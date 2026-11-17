package de.in.fcb.e2e;

import static com.microsoft.playwright.assertions.PlaywrightAssertions.assertThat;
import static org.assertj.core.api.Assertions.assertThat;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.List;
import java.util.Set;
import java.util.UUID;

import javax.imageio.ImageIO;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import com.microsoft.playwright.Browser;
import com.microsoft.playwright.BrowserContext;
import com.microsoft.playwright.BrowserType;
import com.microsoft.playwright.Download;
import com.microsoft.playwright.Locator;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.Playwright;
import com.microsoft.playwright.Tracing;
import com.microsoft.playwright.options.AriaRole;

import de.in.fcb.ai.AiClient;
import de.in.fcb.ai.AiConfig;
import de.in.fcb.ai.ExtractedTask;
import de.in.fcb.ai.GradingResult;
import de.in.fcb.exams.TaskEvaluation;
import de.in.fcb.exams.TestDefinition;
import de.in.fcb.exams.TestTask;
import de.in.fcb.security.AppRole;
import de.in.fcb.security.AppUser;
import de.in.fcb.security.AppUserRepository;
import de.in.fcb.security.AuthProvider;

/**
 * Full end-to-end Playwright browser click-workflow test covering login,
 * AI settings, classes & pseudonymized students, test definitions with AI task
 * extraction, exam assignment, scan upload & redaction, AI batch grading,
 * split-view review, PDF/CSV export, and PoC SQLite database migration.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@Tag("e2e")
class ExamWorkflowE2ETest {

    private static final String POSTGRES_IMAGE = "postgres:16-alpine";
    private static final Path ARTIFACTS_DIR = Path.of("target", "playwright-artifacts");

    @Container
    static final PostgreSQLContainer<?> POSTGRES = new PostgreSQLContainer<>(POSTGRES_IMAGE);

    @TempDir
    static Path tempStorageDir;

    private static Playwright playwright;
    private static Browser browser;

    @LocalServerPort
    private int port;

    @Autowired
    private AppUserRepository appUserRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @DynamicPropertySource
    static void registerProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", POSTGRES::getJdbcUrl);
        registry.add("spring.datasource.username", POSTGRES::getUsername);
        registry.add("spring.datasource.password", POSTGRES::getPassword);
        registry.add("flaicheckbot.storage.dir", () -> tempStorageDir.resolve("scans").toAbsolutePath().toString());
        registry.add("flaicheckbot.migration.enabled", () -> "true");
    }

    @TestConfiguration
    static class StubAiConfig {

        @Bean
        @Primary
        AiClient stubAiClient() {
            return new AiClient() {
                @Override
                public List<String> listModels(AiConfig config) {
                    return List.of("gemini-2.5-flash", "gemini-2.5-pro");
                }

                @Override
                public List<ExtractedTask> extractTasksFromText(
                        AiConfig config, String documentText, String language) {
                    return List.of(
                            new ExtractedTask(
                                    1,
                                    "Reading Comprehension",
                                    "Summarize the main idea.",
                                    "The protagonist moves to London.",
                                    "Award 2 pts for key fact.",
                                    new java.math.BigDecimal("10.00")),
                            new ExtractedTask(
                                    2,
                                    "Vocabulary",
                                    "Explain two idioms.",
                                    "Break a leg; piece of cake.",
                                    "5 pts per idiom.",
                                    new java.math.BigDecimal("10.00")));
                }

                @Override
                public GradingResult gradeStudentWork(
                        AiConfig config,
                        TestDefinition test,
                        List<TestTask> tasks,
                        String studentExternalId,
                        List<byte[]> pagePngBytes) {
                    return new GradingResult(
                            "The protagonist moves to London. Break a leg means good luck.",
                            "Good overall structure and clear vocabulary.",
                            List.of(
                                    new TaskEvaluation(
                                            1,
                                            "Reading Comprehension",
                                            "The protagonist moves to London.",
                                            new java.math.BigDecimal("9.00"),
                                            new java.math.BigDecimal("10.00"),
                                            "Clear summary.",
                                            ""),
                                    new TaskEvaluation(
                                            2,
                                            "Vocabulary",
                                            "Break a leg means good luck.",
                                            new java.math.BigDecimal("8.50"),
                                            new java.math.BigDecimal("10.00"),
                                            "One idiom explained accurately.",
                                            "Second idiom missing.")),
                            350,
                            120,
                            "gemini-2.5-flash");
                }
            };
        }
    }

    @BeforeAll
    static void startBrowser() {
        playwright = Playwright.create();
        browser = playwright.chromium().launch(new BrowserType.LaunchOptions().setHeadless(true));
    }

    @AfterAll
    static void stopBrowser() {
        if (browser != null) {
            browser.close();
        }
        if (playwright != null) {
            playwright.close();
        }
    }

    @Test
    void fullTeacherExamAndMigrationWorkflow() throws Exception {
        ensureAdminUser();
        Files.createDirectories(ARTIFACTS_DIR);

        try (BrowserContext context = browser.newContext(
                new Browser.NewContextOptions()
                        .setAcceptDownloads(true)
                        .setLocale("de-DE")
                        .setViewportSize(1600, 1000))) {
            context.tracing().start(new Tracing.StartOptions().setScreenshots(true).setSnapshots(true));
            Page page = context.newPage();
            page.setDefaultTimeout(20_000);

            try {
                // 1. Login & Dashboard
                page.navigate(baseUrl("/login"));
                page.locator("input[name='username']").fill("e2eadmin");
                page.locator("input[name='password']").fill("E2eAdminPass123!");
                page.locator("vaadin-button[slot='submit'], button[type='submit']").first().click();
                assertThat(page.getByText("Willkommen bei flAICheckBot")).isVisible();

                // 2. Admin -> AI Settings
                page.navigate(baseUrl("/admin"));
                page.getByRole(AriaRole.TAB, new Page.GetByRoleOptions().setName("Einstellungen")).click();
                selectComboBoxItem(page, "Anbieter / Modus", "Google AI Studio / Gemini (Cloud)");
                page.getByLabel("API-Key").fill("e2e-dummy-key");
                page.getByRole(AriaRole.BUTTON, new Page.GetByRoleOptions().setName("Modelle abrufen")).click();
                assertThat(page.getByText("2 Modelle geladen.")).isVisible();
                page.getByRole(AriaRole.BUTTON, new Page.GetByRoleOptions().setName("Speichern")).click();
                assertThat(page.getByText("Einstellungen gespeichert")).isVisible();

                // 3. Classes & Batch Students
                page.navigate(baseUrl("/classes"));
                page.getByRole(AriaRole.BUTTON, new Page.GetByRoleOptions().setName("Neue Klasse")).click();
                Locator classDialog = page.locator("vaadin-dialog[opened]").last();
                classDialog.getByLabel("Klassenname").fill("8b-E2E");
                classDialog.getByLabel("Schuljahr").fill("2025/2026");
                classDialog.getByRole(AriaRole.BUTTON, new Locator.GetByRoleOptions().setName("Speichern")).click();
                assertThat(page.getByText("8b-E2E").first()).isVisible();

                page.getByRole(AriaRole.BUTTON, new Page.GetByRoleOptions().setName("Batch anlegen")).click();
                Locator batchDialog = page.locator("vaadin-dialog[opened]").last();
                batchDialog.getByLabel("Bis Nummer").fill("2");
                batchDialog.getByRole(AriaRole.BUTTON, new Locator.GetByRoleOptions().setName("Generieren")).click();
                assertThat(page.getByText("S01").first()).isVisible();
                assertThat(page.getByText("S02").first()).isVisible();

                // 4. Test Definitions & AI Task Extraction
                page.navigate(baseUrl("/tests"));
                page.getByRole(AriaRole.BUTTON, new Page.GetByRoleOptions().setName("Neuer Test")).click();
                Locator testDialog = page.locator("vaadin-dialog[opened]").last();
                testDialog.getByLabel("Titel").fill("E2E Reading Test");
                testDialog.getByRole(AriaRole.BUTTON, new Locator.GetByRoleOptions().setName("Speichern")).click();
                assertThat(page.getByText("E2E Reading Test").first()).isVisible();

                page.getByLabel("Aufgabentext / Erwartungshorizont (Rohtext)")
                        .fill("Task 1: Summarize the story. Task 2: Explain two idioms.");
                page.getByRole(AriaRole.BUTTON, new Page.GetByRoleOptions().setName("Text speichern")).click();
                page.getByRole(AriaRole.BUTTON, new Page.GetByRoleOptions().setName("Aufgaben per KI strukturieren"))
                        .click();
                assertThat(page.getByText("Reading Comprehension").first()).isVisible();
                assertThat(page.getByText("Vocabulary").first()).isVisible();

                // 5. Assignments -> Scan Upload -> Redact -> AI Batch Grade -> Review -> Export
                page.navigate(baseUrl("/assignments"));
                page.getByRole(AriaRole.BUTTON, new Page.GetByRoleOptions().setName("Neue Klausur")).click();
                Locator assignDialog = page.locator("vaadin-dialog[opened]").last();
                assignDialog.getByLabel("Klausurtitel").fill("E2E Klausur 1");
                selectComboBoxItem(assignDialog, "Klasse", "8b-E2E");
                selectComboBoxItem(assignDialog, "Test-Definition", "E2E Reading Test");
                assignDialog.getByRole(AriaRole.BUTTON, new Locator.GetByRoleOptions().setName("Speichern")).click();
                assertThat(page.getByText("E2E Klausur 1").first()).isVisible();
                assertThat(page.getByText("S01").first()).isVisible();

                Path sampleScan = createSamplePngFile(tempStorageDir.resolve("scan-s01.png"));
                page.getByTitle("Scans hochladen & schwärzen").first().click();
                Locator scanDialog = page.locator("vaadin-dialog[opened]").last();
                scanDialog.locator("vaadin-upload input[type='file']").setInputFiles(sampleScan);
                assertThat(scanDialog.getByRole(AriaRole.BUTTON,
                        new Locator.GetByRoleOptions().setName("Kopfzeile schwärzen (obere 15%)"))).isEnabled();
                scanDialog.getByRole(AriaRole.BUTTON,
                        new Locator.GetByRoleOptions().setName("Kopfzeile schwärzen (obere 15%)")).click();
                assertThat(scanDialog.getByText("Schwärzung dauerhaft im Bild gespeichert.")).isVisible();
                scanDialog.getByRole(AriaRole.BUTTON, new Locator.GetByRoleOptions().setName("Schließen")).click();

                page.getByRole(AriaRole.BUTTON,
                        new Page.GetByRoleOptions().setName("Alle hochgeladenen per KI bewerten")).click();
                assertThat(page.getByText("KI-bewertet").first()).isVisible();

                page.getByTitle("Review & Korrektur (Split-View)").first().click();
                Locator reviewDialog = page.locator("vaadin-dialog[opened]").last();
                assertThat(reviewDialog.getByText("Reading Comprehension").first()).isVisible();
                reviewDialog.getByRole(AriaRole.BUTTON,
                        new Locator.GetByRoleOptions().setName("Als geprüft abschließen")).click();
                assertThat(page.getByText("Geprüft").first()).isVisible();

                Download pdfDownload = page.waitForDownload(() -> page.getByRole(
                        AriaRole.BUTTON, new Page.GetByRoleOptions().setName("PDF-Bericht")).click());
                assertThat(pdfDownload.suggestedFilename()).endsWith(".pdf");
                assertThat(Files.size(pdfDownload.path())).isGreaterThan(500);

                Download csvDownload = page.waitForDownload(() -> page.getByRole(
                        AriaRole.BUTTON, new Page.GetByRoleOptions().setName("CSV-Export")).click());
                assertThat(csvDownload.suggestedFilename()).endsWith(".csv");
                assertThat(Files.readString(csvDownload.path())).contains("S01");

                // 6. Admin -> PoC SQLite Data Migration
                Path pocDb = createSamplePocDb(tempStorageDir.resolve("flaicheckbot-e2e.db"), sampleScan);
                page.navigate(baseUrl("/admin"));
                page.getByRole(AriaRole.TAB, new Page.GetByRoleOptions().setName("Datenmigration (PoC)")).click();
                page.locator("vaadin-upload input[type='file']").setInputFiles(pocDb);
                assertThat(page.getByText("PoC-Klasse-9a").first()).isVisible();
                page.getByRole(AriaRole.BUTTON, new Page.GetByRoleOptions().setName("Migration starten")).click();
                assertThat(page.getByText("Ergebnis der Datenmigration")).isVisible();

                context.tracing().stop();
            } catch (Throwable error) {
                page.screenshot(new Page.ScreenshotOptions()
                        .setPath(ARTIFACTS_DIR.resolve("failure-screenshot.png"))
                        .setFullPage(true));
                context.tracing().stop(new Tracing.StopOptions()
                        .setPath(ARTIFACTS_DIR.resolve("failure-trace.zip")));
                throw error;
            }
        }
    }

    private void ensureAdminUser() {
        if (appUserRepository.findByUsername("e2eadmin").isPresent()) {
            return;
        }
        AppUser admin = new AppUser();
        admin.setUsername("e2eadmin");
        admin.setEmail("e2eadmin@example.com");
        admin.setPasswordHash(passwordEncoder.encode("E2eAdminPass123!"));
        admin.setRoles(Set.of(AppRole.ROLE_ADMIN, AppRole.ROLE_USER));
        admin.setAuthProvider(AuthProvider.LOCAL);
        admin.setEnabled(true);
        appUserRepository.save(admin);
    }

    private String baseUrl(String path) {
        return "http://localhost:" + port + path;
    }

    private static void selectComboBoxItem(Page page, String label, String itemText) {
        page.getByLabel(label).click();
        page.locator("vaadin-combo-box-item", new Page.LocatorOptions().setHasText(itemText)).first().click();
    }

    private static void selectComboBoxItem(Locator root, String label, String itemText) {
        root.getByLabel(label).click();
        root.page().locator("vaadin-combo-box-item", new Page.LocatorOptions().setHasText(itemText)).first().click();
    }

    private static Path createSamplePngFile(Path target) throws Exception {
        BufferedImage image = new BufferedImage(120, 80, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        try {
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, 120, 80);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Student S01", 10, 20);
            g2d.drawString("Answer text", 10, 50);
        } finally {
            g2d.dispose();
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(image, "png", out);
        Files.write(target, out.toByteArray());
        return target;
    }

    private static Path createSamplePocDb(Path dbFile, Path pngFile) throws Exception {
        Files.deleteIfExists(dbFile);
        byte[] pngBytes = Files.readAllBytes(pngFile);
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbFile.toAbsolutePath());
                Statement stmt = conn.createStatement()) {
            stmt.execute("CREATE TABLE classes (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, academic_year TEXT)");
            stmt.execute("CREATE TABLE students (id INTEGER PRIMARY KEY AUTOINCREMENT, external_id TEXT UNIQUE, class_id INTEGER)");
            stmt.execute("CREATE TABLE test_definitions (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, grade_level TEXT, learning_unit TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");
            stmt.execute("CREATE TABLE test_tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, position INTEGER, task_text TEXT, reference_text TEXT, max_points INTEGER, resource_info TEXT)");
            stmt.execute("CREATE TABLE assignments (id INTEGER PRIMARY KEY AUTOINCREMENT, class_id INTEGER, test_id INTEGER, title TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");
            stmt.execute("CREATE TABLE student_work (id INTEGER PRIMARY KEY AUTOINCREMENT, assignment_id INTEGER, student_id INTEGER, image_data BLOB, recognized_text TEXT, grading_json TEXT, status TEXT DEFAULT 'NEW')");

            stmt.execute("INSERT INTO classes (id, name, academic_year) VALUES (1, 'PoC-Klasse-9a', '2025/2026')");
            stmt.execute("INSERT INTO students (id, external_id, class_id) VALUES (1, 'POC-' || '" + UUID.randomUUID().toString().substring(0, 4) + "', 1)");
            stmt.execute("INSERT INTO test_definitions (id, title, grade_level, learning_unit) VALUES (1, 'PoC Unit 4', '9', 'Unit 4')");
            stmt.execute("INSERT INTO test_tasks (id, test_id, position, task_text, reference_text, max_points, resource_info) VALUES (1, 1, 1, 'Write a summary.', 'Summary key.', 10, 'Hint')");
            stmt.execute("INSERT INTO assignments (id, class_id, test_id, title, created_at) VALUES (1, 1, 1, 'PoC Klausur', '2026-04-01 10:00:00')");
            try (PreparedStatement ps = conn.prepareStatement(
                    "INSERT INTO student_work (id, assignment_id, student_id, image_data, status) VALUES (1, 1, 1, ?, 'NEW')")) {
                ps.setBytes(1, pngBytes);
                ps.executeUpdate();
            }
        }
        return dbFile;
    }
}
