package de.in.fcb.migration;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

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
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import javax.imageio.ImageIO;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import de.in.fcb.exams.Assignment;
import de.in.fcb.exams.AssignmentRepository;
import de.in.fcb.exams.AssignmentService;
import de.in.fcb.exams.ForeignLanguage;
import de.in.fcb.exams.ScanStorageService;
import de.in.fcb.exams.SchoolClass;
import de.in.fcb.exams.SchoolClassRepository;
import de.in.fcb.exams.Student;
import de.in.fcb.exams.StudentRepository;
import de.in.fcb.exams.StudentWork;
import de.in.fcb.exams.StudentWorkPage;
import de.in.fcb.exams.StudentWorkPageRepository;
import de.in.fcb.exams.StudentWorkRepository;
import de.in.fcb.exams.StudentWorkStatus;
import de.in.fcb.exams.TestDefinition;
import de.in.fcb.exams.TestDefinitionRepository;
import de.in.fcb.exams.TestTask;
import de.in.fcb.exams.TestTaskRepository;

class PocMigrationServiceTest {

    @TempDir
    Path tempDir;

    private final List<SchoolClass> classes = new ArrayList<>();
    private final List<Student> students = new ArrayList<>();
    private final List<TestDefinition> tests = new ArrayList<>();
    private final List<TestTask> tasks = new ArrayList<>();
    private final List<Assignment> assignments = new ArrayList<>();
    private final List<StudentWork> works = new ArrayList<>();
    private final List<StudentWorkPage> pages = new ArrayList<>();

    private ScanStorageService scanStorageService;
    private PocMigrationService service;

    @BeforeEach
    void setUp() {
        SchoolClassRepository classRepository = mock(SchoolClassRepository.class);
        StudentRepository studentRepository = mock(StudentRepository.class);
        TestDefinitionRepository testRepository = mock(TestDefinitionRepository.class);
        TestTaskRepository taskRepository = mock(TestTaskRepository.class);
        AssignmentRepository assignmentRepository = mock(AssignmentRepository.class);
        StudentWorkRepository studentWorkRepository = mock(StudentWorkRepository.class);
        StudentWorkPageRepository pageRepository = mock(StudentWorkPageRepository.class);

        when(classRepository.save(any(SchoolClass.class))).thenAnswer(inv -> {
            SchoolClass sc = inv.getArgument(0);
            if (sc.getId() == null) {
                sc.setId(UUID.randomUUID());
            }
            classes.removeIf(existing -> existing.getId().equals(sc.getId()));
            classes.add(sc);
            return sc;
        });
        when(classRepository.findByNameIgnoreCase(any())).thenAnswer(inv -> {
            String name = inv.getArgument(0);
            return classes.stream().filter(c -> c.getName().equalsIgnoreCase(name)).findFirst();
        });

        when(studentRepository.save(any(Student.class))).thenAnswer(inv -> {
            Student s = inv.getArgument(0);
            if (s.getId() == null) {
                s.setId(UUID.randomUUID());
            }
            students.removeIf(existing -> existing.getId().equals(s.getId()));
            students.add(s);
            return s;
        });
        when(studentRepository.findBySchoolClassIdOrderByExternalIdAsc(any())).thenAnswer(inv -> {
            UUID classId = inv.getArgument(0);
            return students.stream().filter(s -> s.getSchoolClassId().equals(classId)).toList();
        });
        when(studentRepository.findBySchoolClassIdAndExternalId(any(), any())).thenAnswer(inv -> {
            UUID classId = inv.getArgument(0);
            String extId = inv.getArgument(1);
            return students.stream()
                    .filter(s -> s.getSchoolClassId().equals(classId) && s.getExternalId().equals(extId))
                    .findFirst();
        });

        when(testRepository.save(any(TestDefinition.class))).thenAnswer(inv -> {
            TestDefinition td = inv.getArgument(0);
            if (td.getId() == null) {
                td.setId(UUID.randomUUID());
            }
            tests.removeIf(existing -> existing.getId().equals(td.getId()));
            tests.add(td);
            return td;
        });
        when(testRepository.findFirstByTitleIgnoreCase(any())).thenAnswer(inv -> {
            String title = inv.getArgument(0);
            return tests.stream().filter(t -> t.getTitle().equalsIgnoreCase(title)).findFirst();
        });

        when(taskRepository.save(any(TestTask.class))).thenAnswer(inv -> {
            TestTask tt = inv.getArgument(0);
            if (tt.getId() == null) {
                tt.setId(UUID.randomUUID());
            }
            tasks.removeIf(existing -> existing.getId().equals(tt.getId()));
            tasks.add(tt);
            return tt;
        });
        when(taskRepository.findByTestDefinitionIdOrderByPositionAsc(any())).thenAnswer(inv -> {
            UUID testId = inv.getArgument(0);
            return tasks.stream().filter(t -> t.getTestDefinitionId().equals(testId)).toList();
        });
        when(taskRepository.countByTestDefinitionId(any())).thenAnswer(inv -> {
            UUID testId = inv.getArgument(0);
            return tasks.stream().filter(t -> t.getTestDefinitionId().equals(testId)).count();
        });

        when(assignmentRepository.save(any(Assignment.class))).thenAnswer(inv -> {
            Assignment a = inv.getArgument(0);
            if (a.getId() == null) {
                a.setId(UUID.randomUUID());
            }
            assignments.removeIf(existing -> existing.getId().equals(a.getId()));
            assignments.add(a);
            return a;
        });
        when(assignmentRepository.findFirstByTitleIgnoreCaseAndSchoolClassId(any(), any())).thenAnswer(inv -> {
            String title = inv.getArgument(0);
            UUID classId = inv.getArgument(1);
            return assignments.stream()
                    .filter(a -> a.getTitle().equalsIgnoreCase(title) && a.getSchoolClassId().equals(classId))
                    .findFirst();
        });

        when(studentWorkRepository.save(any(StudentWork.class))).thenAnswer(inv -> {
            StudentWork w = inv.getArgument(0);
            if (w.getId() == null) {
                w.setId(UUID.randomUUID());
            }
            works.removeIf(existing -> existing.getId().equals(w.getId()));
            works.add(w);
            return w;
        });
        when(studentWorkRepository.findById(any())).thenAnswer(inv -> {
            UUID id = inv.getArgument(0);
            return works.stream().filter(w -> w.getId().equals(id)).findFirst();
        });
        when(studentWorkRepository.findByAssignmentId(any())).thenAnswer(inv -> {
            UUID assignmentId = inv.getArgument(0);
            return works.stream().filter(w -> w.getAssignmentId().equals(assignmentId)).toList();
        });
        when(studentWorkRepository.findByAssignmentIdAndStudentId(any(), any())).thenAnswer(inv -> {
            UUID assignmentId = inv.getArgument(0);
            UUID studentId = inv.getArgument(1);
            return works.stream()
                    .filter(w -> w.getAssignmentId().equals(assignmentId) && w.getStudentId().equals(studentId))
                    .findFirst();
        });

        when(pageRepository.save(any(StudentWorkPage.class))).thenAnswer(inv -> {
            StudentWorkPage p = inv.getArgument(0);
            if (p.getId() == null) {
                p.setId(UUID.randomUUID());
            }
            pages.removeIf(existing -> existing.getId().equals(p.getId()));
            pages.add(p);
            return p;
        });
        when(pageRepository.findByStudentWorkIdOrderByPageNumberAsc(any())).thenAnswer(inv -> {
            UUID workId = inv.getArgument(0);
            return pages.stream().filter(p -> p.getStudentWorkId().equals(workId)).toList();
        });
        when(pageRepository.countByStudentWorkId(any())).thenAnswer(inv -> {
            UUID workId = inv.getArgument(0);
            return pages.stream().filter(p -> p.getStudentWorkId().equals(workId)).count();
        });

        Path scanRoot = tempDir.resolve("scans");
        scanStorageService = new ScanStorageService(scanRoot);
        AssignmentService assignmentService = new AssignmentService(
                assignmentRepository,
                studentRepository,
                taskRepository,
                studentWorkRepository,
                pageRepository,
                scanStorageService);

        service = new PocMigrationService(
                classRepository,
                studentRepository,
                testRepository,
                taskRepository,
                assignmentRepository,
                studentWorkRepository,
                pageRepository,
                assignmentService);
    }

    @Test
    void previewsMigratesAndRemainsIdempotentOnSecondRun() throws Exception {
        Path sqliteDb = tempDir.resolve("flaicheckbot-poc.db");
        byte[] samplePng = createSamplePng();
        createSamplePocDatabase(sqliteDb, samplePng);

        PocMigrationPreview preview = service.analyzeSqliteFile(sqliteDb);
        assertThat(preview.classes()).hasSize(1);
        assertThat(preview.classes().get(0).studentCount()).isEqualTo(2);
        assertThat(preview.classes().get(0).alreadyExists()).isFalse();
        assertThat(preview.tests()).hasSize(1);
        assertThat(preview.tests().get(0).taskCount()).isEqualTo(2);
        assertThat(preview.assignments()).hasSize(1);
        assertThat(preview.assignments().get(0).workCount()).isEqualTo(2);
        assertThat(preview.assignments().get(0).scanBlobCount()).isEqualTo(1);

        PocMigrationResult firstRun = service.migrateSqliteFile(sqliteDb, PocMigrationOptions.defaults());
        assertThat(firstRun.classesCreated()).isEqualTo(1);
        assertThat(firstRun.studentsCreated()).isEqualTo(2);
        assertThat(firstRun.testsCreated()).isEqualTo(1);
        assertThat(firstRun.tasksCreated()).isEqualTo(2);
        assertThat(firstRun.assignmentsCreated()).isEqualTo(1);
        assertThat(firstRun.scansImported()).isEqualTo(1);
        assertThat(firstRun.scansSkipped()).isZero();

        assertThat(classes).hasSize(1);
        assertThat(classes.get(0).getForeignLanguage()).isEqualTo(ForeignLanguage.EN);
        assertThat(tasks).hasSize(2);
        assertThat(tasks.get(0).getGradingHints()).isEqualTo("Text A");
        assertThat(pages).hasSize(1);
        assertThat(scanStorageService.readImageBytes(pages.get(0).getStoragePath())).isNotEmpty();

        StudentWork uploadedWork = works.stream()
                .filter(w -> w.getStatus() == StudentWorkStatus.UPLOADED)
                .findFirst()
                .orElseThrow();
        assertThat(uploadedWork.getMaxPoints()).isEqualByComparingTo("15.0");

        PocMigrationPreview afterPreview = service.analyzeSqliteFile(sqliteDb);
        assertThat(afterPreview.classes().get(0).alreadyExists()).isTrue();
        assertThat(afterPreview.tests().get(0).alreadyExists()).isTrue();
        assertThat(afterPreview.assignments().get(0).alreadyExists()).isTrue();

        PocMigrationResult secondRun = service.migrateSqliteFile(sqliteDb, PocMigrationOptions.defaults());
        assertThat(secondRun.classesCreated()).isZero();
        assertThat(secondRun.classesSkipped()).isEqualTo(1);
        assertThat(secondRun.studentsCreated()).isZero();
        assertThat(secondRun.studentsSkipped()).isEqualTo(2);
        assertThat(secondRun.testsCreated()).isZero();
        assertThat(secondRun.testsSkipped()).isEqualTo(1);
        assertThat(secondRun.tasksCreated()).isZero();
        assertThat(secondRun.assignmentsCreated()).isZero();
        assertThat(secondRun.assignmentsSkipped()).isEqualTo(1);
        assertThat(secondRun.scansImported()).isZero();
        assertThat(secondRun.scansSkipped()).isEqualTo(1);
        assertThat(pages).hasSize(1);
    }

    private void createSamplePocDatabase(Path dbFile, byte[] pngBlob) throws Exception {
        Files.deleteIfExists(dbFile);
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbFile.toAbsolutePath());
                Statement stmt = conn.createStatement()) {
            stmt.execute("""
                    CREATE TABLE classes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        academic_year TEXT
                    )
                    """);
            stmt.execute("""
                    CREATE TABLE students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        external_id TEXT UNIQUE,
                        class_id INTEGER
                    )
                    """);
            stmt.execute("""
                    CREATE TABLE test_definitions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        grade_level TEXT,
                        learning_unit TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """);
            stmt.execute("""
                    CREATE TABLE test_tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        test_id INTEGER,
                        position INTEGER,
                        task_text TEXT,
                        reference_text TEXT,
                        max_points INTEGER,
                        resource_info TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """);
            stmt.execute("""
                    CREATE TABLE assignments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        class_id INTEGER,
                        test_id INTEGER,
                        title TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """);
            stmt.execute("""
                    CREATE TABLE student_work (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        assignment_id INTEGER,
                        student_id INTEGER,
                        image_data BLOB,
                        recognized_text TEXT,
                        grading_json TEXT,
                        status TEXT DEFAULT 'NEW'
                    )
                    """);

            stmt.execute("INSERT INTO classes (id, name, academic_year) VALUES (1, '2025KL8b', '2025/2026')");
            stmt.execute("INSERT INTO students (id, external_id, class_id) VALUES (10, 'S01', 1)");
            stmt.execute("INSERT INTO students (id, external_id, class_id) VALUES (11, 'S02', 1)");
            stmt.execute("""
                    INSERT INTO test_definitions (id, title, grade_level, learning_unit)
                    VALUES (1, 'Unit 3 Reading Test', '8', 'Unit 3')
                    """);
            stmt.execute("""
                    INSERT INTO test_tasks (id, test_id, position, task_text, reference_text, max_points, resource_info)
                    VALUES (1, 1, 1, 'Summarize paragraph 1.', 'The author visits London.', 5, 'Text A')
                    """);
            stmt.execute("""
                    INSERT INTO test_tasks (id, test_id, position, task_text, reference_text, max_points, resource_info)
                    VALUES (2, 1, 2, 'Explain the ending.', 'She decides to stay.', 10, NULL)
                    """);
            stmt.execute("""
                    INSERT INTO assignments (id, class_id, test_id, title, created_at)
                    VALUES (1, 1, 1, 'Klausur Unit 3', '2026-03-10 09:00:00')
                    """);
            try (PreparedStatement ps = conn.prepareStatement(
                    "INSERT INTO student_work (id, assignment_id, student_id, image_data, recognized_text, status) "
                            + "VALUES (?, ?, ?, ?, ?, ?)")) {
                ps.setInt(1, 100);
                ps.setInt(2, 1);
                ps.setInt(3, 10);
                ps.setBytes(4, pngBlob);
                ps.setString(5, "Legacy OCR text");
                ps.setString(6, "OCR_DONE");
                ps.executeUpdate();

                ps.setInt(1, 101);
                ps.setInt(2, 1);
                ps.setInt(3, 11);
                ps.setBytes(4, null);
                ps.setString(5, null);
                ps.setString(6, "NEW");
                ps.executeUpdate();
            }
        }
    }

    private static byte[] createSamplePng() throws Exception {
        BufferedImage image = new BufferedImage(80, 40, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        try {
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, 80, 40);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Test", 10, 24);
        } finally {
            g2d.dispose();
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(image, "png", out);
        return out.toByteArray();
    }
}
