package de.in.fcb.ui.admin;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.vaadin.flow.component.UI;

import de.in.fcb.exams.ForeignLanguage;
import de.in.fcb.migration.PocMigrationOptions;
import de.in.fcb.migration.PocMigrationPreview;
import de.in.fcb.migration.PocMigrationResult;
import de.in.fcb.migration.PocMigrationService;

class DataMigrationTabTest {

    @TempDir
    Path tempDir;

    private UI ui;
    private PocMigrationService migrationService;
    private DataMigrationTab tab;

    @BeforeEach
    void setUp() {
        ui = new UI();
        UI.setCurrent(ui);
        migrationService = mock(PocMigrationService.class);
        tab = new DataMigrationTab(migrationService);
    }

    @AfterEach
    void tearDown() {
        UI.setCurrent(null);
        ui = null;
    }

    @Test
    void analyzesSqliteFilePopulatesPreviewAndRunsImport() throws Exception {
        Path fakeDb = Files.createFile(tempDir.resolve("flaicheckbot.db"));

        PocMigrationPreview initialPreview = new PocMigrationPreview(
                List.of(new PocMigrationPreview.ClassSummary(1, "8b", "2025/2026", 12, false)),
                List.of(new PocMigrationPreview.TestSummary(1, "Unit 1", "8", "Unit 1", 3, false)),
                List.of(new PocMigrationPreview.AssignmentSummary(1, "Klassenarbeit 1", "8b", "Unit 1", 12, 10, false)));

        PocMigrationPreview refreshedPreview = new PocMigrationPreview(
                List.of(new PocMigrationPreview.ClassSummary(1, "8b", "2025/2026", 12, true)),
                List.of(new PocMigrationPreview.TestSummary(1, "Unit 1", "8", "Unit 1", 3, true)),
                List.of(new PocMigrationPreview.AssignmentSummary(1, "Klassenarbeit 1", "8b", "Unit 1", 12, 10, true)));

        when(migrationService.analyzeSqliteFile(fakeDb)).thenReturn(initialPreview, refreshedPreview);
        when(migrationService.migrateSqliteFile(eq(fakeDb), any(PocMigrationOptions.class)))
                .thenReturn(new PocMigrationResult(1, 0, 12, 0, 1, 0, 3, 1, 0, 10, 0));

        assertThat(tab.runImportButton.isEnabled()).isFalse();
        assertThat(tab.previewSection.isVisible()).isFalse();

        tab.analyzeFile(fakeDb, "flaicheckbot.db");

        assertThat(tab.previewSection.isVisible()).isTrue();
        assertThat(tab.runImportButton.isEnabled()).isTrue();
        assertThat(tab.currentClassesPreview()).hasSize(1);
        assertThat(tab.currentClassesPreview().get(0).alreadyExists()).isFalse();

        tab.defaultLanguageBox.setValue(ForeignLanguage.FR);
        tab.runImport();

        verify(migrationService).migrateSqliteFile(
                eq(fakeDb),
                eq(new PocMigrationOptions(true, true, true, false, ForeignLanguage.FR)));
        assertThat(tab.resultBanner.isVisible()).isTrue();
        assertThat(tab.currentClassesPreview().get(0).alreadyExists()).isTrue();
    }
}
