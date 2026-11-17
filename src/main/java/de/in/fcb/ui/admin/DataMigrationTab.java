package de.in.fcb.ui.admin;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import com.vaadin.flow.component.Component;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.checkbox.Checkbox;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.grid.GridVariant;
import com.vaadin.flow.component.html.Div;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Paragraph;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.FlexLayout;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.upload.Upload;
import com.vaadin.flow.data.renderer.ComponentRenderer;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.exams.ForeignLanguage;
import de.in.fcb.migration.PocMigrationOptions;
import de.in.fcb.migration.PocMigrationPreview;
import de.in.fcb.migration.PocMigrationResult;
import de.in.fcb.migration.PocMigrationService;

/**
 * Administration tab for inspecting and importing legacy PoC SQLite databases
 * ({@code flaicheckbot.db}) in a two-step preview-then-import workflow.
 */
public class DataMigrationTab extends VerticalLayout implements AdminTab {

    private static final int MAX_UPLOAD_BYTES = 200 * 1024 * 1024;

    private final PocMigrationService migrationService;

    final Upload sqliteUpload;
    final Span fileStatusBadge = new Span();
    final HorizontalLayout summaryBar = new HorizontalLayout();
    final Grid<PocMigrationPreview.ClassSummary> classesGrid = new Grid<>(PocMigrationPreview.ClassSummary.class, false);
    final Grid<PocMigrationPreview.TestSummary> testsGrid = new Grid<>(PocMigrationPreview.TestSummary.class, false);
    final Grid<PocMigrationPreview.AssignmentSummary> assignmentsGrid = new Grid<>(
            PocMigrationPreview.AssignmentSummary.class, false);

    final Checkbox importClassesBox = new Checkbox();
    final Checkbox importTestsBox = new Checkbox();
    final Checkbox importAssignmentsBox = new Checkbox();
    final Checkbox preserveOcrBox = new Checkbox();
    final ComboBox<ForeignLanguage> defaultLanguageBox = new ComboBox<>();
    final Button runImportButton = new Button();

    final VerticalLayout previewSection = new VerticalLayout();
    final Div resultBanner = new Div();

    private Path tempSqliteFile;
    private String uploadedFileName;

    public DataMigrationTab(PocMigrationService migrationService) {
        this.migrationService = migrationService;

        setWidthFull();
        setHeightFull();
        setPadding(false);
        setSpacing(true);
        getStyle().set("overflow-y", "auto");

        sqliteUpload = createUploadComponent();
        configureGrids();
        configureOptionsSection();
        buildPreviewSection();

        resultBanner.setWidthFull();
        resultBanner.setVisible(false);
        resultBanner.addClassNames(
                LumoUtility.Padding.MEDIUM,
                LumoUtility.BorderRadius.MEDIUM,
                LumoUtility.Background.SUCCESS_10);

        add(createHeader(), sqliteUpload, fileStatusBadge, previewSection, resultBanner);

        addDetachListener(event -> cleanupTempFile());
    }

    @Override
    public String getLabelKey() {
        return "admin.tab.migration";
    }

    @Override
    public Component asComponent() {
        return this;
    }

    @Override
    public void refresh() {
        if (tempSqliteFile != null && Files.isRegularFile(tempSqliteFile)) {
            analyzeFile(tempSqliteFile, uploadedFileName);
        }
    }

    private VerticalLayout createHeader() {
        H3 title = new H3(getTranslation("admin.migration.title"));
        title.addClassNames(LumoUtility.Margin.NONE);
        title.addClassName("admin-tab-title");

        Paragraph description = new Paragraph(getTranslation("admin.migration.description"));
        description.addClassNames(LumoUtility.TextColor.SECONDARY, LumoUtility.Margin.NONE);

        VerticalLayout header = new VerticalLayout(title, description);
        header.setPadding(false);
        header.setSpacing(false);
        header.setWidthFull();
        return header;
    }

    private Upload createUploadComponent() {
        Upload upload = new Upload(this::receiveUpload);
        upload.setAcceptedFileTypes(".db", ".sqlite", ".sqlite3");
        upload.setMaxFiles(1);
        upload.setMaxFileSize(MAX_UPLOAD_BYTES);
        upload.setWidthFull();
        upload.setUploadButton(new Button(getTranslation("admin.migration.upload.button"), VaadinIcon.UPLOAD.create()));
        upload.setDropLabel(new Span(getTranslation("admin.migration.upload.drop")));
        upload.addSucceededListener(event -> {
            upload.clearFileList();
            analyzeFile(tempSqliteFile, event.getFileName());
        });
        upload.addFailedListener(event -> {
            cleanupTempFile();
            showError(event.getReason() != null ? event.getReason().getMessage() : "Upload failed");
        });
        upload.addFileRejectedListener(event -> showError(event.getErrorMessage()));
        return upload;
    }

    private OutputStream receiveUpload(String fileName, String mimeType) {
        cleanupTempFile();
        try {
            tempSqliteFile = Files.createTempFile("fcb-poc-import-", ".db");
            uploadedFileName = fileName;
            return Files.newOutputStream(tempSqliteFile);
        } catch (IOException ex) {
            throw new UncheckedIOException("Failed to create temporary SQLite file", ex);
        }
    }

    private void configureGrids() {
        classesGrid.addColumn(PocMigrationPreview.ClassSummary::name)
                .setHeader(getTranslation("exams.classes.col.name"))
                .setAutoWidth(true)
                .setFlexGrow(1);
        classesGrid.addColumn(PocMigrationPreview.ClassSummary::academicYear)
                .setHeader(getTranslation("exams.classes.col.year"))
                .setAutoWidth(true);
        classesGrid.addColumn(PocMigrationPreview.ClassSummary::studentCount)
                .setHeader(getTranslation("exams.classes.col.student-count"))
                .setAutoWidth(true);
        classesGrid.addColumn(new ComponentRenderer<>(item -> statusBadge(item.alreadyExists())))
                .setHeader(getTranslation("admin.grid.column.status"))
                .setAutoWidth(true);
        classesGrid.addThemeVariants(GridVariant.LUMO_COMPACT, GridVariant.LUMO_ROW_STRIPES);
        classesGrid.setHeight("200px");
        classesGrid.setWidthFull();

        testsGrid.addColumn(PocMigrationPreview.TestSummary::title)
                .setHeader(getTranslation("exams.tests.col.title"))
                .setAutoWidth(true)
                .setFlexGrow(1);
        testsGrid.addColumn(item -> item.learningUnit() != null ? item.learningUnit() : "-")
                .setHeader(getTranslation("exams.tests.col.unit"))
                .setAutoWidth(true);
        testsGrid.addColumn(PocMigrationPreview.TestSummary::taskCount)
                .setHeader(getTranslation("exams.tests.col.task-count"))
                .setAutoWidth(true);
        testsGrid.addColumn(new ComponentRenderer<>(item -> statusBadge(item.alreadyExists())))
                .setHeader(getTranslation("admin.grid.column.status"))
                .setAutoWidth(true);
        testsGrid.addThemeVariants(GridVariant.LUMO_COMPACT, GridVariant.LUMO_ROW_STRIPES);
        testsGrid.setHeight("200px");
        testsGrid.setWidthFull();

        assignmentsGrid.addColumn(PocMigrationPreview.AssignmentSummary::title)
                .setHeader(getTranslation("exams.assignments.col.title"))
                .setAutoWidth(true)
                .setFlexGrow(1);
        assignmentsGrid.addColumn(PocMigrationPreview.AssignmentSummary::className)
                .setHeader(getTranslation("exams.assignments.col.class"))
                .setAutoWidth(true);
        assignmentsGrid.addColumn(PocMigrationPreview.AssignmentSummary::testTitle)
                .setHeader(getTranslation("exams.assignments.col.test"))
                .setAutoWidth(true);
        assignmentsGrid.addColumn(PocMigrationPreview.AssignmentSummary::workCount)
                .setHeader(getTranslation("admin.migration.col.works"))
                .setAutoWidth(true);
        assignmentsGrid.addColumn(PocMigrationPreview.AssignmentSummary::scanBlobCount)
                .setHeader(getTranslation("admin.migration.col.scans"))
                .setAutoWidth(true);
        assignmentsGrid.addColumn(new ComponentRenderer<>(item -> statusBadge(item.alreadyExists())))
                .setHeader(getTranslation("admin.grid.column.status"))
                .setAutoWidth(true);
        assignmentsGrid.addThemeVariants(GridVariant.LUMO_COMPACT, GridVariant.LUMO_ROW_STRIPES);
        assignmentsGrid.setHeight("200px");
        assignmentsGrid.setWidthFull();
    }

    private void configureOptionsSection() {
        importClassesBox.setLabel(getTranslation("admin.migration.option.classes"));
        importClassesBox.setValue(true);

        importTestsBox.setLabel(getTranslation("admin.migration.option.tests"));
        importTestsBox.setValue(true);

        importAssignmentsBox.setLabel(getTranslation("admin.migration.option.assignments"));
        importAssignmentsBox.setValue(true);

        preserveOcrBox.setLabel(getTranslation("admin.migration.option.preserve-ocr"));
        preserveOcrBox.setValue(false);

        defaultLanguageBox.setLabel(getTranslation("admin.migration.option.language"));
        defaultLanguageBox.setItems(ForeignLanguage.values());
        defaultLanguageBox.setValue(ForeignLanguage.EN);
        defaultLanguageBox.setWidth("150px");

        runImportButton.setText(getTranslation("admin.migration.button.run"));
        runImportButton.setIcon(VaadinIcon.PLAY.create());
        runImportButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        runImportButton.setEnabled(false);
        runImportButton.addClickListener(event -> runImport());
    }

    private void buildPreviewSection() {
        fileStatusBadge.setVisible(false);

        summaryBar.setWidthFull();
        summaryBar.setSpacing(true);
        summaryBar.getStyle().set("flex-wrap", "wrap");

        Span classesTitle = new Span(getTranslation("admin.migration.preview.classes"));
        classesTitle.addClassNames(LumoUtility.FontWeight.SEMIBOLD);
        VerticalLayout classesBox = new VerticalLayout(classesTitle, classesGrid);
        classesBox.setPadding(false);
        classesBox.setSpacing(false);
        classesBox.setWidth("48%");
        classesBox.getStyle().set("min-width", "320px");
        classesBox.getStyle().set("flex", "1 1 320px");

        Span testsTitle = new Span(getTranslation("admin.migration.preview.tests"));
        testsTitle.addClassNames(LumoUtility.FontWeight.SEMIBOLD);
        VerticalLayout testsBox = new VerticalLayout(testsTitle, testsGrid);
        testsBox.setPadding(false);
        testsBox.setSpacing(false);
        testsBox.setWidth("48%");
        testsBox.getStyle().set("min-width", "320px");
        testsBox.getStyle().set("flex", "1 1 320px");

        FlexLayout masterDataRow = new FlexLayout(classesBox, testsBox);
        masterDataRow.setWidthFull();
        masterDataRow.setFlexWrap(FlexLayout.FlexWrap.WRAP);
        masterDataRow.getStyle().set("gap", "1rem");

        Span assignmentsTitle = new Span(getTranslation("admin.migration.preview.assignments"));
        assignmentsTitle.addClassNames(LumoUtility.FontWeight.SEMIBOLD);

        FlexLayout optionsRow = new FlexLayout(
                importClassesBox,
                importTestsBox,
                importAssignmentsBox,
                preserveOcrBox,
                defaultLanguageBox,
                runImportButton);
        optionsRow.setWidthFull();
        optionsRow.setFlexWrap(FlexLayout.FlexWrap.WRAP);
        optionsRow.setAlignItems(Alignment.END);
        optionsRow.getStyle().set("gap", "1rem");
        optionsRow.addClassNames(
                LumoUtility.Padding.MEDIUM,
                LumoUtility.BorderRadius.MEDIUM,
                LumoUtility.Background.CONTRAST_5);

        previewSection.setWidthFull();
        previewSection.setPadding(false);
        previewSection.setSpacing(true);
        previewSection.setVisible(false);
        previewSection.add(summaryBar, masterDataRow, assignmentsTitle, assignmentsGrid, optionsRow);
    }

    void analyzeFile(Path file, String displayName) {
        if (file == null || !Files.isRegularFile(file)) {
            return;
        }
        try {
            tempSqliteFile = file;
            uploadedFileName = displayName != null ? displayName : file.getFileName().toString();
            PocMigrationPreview preview = migrationService.analyzeSqliteFile(file);
            renderPreview(preview, uploadedFileName);
            resultBanner.setVisible(false);
        } catch (Exception ex) {
            previewSection.setVisible(false);
            runImportButton.setEnabled(false);
            fileStatusBadge.setVisible(false);
            showError(ex.getMessage());
        }
    }

    void renderPreview(PocMigrationPreview preview, String fileName) {
        fileStatusBadge.setText(getTranslation("admin.migration.analyzed", fileName));
        fileStatusBadge.getElement().getThemeList().clear();
        fileStatusBadge.getElement().getThemeList().add("badge");
        fileStatusBadge.getElement().getThemeList().add("primary");
        fileStatusBadge.setVisible(true);

        summaryBar.removeAll();
        summaryBar.add(
                summaryPill(getTranslation("admin.migration.summary.classes",
                        preview.classes().size(), preview.totalStudents())),
                summaryPill(getTranslation("admin.migration.summary.tests",
                        preview.tests().size(), preview.totalTasks())),
                summaryPill(getTranslation("admin.migration.summary.assignments",
                        preview.assignments().size(), preview.totalWorks(), preview.totalScanBlobs())));

        classesGrid.setItems(preview.classes());
        testsGrid.setItems(preview.tests());
        assignmentsGrid.setItems(preview.assignments());

        boolean hasAnyData = !preview.classes().isEmpty()
                || !preview.tests().isEmpty()
                || !preview.assignments().isEmpty();
        runImportButton.setEnabled(hasAnyData);
        previewSection.setVisible(true);
    }

    void runImport() {
        if (tempSqliteFile == null || !Files.isRegularFile(tempSqliteFile)) {
            return;
        }
        PocMigrationOptions options = new PocMigrationOptions(
                importClassesBox.getValue(),
                importTestsBox.getValue(),
                importAssignmentsBox.getValue(),
                preserveOcrBox.getValue(),
                defaultLanguageBox.getValue());

        try {
            PocMigrationResult result = migrationService.migrateSqliteFile(tempSqliteFile, options);
            showResult(result);
            Notification.show(
                    getTranslation("admin.migration.notification.success",
                            result.classesCreated(),
                            result.testsCreated(),
                            result.assignmentsCreated(),
                            result.scansImported()),
                    5000,
                    Notification.Position.MIDDLE);
            PocMigrationPreview refreshed = migrationService.analyzeSqliteFile(tempSqliteFile);
            renderPreview(refreshed, uploadedFileName);
        } catch (Exception ex) {
            showError(ex.getMessage());
        }
    }

    private void showResult(PocMigrationResult result) {
        resultBanner.removeAll();
        Span title = new Span(getTranslation("admin.migration.result.title"));
        title.addClassNames(LumoUtility.FontWeight.BOLD);
        Paragraph details = new Paragraph(getTranslation(
                "admin.migration.result.details",
                result.classesCreated(),
                result.classesSkipped(),
                result.studentsCreated(),
                result.studentsSkipped(),
                result.testsCreated(),
                result.testsSkipped(),
                result.tasksCreated(),
                result.assignmentsCreated(),
                result.assignmentsSkipped(),
                result.scansImported(),
                result.scansSkipped()));
        details.addClassNames(LumoUtility.Margin.Bottom.NONE, LumoUtility.Margin.Top.XSMALL);
        resultBanner.add(title, details);
        resultBanner.setVisible(true);
    }

    private Span summaryPill(String text) {
        Span pill = new Span(text);
        pill.getElement().getThemeList().add("badge");
        return pill;
    }

    private Span statusBadge(boolean alreadyExists) {
        Span badge = new Span(getTranslation(alreadyExists
                ? "admin.migration.status.exists"
                : "admin.migration.status.new"));
        badge.getElement().getThemeList().add("badge");
        badge.getElement().getThemeList().add(alreadyExists ? "contrast" : "success");
        return badge;
    }

    private void showError(String message) {
        Notification.show(
                getTranslation("admin.notification.error", message != null ? message : "Unknown error"),
                6000,
                Notification.Position.MIDDLE);
    }

    void cleanupTempFile() {
        if (tempSqliteFile != null) {
            try {
                Files.deleteIfExists(tempSqliteFile);
            } catch (IOException ignored) {
                // Best-effort temp file cleanup
            }
            tempSqliteFile = null;
        }
    }

    List<PocMigrationPreview.ClassSummary> currentClassesPreview() {
        return classesGrid.getListDataView().getItems().toList();
    }
}
