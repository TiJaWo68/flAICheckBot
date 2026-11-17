package de.in.fcb.ui.exams;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import com.vaadin.flow.component.UI;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.grid.GridVariant;
import com.vaadin.flow.component.html.Anchor;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.progressbar.ProgressBar;
import com.vaadin.flow.data.renderer.ComponentRenderer;
import com.vaadin.flow.router.BeforeEvent;
import com.vaadin.flow.router.HasDynamicTitle;
import com.vaadin.flow.router.HasUrlParameter;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.server.streams.DownloadHandler;
import com.vaadin.flow.server.streams.DownloadResponse;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.exams.Assignment;
import de.in.fcb.exams.AssignmentExportService;
import de.in.fcb.exams.AssignmentService;
import de.in.fcb.exams.BatchGradingRunner;
import de.in.fcb.exams.ScanStorageService;
import de.in.fcb.exams.SchoolClass;
import de.in.fcb.exams.SchoolClassService;
import de.in.fcb.exams.Student;
import de.in.fcb.exams.StudentWork;
import de.in.fcb.exams.StudentWorkStatus;
import de.in.fcb.exams.TestDefinition;
import de.in.fcb.exams.TestDefinitionService;
import de.in.fcb.exams.TestTask;
import de.in.fcb.ui.GridActionButton;
import de.in.fcb.ui.layout.MainLayout;
import jakarta.annotation.security.PermitAll;

/**
 * Exam evaluation workspace for uploading/redacting scans, running
 * asynchronous AI grading batches with live {@code @Push} progress, reviewing
 * submissions side-by-side, and exporting PDF/CSV reports.
 */
@Route(value = "evaluation", layout = MainLayout.class)
@PermitAll
public class EvaluationView extends VerticalLayout implements HasUrlParameter<String>, HasDynamicTitle {

    private final AssignmentService assignmentService;
    private final SchoolClassService classService;
    private final TestDefinitionService testService;
    private final ScanStorageService scanStorageService;
    private final BatchGradingRunner batchGradingRunner;
    private final AssignmentExportService exportService;

    private final H2 titleHeading = new H2();
    private final Span subtitleSpan = new Span();
    private final Span kpiSummary = new Span();
    private final ComboBox<StudentWorkStatus> statusFilter = new ComboBox<>();
    private final Button batchGradeButton = new Button();
    private final ProgressBar batchProgressBar = new ProgressBar();
    private final Span batchProgressLabel = new Span();
    private final Grid<StudentWork> workGrid = new Grid<>(StudentWork.class, false);

    private final ScanRedactionDialog scanDialog;
    private final ReviewDialog reviewDialog;

    private Assignment assignment;
    private SchoolClass schoolClass;
    private TestDefinition testDefinition;
    private List<TestTask> definedTasks = List.of();
    private Map<UUID, Student> studentsById = Map.of();
    private List<StudentWork> allWorks = List.of();

    public EvaluationView(
            AssignmentService assignmentService,
            SchoolClassService classService,
            TestDefinitionService testService,
            ScanStorageService scanStorageService,
            BatchGradingRunner batchGradingRunner,
            AssignmentExportService exportService) {
        this.assignmentService = assignmentService;
        this.classService = classService;
        this.testService = testService;
        this.scanStorageService = scanStorageService;
        this.batchGradingRunner = batchGradingRunner;
        this.exportService = exportService;

        this.scanDialog = new ScanRedactionDialog(assignmentService, scanStorageService);
        this.reviewDialog = new ReviewDialog(assignmentService, scanStorageService);

        setSizeFull();
        setPadding(true);
        setSpacing(true);

        add(createHeader(), createActionBar(), createProgressRow(), workGrid);
        configureGrid();
        expand(workGrid);
    }

    @Override
    public void setParameter(BeforeEvent event, String parameter) {
        try {
            UUID assignmentId = UUID.fromString(parameter);
            this.assignment = assignmentService.findAssignment(assignmentId);
            this.schoolClass = classService.findClass(assignment.getSchoolClassId());
            this.testDefinition = testService.findTest(assignment.getTestDefinitionId());
            this.definedTasks = testService.findTasks(testDefinition.getId());
            this.studentsById = classService.findStudents(schoolClass.getId()).stream()
                    .collect(Collectors.toMap(Student::getId, s -> s));

            titleHeading.setText(assignment.getTitle());
            subtitleSpan.setText(schoolClass.getName() + " · " + testDefinition.getTitle()
                    + " · " + assignment.getExamDate());
            refresh();
        } catch (Exception ex) {
            Notification.show(ex.getMessage(), 4000, Notification.Position.MIDDLE);
            getUI().ifPresent(ui -> ui.navigate(AssignmentsView.class));
        }
    }

    @Override
    public String getPageTitle() {
        return getTranslation("exams.evaluation.page-title");
    }

    private HorizontalLayout createHeader() {
        Button backBtn = new Button(VaadinIcon.ARROW_LEFT.create(),
                e -> getUI().ifPresent(ui -> ui.navigate(AssignmentsView.class)));
        backBtn.setTooltipText(getTranslation("exams.evaluation.back"));

        titleHeading.addClassNames(LumoUtility.FontSize.XLARGE, LumoUtility.Margin.NONE);
        subtitleSpan.addClassNames(LumoUtility.TextColor.SECONDARY, LumoUtility.FontSize.SMALL);

        VerticalLayout titles = new VerticalLayout(titleHeading, subtitleSpan);
        titles.setPadding(false);
        titles.setSpacing(false);

        kpiSummary.getElement().getThemeList().add("badge pill");

        HorizontalLayout header = new HorizontalLayout(backBtn, titles, kpiSummary);
        header.setWidthFull();
        header.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        header.expand(titles);
        return header;
    }

    private HorizontalLayout createActionBar() {
        statusFilter.setPlaceholder(getTranslation("exams.evaluation.filter.status"));
        statusFilter.setItems(StudentWorkStatus.values());
        statusFilter.setItemLabelGenerator(s -> getTranslation("exams.status." + s.name()));
        statusFilter.setClearButtonVisible(true);
        statusFilter.addValueChangeListener(e -> applyFilter());

        batchGradeButton.setText(getTranslation("exams.evaluation.button.batch-grade"));
        batchGradeButton.setIcon(VaadinIcon.MAGIC.create());
        batchGradeButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        batchGradeButton.addClickListener(e -> runBatchGrading());

        Anchor pdfAnchor = createExportAnchor("flaicheckbot-bericht.pdf", "application/pdf",
                getTranslation("exams.evaluation.button.export-pdf"), VaadinIcon.FILE_TEXT, true);
        Anchor csvAnchor = createExportAnchor("flaicheckbot-noten.csv", "text/csv; charset=utf-8",
                getTranslation("exams.evaluation.button.export-csv"), VaadinIcon.TABLE, false);

        HorizontalLayout bar = new HorizontalLayout(statusFilter, batchGradeButton, pdfAnchor, csvAnchor);
        bar.setWidthFull();
        bar.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        bar.getStyle().set("flex-wrap", "wrap");
        return bar;
    }

    private HorizontalLayout createProgressRow() {
        batchProgressBar.setVisible(false);
        batchProgressBar.setWidth("280px");
        batchProgressLabel.setVisible(false);
        batchProgressLabel.addClassNames(LumoUtility.FontSize.SMALL, LumoUtility.TextColor.SECONDARY);

        HorizontalLayout row = new HorizontalLayout(batchProgressBar, batchProgressLabel);
        row.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        return row;
    }

    private Anchor createExportAnchor(
            String fileName,
            String contentType,
            String label,
            VaadinIcon icon,
            boolean isPdf) {
        DownloadHandler handler = DownloadHandler.fromInputStream(event -> {
            byte[] bytes = generateExportBytes(isPdf);
            return new DownloadResponse(new ByteArrayInputStream(bytes), fileName, contentType, bytes.length);
        });
        Anchor anchor = new Anchor(handler, "");
        anchor.getElement().setAttribute("download", true);
        Button button = new Button(label, icon.create());
        anchor.add(button);
        return anchor;
    }

    private byte[] generateExportBytes(boolean isPdf) throws IOException {
        List<StudentWork> currentWorks = assignmentService.findWorks(assignment.getId());
        if (isPdf) {
            return exportService.exportPdf(assignment, schoolClass, testDefinition, currentWorks, studentsById);
        }
        return exportService.exportCsv(assignment, currentWorks, studentsById);
    }

    private void configureGrid() {
        workGrid.addColumn(w -> {
            Student student = studentsById.get(w.getStudentId());
            return student != null ? student.getExternalId() : "-";
        }).setHeader(getTranslation("exams.students.col.external-id")).setAutoWidth(true);

        workGrid.addColumn(w -> {
            Student student = studentsById.get(w.getStudentId());
            return student != null && student.getDisplayAlias() != null ? student.getDisplayAlias() : "";
        }).setHeader(getTranslation("exams.students.col.alias")).setAutoWidth(true);

        workGrid.addColumn(w -> assignmentService.countPages(w.getId()))
                .setHeader(getTranslation("exams.evaluation.col.pages"))
                .setAutoWidth(true);

        workGrid.addColumn(new ComponentRenderer<>(this::renderStatusBadge))
                .setHeader(getTranslation("admin.grid.column.status"))
                .setAutoWidth(true);

        workGrid.addColumn(w -> formatPoints(w))
                .setHeader(getTranslation("exams.evaluation.col.points"))
                .setAutoWidth(true);

        workGrid.addColumn(w -> formatTokens(w))
                .setHeader(getTranslation("exams.evaluation.col.ai-usage"))
                .setAutoWidth(true);

        workGrid.addColumn(new ComponentRenderer<>(this::renderActions))
                .setHeader(getTranslation("admin.grid.column.actions"))
                .setAutoWidth(true);

        workGrid.addThemeVariants(GridVariant.LUMO_ROW_STRIPES, GridVariant.LUMO_COLUMN_BORDERS);
    }

    private Span renderStatusBadge(StudentWork work) {
        Span badge = new Span(getTranslation("exams.status." + work.getStatus().name()));
        badge.getElement().getThemeList().add("badge pill");
        switch (work.getStatus()) {
            case REVIEWED -> badge.getElement().getThemeList().add("success");
            case AI_GRADED -> badge.getElement().getThemeList().add("primary");
            case ERROR -> badge.getElement().getThemeList().add("error");
            default -> badge.getElement().getThemeList().add("contrast");
        }
        return badge;
    }

    private HorizontalLayout renderActions(StudentWork work) {
        Student student = studentsById.get(work.getStudentId());
        Button scansBtn = GridActionButton.create(
                VaadinIcon.FILE_PICTURE,
                getTranslation("exams.evaluation.action.scans"),
                e -> scanDialog.openForWork(work, student, this::refresh));

        Button gradeBtn = GridActionButton.create(
                VaadinIcon.MAGIC,
                getTranslation("exams.evaluation.action.grade-single"),
                e -> runSingleGrading(work));
        gradeBtn.setEnabled(assignmentService.countPages(work.getId()) > 0);

        Button reviewBtn = GridActionButton.create(
                VaadinIcon.EDIT,
                getTranslation("exams.evaluation.action.review"),
                e -> reviewDialog.openForReview(work, student, definedTasks, this::refresh));

        return new HorizontalLayout(scansBtn, gradeBtn, reviewBtn);
    }

    private void runSingleGrading(StudentWork work) {
        UI ui = UI.getCurrent();
        batchProgressBar.setVisible(true);
        batchProgressBar.setIndeterminate(true);
        batchProgressLabel.setVisible(true);
        batchProgressLabel.setText(getTranslation("exams.evaluation.progress.single"));

        batchGradingRunner.gradeSingleAsync(work.getId(), () -> ui.access(() -> {
            batchProgressBar.setVisible(false);
            batchProgressLabel.setVisible(false);
            refresh();
        }));
    }

    private void runBatchGrading() {
        if (assignment == null) {
            return;
        }
        UI ui = UI.getCurrent();
        batchGradeButton.setEnabled(false);
        batchProgressBar.setVisible(true);
        batchProgressBar.setIndeterminate(true);
        batchProgressLabel.setVisible(true);
        batchProgressLabel.setText(getTranslation("exams.evaluation.progress.starting"));

        batchGradingRunner.gradeAssignmentBatchAsync(
                assignment.getId(),
                progress -> ui.access(() -> {
                    batchProgressBar.setIndeterminate(false);
                    batchProgressBar.setMax(Math.max(1, progress.total()));
                    batchProgressBar.setValue(progress.current());
                    batchProgressLabel.setText(getTranslation("exams.evaluation.progress.item",
                            progress.current(), progress.total()));
                    refresh();
                }),
                () -> ui.access(() -> {
                    batchGradeButton.setEnabled(true);
                    batchProgressBar.setVisible(false);
                    batchProgressLabel.setVisible(false);
                    refresh();
                    Notification.show(getTranslation("exams.evaluation.progress.done"), 3000,
                            Notification.Position.MIDDLE);
                }));
    }

    private void refresh() {
        if (assignment == null) {
            return;
        }
        allWorks = assignmentService.findWorks(assignment.getId());
        long reviewed = allWorks.stream().filter(w -> w.getStatus() == StudentWorkStatus.REVIEWED).count();
        long aiGraded = allWorks.stream().filter(w -> w.getStatus() == StudentWorkStatus.AI_GRADED).count();
        long tokens = allWorks.stream().mapToLong(w -> w.getPromptTokens() + w.getCompletionTokens()).sum();

        kpiSummary.setText(getTranslation("exams.evaluation.kpi-summary",
                aiGraded + reviewed, allWorks.size(), reviewed, tokens));
        applyFilter();
    }

    private void applyFilter() {
        StudentWorkStatus filter = statusFilter.getValue();
        if (filter == null) {
            workGrid.setItems(allWorks);
        } else {
            workGrid.setItems(allWorks.stream().filter(w -> w.getStatus() == filter).toList());
        }
    }

    private static String formatPoints(StudentWork work) {
        if (work.getStatus() == StudentWorkStatus.NEW || work.getStatus() == StudentWorkStatus.UPLOADED) {
            return "- / " + work.getMaxPoints().stripTrailingZeros().toPlainString();
        }
        BigDecimal achieved = work.getAchievedPoints() != null ? work.getAchievedPoints() : BigDecimal.ZERO;
        BigDecimal max = work.getMaxPoints() != null ? work.getMaxPoints() : BigDecimal.ZERO;
        String pct = max.signum() > 0
                ? achieved.multiply(new BigDecimal("100")).divide(max, 1, RoundingMode.HALF_UP).toPlainString() + "%"
                : "0%";
        return achieved.stripTrailingZeros().toPlainString() + " / "
                + max.stripTrailingZeros().toPlainString() + " (" + pct + ")";
    }

    private static String formatTokens(StudentWork work) {
        int total = work.getPromptTokens() + work.getCompletionTokens();
        if (total == 0 && work.getModelUsed() == null) {
            return "-";
        }
        return (work.getModelUsed() != null ? work.getModelUsed() + " · " : "") + total + " tok";
    }
}
