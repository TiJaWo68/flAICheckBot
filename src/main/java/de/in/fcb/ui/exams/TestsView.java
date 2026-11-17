package de.in.fcb.ui.exams;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.dialog.Dialog;
import com.vaadin.flow.component.formlayout.FormLayout;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.grid.GridVariant;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.FlexComponent.JustifyContentMode;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.splitlayout.SplitLayout;
import com.vaadin.flow.component.textfield.BigDecimalField;
import com.vaadin.flow.component.textfield.IntegerField;
import com.vaadin.flow.component.textfield.TextArea;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.component.upload.Upload;
import com.vaadin.flow.data.renderer.ComponentRenderer;
import com.vaadin.flow.router.HasDynamicTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.exams.DocumentTextExtractor;
import de.in.fcb.exams.ForeignLanguage;
import de.in.fcb.exams.TestDefinition;
import de.in.fcb.exams.TestDefinitionService;
import de.in.fcb.exams.TestTask;
import de.in.fcb.ui.GridActionButton;
import de.in.fcb.ui.layout.MainLayout;
import jakarta.annotation.security.PermitAll;

/**
 * View for creating reusable {@link TestDefinition} templates, importing exam
 * documents ({@code .pdf}, {@code .docx}, {@code .txt}, {@code .md}),
 * extracting tasks with AI, and editing {@link TestTask}s.
 */
@Route(value = "tests", layout = MainLayout.class)
@PermitAll
public class TestsView extends VerticalLayout implements HasDynamicTitle {

    private final TestDefinitionService testService;
    private final DocumentTextExtractor textExtractor;

    private final Grid<TestDefinition> testGrid = new Grid<>(TestDefinition.class, false);
    private final Grid<TestTask> taskGrid = new Grid<>(TestTask.class, false);

    private final Span selectedTestInfo = new Span();
    private final TextArea sourceTextArea = new TextArea();
    private final Button saveSourceButton = new Button();
    private final Button aiExtractButton = new Button();
    private final Button addTaskButton = new Button();
    private final Upload docUpload;
    private final Map<String, ByteArrayOutputStream> uploadBuffers = new HashMap<>();

    private TestDefinition selectedTest;

    public TestsView(TestDefinitionService testService, DocumentTextExtractor textExtractor) {
        this.testService = testService;
        this.textExtractor = textExtractor;

        setSizeFull();
        setPadding(true);
        setSpacing(true);

        docUpload = new Upload(this::receiveUpload);
        docUpload.setAcceptedFileTypes(".pdf", ".docx", ".txt", ".md");
        docUpload.setMaxFiles(1);
        docUpload.setUploadButton(new Button(getTranslation("exams.tests.upload.button"), VaadinIcon.UPLOAD.create()));
        docUpload.setDropLabel(new Span(getTranslation("exams.tests.upload.drop")));
        docUpload.setWidthFull();
        docUpload.addSucceededListener(event -> handleDocumentUpload(event.getFileName()));

        configureTestGrid();
        configureTaskGrid();

        SplitLayout splitLayout = new SplitLayout(createTestsPane(), createDetailPane());
        splitLayout.setSizeFull();
        splitLayout.setSplitterPosition(40);

        add(splitLayout);
        expand(splitLayout);

        refreshTests(null);
    }

    @Override
    public String getPageTitle() {
        return getTranslation("exams.tests.page-title");
    }

    private VerticalLayout createTestsPane() {
        H2 title = new H2(getTranslation("exams.tests.title"));
        title.addClassNames(LumoUtility.FontSize.LARGE, LumoUtility.Margin.NONE);

        Button newTestButton = new Button(
                getTranslation("exams.tests.button.new"),
                VaadinIcon.PLUS.create(),
                e -> openTestDialog(new TestDefinition()));
        newTestButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        HorizontalLayout header = new HorizontalLayout(title, newTestButton);
        header.setWidthFull();
        header.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        header.expand(title);

        VerticalLayout pane = new VerticalLayout(header, testGrid);
        pane.setSizeFull();
        pane.setPadding(false);
        pane.setSpacing(true);
        pane.expand(testGrid);
        return pane;
    }

    private VerticalLayout createDetailPane() {
        H3 tasksTitle = new H3(getTranslation("exams.tasks.title"));
        tasksTitle.addClassNames(LumoUtility.FontSize.LARGE, LumoUtility.Margin.NONE);

        selectedTestInfo.addClassNames(LumoUtility.TextColor.SECONDARY, LumoUtility.FontSize.SMALL);

        sourceTextArea.setLabel(getTranslation("exams.tests.source-text"));
        sourceTextArea.setWidthFull();
        sourceTextArea.setHeight("140px");

        saveSourceButton.setText(getTranslation("exams.tests.button.save-source"));
        saveSourceButton.setIcon(VaadinIcon.CHECK.create());
        saveSourceButton.addClickListener(e -> saveSourceText());

        aiExtractButton.setText(getTranslation("exams.tests.button.ai-extract"));
        aiExtractButton.setIcon(VaadinIcon.MAGIC.create());
        aiExtractButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        aiExtractButton.addClickListener(e -> runAiTaskExtraction());

        addTaskButton.setText(getTranslation("exams.tasks.button.new"));
        addTaskButton.setIcon(VaadinIcon.PLUS.create());
        addTaskButton.addClickListener(e -> openNewTaskDialog());

        HorizontalLayout sourceActions = new HorizontalLayout(saveSourceButton, aiExtractButton);
        HorizontalLayout taskHeader = new HorizontalLayout(tasksTitle, addTaskButton);
        taskHeader.setWidthFull();
        taskHeader.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        taskHeader.expand(tasksTitle);

        VerticalLayout pane = new VerticalLayout(
                selectedTestInfo, docUpload, sourceTextArea, sourceActions, taskHeader, taskGrid);
        pane.setSizeFull();
        pane.setPadding(false);
        pane.setSpacing(true);
        pane.expand(taskGrid);
        return pane;
    }

    private void configureTestGrid() {
        testGrid.addColumn(TestDefinition::getTitle)
                .setHeader(getTranslation("exams.tests.col.title"))
                .setAutoWidth(true);
        testGrid.addColumn(t -> t.getForeignLanguage().name())
                .setHeader(getTranslation("exams.classes.col.language"))
                .setAutoWidth(true);
        testGrid.addColumn(t -> testService.countTasks(t.getId()))
                .setHeader(getTranslation("exams.tests.col.task-count"))
                .setAutoWidth(true);
        testGrid.addColumn(t -> testService.sumMaxPoints(t.getId()).stripTrailingZeros().toPlainString())
                .setHeader(getTranslation("exams.tests.col.max-points"))
                .setAutoWidth(true);
        testGrid.addColumn(new ComponentRenderer<>(this::renderTestActions))
                .setHeader(getTranslation("admin.grid.column.actions"))
                .setAutoWidth(true);

        testGrid.addThemeVariants(GridVariant.LUMO_ROW_STRIPES, GridVariant.LUMO_COLUMN_BORDERS);
        testGrid.setSelectionMode(Grid.SelectionMode.SINGLE);
        testGrid.asSingleSelect().addValueChangeListener(e -> selectTest(e.getValue()));
    }

    private void configureTaskGrid() {
        taskGrid.addColumn(TestTask::getPosition)
                .setHeader("#")
                .setWidth("60px")
                .setFlexGrow(0);
        taskGrid.addColumn(TestTask::getTitle)
                .setHeader(getTranslation("exams.tasks.col.title"))
                .setAutoWidth(true);
        taskGrid.addColumn(TestTask::getTaskText)
                .setHeader(getTranslation("exams.tasks.col.prompt"))
                .setFlexGrow(1);
        taskGrid.addColumn(t -> t.getMaxPoints().stripTrailingZeros().toPlainString())
                .setHeader(getTranslation("exams.tasks.col.points"))
                .setWidth("90px")
                .setFlexGrow(0);
        taskGrid.addColumn(new ComponentRenderer<>(this::renderTaskActions))
                .setHeader(getTranslation("admin.grid.column.actions"))
                .setAutoWidth(true);

        taskGrid.addThemeVariants(GridVariant.LUMO_ROW_STRIPES, GridVariant.LUMO_COLUMN_BORDERS);
    }

    private HorizontalLayout renderTestActions(TestDefinition test) {
        Button editBtn = GridActionButton.create(VaadinIcon.EDIT, getTranslation("admin.action.edit"),
                e -> openTestDialog(test));
        Button deleteBtn = GridActionButton.createDanger(VaadinIcon.TRASH, getTranslation("admin.action.delete"),
                e -> {
                    try {
                        testService.deleteTest(test.getId());
                        refreshTests(null);
                        Notification.show(getTranslation("admin.notification.deleted"), 3000,
                                Notification.Position.MIDDLE);
                    } catch (IllegalArgumentException ex) {
                        Notification.show(ex.getMessage(), 5000, Notification.Position.MIDDLE);
                    }
                });
        return new HorizontalLayout(editBtn, deleteBtn);
    }

    private HorizontalLayout renderTaskActions(TestTask task) {
        Button editBtn = GridActionButton.create(VaadinIcon.EDIT, getTranslation("admin.action.edit"),
                e -> openTaskDialog(task));
        Button deleteBtn = GridActionButton.createDanger(VaadinIcon.TRASH, getTranslation("admin.action.delete"),
                e -> {
                    testService.deleteTask(task.getId());
                    refreshDetail();
                    refreshTests(selectedTest != null ? selectedTest.getId() : null);
                });
        return new HorizontalLayout(editBtn, deleteBtn);
    }

    private void selectTest(TestDefinition test) {
        this.selectedTest = test;
        refreshDetail();
    }

    private void refreshTests(UUID preferredId) {
        List<TestDefinition> tests = testService.findAllTests();
        testGrid.setItems(tests);

        TestDefinition toSelect = null;
        if (preferredId != null) {
            toSelect = tests.stream().filter(t -> preferredId.equals(t.getId())).findFirst().orElse(null);
        }
        if (toSelect == null && !tests.isEmpty()) {
            toSelect = tests.get(0);
        }
        testGrid.select(toSelect);
        selectTest(toSelect);
    }

    private void refreshDetail() {
        boolean hasTest = selectedTest != null;
        docUpload.setVisible(hasTest);
        sourceTextArea.setEnabled(hasTest);
        saveSourceButton.setEnabled(hasTest);
        aiExtractButton.setEnabled(hasTest);
        addTaskButton.setEnabled(hasTest);

        if (!hasTest) {
            selectedTestInfo.setText(getTranslation("exams.tests.none-selected"));
            sourceTextArea.clear();
            taskGrid.setItems(List.of());
            return;
        }
        selectedTestInfo.setText(selectedTest.getTitle() + " · " + selectedTest.getForeignLanguage()
                + (selectedTest.getLearningUnit() != null ? " · " + selectedTest.getLearningUnit() : ""));
        sourceTextArea.setValue(selectedTest.getSourceText() != null ? selectedTest.getSourceText() : "");
        taskGrid.setItems(testService.findTasks(selectedTest.getId()));
    }

    private OutputStream receiveUpload(String fileName, String mimeType) {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        uploadBuffers.put(fileName, buffer);
        return buffer;
    }

    private void handleDocumentUpload(String fileName) {
        ByteArrayOutputStream buffer = uploadBuffers.remove(fileName);
        docUpload.clearFileList();
        if (buffer == null || selectedTest == null) {
            return;
        }
        try {
            String extracted = textExtractor.extractText(fileName, buffer.toByteArray());
            selectedTest.setSourceText(extracted);
            selectedTest = testService.saveTest(selectedTest);
            sourceTextArea.setValue(extracted);
            Notification.show(getTranslation("exams.tests.upload.success", fileName), 3000,
                    Notification.Position.MIDDLE);
        } catch (Exception ex) {
            Notification.show(getTranslation("admin.notification.error", ex.getMessage()), 5000,
                    Notification.Position.MIDDLE);
        }
    }

    private void saveSourceText() {
        if (selectedTest == null) {
            return;
        }
        selectedTest.setSourceText(sourceTextArea.getValue());
        selectedTest = testService.saveTest(selectedTest);
        Notification.show(getTranslation("admin.notification.saved"), 3000, Notification.Position.MIDDLE);
    }

    private void runAiTaskExtraction() {
        if (selectedTest == null) {
            return;
        }
        selectedTest.setSourceText(sourceTextArea.getValue());
        selectedTest = testService.saveTest(selectedTest);
        try {
            List<TestTask> tasks = testService.extractTasksWithAi(selectedTest.getId());
            refreshTests(selectedTest.getId());
            Notification.show(getTranslation("exams.tests.ai-extract.success", tasks.size()), 4000,
                    Notification.Position.MIDDLE);
        } catch (RuntimeException ex) {
            Notification.show(getTranslation("admin.notification.error", ex.getMessage()), 5000,
                    Notification.Position.MIDDLE);
        }
    }

    private void openTestDialog(TestDefinition model) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle(model.getId() == null
                ? getTranslation("exams.tests.dialog.new")
                : getTranslation("exams.tests.dialog.edit"));

        TextField titleField = new TextField(getTranslation("exams.tests.col.title"));
        ComboBox<ForeignLanguage> langField = new ComboBox<>(getTranslation("exams.classes.col.language"));
        langField.setItems(ForeignLanguage.values());
        TextField gradeField = new TextField(getTranslation("exams.classes.col.grade"));
        TextField unitField = new TextField(getTranslation("exams.tests.col.unit"));

        titleField.setValue(model.getTitle() != null ? model.getTitle() : "");
        langField.setValue(model.getForeignLanguage() != null ? model.getForeignLanguage() : ForeignLanguage.EN);
        gradeField.setValue(model.getGradeLevel() != null ? model.getGradeLevel() : "");
        unitField.setValue(model.getLearningUnit() != null ? model.getLearningUnit() : "");

        FormLayout form = new FormLayout(titleField, langField, gradeField, unitField);
        form.setResponsiveSteps(new FormLayout.ResponsiveStep("0", 1));

        Button cancel = new Button(getTranslation("admin.button.cancel"), e -> dialog.close());
        Button save = new Button(getTranslation("admin.settings.button.save"), e -> {
            try {
                model.setTitle(titleField.getValue());
                model.setForeignLanguage(langField.getValue());
                model.setGradeLevel(gradeField.getValue());
                model.setLearningUnit(unitField.getValue());
                TestDefinition saved = testService.saveTest(model);
                dialog.close();
                refreshTests(saved.getId());
                Notification.show(getTranslation("admin.notification.saved"), 3000, Notification.Position.MIDDLE);
            } catch (IllegalArgumentException ex) {
                Notification.show(ex.getMessage(), 4000, Notification.Position.MIDDLE);
            }
        });
        save.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        HorizontalLayout actions = new HorizontalLayout(cancel, save);
        actions.setWidthFull();
        actions.setJustifyContentMode(JustifyContentMode.END);
        dialog.add(form, actions);
        dialog.open();
    }

    private void openNewTaskDialog() {
        if (selectedTest == null) {
            return;
        }
        List<TestTask> existing = testService.findTasks(selectedTest.getId());
        int nextPos = existing.isEmpty() ? 1 : existing.get(existing.size() - 1).getPosition() + 1;
        TestTask task = new TestTask();
        task.setTestDefinitionId(selectedTest.getId());
        task.setPosition(nextPos);
        task.setMaxPoints(new BigDecimal("10.00"));
        openTaskDialog(task);
    }

    private void openTaskDialog(TestTask task) {
        Dialog dialog = new Dialog();
        dialog.setWidth("640px");
        dialog.setHeaderTitle(task.getId() == null
                ? getTranslation("exams.tasks.dialog.new")
                : getTranslation("exams.tasks.dialog.edit"));

        IntegerField posField = new IntegerField(getTranslation("exams.tasks.col.position"));
        posField.setValue(task.getPosition());
        posField.setMin(1);

        TextField titleField = new TextField(getTranslation("exams.tasks.col.title"));
        titleField.setValue(task.getTitle() != null ? task.getTitle() : "");

        BigDecimalField pointsField = new BigDecimalField(getTranslation("exams.tasks.col.points"));
        pointsField.setValue(task.getMaxPoints() != null ? task.getMaxPoints() : new BigDecimal("10.00"));

        TextArea promptField = new TextArea(getTranslation("exams.tasks.col.prompt"));
        promptField.setValue(task.getTaskText() != null ? task.getTaskText() : "");
        promptField.setHeight("100px");

        TextArea referenceField = new TextArea(getTranslation("exams.tasks.col.reference"));
        referenceField.setValue(task.getReferenceText() != null ? task.getReferenceText() : "");
        referenceField.setHeight("100px");

        TextArea hintsField = new TextArea(getTranslation("exams.tasks.col.hints"));
        hintsField.setValue(task.getGradingHints() != null ? task.getGradingHints() : "");
        hintsField.setHeight("80px");

        FormLayout form = new FormLayout(posField, titleField, pointsField, promptField, referenceField, hintsField);
        form.setResponsiveSteps(new FormLayout.ResponsiveStep("0", 1));

        Button cancel = new Button(getTranslation("admin.button.cancel"), e -> dialog.close());
        Button save = new Button(getTranslation("admin.settings.button.save"), e -> {
            try {
                task.setPosition(posField.getValue() != null ? posField.getValue() : 1);
                task.setTitle(titleField.getValue());
                task.setMaxPoints(pointsField.getValue());
                task.setTaskText(promptField.getValue());
                task.setReferenceText(referenceField.getValue());
                task.setGradingHints(hintsField.getValue());
                testService.saveTask(task);
                dialog.close();
                refreshTests(selectedTest.getId());
                Notification.show(getTranslation("admin.notification.saved"), 3000, Notification.Position.MIDDLE);
            } catch (IllegalArgumentException ex) {
                Notification.show(ex.getMessage(), 4000, Notification.Position.MIDDLE);
            }
        });
        save.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        HorizontalLayout actions = new HorizontalLayout(cancel, save);
        actions.setWidthFull();
        actions.setJustifyContentMode(JustifyContentMode.END);
        dialog.add(form, actions);
        dialog.open();
    }
}
