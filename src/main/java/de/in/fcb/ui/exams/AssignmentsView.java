package de.in.fcb.ui.exams;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.datepicker.DatePicker;
import com.vaadin.flow.component.dialog.Dialog;
import com.vaadin.flow.component.formlayout.FormLayout;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.grid.GridVariant;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.FlexComponent.JustifyContentMode;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.data.renderer.ComponentRenderer;
import com.vaadin.flow.router.HasDynamicTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.exams.Assignment;
import de.in.fcb.exams.AssignmentService;
import de.in.fcb.exams.SchoolClass;
import de.in.fcb.exams.SchoolClassService;
import de.in.fcb.exams.StudentWork;
import de.in.fcb.exams.StudentWorkStatus;
import de.in.fcb.exams.TestDefinition;
import de.in.fcb.exams.TestDefinitionService;
import de.in.fcb.i18n.DateFormatter;
import de.in.fcb.ui.GridActionButton;
import de.in.fcb.ui.layout.MainLayout;
import jakarta.annotation.security.PermitAll;

/**
 * View for creating and opening exam {@link Assignment}s linking a
 * {@link SchoolClass} with a {@link TestDefinition}.
 */
@Route(value = "assignments", layout = MainLayout.class)
@PermitAll
public class AssignmentsView extends VerticalLayout implements HasDynamicTitle {

    private final AssignmentService assignmentService;
    private final SchoolClassService classService;
    private final TestDefinitionService testService;

    private final Grid<Assignment> grid = new Grid<>(Assignment.class, false);

    private Map<UUID, String> classNames = Map.of();
    private Map<UUID, String> testTitles = Map.of();

    public AssignmentsView(
            AssignmentService assignmentService,
            SchoolClassService classService,
            TestDefinitionService testService) {
        this.assignmentService = assignmentService;
        this.classService = classService;
        this.testService = testService;

        setSizeFull();
        setPadding(true);
        setSpacing(true);

        add(createHeader(), grid);
        configureGrid();
        expand(grid);
        refresh();
    }

    @Override
    public String getPageTitle() {
        return getTranslation("exams.assignments.page-title");
    }

    private HorizontalLayout createHeader() {
        H2 title = new H2(getTranslation("exams.assignments.title"));
        title.addClassNames(LumoUtility.FontSize.XLARGE, LumoUtility.Margin.NONE);

        Button newButton = new Button(
                getTranslation("exams.assignments.button.new"),
                VaadinIcon.PLUS.create(),
                e -> openCreateDialog());
        newButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        HorizontalLayout header = new HorizontalLayout(title, newButton);
        header.setWidthFull();
        header.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        header.expand(title);
        return header;
    }

    private void configureGrid() {
        grid.addColumn(Assignment::getTitle)
                .setHeader(getTranslation("exams.assignments.col.title"))
                .setAutoWidth(true);
        grid.addColumn(a -> classNames.getOrDefault(a.getSchoolClassId(), "-"))
                .setHeader(getTranslation("exams.assignments.col.class"))
                .setAutoWidth(true);
        grid.addColumn(a -> testTitles.getOrDefault(a.getTestDefinitionId(), "-"))
                .setHeader(getTranslation("exams.assignments.col.test"))
                .setAutoWidth(true);
        grid.addColumn(a -> DateFormatter.format(a.getExamDate(), getLocale()))
                .setHeader(getTranslation("exams.assignments.col.date"))
                .setAutoWidth(true);
        grid.addColumn(this::formatProgress)
                .setHeader(getTranslation("exams.assignments.col.progress"))
                .setAutoWidth(true);
        grid.addColumn(new ComponentRenderer<>(this::renderActions))
                .setHeader(getTranslation("admin.grid.column.actions"))
                .setAutoWidth(true);

        grid.addThemeVariants(GridVariant.LUMO_ROW_STRIPES, GridVariant.LUMO_COLUMN_BORDERS);
        grid.addItemDoubleClickListener(e -> openEvaluation(e.getItem()));
    }

    private HorizontalLayout renderActions(Assignment assignment) {
        Button openBtn = GridActionButton.create(
                VaadinIcon.ARROW_RIGHT,
                getTranslation("exams.assignments.action.open"),
                e -> openEvaluation(assignment));
        Button deleteBtn = GridActionButton.createDanger(
                VaadinIcon.TRASH,
                getTranslation("admin.action.delete"),
                e -> confirmDelete(assignment));
        return new HorizontalLayout(openBtn, deleteBtn);
    }

    private void openEvaluation(Assignment assignment) {
        getUI().ifPresent(ui -> ui.navigate(EvaluationView.class, assignment.getId().toString()));
    }

    private String formatProgress(Assignment assignment) {
        List<StudentWork> works = assignmentService.findWorks(assignment.getId());
        long graded = works.stream()
                .filter(w -> w.getStatus() == StudentWorkStatus.AI_GRADED
                        || w.getStatus() == StudentWorkStatus.REVIEWED)
                .count();
        return graded + " / " + works.size();
    }

    private void refresh() {
        classNames = classService.findAllClasses().stream()
                .collect(Collectors.toMap(SchoolClass::getId, SchoolClass::getName));
        testTitles = testService.findAllTests().stream()
                .collect(Collectors.toMap(TestDefinition::getId, TestDefinition::getTitle));
        grid.setItems(assignmentService.findAllAssignments());
    }

    private void openCreateDialog() {
        List<SchoolClass> classes = classService.findAllClasses();
        List<TestDefinition> tests = testService.findAllTests();
        if (classes.isEmpty() || tests.isEmpty()) {
            Notification.show(getTranslation("exams.assignments.error.missing-masterdata"), 4000,
                    Notification.Position.MIDDLE);
            return;
        }

        Dialog dialog = new Dialog();
        dialog.setHeaderTitle(getTranslation("exams.assignments.dialog.new"));

        TextField titleField = new TextField(getTranslation("exams.assignments.col.title"));
        ComboBox<SchoolClass> classBox = new ComboBox<>(getTranslation("exams.assignments.col.class"));
        classBox.setItems(classes);
        classBox.setItemLabelGenerator(SchoolClass::getName);
        classBox.setValue(classes.get(0));

        ComboBox<TestDefinition> testBox = new ComboBox<>(getTranslation("exams.assignments.col.test"));
        testBox.setItems(tests);
        testBox.setItemLabelGenerator(TestDefinition::getTitle);
        testBox.setValue(tests.get(0));

        DatePicker datePicker = new DatePicker(getTranslation("exams.assignments.col.date"));
        datePicker.setValue(LocalDate.now());

        FormLayout form = new FormLayout(titleField, classBox, testBox, datePicker);
        form.setResponsiveSteps(new FormLayout.ResponsiveStep("0", 1));

        Button cancel = new Button(getTranslation("admin.button.cancel"), e -> dialog.close());
        Button save = new Button(getTranslation("admin.settings.button.save"), e -> {
            try {
                Assignment saved = assignmentService.createAssignment(
                        titleField.getValue(),
                        classBox.getValue() != null ? classBox.getValue().getId() : null,
                        testBox.getValue() != null ? testBox.getValue().getId() : null,
                        datePicker.getValue());
                dialog.close();
                refresh();
                openEvaluation(saved);
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

    private void confirmDelete(Assignment assignment) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle(getTranslation("admin.action.delete"));
        dialog.add(new Span(getTranslation("exams.assignments.confirm-delete", assignment.getTitle())));

        Button cancel = new Button(getTranslation("admin.button.cancel"), e -> dialog.close());
        Button delete = new Button(getTranslation("admin.action.delete"), e -> {
            assignmentService.deleteAssignment(assignment.getId());
            dialog.close();
            refresh();
            Notification.show(getTranslation("admin.notification.deleted"), 3000, Notification.Position.MIDDLE);
        });
        delete.addThemeVariants(ButtonVariant.LUMO_PRIMARY, ButtonVariant.LUMO_ERROR);

        HorizontalLayout actions = new HorizontalLayout(cancel, delete);
        actions.setWidthFull();
        actions.setJustifyContentMode(JustifyContentMode.END);
        dialog.add(actions);
        dialog.open();
    }
}
