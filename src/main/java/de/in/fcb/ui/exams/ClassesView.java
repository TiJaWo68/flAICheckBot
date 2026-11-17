package de.in.fcb.ui.exams;

import java.util.List;
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
import com.vaadin.flow.component.textfield.IntegerField;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.data.renderer.ComponentRenderer;
import com.vaadin.flow.router.HasDynamicTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.exams.ForeignLanguage;
import de.in.fcb.exams.SchoolClass;
import de.in.fcb.exams.SchoolClassService;
import de.in.fcb.exams.Student;
import de.in.fcb.ui.GridActionButton;
import de.in.fcb.ui.layout.MainLayout;
import jakarta.annotation.security.PermitAll;

/**
 * View for managing {@link SchoolClass} groups and their pseudonymized
 * {@link Student} rosters.
 */
@Route(value = "classes", layout = MainLayout.class)
@PermitAll
public class ClassesView extends VerticalLayout implements HasDynamicTitle {

    private final SchoolClassService classService;

    private final Grid<SchoolClass> classGrid = new Grid<>(SchoolClass.class, false);
    private final Grid<Student> studentGrid = new Grid<>(Student.class, false);
    private final Span selectedClassLabel = new Span();
    private final Button addStudentButton = new Button();
    private final Button batchStudentsButton = new Button();

    private SchoolClass selectedClass;

    public ClassesView(SchoolClassService classService) {
        this.classService = classService;

        setSizeFull();
        setPadding(true);
        setSpacing(true);

        configureClassGrid();
        configureStudentGrid();

        VerticalLayout leftPane = createClassesPane();
        VerticalLayout rightPane = createStudentsPane();

        SplitLayout splitLayout = new SplitLayout(leftPane, rightPane);
        splitLayout.setSizeFull();
        splitLayout.setSplitterPosition(48);

        add(splitLayout);
        expand(splitLayout);

        refreshClasses(null);
    }

    @Override
    public String getPageTitle() {
        return getTranslation("exams.classes.page-title");
    }

    private VerticalLayout createClassesPane() {
        H2 title = new H2(getTranslation("exams.classes.title"));
        title.addClassNames(LumoUtility.FontSize.LARGE, LumoUtility.Margin.NONE);

        Button newClassButton = new Button(
                getTranslation("exams.classes.button.new"),
                VaadinIcon.PLUS.create(),
                event -> openClassDialog(new SchoolClass()));
        newClassButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        HorizontalLayout header = new HorizontalLayout(title, newClassButton);
        header.setWidthFull();
        header.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        header.expand(title);

        VerticalLayout pane = new VerticalLayout(header, classGrid);
        pane.setSizeFull();
        pane.setPadding(false);
        pane.setSpacing(true);
        pane.expand(classGrid);
        return pane;
    }

    private VerticalLayout createStudentsPane() {
        H3 title = new H3(getTranslation("exams.students.title"));
        title.addClassNames(LumoUtility.FontSize.LARGE, LumoUtility.Margin.NONE);

        selectedClassLabel.addClassNames(LumoUtility.TextColor.SECONDARY, LumoUtility.FontSize.SMALL);

        addStudentButton.setText(getTranslation("exams.students.button.new"));
        addStudentButton.setIcon(VaadinIcon.USER.create());
        addStudentButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        addStudentButton.addClickListener(event -> {
            if (selectedClass != null) {
                Student student = new Student();
                student.setSchoolClassId(selectedClass.getId());
                openStudentDialog(student);
            }
        });

        batchStudentsButton.setText(getTranslation("exams.students.button.batch"));
        batchStudentsButton.setIcon(VaadinIcon.USERS.create());
        batchStudentsButton.addClickListener(event -> {
            if (selectedClass != null) {
                openBatchDialog(selectedClass);
            }
        });

        HorizontalLayout actions = new HorizontalLayout(batchStudentsButton, addStudentButton);
        HorizontalLayout header = new HorizontalLayout(title, actions);
        header.setWidthFull();
        header.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        header.expand(title);

        VerticalLayout pane = new VerticalLayout(header, selectedClassLabel, studentGrid);
        pane.setSizeFull();
        pane.setPadding(false);
        pane.setSpacing(true);
        pane.expand(studentGrid);
        return pane;
    }

    private void configureClassGrid() {
        classGrid.addColumn(SchoolClass::getName)
                .setHeader(getTranslation("exams.classes.col.name"))
                .setAutoWidth(true);
        classGrid.addColumn(SchoolClass::getAcademicYear)
                .setHeader(getTranslation("exams.classes.col.year"))
                .setAutoWidth(true);
        classGrid.addColumn(c -> c.getForeignLanguage().name())
                .setHeader(getTranslation("exams.classes.col.language"))
                .setAutoWidth(true);
        classGrid.addColumn(c -> classService.countStudents(c.getId()))
                .setHeader(getTranslation("exams.classes.col.student-count"))
                .setAutoWidth(true);
        classGrid.addColumn(new ComponentRenderer<>(this::renderClassActions))
                .setHeader(getTranslation("admin.grid.column.actions"))
                .setAutoWidth(true);

        classGrid.addThemeVariants(GridVariant.LUMO_ROW_STRIPES, GridVariant.LUMO_COLUMN_BORDERS);
        classGrid.setSelectionMode(Grid.SelectionMode.SINGLE);
        classGrid.asSingleSelect().addValueChangeListener(event -> selectClass(event.getValue()));
    }

    private void configureStudentGrid() {
        studentGrid.addColumn(Student::getExternalId)
                .setHeader(getTranslation("exams.students.col.external-id"))
                .setAutoWidth(true);
        studentGrid.addColumn(s -> s.getDisplayAlias() != null ? s.getDisplayAlias() : "")
                .setHeader(getTranslation("exams.students.col.alias"))
                .setAutoWidth(true);
        studentGrid.addColumn(new ComponentRenderer<>(this::renderStudentActions))
                .setHeader(getTranslation("admin.grid.column.actions"))
                .setAutoWidth(true);

        studentGrid.addThemeVariants(GridVariant.LUMO_ROW_STRIPES, GridVariant.LUMO_COLUMN_BORDERS);
    }

    private HorizontalLayout renderClassActions(SchoolClass schoolClass) {
        Button editBtn = GridActionButton.create(VaadinIcon.EDIT, getTranslation("admin.action.edit"),
                e -> openClassDialog(schoolClass));
        Button deleteBtn = GridActionButton.createDanger(VaadinIcon.TRASH, getTranslation("admin.action.delete"),
                e -> confirmDeleteClass(schoolClass));
        return new HorizontalLayout(editBtn, deleteBtn);
    }

    private HorizontalLayout renderStudentActions(Student student) {
        Button editBtn = GridActionButton.create(VaadinIcon.EDIT, getTranslation("admin.action.edit"),
                e -> openStudentDialog(student));
        Button deleteBtn = GridActionButton.createDanger(VaadinIcon.TRASH, getTranslation("admin.action.delete"),
                e -> {
                    classService.deleteStudent(student.getId());
                    refreshStudents();
                    refreshClasses(selectedClass != null ? selectedClass.getId() : null);
                });
        return new HorizontalLayout(editBtn, deleteBtn);
    }

    private void selectClass(SchoolClass schoolClass) {
        this.selectedClass = schoolClass;
        refreshStudents();
    }

    private void refreshClasses(UUID preferredSelectionId) {
        List<SchoolClass> classes = classService.findAllClasses();
        classGrid.setItems(classes);

        SchoolClass toSelect = null;
        if (preferredSelectionId != null) {
            toSelect = classes.stream().filter(c -> preferredSelectionId.equals(c.getId())).findFirst().orElse(null);
        }
        if (toSelect == null && !classes.isEmpty()) {
            toSelect = classes.get(0);
        }
        classGrid.select(toSelect);
        selectClass(toSelect);
    }

    private void refreshStudents() {
        boolean hasClass = selectedClass != null;
        addStudentButton.setEnabled(hasClass);
        batchStudentsButton.setEnabled(hasClass);
        if (!hasClass) {
            selectedClassLabel.setText(getTranslation("exams.students.none-selected"));
            studentGrid.setItems(List.of());
            return;
        }
        selectedClassLabel.setText(selectedClass.getName() + " (" + selectedClass.getAcademicYear() + ", "
                + selectedClass.getForeignLanguage() + ")");
        studentGrid.setItems(classService.findStudents(selectedClass.getId()));
    }

    private void openClassDialog(SchoolClass model) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle(model.getId() == null
                ? getTranslation("exams.classes.dialog.new")
                : getTranslation("exams.classes.dialog.edit"));

        TextField nameField = new TextField(getTranslation("exams.classes.col.name"));
        TextField yearField = new TextField(getTranslation("exams.classes.col.year"));
        TextField gradeField = new TextField(getTranslation("exams.classes.col.grade"));
        ComboBox<ForeignLanguage> langField = new ComboBox<>(getTranslation("exams.classes.col.language"));
        langField.setItems(ForeignLanguage.values());

        nameField.setValue(model.getName() != null ? model.getName() : "");
        yearField.setValue(model.getAcademicYear() != null ? model.getAcademicYear() : "2026/2027");
        gradeField.setValue(model.getGradeLevel() != null ? model.getGradeLevel() : "");
        langField.setValue(model.getForeignLanguage() != null ? model.getForeignLanguage() : ForeignLanguage.EN);

        FormLayout form = new FormLayout(nameField, yearField, gradeField, langField);
        form.setResponsiveSteps(new FormLayout.ResponsiveStep("0", 1));

        Button cancel = new Button(getTranslation("admin.button.cancel"), e -> dialog.close());
        Button save = new Button(getTranslation("admin.settings.button.save"), e -> {
            try {
                model.setName(nameField.getValue());
                model.setAcademicYear(yearField.getValue());
                model.setGradeLevel(gradeField.getValue());
                model.setForeignLanguage(langField.getValue());
                SchoolClass saved = classService.saveClass(model);
                dialog.close();
                refreshClasses(saved.getId());
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

    private void openStudentDialog(Student student) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle(student.getId() == null
                ? getTranslation("exams.students.dialog.new")
                : getTranslation("exams.students.dialog.edit"));

        TextField extIdField = new TextField(getTranslation("exams.students.col.external-id"));
        TextField aliasField = new TextField(getTranslation("exams.students.col.alias"));
        aliasField.setHelperText(getTranslation("exams.students.alias.hint"));

        extIdField.setValue(student.getExternalId() != null ? student.getExternalId() : "");
        aliasField.setValue(student.getDisplayAlias() != null ? student.getDisplayAlias() : "");

        FormLayout form = new FormLayout(extIdField, aliasField);
        form.setResponsiveSteps(new FormLayout.ResponsiveStep("0", 1));

        Button cancel = new Button(getTranslation("admin.button.cancel"), e -> dialog.close());
        Button save = new Button(getTranslation("admin.settings.button.save"), e -> {
            try {
                student.setExternalId(extIdField.getValue());
                student.setDisplayAlias(aliasField.getValue());
                classService.saveStudent(student);
                dialog.close();
                refreshClasses(selectedClass.getId());
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

    private void openBatchDialog(SchoolClass schoolClass) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle(getTranslation("exams.students.batch.title"));

        TextField prefixField = new TextField(getTranslation("exams.students.batch.prefix"));
        prefixField.setValue("S");
        IntegerField fromField = new IntegerField(getTranslation("exams.students.batch.from"));
        fromField.setValue(1);
        fromField.setMin(1);
        IntegerField toField = new IntegerField(getTranslation("exams.students.batch.to"));
        toField.setValue(25);
        toField.setMin(1);

        FormLayout form = new FormLayout(prefixField, fromField, toField);
        form.setResponsiveSteps(new FormLayout.ResponsiveStep("0", 1));

        Button cancel = new Button(getTranslation("admin.button.cancel"), e -> dialog.close());
        Button generate = new Button(getTranslation("exams.students.batch.generate"), e -> {
            try {
                int created = classService.generateStudentsBatch(
                        schoolClass.getId(),
                        prefixField.getValue(),
                        fromField.getValue() != null ? fromField.getValue() : 1,
                        toField.getValue() != null ? toField.getValue() : 25);
                dialog.close();
                refreshClasses(schoolClass.getId());
                Notification.show(getTranslation("exams.students.batch.created", created), 3000,
                        Notification.Position.MIDDLE);
            } catch (IllegalArgumentException ex) {
                Notification.show(ex.getMessage(), 4000, Notification.Position.MIDDLE);
            }
        });
        generate.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        HorizontalLayout actions = new HorizontalLayout(cancel, generate);
        actions.setWidthFull();
        actions.setJustifyContentMode(JustifyContentMode.END);

        dialog.add(form, actions);
        dialog.open();
    }

    private void confirmDeleteClass(SchoolClass schoolClass) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle(getTranslation("admin.action.delete"));
        dialog.add(new Span(getTranslation("exams.classes.confirm-delete", schoolClass.getName())));

        Button cancel = new Button(getTranslation("admin.button.cancel"), e -> dialog.close());
        Button delete = new Button(getTranslation("admin.action.delete"), e -> {
            classService.deleteClass(schoolClass.getId());
            dialog.close();
            refreshClasses(null);
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
