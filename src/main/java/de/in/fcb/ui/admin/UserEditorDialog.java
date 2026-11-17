package de.in.fcb.ui.admin;

import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.checkbox.Checkbox;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.combobox.MultiSelectComboBox;
import com.vaadin.flow.component.dialog.Dialog;
import com.vaadin.flow.component.formlayout.FormLayout;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.FlexComponent.JustifyContentMode;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.textfield.PasswordField;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.data.binder.Binder;

import de.in.fcb.admin.UserService;
import de.in.fcb.security.AppRole;
import de.in.fcb.security.AppUser;
import de.in.fcb.security.AuthProvider;

/**
 * Modal editor for creating and updating a technical {@link AppUser} account.
 */
public class UserEditorDialog extends Dialog {

    private final UserService userService;

    private final Binder<AppUser> binder = new Binder<>(AppUser.class);

    private final TextField username = new TextField();
    private final TextField email = new TextField();
    private final ComboBox<AuthProvider> authProvider = new ComboBox<>();
    private final MultiSelectComboBox<AppRole> roles = new MultiSelectComboBox<>();
    private final Checkbox enabled = new Checkbox();
    private final PasswordField password = new PasswordField();

    private final Button saveButton = new Button();
    private final Button cancelButton = new Button();

    private AppUser currentUser;
    private Runnable onSaved;

    public UserEditorDialog(UserService userService) {
        this.userService = userService;

        configureFields();
        configureBinder();

        authProvider.addValueChangeListener(event -> updateFieldVisibility());

        saveButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        saveButton.addClickListener(event -> save());
        cancelButton.addClickListener(event -> close());

        FormLayout form = new FormLayout(username, email, authProvider, roles, enabled, password);
        form.setResponsiveSteps(new FormLayout.ResponsiveStep("0", 1));

        HorizontalLayout actions = new HorizontalLayout(cancelButton, saveButton);
        actions.setWidthFull();
        actions.setJustifyContentMode(JustifyContentMode.END);

        add(form, actions);
        setCloseOnEsc(true);
        setCloseOnOutsideClick(false);
    }

    public void openForCreate(Runnable onSaved) {
        this.onSaved = onSaved;
        this.currentUser = new AppUser();
        setHeaderTitle(getTranslation("admin.user.editor.title.new"));
        binder.readBean(currentUser);
        password.clear();
        updateFieldVisibility();
        open();
    }

    public void openForEdit(AppUser user, Runnable onSaved) {
        this.onSaved = onSaved;
        this.currentUser = user;
        setHeaderTitle(getTranslation("admin.user.editor.title.edit"));
        binder.readBean(user);
        password.clear();
        updateFieldVisibility();
        open();
    }

    private void configureFields() {
        username.setLabel(getTranslation("admin.user.editor.field.username"));
        email.setLabel(getTranslation("admin.user.editor.field.email"));
        authProvider.setLabel(getTranslation("admin.user.editor.field.auth-provider"));
        roles.setLabel(getTranslation("admin.user.editor.field.roles"));
        enabled.setLabel(getTranslation("admin.user.editor.field.enabled"));
        password.setLabel(getTranslation("admin.user.editor.field.password"));
        saveButton.setText(getTranslation("admin.user.editor.button.save"));
        cancelButton.setText(getTranslation("admin.user.editor.button.cancel"));

        authProvider.setItems(AuthProvider.values());
        authProvider.setItemLabelGenerator(provider -> getTranslation("admin.auth." + provider.name()));

        roles.setItems(AppRole.values());
        roles.setItemLabelGenerator(role -> getTranslation("admin.role." + role.name()));
    }

    private void configureBinder() {
        binder.forField(username)
                .asRequired(getTranslation("admin.user.editor.error.username-required"))
                .bind(AppUser::getUsername, AppUser::setUsername);

        binder.forField(email)
                .asRequired(getTranslation("admin.user.editor.error.email-required"))
                .bind(AppUser::getEmail, AppUser::setEmail);

        binder.forField(authProvider)
                .asRequired(getTranslation("admin.user.editor.error.auth-provider-required"))
                .bind(AppUser::getAuthProvider, AppUser::setAuthProvider);

        binder.forField(roles)
                .bind(AppUser::getRoles, AppUser::setRoles);

        binder.forField(enabled)
                .bind(AppUser::isEnabled, AppUser::setEnabled);
    }

    private void updateFieldVisibility() {
        boolean local = AuthProvider.LOCAL.equals(authProvider.getValue());
        password.setVisible(local);
        if (!local) {
            password.clear();
        }
    }

    private void save() {
        if (!binder.writeBeanIfValid(currentUser)) {
            return;
        }

        String passwordValue = password.getValue();
        if (currentUser.getId() == null
                && AuthProvider.LOCAL.equals(currentUser.getAuthProvider())
                && (passwordValue == null || passwordValue.isBlank())) {
            Notification.show(getTranslation("admin.user.editor.error.password-required"), 4000,
                    Notification.Position.MIDDLE);
            return;
        }

        try {
            if (currentUser.getId() == null) {
                userService.create(currentUser, passwordValue);
            } else {
                userService.update(currentUser, passwordValue);
            }
            close();
            if (onSaved != null) {
                onSaved.run();
            }
        } catch (IllegalArgumentException exception) {
            Notification.show(getTranslation("admin.notification.error", exception.getMessage()), 5000,
                    Notification.Position.MIDDLE);
        }
    }
}
