package de.in.fcb.ui.admin;

import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;
import java.util.Locale;
import java.util.UUID;

import com.vaadin.flow.component.Component;
import com.vaadin.flow.component.UI;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.dialog.Dialog;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.grid.GridVariant;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.FlexComponent.JustifyContentMode;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.PasswordField;
import com.vaadin.flow.data.renderer.ComponentRenderer;
import com.vaadin.flow.spring.security.AuthenticationContext;

import de.in.fcb.admin.UserService;
import de.in.fcb.security.AppRole;
import de.in.fcb.security.AppUser;
import de.in.fcb.security.AuthProvider;
import de.in.fcb.ui.GridActionButton;

/**
 * User accounts, roles and access management tab.
 */
public class UsersTab extends VerticalLayout implements AdminTab {

    private final UserService userService;
    private final AuthenticationContext authenticationContext;
    private final UserEditorDialog editorDialog;

    private final Grid<AppUser> grid = new Grid<>(AppUser.class, false);

    private UUID currentUserId;

    public UsersTab(UserService userService, AuthenticationContext authenticationContext) {
        this.userService = userService;
        this.authenticationContext = authenticationContext;
        this.editorDialog = new UserEditorDialog(userService);

        setSizeFull();
        setPadding(false);
        setSpacing(false);

        add(createHeader(), grid);
        configureGrid();
    }

    @Override
    public String getLabelKey() {
        return "admin.tab.users";
    }

    @Override
    public Component asComponent() {
        return this;
    }

    @Override
    public void refresh() {
        currentUserId = resolveCurrentUserId();
        grid.setItems(userService.findAll());
    }

    private HorizontalLayout createHeader() {
        Span title = new Span(getTranslation("admin.users.title"));
        title.addClassName("admin-tab-title");

        Button newUserButton = new Button(
                getTranslation("admin.users.button.new-user"),
                VaadinIcon.PLUS.create());
        newUserButton.setTooltipText(getTranslation("admin.users.button.new-user"));
        newUserButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        newUserButton.addClickListener(event -> editorDialog.openForCreate(this::refresh));

        HorizontalLayout header = new HorizontalLayout(title, newUserButton);
        header.setWidthFull();
        header.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        header.expand(title);
        return header;
    }

    private void configureGrid() {
        grid.addColumn(AppUser::getUsername)
                .setHeader(getTranslation("admin.grid.column.username"))
                .setAutoWidth(true);

        grid.addColumn(AppUser::getEmail)
                .setHeader(getTranslation("admin.grid.column.email"))
                .setAutoWidth(true);

        grid.addColumn(new ComponentRenderer<>(this::renderRoleBadges))
                .setHeader(getTranslation("admin.grid.column.roles"))
                .setAutoWidth(true);

        grid.addColumn(new ComponentRenderer<>(this::renderAuthProvider))
                .setHeader(getTranslation("admin.grid.column.auth-provider"))
                .setAutoWidth(true);

        grid.addColumn(new ComponentRenderer<>(this::renderStatus))
                .setHeader(getTranslation("admin.grid.column.status"))
                .setAutoWidth(true);

        grid.addColumn(user -> formatCreatedAt(user.getCreatedAt()))
                .setHeader(getTranslation("admin.grid.column.created-at"))
                .setAutoWidth(true);

        grid.addColumn(new ComponentRenderer<>(this::renderActions))
                .setHeader(getTranslation("admin.grid.column.actions"))
                .setAutoWidth(true);

        grid.addThemeVariants(GridVariant.LUMO_ROW_STRIPES, GridVariant.LUMO_COLUMN_BORDERS);
    }

    private Component renderRoleBadges(AppUser user) {
        HorizontalLayout badges = new HorizontalLayout();
        badges.setSpacing(false);
        for (AppRole role : user.getRoles()) {
            Span badge = new Span(getTranslation("admin.role." + role.name()));
            badge.getElement().getThemeList().add("badge");
            badge.getElement().getThemeList().add("pill");
            badge.getElement().getThemeList().add("success");
            badges.add(badge);
        }
        return badges;
    }

    private Span renderAuthProvider(AppUser user) {
        Span badge = new Span(getTranslation("admin.auth." + user.getAuthProvider().name()));
        badge.getElement().getThemeList().add("badge");
        badge.getElement().getThemeList().add("pill");
        badge.getElement().getThemeList().add("contrast");
        return badge;
    }

    private Span renderStatus(AppUser user) {
        String key = user.isEnabled() ? "admin.user.status.enabled" : "admin.user.status.disabled";
        Span badge = new Span(getTranslation(key));
        badge.getElement().getThemeList().add("badge");
        badge.getElement().getThemeList().add("pill");
        badge.getElement().getThemeList().add(user.isEnabled() ? "success" : "error");
        return badge;
    }

    private HorizontalLayout renderActions(AppUser user) {
        boolean self = user.getId() != null && user.getId().equals(currentUserId);

        Button editButton = GridActionButton.create(
                VaadinIcon.EDIT,
                getTranslation("admin.action.edit"),
                event -> editorDialog.openForEdit(user, this::refresh));

        Button resetButton = GridActionButton.create(
                VaadinIcon.KEY,
                getTranslation("admin.user.action.reset-password"),
                event -> openResetPasswordDialog(user));
        resetButton.setEnabled(!self && user.getAuthProvider() == AuthProvider.LOCAL);

        String toggleKey = user.isEnabled() ? "admin.user.action.disable" : "admin.user.action.enable";
        VaadinIcon toggleIcon = user.isEnabled() ? VaadinIcon.LOCK : VaadinIcon.UNLOCK;
        Button toggleButton = GridActionButton.create(
                toggleIcon,
                getTranslation(toggleKey),
                event -> toggleEnabled(user));
        toggleButton.setEnabled(!self);

        Button deleteButton = GridActionButton.createDanger(
                VaadinIcon.TRASH,
                getTranslation("admin.action.delete"),
                event -> confirmDelete(user));
        deleteButton.setEnabled(!self);

        return new HorizontalLayout(editButton, resetButton, toggleButton, deleteButton);
    }

    private void toggleEnabled(AppUser user) {
        try {
            userService.setEnabled(user.getId(), !user.isEnabled(), currentUserId);
            Notification.show(getTranslation("admin.notification.saved"), 3000, Notification.Position.MIDDLE);
            refresh();
        } catch (IllegalArgumentException exception) {
            Notification.show(getTranslation("admin.notification.error", exception.getMessage()), 5000,
                    Notification.Position.MIDDLE);
        }
    }

    private void confirmDelete(AppUser user) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle(getTranslation("admin.user.confirm.delete.title"));

        VerticalLayout content = new VerticalLayout(
                new Span(getTranslation("admin.user.confirm.delete.text", user.getUsername())));

        Button cancelButton = new Button(getTranslation("admin.button.cancel"), event -> dialog.close());
        Button deleteButton = new Button(getTranslation("admin.action.delete"), event -> {
            dialog.close();
            deleteUser(user);
        });
        deleteButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY, ButtonVariant.LUMO_ERROR);

        HorizontalLayout actions = new HorizontalLayout(cancelButton, deleteButton);
        actions.setWidthFull();
        actions.setJustifyContentMode(JustifyContentMode.END);

        dialog.add(content, actions);
        dialog.open();
    }

    private void deleteUser(AppUser user) {
        try {
            userService.delete(user.getId(), currentUserId);
            Notification.show(getTranslation("admin.notification.deleted"), 3000, Notification.Position.MIDDLE);
            refresh();
        } catch (IllegalArgumentException exception) {
            Notification.show(getTranslation("admin.notification.error", exception.getMessage()), 5000,
                    Notification.Position.MIDDLE);
        }
    }

    private void openResetPasswordDialog(AppUser user) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle(getTranslation("admin.user.reset-password.title", user.getUsername()));

        PasswordField newPassword = new PasswordField(getTranslation("admin.user.reset-password.field"));
        newPassword.setWidthFull();

        Button cancelButton = new Button(getTranslation("admin.button.cancel"), event -> dialog.close());
        Button resetButton = new Button(getTranslation("admin.user.action.reset-password"), event -> {
            try {
                userService.resetPassword(user.getId(), newPassword.getValue());
                dialog.close();
                Notification.show(getTranslation("admin.notification.saved"), 3000, Notification.Position.MIDDLE);
            } catch (IllegalArgumentException exception) {
                Notification.show(getTranslation("admin.notification.error", exception.getMessage()), 5000,
                        Notification.Position.MIDDLE);
            }
        });
        resetButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        HorizontalLayout actions = new HorizontalLayout(cancelButton, resetButton);
        actions.setWidthFull();
        actions.setJustifyContentMode(JustifyContentMode.END);

        dialog.add(newPassword, actions);
        dialog.open();
    }

    private UUID resolveCurrentUserId() {
        return authenticationContext.getPrincipalName()
                .map(userService::findByUsername)
                .map(AppUser::getId)
                .orElse(null);
    }

    private String formatCreatedAt(OffsetDateTime createdAt) {
        if (createdAt == null) {
            return "";
        }
        Locale locale = UI.getCurrent() != null ? UI.getCurrent().getLocale() : Locale.GERMAN;
        return DateTimeFormatter.ofLocalizedDateTime(FormatStyle.SHORT).withLocale(locale).format(createdAt);
    }
}
