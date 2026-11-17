package de.in.fcb.ui.admin;

import com.vaadin.flow.component.Component;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.formlayout.FormLayout;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.TextField;

import de.in.fcb.admin.SystemHealth;
import de.in.fcb.admin.SystemHealthService;

/**
 * Read-only runtime diagnostics tab for the administrator.
 */
public class SystemHealthTab extends VerticalLayout implements AdminTab {

    private final SystemHealthService systemHealthService;

    private final TextField version = new TextField();
    private final TextField activeProfile = new TextField();
    private final TextField javaRuntime = new TextField();
    private final TextField databaseStatus = new TextField();
    private final TextField lastFlywayMigration = new TextField();
    private final TextField loggingConfig = new TextField();

    public SystemHealthTab(SystemHealthService systemHealthService) {
        this.systemHealthService = systemHealthService;

        setSizeFull();
        setPadding(false);
        setSpacing(false);

        add(createHeader(), createForm());
    }

    @Override
    public String getLabelKey() {
        return "admin.tab.health";
    }

    @Override
    public Component asComponent() {
        return this;
    }

    @Override
    public void refresh() {
        SystemHealth health = systemHealthService.collect();

        version.setValue(health.version());
        activeProfile.setValue(health.activeProfile());
        javaRuntime.setValue(health.javaRuntime());
        databaseStatus.setValue(health.databaseOnline()
                ? getTranslation("admin.health.db.online", health.databaseLatencyMs())
                : getTranslation("admin.health.db.offline"));
        lastFlywayMigration.setValue(health.lastFlywayMigration() != null ? health.lastFlywayMigration() : "-");
        loggingConfig.setValue(health.loggingTcpEnabled()
                ? getTranslation("admin.health.logging.on", health.loggingTcpHost(), health.loggingTcpPort())
                : getTranslation("admin.health.logging.off"));
    }

    private HorizontalLayout createHeader() {
        Span title = new Span(getTranslation("admin.health.title"));
        title.addClassName("admin-tab-title");

        Button renewTokenButton = new Button(
                getTranslation("admin.health.button.renew-token"),
                VaadinIcon.REFRESH.create());
        renewTokenButton.setTooltipText(getTranslation("admin.health.button.renew-token"));
        renewTokenButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        renewTokenButton.addClickListener(event -> renewSetupToken());

        HorizontalLayout header = new HorizontalLayout(title, renewTokenButton);
        header.setWidthFull();
        header.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        header.expand(title);
        return header;
    }

    private FormLayout createForm() {
        version.setLabel(getTranslation("admin.health.field.version"));
        activeProfile.setLabel(getTranslation("admin.health.field.profile"));
        javaRuntime.setLabel(getTranslation("admin.health.field.java-runtime"));
        databaseStatus.setLabel(getTranslation("admin.health.field.database"));
        lastFlywayMigration.setLabel(getTranslation("admin.health.field.flyway"));
        loggingConfig.setLabel(getTranslation("admin.health.field.logging"));

        version.setReadOnly(true);
        activeProfile.setReadOnly(true);
        javaRuntime.setReadOnly(true);
        databaseStatus.setReadOnly(true);
        lastFlywayMigration.setReadOnly(true);
        loggingConfig.setReadOnly(true);

        FormLayout form = new FormLayout(version, activeProfile, javaRuntime, databaseStatus, lastFlywayMigration,
                loggingConfig);
        form.setResponsiveSteps(new FormLayout.ResponsiveStep("0", 1));
        return form;
    }

    private void renewSetupToken() {
        systemHealthService.renewSetupToken();
        Notification.show(getTranslation("admin.health.notification.token-renewed"), 4000,
                Notification.Position.MIDDLE);
    }
}
