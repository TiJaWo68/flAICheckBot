package de.in.fcb.ui.dashboard;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;

import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.html.Paragraph;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.router.HasDynamicTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.admin.AppSettingService;
import de.in.fcb.ai.AiConfig;
import de.in.fcb.ui.ResponsiveStyles;
import de.in.fcb.ui.admin.AdminView;
import de.in.fcb.ui.layout.MainLayout;
import jakarta.annotation.security.PermitAll;

/**
 * Initial landing dashboard shown after authentication.
 */
@Route(value = "", layout = MainLayout.class)
@PermitAll
public class DashboardView extends VerticalLayout implements HasDynamicTitle {

    private static final String ROLE_ADMIN = "ROLE_ADMIN";

    public DashboardView() {
        this(null);
    }

    @Autowired
    public DashboardView(@Autowired(required = false) AppSettingService appSettingService) {
        addClassName(ResponsiveStyles.DASHBOARD_VIEW);
        setSizeFull();
        setPadding(true);
        setSpacing(true);

        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        String username = auth != null ? auth.getName() : "";

        VerticalLayout card = new VerticalLayout();
        card.setMaxWidth("720px");
        card.setPadding(true);
        card.setSpacing(true);
        card.addClassNames(
                LumoUtility.Background.CONTRAST_5,
                LumoUtility.BorderRadius.LARGE,
                LumoUtility.Padding.LARGE);

        H2 heading = new H2(getTranslation("ui.dashboard.title"));
        heading.addClassNames(LumoUtility.Margin.NONE);

        Span subtitle = new Span(getTranslation("ui.dashboard.subtitle"));
        subtitle.addClassNames(LumoUtility.TextColor.SECONDARY, LumoUtility.FontSize.SMALL);

        Paragraph welcome = new Paragraph(getTranslation("ui.dashboard.welcome", username));
        welcome.addClassNames(LumoUtility.Margin.Bottom.NONE);

        card.add(heading, subtitle, welcome);

        if (appSettingService != null) {
            AiConfig aiConfig = appSettingService.getAiConfig();
            Span aiBadge = new Span(aiConfig.isConfigured()
                    ? getTranslation("ui.dashboard.ai-status.configured",
                            getTranslation("admin.settings.ai.preset." + aiConfig.preset().name()),
                            aiConfig.normalizedModel())
                    : getTranslation("ui.dashboard.ai-status.unconfigured"));
            aiBadge.getElement().getThemeList().add("badge");
            aiBadge.getElement().getThemeList().add(aiConfig.isConfigured() ? "success" : "contrast");
            card.add(aiBadge);
        }

        if (isAdmin(auth)) {
            Button adminButton = new Button(
                    getTranslation("ui.dashboard.admin-link"),
                    VaadinIcon.COG.create(),
                    event -> getUI().ifPresent(ui -> ui.navigate(AdminView.class)));
            adminButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
            card.add(adminButton);
        }

        add(card);
    }

    @Override
    public String getPageTitle() {
        return getTranslation("ui.dashboard.page-title");
    }

    private static boolean isAdmin(Authentication auth) {
        if (auth == null) {
            return false;
        }
        return auth.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .anyMatch(ROLE_ADMIN::equals);
    }
}
