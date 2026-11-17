package de.in.fcb.ui.dashboard;

import java.util.List;

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
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.router.HasDynamicTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.admin.AppSettingService;
import de.in.fcb.ai.AiConfig;
import de.in.fcb.exams.AssignmentRepository;
import de.in.fcb.exams.SchoolClassRepository;
import de.in.fcb.exams.StudentWorkRepository;
import de.in.fcb.exams.StudentWorkStatus;
import de.in.fcb.exams.TestDefinitionRepository;
import de.in.fcb.ui.ResponsiveStyles;
import de.in.fcb.ui.admin.AdminView;
import de.in.fcb.ui.exams.AssignmentsView;
import de.in.fcb.ui.exams.ClassesView;
import de.in.fcb.ui.exams.TestsView;
import de.in.fcb.ui.layout.MainLayout;
import jakarta.annotation.security.PermitAll;

/**
 * Initial landing dashboard shown after authentication with exam KPIs and
 * quick navigation links.
 */
@Route(value = "", layout = MainLayout.class)
@PermitAll
public class DashboardView extends VerticalLayout implements HasDynamicTitle {

    private static final String ROLE_ADMIN = "ROLE_ADMIN";

    public DashboardView() {
        this(null, null, null, null, null);
    }

    @Autowired
    public DashboardView(
            @Autowired(required = false) AppSettingService appSettingService,
            @Autowired(required = false) SchoolClassRepository classRepository,
            @Autowired(required = false) TestDefinitionRepository testRepository,
            @Autowired(required = false) AssignmentRepository assignmentRepository,
            @Autowired(required = false) StudentWorkRepository studentWorkRepository) {
        addClassName(ResponsiveStyles.DASHBOARD_VIEW);
        setSizeFull();
        setPadding(true);
        setSpacing(true);

        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        String username = auth != null ? auth.getName() : "";

        VerticalLayout card = new VerticalLayout();
        card.setMaxWidth("920px");
        card.setWidthFull();
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

        if (classRepository != null && testRepository != null
                && assignmentRepository != null && studentWorkRepository != null) {
            HorizontalLayout kpiRow = new HorizontalLayout(
                    createKpiCard(getTranslation("ui.dashboard.kpi.classes"), classRepository.count()),
                    createKpiCard(getTranslation("ui.dashboard.kpi.tests"), testRepository.count()),
                    createKpiCard(getTranslation("ui.dashboard.kpi.assignments"), assignmentRepository.count()),
                    createKpiCard(getTranslation("ui.dashboard.kpi.graded"),
                            studentWorkRepository.countByStatusIn(
                                    List.of(StudentWorkStatus.AI_GRADED, StudentWorkStatus.REVIEWED))));
            kpiRow.setWidthFull();
            kpiRow.getStyle().set("flex-wrap", "wrap");
            card.add(kpiRow);
        }

        HorizontalLayout quickActions = new HorizontalLayout();
        quickActions.getStyle().set("flex-wrap", "wrap");
        Button classesBtn = new Button(getTranslation("ui.nav.classes"), VaadinIcon.GROUP.create(),
                e -> getUI().ifPresent(ui -> ui.navigate(ClassesView.class)));
        Button testsBtn = new Button(getTranslation("ui.nav.tests"), VaadinIcon.CLIPBOARD_TEXT.create(),
                e -> getUI().ifPresent(ui -> ui.navigate(TestsView.class)));
        Button assignmentsBtn = new Button(getTranslation("ui.nav.assignments"), VaadinIcon.ACADEMY_CAP.create(),
                e -> getUI().ifPresent(ui -> ui.navigate(AssignmentsView.class)));
        assignmentsBtn.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        quickActions.add(classesBtn, testsBtn, assignmentsBtn);

        if (isAdmin(auth)) {
            Button adminButton = new Button(
                    getTranslation("ui.dashboard.admin-link"),
                    VaadinIcon.COG.create(),
                    event -> getUI().ifPresent(ui -> ui.navigate(AdminView.class)));
            quickActions.add(adminButton);
        }

        card.add(quickActions);
        add(card);
    }

    private VerticalLayout createKpiCard(String label, long value) {
        VerticalLayout kpi = new VerticalLayout();
        kpi.setPadding(true);
        kpi.setSpacing(false);
        kpi.getStyle().set("min-width", "150px");
        kpi.getStyle().set("flex", "1 1 150px");
        kpi.addClassNames(LumoUtility.Background.CONTRAST_10, LumoUtility.BorderRadius.MEDIUM);

        Span lbl = new Span(label);
        lbl.addClassNames(LumoUtility.FontSize.SMALL, LumoUtility.TextColor.SECONDARY);

        Span val = new Span(String.valueOf(value));
        val.addClassNames(LumoUtility.FontSize.XXLARGE, LumoUtility.FontWeight.BOLD);

        kpi.add(lbl, val);
        return kpi;
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
