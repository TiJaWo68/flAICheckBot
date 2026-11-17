package de.in.fcb.ui.admin;

import java.util.LinkedHashMap;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;

import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.tabs.Tab;
import com.vaadin.flow.component.tabs.Tabs;
import com.vaadin.flow.router.HasDynamicTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.spring.security.AuthenticationContext;

import de.in.fcb.admin.AppSettingService;
import de.in.fcb.admin.SystemHealthService;
import de.in.fcb.admin.UserService;
import de.in.fcb.ai.AiClient;
import de.in.fcb.ui.layout.MainLayout;
import jakarta.annotation.security.RolesAllowed;

/**
 * Administration area: user management, AI settings and system diagnostics,
 * restricted to {@code ROLE_ADMIN}.
 */
@Route(value = "admin", layout = MainLayout.class)
@RolesAllowed("ROLE_ADMIN")
public class AdminView extends VerticalLayout implements HasDynamicTitle {

    private final Map<Tab, AdminTab> tabs = new LinkedHashMap<>();
    private final Tabs tabBar = new Tabs();
    private final VerticalLayout content = new VerticalLayout();

    public AdminView(UserService userService,
            SystemHealthService systemHealthService,
            AuthenticationContext authenticationContext,
            AppSettingService appSettingService) {
        this(userService, systemHealthService, authenticationContext, appSettingService, null);
    }

    @Autowired
    public AdminView(UserService userService,
            SystemHealthService systemHealthService,
            AuthenticationContext authenticationContext,
            AppSettingService appSettingService,
            AiClient aiClient) {
        setSizeFull();
        setPadding(true);
        setSpacing(false);

        tabBar.setWidthFull();
        content.setSizeFull();
        content.setPadding(false);

        registerTab(new UsersTab(userService, authenticationContext));
        registerTab(new SettingsTab(appSettingService, aiClient));
        registerTab(new SystemHealthTab(systemHealthService));

        tabBar.addSelectedChangeListener(event -> showSelectedTab());

        add(tabBar, content);
        expand(content);

        showSelectedTab();
    }

    @Override
    public String getPageTitle() {
        return getTranslation("admin.view.page-title");
    }

    private void registerTab(AdminTab adminTab) {
        Tab tab = new Tab(getTranslation(adminTab.getLabelKey()));
        tabs.put(tab, adminTab);
        tabBar.add(tab);
    }

    private void showSelectedTab() {
        AdminTab selected = tabs.get(tabBar.getSelectedTab());
        if (selected == null) {
            return;
        }
        content.removeAll();
        content.add(selected.asComponent());
        selected.refresh();
    }
}
