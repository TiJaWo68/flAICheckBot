package de.in.fcb.ui.admin;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import org.junit.jupiter.api.Test;

import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.tabs.Tabs;
import com.vaadin.flow.spring.security.AuthenticationContext;

import de.in.fcb.admin.AppSettingService;
import de.in.fcb.admin.SystemHealthService;
import de.in.fcb.admin.UserService;
import de.in.fcb.ai.AiClient;
import de.in.fcb.migration.PocMigrationService;

class AdminViewTest {

    @Test
    void rendersSelectedTabContentImmediatelyOnInitialLoad() {
        UserService userService = mock(UserService.class);
        SystemHealthService systemHealthService = mock(SystemHealthService.class);
        AuthenticationContext authenticationContext = mock(AuthenticationContext.class);
        AppSettingService appSettingService = mock(AppSettingService.class);

        AdminView view = new AdminView(userService, systemHealthService, authenticationContext, appSettingService);

        VerticalLayout content = (VerticalLayout) view.getChildren()
                .filter(VerticalLayout.class::isInstance)
                .findFirst()
                .orElseThrow();

        assertThat(content.getChildren())
                .as("AdminView must immediately render the active tab content on first load without requiring a tab click")
                .isNotEmpty();
    }

    @Test
    void includesMigrationTabOnlyWhenEnabled() {
        UserService userService = mock(UserService.class);
        SystemHealthService systemHealthService = mock(SystemHealthService.class);
        AuthenticationContext authenticationContext = mock(AuthenticationContext.class);
        AppSettingService appSettingService = mock(AppSettingService.class);
        AiClient aiClient = mock(AiClient.class);
        PocMigrationService migrationService = mock(PocMigrationService.class);

        AdminView enabledView = new AdminView(
                userService, systemHealthService, authenticationContext,
                appSettingService, aiClient, migrationService, true);
        AdminView disabledView = new AdminView(
                userService, systemHealthService, authenticationContext,
                appSettingService, aiClient, migrationService, false);

        Tabs enabledTabs = enabledView.getChildren()
                .filter(Tabs.class::isInstance)
                .map(Tabs.class::cast)
                .findFirst()
                .orElseThrow();
        Tabs disabledTabs = disabledView.getChildren()
                .filter(Tabs.class::isInstance)
                .map(Tabs.class::cast)
                .findFirst()
                .orElseThrow();

        assertThat(enabledTabs.getComponentCount()).isEqualTo(4);
        assertThat(disabledTabs.getComponentCount()).isEqualTo(3);
    }
}
