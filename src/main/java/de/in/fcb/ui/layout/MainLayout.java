package de.in.fcb.ui.layout;

import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.oauth2.core.user.OAuth2User;

import com.vaadin.flow.component.applayout.AppLayout;
import com.vaadin.flow.component.applayout.DrawerToggle;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.html.H1;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.Icon;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.sidenav.SideNav;
import com.vaadin.flow.component.sidenav.SideNavItem;
import com.vaadin.flow.spring.security.AuthenticationContext;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.ui.ResponsiveStyles;
import jakarta.annotation.security.PermitAll;

/**
 * Application shell for all protected views: responsive navigation drawer,
 * header with session information and a logout action.
 */
@PermitAll
public class MainLayout extends AppLayout {

    private static final String ROLE_ADMIN = "ROLE_ADMIN";

    private final AuthenticationContext authenticationContext;
    private SideNav sideNav;

    public MainLayout(AuthenticationContext authenticationContext) {
        this.authenticationContext = authenticationContext;
        addToNavbar(createHeader());
        addToDrawer(createDrawer());
    }

    private HorizontalLayout createHeader() {
        DrawerToggle drawerToggle = new DrawerToggle();
        drawerToggle.setTooltipText(getTranslation("ui.header.toggle"));

        H1 title = new H1(getTranslation("ui.app.title"));
        title.addClassNames(LumoUtility.FontSize.MEDIUM, LumoUtility.Margin.NONE);

        String displayName = resolveDisplayName(SecurityContextHolder.getContext().getAuthentication());
        Span userBadge = new Span(getTranslation("ui.header.logged-in-as", displayName));
        userBadge.addClassNames(LumoUtility.TextColor.SECONDARY, LumoUtility.FontSize.SMALL,
                ResponsiveStyles.HIDE_ON_SHORT_SCREEN);

        Button logoutButton = new Button(
                VaadinIcon.SIGN_OUT.create(),
                event -> authenticationContext.logout());
        logoutButton.setTooltipText(getTranslation("ui.header.logout"));
        logoutButton.setAriaLabel(getTranslation("ui.header.logout"));

        HorizontalLayout header = new HorizontalLayout(drawerToggle, title, userBadge, logoutButton);
        header.setWidthFull();
        header.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        header.expand(title);
        return header;
    }

    private SideNav createDrawer() {
        sideNav = new SideNav(getTranslation("ui.nav.title"));
        populateDrawer(sideNav);
        return sideNav;
    }

    private void populateDrawer(SideNav targetNav) {
        Set<String> grantedAuthorities = authenticationContext.getGrantedAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .collect(Collectors.toSet());

        navigationItems().stream()
                .filter(item -> isItemVisible(item, grantedAuthorities))
                .forEach(item -> targetNav.addItem(toSideNavItem(item)));
    }

    private SideNavItem toSideNavItem(NavigationItem item) {
        SideNavItem navItem = new SideNavItem(getTranslation(item.labelKey()), item.path());
        navItem.setPrefixComponent(new Icon(item.icon()));
        navItem.setTooltipText(getTranslation(item.labelKey()));
        return navItem;
    }

    static List<NavigationItem> navigationItems() {
        return List.of(
                new NavigationItem("ui.nav.dashboard", "", VaadinIcon.DASHBOARD, Set.of()),
                new NavigationItem("ui.nav.classes", "classes", VaadinIcon.GROUP, Set.of()),
                new NavigationItem("ui.nav.tests", "tests", VaadinIcon.CLIPBOARD_TEXT, Set.of()),
                new NavigationItem("ui.nav.assignments", "assignments", VaadinIcon.ACADEMY_CAP, Set.of()),
                new NavigationItem("ui.nav.admin", "admin", VaadinIcon.COG, Set.of(ROLE_ADMIN)));
    }

    static boolean isItemVisible(NavigationItem item, Collection<String> grantedAuthorities) {
        return item.requiredRoles().isEmpty()
                || grantedAuthorities.stream().anyMatch(item.requiredRoles()::contains);
    }

    static String resolveDisplayName(Authentication authentication) {
        if (authentication == null || !authentication.isAuthenticated() || authentication.getPrincipal() == null) {
            return "";
        }

        Object principal = authentication.getPrincipal();
        if (principal instanceof UserDetails userDetails) {
            return userDetails.getUsername();
        }
        if (principal instanceof OAuth2User oAuth2User) {
            Object email = oAuth2User.getAttribute("email");
            return email != null ? email.toString() : oAuth2User.getName();
        }
        return authentication.getName();
    }

    record NavigationItem(String labelKey, String path, VaadinIcon icon, Set<String> requiredRoles) {
    }
}
