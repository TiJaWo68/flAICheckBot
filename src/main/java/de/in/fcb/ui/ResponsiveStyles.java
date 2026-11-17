package de.in.fcb.ui;

/**
 * CSS hooks for responsive behavior that cannot be expressed with Lumo utility
 * classes alone.
 */
public final class ResponsiveStyles {

    /** Application stylesheet, resolved relative to {@code META-INF/resources}. */
    public static final String STYLESHEET = "styles/responsive.css";

    /**
     * Hides a component on short viewports (e.g. smartphones in landscape
     * orientation).
     */
    public static final String HIDE_ON_SHORT_SCREEN = "fcb-hide-on-short-screen";

    /** Root container class for compact mobile padding on the dashboard. */
    public static final String DASHBOARD_VIEW = "fcb-dashboard-view";

    public static final String CSS_RULES = """
            @media (orientation: landscape) and (max-height: 500px) {
                .""" + HIDE_ON_SHORT_SCREEN + """
                 {
                    display: none !important;
                }
            }
            @media (max-width: 699px) {
                .""" + DASHBOARD_VIEW + """
                 {
                    padding: var(--lumo-space-xs) var(--lumo-space-s) !important;
                }
            }
            """;

    private ResponsiveStyles() {
    }
}
