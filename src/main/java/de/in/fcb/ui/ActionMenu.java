package de.in.fcb.ui;

import java.util.function.Consumer;

import com.vaadin.flow.component.contextmenu.MenuItem;
import com.vaadin.flow.component.contextmenu.SubMenu;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.icon.Icon;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.menubar.MenuBar;
import com.vaadin.flow.component.menubar.MenuBarVariant;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.theme.lumo.LumoUtility;

/**
 * Compact header action menu ({@code ⋮}) that consolidates view actions into
 * a single dropdown menu.
 */
public class ActionMenu {

    private static final String ICON_SIZE = "16px";
    private static final String ICON_MARGIN_RIGHT = "var(--lumo-space-s)";

    private final MenuBar menuBar = new MenuBar();
    private final SubMenu subMenu;

    public ActionMenu(String tooltip) {
        menuBar.addThemeVariants(
                MenuBarVariant.LUMO_TERTIARY,
                MenuBarVariant.LUMO_ICON);

        MenuItem trigger = menuBar.addItem(VaadinIcon.ELLIPSIS_DOTS_V.create());
        trigger.setAriaLabel(tooltip);
        trigger.getElement().setAttribute("title", tooltip);
        this.subMenu = trigger.getSubMenu();
    }

    public HorizontalLayout createTitleHeader(String titleText) {
        H2 title = new H2(titleText);
        title.addClassNames(LumoUtility.FontSize.XLARGE, LumoUtility.Margin.NONE);

        HorizontalLayout header = new HorizontalLayout(title, menuBar);
        header.setWidthFull();
        header.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        header.addClassName(LumoUtility.Margin.Bottom.XSMALL);
        return header;
    }

    public MenuBar getMenuBar() {
        return menuBar;
    }

    public SubMenu getSubMenu() {
        return subMenu;
    }

    public void clear() {
        subMenu.removeAll();
    }

    public MenuItem addAction(String label, VaadinIcon iconType, Runnable action) {
        MenuItem item = subMenu.addItem(label, event -> action.run());
        if (iconType != null) {
            Icon icon = iconType.create();
            icon.setSize(ICON_SIZE);
            icon.getStyle().set("margin-right", ICON_MARGIN_RIGHT);
            item.addComponentAsFirst(icon);
        }
        return item;
    }

    public MenuItem addToggle(String label, boolean checked, Consumer<Boolean> onToggle) {
        MenuItem item = subMenu.addItem(label, event -> onToggle.accept(event.getSource().isChecked()));
        item.setCheckable(true);
        item.setChecked(checked);
        return item;
    }

    public void addSeparator() {
        subMenu.addSeparator();
    }
}
