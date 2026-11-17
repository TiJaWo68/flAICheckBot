package de.in.fcb.ui.admin;

import com.vaadin.flow.component.Component;

/**
 * A single content area of the {@link AdminView} multi-tab layout.
 */
public interface AdminTab {

    String getLabelKey();

    Component asComponent();

    void refresh();
}
