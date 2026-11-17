package de.in.fcb;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import com.vaadin.flow.component.dependency.StyleSheet;
import com.vaadin.flow.component.page.AppShellConfigurator;
import com.vaadin.flow.component.page.Inline;
import com.vaadin.flow.component.page.Push;
import com.vaadin.flow.server.AppShellSettings;
import com.vaadin.flow.server.LoadDependenciesOnStartup;
import com.vaadin.flow.theme.Theme;
import com.vaadin.flow.theme.lumo.Lumo;

import de.in.fcb.ui.ResponsiveStyles;

/**
 * Application entry point and Vaadin AppShell configuration.
 * <p>
 * All web components are loaded eagerly so that the root route renders all
 * components immediately without lazy chunk delays.
 */
@SpringBootApplication
@Push
@Theme(variant = Lumo.DARK)
@StyleSheet(Lumo.UTILITY_STYLESHEET)
@StyleSheet(ResponsiveStyles.STYLESHEET)
@LoadDependenciesOnStartup
public class FlAiCheckBotApplication implements AppShellConfigurator {

    public static void main(String[] args) {
        SpringApplication.run(FlAiCheckBotApplication.class, args);
    }

    @Override
    public void configurePage(AppShellSettings settings) {
        settings.addFavIcon("icon", "/icons/flaicheckbot-logo.svg", "any");
        settings.addInlineWithContents(ResponsiveStyles.CSS_RULES, Inline.Wrapping.STYLESHEET);
    }
}
