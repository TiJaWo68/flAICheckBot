package de.in.fcb;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import com.vaadin.flow.server.LoadDependenciesOnStartup;

class FlAiCheckBotApplicationTest {

    @Test
    void loadsAllFrontendDependenciesOnStartup() {
        assertThat(FlAiCheckBotApplication.class.isAnnotationPresent(LoadDependenciesOnStartup.class)).isTrue();
    }
}
