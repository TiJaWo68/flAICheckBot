package de.in.fcb.ui.exams;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;

import org.junit.jupiter.api.Test;

import com.vaadin.flow.component.splitlayout.SplitLayout;

import de.in.fcb.exams.SchoolClass;
import de.in.fcb.exams.SchoolClassService;

class ClassesViewTest {

    @Test
    void rendersSplitLayoutAndLoadsClassesOnInit() {
        SchoolClassService classService = mock(SchoolClassService.class);
        when(classService.findAllClasses()).thenReturn(List.of(new SchoolClass()));

        ClassesView view = new ClassesView(classService);

        verify(classService).findAllClasses();
        assertThat(view.getChildren().anyMatch(SplitLayout.class::isInstance)).isTrue();
    }
}
