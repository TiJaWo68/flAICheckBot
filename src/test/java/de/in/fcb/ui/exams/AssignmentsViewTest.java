package de.in.fcb.ui.exams;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;

import org.junit.jupiter.api.Test;

import com.vaadin.flow.component.grid.Grid;

import de.in.fcb.exams.AssignmentService;
import de.in.fcb.exams.SchoolClassService;
import de.in.fcb.exams.TestDefinitionService;

class AssignmentsViewTest {

    @Test
    void rendersAssignmentsGridAndLoadsAssignmentsOnInit() {
        AssignmentService assignmentService = mock(AssignmentService.class);
        SchoolClassService classService = mock(SchoolClassService.class);
        TestDefinitionService testService = mock(TestDefinitionService.class);

        when(classService.findAllClasses()).thenReturn(List.of());
        when(testService.findAllTests()).thenReturn(List.of());
        when(assignmentService.findAllAssignments()).thenReturn(List.of());

        AssignmentsView view = new AssignmentsView(assignmentService, classService, testService);

        verify(assignmentService).findAllAssignments();
        assertThat(view.getChildren().anyMatch(Grid.class::isInstance)).isTrue();
    }
}
