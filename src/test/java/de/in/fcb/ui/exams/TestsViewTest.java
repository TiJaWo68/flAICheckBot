package de.in.fcb.ui.exams;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;

import org.junit.jupiter.api.Test;

import com.vaadin.flow.component.splitlayout.SplitLayout;

import de.in.fcb.exams.DocumentTextExtractor;
import de.in.fcb.exams.TestDefinitionService;

class TestsViewTest {

    @Test
    void rendersSplitLayoutAndLoadsTestsOnInit() {
        TestDefinitionService testService = mock(TestDefinitionService.class);
        DocumentTextExtractor extractor = new DocumentTextExtractor();
        when(testService.findAllTests()).thenReturn(List.of());

        TestsView view = new TestsView(testService, extractor);

        verify(testService).findAllTests();
        assertThat(view.getChildren().anyMatch(SplitLayout.class::isInstance)).isTrue();
    }
}
