package de.in.fcb.exams;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import de.in.fcb.admin.AppSettingService;
import de.in.fcb.ai.AiClient;
import de.in.fcb.ai.AiConfig;
import de.in.fcb.ai.AiPreset;
import de.in.fcb.ai.ExtractedTask;

class TestDefinitionServiceTest {

    private TestDefinitionRepository testRepository;
    private TestTaskRepository taskRepository;
    private AssignmentRepository assignmentRepository;
    private AppSettingService appSettingService;
    private AiClient aiClient;
    private TestDefinitionService service;

    @BeforeEach
    void setUp() {
        testRepository = mock(TestDefinitionRepository.class);
        taskRepository = mock(TestTaskRepository.class);
        assignmentRepository = mock(AssignmentRepository.class);
        appSettingService = mock(AppSettingService.class);
        aiClient = mock(AiClient.class);
        service = new TestDefinitionService(
                testRepository, taskRepository, assignmentRepository, appSettingService, aiClient);
    }

    @Test
    void extractsTasksViaAiAndReplacesExistingTasks() {
        UUID testId = UUID.randomUUID();
        TestDefinition test = new TestDefinition();
        test.setId(testId);
        test.setTitle("English Test 1");
        test.setForeignLanguage(ForeignLanguage.EN);
        test.setSourceText("1. Describe the picture (10 pts)");

        AiConfig config = new AiConfig(AiPreset.GEMINI, "https://example/v1", "gemini-2.5-flash", "key");
        when(testRepository.findById(testId)).thenReturn(Optional.of(test));
        when(appSettingService.getAiConfig()).thenReturn(config);
        when(aiClient.extractTasksFromText(config, test.getSourceText(), "EN")).thenReturn(List.of(
                new ExtractedTask(1, "Picture Description", "Describe the picture", "Sample text", "",
                        new BigDecimal("10.00"))));
        when(taskRepository.save(any(TestTask.class))).thenAnswer(inv -> inv.getArgument(0));

        List<TestTask> result = service.extractTasksWithAi(testId);

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getTitle()).isEqualTo("Picture Description");
        assertThat(result.get(0).getMaxPoints()).isEqualByComparingTo("10.00");
        verify(taskRepository).deleteByTestDefinitionId(testId);
    }

    @Test
    void preventsDeletingTestInUseByAssignment() {
        UUID testId = UUID.randomUUID();
        when(assignmentRepository.countByTestDefinitionId(testId)).thenReturn(1L);

        assertThatThrownBy(() -> service.deleteTest(testId))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("used by an assignment");
    }
}
