package de.in.fcb.exams;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import de.in.fcb.admin.AppSettingService;
import de.in.fcb.ai.AiClient;
import de.in.fcb.ai.AiConfig;
import de.in.fcb.ai.ExtractedTask;

/**
 * Service for managing reusable {@link TestDefinition}s, ordered
 * {@link TestTask}s, and AI-assisted task structuring from uploaded documents.
 */
@Service
public class TestDefinitionService {

    private final TestDefinitionRepository testRepository;
    private final TestTaskRepository taskRepository;
    private final AssignmentRepository assignmentRepository;
    private final AppSettingService appSettingService;
    private final AiClient aiClient;

    public TestDefinitionService(
            TestDefinitionRepository testRepository,
            TestTaskRepository taskRepository,
            AssignmentRepository assignmentRepository,
            AppSettingService appSettingService,
            AiClient aiClient) {
        this.testRepository = testRepository;
        this.taskRepository = taskRepository;
        this.assignmentRepository = assignmentRepository;
        this.appSettingService = appSettingService;
        this.aiClient = aiClient;
    }

    @Transactional(readOnly = true)
    public List<TestDefinition> findAllTests() {
        return testRepository.findAllByOrderByCreatedAtDesc();
    }

    @Transactional(readOnly = true)
    public TestDefinition findTest(UUID id) {
        return testRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Test definition not found: " + id));
    }

    @Transactional
    public TestDefinition saveTest(TestDefinition test) {
        if (test == null) {
            throw new IllegalArgumentException("Test definition must not be null");
        }
        String title = test.getTitle() == null ? "" : test.getTitle().trim();
        if (title.isBlank()) {
            throw new IllegalArgumentException("Test title is required");
        }
        test.setTitle(title);
        if (test.getForeignLanguage() == null) {
            test.setForeignLanguage(ForeignLanguage.EN);
        }
        return testRepository.save(test);
    }

    @Transactional
    public void deleteTest(UUID id) {
        if (assignmentRepository.countByTestDefinitionId(id) > 0) {
            throw new IllegalArgumentException("Cannot delete test definition that is used by an assignment");
        }
        testRepository.deleteById(id);
    }

    @Transactional(readOnly = true)
    public List<TestTask> findTasks(UUID testId) {
        return taskRepository.findByTestDefinitionIdOrderByPositionAsc(testId);
    }

    @Transactional(readOnly = true)
    public long countTasks(UUID testId) {
        return taskRepository.countByTestDefinitionId(testId);
    }

    @Transactional(readOnly = true)
    public BigDecimal sumMaxPoints(UUID testId) {
        return findTasks(testId).stream()
                .map(TestTask::getMaxPoints)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    @Transactional
    public TestTask saveTask(TestTask task) {
        if (task == null || task.getTestDefinitionId() == null) {
            throw new IllegalArgumentException("Task and test definition ID are required");
        }
        String title = task.getTitle() == null ? "" : task.getTitle().trim();
        if (title.isBlank()) {
            throw new IllegalArgumentException("Task title is required");
        }
        String taskText = task.getTaskText() == null ? "" : task.getTaskText().trim();
        if (taskText.isBlank()) {
            throw new IllegalArgumentException("Task prompt text is required");
        }
        if (task.getPosition() <= 0) {
            throw new IllegalArgumentException("Task position must be positive");
        }
        if (task.getMaxPoints() == null || task.getMaxPoints().signum() <= 0) {
            throw new IllegalArgumentException("Max points must be positive");
        }
        task.setTitle(title);
        task.setTaskText(taskText);
        return taskRepository.save(task);
    }

    @Transactional
    public void deleteTask(UUID taskId) {
        taskRepository.deleteById(taskId);
    }

    @Transactional
    public List<TestTask> extractTasksWithAi(UUID testId) {
        TestDefinition test = findTest(testId);
        String sourceText = test.getSourceText();
        if (sourceText == null || sourceText.isBlank()) {
            throw new IllegalArgumentException("Import or enter exam document text before running AI extraction");
        }
        AiConfig config = appSettingService.getAiConfig();
        List<ExtractedTask> extracted = aiClient.extractTasksFromText(
                config, sourceText, test.getForeignLanguage().name());
        return replaceTasksFromExtracted(testId, extracted);
    }

    @Transactional
    public List<TestTask> replaceTasksFromExtracted(UUID testId, List<ExtractedTask> extracted) {
        if (extracted == null || extracted.isEmpty()) {
            throw new IllegalArgumentException("AI returned no tasks from document text");
        }
        taskRepository.deleteByTestDefinitionId(testId);
        taskRepository.flush();

        List<TestTask> saved = new ArrayList<>(extracted.size());
        int pos = 1;
        for (ExtractedTask item : extracted) {
            TestTask task = new TestTask();
            task.setTestDefinitionId(testId);
            task.setPosition(pos++);
            task.setTitle(item.title());
            task.setTaskText(item.taskText());
            task.setReferenceText(item.referenceText());
            task.setGradingHints(item.gradingHints());
            task.setMaxPoints(item.maxPoints() != null && item.maxPoints().signum() > 0
                    ? item.maxPoints()
                    : new BigDecimal("10.00"));
            saved.add(taskRepository.save(task));
        }
        return saved;
    }
}
