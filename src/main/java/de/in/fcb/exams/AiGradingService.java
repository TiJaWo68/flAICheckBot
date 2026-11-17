package de.in.fcb.exams;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import de.in.fcb.admin.AppSettingService;
import de.in.fcb.ai.AiClient;
import de.in.fcb.ai.AiConfig;
import de.in.fcb.ai.GradingResult;

/**
 * Executes synchronous multimodal AI grading for a single {@link StudentWork},
 * strictly transmitting only pseudonymized student IDs and scan images.
 */
@Service
public class AiGradingService {

    private static final Logger LOGGER = LoggerFactory.getLogger(AiGradingService.class);

    private final AssignmentService assignmentService;
    private final AssignmentRepository assignmentRepository;
    private final TestDefinitionRepository testDefinitionRepository;
    private final TestTaskRepository testTaskRepository;
    private final StudentRepository studentRepository;
    private final ScanStorageService scanStorageService;
    private final AppSettingService appSettingService;
    private final AiClient aiClient;

    public AiGradingService(
            AssignmentService assignmentService,
            AssignmentRepository assignmentRepository,
            TestDefinitionRepository testDefinitionRepository,
            TestTaskRepository testTaskRepository,
            StudentRepository studentRepository,
            ScanStorageService scanStorageService,
            AppSettingService appSettingService,
            AiClient aiClient) {
        this.assignmentService = assignmentService;
        this.assignmentRepository = assignmentRepository;
        this.testDefinitionRepository = testDefinitionRepository;
        this.testTaskRepository = testTaskRepository;
        this.studentRepository = studentRepository;
        this.scanStorageService = scanStorageService;
        this.appSettingService = appSettingService;
        this.aiClient = aiClient;
    }

    public StudentWork gradeStudentWork(UUID studentWorkId) {
        StudentWork work = assignmentService.findWork(studentWorkId);
        try {
            AiConfig config = appSettingService.getAiConfig();
            if (!config.isConfigured()) {
                throw new IllegalStateException("AI integration is not configured in Admin > Settings");
            }

            Assignment assignment = assignmentRepository.findById(work.getAssignmentId())
                    .orElseThrow(() -> new IllegalStateException("Assignment not found"));
            TestDefinition test = testDefinitionRepository.findById(assignment.getTestDefinitionId())
                    .orElseThrow(() -> new IllegalStateException("Test definition not found"));
            List<TestTask> tasks = testTaskRepository.findByTestDefinitionIdOrderByPositionAsc(test.getId());
            if (tasks.isEmpty()) {
                throw new IllegalStateException("Test definition has no tasks defined");
            }

            Student student = studentRepository.findById(work.getStudentId())
                    .orElseThrow(() -> new IllegalStateException("Student not found"));

            List<StudentWorkPage> pages = assignmentService.findPages(studentWorkId);
            if (pages.isEmpty()) {
                throw new IllegalStateException("No scan pages uploaded for student " + student.getExternalId());
            }

            List<byte[]> pageBytes = new ArrayList<>(pages.size());
            for (StudentWorkPage page : pages) {
                pageBytes.add(scanStorageService.readImageBytes(page.getStoragePath()));
            }

            // Strictly transmit externalId, never displayAlias
            GradingResult result = aiClient.gradeStudentWork(
                    config, test, tasks, student.getExternalId(), pageBytes);
            return assignmentService.saveAiGradingResult(studentWorkId, result);
        } catch (Exception exception) {
            LOGGER.warn("AI grading failed for studentWork {}: {}", studentWorkId, exception.getMessage());
            return assignmentService.markGradingError(studentWorkId, exception.getMessage());
        }
    }
}
