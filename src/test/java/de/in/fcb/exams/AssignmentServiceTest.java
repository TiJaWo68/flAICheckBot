package de.in.fcb.exams;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import de.in.fcb.ai.GradingResult;

class AssignmentServiceTest {

    private AssignmentRepository assignmentRepository;
    private StudentRepository studentRepository;
    private TestTaskRepository testTaskRepository;
    private StudentWorkRepository studentWorkRepository;
    private StudentWorkPageRepository pageRepository;
    private ScanStorageService scanStorageService;
    private AssignmentService service;

    @BeforeEach
    void setUp() {
        assignmentRepository = mock(AssignmentRepository.class);
        studentRepository = mock(StudentRepository.class);
        testTaskRepository = mock(TestTaskRepository.class);
        studentWorkRepository = mock(StudentWorkRepository.class);
        pageRepository = mock(StudentWorkPageRepository.class);
        scanStorageService = mock(ScanStorageService.class);
        service = new AssignmentService(
                assignmentRepository,
                studentRepository,
                testTaskRepository,
                studentWorkRepository,
                pageRepository,
                scanStorageService);
    }

    @Test
    void createsAssignmentAndInitializesStudentWorksWithMaxPoints() {
        UUID classId = UUID.randomUUID();
        UUID testId = UUID.randomUUID();

        TestTask t1 = new TestTask();
        t1.setMaxPoints(new BigDecimal("10.00"));
        TestTask t2 = new TestTask();
        t2.setMaxPoints(new BigDecimal("15.00"));
        when(testTaskRepository.findByTestDefinitionIdOrderByPositionAsc(testId)).thenReturn(List.of(t1, t2));

        Student s1 = new Student();
        s1.setId(UUID.randomUUID());
        Student s2 = new Student();
        s2.setId(UUID.randomUUID());
        when(studentRepository.findBySchoolClassIdOrderByExternalIdAsc(classId)).thenReturn(List.of(s1, s2));

        when(assignmentRepository.save(any(Assignment.class))).thenAnswer(inv -> {
            Assignment a = inv.getArgument(0);
            a.setId(UUID.randomUUID());
            return a;
        });

        Assignment created = service.createAssignment("Klassenarbeit 1", classId, testId, LocalDate.of(2026, 11, 17));

        assertThat(created.getId()).isNotNull();
        verify(studentWorkRepository, times(2)).save(any(StudentWork.class));
    }

    @Test
    void savesAiGradingResultAndUpdatesScoreAndTokens() {
        UUID workId = UUID.randomUUID();
        StudentWork work = new StudentWork();
        work.setId(workId);
        work.setStatus(StudentWorkStatus.UPLOADED);
        when(studentWorkRepository.findById(workId)).thenReturn(Optional.of(work));
        when(studentWorkRepository.save(any(StudentWork.class))).thenAnswer(inv -> inv.getArgument(0));

        GradingResult result = new GradingResult(
                "My text",
                "Gut gemacht",
                List.of(
                        new TaskEvaluation(1, "Task 1", "Ans 1", new BigDecimal("8.5"), new BigDecimal("10"), "Ok", ""),
                        new TaskEvaluation(2, "Task 2", "Ans 2", new BigDecimal("9.0"), new BigDecimal("10"), "Super",
                                "")),
                420,
                180,
                "gemini-2.5-flash");

        StudentWork saved = service.saveAiGradingResult(workId, result);

        assertThat(saved.getStatus()).isEqualTo(StudentWorkStatus.AI_GRADED);
        assertThat(saved.getAchievedPoints()).isEqualByComparingTo("17.5");
        assertThat(saved.getMaxPoints()).isEqualByComparingTo("20");
        assertThat(saved.getPromptTokens()).isEqualTo(420);
        assertThat(saved.getCompletionTokens()).isEqualTo(180);
        assertThat(saved.getModelUsed()).isEqualTo("gemini-2.5-flash");
        assertThat(service.parseTaskEvaluations(saved.getGradingJson())).hasSize(2);
    }

    @Test
    void updatesTeacherReviewAndMarksReviewed() {
        UUID workId = UUID.randomUUID();
        StudentWork work = new StudentWork();
        work.setId(workId);
        work.setStatus(StudentWorkStatus.AI_GRADED);
        when(studentWorkRepository.findById(workId)).thenReturn(Optional.of(work));
        when(studentWorkRepository.save(any(StudentWork.class))).thenAnswer(inv -> inv.getArgument(0));

        StudentWork updated = service.updateReview(
                workId,
                "Corrected text",
                "Sehr gut überarbeitet",
                List.of(new TaskEvaluation(1, "Task 1", "Ans", new BigDecimal("9.5"), new BigDecimal("10"), "Stark",
                        "")),
                StudentWorkStatus.REVIEWED);

        assertThat(updated.getStatus()).isEqualTo(StudentWorkStatus.REVIEWED);
        assertThat(updated.getAchievedPoints()).isEqualByComparingTo("9.5");
        assertThat(updated.getRecognizedText()).isEqualTo("Corrected text");
    }
}
