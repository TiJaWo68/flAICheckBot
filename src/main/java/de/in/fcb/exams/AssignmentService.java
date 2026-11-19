package de.in.fcb.exams;

import java.io.IOException;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.in.fcb.ai.GradingResult;

/**
 * Service coordinating exam {@link Assignment}s, {@link StudentWork} records,
 * scan page uploads/redactions, AI grading persistence, and teacher reviews.
 */
@Service
public class AssignmentService {

    private final AssignmentRepository assignmentRepository;
    private final StudentRepository studentRepository;
    private final TestTaskRepository testTaskRepository;
    private final StudentWorkRepository studentWorkRepository;
    private final StudentWorkPageRepository pageRepository;
    private final ScanStorageService scanStorageService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public AssignmentService(
            AssignmentRepository assignmentRepository,
            StudentRepository studentRepository,
            TestTaskRepository testTaskRepository,
            StudentWorkRepository studentWorkRepository,
            StudentWorkPageRepository pageRepository,
            ScanStorageService scanStorageService) {
        this.assignmentRepository = assignmentRepository;
        this.studentRepository = studentRepository;
        this.testTaskRepository = testTaskRepository;
        this.studentWorkRepository = studentWorkRepository;
        this.pageRepository = pageRepository;
        this.scanStorageService = scanStorageService;
    }

    @Transactional(readOnly = true)
    public List<Assignment> findAllAssignments() {
        return assignmentRepository.findAllByOrderByExamDateDescCreatedAtDesc();
    }

    @Transactional(readOnly = true)
    public Assignment findAssignment(UUID id) {
        return assignmentRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Assignment not found: " + id));
    }

    @Transactional
    public Assignment createAssignment(String title, UUID schoolClassId, UUID testDefinitionId, LocalDate examDate) {
        String trimmedTitle = title == null ? "" : title.trim();
        if (trimmedTitle.isBlank()) {
            throw new IllegalArgumentException("Assignment title is required");
        }
        if (schoolClassId == null || testDefinitionId == null) {
            throw new IllegalArgumentException("Class and test definition are required");
        }

        Assignment assignment = new Assignment();
        assignment.setTitle(trimmedTitle);
        assignment.setSchoolClassId(schoolClassId);
        assignment.setTestDefinitionId(testDefinitionId);
        assignment.setExamDate(examDate != null ? examDate : LocalDate.now());
        Assignment saved = assignmentRepository.save(assignment);

        BigDecimal maxPoints = testTaskRepository.findByTestDefinitionIdOrderByPositionAsc(testDefinitionId).stream()
                .map(TestTask::getMaxPoints)
                .reduce(BigDecimal.ZERO, BigDecimal::add);

        List<Student> students = studentRepository.findBySchoolClassIdOrderByExternalIdAsc(schoolClassId);
        for (Student student : students) {
            StudentWork work = new StudentWork();
            work.setAssignmentId(saved.getId());
            work.setStudentId(student.getId());
            work.setStatus(StudentWorkStatus.NEW);
            work.setMaxPoints(maxPoints);
            studentWorkRepository.save(work);
        }
        return saved;
    }

    @Transactional
    public void deleteAssignment(UUID assignmentId) {
        List<StudentWork> works = studentWorkRepository.findByAssignmentId(assignmentId);
        for (StudentWork work : works) {
            List<StudentWorkPage> pages = pageRepository.findByStudentWorkIdOrderByPageNumberAsc(work.getId());
            for (StudentWorkPage page : pages) {
                scanStorageService.deleteFileIfExists(page.getStoragePath());
            }
        }
        assignmentRepository.deleteById(assignmentId);
    }

    @Transactional(readOnly = true)
    public List<StudentWork> findWorks(UUID assignmentId) {
        return studentWorkRepository.findByAssignmentId(assignmentId);
    }

    @Transactional(readOnly = true)
    public StudentWork findWork(UUID studentWorkId) {
        return studentWorkRepository.findById(studentWorkId)
                .orElseThrow(() -> new IllegalArgumentException("Student work not found: " + studentWorkId));
    }

    @Transactional(readOnly = true)
    public List<StudentWorkPage> findPages(UUID studentWorkId) {
        return pageRepository.findByStudentWorkIdOrderByPageNumberAsc(studentWorkId);
    }

    @Transactional(readOnly = true)
    public long countPages(UUID studentWorkId) {
        return pageRepository.countByStudentWorkId(studentWorkId);
    }

    @Transactional
    public List<StudentWorkPage> uploadPages(UUID studentWorkId, String filename, byte[] bytes) throws IOException {
        StudentWork work = findWork(studentWorkId);
        List<StudentWorkPage> existing = pageRepository.findByStudentWorkIdOrderByPageNumberAsc(studentWorkId);
        int nextPageNum = existing.isEmpty() ? 1 : existing.get(existing.size() - 1).getPageNumber() + 1;

        List<String> storedPaths = scanStorageService.importUpload(
                work.getAssignmentId(), studentWorkId, nextPageNum, filename, bytes);

        List<StudentWorkPage> created = new ArrayList<>(storedPaths.size());
        int pageNum = nextPageNum;
        for (String path : storedPaths) {
            StudentWorkPage page = new StudentWorkPage();
            page.setStudentWorkId(studentWorkId);
            page.setPageNumber(pageNum++);
            page.setStoragePath(path);
            page.setMimeType("image/png");
            created.add(pageRepository.save(page));
        }

        if (work.getStatus() == StudentWorkStatus.NEW || work.getStatus() == StudentWorkStatus.ERROR) {
            work.setStatus(StudentWorkStatus.UPLOADED);
            studentWorkRepository.save(work);
        }
        return created;
    }

    @Transactional
    public void rotatePage(UUID pageId, int degreesClockwise) throws IOException {
        StudentWorkPage page = pageRepository.findById(pageId)
                .orElseThrow(() -> new IllegalArgumentException("Page not found: " + pageId));
        scanStorageService.rotatePage(page.getStoragePath(), degreesClockwise);
        List<RedactionBox> boxes = parseRedactionBoxes(page.getRedactionBoxesJson());
        if (!boxes.isEmpty()) {
            List<RedactionBox> rotated = boxes.stream()
                    .map(box -> box.rotateClockwise(degreesClockwise))
                    .toList();
            page.setRedactionBoxesJson(objectMapper.writeValueAsString(rotated));
            pageRepository.save(page);
        }
    }

    @Transactional
    public void redactPage(UUID pageId, double normX, double normY, double normWidth, double normHeight)
            throws IOException {
        StudentWorkPage page = pageRepository.findById(pageId)
                .orElseThrow(() -> new IllegalArgumentException("Page not found: " + pageId));
        List<RedactionBox> boxes = parseRedactionBoxes(page.getRedactionBoxesJson());
        boxes.add(new RedactionBox(normX, normY, normWidth, normHeight));
        scanStorageService.renderRedactedImage(page.getStoragePath(), boxes);
        page.setRedactionBoxesJson(objectMapper.writeValueAsString(boxes));
        pageRepository.save(page);
    }

    @Transactional
    public boolean undoLastRedaction(UUID pageId) throws IOException {
        StudentWorkPage page = pageRepository.findById(pageId)
                .orElseThrow(() -> new IllegalArgumentException("Page not found: " + pageId));
        List<RedactionBox> boxes = parseRedactionBoxes(page.getRedactionBoxesJson());
        if (boxes.isEmpty()) {
            return false;
        }
        boxes.remove(boxes.size() - 1);
        scanStorageService.renderRedactedImage(page.getStoragePath(), boxes);
        page.setRedactionBoxesJson(boxes.isEmpty() ? null : objectMapper.writeValueAsString(boxes));
        pageRepository.save(page);
        return true;
    }

    @Transactional
    public boolean clearRedactions(UUID pageId) throws IOException {
        StudentWorkPage page = pageRepository.findById(pageId)
                .orElseThrow(() -> new IllegalArgumentException("Page not found: " + pageId));
        List<RedactionBox> boxes = parseRedactionBoxes(page.getRedactionBoxesJson());
        if (boxes.isEmpty()) {
            return false;
        }
        scanStorageService.renderRedactedImage(page.getStoragePath(), List.of());
        page.setRedactionBoxesJson(null);
        pageRepository.save(page);
        return true;
    }

    @Transactional
    public void deletePage(UUID pageId) {
        StudentWorkPage page = pageRepository.findById(pageId)
                .orElseThrow(() -> new IllegalArgumentException("Page not found: " + pageId));
        UUID workId = page.getStudentWorkId();
        scanStorageService.deleteFileIfExists(page.getStoragePath());
        pageRepository.delete(page);
        pageRepository.flush();

        List<StudentWorkPage> remaining = pageRepository.findByStudentWorkIdOrderByPageNumberAsc(workId);
        int num = 1;
        for (StudentWorkPage rem : remaining) {
            rem.setPageNumber(num++);
            pageRepository.save(rem);
        }
        if (remaining.isEmpty()) {
            StudentWork work = findWork(workId);
            work.setStatus(StudentWorkStatus.NEW);
            studentWorkRepository.save(work);
        }
    }

    @Transactional
    public StudentWork saveAiGradingResult(UUID studentWorkId, GradingResult result) {
        StudentWork work = findWork(studentWorkId);
        List<TaskEvaluation> tasks = result.tasks() == null ? List.of() : result.tasks();

        BigDecimal achieved = tasks.stream()
                .map(t -> t.awardedPoints() != null ? t.awardedPoints() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        BigDecimal max = tasks.stream()
                .map(t -> t.maxPoints() != null ? t.maxPoints() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);

        work.setRecognizedText(result.recognizedText());
        work.setOverallFeedback(result.overallFeedback());
        work.setGradingJson(serializeTaskEvaluations(tasks));
        work.setAchievedPoints(achieved);
        if (max.signum() > 0) {
            work.setMaxPoints(max);
        }
        work.setPromptTokens(work.getPromptTokens() + result.promptTokens());
        work.setCompletionTokens(work.getCompletionTokens() + result.completionTokens());
        work.setModelUsed(result.modelUsed());
        work.setStatus(StudentWorkStatus.AI_GRADED);
        return studentWorkRepository.save(work);
    }

    @Transactional
    public StudentWork markGradingError(UUID studentWorkId, String errorMessage) {
        StudentWork work = findWork(studentWorkId);
        work.setStatus(StudentWorkStatus.ERROR);
        work.setOverallFeedback(errorMessage);
        return studentWorkRepository.save(work);
    }

    @Transactional
    public StudentWork updateReview(
            UUID studentWorkId,
            String recognizedText,
            String overallFeedback,
            List<TaskEvaluation> evaluations,
            StudentWorkStatus newStatus) {
        StudentWork work = findWork(studentWorkId);
        List<TaskEvaluation> effectiveTasks = evaluations == null ? List.of() : evaluations;
        BigDecimal achieved = effectiveTasks.stream()
                .map(t -> t.awardedPoints() != null ? t.awardedPoints() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        BigDecimal max = effectiveTasks.stream()
                .map(t -> t.maxPoints() != null ? t.maxPoints() : BigDecimal.ZERO)
                .reduce(BigDecimal.ZERO, BigDecimal::add);

        work.setRecognizedText(recognizedText);
        work.setOverallFeedback(overallFeedback);
        work.setGradingJson(serializeTaskEvaluations(effectiveTasks));
        work.setAchievedPoints(achieved);
        if (max.signum() > 0) {
            work.setMaxPoints(max);
        }
        if (newStatus != null) {
            work.setStatus(newStatus);
        }
        return studentWorkRepository.save(work);
    }

    public List<TaskEvaluation> parseTaskEvaluations(String gradingJson) {
        if (gradingJson == null || gradingJson.isBlank()) {
            return List.of();
        }
        try {
            return objectMapper.readValue(gradingJson, new TypeReference<>() {
            });
        } catch (IOException exception) {
            return List.of();
        }
    }

    public String serializeTaskEvaluations(List<TaskEvaluation> evaluations) {
        try {
            return objectMapper.writeValueAsString(evaluations == null ? List.of() : evaluations);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to serialize task evaluations", exception);
        }
    }

    public List<RedactionBox> parseRedactionBoxes(String json) {
        if (json == null || json.isBlank()) {
            return new ArrayList<>();
        }
        try {
            return objectMapper.readValue(json, new TypeReference<>() {
            });
        } catch (IOException exception) {
            return new ArrayList<>();
        }
    }
}
