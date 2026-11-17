package de.in.fcb.exams;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

import org.springframework.stereotype.Service;

/**
 * Asynchronous background runner for single-student and full-assignment batch
 * AI grading, isolating failures per student so partial progress is preserved.
 */
@Service
public class BatchGradingRunner {

    private final AiGradingService aiGradingService;
    private final AssignmentService assignmentService;

    public BatchGradingRunner(AiGradingService aiGradingService, AssignmentService assignmentService) {
        this.aiGradingService = aiGradingService;
        this.assignmentService = assignmentService;
    }

    public CompletableFuture<StudentWork> gradeSingleAsync(UUID studentWorkId, Runnable onCompleted) {
        return CompletableFuture.supplyAsync(() -> aiGradingService.gradeStudentWork(studentWorkId))
                .whenComplete((res, ex) -> {
                    if (onCompleted != null) {
                        onCompleted.run();
                    }
                });
    }

    public CompletableFuture<Void> gradeAssignmentBatchAsync(
            UUID assignmentId,
            Consumer<ProgressUpdate> onProgress,
            Runnable onCompleted) {
        return CompletableFuture.runAsync(() -> {
            List<StudentWork> candidates = assignmentService.findWorks(assignmentId).stream()
                    .filter(work -> (work.getStatus() == StudentWorkStatus.UPLOADED
                            || work.getStatus() == StudentWorkStatus.ERROR)
                            && assignmentService.countPages(work.getId()) > 0)
                    .toList();

            int total = candidates.size();
            int current = 0;
            for (StudentWork work : candidates) {
                current++;
                if (onProgress != null) {
                    onProgress.accept(new ProgressUpdate(current, total));
                }
                aiGradingService.gradeStudentWork(work.getId());
            }
        }).whenComplete((res, ex) -> {
            if (onCompleted != null) {
                onCompleted.run();
            }
        });
    }

    public record ProgressUpdate(int current, int total) {
    }
}
