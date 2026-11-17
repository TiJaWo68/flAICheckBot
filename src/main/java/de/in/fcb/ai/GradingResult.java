package de.in.fcb.ai;

import java.util.List;

import de.in.fcb.exams.TaskEvaluation;

/**
 * Structured multimodal AI grading output including recognized transcription,
 * per-task evaluations, overall feedback, and token consumption metadata.
 */
public record GradingResult(
        String recognizedText,
        String overallFeedback,
        List<TaskEvaluation> tasks,
        int promptTokens,
        int completionTokens,
        String modelUsed) {
}
