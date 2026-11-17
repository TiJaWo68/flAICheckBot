package de.in.fcb.exams;

import java.math.BigDecimal;

/**
 * Per-task evaluation result stored in {@link StudentWork#getGradingJson()}.
 */
public record TaskEvaluation(
        int position,
        String taskTitle,
        String recognizedAnswer,
        BigDecimal awardedPoints,
        BigDecimal maxPoints,
        String feedback,
        String errorAnnotations) {
}
