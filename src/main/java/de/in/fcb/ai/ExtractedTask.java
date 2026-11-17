package de.in.fcb.ai;

import java.math.BigDecimal;

/**
 * Structured task candidate extracted by the AI from an uploaded exam document.
 */
public record ExtractedTask(
        int position,
        String title,
        String taskText,
        String referenceText,
        String gradingHints,
        BigDecimal maxPoints) {
}
