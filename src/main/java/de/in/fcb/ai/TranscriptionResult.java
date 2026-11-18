package de.in.fcb.ai;

import java.util.List;

/**
 * Result of a standalone verbatim handwriting transcription run on one or more
 * scanned pages.
 */
public record TranscriptionResult(
        String recognizedText,
        List<String> illegibleSegments,
        int promptTokens,
        int completionTokens,
        String modelUsed) {
}
