package de.in.fcb.exams;

/**
 * Lifecycle state of an individual student exam submission.
 */
public enum StudentWorkStatus {
    NEW,
    UPLOADED,
    AI_GRADED,
    REVIEWED,
    ERROR
}
