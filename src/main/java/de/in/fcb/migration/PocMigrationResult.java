package de.in.fcb.migration;

/**
 * Counters returned after executing a PoC SQLite migration run.
 */
public record PocMigrationResult(
        int classesCreated,
        int classesSkipped,
        int studentsCreated,
        int studentsSkipped,
        int testsCreated,
        int testsSkipped,
        int tasksCreated,
        int assignmentsCreated,
        int assignmentsSkipped,
        int scansImported,
        int scansSkipped) {
}
