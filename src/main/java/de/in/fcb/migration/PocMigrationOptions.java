package de.in.fcb.migration;

import de.in.fcb.exams.ForeignLanguage;

/**
 * Options controlling which entities and metadata to import from a PoC SQLite database.
 */
public record PocMigrationOptions(
        boolean importClassesAndStudents,
        boolean importTestsAndTasks,
        boolean importAssignmentsAndScans,
        boolean preserveRecognizedText,
        ForeignLanguage defaultLanguage) {

    public static PocMigrationOptions defaults() {
        return new PocMigrationOptions(true, true, true, false, ForeignLanguage.EN);
    }
}
