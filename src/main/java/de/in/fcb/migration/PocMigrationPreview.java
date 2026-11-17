package de.in.fcb.migration;

import java.util.List;

/**
 * Read-only inspection summary of a PoC SQLite database before running migration.
 */
public record PocMigrationPreview(
        List<ClassSummary> classes,
        List<TestSummary> tests,
        List<AssignmentSummary> assignments) {

    public record ClassSummary(
            int pocId,
            String name,
            String academicYear,
            int studentCount,
            boolean alreadyExists) {
    }

    public record TestSummary(
            int pocId,
            String title,
            String gradeLevel,
            String learningUnit,
            int taskCount,
            boolean alreadyExists) {
    }

    public record AssignmentSummary(
            int pocId,
            String title,
            String className,
            String testTitle,
            int workCount,
            int scanBlobCount,
            boolean alreadyExists) {
    }

    public int totalStudents() {
        return classes.stream().mapToInt(ClassSummary::studentCount).sum();
    }

    public int totalTasks() {
        return tests.stream().mapToInt(TestSummary::taskCount).sum();
    }

    public int totalWorks() {
        return assignments.stream().mapToInt(AssignmentSummary::workCount).sum();
    }

    public int totalScanBlobs() {
        return assignments.stream().mapToInt(AssignmentSummary::scanBlobCount).sum();
    }
}
