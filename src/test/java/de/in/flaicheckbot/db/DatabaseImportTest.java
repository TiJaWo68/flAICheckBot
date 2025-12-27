package de.in.flaicheckbot.db;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.sql.SQLException;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit and integration tests for database operations, including schema
 * self-healing.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class DatabaseImportTest {

    @TempDir
    Path tempDir;

    private String dbPath;
    private DatabaseManager dbManager;

    @BeforeEach
    void setUp() {
        dbPath = tempDir.resolve("test-flaicheckbot.db").toString();
        dbManager = new DatabaseManager(dbPath);
    }

    @Test
    void testCreateTrainingSetAndSample() throws SQLException {
        int setId = dbManager.createTrainingSet("Test Title", "Original Text Content");
        assertTrue(setId > 0, "Set ID should be positive");

        byte[] dummyImage = new byte[] { 0x01, 0x02, 0x03 };
        assertDoesNotThrow(() -> {
            dbManager.addTrainingSample(setId, dummyImage, "image/png", "Test Sample Text");
        });
    }

    @Test
    void testAssignmentAndStudentWorkFlow() throws SQLException {
        // 1. Setup metadata
        int classId = dbManager.getOrCreateClass("10A");
        int testId = dbManager.createTestDefinition("Math Test", "10", "Algebra");
        int assignmentId = dbManager.getOrCreateAssignment(classId, testId, "2025-12-27 - Final Exam");

        // 2. Add student and work
        int studentId = dbManager.getOrCreateStudent("STUDENT_001", classId);
        byte[] dummyImage = new byte[] { (byte) 0xCA, (byte) 0xFE, (byte) 0xBA, (byte) 0xBE };
        dbManager.addStudentWork(assignmentId, studentId, dummyImage);

        // 3. Verify retrieval
        java.util.List<DatabaseManager.StudentWorkInfo> works = dbManager.getStudentWorkForAssignment(assignmentId);
        assertEquals(1, works.size(), "Should have exactly one student work");
        DatabaseManager.StudentWorkInfo work = works.get(0);
        assertEquals("STUDENT_001", work.studentExternalId);
        assertArrayEquals(dummyImage, work.imageData);
        assertFalse(work.isEvaluated, "Default status should be not evaluated");

        // 4. Update status and verify
        dbManager.updateStudentWorkStatus(work.id, true);
        works = dbManager.getStudentWorkForAssignment(assignmentId);
        assertTrue(works.get(0).isEvaluated, "Status should be updated to evaluated");
    }

    @Test
    void testSelfHealingWithLegacyConstraint() throws SQLException {
        // 1. Manually create a "broken" database with the legacy column and NOT NULL
        // constraint
        String legacyDbPath = tempDir.resolve("legacy-flaicheckbot.db").toString();
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + legacyDbPath);
                Statement stmt = conn.createStatement()) {

            stmt.execute("CREATE TABLE training_samples (" +
                    "id INTEGER PRIMARY KEY AUTOINCREMENT," +
                    "training_set_id INTEGER," +
                    "file_path TEXT NOT NULL" + // This is the problematic legacy column
                    ")");
        }

        // 2. Initialize DatabaseManager on this legacy DB
        DatabaseManager legacyDbManager = new DatabaseManager(legacyDbPath);

        // 3. Try to add a sample. This should fail if not healed.
        int setId = legacyDbManager.createTrainingSet("Legacy Test", "Some text");

        // This is expected to fail currently, which reproduces the user's issue
        byte[] dummyImage = new byte[] { 0x01 };

        // We want this to NOT throw an exception after our fix
        assertDoesNotThrow(() -> {
            legacyDbManager.addTrainingSample(setId, dummyImage, "image/png", null);
        });
    }
}
