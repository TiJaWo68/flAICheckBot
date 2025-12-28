package de.in.flaicheckbot.db;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.sqlite.SQLiteDataSource;

/**
 * Core persistence layer managing the SQLite database,
 * schema migrations, and all CRUD operations.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class DatabaseManager {
    private static final Logger logger = LogManager.getLogger(DatabaseManager.class);
    private final SQLiteDataSource dataSource;

    public DatabaseManager(String dbPath) {
        dataSource = new SQLiteDataSource();
        dataSource.setUrl("jdbc:sqlite:" + dbPath);
        initializeSchema();
    }

    private void initializeSchema() {
        try (Connection conn = dataSource.getConnection();
                InputStream is = getClass().getResourceAsStream("/schema.sql")) {
            if (is == null) {
                logger.error("Schema file not found!");
                return;
            }
            String schema = new BufferedReader(new InputStreamReader(is))
                    .lines().collect(Collectors.joining("\n"));
            try (Statement stmt = conn.createStatement()) {
                stmt.executeUpdate(schema);
            }
            ensureColumnsExist(conn);
        } catch (Exception e) {
            logger.error("Database initialization failed", e);
        }
    }

    private void ensureColumnsExist(Connection conn) {
        // Self-healing: Check for missing columns and add them
        try {
            // Add new required columns
            addColumnIfMissing(conn, "training_samples", "image_data", "BLOB NOT NULL DEFAULT ''");
            addColumnIfMissing(conn, "training_samples", "mime_type", "TEXT");
            addColumnIfMissing(conn, "training_samples", "sample_text", "TEXT");
            addColumnIfMissing(conn, "student_work", "image_data", "BLOB");
            addColumnIfMissing(conn, "student_work", "recognized_text", "TEXT");
            addColumnIfMissing(conn, "student_work", "grading_json", "TEXT");
            addColumnIfMissing(conn, "student_work", "score", "INTEGER");
            addColumnIfMissing(conn, "student_work", "feedback", "TEXT");
            addColumnIfMissing(conn, "student_work", "status", "TEXT DEFAULT 'NEW'");
            addColumnIfMissing(conn, "student_work", "is_evaluated", "BOOLEAN DEFAULT FALSE");

            // Schema migration for assignments
            addColumnIfMissing(conn, "assignments", "test_id", "INTEGER");
            addColumnIfMissing(conn, "assignments", "last_import_path", "TEXT");

            // Advanced Test Tasks migration
            addColumnIfMissing(conn, "test_tasks", "task_text", "TEXT");
            addColumnIfMissing(conn, "test_tasks", "position", "INTEGER");

            // Create new tables if initialization failed or for migration
            try (Statement stmt = conn.createStatement()) {
                stmt.executeUpdate(
                        "CREATE TABLE IF NOT EXISTS test_definitions (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, grade_level TEXT, learning_unit TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");
                stmt.executeUpdate(
                        "CREATE TABLE IF NOT EXISTS test_tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, position INTEGER, task_text TEXT, reference_text TEXT, max_points INTEGER, resource_info TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(test_id) REFERENCES test_definitions(id))");
            }

            // Legacy cleanup: Drop columns that cause conflicts (like NOT NULL without
            // default)
            dropColumnIfExists(conn, "training_samples", "file_path");
        } catch (SQLException e) {
            logger.error("Error during self-healing", e);
        }
    }

    private void dropColumnIfExists(Connection conn, String table, String column) throws SQLException {
        String checkSql = "PRAGMA table_info(" + table + ")";
        boolean exists = false;
        try (Statement stmt = conn.createStatement();
                ResultSet rs = stmt.executeQuery(checkSql)) {
            while (rs.next()) {
                if (column.equalsIgnoreCase(rs.getString("name"))) {
                    exists = true;
                    break;
                }
            }
        }
        if (exists) {
            String dropSql = "ALTER TABLE " + table + " DROP COLUMN " + column;
            try (Statement stmt = conn.createStatement()) {
                stmt.executeUpdate(dropSql);
                logger.info("Self-healing: Dropped legacy column [{}] from table [{}]", column, table);
            } catch (SQLException e) {
                logger.warn(
                        "Could not drop legacy column [{}]. This is expected on very old SQLite versions. Error: {}",
                        column, e.getMessage());
            }
        }
    }

    private void addColumnIfMissing(Connection conn, String table, String column, String type) throws SQLException {
        String checkSql = "PRAGMA table_info(" + table + ")";
        boolean exists = false;
        try (Statement stmt = conn.createStatement();
                ResultSet rs = stmt.executeQuery(checkSql)) {
            while (rs.next()) {
                if (column.equalsIgnoreCase(rs.getString("name"))) {
                    exists = true;
                    break;
                }
            }
        }
        if (!exists) {
            String alterSql = "ALTER TABLE " + table + " ADD COLUMN " + column + " " + type;
            try (Statement stmt = conn.createStatement()) {
                stmt.executeUpdate(alterSql);
                logger.info("Self-healing: Added column [{}] to table [{}]", column, table);
            }
        }
    }

    public Connection getConnection() throws SQLException {
        return dataSource.getConnection();
    }

    // --- Helper Methods using simple JDBC for now ---

    public int createTrainingSet(String title, String originalText) throws SQLException {
        String sql = "INSERT INTO training_sets (title, original_text) VALUES (?, ?)";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql, Statement.RETURN_GENERATED_KEYS)) {
            pstmt.setString(1, title);
            pstmt.setString(2, originalText);
            pstmt.executeUpdate();
            try (ResultSet generatedKeys = pstmt.getGeneratedKeys()) {
                if (generatedKeys.next()) {
                    return generatedKeys.getInt(1);
                }
            }
        }
        return -1;
    }

    public void addTrainingSample(int trainingSetId, byte[] imageData, String mimeType, String sampleText)
            throws SQLException {
        String sql = "INSERT INTO training_samples (training_set_id, image_data, mime_type, sample_text) VALUES (?, ?, ?, ?)";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, trainingSetId);
            pstmt.setBytes(2, imageData);
            pstmt.setString(3, mimeType);
            pstmt.setString(4, sampleText);
            pstmt.executeUpdate();
        }
    }

    public int createAssignment(int classId, int testId, String title) throws SQLException {
        String sql = "INSERT INTO assignments (class_id, test_id, title) VALUES (?, ?, ?)";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql, Statement.RETURN_GENERATED_KEYS)) {
            pstmt.setInt(1, classId);
            pstmt.setInt(2, testId);
            pstmt.setString(3, title);
            pstmt.executeUpdate();
            try (ResultSet generatedKeys = pstmt.getGeneratedKeys()) {
                if (generatedKeys.next()) {
                    return generatedKeys.getInt(1);
                }
            }
        }
        return -1;
    }

    public int getOrCreateAssignment(int classId, int testId, String title) throws SQLException {
        String query = "SELECT id FROM assignments WHERE class_id = ? AND test_id = ? AND title = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(query)) {
            pstmt.setInt(1, classId);
            pstmt.setInt(2, testId);
            pstmt.setString(3, title);
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next())
                    return rs.getInt("id");
            }
        }
        return createAssignment(classId, testId, title);
    }

    public int createTestDefinition(String title, String gradeLevel, String learningUnit) throws SQLException {
        String sql = "INSERT INTO test_definitions (title, grade_level, learning_unit) VALUES (?, ?, ?)";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql, Statement.RETURN_GENERATED_KEYS)) {
            pstmt.setString(1, title);
            pstmt.setString(2, gradeLevel);
            pstmt.setString(3, learningUnit);
            pstmt.executeUpdate();
            try (ResultSet generatedKeys = pstmt.getGeneratedKeys()) {
                if (generatedKeys.next()) {
                    return generatedKeys.getInt(1);
                }
            }
        }
        return -1;
    }

    public void addTestTask(int testId, int position, String taskText, String referenceText, int maxPoints,
            String resourceInfo)
            throws SQLException {
        String sql = "INSERT INTO test_tasks (test_id, position, task_text, reference_text, max_points, resource_info) VALUES (?, ?, ?, ?, ?, ?)";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, testId);
            pstmt.setInt(2, position);
            pstmt.setString(3, taskText);
            pstmt.setString(4, referenceText);
            pstmt.setInt(5, maxPoints);
            pstmt.setString(6, resourceInfo);
            pstmt.executeUpdate();
        }
    }

    public void updateTestDefinitionWithTasks(int testId, String title, String gradeLevel, String learningUnit,
            List<TaskInfo> tasks) throws SQLException {
        try (Connection conn = getConnection()) {
            conn.setAutoCommit(false);
            try {
                // Update metadata
                String sqlMeta = "UPDATE test_definitions SET title = ?, grade_level = ?, learning_unit = ? WHERE id = ?";
                try (PreparedStatement pstmt = conn.prepareStatement(sqlMeta)) {
                    pstmt.setString(1, title);
                    pstmt.setString(2, gradeLevel);
                    pstmt.setString(3, learningUnit);
                    pstmt.setInt(4, testId);
                    int affected = pstmt.executeUpdate();
                    if (affected == 0) {
                        throw new SQLException("Test definition ID " + testId + " not found.");
                    }
                }

                // Delete old tasks
                String sqlDel = "DELETE FROM test_tasks WHERE test_id = ?";
                try (PreparedStatement pstmt = conn.prepareStatement(sqlDel)) {
                    pstmt.setInt(1, testId);
                    pstmt.executeUpdate();
                }

                // Insert new tasks
                String sqlTask = "INSERT INTO test_tasks (test_id, position, task_text, reference_text, max_points, resource_info) VALUES (?, ?, ?, ?, ?, ?)";
                try (PreparedStatement pstmt = conn.prepareStatement(sqlTask)) {
                    for (int i = 0; i < tasks.size(); i++) {
                        TaskInfo task = tasks.get(i);
                        pstmt.setInt(1, testId);
                        pstmt.setInt(2, i + 1);
                        pstmt.setString(3, task.taskText);
                        pstmt.setString(4, task.referenceText);
                        pstmt.setInt(5, task.maxPoints);
                        pstmt.setString(6, task.resourceInfo);
                        pstmt.executeUpdate();
                    }
                }

                conn.commit();
                logger.info("Updated test definition ID {} with {} tasks", testId, tasks.size());
            } catch (SQLException e) {
                conn.rollback();
                throw e;
            } finally {
                conn.setAutoCommit(true);
            }
        }
    }

    public void deleteStudentWork(int workId) throws SQLException {
        String sql = "DELETE FROM student_work WHERE id = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, workId);
            pstmt.executeUpdate();
        }
    }

    public void deleteTestDefinition(int testId) throws SQLException {
        try (Connection conn = getConnection()) {
            conn.setAutoCommit(false);
            try {
                // Delete tasks first
                String sqlTasks = "DELETE FROM test_tasks WHERE test_id = ?";
                try (PreparedStatement pstmt = conn.prepareStatement(sqlTasks)) {
                    pstmt.setInt(1, testId);
                    pstmt.executeUpdate();
                }
                // Delete definition
                String sqlDef = "DELETE FROM test_definitions WHERE id = ?";
                try (PreparedStatement pstmt = conn.prepareStatement(sqlDef)) {
                    pstmt.setInt(1, testId);
                    pstmt.executeUpdate();
                }
                conn.commit();
                logger.info("Deleted test definition ID {}", testId);
            } catch (SQLException e) {
                conn.rollback();
                throw e;
            } finally {
                conn.setAutoCommit(true);
            }
        }
    }

    public boolean isTestInUse(int testId) throws SQLException {
        String sql = "SELECT COUNT(*) FROM assignments WHERE test_id = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, testId);
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    return rs.getInt(1) > 0;
                }
            }
        }
        return false;
    }

    public java.util.List<TaskInfo> getTasksForTest(int testId) throws SQLException {
        java.util.List<TaskInfo> tasks = new java.util.ArrayList<>();
        String sql = "SELECT position, task_text, reference_text, max_points, resource_info FROM test_tasks WHERE test_id = ? ORDER BY position ASC";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, testId);
            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    tasks.add(new TaskInfo(
                            rs.getInt("position"),
                            rs.getString("task_text"),
                            rs.getString("reference_text"),
                            rs.getInt("max_points"),
                            rs.getString("resource_info")));
                }
            }
        }
        return tasks;
    }

    public static class TaskInfo {
        public final int position;
        public final String taskText;
        public final String referenceText;
        public final int maxPoints;
        public final String resourceInfo;
        public String context; // Optional: e.g. Test title

        public TaskInfo(int position, String taskText, String referenceText, int maxPoints, String resourceInfo) {
            this(position, taskText, referenceText, maxPoints, resourceInfo, null);
        }

        public TaskInfo(int position, String taskText, String referenceText, int maxPoints, String resourceInfo,
                String context) {
            this.position = position;
            this.taskText = taskText;
            this.referenceText = referenceText;
            this.maxPoints = maxPoints;
            this.resourceInfo = resourceInfo;
            this.context = context;
        }

        @Override
        public String toString() {
            String shortText = taskText.length() > 50 ? taskText.substring(0, 47) + "..." : taskText;
            return (context != null ? "[" + context + "] " : "") + shortText;
        }
    }

    public java.util.List<TaskInfo> getAllTasks() throws SQLException {
        java.util.List<TaskInfo> tasks = new java.util.ArrayList<>();
        String sql = "SELECT t.position, t.task_text, t.reference_text, t.max_points, t.resource_info, d.title as test_title "
                +
                "FROM test_tasks t JOIN test_definitions d ON t.test_id = d.id " +
                "ORDER BY t.created_at DESC";
        try (Connection conn = getConnection();
                Statement stmt = conn.createStatement();
                ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                tasks.add(new TaskInfo(
                        rs.getInt("position"),
                        rs.getString("task_text"),
                        rs.getString("reference_text"),
                        rs.getInt("max_points"),
                        rs.getString("resource_info"),
                        rs.getString("test_title")));
            }
        }
        return tasks;
    }

    public java.util.List<TestInfo> getAllTests() throws SQLException {
        java.util.List<TestInfo> tests = new java.util.ArrayList<>();
        String sql = "SELECT id, title, grade_level, learning_unit FROM test_definitions ORDER BY title";
        try (Connection conn = getConnection();
                Statement stmt = conn.createStatement();
                ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                tests.add(new TestInfo(rs.getInt("id"), rs.getString("title"), rs.getString("grade_level"),
                        rs.getString("learning_unit")));
            }
        }
        return tests;
    }

    public static class TestInfo {
        public final int id;
        public final String title;
        public final String gradeLevel;
        public final String learningUnit;

        public TestInfo(int id, String title, String gradeLevel, String learningUnit) {
            this.id = id;
            this.title = title;
            this.gradeLevel = gradeLevel;
            this.learningUnit = learningUnit;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(title);
            boolean hasGrade = gradeLevel != null && !gradeLevel.isEmpty();
            boolean hasUnit = learningUnit != null && !learningUnit.isEmpty();

            if (hasGrade || hasUnit) {
                sb.append(" (");
                if (hasGrade) {
                    sb.append("Kl. ").append(gradeLevel);
                }
                if (hasGrade && hasUnit) {
                    sb.append(", ");
                }
                if (hasUnit) {
                    sb.append(learningUnit);
                }
                sb.append(")");
            }
            return sb.toString();
        }
    }

    public List<AssignmentInfo> getAssignmentsForTest(int testId) throws SQLException {
        List<AssignmentInfo> assignments = new java.util.ArrayList<>();
        String sql = "SELECT a.id, a.title, c.name as class_name, t.title as test_title, a.last_import_path " +
                "FROM assignments a " +
                "JOIN classes c ON a.class_id = c.id " +
                "JOIN test_definitions t ON a.test_id = t.id " +
                "WHERE a.test_id = ? ORDER BY a.created_at DESC";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, testId);
            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    assignments.add(new AssignmentInfo(
                            rs.getInt("id"),
                            rs.getString("title"),
                            rs.getString("class_name"),
                            rs.getString("test_title"),
                            rs.getString("last_import_path")));
                }
            }
        }
        return assignments;
    }

    public void updateAssignmentImportPath(int assignmentId, String path) throws SQLException {
        String sql = "UPDATE assignments SET last_import_path = ? WHERE id = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, path);
            pstmt.setInt(2, assignmentId);
            pstmt.executeUpdate();
        }
    }

    public static class AssignmentInfo {
        public final int id;
        public final String title;
        public final String className;
        public final String testTitle;
        public final String lastImportPath;

        public AssignmentInfo(int id, String title, String className, String testTitle, String lastImportPath) {
            this.id = id;
            this.title = title;
            this.className = className;
            this.testTitle = testTitle;
            this.lastImportPath = lastImportPath;
        }

        @Override
        public String toString() {
            return title + " (" + className + " - " + testTitle + ")";
        }
    }

    public java.util.List<StudentWorkInfo> getStudentWorkForAssignment(int assignmentId) throws SQLException {
        java.util.List<StudentWorkInfo> workList = new java.util.ArrayList<>();
        String sql = "SELECT sw.id, sw.student_id, sw.image_data, sw.recognized_text, sw.grading_json, sw.score, sw.feedback, sw.status, sw.is_evaluated, s.external_id "
                +
                "FROM student_work sw JOIN students s ON sw.student_id = s.id " +
                "WHERE sw.assignment_id = ? ORDER BY s.external_id";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, assignmentId);
            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    workList.add(new StudentWorkInfo(
                            rs.getInt("id"),
                            rs.getInt("student_id"),
                            rs.getString("external_id"),
                            rs.getBytes("image_data"),
                            rs.getString("recognized_text"),
                            rs.getString("grading_json"),
                            rs.getInt("score"),
                            rs.getString("feedback"),
                            rs.getString("status"),
                            rs.getBoolean("is_evaluated")));
                }
            }
        }
        return workList;
    }

    public static class StudentWorkInfo {
        public final int id;
        public final int studentId;
        public final String studentExternalId;
        public final byte[] imageData;
        public final String recognizedText;
        public final String gradingJson;
        public final int score;
        public final String feedback;
        public final String status;
        public final boolean isEvaluated;

        public StudentWorkInfo(int id, int studentId, String studentExternalId, byte[] imageData, String recognizedText,
                String gradingJson, int score, String feedback, String status, boolean isEvaluated) {
            this.id = id;
            this.studentId = studentId;
            this.studentExternalId = studentExternalId;
            this.imageData = imageData;
            this.recognizedText = recognizedText;
            this.gradingJson = gradingJson;
            this.score = score;
            this.feedback = feedback;
            this.status = status;
            this.isEvaluated = isEvaluated;
        }
    }

    public void updateGrading(int studentWorkId, String gradingJson, int score, String feedback) throws SQLException {
        String sql = "UPDATE student_work SET grading_json = ?, score = ?, feedback = ? WHERE id = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, gradingJson);
            pstmt.setInt(2, score);
            pstmt.setString(3, feedback);
            pstmt.setInt(4, studentWorkId);
            pstmt.executeUpdate();
        }
    }

    public void updateGradingManual(int studentWorkId, int score, String feedback) throws SQLException {
        String sql = "UPDATE student_work SET score = ?, feedback = ? WHERE id = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, score);
            pstmt.setString(2, feedback);
            pstmt.setInt(3, studentWorkId);
            pstmt.executeUpdate();
        }
    }

    public void updateRecognizedText(int studentWorkId, String text) throws SQLException {
        String sql = "UPDATE student_work SET recognized_text = ? WHERE id = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, text);
            pstmt.setInt(2, studentWorkId);
            pstmt.executeUpdate();
        }
    }

    public void updateStudentWorkStatus(int workId, boolean evaluated) throws SQLException {
        String sql = "UPDATE student_work SET is_evaluated = ? WHERE id = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setBoolean(1, evaluated);
            pstmt.setInt(2, workId);
            pstmt.executeUpdate();
        }
    }

    public void addStudentWork(int assignmentId, int studentId, byte[] imageData) throws SQLException {
        String sql = "INSERT INTO student_work (assignment_id, student_id, image_data) VALUES (?, ?, ?)";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, assignmentId);
            pstmt.setInt(2, studentId);
            pstmt.setBytes(3, imageData);
            pstmt.executeUpdate();
        }
    }

    // Quick method to ensure a class exists
    public int getOrCreateClass(String name) throws SQLException {
        String query = "SELECT id FROM classes WHERE name = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(query)) {
            pstmt.setString(1, name);
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    return rs.getInt("id");
                }
            }
        }
        String insert = "INSERT INTO classes (name) VALUES (?)";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(insert, Statement.RETURN_GENERATED_KEYS)) {
            pstmt.setString(1, name);
            pstmt.executeUpdate();
            try (ResultSet rs = pstmt.getGeneratedKeys()) {
                if (rs.next()) {
                    return rs.getInt(1);
                }
            }
        }
        return -1;
    }

    public int getOrCreateStudent(String externalId, int classId) throws SQLException {
        String query = "SELECT id FROM students WHERE external_id = ? AND class_id = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(query)) {
            pstmt.setString(1, externalId);
            pstmt.setInt(2, classId);
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    return rs.getInt("id");
                }
            }
        }
        String insert = "INSERT INTO students (external_id, class_id) VALUES (?, ?)";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(insert, Statement.RETURN_GENERATED_KEYS)) {
            pstmt.setString(1, externalId);
            pstmt.setInt(2, classId);
            pstmt.executeUpdate();
            try (ResultSet rs = pstmt.getGeneratedKeys()) {
                if (rs.next()) {
                    return rs.getInt(1);
                }
            }
        }
        return -1;
    }

    public void updateStudentExternalId(int studentId, String newExternalId) throws SQLException {
        String sql = "UPDATE students SET external_id = ? WHERE id = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, newExternalId);
            pstmt.setInt(2, studentId);
            pstmt.executeUpdate();
        }
    }

    public String getSetting(String key) throws SQLException {
        String sql = "SELECT value FROM settings WHERE key = ?";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, key);
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    return rs.getString("value");
                }
            }
        }
        return null;
    }

    public void setSetting(String key, String value) throws SQLException {
        String sql = "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)";
        try (Connection conn = getConnection();
                PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, key);
            pstmt.setString(2, value);
            pstmt.executeUpdate();
        }
    }
}
