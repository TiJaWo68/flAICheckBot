package de.in.flaicheckbot.util;

import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import de.in.flaicheckbot.db.DatabaseManager;

/**
 * Utility class for database maintenance and cleanup operations.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class DatabaseCleanupUtil {
	private static final Logger LOGGER = LogManager.getLogger(DatabaseCleanupUtil.class);

	/**
	 * Deletes all assignments and associated student work from the database. This operation is irreversible.
	 * 
	 * @param dbManager The database manager to use.
	 * @throws SQLException If a database error occurs.
	 */
	public static void deleteAllAssignments(DatabaseManager dbManager) throws SQLException {
		try (Connection conn = dbManager.getConnection()) {
			conn.setAutoCommit(false);
			try (Statement stmt = conn.createStatement()) {
				LOGGER.info("Cleaning up all assignments and student work...");

				// student_work has a foreign key to assignments
				int deletedWork = stmt.executeUpdate("DELETE FROM student_work");
				int deletedAssignments = stmt.executeUpdate("DELETE FROM assignments");

				conn.commit();
				LOGGER.info("Cleanup successful: deleted {} student work entries and {} assignments", deletedWork, deletedAssignments);
			} catch (SQLException e) {
				conn.rollback();
				LOGGER.error("Cleanup failed, changes rolled back.", e);
				throw e;
			} finally {
				conn.setAutoCommit(true);
			}
		}
	}

	public static void main(String[] args) throws SQLException {
		LOGGER.info("Initializing Database...");
		DatabaseManager dm = new DatabaseManager("flaicheckbot.db");
		deleteAllAssignments(dm);
	}
}
