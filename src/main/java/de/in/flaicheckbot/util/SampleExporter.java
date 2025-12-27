package de.in.flaicheckbot.util;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;

/**
 * Maintenance utility for exporting training data from the database
 * and clearing training tables.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class SampleExporter {

	private static final String DB_PATH = "flaicheckbot.db"; // Assumes run from project root

	public static void main(String[] args) {
		System.out.println("Starting Sample Export...");
		File exportDir = new File("exported_samples");
		if (!exportDir.exists()) {
			exportDir.mkdirs();
		}

		String url = "jdbc:sqlite:" + DB_PATH;

		try (Connection conn = DriverManager.getConnection(url)) {
			String sql = """
					SELECT s.id, s.image_data, s.mime_type, COALESCE(s.sample_text, t.original_text) as text \
					FROM training_samples s \
					JOIN training_sets t ON s.training_set_id = t.id""";

			try (PreparedStatement start = conn.prepareStatement(sql); ResultSet rs = start.executeQuery()) {

				int count = 0;
				while (rs.next()) {
					int id = rs.getInt("id");
					byte[] imgData = rs.getBytes("image_data");
					String text = rs.getString("text");
					String mime = rs.getString("mime_type");

					String ext = ".png"; // Default
					if (mime != null && mime.contains("jpeg"))
						ext = ".jpg";

					// Save Image
					File imgFile = new File(exportDir, "sample_" + id + ext);
					try (FileOutputStream fos = new FileOutputStream(imgFile)) {
						fos.write(imgData);
					}

					// Save Text
					File txtFile = new File(exportDir, "sample_" + id + ".txt");
					try (FileOutputStream fos = new FileOutputStream(txtFile)) {
						fos.write(text.getBytes(StandardCharsets.UTF_8));
					}

					System.out.println("Exported ID " + id + ": " + imgFile.getName());
					count++;
				}
				System.out.println("Export complete! " + count + " samples exported to " + exportDir.getAbsolutePath());

				// Clear Database
				System.out.println("Clearing database...");
				try (java.sql.Statement stmt = conn.createStatement()) {
					// Delete samples first due to FK constraint
					int deletedSamples = stmt.executeUpdate("DELETE FROM training_samples");
					int deletedSets = stmt.executeUpdate("DELETE FROM training_sets");
					System.out.println(
							"Database cleared: " + deletedSamples + " samples and " + deletedSets + " sets deleted.");
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
			System.err.println("Export failed: " + e.getMessage());
		}
	}
}
