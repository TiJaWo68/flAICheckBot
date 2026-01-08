package de.in.flaicheckbot;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import java.io.File;
import java.nio.file.Files;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.util.concurrent.ExecutionException;

/**
 * JUnit test for AIEngineClient.
 * This test verifies that the client correctly handles both empty and valid
 * image files.
 */
public class AIEngineClientTest {

    @Test
    public void testRecognizeEmptyFile() throws IOException, ExecutionException, InterruptedException {
        File emptyFile = File.createTempFile("junit_test_empty_", ".png");
        try {
            AIEngineClient client = new AIEngineClient();
            String response = client.recognizeHandwriting(emptyFile, "de").get();

            // The backend should return an error message about the empty file
            // instead of a 500 error or crash.
            assertTrue(response.contains("empty") || response.contains("error"),
                    "Response should indicate an error for empty file: " + response);
        } finally {
            Files.deleteIfExists(emptyFile.toPath());
        }
    }

    @Test
    public void testRecognizeValidFile() throws IOException, ExecutionException, InterruptedException {
        File tempFile = File.createTempFile("junit_test_valid_", ".png");
        try {
            // Create a small 100x100 white image
            BufferedImage img = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
            java.awt.Graphics2D g2d = img.createGraphics();
            g2d.setColor(java.awt.Color.WHITE);
            g2d.fillRect(0, 0, 100, 100);
            g2d.dispose();

            ImageIO.write(img, "png", tempFile);
            assertTrue(tempFile.length() > 0, "Temp file should not be empty");

            AIEngineClient client = new AIEngineClient();
            String response = client.recognizeHandwriting(tempFile, "de").get();

            // Expected success (status: success)
            assertTrue(response.contains("success"), "Response should be success: " + response);
        } finally {
            Files.deleteIfExists(tempFile.toPath());
        }
    }

    @Test
    public void testRecognizeRealScan() throws IOException, ExecutionException, InterruptedException {
        // Path to the actual student scan that was problematic
        File realScan = new File(
                "/home/t68/.gemini/antigravity/brain/f4a8f8cd-58e5-460a-ab57-9763a9f72929/uploaded_image_1767807161120.png");
        if (!realScan.exists()) {
            System.out.println("Real scan not found, skipping integration test: " + realScan.getAbsolutePath());
            return;
        }

        AIEngineClient client = new AIEngineClient();
        String response = client.recognizeHandwriting(realScan, "en").get();

        System.out.println("--- JAVA E2E INTEGRATION RESPONSE ---");
        System.out.println(response);
        System.out.println("--------------------------------------");

        assertTrue(response.contains("success"), "Real scan should return success");
        // Ensure some actual text is found and it doesn't just contain junk
        assertTrue(response.contains("Four generations") || response.contains("Lina Weaver"),
                "Should recognize key parts of the document");
        assertFalse(response.contains("1961 62") && response.split("1961 62").length > 3,
                "Response contains too many hallucinations (1961 62)");
    }

    @Test
    public void testRecognizeNonAsciiFilename()
            throws java.io.IOException, java.util.concurrent.ExecutionException, InterruptedException {
        File tempFile = File.createTempFile("Müller_", ".png");
        try {
            BufferedImage img = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
            java.awt.Graphics2D g2d = img.createGraphics();
            g2d.setColor(java.awt.Color.WHITE);
            g2d.fillRect(0, 0, 100, 100);
            g2d.dispose();

            javax.imageio.ImageIO.write(img, "png", tempFile);
            assertTrue(tempFile.length() > 0);

            AIEngineClient client = new AIEngineClient();
            String response = client.recognizeHandwriting(tempFile, "de").get();

            assertTrue(response.contains("success"),
                    "Response should be success even with non-ASCII filename: " + response);
        } finally {
            java.nio.file.Files.deleteIfExists(tempFile.toPath());
        }
    }
}
