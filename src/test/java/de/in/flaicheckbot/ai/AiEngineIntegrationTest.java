package de.in.flaicheckbot.ai;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Files;
import java.util.concurrent.CompletableFuture;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import de.in.flaicheckbot.AIEngineClient;

/**
 * Integration test to verify connectivity and basic functionality
 * of the AI engine endpoints.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class AiEngineIntegrationTest {
    private static final Logger logger = LogManager.getLogger(AiEngineIntegrationTest.class);

    @BeforeAll
    public static void checkEngine() {
        AiProcessManager manager = new AiProcessManager();
        if (!manager.isEngineRunning()) {
            logger.info("AI Engine not running, starting for test...");
            manager.startEngine();
            assertTrue(manager.waitForEngine(30), "AI Engine failed to start for test");
        }
    }

    @Test
    public void testRecognitionEndpoint() throws Exception {
        AIEngineClient client = new AIEngineClient();

        // Create a dummy image for testing
        File tempFile = File.createTempFile("test_handwriting", ".png");
        try (FileOutputStream fos = new FileOutputStream(tempFile)) {
            byte[] dummyPng = getDummyPng();
            fos.write(dummyPng);
        }

        try {
            CompletableFuture<String> future = client.recognizeHandwriting(tempFile, "en");
            String result = future.get(30, java.util.concurrent.TimeUnit.SECONDS);

            assertNotNull(result, "Recognition result should not be null");
            logger.info("Integration Test - Recognition Result: {}", result);
        } finally {
            Files.deleteIfExists(tempFile.toPath());
        }
    }

    @Test
    public void testStreamingRecognition() throws Exception {
        AIEngineClient client = new AIEngineClient();
        File tempFile = File.createTempFile("test_stream", ".png");
        try (FileOutputStream fos = new FileOutputStream(tempFile)) {
            fos.write(getDummyPng());
        }

        final int[] lineCount = { 0 };
        try {
            CompletableFuture<String> future = client.recognizeHandwritingStreaming(tempFile, "en", true,
                    (index, total, text, bbox) -> {
                        lineCount[0]++;
                        logger.info("Stream Item: {}/{} - {} (BBox: {})", index, total, text, bbox);
                    });

            String result = future.get(30, java.util.concurrent.TimeUnit.SECONDS);
            assertNotNull(result, "Streaming result should not be null");
            // Since it is a dummy image, it might not find lines or might find noise.
            // But if it finds lines, our listener should be called.
            logger.info("Streaming Test - Final Result: {}, Lines Callback Count: {}", result, lineCount[0]);
        } finally {
            Files.deleteIfExists(tempFile.toPath());
        }
    }

    private byte[] getDummyPng() {
        return new byte[] {
                (byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0x0D, 0x49, 0x48, 0x44, 0x52,
                0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0, (byte) 0x90, 0x77, 0x53, (byte) 0xDE, 0, 0, 0, 0x0C,
                0x49, 0x44, 0x41, 0x54, 0x08, (byte) 0xD7, 0x63, (byte) 0xF8, (byte) 0xFF, (byte) 0xFF, 0x3F, 0,
                0x05, (byte) 0xFE, 0x02, (byte) 0xFE, (byte) 0xDC, 0x44, 0x74, 0x06,
                0, 0, 0, 0, 0x49, 0x45, 0x4E, 0x44, (byte) 0xAE, 0x42, 0x60, (byte) 0x82
        };
    }

    @Test
    public void testTrainingEndpoint() throws Exception {
        TrainingManager trainingManager = new TrainingManager();

        // Note: This actually triggers training if there is data in the DB
        // In a real CI environment, we'd use a test DB
        CompletableFuture<String> future = trainingManager.startTraining("de");
        String response = future.get(60, java.util.concurrent.TimeUnit.SECONDS);

        assertNotNull(response, "Training response should not be null");
        assertTrue(response.contains("success") || response.contains("No training samples"),
                "Response should indicate success or 'no samples', not a crash. Got: " + response);
        logger.info("Integration Test - Training Response: {}", response);
    }
}
