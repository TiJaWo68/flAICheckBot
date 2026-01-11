
package de.in.flaicheckbot;

import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;

/**
 * Reproduction test for the silent failure issue during local recognition.
 * Tries to perform a recognition request against the local AI engine.
 * 
 * Protocol step: 1. Reproduction (Test-First Fix)
 */
public class ReproduceSilentFailureTest {

    @Test
    @Tag("reproduction")
    public void testSilentFailureReproduction() {
        System.out.println("Starting reproduction test...");

        // 1. Setup a dummy image
        BufferedImage dummyImage = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);

        // 2. Initialize Client
        AIEngineClient client = new AIEngineClient();

        // 3. Attempt recognition with a timeout
        // The user reports the UI hangs/does nothing, so we expect this might hang or
        // fail silently
        // if the backend is in a bad state.
        try {
            System.out.println("Sending request to backend...");
            CompletableFuture<String> future = client.recognizeHandwriting(dummyImage, "en");

            // Wait for max 10 seconds (UI seemingly waits forever)
            String result = future.get(10, TimeUnit.SECONDS);

            System.out.println("Received result: " + result);
            assertNotNull(result, "Result should not be null");
            // If the backend is working, we expect some JSON. If it failed silently, we
            // might get empty string or invalid JSON.
            assertTrue(result.contains("status") || result.contains("done") || result.contains("line"),
                    "Result should contain valid JSON response components");

        } catch (java.util.concurrent.ExecutionException e) {
            // This is EXPECTED now. The client should throw an exception if response is
            // empty.
            String msg = e.getCause().getMessage();
            System.out.println("Caught expected exception: " + msg);
            assertTrue(msg.contains("empty response") || msg.contains("Check server logs"),
                    "Exception message should mention empty response");
        } catch (Exception e) {
            e.printStackTrace();
            fail("Caught unexpected exception: " + e.getClass().getName() + ": " + e.getMessage());
        }
    }
}
