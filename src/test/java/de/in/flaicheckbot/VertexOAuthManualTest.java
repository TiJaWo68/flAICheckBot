package de.in.flaicheckbot;

import com.google.auth.Credentials;
import com.google.auth.oauth2.AccessToken;
import com.google.auth.oauth2.GoogleCredentials;
import de.in.flaicheckbot.util.GoogleLoginManager;
import java.util.concurrent.CompletableFuture;

/**
 * Manual test to verify the Vertex AI (GCP) path using OAuth.
 * This class expects a 'client_secret.json' in the project root.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class VertexOAuthManualTest {

    public static void main(String[] args) {
        System.out.println("Starting Vertex AI OAuth Manual Test...");

        try {
            // 1. Perform Login (uses client_secret.json)
            System.out.println("Step 1: Authenticating via GoogleLoginManager...");
            Credentials credentials = GoogleLoginManager.login();
            String projectId = GoogleLoginManager.getProjectId();

            if (credentials == null || projectId == null) {
                System.err.println("❌ Authentication failed: Credentials or Project ID missing.");
                return;
            }

            System.out.println("✅ Authenticated. Project ID: " + projectId);

            // 2. Get Access Token
            String token = "";
            if (credentials instanceof GoogleCredentials) {
                GoogleCredentials gcreds = (GoogleCredentials) credentials;
                AccessToken at = gcreds.refreshAccessToken();
                token = at.getTokenValue();
            }

            if (token.isEmpty()) {
                System.err.println("❌ Failed to retrieve Access Token.");
                return;
            }

            // 3. Send Grading Request to AI Engine
            System.out.println("Step 2: Sending Grading Request to local AI Engine (Vertex Path)...");
            AIEngineClient client = new AIEngineClient();

            String task = "Aufgabe: Beschreiben Sie die Photosynthese (Max: 10 Punkte)";
            String actual = "Pflanzen wandeln Licht in Energie um.";

            CompletableFuture<String> future = client.gradeStudentWorkVertexAI(
                    task,
                    "",
                    actual,
                    token,
                    projectId,
                    null // No API Key, we want to test OAuth
            );

            String response = future.join();
            System.out.println("\n--- AI Engine Response ---");
            System.out.println(response);
            System.out.println("--------------------------");

            if (response.contains("\"status\":\"success\"") || response.contains("\"feedback\"")) {
                System.out.println("\n✅ Vertex AI OAuth Path is working correctly!");
            } else {
                System.out.println("\n❌ Vertex AI OAuth Path failed. Check the response above.");
            }

        } catch (Exception e) {
            System.err.println("\n❌ Critical error during manual test:");
            e.printStackTrace();
        }
    }
}
