package de.in.flaicheckbot;

import de.in.flaicheckbot.db.DatabaseManager;
import java.util.concurrent.CompletableFuture;
import java.io.File;

/**
 * Manual test to verify the Gemini API Key (AI Studio) path.
 * This class fetches the API key from the local database or allows manual
 * entry.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class GeminiApiKeyManualTest {

    public static void main(String[] args) {
        System.out.println("Starting Gemini API Key (AI Studio) Manual Test...");

        try {
            // 1. Initialize Database to get the stored key
            File dbFile = new File("flaicheckbot.db");
            if (!dbFile.exists()) {
                System.err.println("❌ Database file 'flaicheckbot.db' not found in project root.");
                System.err.println("Please run the MainApp once or ensure the DB is present.");
                return;
            }

            DatabaseManager dbManager = new DatabaseManager("flaicheckbot.db");
            String apiKey = dbManager.getSetting("gemini_api_key");

            if (apiKey == null || apiKey.isEmpty()) {
                System.out.println("⚠️ No API Key found in database.");
                if (args.length > 0) {
                    apiKey = args[0];
                    System.out.println("Using API Key from command line arguments.");
                } else {
                    System.err.println("❌ Error: No API Key provided.");
                    System.err.println(
                            "Usage: mvn exec:java -Dexec.mainClass=\"de.in.flaicheckbot.GeminiApiKeyManualTest\" -Dexec.args=\"YOUR_API_KEY\"");
                    return;
                }
            } else {
                System.out.println("✅ API Key loaded from database.");
            }

            // 2. Send Grading Request to AI Engine
            System.out.println("Step 2: Sending Grading Request to local AI Engine (API Key Path)...");
            AIEngineClient client = new AIEngineClient();

            String task = "Aufgabe: Rechnen Sie 15 * 4 (Max: 2 Punkte)";
            String actual = "Das Ergebnis ist 60.";

            // In API Key mode, we pass null for token and projectId, but provide the apiKey
            CompletableFuture<String> future = client.gradeStudentWorkVertexAI(
                    task,
                    "",
                    actual,
                    null, // No OAuth token
                    null, // No Project ID
                    apiKey);

            String response = future.join();
            System.out.println("\n--- AI Engine Response ---");
            System.out.println(response);
            System.out.println("--------------------------");

            if (response.contains("\"status\":\"error\"") && response.contains("\"code\":\"MODEL_NOT_FOUND\"")) {
                System.out.println("\n💡 Tipp: Das Modell wurde nicht gefunden.");
                System.out.println("Rufen Sie im Browser 'http://127.0.0.1:8000/models?apiKey=" + apiKey + "' auf,");
                System.out.println("um zu sehen, welche Modelle für Ihren Key verfügbar sind.");
            }

            if (response.contains("\"status\":\"success\"") || response.contains("\"feedback\"")) {
                System.out.println("\n✅ Gemini API Key Path is working correctly!");
            } else {
                System.out.println("\n❌ Gemini API Key Path failed. Check the response above.");
            }

        } catch (Exception e) {
            System.err.println("\n❌ Critical error during manual test:");
            e.printStackTrace();
        }
    }
}
