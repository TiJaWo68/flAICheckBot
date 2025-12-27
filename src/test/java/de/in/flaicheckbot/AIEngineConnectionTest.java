package de.in.flaicheckbot;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

/**
 * Utility class for verifying the connection to the AI engine
 * and testing basic API responses.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class AIEngineConnectionTest {

    public static void main(String[] args) {
        System.out.println("Starting AI Engine Connection Test...");
        String engineUrl = "http://127.0.0.1:8000";
        HttpClient client = HttpClient.newHttpClient();

        try {
            // 1. Test Ping
            System.out.println("Testing /ping endpoint...");
            HttpRequest pingRequest = HttpRequest.newBuilder()
                    .uri(URI.create(engineUrl + "/ping"))
                    .GET()
                    .build();

            HttpResponse<String> pingResponse = client.send(pingRequest, HttpResponse.BodyHandlers.ofString());
            System.out.println("Ping HTTP Status: " + pingResponse.statusCode());
            System.out.println("Ping Response Body: " + pingResponse.body());

            if (pingResponse.statusCode() == 200) {
                System.out.println("✅ Ping successful!");
            } else {
                System.out.println("❌ Ping failed!");
            }

            // (Further tests like /recognize could be added here if needed)

        } catch (Exception e) {
            System.err.println("❌ Critical Error during connection test:");
            e.printStackTrace();
        }
    }
}
