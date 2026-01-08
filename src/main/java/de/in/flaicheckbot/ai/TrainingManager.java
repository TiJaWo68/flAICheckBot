package de.in.flaicheckbot.ai;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.concurrent.CompletableFuture;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Client-side manager for triggering AI model training (fine-tuning) on the
 * local AI server.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class TrainingManager {
    private static final Logger logger = LogManager.getLogger(TrainingManager.class);
    private static final String ENGINE_URL = "http://127.0.0.1:8000";
    private final HttpClient client;

    public TrainingManager() {
        this.client = HttpClient.newHttpClient();
    }

    public CompletableFuture<String> startTraining(String language) {
        String boundary = "---" + java.util.UUID.randomUUID().toString();
        String langPart = "--" + boundary + "\r\n" +
                "Content-Disposition: form-data; name=\"language\"\r\n\r\n" +
                (language != null ? language : "de") + "\r\n";
        String foot = "--" + boundary + "--\r\n";
        String requestBody = langPart + foot;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(ENGINE_URL + "/train"))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        return client.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    logger.debug("AI Training Response Status: {}", response.statusCode());
                    return response.body();
                });
    }
}
