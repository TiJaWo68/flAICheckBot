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

    public CompletableFuture<String> startTraining() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(ENGINE_URL + "/train"))
                .POST(HttpRequest.BodyPublishers.noBody())
                .build();

        return client.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    logger.debug("AI Training Response Status: {}", response.statusCode());
                    return response.body();
                });
    }
}
