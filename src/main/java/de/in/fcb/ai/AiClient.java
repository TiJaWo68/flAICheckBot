package de.in.fcb.ai;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * OpenAI-compatible AI client for verifying endpoint connectivity and listing
 * available models across Google AI Studio / Gemini, OpenAI, and self-hosted
 * endpoints.
 */
@Service
public class AiClient {

    private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(10);

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public AiClient() {
        this(HttpClient.newBuilder().connectTimeout(CONNECT_TIMEOUT).build());
    }

    AiClient(HttpClient httpClient) {
        this.httpClient = httpClient;
    }

    /**
     * Verifies reachability and credentials against {@code GET {baseUrl}/models}.
     */
    public void testConnection(AiConfig config) {
        if (config == null || !config.isConfigured()) {
            throw new IllegalStateException("AI configuration is incomplete");
        }
        listModels(config);
    }

    /**
     * Queries available model identifiers from {@code GET {baseUrl}/models}.
     * Normalizes optional {@code models/} prefixes returned by Gemini endpoints.
     */
    public List<String> listModels(AiConfig config) {
        if (config == null || !config.canQueryModels()) {
            throw new IllegalStateException("AI endpoint URL or API key is incomplete");
        }
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(config.normalizedBaseUrl() + "/models"))
                .timeout(CONNECT_TIMEOUT)
                .GET();
        String apiKey = config.effectiveApiKey();
        if (!apiKey.isBlank()) {
            builder.header("Authorization", "Bearer " + apiKey);
        }
        try {
            HttpResponse<String> response = httpClient.send(builder.build(), HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new IllegalStateException("HTTP " + response.statusCode() + ": " + truncate(response.body()));
            }
            return parseModelIds(response.body());
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Model query interrupted", exception);
        } catch (IOException exception) {
            throw new IllegalStateException("Model query failed: " + exception.getMessage(), exception);
        }
    }

    List<String> parseModelIds(String rawResponse) throws IOException {
        if (rawResponse == null || rawResponse.isBlank()) {
            return List.of();
        }
        JsonNode root = objectMapper.readTree(rawResponse);
        Set<String> models = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
        collectModelIds(root.path("data"), models);
        collectModelIds(root.path("models"), models);
        return List.copyOf(models);
    }

    private static void collectModelIds(JsonNode arrayNode, Set<String> target) {
        if (!arrayNode.isArray()) {
            return;
        }
        for (JsonNode entry : arrayNode) {
            String rawId = firstNonBlank(entry, "id", "name", "model");
            if (rawId != null) {
                String normalized = rawId.startsWith("models/")
                        ? rawId.substring("models/".length()).trim()
                        : rawId.trim();
                if (!normalized.isBlank()) {
                    target.add(normalized);
                }
            }
        }
    }

    private static String firstNonBlank(JsonNode node, String... fieldNames) {
        for (String field : fieldNames) {
            String value = node.path(field).asText("");
            if (!value.isBlank()) {
                return value.trim();
            }
        }
        return null;
    }

    private static String truncate(String text) {
        if (text == null) {
            return "";
        }
        return text.length() > 200 ? text.substring(0, 200) + "..." : text;
    }
}
