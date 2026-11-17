package de.in.fcb.ai;

import java.io.IOException;
import java.math.BigDecimal;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.in.fcb.exams.TaskEvaluation;
import de.in.fcb.exams.TestDefinition;
import de.in.fcb.exams.TestTask;

/**
 * OpenAI-compatible AI client for model discovery, document task extraction,
 * and multimodal handwriting recognition + exam grading across Google AI Studio
 * (Gemini), OpenAI, and self-hosted endpoints.
 */
@Service
public class AiClient {

    private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(10);
    private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(120);

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

    /**
     * Extracts structured exam tasks and point allocations from raw document
     * text using the configured AI model.
     */
    public List<ExtractedTask> extractTasksFromText(AiConfig config, String documentText, String language) {
        requireConfigured(config);
        if (documentText == null || documentText.isBlank()) {
            throw new IllegalArgumentException("Document text must not be blank");
        }

        String systemPrompt = """
                You are an experienced foreign-language teacher. Parse the supplied exam text and extract the individual tasks.
                Return ONLY valid JSON with shape:
                {
                  "tasks": [
                    {
                      "position": 1,
                      "title": "Short task title",
                      "taskText": "Full prompt / question for the student",
                      "referenceText": "Expected solution or answer key if present in the text, else empty string",
                      "gradingHints": "Grading criteria or hints if present, else empty string",
                      "maxPoints": 10.0
                    }
                  ]
                }
                If point totals are missing, estimate sensible points (default 10.0). Preserve original language of the tasks.
                """;

        String userPrompt = "Target foreign language: " + (language == null ? "EN" : language)
                + "\n\nExam document text:\n" + documentText;

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("model", config.normalizedModel());
        payload.put("temperature", 0.1);
        payload.put("response_format", Map.of("type", "json_object"));
        payload.put("messages", List.of(
                Map.of("role", "system", "content", systemPrompt),
                Map.of("role", "user", "content", userPrompt)));

        String rawResponse = sendChatCompletion(config, payload);
        try {
            return parseExtractedTasksResponse(rawResponse);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to parse extracted tasks JSON: " + exception.getMessage(), exception);
        }
    }

    /**
     * Performs multimodal handwriting transcription and task-by-task grading on
     * scanned student pages using only the pseudonymized {@code studentExternalId}.
     */
    public GradingResult gradeStudentWork(
            AiConfig config,
            TestDefinition test,
            List<TestTask> tasks,
            String studentExternalId,
            List<byte[]> pagePngBytes) {
        requireConfigured(config);
        if (test == null) {
            throw new IllegalArgumentException("Test definition is required");
        }
        if (tasks == null || tasks.isEmpty()) {
            throw new IllegalArgumentException("At least one test task is required for grading");
        }
        if (pagePngBytes == null || pagePngBytes.isEmpty()) {
            throw new IllegalArgumentException("At least one scanned page is required");
        }

        try {
            String requestJson = buildGradingRequestJson(config, test, tasks, studentExternalId, pagePngBytes);
            String rawResponse = sendChatCompletionJson(config, requestJson);
            return parseGradingResponse(rawResponse, tasks, config.normalizedModel());
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to process AI grading response: " + exception.getMessage(), exception);
        }
    }

    String buildGradingRequestJson(
            AiConfig config,
            TestDefinition test,
            List<TestTask> tasks,
            String studentExternalId,
            List<byte[]> pagePngBytes) throws IOException {
        String systemPrompt = """
                You are an experienced, fair foreign-language teacher evaluating a handwritten student exam from scanned page images.
                Read every handwritten answer carefully across all pages, match each answer to its corresponding task, transcribe it accurately, and grade it against the reference solution and grading hints.
                Give helpful, constructive feedback in German (pointing out grammar, vocabulary, spelling, and content errors in the target foreign language).
                Never award more than maxPoints or less than 0 for any task.
                Return ONLY valid JSON matching this schema:
                {
                  "recognizedText": "Complete clean transcription of the student's work across all pages",
                  "overallFeedback": "Concise overall summary and encouragement in German",
                  "tasks": [
                    {
                      "position": 1,
                      "taskTitle": "Task title",
                      "recognizedAnswer": "Transcribed student answer for this task",
                      "awardedPoints": 7.5,
                      "maxPoints": 10.0,
                      "feedback": "Task-specific grading explanation in German",
                      "errorAnnotations": "Key grammar/vocabulary/spelling notes or empty string"
                    }
                  ]
                }
                """;

        List<Map<String, Object>> taskSpecs = new ArrayList<>();
        for (TestTask task : tasks) {
            Map<String, Object> spec = new LinkedHashMap<>();
            spec.put("position", task.getPosition());
            spec.put("title", task.getTitle());
            spec.put("taskText", task.getTaskText());
            spec.put("referenceText", task.getReferenceText() != null ? task.getReferenceText() : "");
            spec.put("gradingHints", task.getGradingHints() != null ? task.getGradingHints() : "");
            spec.put("maxPoints", task.getMaxPoints());
            taskSpecs.add(spec);
        }

        String userText = "Exam: " + test.getTitle()
                + "\nTarget language: " + test.getForeignLanguage()
                + "\nGrade level: " + (test.getGradeLevel() != null ? test.getGradeLevel() : "-")
                + "\nPseudonymized student ID: " + (studentExternalId != null ? studentExternalId : "unknown")
                + "\nTasks:\n" + objectMapper.writeValueAsString(taskSpecs);

        List<Map<String, Object>> contentParts = new ArrayList<>();
        contentParts.add(Map.of("type", "text", "text", userText));
        for (byte[] png : pagePngBytes) {
            String dataUri = "data:image/png;base64," + Base64.getEncoder().encodeToString(png);
            contentParts.add(Map.of("type", "image_url", "image_url", Map.of("url", dataUri)));
        }

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("model", config.normalizedModel());
        payload.put("temperature", 0.1);
        payload.put("response_format", Map.of("type", "json_object"));
        payload.put("messages", List.of(
                Map.of("role", "system", "content", systemPrompt),
                Map.of("role", "user", "content", contentParts)));

        return objectMapper.writeValueAsString(payload);
    }

    List<ExtractedTask> parseExtractedTasksResponse(String rawResponse) throws IOException {
        JsonNode content = extractAssistantJson(rawResponse);
        JsonNode tasksNode = content.path("tasks");
        if (!tasksNode.isArray()) {
            return List.of();
        }

        List<ExtractedTask> result = new ArrayList<>();
        int fallbackPos = 1;
        for (JsonNode node : tasksNode) {
            int position = node.hasNonNull("position") ? node.get("position").asInt(fallbackPos) : fallbackPos;
            String title = node.path("title").asText("Aufgabe " + position).trim();
            String taskText = node.path("taskText").asText("").trim();
            if (taskText.isBlank()) {
                taskText = title;
            }
            String referenceText = node.path("referenceText").asText("").trim();
            String gradingHints = node.path("gradingHints").asText("").trim();
            BigDecimal maxPoints = parseDecimal(node.path("maxPoints"), new BigDecimal("10.00"));
            if (maxPoints.signum() <= 0) {
                maxPoints = new BigDecimal("10.00");
            }
            result.add(new ExtractedTask(position, title, taskText, referenceText, gradingHints, maxPoints));
            fallbackPos++;
        }
        return result;
    }

    GradingResult parseGradingResponse(String rawResponse, List<TestTask> expectedTasks, String modelUsed)
            throws IOException {
        JsonNode root = objectMapper.readTree(rawResponse);
        JsonNode usage = root.path("usage");
        int promptTokens = usage.path("prompt_tokens").asInt(0);
        int completionTokens = usage.path("completion_tokens").asInt(0);

        JsonNode parsed = extractAssistantJson(root);
        String recognizedText = parsed.path("recognizedText").asText("").trim();
        String overallFeedback = parsed.path("overallFeedback").asText("").trim();

        Map<Integer, TestTask> tasksByPos = new LinkedHashMap<>();
        for (TestTask task : expectedTasks) {
            tasksByPos.put(task.getPosition(), task);
        }

        List<TaskEvaluation> evaluations = new ArrayList<>();
        JsonNode tasksNode = parsed.path("tasks");
        if (tasksNode.isArray()) {
            for (JsonNode entry : tasksNode) {
                int pos = entry.path("position").asInt(evaluations.size() + 1);
                TestTask defined = tasksByPos.get(pos);
                BigDecimal maxPts = defined != null
                        ? defined.getMaxPoints()
                        : parseDecimal(entry.path("maxPoints"), new BigDecimal("10.00"));
                BigDecimal awarded = clampPoints(parseDecimal(entry.path("awardedPoints"), BigDecimal.ZERO), maxPts);
                String title = entry.path("taskTitle").asText(defined != null ? defined.getTitle() : "Aufgabe " + pos);
                String answer = entry.path("recognizedAnswer").asText("").trim();
                String feedback = entry.path("feedback").asText("").trim();
                String annotations = entry.path("errorAnnotations").asText("").trim();
                evaluations.add(new TaskEvaluation(pos, title, answer, awarded, maxPts, feedback, annotations));
            }
        }

        return new GradingResult(recognizedText, overallFeedback, evaluations, promptTokens, completionTokens, modelUsed);
    }

    private String sendChatCompletion(AiConfig config, Map<String, Object> payload) {
        try {
            return sendChatCompletionJson(config, objectMapper.writeValueAsString(payload));
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to serialize AI request: " + exception.getMessage(), exception);
        }
    }

    private String sendChatCompletionJson(AiConfig config, String requestBody) {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(config.normalizedBaseUrl() + "/chat/completions"))
                .timeout(REQUEST_TIMEOUT)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody));
        String apiKey = config.effectiveApiKey();
        if (!apiKey.isBlank()) {
            builder.header("Authorization", "Bearer " + apiKey);
        }
        try {
            HttpResponse<String> response = httpClient.send(builder.build(), HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new IllegalStateException("HTTP " + response.statusCode() + ": " + truncate(response.body()));
            }
            return response.body();
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("AI request interrupted", exception);
        } catch (IOException exception) {
            throw new IllegalStateException("AI request failed: " + exception.getMessage(), exception);
        }
    }

    private JsonNode extractAssistantJson(String rawResponse) throws IOException {
        return extractAssistantJson(objectMapper.readTree(rawResponse));
    }

    private JsonNode extractAssistantJson(JsonNode root) throws IOException {
        JsonNode contentNode = root.path("choices").path(0).path("message").path("content");
        if (contentNode.isMissingNode() || contentNode.asText().isBlank()) {
            throw new IOException("AI response contains empty message content");
        }
        String cleaned = stripCodeFences(contentNode.asText());
        return objectMapper.readTree(cleaned);
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

    private static void requireConfigured(AiConfig config) {
        if (config == null || !config.isConfigured()) {
            throw new IllegalStateException("AI configuration is incomplete");
        }
    }

    private static String stripCodeFences(String text) {
        String trimmed = text.trim();
        if (trimmed.startsWith("```")) {
            int firstNewline = trimmed.indexOf('\n');
            if (firstNewline > 0) {
                trimmed = trimmed.substring(firstNewline + 1);
            }
            if (trimmed.endsWith("```")) {
                trimmed = trimmed.substring(0, trimmed.length() - 3);
            }
        }
        return trimmed.trim();
    }

    private static BigDecimal parseDecimal(JsonNode node, BigDecimal fallback) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return fallback;
        }
        if (node.isNumber()) {
            return node.decimalValue();
        }
        String raw = node.asText("").trim().replace(',', '.');
        if (raw.isBlank()) {
            return fallback;
        }
        try {
            return new BigDecimal(raw);
        } catch (NumberFormatException exception) {
            return fallback;
        }
    }

    private static BigDecimal clampPoints(BigDecimal value, BigDecimal maxPoints) {
        if (value == null || value.signum() < 0) {
            return BigDecimal.ZERO;
        }
        if (maxPoints != null && value.compareTo(maxPoints) > 0) {
            return maxPoints;
        }
        return value;
    }

    private static String truncate(String text) {
        if (text == null) {
            return "";
        }
        return text.length() > 200 ? text.substring(0, 200) + "..." : text;
    }
}
