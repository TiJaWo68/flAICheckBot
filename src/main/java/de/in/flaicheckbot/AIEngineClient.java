package de.in.flaicheckbot;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.util.UUID;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * HTTP client for communicating with the AI engine API,
 * handling recognition, preprocessing, and transcription requests.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class AIEngineClient {
    private static final Logger logger = LogManager.getLogger(AIEngineClient.class);
    private static final String ENGINE_URL = "http://127.0.0.1:8000";
    private final HttpClient client;

    public AIEngineClient() {
        this.client = HttpClient.newHttpClient();
    }

    public java.util.concurrent.CompletableFuture<String> recognizeHandwriting(File imageFile) {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try {
                String boundary = "---" + UUID.randomUUID().toString();
                byte[] fileBytes = Files.readAllBytes(imageFile.toPath());

                String head = "--" + boundary + "\r\n" +
                        "Content-Disposition: form-data; name=\"file\"; filename=\"" + imageFile.getName() + "\"\r\n" +
                        "Content-Type: image/png\r\n\r\n";
                String foot = "\r\n--" + boundary + "--\r\n";

                byte[] requestBody = new byte[head.length() + fileBytes.length + foot.length()];
                System.arraycopy(head.getBytes(), 0, requestBody, 0, head.length());
                System.arraycopy(fileBytes, 0, requestBody, head.length(), fileBytes.length);
                System.arraycopy(foot.getBytes(), 0, requestBody, head.length() + fileBytes.length, foot.length());

                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(ENGINE_URL + "/recognize"))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(requestBody))
                        .build();

                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
                logger.debug("AI Engine response: {}", response.statusCode());
                return response.body();
            } catch (IOException | InterruptedException e) {
                logger.error("Failed to communicate with AI Engine", e);
                throw new RuntimeException(e);
            }
        });
    }

    public java.util.concurrent.CompletableFuture<String> transcribeAudio(File audioFile, String language) {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try {
                String boundary = "---" + UUID.randomUUID().toString();
                byte[] fileBytes = Files.readAllBytes(audioFile.toPath());

                String langPart = "--" + boundary + "\r\n" +
                        "Content-Disposition: form-data; name=\"language\"\r\n\r\n" +
                        (language != null ? language : "de") + "\r\n";

                String filePart = "--" + boundary + "\r\n" +
                        "Content-Disposition: form-data; name=\"file\"; filename=\"" + audioFile.getName() + "\"\r\n" +
                        "Content-Type: audio/wav\r\n\r\n";
                String foot = "\r\n--" + boundary + "--\r\n";

                byte[] requestBody = new byte[langPart.length() + filePart.length() + fileBytes.length + foot.length()];
                int pos = 0;
                System.arraycopy(langPart.getBytes(), 0, requestBody, pos, langPart.length());
                pos += langPart.length();
                System.arraycopy(filePart.getBytes(), 0, requestBody, pos, filePart.length());
                pos += filePart.length();
                System.arraycopy(fileBytes, 0, requestBody, pos, fileBytes.length);
                pos += fileBytes.length;
                System.arraycopy(foot.getBytes(), 0, requestBody, pos, foot.length());

                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(ENGINE_URL + "/transcribe"))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(requestBody))
                        .build();

                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
                logger.debug("AI Engine transcribe response: {}", response.statusCode());
                return response.body();
            } catch (IOException | InterruptedException e) {
                logger.error("Failed to communicate with AI Engine for transcription", e);
                throw new RuntimeException(e);
            }
        });
    }

    public String calibrate(String studentId, File imageFile) throws IOException, InterruptedException {
        // (Keep existing calibrate logic if needed, but recognize is primary now)
        return "Not implemented in prototype (use recognize/train)";
    }

    public java.util.concurrent.CompletableFuture<byte[]> preprocessImage(File imageFile) {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try {
                String boundary = "---" + UUID.randomUUID().toString();
                byte[] fileBytes = Files.readAllBytes(imageFile.toPath());

                String head = "--" + boundary + "\r\n" +
                        "Content-Disposition: form-data; name=\"file\"; filename=\"" + imageFile.getName() + "\"\r\n" +
                        "Content-Type: image/png\r\n\r\n";
                String foot = "\r\n--" + boundary + "--\r\n";

                byte[] requestBody = new byte[head.length() + fileBytes.length + foot.length()];
                System.arraycopy(head.getBytes(), 0, requestBody, 0, head.length());
                System.arraycopy(fileBytes, 0, requestBody, head.length(), fileBytes.length);
                System.arraycopy(foot.getBytes(), 0, requestBody, head.length() + fileBytes.length, foot.length());

                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(ENGINE_URL + "/preprocess"))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(requestBody))
                        .build();

                HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
                if (response.statusCode() == 200) {
                    return response.body();
                } else {
                    throw new RuntimeException("AI Engine error (preprocess): " + response.statusCode());
                }
            } catch (IOException | InterruptedException e) {
                logger.error("Failed to communicate with AI Engine for preprocessing", e);
                throw new RuntimeException(e);
            }
        });
    }

    public java.util.concurrent.CompletableFuture<String> gradeStudentWork(String task, String expected,
            String actual) {
        return gradeWithEndpoint("/grade", task, expected, actual);
    }

    public java.util.concurrent.CompletableFuture<String> gradeStudentWorkVertexAI(String task, String expected,
            String actual, String accessToken, String projectId, String apiKey) {
        return gradeWithEndpoint("/grade_vertex", task, expected, actual, accessToken, projectId, apiKey);
    }

    private java.util.concurrent.CompletableFuture<String> gradeWithEndpoint(String endpoint, String task,
            String expected, String actual) {
        return gradeWithEndpoint(endpoint, task, expected, actual, null, null, null);
    }

    private java.util.concurrent.CompletableFuture<String> gradeWithEndpoint(String endpoint, String task,
            String expected, String actual, String accessToken, String projectId, String apiKey) {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try {
                String boundary = "---" + UUID.randomUUID().toString();

                String taskPart = "--" + boundary + "\r\n" +
                        "Content-Disposition: form-data; name=\"task\"\r\n\r\n" +
                        task + "\r\n";
                String expectedPart = "--" + boundary + "\r\n" +
                        "Content-Disposition: form-data; name=\"expected\"\r\n\r\n" +
                        expected + "\r\n";
                String actualPart = "--" + boundary + "\r\n" +
                        "Content-Disposition: form-data; name=\"actual\"\r\n\r\n" +
                        actual + "\r\n";
                String tokenPart = "";
                if (accessToken != null && !accessToken.isEmpty()) {
                    tokenPart = "--" + boundary + "\r\n" +
                            "Content-Disposition: form-data; name=\"token\"\r\n\r\n" +
                            accessToken + "\r\n";
                }
                String projectPart = "";
                if (projectId != null && !projectId.isEmpty()) {
                    projectPart = "--" + boundary + "\r\n" +
                            "Content-Disposition: form-data; name=\"projectId\"\r\n\r\n" +
                            projectId + "\r\n";
                }
                String apiKeyPart = "";
                if (apiKey != null && !apiKey.isEmpty()) {
                    apiKeyPart = "--" + boundary + "\r\n" +
                            "Content-Disposition: form-data; name=\"apiKey\"\r\n\r\n" +
                            apiKey + "\r\n";
                }
                String foot = "--" + boundary + "--\r\n";

                String requestBodyStr = taskPart + expectedPart + actualPart + tokenPart + projectPart + apiKeyPart
                        + foot;

                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(ENGINE_URL + endpoint))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofString(requestBodyStr))
                        .build();

                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
                logger.debug("AI Engine grade response ({}): {}", endpoint, response.statusCode());
                return response.body();
            } catch (IOException | InterruptedException e) {
                logger.error("Failed to communicate with AI Engine for grading at " + endpoint, e);
                throw new RuntimeException(e);
            }
        });
    }

    public java.util.concurrent.CompletableFuture<String> resetTraining() {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try {
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(ENGINE_URL + "/reset"))
                        .POST(HttpRequest.BodyPublishers.noBody())
                        .build();

                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
                logger.debug("AI Engine reset response: {}", response.statusCode());
                return response.body();
            } catch (IOException | InterruptedException e) {
                logger.error("Failed to reset AI Engine", e);
                throw new RuntimeException(e);
            }
        });
    }
}
