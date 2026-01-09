package de.in.flaicheckbot;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.Scanner;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import javax.imageio.ImageIO;

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

    /**
     * Interface for receiving real-time progress updates during OCR.
     */
    public interface ProgressListener {
        /**
         * Called when a line has been recognized.
         * 
         * @param page The 0-based index of the page.
         * @param text The recognized text.
         * @param bbox The bounding box in the original image.
         */
        void onLineRecognized(int page, int index, int total, String text, java.awt.Rectangle bbox);
    }

    /**
     * Consolidates OCR recognition by handling BufferedImage -> Temp File
     * conversion
     * and automatic cleanup.
     * 
     * @param image    The image to recognize.
     * @param language The language code (de, en, fr, es).
     * @return CompletableFuture with the raw JSON response.
     */
    public java.util.concurrent.CompletableFuture<String> recognizeHandwriting(BufferedImage image, String language) {
        return recognizeHandwriting(image, language, true);
    }

    public java.util.concurrent.CompletableFuture<String> recognizeHandwriting(BufferedImage image, String language,
            boolean usePreprocessing) {
        return recognizeHandwritingStreaming(image, language, usePreprocessing, null);
    }

    public CompletableFuture<String> recognizeHandwritingStreaming(BufferedImage image, String language,
            boolean usePreprocessing, ProgressListener listener) {
        return CompletableFuture.supplyAsync(() -> {
            File tempFile = null;
            try {
                tempFile = File.createTempFile("ocr_temp_", ".png");
                ImageIO.write(image, "png", tempFile);
                return recognizeHandwritingStreaming(tempFile, language, usePreprocessing, listener).get();
            } catch (Exception e) {
                logger.error("Failed to perform consolidated OCR", e);
                throw new RuntimeException(e);
            } finally {
                if (tempFile != null && tempFile.exists()) {
                    tempFile.delete();
                }
            }
        });
    }

    public java.util.concurrent.CompletableFuture<String> recognizeHandwriting(File imageFile, String language) {
        return recognizeHandwriting(imageFile, language, true);
    }

    public java.util.concurrent.CompletableFuture<String> recognizeHandwriting(File imageFile, String language,
            boolean usePreprocessing) {
        return recognizeHandwritingStreaming(imageFile, language, usePreprocessing, null);
    }

    /**
     * Performs OCR and provides real-time updates via the listener.
     */
    public CompletableFuture<String> recognizeHandwritingStreaming(File imageFile, String language,
            boolean usePreprocessing, ProgressListener listener) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                String boundary = "---" + UUID.randomUUID().toString();
                byte[] fileBytes = Files.readAllBytes(imageFile.toPath());

                java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                java.nio.charset.Charset utf8 = java.nio.charset.StandardCharsets.UTF_8;

                // Language part
                baos.write(("--" + boundary + "\r\n").getBytes(utf8));
                baos.write("Content-Disposition: form-data; name=\"language\"\r\n\r\n".getBytes(utf8));
                baos.write(((language != null ? language : "de") + "\r\n").getBytes(utf8));

                // Preprocess part
                baos.write(("--" + boundary + "\r\n").getBytes(utf8));
                baos.write("Content-Disposition: form-data; name=\"preprocess\"\r\n\r\n".getBytes(utf8));
                baos.write((String.valueOf(usePreprocessing) + "\r\n").getBytes(utf8));

                // File part
                String contentType = "image/png";
                if (imageFile.getName().toLowerCase().endsWith(".pdf")) {
                    contentType = "application/pdf";
                }

                baos.write(("--" + boundary + "\r\n").getBytes(utf8));
                baos.write(
                        ("Content-Disposition: form-data; name=\"file\"; filename=\"" + imageFile.getName() + "\"\r\n")
                                .getBytes(utf8));
                baos.write(("Content-Type: " + contentType + "\r\n\r\n").getBytes(utf8));
                baos.write(fileBytes);
                baos.write(("\r\n--" + boundary + "--\r\n").getBytes(utf8));

                HttpRequest request = HttpRequest.newBuilder().uri(URI.create(ENGINE_URL + "/recognize"))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(baos.toByteArray())).build();

                HttpResponse<java.io.InputStream> response = client.send(request,
                        HttpResponse.BodyHandlers.ofInputStream());

                if (response.statusCode() != 200) {
                    logger.error("AI Engine error: {}", response.statusCode());
                    throw new RuntimeException("AI Engine error: " + response.statusCode());
                }

                ObjectMapper mapper = new ObjectMapper();
                StringBuilder finalResult = new StringBuilder();

                try (Scanner scanner = new Scanner(response.body(), utf8.name())) {
                    while (scanner.hasNextLine()) {
                        String line = scanner.nextLine();
                        if (line.isEmpty())
                            continue;
                        try {
                            JsonNode node = mapper.readTree(line);
                            String type = node.path("type").asText();

                            if ("line".equals(type) && listener != null) {
                                int index = node.path("index").asInt();
                                int total = node.path("total").asInt();
                                String text = node.path("text").asText();
                                JsonNode bboxNode = node.path("bbox");
                                java.awt.Rectangle bbox = null;
                                if (bboxNode.isArray() && bboxNode.size() == 4) {
                                    bbox = new java.awt.Rectangle(
                                            bboxNode.get(0).asInt(),
                                            bboxNode.get(1).asInt(),
                                            bboxNode.get(2).asInt(),
                                            bboxNode.get(3).asInt());
                                }
                                int page = node.path("page").asInt(0);
                                listener.onLineRecognized(page, index, total, text, bbox);
                            } else if ("final".equals(type)) {
                                finalResult.append(line);
                            }
                        } catch (Exception e) {
                            logger.warn("Failed to parse streaming line: {}", line, e);
                        }
                    }
                }

                return finalResult.toString();
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

                java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                java.nio.charset.Charset utf8 = java.nio.charset.StandardCharsets.UTF_8;

                // Language part
                baos.write(("--" + boundary + "\r\n").getBytes(utf8));
                baos.write("Content-Disposition: form-data; name=\"language\"\r\n\r\n".getBytes(utf8));
                baos.write(((language != null ? language : "de") + "\r\n").getBytes(utf8));

                // File part
                baos.write(("--" + boundary + "\r\n").getBytes(utf8));
                baos.write(
                        ("Content-Disposition: form-data; name=\"file\"; filename=\"" + audioFile.getName() + "\"\r\n")
                                .getBytes(utf8));
                baos.write("Content-Type: audio/wav\r\n\r\n".getBytes(utf8));
                baos.write(fileBytes);
                baos.write(("\r\n--" + boundary + "--\r\n").getBytes(utf8));

                HttpRequest request = HttpRequest.newBuilder().uri(URI.create(ENGINE_URL + "/transcribe"))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(baos.toByteArray())).build();

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

                java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                java.nio.charset.Charset utf8 = java.nio.charset.StandardCharsets.UTF_8;

                // File part
                baos.write(("--" + boundary + "\r\n").getBytes(utf8));
                baos.write(
                        ("Content-Disposition: form-data; name=\"file\"; filename=\"" + imageFile.getName() + "\"\r\n")
                                .getBytes(utf8));
                baos.write("Content-Type: image/png\r\n\r\n".getBytes(utf8));
                baos.write(fileBytes);
                baos.write(("\r\n--" + boundary + "--\r\n").getBytes(utf8));

                HttpRequest request = HttpRequest.newBuilder().uri(URI.create(ENGINE_URL + "/preprocess"))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(baos.toByteArray())).build();

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

    public java.util.concurrent.CompletableFuture<String> trainModel(String language, String dataPath) {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try {
                String boundary = "---" + UUID.randomUUID().toString();
                java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                java.nio.charset.Charset utf8 = java.nio.charset.StandardCharsets.UTF_8;

                // Language part
                baos.write(("--" + boundary + "\r\n").getBytes(utf8));
                baos.write("Content-Disposition: form-data; name=\"language\"\r\n\r\n".getBytes(utf8));
                baos.write(((language != null ? language : "de") + "\r\n").getBytes(utf8));

                // Data path part
                if (dataPath != null && !dataPath.isEmpty()) {
                    baos.write(("--" + boundary + "\r\n").getBytes(utf8));
                    baos.write("Content-Disposition: form-data; name=\"data_path\"\r\n\r\n".getBytes(utf8));
                    baos.write((dataPath + "\r\n").getBytes(utf8));
                }

                baos.write(("--" + boundary + "--\r\n").getBytes(utf8));

                HttpRequest request = HttpRequest.newBuilder().uri(URI.create(ENGINE_URL + "/train"))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(baos.toByteArray())).build();

                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
                logger.debug("AI Engine train response: {}", response.statusCode());
                return response.body();
            } catch (IOException | InterruptedException e) {
                logger.error("Failed to communicate with AI Engine for training", e);
                throw new RuntimeException(e);
            }
        });
    }

    public java.util.concurrent.CompletableFuture<String> resetTraining(String language) {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try {
                String url = ENGINE_URL + "/reset";
                HttpRequest.Builder builder = HttpRequest.newBuilder().uri(URI.create(url));

                if (language != null && !language.isEmpty()) {
                    String boundary = "---" + UUID.randomUUID().toString();
                    String langPart = "--" + boundary + "\r\n" +
                            "Content-Disposition: form-data; name=\"language\"\r\n\r\n" +
                            language + "\r\n--" + boundary + "--\r\n";
                    builder.header("Content-Type", "multipart/form-data; boundary=" + boundary)
                            .POST(HttpRequest.BodyPublishers.ofString(langPart));
                } else {
                    builder.POST(HttpRequest.BodyPublishers.noBody());
                }

                HttpRequest request = builder.build();
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
