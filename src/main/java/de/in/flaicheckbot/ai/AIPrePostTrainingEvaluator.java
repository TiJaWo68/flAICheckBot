package de.in.flaicheckbot.ai;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import de.in.flaicheckbot.AIEngineClient;

/**
 * Evaluates the improvement of the local AI model by running recognition
 * before and after training on a set of exported samples.
 * 
 * Generates an HTML report with the findings.
 */
public class AIPrePostTrainingEvaluator {

    private static final Logger logger = LogManager.getLogger(AIPrePostTrainingEvaluator.class);
    private final AIEngineClient client;
    private final File samplesDir;
    private final String language;

    public AIPrePostTrainingEvaluator(File samplesDir, String language) {
        this.client = new AIEngineClient();
        this.samplesDir = samplesDir;
        this.language = language;
    }

    public static void main(String[] args) {
        File baseDir = new File("exported_samples");
        if (!baseDir.exists()) {
            System.err.println("Base samples directory not found: " + baseDir.getAbsolutePath());
            return;
        }

        File[] langDirs = baseDir.listFiles(File::isDirectory);
        if (langDirs == null || langDirs.length == 0) {
            System.err.println("No language subdirectories found in " + baseDir.getAbsolutePath());
            return;
        }

        for (File dir : langDirs) {
            String lang = dir.getName();
            logger.info("Starting Evaluation for language: " + lang);
            AIPrePostTrainingEvaluator evaluator = new AIPrePostTrainingEvaluator(dir, lang);
            try {
                evaluator.runEvaluation();
            } catch (Exception e) {
                logger.error("Evaluation failed for " + lang, e);
            }
        }
    }

    public void runEvaluation() throws Exception {
        logger.info("Starting AI Evaluation for " + language + " in " + samplesDir.getAbsolutePath());

        // 1. Load Samples
        Map<File, String> samples = loadSamples();
        if (samples.isEmpty()) {
            logger.warn("No samples found (pairs of .png and .txt) in " + samplesDir.getAbsolutePath());
            return;
        }
        logger.info("Loaded " + samples.size() + " samples.");

        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmm").format(new Date());
        String modelId = language + "-eval-" + timestamp;

        // 2. Baseline Test (Untrained, Preprocessed)
        logger.info("Running Baseline Test (Untrained, Preprocessed)...");
        Map<File, String> baselineResults = runRecognitionTest(samples, language, true);

        // 3. Raw Test (Untrained, NO Preprocessing)
        logger.info("Running Raw Test (Untrained, NO Preprocessing)...");
        Map<File, String> rawResults = runRecognitionTest(samples, language, false);

        // 4. Train Model
        logger.info("Training Model '" + modelId + "'...");
        File trainingTempDir = prepareTrainingData(samples);
        try {
            String trainResponse = client.trainModel(language, trainingTempDir.getAbsolutePath()).get();
            logger.info("Training finished: " + trainResponse);
        } finally {
            // cleanup temp dir? maybe keep for debug
        }

        // 5. Post-Training Test (Trained, Preprocessed)
        logger.info("Running Post-Training Test...");
        Map<File, String> trainedResults = runRecognitionTest(samples, language, true);

        // 6. Generate Report
        generateHtmlReport(samples, baselineResults, rawResults, trainedResults, modelId, timestamp);
        logger.info("Evaluation Complete for " + language);
    }

    private Map<File, String> loadSamples() throws IOException {
        Map<File, String> loaded = new HashMap<>();
        File[] files = samplesDir.listFiles();
        if (files == null)
            return loaded;

        for (File f : files) {
            if (f.getName().toLowerCase().endsWith(".png")) {
                String baseName = f.getName().substring(0, f.getName().lastIndexOf('.'));
                File txtFile = new File(samplesDir, baseName + ".txt");
                if (txtFile.exists()) {
                    String text = Files.readString(txtFile.toPath(), StandardCharsets.UTF_8).trim();
                    loaded.put(f, text);
                }
            }
        }
        return loaded;
    }

    private File prepareTrainingData(Map<File, String> samples) throws IOException {
        File tempDir = Files.createTempDirectory("ai_training_" + System.currentTimeMillis()).toFile();
        for (Map.Entry<File, String> entry : samples.entrySet()) {
            File srcImg = entry.getKey();
            String text = entry.getValue();

            File destImg = new File(tempDir, srcImg.getName());
            File destTxt = new File(tempDir, srcImg.getName().replace(".png", ".txt"));

            Files.copy(srcImg.toPath(), destImg.toPath(), StandardCopyOption.REPLACE_EXISTING);
            Files.writeString(destTxt.toPath(), text, StandardCharsets.UTF_8);
        }
        return tempDir;
    }

    private Map<File, String> runRecognitionTest(Map<File, String> samples, String language, boolean preprocess) {
        Map<File, String> results = new HashMap<>();
        for (File img : samples.keySet()) {
            try {
                // Using the file-based overload with preprocessing flag
                String jsonResult = client.recognizeHandwriting(img, language, preprocess).get();
                // Simple parsing to extract text
                String text = extractTextFromJson(jsonResult);
                results.put(img, text);
            } catch (Exception e) {
                logger.error("Error recognizing " + img.getName(), e);
                results.put(img, "ERROR: " + e.getMessage());
            }
        }
        return results;
    }

    private String extractTextFromJson(String json) {
        // Quick and dirty JSON parsing to avoid heavy deps if not available,
        // or use Regex. The server returns {"status": "success", "text": "...", ...}
        // Need to handle escaped quotes if present.
        try {
            int textIndex = json.indexOf("\"text\":");
            if (textIndex == -1)
                return "";

            int startQuote = json.indexOf("\"", textIndex + 7);
            // This is brittle for complex JSON strings, but sufficient for this prototype
            // context
            // Ideally use a proper JSON parser (Jackson/Gson) if available in classpath
            // Let's assume Jackson is on CP since it's a standard java project
            // But to be safe and dependency-free for this single class:

            // Let's assume standard simple JSON structure from our python server
            // We can use a slightly more robust regex or logic
            // Or better yet, rely on the fact that we can fix this if it breaks.

            // Let's try to find end quote by iterating and skipping escaped quotes
            StringBuilder sb = new StringBuilder();
            boolean escaped = false;
            for (int i = startQuote + 1; i < json.length(); i++) {
                char c = json.charAt(i);
                if (escaped) {
                    sb.append(c);
                    escaped = false;
                } else {
                    if (c == '\\') {
                        escaped = true;
                    } else if (c == '"') {
                        break; // End of string
                    } else {
                        sb.append(c);
                    }
                }
            }
            // Handle newline escapes manually if regex didn't catch them
            return sb.toString().replace("\\n", "\n");
        } catch (Exception e) {
            return "JSON Parsing Error";
        }
    }

    private void generateHtmlReport(Map<File, String> samples,
            Map<File, String> baseline,
            Map<File, String> raw,
            Map<File, String> trained,
            String modelId, String timestamp) throws IOException {

        StringBuilder html = new StringBuilder();
        html.append("<html><head><style>")
                .append("body { font-family: sans-serif; }")
                .append("table { border-collapse: collapse; width: 100%; }")
                .append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }")
                .append("th { background-color: #f2f2f2; }")
                .append(".diff-loss { color: red; }")
                .append(".diff-gain { color: green; }")
                .append("img { max-width: 300px; }")
                .append("</style></head><body>");

        html.append("<h1>AI Training & Preprocessing Evaluation Report</h1>");
        html.append("<p>Date: ").append(timestamp).append("</p>");
        html.append("<p>Model ID: ").append(modelId).append("</p>");

        html.append("<table>");
        html.append(
                "<tr><th>Image</th><th>Ground Truth</th><th>Baseline (Preproc)</th><th>Raw (No Preproc)</th><th>Trained (Preproc)</th><th>Improvement (T vs B)</th></tr>");

        double totalSimBefore = 0;
        double totalSimRaw = 0;
        double totalSimAfter = 0;
        int count = 0;

        for (File img : samples.keySet()) {
            String truth = samples.get(img);
            String pre = baseline.get(img);
            String rw = raw.get(img);
            String post = trained.get(img);

            int distBefore = levenshteinDistance(truth, pre);
            int distRaw = levenshteinDistance(truth, rw);
            int distAfter = levenshteinDistance(truth, post);

            // Normalized Similarity (0 to 1)
            double simBefore = 1.0 - ((double) distBefore / Math.max(truth.length(), pre.length()));
            double simRaw = 1.0 - ((double) distRaw / Math.max(truth.length(), rw.length()));
            double simAfter = 1.0 - ((double) distAfter / Math.max(truth.length(), post.length()));

            if (simBefore < 0)
                simBefore = 0;
            if (simRaw < 0)
                simRaw = 0;
            if (simAfter < 0)
                simAfter = 0;

            totalSimBefore += simBefore;
            totalSimRaw += simRaw;
            totalSimAfter += simAfter;
            count++;

            html.append("<tr>");
            html.append("<td><img src='file://").append(img.getAbsolutePath()).append("'><br>").append(img.getName())
                    .append("</td>");
            html.append("<td><pre>").append(escapeHtml(truth)).append("</pre></td>");
            html.append("<td><pre>").append(escapeHtml(pre)).append("</pre><br><i>Dist: ").append(distBefore)
                    .append(" (").append(String.format("%.1f%%", simBefore * 100)).append(")</i></td>");
            html.append("<td><pre>").append(escapeHtml(rw)).append("</pre><br><i>Dist: ").append(distRaw)
                    .append(" (").append(String.format("%.1f%%", simRaw * 100)).append(")</i></td>");
            html.append("<td><pre>").append(escapeHtml(post)).append("</pre><br><i>Dist: ").append(distAfter)
                    .append(" (").append(String.format("%.1f%%", simAfter * 100)).append(")</i></td>");

            double improvement = (simAfter - simBefore) * 100;
            String color = improvement > 0 ? "green" : (improvement < 0 ? "red" : "black");
            html.append("<td style='color:").append(color).append("'>").append(String.format("%+.1f%%", improvement))
                    .append("</td>");
            html.append("</tr>");
        }
        html.append("</table>");

        if (count > 0) {
            double avgBefore = (totalSimBefore / count) * 100;
            double avgRaw = (totalSimRaw / count) * 100;
            double avgAfter = (totalSimAfter / count) * 100;
            html.append("<h2>Summary</h2>");
            html.append("<p>Average Accuracy Baseline (Preproc): <b>").append(String.format("%.2f%%", avgBefore))
                    .append("</b></p>");
            html.append("<p>Average Accuracy Raw (No Preproc): <b>").append(String.format("%.2f%%", avgRaw))
                    .append("</b></p>");
            html.append("<p>Average Accuracy After Training: <b>").append(String.format("%.2f%%", avgAfter))
                    .append("</b></p>");

            double rawVsBase = avgRaw - avgBefore;
            String rawColor = rawVsBase > 0 ? "green" : (rawVsBase < 0 ? "red" : "black");
            html.append("<p>Impact of Preprocessing: <b style='color:").append(rawColor).append("'>")
                    .append(String.format("%+.2f%%", -rawVsBase))
                    .append("</b> (negative means preprocessing is hurting)</p>");
        }

        html.append("</body></html>");

        File reportFile = new File("evaluation_report_" + timestamp + ".html");
        try (FileWriter fw = new FileWriter(reportFile)) {
            fw.write(html.toString());
        }
        logger.info("Report saved to: " + reportFile.getAbsolutePath());
    }

    private int levenshteinDistance(String a, String b) {
        // Standard implementation
        int[][] dp = new int[a.length() + 1][b.length() + 1];
        for (int i = 0; i <= a.length(); i++)
            dp[i][0] = i;
        for (int j = 0; j <= b.length(); j++)
            dp[0][j] = j;
        for (int i = 1; i <= a.length(); i++) {
            for (int j = 1; j <= b.length(); j++) {
                int cost = (a.charAt(i - 1) == b.charAt(j - 1)) ? 0 : 1;
                dp[i][j] = Math.min(Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + cost);
            }
        }
        return dp[a.length()][b.length()];
    }

    private String escapeHtml(String text) {
        if (text == null)
            return "";
        return text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;");
    }
}
