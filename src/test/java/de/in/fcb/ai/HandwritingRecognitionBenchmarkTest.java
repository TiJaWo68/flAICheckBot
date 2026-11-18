package de.in.fcb.ai;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Live handwriting recognition benchmark evaluating {@link AiClient#transcribeHandwriting}
 * against exported PoC sample pairs ({@code *.png} + reference {@code *.txt}),
 * specifically measuring verbatim 1:1 preservation of authentic student errors and
 * {@code [unleserlich]} annotations.
 * <p>
 * Executed only via the Maven {@code -Pbenchmark-tests} profile.
 */
@Tag("benchmark")
class HandwritingRecognitionBenchmarkTest {

    private static final String DEFAULT_SAMPLES_DIR = "/mnt/BIFROST/Projekte/flAICheckbot/exported_samples";
    private static final Path REPORT_PATH = Path.of("target", "ocr-benchmark-report.md");

    private final AiClient aiClient = new AiClient();

    @Test
    void evaluateVerbatimTranscriptionAgainstExportedSamples() throws Exception {
        Path samplesRoot = Path.of(System.getProperty("fcb.samples.dir", DEFAULT_SAMPLES_DIR));
        Assumptions.assumeTrue(Files.isDirectory(samplesRoot),
                "Samples directory not available: " + samplesRoot);

        AiConfig config = resolveAiConfig();
        Assumptions.assumeTrue(config.isConfigured(),
                "AI endpoint/key not configured for benchmark (set FLAICHECKBOT_AI_API_KEY)");

        List<SamplePair> samples = discoverSamples(samplesRoot, System.getProperty("fcb.samples.filter", ""));
        assertThat(samples)
                .as("No matching PNG/TXT sample pairs found in %s", samplesRoot)
                .isNotEmpty();

        List<SampleResult> results = new ArrayList<>();
        for (SamplePair sample : samples) {
            byte[] pngBytes = Files.readAllBytes(sample.imagePath());
            String referenceText = Files.readString(sample.textPath(), StandardCharsets.UTF_8);

            long startNs = System.nanoTime();
            TranscriptionResult transcription = aiClient.transcribeHandwriting(
                    config, List.of(pngBytes), sample.language().toUpperCase(Locale.ROOT));
            long durationMs = (System.nanoTime() - startNs) / 1_000_000L;

            HandwritingSimilarityMetrics.Metrics metrics = HandwritingSimilarityMetrics.compute(
                    referenceText, transcription.recognizedText());
            results.add(new SampleResult(sample, referenceText, transcription, metrics, durationMs));

            System.out.printf(Locale.ROOT,
                    "[BENCHMARK] %-14s | WordAcc=%5.1f%% | CER=%5.1f%% | WER=%5.1f%% | illegible=%d | tokens=%d+%d | %d ms%n",
                    sample.id(),
                    metrics.wordAccuracy() * 100.0,
                    metrics.cer() * 100.0,
                    metrics.wer() * 100.0,
                    metrics.illegibleCount(),
                    transcription.promptTokens(),
                    transcription.completionTokens(),
                    durationMs);
        }

        writeReport(config, results);
        assertThat(Files.exists(REPORT_PATH)).isTrue();
    }

    private static AiConfig resolveAiConfig() {
        String presetRaw = firstNonBlank(
                System.getProperty("fcb.ai.preset"),
                System.getenv("FLAICHECKBOT_AI_PRESET"),
                AiPreset.GEMINI.name());
        AiPreset preset = AiPreset.fromKey(presetRaw);
        String baseUrl = firstNonBlank(
                System.getProperty("fcb.ai.base-url"),
                System.getenv("FLAICHECKBOT_AI_BASE_URL"),
                preset.defaultBaseUrl());
        String model = firstNonBlank(
                System.getProperty("fcb.ai.model"),
                System.getenv("FLAICHECKBOT_AI_MODEL"),
                preset.defaultModel());
        String apiKey = firstNonBlank(
                System.getProperty("fcb.ai.api-key"),
                System.getenv("FLAICHECKBOT_AI_API_KEY"),
                "");
        return new AiConfig(preset, baseUrl, model, apiKey);
    }

    private static List<SamplePair> discoverSamples(Path root, String filterExpr) throws IOException {
        Set<String> filterSet = Arrays.stream((filterExpr == null ? "" : filterExpr).split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .collect(Collectors.toSet());

        List<SamplePair> found = new ArrayList<>();
        for (String lang : List.of("en", "de")) {
            Path langDir = root.resolve(lang);
            if (!Files.isDirectory(langDir)) {
                continue;
            }
            try (Stream<Path> stream = Files.list(langDir)) {
                List<Path> pngs = stream
                        .filter(p -> Files.isRegularFile(p) && p.getFileName().toString().endsWith(".png"))
                        .sorted(Comparator.comparingInt(HandwritingRecognitionBenchmarkTest::extractSampleOrder)
                                .thenComparing(p -> p.getFileName().toString()))
                        .toList();
                for (Path png : pngs) {
                    String fileName = png.getFileName().toString();
                    String baseName = fileName.substring(0, fileName.length() - 4);
                    Path txt = langDir.resolve(baseName + ".txt");
                    if (!Files.isRegularFile(txt)) {
                        continue;
                    }
                    String id = lang + "/" + baseName;
                    if (!filterSet.isEmpty() && !filterSet.contains(id) && !filterSet.contains(baseName)) {
                        continue;
                    }
                    found.add(new SamplePair(id, lang, png, txt));
                }
            }
        }
        return found;
    }

    private static int extractSampleOrder(Path path) {
        String name = path.getFileName().toString();
        String digits = name.replaceAll("\\D+", "");
        if (digits.isEmpty()) {
            return Integer.MAX_VALUE;
        }
        try {
            return Integer.parseInt(digits);
        } catch (NumberFormatException ex) {
            return Integer.MAX_VALUE;
        }
    }

    private static void writeReport(AiConfig config, List<SampleResult> results) throws IOException {
        Files.createDirectories(REPORT_PATH.getParent());

        double avgWordAcc = results.stream().mapToDouble(r -> r.metrics().wordAccuracy()).average().orElse(0.0);
        double avgCer = results.stream().mapToDouble(r -> r.metrics().cer()).average().orElse(0.0);
        double avgWer = results.stream().mapToDouble(r -> r.metrics().wer()).average().orElse(0.0);
        int totalIllegible = results.stream().mapToInt(r -> r.metrics().illegibleCount()).sum();
        int totalPromptTokens = results.stream().mapToInt(r -> r.transcription().promptTokens()).sum();
        int totalCompletionTokens = results.stream().mapToInt(r -> r.transcription().completionTokens()).sum();
        long totalDurationMs = results.stream().mapToLong(SampleResult::durationMs).sum();

        StringBuilder md = new StringBuilder();
        md.append("# Verbatim Handwriting OCR Benchmark Report\n\n");
        md.append("- **Generated:** ").append(OffsetDateTime.now()).append('\n');
        md.append("- **Provider:** ").append(config.preset()).append(" (`").append(config.normalizedBaseUrl()).append("`)\n");
        md.append("- **Model:** `").append(config.normalizedModel()).append("`\n");
        md.append("- **Samples evaluated:** ").append(results.size()).append('\n');
        md.append(String.format(Locale.ROOT,
                "- **Average Word Accuracy:** %.1f%% · **Avg CER:** %.1f%% · **Avg WER:** %.1f%%\n",
                avgWordAcc * 100.0, avgCer * 100.0, avgWer * 100.0));
        md.append(String.format(Locale.ROOT,
                "- **Total `[unleserlich]` markers:** %d · **Total tokens:** %d prompt + %d completion · **Total time:** %.1f s\n\n",
                totalIllegible, totalPromptTokens, totalCompletionTokens, totalDurationMs / 1000.0));

        md.append("## Summary\n\n");
        md.append("| Sample | Ref Words | Hyp Words | Word Accuracy | CER | WER | `[unleserlich]` | Tokens (in/out) | Time (ms) |\n");
        md.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for (SampleResult r : results) {
            HandwritingSimilarityMetrics.Metrics m = r.metrics();
            TranscriptionResult t = r.transcription();
            md.append(String.format(Locale.ROOT,
                    "| `%s` | %d | %d | %.1f%% | %.1f%% | %.1f%% | %d | %d / %d | %d |\n",
                    r.sample().id(),
                    m.refWords(),
                    m.hypWords(),
                    m.wordAccuracy() * 100.0,
                    m.cer() * 100.0,
                    m.wer() * 100.0,
                    m.illegibleCount(),
                    t.promptTokens(),
                    t.completionTokens(),
                    r.durationMs()));
        }

        md.append("\n## Per-Sample Comparison\n");
        for (SampleResult r : results) {
            HandwritingSimilarityMetrics.Metrics m = r.metrics();
            md.append("\n### `").append(r.sample().id()).append("` ");
            md.append(String.format(Locale.ROOT, "(WordAcc %.1f%%, CER %.1f%%, `[unleserlich]`=%d)\n\n",
                    m.wordAccuracy() * 100.0, m.cer() * 100.0, m.illegibleCount()));
            if (!r.transcription().illegibleSegments().isEmpty()) {
                md.append("**Illegible annotations:**\n");
                for (String seg : r.transcription().illegibleSegments()) {
                    md.append("- ").append(seg).append('\n');
                }
                md.append('\n');
            }
            md.append("**Reference (`.txt`):**\n```text\n")
                    .append(r.referenceText().trim())
                    .append("\n```\n\n");
            md.append("**Recognized (`").append(config.normalizedModel()).append("`):**\n```text\n")
                    .append(r.transcription().recognizedText().trim())
                    .append("\n```\n");
        }

        Files.writeString(REPORT_PATH, md.toString(), StandardCharsets.UTF_8);
        System.out.println("[BENCHMARK] Wrote report to " + REPORT_PATH.toAbsolutePath());
    }

    private static String firstNonBlank(String... candidates) {
        for (String candidate : candidates) {
            if (candidate != null && !candidate.isBlank()) {
                return candidate.trim();
            }
        }
        return "";
    }

    private record SamplePair(String id, String language, Path imagePath, Path textPath) {
    }

    private record SampleResult(
            SamplePair sample,
            String referenceText,
            TranscriptionResult transcription,
            HandwritingSimilarityMetrics.Metrics metrics,
            long durationMs) {
    }
}
