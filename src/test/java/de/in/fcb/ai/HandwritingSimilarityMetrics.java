package de.in.fcb.ai;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Character-level (CER) and word-level (WER / Word Accuracy) similarity
 * metrics plus {@code [unleserlich]} marker tracking for handwriting OCR benchmarks.
 */
final class HandwritingSimilarityMetrics {

    private static final Pattern ILLEGIBLE_MARKER = Pattern.compile("\\[unleserlich]", Pattern.CASE_INSENSITIVE);

    private HandwritingSimilarityMetrics() {
    }

    record Metrics(
            int refChars,
            int hypChars,
            double cer,
            int refWords,
            int hypWords,
            double wer,
            double wordAccuracy,
            int illegibleCount) {
    }

    static Metrics compute(String referenceText, String recognizedText) {
        String normRef = normalizeWhitespace(referenceText);
        String normHyp = normalizeWhitespace(recognizedText);

        int charDist = levenshteinDistance(normRef, normHyp);
        double cer = charDist / (double) Math.max(1, normRef.length());

        List<String> refTokens = tokenizeWords(normRef);
        List<String> hypTokens = tokenizeWords(normHyp);
        int wordDist = wordLevenshteinDistance(refTokens, hypTokens);
        double wer = wordDist / (double) Math.max(1, refTokens.size());
        double wordAccuracy = Math.max(0.0, 1.0 - wer);

        int illegibleCount = countIllegibleMarkers(recognizedText);
        return new Metrics(
                normRef.length(),
                normHyp.length(),
                cer,
                refTokens.size(),
                hypTokens.size(),
                wer,
                wordAccuracy,
                illegibleCount);
    }

    static String normalizeWhitespace(String text) {
        if (text == null || text.isBlank()) {
            return "";
        }
        return text.trim().replaceAll("\\s+", " ");
    }

    static List<String> tokenizeWords(String normalizedText) {
        if (normalizedText == null || normalizedText.isBlank()) {
            return List.of();
        }
        return Arrays.asList(normalizedText.split(" "));
    }

    static int countIllegibleMarkers(String text) {
        if (text == null || text.isBlank()) {
            return 0;
        }
        Matcher matcher = ILLEGIBLE_MARKER.matcher(text.toLowerCase(Locale.ROOT));
        int count = 0;
        while (matcher.find()) {
            count++;
        }
        return count;
    }

    static int levenshteinDistance(String a, String b) {
        String left = a == null ? "" : a;
        String right = b == null ? "" : b;
        int n = left.length();
        int m = right.length();
        if (n == 0) {
            return m;
        }
        if (m == 0) {
            return n;
        }

        int[] prev = new int[m + 1];
        int[] curr = new int[m + 1];
        for (int j = 0; j <= m; j++) {
            prev[j] = j;
        }
        for (int i = 1; i <= n; i++) {
            curr[0] = i;
            char lc = left.charAt(i - 1);
            for (int j = 1; j <= m; j++) {
                int cost = (lc == right.charAt(j - 1)) ? 0 : 1;
                curr[j] = Math.min(Math.min(curr[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
            }
            int[] tmp = prev;
            prev = curr;
            curr = tmp;
        }
        return prev[m];
    }

    static int wordLevenshteinDistance(List<String> a, List<String> b) {
        List<String> left = a == null ? List.of() : a;
        List<String> right = b == null ? List.of() : b;
        int n = left.size();
        int m = right.size();
        if (n == 0) {
            return m;
        }
        if (m == 0) {
            return n;
        }

        int[] prev = new int[m + 1];
        int[] curr = new int[m + 1];
        for (int j = 0; j <= m; j++) {
            prev[j] = j;
        }
        for (int i = 1; i <= n; i++) {
            curr[0] = i;
            String lw = left.get(i - 1);
            for (int j = 1; j <= m; j++) {
                int cost = lw.equals(right.get(j - 1)) ? 0 : 1;
                curr[j] = Math.min(Math.min(curr[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
            }
            int[] tmp = prev;
            prev = curr;
            curr = tmp;
        }
        return prev[m];
    }
}
