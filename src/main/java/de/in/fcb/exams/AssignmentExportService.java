package de.in.fcb.exams;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.font.PDFont;
import org.apache.pdfbox.pdmodel.font.PDType0Font;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;
import org.springframework.stereotype.Service;

/**
 * Exports assignment evaluation results as UTF-8 PDF reports and CSV gradebooks.
 */
@Service
public class AssignmentExportService {

    private static final List<String> REGULAR_FONT_CANDIDATES = List.of(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf");

    private static final List<String> BOLD_FONT_CANDIDATES = List.of(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf");

    private final AssignmentService assignmentService;

    public AssignmentExportService(AssignmentService assignmentService) {
        this.assignmentService = assignmentService;
    }

    public byte[] exportPdf(
            Assignment assignment,
            SchoolClass schoolClass,
            TestDefinition test,
            List<StudentWork> works,
            Map<UUID, Student> studentsById) throws IOException {
        try (PDDocument document = new PDDocument();
                ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            FontPair fonts = loadFonts(document);
            PdfCursor cursor = new PdfCursor(document, fonts);

            cursor.writeLine("flAICheckBot - Korrekturbericht", true, 16);
            cursor.y -= 6;
            cursor.writeLine("Klausur: " + assignment.getTitle(), true, 12);
            cursor.writeLine("Klasse: " + (schoolClass != null ? schoolClass.getName() : "-")
                    + "   |   Test: " + (test != null ? test.getTitle() : "-")
                    + "   |   Datum: " + assignment.getExamDate(), false, 10);
            cursor.y -= 12;

            for (StudentWork work : works) {
                cursor.ensureSpace(90);
                cursor.drawSeparator();

                Student student = studentsById.get(work.getStudentId());
                String label = student != null ? student.formatTeacherLabel() : work.getStudentId().toString();
                String score = formatScore(work.getAchievedPoints(), work.getMaxPoints());
                cursor.writeLine(label + "   [" + work.getStatus() + "]   —   " + score, true, 11);

                if (work.getOverallFeedback() != null && !work.getOverallFeedback().isBlank()) {
                    cursor.writeWrapped("Gesamtfeedback: " + work.getOverallFeedback(), false, 9.5f);
                }

                List<TaskEvaluation> evaluations = assignmentService.parseTaskEvaluations(work.getGradingJson());
                for (TaskEvaluation eval : evaluations) {
                    cursor.ensureSpace(40);
                    String taskLine = "  • Aufgabe " + eval.position() + " (" + eval.taskTitle() + "): "
                            + formatScore(eval.awardedPoints(), eval.maxPoints());
                    cursor.writeWrapped(taskLine, true, 9.5f);
                    if (eval.feedback() != null && !eval.feedback().isBlank()) {
                        cursor.writeWrapped("    " + eval.feedback(), false, 9f);
                    }
                }
                cursor.y -= 8;
            }

            cursor.close();
            document.save(out);
            return out.toByteArray();
        }
    }

    public byte[] exportCsv(
            Assignment assignment,
            List<StudentWork> works,
            Map<UUID, Student> studentsById) {
        StringBuilder csv = new StringBuilder();
        csv.append('\uFEFF');
        csv.append("Assignment;External-ID;Alias;Status;AchievedPoints;MaxPoints;Percent;OverallFeedback\n");

        for (StudentWork work : works) {
            Student student = studentsById.get(work.getStudentId());
            String extId = student != null ? student.getExternalId() : "";
            String alias = student != null && student.getDisplayAlias() != null ? student.getDisplayAlias() : "";
            String pct = computePercent(work.getAchievedPoints(), work.getMaxPoints());

            csv.append(csvField(assignment.getTitle())).append(';')
                    .append(csvField(extId)).append(';')
                    .append(csvField(alias)).append(';')
                    .append(csvField(work.getStatus().name())).append(';')
                    .append(work.getAchievedPoints().toPlainString()).append(';')
                    .append(work.getMaxPoints().toPlainString()).append(';')
                    .append(pct).append(';')
                    .append(csvField(work.getOverallFeedback())).append('\n');
        }
        return csv.toString().getBytes(StandardCharsets.UTF_8);
    }

    private static String formatScore(BigDecimal achieved, BigDecimal max) {
        BigDecimal a = achieved != null ? achieved : BigDecimal.ZERO;
        BigDecimal m = max != null ? max : BigDecimal.ZERO;
        return a.stripTrailingZeros().toPlainString() + " / " + m.stripTrailingZeros().toPlainString() + " Pkt ("
                + computePercent(a, m) + "%)";
    }

    private static String computePercent(BigDecimal achieved, BigDecimal max) {
        if (achieved == null || max == null || max.signum() <= 0) {
            return "0.0";
        }
        return achieved.multiply(new BigDecimal("100"))
                .divide(max, 1, RoundingMode.HALF_UP)
                .toPlainString();
    }

    private static String csvField(String value) {
        if (value == null) {
            return "\"\"";
        }
        String sanitized = value.replace("\r\n", " ").replace('\n', ' ').replace("\"", "\"\"");
        return "\"" + sanitized + "\"";
    }

    private static FontPair loadFonts(PDDocument document) throws IOException {
        Path regularPath = findFirstExisting(REGULAR_FONT_CANDIDATES);
        Path boldPath = findFirstExisting(BOLD_FONT_CANDIDATES);
        if (regularPath != null) {
            PDFont regular = PDType0Font.load(document, regularPath.toFile());
            PDFont bold = boldPath != null ? PDType0Font.load(document, boldPath.toFile()) : regular;
            return new FontPair(regular, bold, true);
        }
        return new FontPair(
                new PDType1Font(Standard14Fonts.FontName.HELVETICA),
                new PDType1Font(Standard14Fonts.FontName.HELVETICA_BOLD),
                false);
    }

    private static Path findFirstExisting(List<String> candidates) {
        for (String candidate : candidates) {
            Path path = Path.of(candidate);
            if (Files.isRegularFile(path)) {
                return path;
            }
        }
        return null;
    }

    private record FontPair(PDFont regular, PDFont bold, boolean unicode) {
    }

    private static class PdfCursor {
        private static final float MARGIN = 48f;
        private static final float LINE_LEADING = 13.5f;

        private final PDDocument document;
        private final FontPair fonts;
        private final float contentWidth;
        private PDPageContentStream stream;
        private float y;

        PdfCursor(PDDocument document, FontPair fonts) throws IOException {
            this.document = document;
            this.fonts = fonts;
            this.contentWidth = PDRectangle.A4.getWidth() - 2 * MARGIN;
            startNewPage();
        }

        void startNewPage() throws IOException {
            if (stream != null) {
                stream.close();
            }
            PDPage page = new PDPage(PDRectangle.A4);
            document.addPage(page);
            stream = new PDPageContentStream(document, page);
            y = PDRectangle.A4.getHeight() - MARGIN;
        }

        void ensureSpace(float needed) throws IOException {
            if (y - needed < MARGIN) {
                startNewPage();
            }
        }

        void drawSeparator() throws IOException {
            ensureSpace(16);
            stream.setLineWidth(0.6f);
            stream.moveTo(MARGIN, y);
            stream.lineTo(MARGIN + contentWidth, y);
            stream.stroke();
            y -= 12;
        }

        void writeLine(String text, boolean bold, float fontSize) throws IOException {
            ensureSpace(LINE_LEADING + 4);
            PDFont font = bold ? fonts.bold() : fonts.regular();
            String safeText = sanitizeForFont(text, fonts.unicode());
            stream.beginText();
            stream.setFont(font, fontSize);
            stream.newLineAtOffset(MARGIN, y);
            stream.showText(safeText);
            stream.endText();
            y -= LINE_LEADING;
        }

        void writeWrapped(String text, boolean bold, float fontSize) throws IOException {
            PDFont font = bold ? fonts.bold() : fonts.regular();
            String safe = sanitizeForFont(text, fonts.unicode());
            for (String line : wrap(safe, font, fontSize, contentWidth)) {
                ensureSpace(LINE_LEADING + 2);
                stream.beginText();
                stream.setFont(font, fontSize);
                stream.newLineAtOffset(MARGIN, y);
                stream.showText(line);
                stream.endText();
                y -= LINE_LEADING;
            }
        }

        void close() throws IOException {
            if (stream != null) {
                stream.close();
                stream = null;
            }
        }

        private static List<String> wrap(String text, PDFont font, float fontSize, float maxWidth) throws IOException {
            List<String> lines = new ArrayList<>();
            for (String paragraph : text.split("\\r?\\n")) {
                if (paragraph.isEmpty()) {
                    lines.add("");
                    continue;
                }
                StringBuilder current = new StringBuilder();
                for (String word : paragraph.split("\\s+")) {
                    String candidate = current.isEmpty() ? word : current + " " + word;
                    float width = font.getStringWidth(candidate) / 1000f * fontSize;
                    if (width > maxWidth && !current.isEmpty()) {
                        lines.add(current.toString());
                        current = new StringBuilder(word);
                    } else {
                        current = new StringBuilder(candidate);
                    }
                }
                if (!current.isEmpty()) {
                    lines.add(current.toString());
                }
            }
            return lines;
        }

        private static String sanitizeForFont(String text, boolean unicode) {
            if (text == null) {
                return "";
            }
            String cleaned = text.replace('\t', ' ').replaceAll("[\\p{Cntrl}&&[^\r\n]]", "");
            if (unicode) {
                return cleaned;
            }
            String normalized = Normalizer.normalize(cleaned, Normalizer.Form.NFC);
            StringBuilder out = new StringBuilder(normalized.length());
            for (int i = 0; i < normalized.length(); i++) {
                char c = normalized.charAt(i);
                out.append(c <= 0xFF ? c : '?');
            }
            return out.toString();
        }
    }
}
