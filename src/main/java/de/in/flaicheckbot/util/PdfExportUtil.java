package de.in.flaicheckbot.util;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;

import de.in.flaicheckbot.db.DatabaseManager;

/**
 * Utility for exporting evaluation results to PDF using Apache PDFBox.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class PdfExportUtil {

    public static void export(File file, DatabaseManager.AssignmentInfo assignment, List<StudentResult> results)
            throws IOException {
        try (PDDocument document = new PDDocument()) {
            ExportState state = new ExportState(document);

            // Header
            state.writeBoldLine("Evaluationsergebnisse", 18);
            state.y -= 10;
            state.writeBoldLine("Test: " + assignment.title, 12);
            state.writeLine("Klasse: " + assignment.className, 11);
            state.y -= 20;

            // Results
            for (StudentResult res : results) {
                state.checkPageBreak(100);

                // Separator Line
                state.contentStream.setLineWidth(1f);
                state.contentStream.moveTo(state.margin, state.y);
                state.contentStream.lineTo(state.margin + state.width, state.y);
                state.contentStream.stroke();
                state.y -= 15;

                // Student Name & Score
                state.writeBoldLine("Schüler: " + res.name, 12, false);
                float scoreWidth = state.fontBold.getStringWidth(res.score) / 1000 * 12;
                state.writeTextAt(state.margin + state.width - scoreWidth, state.y, res.score, state.fontBold, 12);
                state.y -= 20;

                // Feedback
                state.writeLine("Bewertung / Feedback:", 10);
                state.y -= 5;

                List<String> wrappedLines = wrapText(res.feedback, state.fontNormal, 10, state.width);
                for (String line : wrappedLines) {
                    state.checkPageBreak(30);
                    state.writeLine(line, 10);
                }
                state.y -= 10;
            }

            state.close();
            document.save(file);
        }
    }

    private static class ExportState {
        PDDocument document;
        PDPage page;
        PDPageContentStream contentStream;
        float y;
        float margin = 50;
        float width;
        PDType1Font fontBold;
        PDType1Font fontNormal;

        ExportState(PDDocument doc) throws IOException {
            this.document = doc;
            this.fontBold = new PDType1Font(Standard14Fonts.FontName.HELVETICA_BOLD);
            this.fontNormal = new PDType1Font(Standard14Fonts.FontName.HELVETICA);
            newPage();
        }

        void newPage() throws IOException {
            if (contentStream != null) {
                contentStream.close();
            }
            page = new PDPage(PDRectangle.A4);
            document.addPage(page);
            contentStream = new PDPageContentStream(document, page);
            y = page.getMediaBox().getHeight() - 50;
            width = page.getMediaBox().getWidth() - 2 * margin;
        }

        void checkPageBreak(float requiredSpace) throws IOException {
            if (y < requiredSpace) {
                newPage();
            }
        }

        void writeTextAt(float x, float y, String text, PDType1Font font, float size) throws IOException {
            contentStream.beginText();
            contentStream.setFont(font, size);
            contentStream.newLineAtOffset(x, y);
            contentStream.showText(safeText(text));
            contentStream.endText();
        }

        private String safeText(String text) {
            if (text == null)
                return "";
            // Remove control characters and non-WinAnsiEncoding characters that PDFBox Type
            // 1 fonts can't handle
            return text.replaceAll("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F]", "");
        }

        void writeLine(String text, float size) throws IOException {
            writeLine(text, size, true);
        }

        void writeBoldLine(String text, float size) throws IOException {
            writeBoldLine(text, size, true);
        }

        void writeLine(String text, float size, boolean advance) throws IOException {
            writeTextAt(margin, y, text, fontNormal, size);
            if (advance)
                y -= (size * 1.2f);
        }

        void writeBoldLine(String text, float size, boolean advance) throws IOException {
            writeTextAt(margin, y, text, fontBold, size);
            if (advance)
                y -= (size * 1.2f);
        }

        void close() throws IOException {
            if (contentStream != null) {
                contentStream.close();
            }
        }
    }

    private static List<String> wrapText(String text, PDType1Font font, float fontSize, float width)
            throws IOException {
        List<String> result = new ArrayList<>();
        if (text == null || text.isEmpty())
            return result;

        String[] lines = text.split("\\r?\\n");
        for (String line : lines) {
            String[] words = line.split("\\s+");
            StringBuilder sb = new StringBuilder();
            for (String word : words) {
                if (word.isEmpty())
                    continue;
                String potential = sb.length() == 0 ? word : sb.toString() + " " + word;
                float w = 0;
                try {
                    w = font.getStringWidth(potential) / 1000 * fontSize;
                } catch (IllegalArgumentException e) {
                    // Fallback for unknown characters
                    w = potential.length() * (fontSize * 0.6f);
                }

                if (w > width) {
                    if (sb.length() > 0)
                        result.add(sb.toString());
                    sb = new StringBuilder(word);
                } else {
                    if (sb.length() > 0)
                        sb.append(" ");
                    sb.append(word);
                }
            }
            if (sb.length() > 0)
                result.add(sb.toString());
        }
        return result;
    }

    public static class StudentResult {
        public String name;
        public String feedback;
        public String score;

        public StudentResult(String name, String feedback, String score) {
            this.name = name;
            this.feedback = feedback;
            this.score = score;
        }
    }
}
