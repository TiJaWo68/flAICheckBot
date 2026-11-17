package de.in.fcb.exams;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;
import org.junit.jupiter.api.Test;

class DocumentTextExtractorTest {

    private final DocumentTextExtractor extractor = new DocumentTextExtractor();

    @Test
    void extractsPlainTextAndMarkdown() throws Exception {
        byte[] bytes = "Task 1: Translate into French.".getBytes(StandardCharsets.UTF_8);

        assertThat(extractor.extractText("exam.txt", bytes)).isEqualTo("Task 1: Translate into French.");
        assertThat(extractor.extractText("exam.md", bytes)).isEqualTo("Task 1: Translate into French.");
    }

    @Test
    void extractsParagraphsFromDocxArchive() throws Exception {
        String xml = """
                <?xml version="1.0" encoding="UTF-8"?>
                <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                  <w:body>
                    <w:p><w:r><w:t>Aufgabe 1: Vocabulary</w:t></w:r></w:p>
                    <w:p><w:r><w:t>Aufgabe 2: Essay</w:t></w:r></w:p>
                  </w:body>
                </w:document>
                """;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ZipOutputStream zip = new ZipOutputStream(baos)) {
            zip.putNextEntry(new ZipEntry("word/document.xml"));
            zip.write(xml.getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }

        String text = extractor.extractText("klausur.docx", baos.toByteArray());

        assertThat(text).contains("Aufgabe 1: Vocabulary");
        assertThat(text).contains("Aufgabe 2: Essay");
    }

    @Test
    void extractsTextFromPdfDocument() throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (PDDocument doc = new PDDocument()) {
            PDPage page = new PDPage();
            doc.addPage(page);
            try (PDPageContentStream cs = new PDPageContentStream(doc, page)) {
                cs.beginText();
                cs.setFont(new PDType1Font(Standard14Fonts.FontName.HELVETICA), 12);
                cs.newLineAtOffset(50, 700);
                cs.showText("Task 1: Grammar Check");
                cs.endText();
            }
            doc.save(baos);
        }

        String text = extractor.extractText("exam.pdf", baos.toByteArray());

        assertThat(text).contains("Task 1: Grammar Check");
    }

    @Test
    void rejectsUnsupportedExtension() {
        assertThatThrownBy(() -> extractor.extractText("exam.exe", new byte[] { 1, 2, 3 }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Unsupported");
    }
}
