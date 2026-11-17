package de.in.fcb.exams;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.stereotype.Component;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * Extracts plain text from uploaded exam definitions ({@code .pdf},
 * {@code .docx}, {@code .txt}, {@code .md}).
 */
@Component
public class DocumentTextExtractor {

    private static final String DOCX_DOCUMENT_ENTRY = "word/document.xml";

    public String extractText(String filename, byte[] content) throws IOException {
        if (filename == null || filename.isBlank()) {
            throw new IllegalArgumentException("Filename must not be blank");
        }
        if (content == null || content.length == 0) {
            throw new IllegalArgumentException("Uploaded document is empty");
        }

        String lower = filename.trim().toLowerCase(Locale.ROOT);
        if (lower.endsWith(".pdf")) {
            return extractFromPdf(content);
        }
        if (lower.endsWith(".docx")) {
            return extractFromDocx(content);
        }
        if (lower.endsWith(".txt") || lower.endsWith(".md")) {
            return new String(content, StandardCharsets.UTF_8).trim();
        }
        throw new IllegalArgumentException("Unsupported document format: " + filename);
    }

    private String extractFromPdf(byte[] content) throws IOException {
        try (PDDocument document = Loader.loadPDF(content)) {
            PDFTextStripper stripper = new PDFTextStripper();
            return stripper.getText(document).trim();
        }
    }

    private String extractFromDocx(byte[] content) throws IOException {
        try (ZipInputStream zip = new ZipInputStream(new ByteArrayInputStream(content))) {
            ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                if (DOCX_DOCUMENT_ENTRY.equals(entry.getName())) {
                    return parseWordDocumentXml(zip);
                }
            }
        }
        throw new IOException("Invalid DOCX archive: missing " + DOCX_DOCUMENT_ENTRY);
    }

    private String parseWordDocumentXml(InputStream xmlStream) throws IOException {
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
            factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            factory.setNamespaceAware(true);
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document doc = builder.parse(xmlStream);

            StringBuilder result = new StringBuilder();
            NodeList paragraphs = doc.getElementsByTagNameNS("*", "p");
            for (int i = 0; i < paragraphs.getLength(); i++) {
                Element paragraph = (Element) paragraphs.item(i);
                NodeList textNodes = paragraph.getElementsByTagNameNS("*", "t");
                StringBuilder line = new StringBuilder();
                for (int j = 0; j < textNodes.getLength(); j++) {
                    String segment = textNodes.item(j).getTextContent();
                    if (segment != null) {
                        line.append(segment);
                    }
                }
                if (!line.isEmpty()) {
                    if (!result.isEmpty()) {
                        result.append('\n');
                    }
                    result.append(line);
                }
            }
            return result.toString().trim();
        } catch (ParserConfigurationException | SAXException exception) {
            throw new IOException("Failed to parse DOCX XML content", exception);
        }
    }
}
