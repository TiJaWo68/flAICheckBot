package de.in.flaicheckbot.util;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import javax.swing.text.DefaultStyledDocument;
import javax.swing.text.rtf.RTFEditorKit;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.poi.hwpf.HWPFDocument;
import org.apache.poi.hwpf.extractor.WordExtractor;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;
import org.odftoolkit.odfdom.doc.OdfTextDocument;

/**
 * Utility class for extracting text from various document formats
 * like PDF, Word, ODT, and RTF.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class DocumentTextExtractor {

    public static String extractText(File file) throws Exception {
        String name = file.getName().toLowerCase();
        if (name.endsWith(".pdf")) {
            return extractFromPdf(file);
        } else if (name.endsWith(".docx")) {
            return extractFromDocx(file);
        } else if (name.endsWith(".doc")) {
            return extractFromDoc(file);
        } else if (name.endsWith(".odt")) {
            return extractFromOdt(file);
        } else if (name.endsWith(".txt") || name.endsWith(".md") || name.endsWith(".log")) {
            return extractFromPlainText(file);
        } else if (name.endsWith(".rtf")) {
            return extractFromRtf(file);
        } else {
            throw new IllegalArgumentException("Unsupported file format: " + name);
        }
    }

    private static String extractFromPdf(File file) throws Exception {
        try (PDDocument document = Loader.loadPDF(file)) {
            PDFTextStripper stripper = new PDFTextStripper();
            return stripper.getText(document);
        }
    }

    private static String extractFromDocx(File file) throws Exception {
        try (FileInputStream fis = new FileInputStream(file);
                XWPFDocument document = new XWPFDocument(fis)) {
            return document.getParagraphs().stream()
                    .map(XWPFParagraph::getText)
                    .collect(Collectors.joining("\n"));
        }
    }

    private static String extractFromDoc(File file) throws Exception {
        try (FileInputStream fis = new FileInputStream(file);
                HWPFDocument document = new HWPFDocument(fis);
                WordExtractor extractor = new WordExtractor(document)) {
            return extractor.getText();
        }
    }

    private static String extractFromOdt(File file) throws Exception {
        OdfTextDocument odt = OdfTextDocument.loadDocument(file);
        return odt.getContentRoot().getTextContent();
    }

    private static String extractFromPlainText(File file) throws Exception {
        return Files.readString(file.toPath(), StandardCharsets.UTF_8);
    }

    private static String extractFromRtf(File file) throws Exception {
        RTFEditorKit rtfKit = new RTFEditorKit();
        DefaultStyledDocument doc = new DefaultStyledDocument();
        try (InputStream is = new FileInputStream(file)) {
            rtfKit.read(is, doc, 0);
            return doc.getText(0, doc.getLength());
        }
    }

    /**
     * Renders all pages of a PDF to a single, vertically concatenated
     * BufferedImage.
     */
    public static List<BufferedImage> renderPdfToImages(File file) throws Exception {
        try (PDDocument document = Loader.loadPDF(file)) {
            PDFRenderer renderer = new PDFRenderer(document);
            List<BufferedImage> pageImages = new ArrayList<>();
            for (int i = 0; i < document.getNumberOfPages(); i++) {
                pageImages.add(renderer.renderImageWithDPI(i, 150));
            }
            return pageImages;
        }
    }

    public static BufferedImage renderPdfToImage(File file) throws Exception {
        List<BufferedImage> pageImages = renderPdfToImages(file);
        if (pageImages.isEmpty()) {
            return null;
        }

        int totalWidth = 0;
        int totalHeight = 0;
        for (BufferedImage img : pageImages) {
            totalWidth = Math.max(totalWidth, img.getWidth());
            totalHeight += img.getHeight();
        }

        BufferedImage combined = new BufferedImage(totalWidth, totalHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = combined.createGraphics();
        int currentY = 0;
        for (BufferedImage img : pageImages) {
            g2.drawImage(img, 0, currentY, null);
            currentY += img.getHeight();
        }
        g2.dispose();
        return combined;
    }
}
