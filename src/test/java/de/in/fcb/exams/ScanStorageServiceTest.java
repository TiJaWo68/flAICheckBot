package de.in.fcb.exams;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;

import javax.imageio.ImageIO;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ScanStorageServiceTest {

    @TempDir
    Path tempDir;

    @Test
    void importsImageRotatesAndAppliesBlackRedactionBox() throws Exception {
        ScanStorageService service = new ScanStorageService(tempDir);

        BufferedImage img = new BufferedImage(200, 100, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = img.createGraphics();
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, 200, 100);
        g2d.dispose();

        ByteArrayOutputStream pngOut = new ByteArrayOutputStream();
        ImageIO.write(img, "png", pngOut);

        List<String> paths = service.importUpload(
                UUID.randomUUID(), UUID.randomUUID(), 1, "scan.png", pngOut.toByteArray());

        assertThat(paths).hasSize(1);
        String storedPath = paths.get(0);

        service.rotatePage(storedPath, 90);
        BufferedImage rotated = ImageIO.read(new ByteArrayInputStream(service.readImageBytes(storedPath)));
        assertThat(rotated.getWidth()).isEqualTo(100);
        assertThat(rotated.getHeight()).isEqualTo(200);

        service.applyRedactionBox(storedPath, 0.0, 0.0, 1.0, 0.20);
        BufferedImage redacted = ImageIO.read(new ByteArrayInputStream(service.readImageBytes(storedPath)));
        assertThat(new Color(redacted.getRGB(50, 10))).isEqualTo(Color.BLACK);
        assertThat(new Color(redacted.getRGB(50, 150))).isEqualTo(Color.WHITE);

        service.renderRedactedImage(storedPath, List.of(new RedactionBox(0.0, 0.5, 1.0, 0.25)));
        BufferedImage reRendered = ImageIO.read(new ByteArrayInputStream(service.readImageBytes(storedPath)));
        assertThat(new Color(reRendered.getRGB(50, 10))).isEqualTo(Color.WHITE);
        assertThat(new Color(reRendered.getRGB(50, 110))).isEqualTo(Color.BLACK);

        service.renderRedactedImage(storedPath, List.of());
        BufferedImage restored = ImageIO.read(new ByteArrayInputStream(service.readImageBytes(storedPath)));
        assertThat(new Color(restored.getRGB(50, 10))).isEqualTo(Color.WHITE);
        assertThat(new Color(restored.getRGB(50, 110))).isEqualTo(Color.WHITE);
    }

    @Test
    void rendersEachPdfPageIntoStoredPng() throws Exception {
        ScanStorageService service = new ScanStorageService(tempDir);

        ByteArrayOutputStream pdfOut = new ByteArrayOutputStream();
        try (PDDocument doc = new PDDocument()) {
            doc.addPage(new PDPage());
            doc.addPage(new PDPage());
            doc.save(pdfOut);
        }

        List<String> paths = service.importUpload(
                UUID.randomUUID(), UUID.randomUUID(), 1, "student.pdf", pdfOut.toByteArray());

        assertThat(paths).hasSize(2);
        assertThat(service.readImageBytes(paths.get(0))).isNotEmpty();
        assertThat(service.readImageBytes(paths.get(1))).isNotEmpty();
    }
}
