package de.in.fcb.exams;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

import javax.imageio.ImageIO;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.drew.imaging.ImageMetadataReader;
import com.drew.metadata.Directory;
import com.drew.metadata.Metadata;
import com.drew.metadata.exif.ExifIFD0Directory;

/**
 * Stores student scan pages as PNG files on the filesystem, renders PDF pages,
 * corrects EXIF orientation on camera photos, rotates pages, and burns black
 * redaction boxes onto images prior to AI transmission.
 */
@Service
public class ScanStorageService {

    private static final float PDF_RENDER_DPI = 150f;
    private static final String PNG_FORMAT = "png";

    private final Path storageRoot;

    @Autowired
    public ScanStorageService(@Value("${flaicheckbot.storage.dir:./data/scans}") String storageDir) {
        this(Path.of(storageDir));
    }

    public ScanStorageService(Path storageRoot) {
        this.storageRoot = storageRoot.toAbsolutePath().normalize();
    }

    /**
     * Imports a PDF or image upload, converts each page to an EXIF-corrected
     * PNG under the student work folder, and returns the relative storage paths.
     */
    public List<String> importUpload(
            UUID assignmentId,
            UUID studentWorkId,
            int startPageNumber,
            String filename,
            byte[] content) throws IOException {
        if (assignmentId == null || studentWorkId == null) {
            throw new IllegalArgumentException("Assignment and student work IDs are required");
        }
        if (content == null || content.length == 0) {
            throw new IllegalArgumentException("Uploaded file is empty");
        }

        String lower = filename == null ? "" : filename.trim().toLowerCase(Locale.ROOT);
        List<BufferedImage> images = new ArrayList<>();
        if (lower.endsWith(".pdf")) {
            try (PDDocument doc = Loader.loadPDF(content)) {
                PDFRenderer renderer = new PDFRenderer(doc);
                for (int page = 0; page < doc.getNumberOfPages(); page++) {
                    images.add(renderer.renderImageWithDPI(page, PDF_RENDER_DPI));
                }
            }
        } else {
            BufferedImage image = ImageIO.read(new ByteArrayInputStream(content));
            if (image == null) {
                throw new IllegalArgumentException("Unsupported or corrupt image file: " + filename);
            }
            int orientation = readExifOrientation(content);
            images.add(applyExifOrientation(image, orientation));
        }

        if (images.isEmpty()) {
            throw new IllegalArgumentException("No pages found in upload: " + filename);
        }

        Path workDir = storageRoot.resolve(assignmentId.toString()).resolve(studentWorkId.toString()).normalize();
        Files.createDirectories(workDir);

        List<String> storedPaths = new ArrayList<>(images.size());
        int pageNum = startPageNumber;
        for (BufferedImage pageImage : images) {
            String fileName = "page-" + pageNum + "-" + UUID.randomUUID().toString().substring(0, 8) + ".png";
            Path target = workDir.resolve(fileName).normalize();
            BufferedImage rgb = toRgbImage(pageImage);
            ImageIO.write(rgb, PNG_FORMAT, target.toFile());
            ImageIO.write(rgb, PNG_FORMAT, resolveOriginalPath(target).toFile());
            storedPaths.add(storageRoot.relativize(target).toString().replace('\\', '/'));
            pageNum++;
        }
        return storedPaths;
    }

    /**
     * Rotates an already stored PNG image (and its untouched original backup, if
     * present) clockwise in 90-degree increments.
     */
    public void rotatePage(String storagePath, int degreesClockwise) throws IOException {
        int normalized = ((degreesClockwise % 360) + 360) % 360;
        if (normalized == 0) {
            return;
        }
        if (normalized != 90 && normalized != 180 && normalized != 270) {
            throw new IllegalArgumentException("Rotation must be a multiple of 90 degrees");
        }

        Path path = resolveSafePath(storagePath);
        rotateFileInPlace(path, normalized);
        Path origPath = resolveOriginalPath(path);
        if (Files.exists(origPath)) {
            rotateFileInPlace(origPath, normalized);
        }
    }

    /**
     * Burns a single solid black rectangle onto the stored PNG image using
     * normalized coordinates ({@code 0.0..1.0}).
     */
    public void applyRedactionBox(
            String storagePath,
            double normX,
            double normY,
            double normWidth,
            double normHeight) throws IOException {
        Path path = resolveSafePath(storagePath);
        ensureOriginalBackup(path);
        BufferedImage source = ImageIO.read(path.toFile());
        if (source == null) {
            throw new IllegalArgumentException("Cannot read image at " + storagePath);
        }

        BufferedImage canvas = toRgbImage(source);
        drawRedactionBoxes(canvas, List.of(new RedactionBox(normX, normY, normWidth, normHeight)));
        ImageIO.write(canvas, PNG_FORMAT, path.toFile());
    }

    /**
     * Re-renders the stored PNG from its unredacted original backup by burning
     * all supplied {@link RedactionBox} entries onto it, enabling lossless undo
     * and full redaction reset without leaking unredacted pixels into the
     * AI-facing image file.
     */
    public void renderRedactedImage(String storagePath, List<RedactionBox> boxes) throws IOException {
        Path path = resolveSafePath(storagePath);
        Path origPath = ensureOriginalBackup(path);
        BufferedImage source = ImageIO.read(origPath.toFile());
        if (source == null) {
            throw new IOException("Cannot read original image at " + origPath.getFileName());
        }

        BufferedImage canvas = toRgbImage(source);
        if (boxes != null && !boxes.isEmpty()) {
            drawRedactionBoxes(canvas, boxes);
        }
        ImageIO.write(canvas, PNG_FORMAT, path.toFile());
    }

    public byte[] readImageBytes(String storagePath) throws IOException {
        Path path = resolveSafePath(storagePath);
        return Files.readAllBytes(path);
    }

    public void deleteFileIfExists(String storagePath) {
        if (storagePath == null || storagePath.isBlank()) {
            return;
        }
        try {
            Path path = resolveSafePath(storagePath);
            Files.deleteIfExists(path);
            Files.deleteIfExists(resolveOriginalPath(path));
        } catch (IOException ignored) {
            // Best-effort cleanup
        }
    }

    private void rotateFileInPlace(Path path, int degreesClockwise) throws IOException {
        BufferedImage source = ImageIO.read(path.toFile());
        if (source == null) {
            throw new IOException("Cannot read image at " + path.getFileName());
        }
        BufferedImage rotated = rotateClockwise(source, degreesClockwise);
        ImageIO.write(rotated, PNG_FORMAT, path.toFile());
    }

    private static Path ensureOriginalBackup(Path targetPath) throws IOException {
        Path origPath = resolveOriginalPath(targetPath);
        if (!Files.exists(origPath) && Files.exists(targetPath)) {
            Files.copy(targetPath, origPath, StandardCopyOption.REPLACE_EXISTING);
        }
        return origPath;
    }

    static Path resolveOriginalPath(Path targetPath) {
        String name = targetPath.getFileName().toString();
        String origName = name.endsWith(".png")
                ? name.substring(0, name.length() - 4) + ".orig.png"
                : name + ".orig.png";
        return targetPath.resolveSibling(origName);
    }

    private static void drawRedactionBoxes(BufferedImage canvas, List<RedactionBox> boxes) {
        int width = canvas.getWidth();
        int height = canvas.getHeight();
        Graphics2D g2d = canvas.createGraphics();
        try {
            g2d.setColor(Color.BLACK);
            for (RedactionBox box : boxes) {
                double x0 = clamp(box.x());
                double y0 = clamp(box.y());
                double wNorm = clamp(box.width());
                double hNorm = clamp(box.height());
                if (wNorm <= 0 || hNorm <= 0) {
                    throw new IllegalArgumentException("Redaction rectangle width and height must be positive");
                }
                int px = (int) Math.round(x0 * width);
                int py = (int) Math.round(y0 * height);
                int pw = Math.max(1, (int) Math.round(Math.min(1.0 - x0, wNorm) * width));
                int ph = Math.max(1, (int) Math.round(Math.min(1.0 - y0, hNorm) * height));
                g2d.fillRect(px, py, pw, ph);
            }
        } finally {
            g2d.dispose();
        }
    }

    private Path resolveSafePath(String storagePath) {
        if (storagePath == null || storagePath.isBlank()) {
            throw new IllegalArgumentException("Storage path must not be blank");
        }
        Path resolved = storageRoot.resolve(storagePath).normalize();
        if (!resolved.startsWith(storageRoot)) {
            throw new IllegalArgumentException("Invalid storage path outside scan root: " + storagePath);
        }
        return resolved;
    }

    private int readExifOrientation(byte[] imageBytes) {
        try {
            Metadata metadata = ImageMetadataReader.readMetadata(new ByteArrayInputStream(imageBytes));
            Directory directory = metadata.getFirstDirectoryOfType(ExifIFD0Directory.class);
            if (directory != null && directory.containsTag(ExifIFD0Directory.TAG_ORIENTATION)) {
                return directory.getInt(ExifIFD0Directory.TAG_ORIENTATION);
            }
        } catch (Exception ignored) {
            // Non-JPEG or image without EXIF metadata
        }
        return 1;
    }

    private BufferedImage applyExifOrientation(BufferedImage image, int orientation) {
        return switch (orientation) {
            case 3 -> rotateClockwise(image, 180);
            case 6 -> rotateClockwise(image, 90);
            case 8 -> rotateClockwise(image, 270);
            default -> image;
        };
    }

    private BufferedImage rotateClockwise(BufferedImage source, int degrees) {
        int srcW = source.getWidth();
        int srcH = source.getHeight();
        int dstW = (degrees == 90 || degrees == 270) ? srcH : srcW;
        int dstH = (degrees == 90 || degrees == 270) ? srcW : srcH;

        BufferedImage result = new BufferedImage(dstW, dstH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = result.createGraphics();
        try {
            AffineTransform transform = new AffineTransform();
            transform.translate(dstW / 2.0, dstH / 2.0);
            transform.rotate(Math.toRadians(degrees));
            transform.translate(-srcW / 2.0, -srcH / 2.0);
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, dstW, dstH);
            g2d.drawRenderedImage(source, transform);
        } finally {
            g2d.dispose();
        }
        return result;
    }

    private static BufferedImage toRgbImage(BufferedImage source) {
        if (source.getType() == BufferedImage.TYPE_INT_RGB) {
            return source;
        }
        BufferedImage rgb = new BufferedImage(source.getWidth(), source.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = rgb.createGraphics();
        try {
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, source.getWidth(), source.getHeight());
            g2d.drawImage(source, 0, 0, null);
        } finally {
            g2d.dispose();
        }
        return rgb;
    }

    private static double clamp(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }
}
