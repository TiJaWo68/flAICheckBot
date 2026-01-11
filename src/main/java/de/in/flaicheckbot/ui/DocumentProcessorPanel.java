package de.in.flaicheckbot.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JOptionPane;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import java.util.Base64;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.api.gax.core.FixedCredentialsProvider;
import com.google.cloud.vision.v1.AnnotateImageRequest;
import com.google.cloud.vision.v1.AnnotateImageResponse;
import com.google.cloud.vision.v1.BatchAnnotateImagesResponse;
import com.google.cloud.vision.v1.Feature;
import com.google.cloud.vision.v1.ImageAnnotatorClient;
import com.google.cloud.vision.v1.ImageAnnotatorSettings;
import com.google.protobuf.ByteString;

import de.in.flaicheckbot.AIEngineClient;
import de.in.flaicheckbot.db.DatabaseManager;
import de.in.flaicheckbot.util.DocumentTextExtractor;
import de.in.flaicheckbot.util.UndoHelper;
import de.in.utils.gui.ExceptionMessage;
import de.in.utils.gui.WrapLayout;

/**
 * Base panel for components that process documents (Image + OCR).
 * Consolidates image viewing, preprocessing, and OCR logic.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public abstract class DocumentProcessorPanel extends JPanel {
    protected static final Logger logger = LogManager.getLogger(DocumentProcessorPanel.class);

    protected final DatabaseManager dbManager;
    protected JTextArea txtResult;
    protected JPanel pagesPanel;
    protected JPanel segmentsPanel;
    protected List<ZoomableImagePanel> imagePanels = new ArrayList<>();
    protected File currentFile;
    protected byte[] currentRawData;

    public DocumentProcessorPanel(DatabaseManager dbManager) {
        this.dbManager = dbManager;
    }

    protected void initBaseUI() {
        setLayout(new BorderLayout(10, 10));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        splitPane.setResizeWeight(0.5);

        // Left: Image View
        JPanel leftPanel = new JPanel(new BorderLayout());
        leftPanel.setBorder(BorderFactory.createTitledBorder("Vorschau & Bearbeitung"));

        leftPanel.add(createImageToolbar(), BorderLayout.NORTH);

        pagesPanel = new JPanel();
        pagesPanel.setLayout(new BoxLayout(pagesPanel, BoxLayout.Y_AXIS));
        JScrollPane scrollPane = new JScrollPane(pagesPanel);
        scrollPane.getVerticalScrollBar().setUnitIncrement(16);
        leftPanel.add(scrollPane, BorderLayout.CENTER);

        splitPane.setLeftComponent(leftPanel);

        // Right: OCR Text
        JPanel rightPanel = new JPanel(new BorderLayout());
        rightPanel.setBorder(BorderFactory.createTitledBorder("Erkannter Text"));

        rightPanel.add(createOcrToolbar(), BorderLayout.NORTH);

        txtResult = new JTextArea();
        txtResult.setFont(txtResult.getFont().deriveFont(txtResult.getFont().getSize2D() + 2f));
        txtResult.setLineWrap(true);
        txtResult.setWrapStyleWord(true);
        UndoHelper.addUndoSupport(txtResult);

        setupTextEventListeners(txtResult);

        rightPanel.add(new JScrollPane(txtResult), BorderLayout.CENTER);

        // Right Most: Segments (Visual Debug)
        JPanel debugPanel = new JPanel(new BorderLayout());
        debugPanel.setBorder(BorderFactory.createTitledBorder("Zeilen-Segmente (Debug)"));
        segmentsPanel = new JPanel();
        segmentsPanel.setLayout(new BoxLayout(segmentsPanel, BoxLayout.Y_AXIS));
        debugPanel.add(new JScrollPane(segmentsPanel), BorderLayout.CENTER);

        JSplitPane innerSplit = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, rightPanel, debugPanel);
        innerSplit.setResizeWeight(0.5);

        splitPane.setRightComponent(innerSplit);

        add(splitPane, BorderLayout.CENTER);

        setupDropTargets();
    }

    protected abstract JPanel createImageToolbar();

    protected abstract JPanel createOcrToolbar();

    protected JPanel createStandardImageControls() {
        JPanel editRow = new JPanel(new WrapLayout(FlowLayout.LEFT, 5, 2));

        JButton btnZoomIn = createIconButton("🔍+", "Vergrößern", e -> imagePanels.forEach(p -> p.adjustZoom(0.2)));
        JButton btnZoomOut = createIconButton("🔍-", "Verkleinern", e -> imagePanels.forEach(p -> p.adjustZoom(-0.2)));
        JButton btnFit = createIconButton("⛶", "Einpassen", e -> imagePanels.forEach(p -> p.fitToScreen()));
        JButton btnFitWidth = createIconButton("↔", "Auf Breite einpassen",
                e -> imagePanels.forEach(p -> p.fitToWidth()));
        JButton btnCrop = createIconButton("✂", "Zuschneiden", e -> imagePanels.forEach(p -> p.cropSelection()));
        JButton btnReset = createIconButton("↺", "Bild zurücksetzen", e -> resetImages());
        JButton btnPreprocess = createIconButton("✨", "Bild entzerren (KI)", e -> runPreprocessing());

        editRow.add(btnZoomIn);
        editRow.add(btnZoomOut);
        editRow.add(btnFit);
        editRow.add(btnFitWidth);
        editRow.add(new javax.swing.JSeparator(javax.swing.SwingConstants.VERTICAL));
        editRow.add(btnCrop);
        editRow.add(btnReset);
        editRow.add(new javax.swing.JSeparator(javax.swing.SwingConstants.VERTICAL));
        editRow.add(btnPreprocess);

        return editRow;
    }

    protected JButton createIconButton(String text, String toolTip, java.awt.event.ActionListener listener) {
        JButton btn = new JButton(text);
        btn.setToolTipText(toolTip);
        btn.setFont(btn.getFont().deriveFont(btn.getFont().getSize() + 4f));
        btn.addActionListener(listener);
        return btn;
    }

    protected void setupTextEventListeners(JTextArea textArea) {
        textArea.addMouseWheelListener(e -> {
            if (e.isControlDown()) {
                adjustFontSize((e.getWheelRotation() < 0) ? 2f : -2f);
            } else {
                if (textArea.getParent() != null) {
                    textArea.getParent().dispatchEvent(e);
                }
            }
        });

        textArea.addKeyListener(new java.awt.event.KeyAdapter() {
            @Override
            public void keyPressed(java.awt.event.KeyEvent e) {
                if (e.isControlDown()) {
                    if (e.getKeyCode() == java.awt.event.KeyEvent.VK_PLUS
                            || e.getKeyCode() == java.awt.event.KeyEvent.VK_ADD) {
                        adjustFontSize(2f);
                    } else if (e.getKeyCode() == java.awt.event.KeyEvent.VK_MINUS
                            || e.getKeyCode() == java.awt.event.KeyEvent.VK_SUBTRACT) {
                        adjustFontSize(-2f);
                    }
                }
            }
        });
    }

    public void adjustFontSize(float delta) {
        float size = txtResult.getFont().getSize2D();
        size = Math.max(8f, size + delta);
        txtResult.setFont(txtResult.getFont().deriveFont(size));
    }

    protected void loadImage(File file) {
        this.currentFile = file;
        this.currentRawData = null;
        try {
            List<BufferedImage> images;
            if (file.getName().toLowerCase().endsWith(".pdf")) {
                images = DocumentTextExtractor.renderPdfToImages(file);
            } else {
                BufferedImage img = ImageIO.read(file);
                images = new ArrayList<>();
                if (img != null)
                    images.add(img);
            }
            displayImages(images);
        } catch (Exception e) {
            logger.error("Failed to load image/PDF", e);
            JOptionPane.showMessageDialog(this, "Fehler beim Laden: " + e.getMessage(), "Fehler",
                    JOptionPane.ERROR_MESSAGE);
        }
    }

    protected void loadRawData(byte[] data, String fileNameHint) {
        this.currentRawData = data;
        this.currentFile = null;
        try {
            List<BufferedImage> images = new ArrayList<>();
            boolean isPdf = data.length > 4 && data[0] == '%' && data[1] == 'P' && data[2] == 'D' && data[3] == 'F';

            if (isPdf) {
                File tempPdf = File.createTempFile("flaicheck_view_", ".pdf");
                java.nio.file.Files.write(tempPdf.toPath(), data);
                try {
                    images = DocumentTextExtractor.renderPdfToImages(tempPdf);
                } finally {
                    tempPdf.delete();
                }
            } else {
                BufferedImage img = ImageIO.read(new ByteArrayInputStream(data));
                if (img != null)
                    images.add(img);
            }
            displayImages(images);
        } catch (Exception e) {
            logger.error("Failed to load raw image data", e);
        }
    }

    private void displayImages(List<BufferedImage> images) {
        pagesPanel.removeAll();
        imagePanels.clear();
        for (BufferedImage img : images) {
            ZoomableImagePanel p = new ZoomableImagePanel();
            p.setImage(img);
            p.setBorder(BorderFactory.createMatteBorder(0, 0, 5, 0, Color.GRAY));
            imagePanels.add(p);
            pagesPanel.add(p);
        }
        SwingUtilities.invokeLater(() -> {
            imagePanels.forEach(ZoomableImagePanel::fitToWidth);
            pagesPanel.revalidate();
            pagesPanel.repaint();
        });
    }

    protected void resetImages() {
        if (currentFile != null)
            loadImage(currentFile);
        else if (currentRawData != null)
            loadRawData(currentRawData, "reset");
    }

    protected void runPreprocessing() {
        if (imagePanels.isEmpty())
            return;
        new Thread(() -> {
            for (ZoomableImagePanel p : imagePanels) {
                BufferedImage imgToProcess = p.getImage();
                File tempFile = null;
                try {
                    tempFile = File.createTempFile("flaicheck_preprocess_", ".png");
                    ImageIO.write(imgToProcess, "png", tempFile);
                    AIEngineClient client = new AIEngineClient();
                    byte[] imageBytes = client.preprocessImage(tempFile).get();
                    if (imageBytes != null && imageBytes.length > 0) {
                        BufferedImage resultImg = ImageIO.read(new ByteArrayInputStream(imageBytes));
                        if (resultImg != null)
                            SwingUtilities.invokeLater(() -> p.updateImage(resultImg));
                    }
                } catch (Exception ex) {
                    logger.error("Preprocessing failed", ex);
                } finally {
                    if (tempFile != null)
                        tempFile.delete();
                }
            }
        }).start();
    }

    public java.util.concurrent.CompletableFuture<Void> runLocalRecognition(String langCode) {
        if (imagePanels.isEmpty())
            return java.util.concurrent.CompletableFuture.completedFuture(null);

        // Clear previous results and debug segments
        txtResult.setText("");
        segmentsPanel.removeAll();
        segmentsPanel.revalidate();
        segmentsPanel.repaint();

        logger.info("Starting local recognition for {} images with language '{}'", imagePanels.size(), langCode);

        return java.util.concurrent.CompletableFuture.runAsync(() -> {
            try {
                AIEngineClient client = new AIEngineClient();

                for (int i = 0; i < imagePanels.size(); i++) {
                    ZoomableImagePanel panel = imagePanels.get(i);
                    BufferedImage imgToProcess = panel.getImage();

                    if (imgToProcess == null) {
                        logger.warn("Image panel {} has no image, skipping.", i);
                        continue;
                    }

                    logger.info("Processing page {}/{} ({}x{})", i + 1, imagePanels.size(), imgToProcess.getWidth(),
                            imgToProcess.getHeight());

                    // Process each page sequentially
                    client.recognizeHandwritingStreaming(imgToProcess, langCode, true,
                            (page, index, total, text, bbox, base64Image, rejected, reason) -> {
                                SwingUtilities.invokeLater(() -> {
                                    if (!rejected) {
                                        txtResult.append(text + "\n");
                                    }
                                    panel.setHighlight(bbox);

                                    // Add to debug segments panel
                                    if (base64Image != null) {
                                        try {
                                            byte[] bytes = Base64.getDecoder().decode(base64Image);
                                            BufferedImage segmentImg = ImageIO.read(new ByteArrayInputStream(bytes));
                                            if (segmentImg != null) {
                                                JPanel segmentBox = new JPanel(new BorderLayout(5, 5));
                                                segmentBox.setBorder(BorderFactory.createCompoundBorder(
                                                        BorderFactory.createLineBorder(
                                                                rejected ? Color.RED : Color.LIGHT_GRAY),
                                                        BorderFactory.createEmptyBorder(5, 5, 5, 5)));

                                                if (rejected) {
                                                    segmentBox.setBackground(new Color(255, 230, 230));
                                                }

                                                // Image (limit size in list)
                                                int dispW = segmentImg.getWidth() / 2;
                                                int dispH = segmentImg.getHeight() / 2;
                                                if (dispW < 50)
                                                    dispW = 50;

                                                JLabel imgLabel = new JLabel(new javax.swing.ImageIcon(
                                                        segmentImg.getScaledInstance(dispW, dispH,
                                                                BufferedImage.SCALE_SMOOTH)));

                                                // Interaction: Highlight on hover and enlarge on click
                                                imgLabel.addMouseListener(new java.awt.event.MouseAdapter() {
                                                    @Override
                                                    public void mouseClicked(java.awt.event.MouseEvent e) {
                                                        // Enlarged popup (3x)
                                                        int scale = 3;
                                                        BufferedImage enlarged = new BufferedImage(
                                                                segmentImg.getWidth() * scale,
                                                                segmentImg.getHeight() * scale,
                                                                BufferedImage.TYPE_INT_ARGB);
                                                        java.awt.Graphics2D g = enlarged.createGraphics();
                                                        g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                                                                java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                                                        g.drawImage(segmentImg, 0, 0, segmentImg.getWidth() * scale,
                                                                segmentImg.getHeight() * scale, null);
                                                        g.dispose();

                                                        JPanel popupPanel = new JPanel(new BorderLayout(10, 10));
                                                        popupPanel.add(new JLabel(new javax.swing.ImageIcon(enlarged)),
                                                                BorderLayout.CENTER);

                                                        String displayText = rejected ? "[VERWORFEN: " + reason + "]"
                                                                : text;
                                                        JLabel textLabel = new JLabel(displayText);
                                                        textLabel.setFont(new java.awt.Font("Monospaced",
                                                                java.awt.Font.BOLD, 18));
                                                        if (rejected)
                                                            textLabel.setForeground(Color.RED);
                                                        popupPanel.add(textLabel, BorderLayout.SOUTH);

                                                        JOptionPane.showMessageDialog(DocumentProcessorPanel.this,
                                                                popupPanel, "Segment-Details",
                                                                JOptionPane.PLAIN_MESSAGE);
                                                    }

                                                    @Override
                                                    public void mouseEntered(java.awt.event.MouseEvent e) {
                                                        panel.setHighlight(bbox);
                                                        segmentBox.setBackground(rejected ? new Color(255, 200, 200)
                                                                : new Color(230, 240, 255));
                                                        imgLabel.setCursor(
                                                                new java.awt.Cursor(java.awt.Cursor.HAND_CURSOR));
                                                    }

                                                    @Override
                                                    public void mouseExited(java.awt.event.MouseEvent e) {
                                                        segmentBox.setBackground(
                                                                rejected ? new Color(255, 230, 230) : null);
                                                    }
                                                });
                                                segmentBox.add(imgLabel, BorderLayout.CENTER);

                                                String labelText = rejected
                                                        ? "<html><font color='red'><b>REJECTED:</b> " + reason
                                                                + "</font></html>"
                                                        : text;
                                                JLabel lblText = new JLabel(labelText);
                                                segmentBox.add(lblText, BorderLayout.SOUTH);

                                                segmentsPanel.add(segmentBox);
                                                segmentsPanel.revalidate();

                                                // Auto-scroll to latest segment
                                                SwingUtilities.invokeLater(() -> segmentBox
                                                        .scrollRectToVisible(new java.awt.Rectangle(0, 0, 1, 1)));
                                            }
                                        } catch (Exception ex) {
                                            logger.warn("Failed to decode segment image", ex);
                                        }
                                    }
                                });
                            }).get();

                    // Update full text from final response - REMOVED: Redundant and overwrites line
                    // breaks
                }

                SwingUtilities.invokeLater(() -> {
                    imagePanels.forEach(p -> p.setHighlight(null));
                });

            } catch (Exception e) {
                logger.error("Local OCR failed", e);
                SwingUtilities.invokeLater(() -> {
                    imagePanels.forEach(p -> p.setHighlight(null));
                    ExceptionMessage.show(this, "Fehler", "Lokale KI Fehler", e);
                });
            }
        });
    }

    protected java.util.concurrent.CompletableFuture<Void> runCloudRecognition() {
        if (imagePanels.isEmpty())
            return java.util.concurrent.CompletableFuture.completedFuture(null);
        com.google.auth.Credentials credentials = de.in.flaicheckbot.MainApp.getCredentials();
        if (credentials == null) {
            JOptionPane.showMessageDialog(this, "Bitte zuerst über das Menü 'Account' bei Google anmelden!",
                    "Login erforderlich", JOptionPane.WARNING_MESSAGE);
            return java.util.concurrent.CompletableFuture.completedFuture(null);
        }

        return java.util.concurrent.CompletableFuture.runAsync(() -> {
            try {
                ImageAnnotatorSettings settings = ImageAnnotatorSettings.newBuilder()
                        .setCredentialsProvider(FixedCredentialsProvider.create(credentials)).build();
                try (ImageAnnotatorClient client = ImageAnnotatorClient.create(settings)) {
                    StringBuilder fullText = new StringBuilder();
                    for (ZoomableImagePanel p : imagePanels) {
                        BufferedImage imgToProcess = p.getImage();
                        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                        ImageIO.write(imgToProcess, "png", baos);
                        ByteString imgBytes = ByteString.copyFrom(baos.toByteArray());

                        com.google.cloud.vision.v1.Image img = com.google.cloud.vision.v1.Image.newBuilder()
                                .setContent(imgBytes).build();
                        Feature feat = Feature.newBuilder().setType(Feature.Type.DOCUMENT_TEXT_DETECTION).build();
                        AnnotateImageRequest request = AnnotateImageRequest.newBuilder().addFeatures(feat).setImage(img)
                                .build();

                        List<AnnotateImageRequest> requests = new ArrayList<>();
                        requests.add(request);

                        BatchAnnotateImagesResponse response = client.batchAnnotateImages(requests);
                        for (AnnotateImageResponse res : response.getResponsesList()) {
                            if (res.hasError())
                                fullText.append("Error: ").append(res.getError().getMessage()).append("\n");
                            else
                                fullText.append(res.getFullTextAnnotation().getText()).append("\n");
                        }
                    }
                    SwingUtilities.invokeLater(() -> txtResult.setText(fullText.toString()));
                }
            } catch (Exception e) {
                logger.error("Cloud OCR failed", e);
                SwingUtilities.invokeLater(() -> ExceptionMessage.show(this, "Fehler", "Cloud API Fehler", e));
            }
        });
    }

    protected void setupDropTargets() {
        addFileDropTarget(pagesPanel, this::loadImage);
        addFileDropTarget(txtResult, f -> {
            try {
                String text = DocumentTextExtractor.extractText(f);
                txtResult.setText(text);
            } catch (Exception e) {
                logger.error("Text extraction from drop failed", e);
            }
        });
    }

    protected void addFileDropTarget(java.awt.Component component,
            java.util.function.Consumer<java.io.File> fileHandler) {
        new java.awt.dnd.DropTarget(component, new java.awt.dnd.DropTargetAdapter() {
            @Override
            public void drop(java.awt.dnd.DropTargetDropEvent dtde) {
                try {
                    dtde.acceptDrop(java.awt.dnd.DnDConstants.ACTION_COPY);
                    java.awt.datatransfer.Transferable transferable = dtde.getTransferable();
                    if (transferable.isDataFlavorSupported(java.awt.datatransfer.DataFlavor.javaFileListFlavor)) {
                        @SuppressWarnings("unchecked")
                        java.util.List<java.io.File> files = (java.util.List<java.io.File>) transferable
                                .getTransferData(java.awt.datatransfer.DataFlavor.javaFileListFlavor);
                        if (files != null && !files.isEmpty())
                            fileHandler.accept(files.get(0));
                    }
                } catch (Exception ex) {
                    logger.error("Drop failed", ex);
                }
            }
        });
    }
}
