package de.in.flaicheckbot.ui;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.api.gax.core.FixedCredentialsProvider;
import com.google.cloud.vision.v1.AnnotateImageRequest;
import com.google.cloud.vision.v1.AnnotateImageResponse;
import com.google.cloud.vision.v1.BatchAnnotateImagesResponse;
import com.google.cloud.vision.v1.Feature;
import com.google.cloud.vision.v1.ImageAnnotatorClient;
import com.google.cloud.vision.v1.ImageAnnotatorSettings;
import com.google.protobuf.ByteString;

import de.in.flaicheckbot.db.DatabaseManager;
import de.in.flaicheckbot.util.DocumentTextExtractor;
import de.in.flaicheckbot.util.UndoHelper;
import de.in.utils.gui.ExceptionMessage;
import de.in.utils.gui.WrapLayout;

/**
 * Main UI panel for AI training interactions, including OCR recognition,
 * image preprocessing, and data archiving.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class TrainingPanel extends JPanel {
	private static final Logger logger = LogManager.getLogger(TrainingPanel.class);
	private final DatabaseManager dbManager;

	private JLabel lblImageStatus;
	private JTextArea txtResult;
	private File selectedImageFile;
	private ZoomableImagePanel imagePanel;
	private javax.swing.JComboBox<String> comboLanguage;

	public TrainingPanel(DatabaseManager dbManager) {
		this.dbManager = dbManager;
		setLayout(new BorderLayout(10, 10));
		setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

		// Center: Preview & Result with SplitPane
		JSplitPane centerSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
		centerSplitPane.setResizeWeight(0.5);

		// Preview (with Toolbar)
		JPanel leftPanel = new JPanel(new BorderLayout());
		leftPanel.setBorder(BorderFactory.createTitledBorder("Vorschau & Bearbeitung"));

		JPanel leftToolbar = new JPanel();
		leftToolbar.setLayout(new javax.swing.BoxLayout(leftToolbar, javax.swing.BoxLayout.Y_AXIS));

		JPanel imgRow = new JPanel(new WrapLayout(FlowLayout.LEFT, 5, 2));
		JButton btnSelectImage = new JButton("Bild auswählen...");
		btnSelectImage.addActionListener(e -> selectImage());
		imgRow.add(btnSelectImage);

		lblImageStatus = new JLabel("Kein Bild ausgewählt");
		imgRow.add(lblImageStatus);

		imgRow.add(Box.createHorizontalStrut(20));
		imgRow.add(new JLabel("Sprache:"));
		comboLanguage = new javax.swing.JComboBox<>(new String[] { "Deutsch", "Englisch", "Französisch", "Spanisch" });
		imgRow.add(comboLanguage);

		JPanel editRow = new JPanel(new WrapLayout(FlowLayout.LEFT, 5, 2));
		editRow.add(new JLabel("Vorschau-Optionen:"));
		JButton btnZoomIn = new JButton("🔍+");
		btnZoomIn.setToolTipText("Zoom In");
		btnZoomIn.addActionListener(e -> imagePanel.adjustZoom(0.2));
		JButton btnZoomOut = new JButton("🔍-");
		btnZoomOut.setToolTipText("Zoom Out");
		btnZoomOut.addActionListener(e -> imagePanel.adjustZoom(-0.2));
		JButton btnFit = new JButton("⛶");
		btnFit.setToolTipText("Einpassen");
		btnFit.addActionListener(e -> imagePanel.fitToScreen());
		JButton btnFitWidth = new JButton("↔");
		btnFitWidth.setToolTipText("Auf Breite einpassen");
		btnFitWidth.addActionListener(e -> imagePanel.fitToWidth());

		JButton btnCrop = new JButton("✂");
		btnCrop.setToolTipText("Auswahl zuschneiden");
		btnCrop.addActionListener(e -> imagePanel.cropSelection());

		JButton btnReset = new JButton("↺");
		btnReset.setToolTipText("Bild zurücksetzen");
		btnReset.addActionListener(e -> imagePanel.reset());

		JButton btnPreprocess = new JButton("Bild entzerren");
		btnPreprocess.setToolTipText("Perspektive / Neigung korrigieren (KI)");
		btnPreprocess.addActionListener(e -> runPreprocessing());

		editRow.add(btnZoomIn);
		editRow.add(btnZoomOut);
		editRow.add(btnFit);
		editRow.add(btnFitWidth);
		editRow.add(new javax.swing.JSeparator(javax.swing.SwingConstants.VERTICAL));
		editRow.add(btnCrop);
		editRow.add(btnReset);
		editRow.add(new javax.swing.JSeparator(javax.swing.SwingConstants.VERTICAL));
		editRow.add(btnPreprocess);

		// Increase font size for icon buttons
		for (java.awt.Component c : editRow.getComponents()) {
			if (c instanceof JButton) {
				JButton b = (JButton) c;
				if (b.getText().length() <= 3 || b.getText().contains("🔍") || b.getText().contains("⛶")
						|| b.getText().contains("↔") || b.getText().contains("✂") || b.getText().contains("↺")) {
					b.setFont(b.getFont().deriveFont(b.getFont().getSize() + 4f));
				}
			}
		}

		leftToolbar.add(imgRow);
		leftToolbar.add(editRow);

		leftPanel.add(leftToolbar, BorderLayout.NORTH);

		imagePanel = new ZoomableImagePanel();
		JScrollPane scrollPane = new JScrollPane(imagePanel);
		leftPanel.add(scrollPane, BorderLayout.CENTER);
		centerSplitPane.setLeftComponent(leftPanel);

		// Result Text
		JPanel resultPanel = new JPanel(new BorderLayout());
		resultPanel.setBorder(BorderFactory.createTitledBorder("Erkannter Text"));
		txtResult = new JTextArea();
		txtResult.setFont(txtResult.getFont().deriveFont(txtResult.getFont().getSize2D() + 2f));
		UndoHelper.addUndoSupport(txtResult);

		// Result Toolbar (Top of Right Panel)
		JPanel resultToolbar = new JPanel();
		resultToolbar.setLayout(new BoxLayout(resultToolbar, BoxLayout.Y_AXIS));

		JPanel row1 = new JPanel(new WrapLayout(FlowLayout.LEFT, 5, 2));

		JButton btnRecognizeCloud = new JButton("Cloud Erkennung (Google)");
		btnRecognizeCloud.addActionListener(e -> runRecognition());
		row1.add(btnRecognizeCloud);

		JButton btnRecognizeLocal = new JButton("lokale Erkennung");
		btnRecognizeLocal.addActionListener(e -> runLocalRecognition());
		row1.add(btnRecognizeLocal);

		// Manage Cloud Button enablement via Listener
		de.in.flaicheckbot.MainApp.addAuthListener(status -> {
			btnRecognizeCloud.setEnabled(status.oauthLoggedIn);
			btnRecognizeCloud.setToolTipText(status.oauthLoggedIn ? "Google Cloud Vision OCR starten"
					: "Bitte zuerst über das Menü 'Account -> Google Login' anmelden (erfordert OAuth/client_secret.json).");
		});

		JPanel row2 = new JPanel(new WrapLayout(FlowLayout.LEFT, 5, 2));
		row2.add(new JLabel("Schrift:"));
		JButton btnFontPlus = new JButton("+");
		btnFontPlus.addActionListener(e -> adjustFontSize(2f));
		JButton btnFontMinus = new JButton("-");
		btnFontMinus.addActionListener(e -> adjustFontSize(-2f));
		row2.add(btnFontPlus);
		row2.add(btnFontMinus);

		resultToolbar.add(row1);
		resultToolbar.add(row2);
		resultPanel.add(resultToolbar, BorderLayout.NORTH);

		// Dynamic Font Resizing (Existing Listeners)
		txtResult.addMouseWheelListener(e -> {
			if (e.isControlDown()) {
				adjustFontSize((e.getWheelRotation() < 0) ? 2f : -2f);
			} else {
				e.getComponent().getParent().dispatchEvent(e);
			}
		});

		txtResult.addKeyListener(new java.awt.event.KeyAdapter() {
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

		resultPanel.add(new JScrollPane(txtResult), BorderLayout.CENTER);
		centerSplitPane.setRightComponent(resultPanel);

		add(centerSplitPane, BorderLayout.CENTER);

		setupDropTargets();

		// Bottom: Save
		JPanel actionPanel = new JPanel();
		actionPanel.setLayout(new BoxLayout(actionPanel, BoxLayout.LINE_AXIS));

		JButton btnAiReset = new JButton("KI Reset");
		btnAiReset.setToolTipText("Setzt das KI-Training zurück (Löscht gelernte Daten)");
		btnAiReset.addActionListener(e -> resetAiTraining());
		actionPanel.add(btnAiReset);

		actionPanel.add(Box.createHorizontalGlue());

		JButton btnTrain = new JButton("KI-Training");
		btnAiReset.setToolTipText("KI-Training (Fine-Tuning) starten");
		btnTrain.addActionListener(e -> runAiTraining());
		actionPanel.add(btnTrain);

		JButton btnSave = new JButton("Als Trainingsdaten speichern");
		btnSave.addActionListener(e -> saveToDatabase());
		actionPanel.add(btnSave);
		add(actionPanel, BorderLayout.SOUTH);
	}

	private void adjustFontSize(float delta) {
		float size = txtResult.getFont().getSize2D();
		size = Math.max(8f, size + delta);
		txtResult.setFont(txtResult.getFont().deriveFont(size));
	}

	private void runAiTraining() {
		String language = (String) comboLanguage.getSelectedItem();
		logger.info("User triggered AI Fine-Tuning (Language: {}).", language);
		int confirm = JOptionPane.showConfirmDialog(this,
				"Möchten Sie das Fine-Tuning der KI für Sprache '" + language
						+ "' mit allen archivierten Daten starten?\nDies kann je nach Datenmenge einige Minuten dauern.",
				"KI Training", JOptionPane.YES_NO_OPTION);

		if (confirm == JOptionPane.YES_OPTION) {
			new Thread(() -> {
				try {
					String langCode = mapLanguageCode(language);
					de.in.flaicheckbot.ai.TrainingManager trainingManager = new de.in.flaicheckbot.ai.TrainingManager();
					String response = trainingManager.startTraining(langCode).get();

					SwingUtilities.invokeLater(() -> {
						logger.info("AI Training response: {}", response);
						try {
							ObjectMapper mapper = new ObjectMapper();
							JsonNode root = mapper.readTree(response);
							String status = root.path("status").asText("Unbekannt");
							String message = root.path("message").asText(response);

							JOptionPane.showMessageDialog(this, message, "Training beendet: " + status,
									JOptionPane.INFORMATION_MESSAGE);
						} catch (Exception ex) {
							logger.warn("Could not parse AI response as JSON: {}", response);
							JOptionPane.showMessageDialog(this, response, "Training beendet",
									JOptionPane.INFORMATION_MESSAGE);
						}
					});
				} catch (Exception e) {
					logger.error("AI Training failed", e);
					SwingUtilities.invokeLater(() -> {
						ExceptionMessage.show(this, "Fehler", "Fehler beim Training", e);
					});
				}
			}).start();
		}

	}

	private void resetAiTraining() {
		logger.info("User triggered AI Training Reset.");
		int confirm = JOptionPane.showConfirmDialog(this,
				"Möchten Sie den gelernten Zustand der lokalen KI wirklich zurücksetzen?\n"
						+ "Das System kehrt zum Basis-Modell zurück. Archivierte Trainingsdaten bleiben erhalten, müssen aber neu trainiert werden.",
				"KI Reset", JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);

		if (confirm == JOptionPane.YES_OPTION) {
			new Thread(() -> {
				try {
					de.in.flaicheckbot.AIEngineClient client = new de.in.flaicheckbot.AIEngineClient();
					String response = client.resetTraining(null).get();

					SwingUtilities.invokeLater(() -> {
						logger.info("AI Reset response: {}", response);
						try {
							ObjectMapper mapper = new ObjectMapper();
							JsonNode root = mapper.readTree(response);
							String message = root.path("message").asText(response);
							JOptionPane.showMessageDialog(this, message, "KI Reset", JOptionPane.INFORMATION_MESSAGE);
						} catch (Exception ex) {
							JOptionPane.showMessageDialog(this, response, "KI Reset", JOptionPane.INFORMATION_MESSAGE);
						}
					});
				} catch (Exception e) {
					logger.error("AI Reset failed", e);
					SwingUtilities.invokeLater(() -> {
						ExceptionMessage.show(this, "Fehler", "Fehler beim Zurücksetzen", e);
					});
				}
			}).start();
		}
	}

	private void selectImage() {
		String lastDir = null;
		try {
			lastDir = dbManager.getSetting("last_import_directory");
		} catch (Exception e) {
			logger.warn("Failed to load last import directory", e);
		}

		JFileChooser chooser = (lastDir != null && new File(lastDir).exists()) ? new JFileChooser(lastDir)
				: new JFileChooser();

		chooser.setFileFilter(
				new javax.swing.filechooser.FileNameExtensionFilter("Images & Documents", "png", "jpg", "jpeg", "pdf"));
		if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
			loadImage(chooser.getSelectedFile());
		}
	}

	private void loadImage(File file) {
		selectedImageFile = file;
		logger.info("Loading image/PDF: {}", file.getAbsolutePath());
		lblImageStatus.setText(selectedImageFile.getName());

		// Save last directory so next JFileChooser opens here
		try {
			dbManager.setSetting("last_import_directory", file.getParent());
		} catch (Exception e) {
			logger.warn("Failed to save last import directory: {}", e.getMessage());
		}

		try {
			BufferedImage img;
			if (selectedImageFile.getName().toLowerCase().endsWith(".pdf")) {
				img = DocumentTextExtractor.renderPdfToImage(selectedImageFile);
			} else {
				img = ImageIO.read(selectedImageFile);
			}
			imagePanel.setImage(img);
		} catch (Exception e) {
			logger.error("Failed to load image", e);
			JOptionPane.showMessageDialog(this, "Fehler beim Laden des Bildes: " + e.getMessage(), "Fehler",
					JOptionPane.ERROR_MESSAGE);
		}
	}

	private void handleTextFile(File file) {
		try {
			String text = DocumentTextExtractor.extractText(file);
			txtResult.setText(text);
			logger.info("Loaded text from dropped file: {}", file.getName());
		} catch (IllegalArgumentException e) {
			// Probably not a text file we support, ignore or show hint
			logger.warn("Unsupported file dropped: {}", file.getName());
		} catch (Exception e) {
			logger.error("Failed to extract text from dropped file", e);
			JOptionPane.showMessageDialog(this, "Fehler beim Extrahieren des Textes: " + e.getMessage(), "Fehler",
					JOptionPane.ERROR_MESSAGE);
		}
	}

	private void runLocalRecognition() {
		BufferedImage imgToProcess = imagePanel.getImage();
		if (imgToProcess == null) {
			JOptionPane.showMessageDialog(this, "Bitte zuerst ein Bild auswählen!", "Hinweis",
					JOptionPane.WARNING_MESSAGE);
			return;
		}

		// Clear result and reset highlight
		txtResult.setText("");
		imagePanel.setHighlight(null);

		new Thread(() -> {
			try {
				String language = (String) comboLanguage.getSelectedItem();
				String langCode = mapLanguageCode(language);
				logger.info("Starting local OCR recognition (Language: {})...", language);
				de.in.flaicheckbot.AIEngineClient client = new de.in.flaicheckbot.AIEngineClient();

				String response = client.recognizeHandwritingStreaming(imgToProcess, langCode, true,
						(index, total, text, bbox) -> {
							SwingUtilities.invokeLater(() -> {
								txtResult.append(text + "\n");
								imagePanel.setHighlight(bbox);
							});
						}).get();

				ObjectMapper mapper = new ObjectMapper();
				JsonNode root = mapper.readTree(response);
				String finalResultText = root.path("text").asText("");

				logger.info("Local OCR recognition completed.");
				SwingUtilities.invokeLater(() -> {
					txtResult.setText(finalResultText); // Final result (ensure consistency)
					imagePanel.setHighlight(null); // Clear last highlight
				});
			} catch (Exception e) {
				logger.error("Local AI Recognition failed", e);
				SwingUtilities.invokeLater(() -> {
					imagePanel.setHighlight(null);
					ExceptionMessage.show(this, "Fehler", "Lokale KI Fehler", e);
				});
			}
		}).start();
	}

	private String mapLanguageCode(String language) {
		if (language == null)
			return "de";
		if ("Englisch".equals(language) || "English".equals(language) || "en".equals(language))
			return "en";
		if ("Französisch".equals(language) || "French".equals(language) || "fr".equals(language))
			return "fr";
		if ("Spanisch".equals(language) || "Spanish".equals(language) || "es".equals(language))
			return "es";
		return "de";
	}

	private void runPreprocessing() {
		BufferedImage imgToProcess = imagePanel.getImage();
		if (imgToProcess == null) {
			JOptionPane.showMessageDialog(this, "Bitte zuerst ein Bild auswählen!", "Hinweis",
					JOptionPane.WARNING_MESSAGE);
			return;
		}

		new Thread(() -> {
			File tempFile = null;
			try {
				tempFile = File.createTempFile("flaicheck_preprocess_", ".png");
				ImageIO.write(imgToProcess, "png", tempFile);

				logger.info("Starting image preprocessing (deskew/AI)...");
				de.in.flaicheckbot.AIEngineClient client = new de.in.flaicheckbot.AIEngineClient();
				byte[] imageBytes = client.preprocessImage(tempFile).get();

				if (imageBytes != null && imageBytes.length > 0) {
					logger.info("Image preprocessing completed successfully.");
					BufferedImage resultImg = ImageIO.read(new java.io.ByteArrayInputStream(imageBytes));
					if (resultImg != null) {
						SwingUtilities.invokeLater(() -> imagePanel.updateImage(resultImg));
					}
				}
			} catch (Exception ex) {
				logger.error("Image preprocessing failed", ex);
				SwingUtilities
						.invokeLater(() -> ExceptionMessage.show(this, "Fehler", "Bild-Entzerrung fehlgeschlagen", ex));
			} finally {
				if (tempFile != null && tempFile.exists()) {
					tempFile.delete();
				}
			}
		}).start();
	}

	private void runRecognition() {
		BufferedImage imgToProcess = imagePanel.getImage();
		if (imgToProcess == null) {
			JOptionPane.showMessageDialog(this, "Bitte zuerst ein Bild auswählen!", "Hinweis",
					JOptionPane.WARNING_MESSAGE);
			return;
		}
		com.google.auth.Credentials credentials = de.in.flaicheckbot.MainApp.getCredentials();
		if (credentials == null) {
			JOptionPane.showMessageDialog(this, "Bitte zuerst über das Menü 'Account' bei Google anmelden!",
					"Login erforderlich",
					JOptionPane.WARNING_MESSAGE);
			return;
		}

		new Thread(() -> {
			try {
				logger.info("Starting Google Cloud Vision recognition...");
				ImageAnnotatorSettings settings = ImageAnnotatorSettings.newBuilder()
						.setCredentialsProvider(FixedCredentialsProvider.create(credentials)).build();

				try (ImageAnnotatorClient client = ImageAnnotatorClient.create(settings)) {
					// Convert BufferedImage to ByteString
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
					List<AnnotateImageResponse> responses = response.getResponsesList();

					StringBuilder sb = new StringBuilder();
					for (AnnotateImageResponse res : responses) {
						if (res.hasError()) {
							sb.append("Error: ").append(res.getError().getMessage()).append("\n");
						} else {
							sb.append(res.getFullTextAnnotation().getText());
						}
					}

					logger.info("Google Cloud recognition completed successfully.");
					SwingUtilities.invokeLater(() -> txtResult.setText(sb.toString()));
				}
			} catch (Exception e) {
				logger.error("Google Cloud Recognition failed", e);
				SwingUtilities.invokeLater(() -> ExceptionMessage.show(this, "Fehler", "Cloud API Fehler", e));
			}
		}).start();
	}

	private void saveToDatabase() {
		if (selectedImageFile == null || txtResult.getText().trim().isEmpty()) {
			return;
		}
		String language = (String) comboLanguage.getSelectedItem();
		String langCode = mapLanguageCode(language);
		String text = txtResult.getText();
		logger.info("Saving training set: '{}' (Language: {}) with text: '{}'...", selectedImageFile.getName(),
				language,
				text.length() > 50 ? text.substring(0, 47) + "..." : text);
		try {
			int setId = dbManager.createTrainingSet(selectedImageFile.getName(), text, langCode);
			// Save the ACTUALLY used image (maybe cropped)
			BufferedImage usedImage = imagePanel.getImage();
			java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
			ImageIO.write(usedImage, "png", baos);
			byte[] imgData = baos.toByteArray();
			String mime = "image/png";

			dbManager.addTrainingSample(setId, imgData, mime, txtResult.getText());

			logger.info("Training set saved successfully. Set ID: {}", setId);
			JOptionPane.showMessageDialog(this, "Gespeichert unter Set ID " + setId);
		} catch (Exception e) {
			logger.error("Save failed", e);
			ExceptionMessage.show(this, "Fehler", "Speichern fehlgeschlagen", e);
		}
	}

	private void setupDropTargets() {
		addFileDropTarget(imagePanel, this::loadImage);
		addFileDropTarget(txtResult, this::handleTextFile);
	}

	private void addFileDropTarget(java.awt.Component component,
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
						if (files != null && !files.isEmpty()) {
							fileHandler.accept(files.get(0));
						}
					}
				} catch (Exception ex) {
					logger.error("Drop failed", ex);
				}
			}
		});
	}
}
