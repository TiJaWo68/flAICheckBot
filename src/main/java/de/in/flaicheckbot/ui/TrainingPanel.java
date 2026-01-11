package de.in.flaicheckbot.ui;

import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.in.flaicheckbot.db.DatabaseManager;
import de.in.flaicheckbot.util.LanguageSelectionProvider;
import de.in.utils.gui.ExceptionMessage;

/**
 * Main UI panel for AI training interactions, including OCR recognition,
 * image preprocessing, and data archiving.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class TrainingPanel extends DocumentProcessorPanel {
	private JLabel lblImageStatus;
	private javax.swing.JComboBox<String> comboLanguage;

	public TrainingPanel(DatabaseManager dbManager) {
		super(dbManager);
		initBaseUI();
		addTrainingActionButtons();
	}

	protected void addTrainingActionButtons() {
		// Bottom: Action Buttons
		JPanel actionPanel = new JPanel();
		actionPanel.setLayout(new javax.swing.BoxLayout(actionPanel, javax.swing.BoxLayout.LINE_AXIS));

		JButton btnAiReset = new JButton("KI Reset");
		btnAiReset.setToolTipText("Setzt das KI-Training zurück (Löscht gelernte Daten)");
		btnAiReset.addActionListener(e -> resetAiTraining());
		actionPanel.add(btnAiReset);

		actionPanel.add(javax.swing.Box.createHorizontalGlue());

		JButton btnTrain = new JButton("KI-Training");
		btnTrain.setToolTipText("KI-Training (Fine-Tuning) starten");
		btnTrain.addActionListener(e -> runAiTraining());
		actionPanel.add(btnTrain);

		JButton btnSave = new JButton("Als Trainingsdaten speichern");
		btnSave.addActionListener(e -> saveToDatabase());
		actionPanel.add(btnSave);
		add(actionPanel, java.awt.BorderLayout.SOUTH);
	}

	@Override
	protected JPanel createImageToolbar() {
		JPanel leftToolbar = new JPanel();
		leftToolbar.setLayout(new javax.swing.BoxLayout(leftToolbar, javax.swing.BoxLayout.Y_AXIS));

		JPanel imgRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 2));
		JButton btnSelectImage = new JButton("Bild auswählen...");
		btnSelectImage.addActionListener(e -> selectImage());
		imgRow.add(btnSelectImage);

		lblImageStatus = new JLabel("Kein Bild ausgewählt");
		imgRow.add(lblImageStatus);

		imgRow.add(Box.createHorizontalStrut(20));
		imgRow.add(new JLabel("Sprache:"));
		comboLanguage = new javax.swing.JComboBox<>(LanguageSelectionProvider.getDisplayNames());
		imgRow.add(comboLanguage);

		leftToolbar.add(imgRow);
		leftToolbar.add(createStandardImageControls());

		return leftToolbar;
	}

	@Override
	protected JPanel createOcrToolbar() {
		JPanel resultToolbar = new JPanel();
		resultToolbar.setLayout(new BoxLayout(resultToolbar, BoxLayout.Y_AXIS));

		JPanel row1 = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 2));

		JButton btnRecognizeCloud = new JButton("Cloud Erkennung (Google)");
		btnRecognizeCloud.addActionListener(e -> runCloudRecognition());
		row1.add(btnRecognizeCloud);

		JButton btnRecognizeLocal = new JButton("lokale Erkennung");
		btnRecognizeLocal.addActionListener(e -> {
			String langCode = LanguageSelectionProvider.mapToIsoCode((String) comboLanguage.getSelectedItem());
			runLocalRecognition(langCode);
		});
		row1.add(btnRecognizeLocal);

		// Manage Cloud Button enablement via Listener
		de.in.flaicheckbot.MainApp.addAuthListener(status -> {
			btnRecognizeCloud.setEnabled(status.oauthLoggedIn);
			btnRecognizeCloud.setToolTipText(status.oauthLoggedIn ? "Google Cloud Vision OCR starten"
					: "Bitte zuerst über das Menü 'Account -> Google Login' anmelden.");
		});

		JPanel row2 = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 2));
		row2.add(new JLabel("Schrift:"));
		JButton btnFontPlus = createIconButton("+", "Größer", e -> adjustFontSize(2f));
		JButton btnFontMinus = createIconButton("-", "Kleiner", e -> adjustFontSize(-2f));
		row2.add(btnFontPlus);
		row2.add(btnFontMinus);

		resultToolbar.add(row1);
		resultToolbar.add(row2);
		return resultToolbar;
	}

	private void runAiTraining() {
		String language = (String) comboLanguage.getSelectedItem();
		int confirm = JOptionPane.showConfirmDialog(this,
				"Möchten Sie das Fine-Tuning der KI für Sprache '" + language
						+ "' mit allen archivierten Daten starten?",
				"KI Training", JOptionPane.YES_NO_OPTION);

		if (confirm == JOptionPane.YES_OPTION) {
			new Thread(() -> {
				try {
					String langCode = LanguageSelectionProvider.mapToIsoCode(language);
					de.in.flaicheckbot.ai.TrainingManager trainingManager = new de.in.flaicheckbot.ai.TrainingManager();
					String response = trainingManager.startTraining(langCode).get();

					SwingUtilities.invokeLater(() -> {
						try {
							ObjectMapper mapper = new ObjectMapper();
							JsonNode root = mapper.readTree(response);
							String message = root.path("message").asText(response);
							JOptionPane.showMessageDialog(this, message, "Training beendet",
									JOptionPane.INFORMATION_MESSAGE);
						} catch (Exception ex) {
							JOptionPane.showMessageDialog(this, response, "Training beendet",
									JOptionPane.INFORMATION_MESSAGE);
						}
					});
				} catch (Exception e) {
					logger.error("AI Training failed", e);
					SwingUtilities.invokeLater(() -> ExceptionMessage.show(this, "Fehler", "Fehler beim Training", e));
				}
			}).start();
		}
	}

	private void resetAiTraining() {
		int confirm = JOptionPane.showConfirmDialog(this,
				"Möchten Sie den gelernten Zustand der lokalen KI wirklich zurücksetzen?",
				"KI Reset", JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);

		if (confirm == JOptionPane.YES_OPTION) {
			new Thread(() -> {
				try {
					de.in.flaicheckbot.AIEngineClient client = new de.in.flaicheckbot.AIEngineClient();
					String response = client.resetTraining(null).get();
					SwingUtilities.invokeLater(() -> JOptionPane.showMessageDialog(this, response, "KI Reset",
							JOptionPane.INFORMATION_MESSAGE));
				} catch (Exception e) {
					logger.error("AI Reset failed", e);
					SwingUtilities
							.invokeLater(() -> ExceptionMessage.show(this, "Fehler", "Fehler beim Zurücksetzen", e));
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
			File file = chooser.getSelectedFile();
			loadImage(file);
			lblImageStatus.setText(file.getName());
			try {
				dbManager.setSetting("last_import_directory", file.getParent());
			} catch (Exception e) {
				logger.warn("Failed to save last import directory: {}", e.getMessage());
			}
		}
	}

	private void saveToDatabase() {
		if (currentFile == null && currentRawData == null || txtResult.getText().trim().isEmpty()
				|| imagePanels.isEmpty()) {
			return;
		}
		String language = (String) comboLanguage.getSelectedItem();
		String langCode = LanguageSelectionProvider.mapToIsoCode(language);
		String text = txtResult.getText();
		String fileName = currentFile != null ? currentFile.getName() : "unbenannt";

		try {
			int setId = dbManager.createTrainingSet(fileName, text, langCode);
			for (ZoomableImagePanel p : imagePanels) {
				BufferedImage usedImage = p.getImage();
				java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
				ImageIO.write(usedImage, "png", baos);
				dbManager.addTrainingSample(setId, baos.toByteArray(), "image/png", text);
			}
			JOptionPane.showMessageDialog(this, "Gespeichert unter Set ID " + setId);
		} catch (Exception e) {
			logger.error("Save failed", e);
			ExceptionMessage.show(this, "Fehler", "Speichern fehlgeschlagen", e);
		}
	}
}
