package de.in.flaicheckbot.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;

import javax.imageio.ImageIO;
import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.api.gax.core.FixedCredentialsProvider;
import com.google.auth.oauth2.AccessToken;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.vision.v1.AnnotateImageRequest;
import com.google.cloud.vision.v1.AnnotateImageResponse;
import com.google.cloud.vision.v1.BatchAnnotateImagesResponse;
import com.google.cloud.vision.v1.Feature;
import com.google.cloud.vision.v1.ImageAnnotatorClient;
import com.google.cloud.vision.v1.ImageAnnotatorSettings;
import com.google.protobuf.ByteString;

import de.in.flaicheckbot.AIEngineClient;
import de.in.flaicheckbot.db.DatabaseManager;
import de.in.flaicheckbot.util.UndoHelper;
import de.in.utils.gui.ExceptionMessage;

/**
 * Three-column panel for evaluating a single student's work: 1. Image View
 * (Left) 2. OCR Text (Middle) 3. Assessment & Feedback (Right)
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class EvaluationStudentPanel extends JPanel {
	private static final Logger logger = LogManager.getLogger(EvaluationStudentPanel.class);
	private final DatabaseManager dbManager;
	private final DatabaseManager.StudentWorkInfo work;
	private final DatabaseManager.TestInfo testInfo;

	private JTextField txtStudentId; // Changed from JLabel lblStudentId to JTextField txtStudentId
	private ZoomableImagePanel imagePanel;
	private JTextArea txtRecognized;
	private JTextArea txtFeedback;
	private JButton btnMaximize;
	private boolean isMaximized = false;
	private BiConsumer<EvaluationStudentPanel, Boolean> maximizeListener;
	private JLabel lblScore;
	private JCheckBox chkEvaluated;
	private static float globalFontSize = -1f; // -1 means use default
	private static final List<EvaluationStudentPanel> activeInstances = new java.util.ArrayList<>();

	public EvaluationStudentPanel(DatabaseManager dbManager, DatabaseManager.StudentWorkInfo work,
			DatabaseManager.TestInfo testInfo) {
		this.dbManager = dbManager;
		this.work = work;
		this.testInfo = testInfo;

		setLayout(new BorderLayout(5, 5));
		setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY));

		// Initial size for the list view
		setPreferredSize(new Dimension(1000, 400));
		setMaximumSize(new Dimension(Integer.MAX_VALUE, 400));
		setMinimumSize(new Dimension(800, 300));

		// Header: Delete (Left), Student ID, Evaluations Status, Maximize (Center)
		JPanel header = new JPanel(new BorderLayout());
		header.setOpaque(false);

		JPanel leftGroup = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
		leftGroup.setOpaque(false);

		JButton btnDelete = new JButton("✖");
		btnDelete.setToolTipText("Diesen Schüler-Eintrag löschen");
		btnDelete.setForeground(Color.RED);
		btnDelete.setFocusable(false);
		btnDelete.setBorderPainted(false);
		btnDelete.setContentAreaFilled(false);
		btnDelete.setFont(btnDelete.getFont().deriveFont(btnDelete.getFont().getSize() + 4f));
		btnDelete.addActionListener(e -> deleteStudentWork());
		leftGroup.add(btnDelete);

		leftGroup.add(new JLabel("Schüler:"));
		txtStudentId = new JTextField(work.studentExternalId);
		txtStudentId.setFont(txtStudentId.getFont().deriveFont(Font.BOLD, 14f));
		txtStudentId.setBorder(null);
		txtStudentId.setOpaque(false);
		txtStudentId.setPreferredSize(new Dimension(200, 25));
		leftGroup.add(txtStudentId);

		leftGroup.add(Box.createHorizontalStrut(10));
		chkEvaluated = new JCheckBox("Bewertet");
		chkEvaluated.setSelected(work.isEvaluated);
		chkEvaluated.setOpaque(false);
		chkEvaluated.setFont(chkEvaluated.getFont().deriveFont(Font.BOLD, 14f));
		chkEvaluated.addActionListener(e -> {
			try {
				dbManager.updateStudentWorkStatus(work.id, chkEvaluated.isSelected());
				firePropertyChange("isEvaluated", !chkEvaluated.isSelected(), chkEvaluated.isSelected());
			} catch (java.sql.SQLException ex) {
				logger.error("Failed to update evaluation status", ex);
			}
		});
		leftGroup.add(chkEvaluated);

		header.add(leftGroup, BorderLayout.WEST);

		JPanel centerGroup = new JPanel(new FlowLayout(FlowLayout.CENTER, 5, 0));
		centerGroup.setOpaque(false);
		btnMaximize = new JButton("🗖");
		btnMaximize.setToolTipText("Maximieren");
		btnMaximize.setFont(btnMaximize.getFont().deriveFont(btnMaximize.getFont().getSize() + 4f));
		btnMaximize.addActionListener(e -> {
			isMaximized = !isMaximized;
			updateMaximizeButton();
			if (maximizeListener != null) {
				maximizeListener.accept(this, isMaximized);
			}
		});
		centerGroup.add(btnMaximize);
		header.add(centerGroup, BorderLayout.CENTER);

		// Spacer on the right to keep centerGroup centered
		JPanel rightSpacer = new JPanel();
		rightSpacer.setOpaque(false);
		rightSpacer.setPreferredSize(leftGroup.getPreferredSize());
		header.add(rightSpacer, BorderLayout.EAST);

		add(header, BorderLayout.NORTH);

		// Center: 3 columns
		JPanel centerPanel = new JPanel(new GridLayout(1, 3, 10, 0));

		// Column 1: Image
		JPanel col1 = new JPanel(new BorderLayout());
		col1.setBorder(BorderFactory.createTitledBorder("Scan / Foto"));
		imagePanel = new ZoomableImagePanel();
		if (work.imageData != null) {
			try {
				BufferedImage img = ImageIO.read(new ByteArrayInputStream(work.imageData));
				imagePanel.setImage(img);
				SwingUtilities.invokeLater(() -> imagePanel.fitToWidth());
			} catch (IOException e) {
				logger.error("Failed to load student image", e);
			}
		}
		col1.add(new JScrollPane(imagePanel), BorderLayout.CENTER);

		JPanel imgToolbar = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 2));
		JButton btnZoomIn = new JButton("🔍+");
		btnZoomIn.addActionListener(e -> imagePanel.adjustZoom(0.2));
		JButton btnZoomOut = new JButton("🔍-");
		btnZoomOut.addActionListener(e -> imagePanel.adjustZoom(-0.2));
		JButton btnFit = new JButton("⛶");
		btnFit.setToolTipText("Einpassen");
		btnFit.addActionListener(e -> imagePanel.fitToScreen());
		JButton btnFitWidth = new JButton("↔");
		btnFitWidth.setToolTipText("Auf Breite einpassen");
		btnFitWidth.addActionListener(e -> imagePanel.fitToWidth());
		JButton btnCrop = new JButton("✂");
		btnCrop.setToolTipText("Zuschneiden");
		btnCrop.addActionListener(e -> imagePanel.cropSelection());
		JButton btnReset = new JButton("↺");
		btnReset.setToolTipText("Zurücksetzen");
		btnReset.addActionListener(e -> resetImage());
		JButton btnPreprocess = new JButton("Entzerren");
		btnPreprocess.addActionListener(e -> runPreprocessing());

		imgToolbar.add(btnZoomIn);
		imgToolbar.add(btnZoomOut);
		imgToolbar.add(btnFit);
		imgToolbar.add(btnFitWidth);
		imgToolbar.add(btnCrop);
		imgToolbar.add(btnReset);
		imgToolbar.add(btnPreprocess);

		// Increase font size for all buttons in the toolbar
		for (java.awt.Component c : imgToolbar.getComponents()) {
			if (c instanceof JButton) {
				c.setFont(c.getFont().deriveFont(c.getFont().getSize() + 4f));
			}
		}

		col1.add(imgToolbar, BorderLayout.NORTH);

		centerPanel.add(col1);

		// Column 2: OCR Text
		JPanel col2 = new JPanel(new BorderLayout());
		col2.setBorder(BorderFactory.createTitledBorder("Erkannter Text"));
		txtRecognized = new JTextArea(work.recognizedText != null ? work.recognizedText : "");
		txtRecognized.setLineWrap(true);
		txtRecognized.setWrapStyleWord(true);
		txtRecognized.setFont(txtRecognized.getFont().deriveFont(txtRecognized.getFont().getSize2D() + 2f));
		UndoHelper.addUndoSupport(txtRecognized);
		if (globalFontSize > 0) {
			txtRecognized.setFont(txtRecognized.getFont().deriveFont(globalFontSize));
		}
		col2.add(new JScrollPane(txtRecognized), BorderLayout.CENTER);

		JPanel ocrToolbar = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 2));

		JButton btnCloud = new JButton("Cloud");
		btnCloud.setToolTipText("Cloud Erkennung (Google)");
		btnCloud.addActionListener(e -> runCloudRecognition());
		ocrToolbar.add(btnCloud);

		JButton btnLocal = new JButton("Lokal");
		btnLocal.setToolTipText("Lokale KI-Erkennung");
		btnLocal.addActionListener(e -> runLocalRecognition());
		ocrToolbar.add(btnLocal);

		// Manage Cloud Button enablement via Listener
		de.in.flaicheckbot.MainApp.addLoginListener(loggedIn -> {
			btnCloud.setEnabled(loggedIn);
			btnCloud.setToolTipText(loggedIn ? "Google Cloud Vision OCR starten"
					: "Bitte zuerst über das Menü 'Account -> Google Login' anmelden.");
		});

		JButton btnFontPlus = new JButton("T+");
		btnFontPlus.addActionListener(e -> adjustFontSize(2f));
		JButton btnFontMinus = new JButton("T-");
		btnFontMinus.addActionListener(e -> adjustFontSize(-2f));
		ocrToolbar.add(btnFontPlus);
		ocrToolbar.add(btnFontMinus);

		JButton btnSaveOcr = new JButton("💾");
		btnSaveOcr.setToolTipText("Text permanent speichern");
		btnSaveOcr.addActionListener(e -> saveOcrText());
		ocrToolbar.add(btnSaveOcr);

		col2.add(ocrToolbar, BorderLayout.NORTH);

		// Add shortcuts to txtRecognized
		txtRecognized.addMouseWheelListener(e -> {
			if (e.isControlDown()) {
				adjustFontSize((e.getWheelRotation() < 0) ? 2f : -2f);
			} else {
				// Redispatch to viewport/scrollpane
				if (txtRecognized.getParent() != null) {
					txtRecognized.getParent().dispatchEvent(e);
				}
			}
		});
		txtRecognized.addKeyListener(new java.awt.event.KeyAdapter() {
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

		centerPanel.add(col2);

		// Column 3: Assessment
		JPanel col3 = new JPanel(new BorderLayout());
		col3.setBorder(BorderFactory.createTitledBorder("Bewertung (KI-gestützt)"));

		txtFeedback = new JTextArea();
		txtFeedback.setLineWrap(true);
		txtFeedback.setWrapStyleWord(true);
		txtFeedback.setEditable(true);
		txtFeedback.setFont(txtFeedback.getFont().deriveFont(txtFeedback.getFont().getSize2D() + 2f));
		UndoHelper.addUndoSupport(txtFeedback);

		if (globalFontSize > 0) {
			txtFeedback.setFont(txtFeedback.getFont().deriveFont(globalFontSize));
		}

		lblScore = new JLabel("Punkte: --", JLabel.CENTER);
		lblScore.setFont(lblScore.getFont().deriveFont(Font.BOLD, 16f));

		// Load existing grading if available
		if (work.feedback != null && !work.feedback.isEmpty()) {
			txtFeedback.setText(work.feedback);
			lblScore.setText("Gesamtpunkte: " + work.score);
		} else if (work.gradingJson != null && !work.gradingJson.isEmpty()) {
			parseAndDisplayGrading(work.gradingJson);
		}

		col3.add(new JScrollPane(txtFeedback), BorderLayout.CENTER);

		// Add shortcuts to txtFeedback
		txtFeedback.addMouseWheelListener(e -> {
			if (e.isControlDown()) {
				adjustFontSize((e.getWheelRotation() < 0) ? 2f : -2f);
			} else {
				// Redispatch to viewport/scrollpane
				if (txtFeedback.getParent() != null) {
					txtFeedback.getParent().dispatchEvent(e);
				}
			}
		});
		txtFeedback.addKeyListener(new java.awt.event.KeyAdapter() {
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

		setupAutoSave();

		JPanel gradeActions = new JPanel(new BorderLayout());
		JPanel btnBox = new JPanel(new GridLayout(1, 2, 5, 0));

		JButton btnLocalGrade = new JButton("Lokal");
		btnLocalGrade.setToolTipText("KI führt die Bewertung lokal auf diesem Rechner durch");
		btnLocalGrade.addActionListener(e -> runLocalKiGrading());
		btnBox.add(btnLocalGrade);

		JButton btnVertexGrade = new JButton("Vertex AI");
		btnVertexGrade.setToolTipText("Cloud-Bewertung via Google Vertex AI");
		btnVertexGrade.addActionListener(e -> runVertexKiGrading());
		btnBox.add(btnVertexGrade);

		// Manage Vertex AI Button enablement via Listener
		de.in.flaicheckbot.MainApp.addLoginListener(loggedIn -> {
			btnVertexGrade.setEnabled(loggedIn);
			btnVertexGrade.setToolTipText(loggedIn ? "Cloud-Bewertung via Google Vertex AI"
					: "Bitte zuerst über das Menü 'Account -> Google Login' anmelden.");
		});

		gradeActions.add(btnBox, BorderLayout.CENTER);
		gradeActions.add(lblScore, BorderLayout.SOUTH);

		col3.add(gradeActions, BorderLayout.NORTH);

		centerPanel.add(col3);

		add(centerPanel, BorderLayout.CENTER);

		setupKeyBindings();
	}

	private void setupKeyBindings() {
		String restoreKey = "RESTORE_VIEW";
		getInputMap(WHEN_IN_FOCUSED_WINDOW).put(KeyStroke.getKeyStroke("ESCAPE"), restoreKey);
		getActionMap().put(restoreKey, new AbstractAction() {
			@Override
			public void actionPerformed(java.awt.event.ActionEvent e) {
				if (isMaximized) {
					isMaximized = false;
					updateMaximizeButton();
					if (maximizeListener != null) {
						maximizeListener.accept(EvaluationStudentPanel.this, isMaximized);
					}
				}
			}
		});
	}

	private void saveOcrText() {
		try {
			dbManager.updateRecognizedText(work.id, txtRecognized.getText());
			logger.info("Saved corrected OCR text for student work id={}", work.id);
		} catch (Exception e) {
			ExceptionMessage.show(this, "Fehler", "Speichern fehlgeschlagen", e);
		}
	}

	private void runLocalKiGrading() {
		runGradingTask(false);
	}

	private void runVertexKiGrading() {
		runGradingTask(true);
	}

	public java.util.concurrent.CompletableFuture<Void> runGradingTask(boolean useVertex) {
		return java.util.concurrent.CompletableFuture.runAsync(() -> {
			try {
				SwingUtilities.invokeLater(() -> {
					txtFeedback.setText(
							"KI bewertet gerade (" + (useVertex ? "Vertex AI" : "Lokal") + ")... bitte warten.");
					lblScore.setText("Punkte: --");
				});

				// Get tasks for this test to have context
				java.util.List<DatabaseManager.TaskInfo> tasks;
				try {
					tasks = dbManager.getTasksForTest(testInfo.id);
				} catch (java.sql.SQLException e) {
					logger.error("Failed to load tasks for grading", e);
					SwingUtilities.invokeLater(() -> {
						txtFeedback.setText("Datenbankfehler beim Laden der Aufgabenstellung.");
					});
					return;
				}

				StringBuilder testConfig = new StringBuilder();
				final int[] totalMaxPoints = { 0 };
				for (DatabaseManager.TaskInfo t : tasks) {
					testConfig.append("Aufgabe ").append(t.position).append(": [").append(t.taskText).append("], ");
					testConfig.append("Erwartung: [").append(t.referenceText).append("], ");
					testConfig.append("Max: ").append(t.maxPoints).append(" Punkte\n");
					totalMaxPoints[0] += t.maxPoints;
				}

				AIEngineClient client = new AIEngineClient();
				java.util.concurrent.CompletableFuture<String> future;
				if (useVertex) {
					String token = "";
					com.google.auth.Credentials creds = de.in.flaicheckbot.MainApp.getCredentials();
					if (creds instanceof GoogleCredentials) {
						GoogleCredentials gcreds = (GoogleCredentials) creds;
						AccessToken at = gcreds.refreshAccessToken();
						token = at.getTokenValue();
					}
					future = client.gradeStudentWorkVertexAI(testConfig.toString(), "", txtRecognized.getText(), token,
							de.in.flaicheckbot.MainApp.getProjectId());
				} else {
					future = client.gradeStudentWork(testConfig.toString(), "", txtRecognized.getText());
				}

				String response = future.get();

				logger.info("KI Grading response ({}): {}", useVertex ? "Vertex" : "Local", response);

				SwingUtilities.invokeLater(() -> {
					try {
						// Parse response to get score and feedback
						int score = -1;
						String feedback = response;
						try {
							ObjectMapper mapper = new ObjectMapper();
							JsonNode root = mapper.readTree(response);
							if (root.isObject() && "success".equals(root.path("status").asText())) {
								feedback = root.path("feedback").asText();
								score = root.path("score").asInt();
							}
						} catch (Exception e) {
							logger.warn("Could not parse AI response JSON for extraction", e);
						}

						dbManager.updateGrading(work.id, response, score, feedback);
						displayGradingWithMax(response, totalMaxPoints[0]);
					} catch (java.sql.SQLException e) {
						logger.error("Failed to update grading in DB", e);
						ExceptionMessage.show(this, "Fehler", "Bewertung konnte nicht gespeichert werden", e);
					}
				});

			} catch (Exception e) {
				logger.error("Grading failed", e);
				SwingUtilities.invokeLater(() -> {
					txtFeedback.setText("Fehler bei der Bewertung: " + e.getMessage());
					ExceptionMessage.show(this, "Fehler", "KI-Bewertung fehlgeschlagen", e);
				});
			}
		});
	}

	private void parseAndDisplayGrading(String json) {
		displayGradingWithMax(json, -1);
	}

	private void displayGradingWithMax(String json, int maxPoints) {
		if (json == null || json.isEmpty())
			return;

		isUpdating = true;
		try {
			ObjectMapper mapper = new ObjectMapper();
			JsonNode root = mapper.readTree(json);
			if (root.isObject() && "success".equals(root.path("status").asText())) {
				txtFeedback.setText(root.path("feedback").asText());
				int score = root.path("score").asInt();

				if (maxPoints > 0) {
					lblScore.setText("Gesamtpunkte: " + score + " / " + maxPoints);
				} else {
					lblScore.setText("Gesamtpunkte: " + score);
				}
			} else {
				txtFeedback.setText(json);
			}
		} catch (Exception e) {
			txtFeedback.setText(json);
		} finally {
			isUpdating = false;
		}
	}

	private boolean isUpdating = false;
	private javax.swing.Timer saveTimer;

	private void setupAutoSave() {
		javax.swing.event.DocumentListener listener = new javax.swing.event.DocumentListener() {
			public void insertUpdate(javax.swing.event.DocumentEvent e) {
				scheduleSave();
			}

			public void removeUpdate(javax.swing.event.DocumentEvent e) {
				scheduleSave();
			}

			public void changedUpdate(javax.swing.event.DocumentEvent e) {
				scheduleSave();
			}
		};
		txtRecognized.getDocument().addDocumentListener(listener);
		txtFeedback.getDocument().addDocumentListener(listener);
		txtStudentId.getDocument().addDocumentListener(listener);
	}

	private void scheduleSave() {
		if (isUpdating)
			return;
		if (saveTimer != null)
			saveTimer.restart();
		else {
			saveTimer = new javax.swing.Timer(1500, e -> {
				saveData();
				saveTimer.stop();
			});
			saveTimer.setRepeats(false);
			saveTimer.start();
		}
	}

	private void saveData() {
		try {
			dbManager.updateRecognizedText(work.id, txtRecognized.getText());
			dbManager.updateGradingManual(work.id, extractScore(), txtFeedback.getText());
			dbManager.updateStudentExternalId(work.studentId, txtStudentId.getText().trim());
			logger.debug("Auto-saved student work id={}", work.id);
		} catch (Exception e) {
			logger.error("Auto-save failed", e);
		}
	}

	private int extractScore() {
		String text = lblScore.getText();
		if (text == null || text.contains("--"))
			return -1;
		try {
			String clean = text.replace("Gesamtpunkte: ", "").replace("Punkte: ", "").trim();
			if (clean.contains("/")) {
				clean = clean.substring(0, clean.indexOf("/")).trim();
			}
			return Integer.parseInt(clean);
		} catch (Exception e) {
			return -1;
		}
	}

	private void resetImage() {
		if (work.imageData != null) {
			try {
				BufferedImage img = ImageIO.read(new ByteArrayInputStream(work.imageData));
				imagePanel.setImage(img);
			} catch (IOException e) {
				logger.error("Failed to reset image", e);
			}
		}
	}

	private void adjustFontSize(float delta) {
		float currentSize = txtRecognized.getFont().getSize2D();
		float newSize = Math.max(8f, currentSize + delta);
		globalFontSize = newSize;

		synchronized (activeInstances) {
			for (EvaluationStudentPanel panel : activeInstances) {
				panel.updateLocalFontSize(newSize);
			}
		}
	}

	private void updateLocalFontSize(float newSize) {
		Font newFont = txtRecognized.getFont().deriveFont(newSize);
		txtRecognized.setFont(newFont);
		txtFeedback.setFont(txtFeedback.getFont().deriveFont(newSize));
		revalidate();
		repaint();
	}

	@Override
	public void addNotify() {
		super.addNotify();
		synchronized (activeInstances) {
			if (!activeInstances.contains(this)) {
				activeInstances.add(this);
			}
		}
		if (globalFontSize > 0) {
			updateLocalFontSize(globalFontSize);
		}
	}

	@Override
	public void removeNotify() {
		super.removeNotify();
		synchronized (activeInstances) {
			activeInstances.remove(this);
		}
	}

	private void runPreprocessing() {
		BufferedImage img = imagePanel.getImage();
		if (img == null)
			return;
		new Thread(() -> {
			File temp = null;
			try {
				temp = File.createTempFile("eval_prep_", ".png");
				ImageIO.write(img, "png", temp);
				AIEngineClient client = new AIEngineClient();
				byte[] result = client.preprocessImage(temp).get();
				if (result != null) {
					BufferedImage resImg = ImageIO.read(new ByteArrayInputStream(result));
					SwingUtilities.invokeLater(() -> imagePanel.updateImage(resImg));
				}
			} catch (Exception e) {
				logger.error("Preprocessing failed", e);
				SwingUtilities.invokeLater(() -> ExceptionMessage.show(this, "Fehler", "Entzerrung fehlgeschlagen", e));
			} finally {
				if (temp != null)
					temp.delete();
			}
		}).start();
	}

	public java.util.concurrent.CompletableFuture<Void> runLocalRecognition() {
		BufferedImage img = imagePanel.getImage();
		if (img == null)
			return java.util.concurrent.CompletableFuture.completedFuture(null);
		return java.util.concurrent.CompletableFuture.runAsync(() -> {
			File temp = null;
			try {
				temp = File.createTempFile("eval_local_", ".png");
				ImageIO.write(img, "png", temp);
				AIEngineClient client = new AIEngineClient();
				String response = client.recognizeHandwriting(temp).get();
				ObjectMapper mapper = new ObjectMapper();
				JsonNode root = mapper.readTree(response);
				if ("success".equals(root.path("status").asText())) {
					String text = root.path("text").asText();
					SwingUtilities.invokeLater(() -> txtRecognized.setText(text));
				}
			} catch (Exception e) {
				logger.error("Local recognition failed", e);
				SwingUtilities
						.invokeLater(() -> ExceptionMessage.show(this, "Fehler", "Lokal-Erkennung fehlgeschlagen", e));
			} finally {
				if (temp != null)
					temp.delete();
			}
		});
	}

	public java.util.concurrent.CompletableFuture<Void> runCloudRecognition() {
		BufferedImage img = imagePanel.getImage();
		com.google.auth.Credentials credentials = de.in.flaicheckbot.MainApp.getCredentials();
		if (img == null || credentials == null)
			return java.util.concurrent.CompletableFuture.completedFuture(null);
		return java.util.concurrent.CompletableFuture.runAsync(() -> {
			try {
				ImageAnnotatorSettings settings = ImageAnnotatorSettings.newBuilder()
						.setCredentialsProvider(FixedCredentialsProvider.create(credentials)).build();
				try (ImageAnnotatorClient client = ImageAnnotatorClient.create(settings)) {
					java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
					ImageIO.write(img, "png", baos);
					ByteString imgBytes = ByteString.copyFrom(baos.toByteArray());
					com.google.cloud.vision.v1.Image visionImg = com.google.cloud.vision.v1.Image.newBuilder()
							.setContent(imgBytes).build();
					Feature feat = Feature.newBuilder().setType(Feature.Type.DOCUMENT_TEXT_DETECTION).build();
					AnnotateImageRequest request = AnnotateImageRequest.newBuilder().addFeatures(feat)
							.setImage(visionImg).build();
					List<AnnotateImageRequest> requests = new ArrayList<>();
					requests.add(request);
					BatchAnnotateImagesResponse response = client.batchAnnotateImages(requests);
					StringBuilder sb = new StringBuilder();
					for (AnnotateImageResponse res : response.getResponsesList()) {
						if (res.hasError())
							sb.append("Error: ").append(res.getError().getMessage()).append("\n");
						else
							sb.append(res.getFullTextAnnotation().getText());
					}
					SwingUtilities.invokeLater(() -> txtRecognized.setText(sb.toString()));
				}
			} catch (Exception e) {
				logger.error("Cloud recognition failed", e);
				SwingUtilities
						.invokeLater(() -> ExceptionMessage.show(this, "Fehler", "Cloud-Erkennung fehlgeschlagen", e));
			}
		});
	}

	private void deleteStudentWork() {
		int confirm = JOptionPane.showConfirmDialog(this,
				"Möchten Sie den Eintrag für Schüler '" + work.studentExternalId + "' wirklich löschen?",
				"Eintrag löschen",
				JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);

		if (confirm == JOptionPane.YES_OPTION) {
			try {
				dbManager.deleteStudentWork(work.id);
				// Remove from UI
				JPanel parent = (JPanel) getParent();
				if (parent != null) {
					parent.remove(this);
					// Also remove the strut if possible, but let's just revalidate
					parent.revalidate();
					parent.repaint();
				}
				logger.info("Deleted student work id={}", work.id);
			} catch (Exception e) {
				logger.error("Failed to delete student work", e);
				ExceptionMessage.show(this, "Fehler", "Löschen fehlgeschlagen", e);
			}
		}
	}

	public void setMaximizeListener(BiConsumer<EvaluationStudentPanel, Boolean> listener) {
		this.maximizeListener = listener;
	}

	public void setMaximized(boolean maximized) {
		this.isMaximized = maximized;
		if (maximized) {
			setMaximumSize(new Dimension(Integer.MAX_VALUE, Integer.MAX_VALUE));
			setPreferredSize(null); // Allow it to fill the card
		} else {
			setMaximumSize(new Dimension(Integer.MAX_VALUE, 400));
			setPreferredSize(new Dimension(1000, 400));
		}
		updateMaximizeButton();
	}

	public String getStudentName() {
		return txtStudentId.getText().trim();
	}

	public String getFeedback() {
		return txtFeedback.getText();
	}

	public String getScore() {
		return lblScore.getText().replace("Gesamtpunkte: ", "").replace("Punkte: ", "").trim();
	}

	public boolean isEvaluated() {
		return chkEvaluated.isSelected();
	}

	private void updateMaximizeButton() {
		if (isMaximized) {
			btnMaximize.setText("🗗");
			btnMaximize.setToolTipText("Wiederherstellen");
		} else {
			btnMaximize.setText("🗖");
			btnMaximize.setToolTipText("Maximieren");
		}
	}
}
