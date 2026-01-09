package de.in.flaicheckbot.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridLayout;
import java.util.function.BiConsumer;

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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.in.flaicheckbot.AIEngineClient;
import de.in.flaicheckbot.db.DatabaseManager;
import de.in.flaicheckbot.util.UndoHelper;
import de.in.utils.gui.ExceptionMessage;

/**
 * Three-column panel for evaluating a single student's work:
 * 1. Image View (Left) 2. OCR Text (Middle) 3. Assessment & Feedback (Right)
 */
public class EvaluationStudentPanel extends DocumentProcessorPanel {
	private final DatabaseManager.StudentWorkInfo work;
	private final DatabaseManager.TestInfo testInfo;

	private JTextField txtStudentId;
	private JTextArea txtFeedback;
	private JButton btnMaximize;
	private boolean isMaximized = false;
	private BiConsumer<EvaluationStudentPanel, Boolean> maximizeListener;
	private JLabel lblScore;
	private JCheckBox chkEvaluated;
	private String languageCode = "de";
	private static float globalFontSize = -1f;
	private static final java.util.List<EvaluationStudentPanel> activeInstances = new java.util.ArrayList<>();

	public EvaluationStudentPanel(DatabaseManager dbManager, DatabaseManager.StudentWorkInfo work,
			DatabaseManager.TestInfo testInfo) {
		super(dbManager);
		this.work = work;
		this.testInfo = testInfo;

		setLayout(new BorderLayout(5, 5));
		setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY));
		setPreferredSize(new Dimension(1000, 400));
		setMaximumSize(new Dimension(Integer.MAX_VALUE, 400));

		add(createHeader(), BorderLayout.NORTH);

		JPanel centerPanel = new JPanel(new GridLayout(1, 3, 10, 0));

		// Col 1: Images
		JPanel col1 = new JPanel(new BorderLayout());
		col1.setBorder(BorderFactory.createTitledBorder("Scan / Foto"));
		col1.add(createImageToolbar(), BorderLayout.NORTH);
		pagesPanel = new JPanel();
		pagesPanel.setLayout(new javax.swing.BoxLayout(pagesPanel, javax.swing.BoxLayout.Y_AXIS));
		col1.add(new JScrollPane(pagesPanel), BorderLayout.CENTER);
		centerPanel.add(col1);

		// Col 2: OCR Text
		JPanel col2 = new JPanel(new BorderLayout());
		col2.setBorder(BorderFactory.createTitledBorder("Erkannter Text"));
		col2.add(createOcrToolbar(), BorderLayout.NORTH);
		txtResult = new JTextArea(work.recognizedText != null ? work.recognizedText : "");
		txtResult.setLineWrap(true);
		txtResult.setWrapStyleWord(true);
		UndoHelper.addUndoSupport(txtResult);
		setupTextEventListeners(txtResult);
		col2.add(new JScrollPane(txtResult), BorderLayout.CENTER);
		centerPanel.add(col2);

		// Col 3: Assessment
		centerPanel.add(createAssessmentColumn());

		add(centerPanel, BorderLayout.CENTER);

		if (work.imageData != null) {
			loadRawData(work.imageData, "work_" + work.id);
		}

		setupAutoSave();
		setupKeyBindings();
	}

	private JPanel createHeader() {
		JPanel header = new JPanel(new BorderLayout());
		header.setOpaque(false);

		JPanel leftGroup = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
		leftGroup.setOpaque(false);

		JButton btnDelete = new JButton("✖");
		btnDelete.setForeground(Color.RED);
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

		chkEvaluated = new JCheckBox("Bewertet");
		chkEvaluated.setSelected(work.isEvaluated);
		chkEvaluated.setOpaque(false);
		chkEvaluated.setFont(chkEvaluated.getFont().deriveFont(Font.BOLD, 14f));
		chkEvaluated.addActionListener(e -> {
			try {
				dbManager.updateStudentWorkStatus(work.id, chkEvaluated.isSelected());
				firePropertyChange("isEvaluated", !chkEvaluated.isSelected(), chkEvaluated.isSelected());
			} catch (Exception ex) {
				logger.error("Status update failed", ex);
			}
		});
		leftGroup.add(chkEvaluated);
		header.add(leftGroup, BorderLayout.WEST);

		btnMaximize = new JButton("🗖");
		btnMaximize.addActionListener(e -> {
			isMaximized = !isMaximized;
			btnMaximize.setText(isMaximized ? "❐" : "🗖");
			if (maximizeListener != null)
				maximizeListener.accept(this, isMaximized);
		});
		JPanel centerGroup = new JPanel(new FlowLayout(FlowLayout.CENTER));
		centerGroup.setOpaque(false);
		centerGroup.add(btnMaximize);
		header.add(centerGroup, BorderLayout.CENTER);

		return header;
	}

	@Override
	protected JPanel createImageToolbar() {
		return createStandardImageControls();
	}

	@Override
	protected JPanel createOcrToolbar() {
		JPanel ocrToolbar = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 2));
		JButton btnCloud = createIconButton("☁", "Cloud (Google)", e -> runCloudRecognition());
		JButton btnLocal = createIconButton("⚙", "Lokal", e -> runLocalRecognition(languageCode));
		JButton btnSaveOcr = createIconButton("💾", "Speichern", e -> saveOcrText());

		ocrToolbar.add(btnCloud);
		ocrToolbar.add(btnLocal);
		ocrToolbar.add(Box.createHorizontalStrut(5));
		ocrToolbar.add(createIconButton("T+", "Größer", e -> adjustFontSize(2f)));
		ocrToolbar.add(createIconButton("T-", "Kleiner", e -> adjustFontSize(-2f)));
		ocrToolbar.add(btnSaveOcr);

		de.in.flaicheckbot.MainApp.addAuthListener(status -> {
			btnCloud.setEnabled(status.oauthLoggedIn);
		});
		return ocrToolbar;
	}

	private JPanel createAssessmentColumn() {
		JPanel col3 = new JPanel(new BorderLayout());
		col3.setBorder(BorderFactory.createTitledBorder("Bewertung (KI-gestützt)"));

		txtFeedback = new JTextArea();
		txtFeedback.setLineWrap(true);
		txtFeedback.setWrapStyleWord(true);
		UndoHelper.addUndoSupport(txtFeedback);
		setupTextEventListeners(txtFeedback);

		lblScore = new JLabel("Punkte: --", JLabel.CENTER);
		lblScore.setFont(lblScore.getFont().deriveFont(Font.BOLD, 16f));

		if (work.feedback != null && !work.feedback.isEmpty()) {
			txtFeedback.setText(work.feedback);
			lblScore.setText("Gesamtpunkte: " + work.score);
		} else if (work.gradingJson != null && !work.gradingJson.isEmpty()) {
			parseAndDisplayGrading(work.gradingJson);
		}

		col3.add(new JScrollPane(txtFeedback), BorderLayout.CENTER);

		JPanel gradeActions = new JPanel(new BorderLayout());
		JPanel btnBox = new JPanel(new GridLayout(1, 2, 5, 0));
		JButton btnLocalGrade = new JButton("Lokal");
		btnLocalGrade.addActionListener(e -> runGradingTask(false));
		btnBox.add(btnLocalGrade);

		JButton btnVertexGrade = new JButton("Vertex AI");
		btnVertexGrade.addActionListener(e -> runGradingTask(true));
		btnBox.add(btnVertexGrade);

		de.in.flaicheckbot.MainApp.addAuthListener(status -> btnVertexGrade.setEnabled(status.isAnyAvailable()));

		gradeActions.add(btnBox, BorderLayout.CENTER);
		gradeActions.add(lblScore, BorderLayout.SOUTH);
		col3.add(gradeActions, BorderLayout.NORTH);

		return col3;
	}

	private void saveOcrText() {
		try {
			dbManager.updateRecognizedText(work.id, txtResult.getText());
		} catch (Exception e) {
			ExceptionMessage.show(this, "Fehler", "Speichern fehlgeschlagen", e);
		}
	}

	public java.util.concurrent.CompletableFuture<Void> runGradingTask(boolean useVertex) {
		return java.util.concurrent.CompletableFuture.runAsync(() -> {
			try {
				SwingUtilities.invokeLater(() -> {
					txtFeedback.setText("KI bewertet gerade...");
					lblScore.setText("Punkte: --");
				});

				java.util.List<DatabaseManager.TaskInfo> tasks = dbManager.getTasksForTest(testInfo.id);
				StringBuilder testConfig = new StringBuilder();
				int totalMax = 0;
				for (DatabaseManager.TaskInfo t : tasks) {
					testConfig.append("Aufgabe ").append(t.position).append(": ").append(t.taskText).append("\n");
					totalMax += t.maxPoints;
				}

				AIEngineClient client = new AIEngineClient();
				String response;
				if (useVertex) {
					response = client.gradeStudentWorkVertexAI(testConfig.toString(), "", txtResult.getText(), null,
							de.in.flaicheckbot.MainApp.getProjectId(), null).get();
				} else {
					response = client.gradeStudentWork(testConfig.toString(), "", txtResult.getText()).get();
				}

				final int finalTotalMax = totalMax;
				SwingUtilities.invokeLater(() -> {
					try {
						ObjectMapper mapper = new ObjectMapper();
						JsonNode root = mapper.readTree(response);
						if ("success".equals(root.path("status").asText())) {
							String feedback = root.path("feedback").asText();
							int score = root.path("score").asInt();
							dbManager.updateGrading(work.id, response, score, feedback);
							displayGradingWithMax(response, finalTotalMax);
						}
					} catch (Exception e) {
						logger.error("JSON parse failed", e);
					}
				});
			} catch (Exception e) {
				logger.error("Grading failed", e);
			}
		});
	}

	private void parseAndDisplayGrading(String json) {
		displayGradingWithMax(json, -1);
	}

	private void displayGradingWithMax(String json, int maxPoints) {
		if (json == null || json.isEmpty())
			return;
		try {
			ObjectMapper mapper = new ObjectMapper();
			JsonNode root = mapper.readTree(json);
			if ("success".equals(root.path("status").asText())) {
				txtFeedback.setText(root.path("feedback").asText());
				int score = root.path("score").asInt();
				lblScore.setText("Gesamtpunkte: " + score + (maxPoints > 0 ? " / " + maxPoints : ""));
			} else {
				txtFeedback.setText(json);
			}
		} catch (Exception e) {
			txtFeedback.setText(json);
		}
	}

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
		txtResult.getDocument().addDocumentListener(listener);
		txtFeedback.getDocument().addDocumentListener(listener);
		txtStudentId.getDocument().addDocumentListener(listener);
	}

	private javax.swing.Timer saveTimer;

	private void scheduleSave() {
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
			dbManager.updateRecognizedText(work.id, txtResult.getText());
			dbManager.updateGradingManual(work.id, extractScore(), txtFeedback.getText());
			dbManager.updateStudentExternalId(work.studentId, txtStudentId.getText().trim());
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
			if (clean.contains("/"))
				clean = clean.substring(0, clean.indexOf("/")).trim();
			return Integer.parseInt(clean);
		} catch (Exception e) {
			return -1;
		}
	}

	private void setupKeyBindings() {
		getInputMap(WHEN_IN_FOCUSED_WINDOW).put(KeyStroke.getKeyStroke("ESCAPE"), "RESTORE");
		getActionMap().put("RESTORE", new AbstractAction() {
			public void actionPerformed(java.awt.event.ActionEvent e) {
				if (isMaximized) {
					isMaximized = false;
					btnMaximize.setText("🗖");
					if (maximizeListener != null)
						maximizeListener.accept(EvaluationStudentPanel.this, false);
				}
			}
		});
	}

	@Override
	public void adjustFontSize(float delta) {
		float currentSize = txtResult.getFont().getSize2D();
		float newSize = Math.max(8f, currentSize + delta);
		globalFontSize = newSize;
		synchronized (activeInstances) {
			for (EvaluationStudentPanel panel : activeInstances) {
				panel.updateLocalFontSize(newSize);
			}
		}
	}

	public void updateLocalFontSize(float newSize) {
		Font font = txtResult.getFont().deriveFont(newSize);
		txtResult.setFont(font);
		txtFeedback.setFont(font);
		revalidate();
		repaint();
	}

	@Override
	public void addNotify() {
		super.addNotify();
		synchronized (activeInstances) {
			if (!activeInstances.contains(this))
				activeInstances.add(this);
		}
		if (globalFontSize > 0)
			updateLocalFontSize(globalFontSize);
	}

	@Override
	public void removeNotify() {
		super.removeNotify();
		synchronized (activeInstances) {
			activeInstances.remove(this);
		}
	}

	public void setLanguage(String languageCode) {
		this.languageCode = languageCode;
	}

	public void setMaximizeListener(BiConsumer<EvaluationStudentPanel, Boolean> l) {
		this.maximizeListener = l;
	}

	public boolean isEvaluated() {
		return chkEvaluated.isSelected();
	}

	public String getStudentName() {
		return txtStudentId.getText();
	}

	public String getFeedback() {
		return txtFeedback.getText();
	}

	public int getScore() {
		return extractScore();
	}

	private void deleteStudentWork() {
		int confirm = JOptionPane.showConfirmDialog(this, "Soll dieser Eintrag wirklich gelöscht werden?", "Löschen",
				JOptionPane.YES_NO_OPTION);
		if (confirm == JOptionPane.YES_OPTION) {
			try {
				dbManager.deleteStudentWork(work.id);
				getParent().remove(this);
				revalidate();
				repaint();
			} catch (Exception e) {
				logger.error("Delete failed", e);
			}
		}
	}
}
