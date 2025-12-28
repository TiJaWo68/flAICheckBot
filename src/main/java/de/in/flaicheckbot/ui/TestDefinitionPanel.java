package de.in.flaicheckbot.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Frame;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.cuberact.swing.layout.Composite;

import de.in.flaicheckbot.db.DatabaseManager;
import de.in.flaicheckbot.util.DocumentTextExtractor;
import de.in.flaicheckbot.util.UndoHelper;
import de.in.utils.gui.ExceptionMessage;

/**
 * Swing UI panel for creating, editing, and managing test templates and their associated tasks.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class TestDefinitionPanel extends JPanel {
	private static final Logger logger = LogManager.getLogger(TestDefinitionPanel.class);
	private final DatabaseManager dbManager;

	private JTextField txtTitle;
	private JTextField txtGrade;
	private JTextField txtLearningUnit;

	private JTabbedPane taskTabPane;
	private List<TaskCard> taskCards = new ArrayList<>();
	private int currentTestId = -1;

	public TestDefinitionPanel(DatabaseManager dbManager) {
		this.dbManager = dbManager;
		setLayout(new BorderLayout(10, 10));
		setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

		// Top: Metadata
		JPanel metaPanel = new JPanel(new GridBagLayout());
		metaPanel.setBorder(BorderFactory.createTitledBorder("Metadaten"));
		GridBagConstraints gbc = new GridBagConstraints();
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.insets = new Insets(5, 5, 5, 5);
		gbc.weightx = 1.0;

		gbc.gridx = 0;
		gbc.gridy = 0;
		gbc.gridwidth = 2;
		gbc.weightx = 1.0;
		metaPanel.add(new JLabel("Titel:"), gbc);

		JPanel titleActionPanel = new JPanel(new BorderLayout(5, 0));
		txtTitle = new JTextField();
		titleActionPanel.add(txtTitle, BorderLayout.CENTER);

		JButton btnLoadTest = new JButton("🗄️");
		btnLoadTest.setToolTipText("Gespeicherten Test aus Datenbank laden");
		btnLoadTest.addActionListener(e -> loadTestFromDb());
		titleActionPanel.add(btnLoadTest, BorderLayout.EAST);

		gbc.gridy = 1;
		metaPanel.add(titleActionPanel, gbc);

		gbc.gridy = 2;
		gbc.gridwidth = 1;
		gbc.weightx = 0.5;
		metaPanel.add(new JLabel("Klassenstufe:"), gbc);

		gbc.gridx = 1;
		metaPanel.add(new JLabel("Lernabschnitt:"), gbc);

		txtGrade = new JTextField();
		gbc.gridx = 0;
		gbc.gridy = 3;
		metaPanel.add(txtGrade, gbc);

		txtLearningUnit = new JTextField();
		gbc.gridx = 1;
		metaPanel.add(txtLearningUnit, gbc);

		add(metaPanel, BorderLayout.NORTH);

		// Center: Tasks List
		taskTabPane = new JTabbedPane();
		JPanel taskContainerPanel = new JPanel(new BorderLayout());
		taskContainerPanel.setBorder(BorderFactory.createTitledBorder("Aufgaben"));
		taskContainerPanel.add(taskTabPane, BorderLayout.CENTER);
		add(taskContainerPanel, BorderLayout.CENTER);

		// Bottom: Actions
		JPanel bottomActionPanel = new JPanel(new BorderLayout());
		bottomActionPanel.setBorder(BorderFactory.createEmptyBorder(5, 0, 0, 0));

		JButton btnAddTask = new JButton("Neue Aufgabe hinzufügen");
		btnAddTask.addActionListener(e -> addTask(null));

		JButton btnNewTest = new JButton("Neuen Test erstellen");
		btnNewTest.addActionListener(e -> {
			if (!txtTitle.getText().trim().isEmpty() || taskCards.size() > 1 || !taskCards.get(0).getTaskText().trim().isEmpty()) {
				int confirm = JOptionPane.showConfirmDialog(this,
						"Soll der aktuelle Test wirklich verworfen werden, um einen neuen zu erstellen?", "Bestätigung",
						JOptionPane.YES_NO_OPTION);
				if (confirm != JOptionPane.YES_OPTION)
					return;
			}
			resetPanel();
		});

		JButton btnAddFromDb = new JButton("Aufgabe aus Datenbank hinzufügen");
		btnAddFromDb.addActionListener(e -> addTaskFromDb());

		JButton btnSave = new JButton("Test speichern");
		btnSave.addActionListener(e -> saveTest());

		JPanel leftButtons = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
		leftButtons.add(btnNewTest);
		leftButtons.add(new javax.swing.JSeparator(javax.swing.SwingConstants.VERTICAL));
		leftButtons.add(btnAddTask);
		leftButtons.add(btnAddFromDb);
		bottomActionPanel.add(leftButtons, BorderLayout.WEST);

		bottomActionPanel.add(btnSave, BorderLayout.EAST);

		add(bottomActionPanel, BorderLayout.SOUTH);

		// Initial task
		addTask(null);
	}

	private void addTask(DatabaseManager.TaskInfo existingTask) {
		int newPos = taskCards.size() + 1;
		TaskCard card = new TaskCard(newPos);
		if (existingTask != null) {
			card.setTaskText(existingTask.taskText);
			card.setReferenceText(existingTask.referenceText);
			card.setResourceInfo(existingTask.resourceInfo);
			card.setMaxPoints(existingTask.maxPoints);
		}
		taskCards.add(card);
		taskTabPane.addTab("Aufgabe " + newPos, card);
		taskTabPane.setSelectedComponent(card);

		revalidate();
		repaint();
	}

	private void loadTestFromDb() {
		try {
			List<DatabaseManager.TestInfo> allTests = dbManager.getAllTests();
			if (allTests.isEmpty()) {
				JOptionPane.showMessageDialog(this, "Keine gespeicherten Tests in der Datenbank gefunden.", "Information",
						JOptionPane.INFORMATION_MESSAGE);
				return;
			}

			TestSelectionDialog dialog = new TestSelectionDialog((Frame) SwingUtilities.getWindowAncestor(this), dbManager, allTests,
					deleted -> {
						if (currentTestId == deleted.id) {
							resetPanel();
						}
					});
			dialog.setVisible(true);

			DatabaseManager.TestInfo selected = dialog.getSelectedTest();

			if (selected != null) {
				// Confirm before overwriting current work if NOT empty
				if (!txtTitle.getText().trim().isEmpty() || !taskCards.isEmpty()) {
					int confirm = JOptionPane.showConfirmDialog(this,
							"Aktuelle Eingaben werden beim Laden eines neuen Tests verworfen. Fortfahren?", "Bestätigung",
							JOptionPane.YES_NO_OPTION);
					if (confirm != JOptionPane.YES_OPTION)
						return;
				}

				// Reset first
				currentTestId = selected.id;
				txtTitle.setText(selected.title);
				txtGrade.setText(selected.gradeLevel);
				txtLearningUnit.setText(selected.learningUnit);

				taskCards.clear();
				taskTabPane.removeAll();

				List<DatabaseManager.TaskInfo> tasks = dbManager.getTasksForTest(selected.id);
				for (DatabaseManager.TaskInfo task : tasks) {
					addTask(task);
				}

				if (tasks.isEmpty()) {
					addTask(null); // Ensure at least one task
				}

				revalidate();
				repaint();

				logger.info("Loaded test definition ID {} with {} tasks", selected.id, tasks.size());
			}
		} catch (Exception e) {
			logger.error("Failed to load tests/tasks from DB", e);
			ExceptionMessage.show(this, "Fehler", "Fehler beim Laden des Tests", e);
		}
	}

	private void addTaskFromDb() {
		try {
			List<DatabaseManager.TaskInfo> allTasks = dbManager.getAllTasks();
			if (allTasks.isEmpty()) {
				JOptionPane.showMessageDialog(this, "Keine Aufgaben in der Datenbank gefunden.", "Information",
						JOptionPane.INFORMATION_MESSAGE);
				return;
			}

			// Simple dialog with a scrollable list
			JList<DatabaseManager.TaskInfo> list = new JList<>(allTasks.toArray(new DatabaseManager.TaskInfo[0]));
			list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

			JScrollPane scroll = new JScrollPane(list);
			scroll.setPreferredSize(new Dimension(500, 300));

			int result = JOptionPane.showConfirmDialog(this, scroll, "Aufgabe auswählen", JOptionPane.OK_CANCEL_OPTION,
					JOptionPane.PLAIN_MESSAGE);
			if (result == JOptionPane.OK_OPTION && list.getSelectedValue() != null) {
				addTask(list.getSelectedValue());
			}

		} catch (Exception e) {
			logger.error("Failed to load tasks from DB", e);
			ExceptionMessage.show(this, "Fehler", "Fehler beim Laden der Aufgaben", e);
		}
	}

	private void removeTask(TaskCard card) {
		taskCards.remove(card);
		taskTabPane.remove(card);
		updatePositions();
	}

	private void moveTask(TaskCard card, int direction) {
		int index = taskCards.indexOf(card);
		int newIndex = index + direction;
		if (newIndex >= 0 && newIndex < taskCards.size()) {
			Collections.swap(taskCards, index, newIndex);

			// Correct way to reorder tabs without losing state/focus
			taskTabPane.remove(card);
			taskTabPane.insertTab("Aufgabe " + (newIndex + 1), null, card, null, newIndex);

			updatePositions();
			taskTabPane.setSelectedIndex(newIndex);
		}
	}

	private void updatePositions() {
		for (int i = 0; i < taskCards.size(); i++) {
			int pos = i + 1;
			taskCards.get(i).setPosition(pos);
			taskTabPane.setTitleAt(i, "Aufgabe " + pos);
		}
	}

	private void saveTest() {
		try {
			String title = txtTitle.getText().trim();
			if (title.isEmpty()) {
				JOptionPane.showMessageDialog(this, "Bitte einen Titel für den Test angeben!", "Fehler", JOptionPane.WARNING_MESSAGE);
				return;
			}

			if (taskCards.isEmpty()) {
				JOptionPane.showMessageDialog(this, "Ein Test muss mindestens eine Aufgabe haben!", "Fehler", JOptionPane.WARNING_MESSAGE);
				return;
			}

			List<DatabaseManager.TaskInfo> tasks = new ArrayList<>();
			for (int i = 0; i < taskCards.size(); i++) {
				TaskCard card = taskCards.get(i);
				tasks.add(new DatabaseManager.TaskInfo(i + 1, card.getTaskText(), card.getReferenceText(), card.getMaxPoints(),
						card.getResourceInfo(), ""));
			}

			if (currentTestId != -1) {
				dbManager.updateTestDefinitionWithTasks(currentTestId, title, txtGrade.getText().trim(), txtLearningUnit.getText().trim(),
						tasks);
				logger.info("Successfully updated test definition ID {} with {} tasks", currentTestId, tasks.size());
				JOptionPane.showMessageDialog(this, "Test erfolgreich aktualisiert!", "Erfolg", JOptionPane.INFORMATION_MESSAGE);
			} else {
				int testId = dbManager.createTestDefinition(title, txtGrade.getText().trim(), txtLearningUnit.getText().trim());
				for (int i = 0; i < tasks.size(); i++) {
					DatabaseManager.TaskInfo task = tasks.get(i);
					dbManager.addTestTask(testId, i + 1, task.taskText, task.referenceText, task.maxPoints, task.resourceInfo);
				}
				logger.info("Successfully created new test definition '{}' with {} tasks", title, tasks.size());
				JOptionPane.showMessageDialog(this, "Test erfolgreich gespeichert!", "Erfolg", JOptionPane.INFORMATION_MESSAGE);
			}

			// Reset
			resetPanel();

		} catch (Exception e) {
			logger.error("Failed to save test definition", e);
			ExceptionMessage.show(this, "Fehler", "Fehler beim Speichern", e);
		}
	}

	private void resetPanel() {
		currentTestId = -1;
		txtTitle.setText("");
		txtGrade.setText("");
		txtLearningUnit.setText("");
		taskCards.clear();
		taskTabPane.removeAll();
		addTask(null);
	}

	private class TaskCard extends JPanel {
		private final JLabel lblNumber;
		private final JTextArea atxtTask;
		private final JTextArea atxtRef;
		private final JTextField txtResource;
		private final JSpinner spinPoints;

		public TaskCard(int position) {
			setLayout(new BorderLayout(10, 5));
			setMaximumSize(new Dimension(Integer.MAX_VALUE, 400));

			// Left: Reordering Buttons and Label
			JPanel leftPanel = new JPanel();
			leftPanel.setLayout(new BoxLayout(leftPanel, BoxLayout.Y_AXIS));
			leftPanel.setBorder(BorderFactory.createEmptyBorder(10, 5, 10, 5));
			lblNumber = new JLabel("#" + position, SwingConstants.CENTER); // Still keep it but maybe it's less prominent? Or remove?
			// Actually, let's keep it as a small indicator or remove the large font.
			lblNumber.setFont(new Font("SansSerif", Font.BOLD, 14));
			lblNumber.setAlignmentX(Component.CENTER_ALIGNMENT);

			JButton btnUp = new JButton("▲");
			btnUp.setToolTipText("Reihenfolge ändern (nach vorne)");
			btnUp.addActionListener(e -> moveTask(this, -1));
			btnUp.setAlignmentX(Component.CENTER_ALIGNMENT);

			JButton btnDown = new JButton("▼");
			btnDown.setToolTipText("Reihenfolge ändern (nach hinten)");
			btnDown.addActionListener(e -> moveTask(this, 1));
			btnDown.setAlignmentX(Component.CENTER_ALIGNMENT);

			JButton btnDel = new JButton("✖");
			btnDel.setForeground(Color.RED);
			btnDel.setToolTipText("Löschen");
			btnDel.addActionListener(e -> removeTask(this));
			btnDel.setAlignmentX(Component.CENTER_ALIGNMENT);

			leftPanel.add(lblNumber);
			leftPanel.add(Box.createVerticalStrut(10));
			leftPanel.add(btnUp);
			leftPanel.add(btnDown);
			leftPanel.add(Box.createVerticalGlue());
			leftPanel.add(btnDel);

			add(leftPanel, BorderLayout.WEST);

			// Center: Content
			Composite contentPanel = new Composite();
			contentPanel.pad(5);

			atxtTask = new JTextArea(6, 20);
			atxtTask.setLineWrap(true);
			atxtTask.setWrapStyleWord(true);
			atxtTask.setFont(atxtTask.getFont().deriveFont(atxtTask.getFont().getSize2D() + 2f));
			UndoHelper.addUndoSupport(atxtTask);

			atxtRef = new JTextArea(12, 20);
			atxtRef.setLineWrap(true);
			atxtRef.setWrapStyleWord(true);
			atxtRef.setFont(atxtRef.getFont().deriveFont(atxtRef.getFont().getSize2D() + 2f));
			UndoHelper.addUndoSupport(atxtRef);

			// Building the layout row by row Row 0: Task Text Header
			contentPanel.addCell(createHeaderPanel("Aufgabentext:", atxtTask)).colspan(2).fillX();
			contentPanel.row();

			// Row 1: Task Text Area
			JScrollPane scrollTask = new JScrollPane(atxtTask);
			scrollTask.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
			scrollTask.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
			contentPanel.addCell(scrollTask).colspan(2).fill().expandY(); // Using expandX/Y for weight
			contentPanel.row();

			// Row 2: Reference Text Header
			contentPanel.addCell(createHeaderPanel("Erwartungshorizont / Musterlösung:", atxtRef)).colspan(2).fillX();
			contentPanel.row();

			// Row 3: Reference Text Area
			JScrollPane scrollRef = new JScrollPane(atxtRef);
			scrollRef.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
			scrollRef.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
			contentPanel.addCell(scrollRef).colspan(2).fill().expandY();
			contentPanel.row();

			// Row 4: Resource and Points labels
			contentPanel.addCell(new JLabel("Resource (z.B. Text A):")).expandX().fillX();
			contentPanel.addCell(new JLabel("Punkte:")).align(org.cuberact.swing.layout.Cell.LEFT).padLeft(10);
			contentPanel.row();

			// Row 5: Resource input and Points spinner
			txtResource = new JTextField();
			contentPanel.addCell(txtResource).expandX().fillX();

			spinPoints = new JSpinner(new SpinnerNumberModel(10, 0, 100, 1));
			JComponent spinEditor = spinPoints.getEditor();
			if (spinEditor instanceof JSpinner.DefaultEditor) {
				((JSpinner.DefaultEditor) spinEditor).getTextField().setColumns(3);
			}
			contentPanel.addCell(spinPoints).align(org.cuberact.swing.layout.Cell.LEFT).padLeft(10);

			add(contentPanel, BorderLayout.CENTER);
		}

		private JPanel createHeaderPanel(String title, JTextArea target) {
			JPanel p = new JPanel(new BorderLayout());
			p.add(new JLabel(title), BorderLayout.WEST);

			JButton btnImport = new JButton("📥"); // Unicode Symbol for Import
			btnImport.setToolTipText("Text importieren...");
			btnImport.setMargin(new Insets(0, 2, 0, 2));
			btnImport.setFocusable(false);
			btnImport.addActionListener(e -> importText(target));
			p.add(btnImport, BorderLayout.EAST);
			return p;
		}

		private void importText(JTextArea target) {
			JFileChooser chooser = new JFileChooser();
			if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
				try {
					String text = DocumentTextExtractor.extractText(chooser.getSelectedFile());
					target.setText(text);
				} catch (Exception e) {
					ExceptionMessage.show(this, "Fehler", "Fehler beim Import", e);
				}
			}
		}

		public void setPosition(int position) {
			lblNumber.setText("#" + position);
		}

		public void setTaskText(String text) {
			atxtTask.setText(text);
		}

		public void setReferenceText(String text) {
			atxtRef.setText(text);
		}

		public void setResourceInfo(String info) {
			txtResource.setText(info);
		}

		public void setMaxPoints(int pts) {
			spinPoints.setValue(pts);
		}

		public String getTaskText() {
			return atxtTask.getText();
		}

		public String getReferenceText() {
			return atxtRef.getText();
		}

		public String getResourceInfo() {
			return txtResource.getText();
		}

		public int getMaxPoints() {
			return (Integer) spinPoints.getValue();
		}
	}
}
