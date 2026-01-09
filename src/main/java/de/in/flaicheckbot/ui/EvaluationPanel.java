package de.in.flaicheckbot.ui;

import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.io.File;
import java.nio.file.Files;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import de.in.flaicheckbot.db.DatabaseManager;
import de.in.flaicheckbot.util.PdfExportUtil;
import de.in.flaicheckbot.util.PdfExportUtil.StudentResult;
import de.in.utils.gui.ExceptionMessage;
import de.in.utils.gui.WrapLayout;

/**
 * Main Evaluation Tab:
 * - North: Filter & Selection (Test & Instance)
 * - Center: List of EvaluationStudentPanels
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class EvaluationPanel extends JPanel {
    private static final Logger logger = LogManager.getLogger(EvaluationPanel.class);
    private final DatabaseManager dbManager;

    private JComboBox<DatabaseManager.TestInfo> comboTests;
    private javax.swing.JSpinner spinDate;
    private JTextField txtClass;
    private JPanel listPanel;
    private JPanel mainCardPanel;
    private CardLayout mainCardLayout;
    private JPanel singleViewPanel;
    private DatabaseManager.AssignmentInfo currentAssignment;
    private JComboBox<String> comboFilter;
    private JComboBox<String> comboLanguage;
    private javax.swing.JProgressBar progressBar;
    private java.util.List<EvaluationStudentPanel> activeStudentPanels = new java.util.ArrayList<>();

    public EvaluationPanel(DatabaseManager dbManager) {
        this.dbManager = dbManager;
        setLayout(new BorderLayout(5, 5));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // North Panel: Filter & Selection
        JPanel northPanel = new JPanel(new WrapLayout(FlowLayout.LEFT, 10, 5));
        northPanel.setBorder(BorderFactory.createTitledBorder("Test & Korrektur auswählen"));

        northPanel.add(new JLabel("Test-Vorlage:"));
        JPanel testSelectionPanel = new JPanel(new BorderLayout(5, 0));
        comboTests = new JComboBox<>();
        comboTests.setPreferredSize(new Dimension(250, 25));
        comboTests.addActionListener(e -> onTestSelected());
        testSelectionPanel.add(comboTests, BorderLayout.CENTER);

        JButton btnSearchTest = new JButton("🗄️");
        btnSearchTest.setToolTipText("Test in Datenbank suchen...");
        btnSearchTest.addActionListener(e -> searchTest());
        testSelectionPanel.add(btnSearchTest, BorderLayout.EAST);
        northPanel.add(testSelectionPanel);

        northPanel.add(new JLabel("Datum:"));
        spinDate = new javax.swing.JSpinner(new javax.swing.SpinnerDateModel());
        spinDate.setEditor(new javax.swing.JSpinner.DateEditor(spinDate, "dd.MM.yyyy"));
        spinDate.setPreferredSize(new Dimension(120, 25));
        northPanel.add(spinDate);

        northPanel.add(new JLabel("Klasse:"));
        txtClass = new JTextField();
        txtClass.setPreferredSize(new Dimension(120, 25));
        northPanel.add(txtClass);

        JButton btnImport = new JButton("Arbeiten importieren");
        northPanel.add(new JLabel("Sprache:"));
        comboLanguage = new JComboBox<>(new String[] { "Deutsch", "Englisch", "Französisch", "Spanisch" });
        comboLanguage.addActionListener(e -> {
            String lang = mapLanguageCode((String) comboLanguage.getSelectedItem());
            for (EvaluationStudentPanel panel : activeStudentPanels) {
                panel.setLanguage(lang);
            }
        });
        northPanel.add(comboLanguage);
        northPanel.add(Box.createHorizontalStrut(10));

        btnImport = new JButton("Schülerarbeiten importieren");
        btnImport.addActionListener(e -> importWorks());
        northPanel.add(btnImport);

        northPanel.add(Box.createHorizontalStrut(10));

        JButton btnNew = new JButton("Neue Korrektur starten");
        btnNew.addActionListener(e -> startNewEvaluation());
        northPanel.add(btnNew);

        JButton btnLoad = new JButton("Korrektur laden");
        btnLoad.addActionListener(e -> loadAssignment());
        northPanel.add(btnLoad);

        JButton btnExport = new JButton("Ergebnisse exportieren");
        btnExport.addActionListener(e -> exportToPdf());
        northPanel.add(btnExport);

        northPanel.add(Box.createHorizontalStrut(10));
        northPanel.add(new JLabel("Filter:"));
        comboFilter = new JComboBox<>(new String[] { "Alle anzeigen", "Bewertete anzeigen", "Unbewertete anzeigen" });
        comboFilter.setSelectedIndex(2); // Default: Unbewertete anzeigen
        comboFilter.addActionListener(e -> applyFilter());
        northPanel.add(comboFilter);

        add(northPanel, BorderLayout.NORTH);

        // Center Panel: Stacked view (List vs Single)
        mainCardLayout = new CardLayout();
        mainCardPanel = new JPanel(mainCardLayout);

        listPanel = new JPanel();
        listPanel.setLayout(new BoxLayout(listPanel, BoxLayout.Y_AXIS));
        JScrollPane scrollPane = new JScrollPane(listPanel);
        scrollPane.getVerticalScrollBar().setUnitIncrement(16);

        singleViewPanel = new JPanel(new BorderLayout());

        mainCardPanel.add(scrollPane, "LIST");
        mainCardPanel.add(singleViewPanel, "SINGLE");

        add(mainCardPanel, BorderLayout.CENTER);

        // South Panel: Batch Actions & Progress (3 Parts: Progress, Recognition,
        // Grading)
        JPanel southPanel = new JPanel(new java.awt.GridLayout(1, 3, 10, 0));
        southPanel.setBorder(BorderFactory.createEmptyBorder(5, 10, 5, 10));

        // Part 1: Progress (Left)
        progressBar = new javax.swing.JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        progressBar.setString("Bereit");
        progressBar.setVisible(false);
        southPanel.add(progressBar);

        // Part 2: Recognition Buttons (Center)
        JPanel recognitionPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 5, 0));
        JButton btnBatchOcrLocal = new JButton("Alle lokal erkennen");
        JButton btnBatchOcrCloud = new JButton("Alle Cloud erkennen");
        recognitionPanel.add(btnBatchOcrLocal);
        recognitionPanel.add(btnBatchOcrCloud);
        southPanel.add(recognitionPanel);

        // Part 3: Grading Buttons (Right)
        JPanel gradingPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 5, 0));
        JButton btnBatchGradeLocal = new JButton("Alle lokal bewerten");
        JButton btnBatchGradeCloud = new JButton("Alle Cloud bewerten");
        gradingPanel.add(btnBatchGradeLocal);
        gradingPanel.add(btnBatchGradeCloud);
        southPanel.add(gradingPanel);

        btnBatchOcrLocal.addActionListener(e -> runBatchAction("OCR_LOCAL"));
        btnBatchOcrCloud.addActionListener(e -> runBatchAction("OCR_CLOUD"));
        btnBatchGradeLocal.addActionListener(e -> runBatchAction("GRADE_LOCAL"));
        btnBatchGradeCloud.addActionListener(e -> runBatchAction("GRADE_CLOUD"));

        // Manage Cloud Buttons enablement via Listener
        de.in.flaicheckbot.MainApp.addAuthListener(status -> {
            btnBatchOcrCloud.setEnabled(status.oauthLoggedIn);
            btnBatchGradeCloud.setEnabled(status.isAnyAvailable());

            btnBatchOcrCloud.setToolTipText(status.oauthLoggedIn ? "Alle Scans via Google Vision OCR erkennen"
                    : "Bitte zuerst über das Menü 'Account -> Google Login' anmelden (erfordert OAuth).");
            btnBatchGradeCloud.setToolTipText(status.isAnyAvailable() ? "Alle Korrekturen via Gemini API starten"
                    : "Bitte zuerst über das Menü 'Account' anmelden oder einen API Key eingeben.");
        });

        add(southPanel, BorderLayout.SOUTH);

        // Initial Data Load
        loadInitialData();
    }

    private void loadInitialData() {
        try {
            List<DatabaseManager.TestInfo> tests = dbManager.getAllTests();
            comboTests.removeAllItems();
            for (DatabaseManager.TestInfo test : tests) {
                comboTests.addItem(test);
            }
        } catch (Exception e) {
            logger.error("Failed to load tests", e);
        }
    }

    private void importWorks() {
        DatabaseManager.TestInfo test = (DatabaseManager.TestInfo) comboTests.getSelectedItem();
        Date selectedDate = (Date) spinDate.getValue();
        String dateStr = new SimpleDateFormat("dd.MM.yyyy").format(selectedDate);
        String classStr = txtClass.getText().trim();

        if (test == null || dateStr.isEmpty() || classStr.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Bitte Test, Datum und Klasse angeben!", "Fehler",
                    JOptionPane.WARNING_MESSAGE);
            return;
        }

        JFileChooser chooser = new JFileChooser();
        chooser.setMultiSelectionEnabled(true);
        if (currentAssignment != null && currentAssignment.lastImportPath != null) {
            File lastFolder = new File(currentAssignment.lastImportPath);
            if (lastFolder.exists()) {
                chooser.setCurrentDirectory(lastFolder);
            }
        }
        chooser.setFileFilter(
                new FileNameExtensionFilter("Unterstützte Formate (Bilder & PDF)", "jpg", "jpeg", "png", "bmp", "pdf"));

        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File[] files = chooser.getSelectedFiles();
            if (files.length == 0)
                return;

            progressBar.setMaximum(files.length);
            progressBar.setValue(0);
            progressBar.setString("Importiere Arbeiten...");
            progressBar.setVisible(true);

            new javax.swing.SwingWorker<Void, Integer>() {
                private DatabaseManager.AssignmentInfo assignmentInfo;

                @Override
                protected Void doInBackground() throws Exception {
                    int classId = dbManager.getOrCreateClass(classStr);
                    String assignmentTitle = dateStr + " - " + classStr;
                    int assignmentId = dbManager.getOrCreateAssignment(classId, test.id, assignmentTitle);

                    for (int i = 0; i < files.length; i++) {
                        File f = files[i];
                        String studentId = f.getName();
                        int studentDbId = dbManager.getOrCreateStudent(studentId, classId);

                        byte[] imageData;
                        if (f.getName().toLowerCase().endsWith(".pdf")) {
                            imageData = Files.readAllBytes(f.toPath());
                        } else {
                            imageData = Files.readAllBytes(f.toPath());
                        }
                        dbManager.addStudentWork(assignmentId, studentDbId, imageData);
                        publish(i + 1);
                    }

                    String lastDirPath = files[0].getParent();
                    dbManager.updateAssignmentImportPath(assignmentId, lastDirPath);
                    assignmentInfo = new DatabaseManager.AssignmentInfo(assignmentId, assignmentTitle, classStr,
                            test.title, lastDirPath);
                    return null;
                }

                @Override
                protected void process(List<Integer> chunks) {
                    int latest = chunks.get(chunks.size() - 1);
                    progressBar.setValue(latest);
                    progressBar.setString("Importiere: " + latest + " / " + files.length);
                }

                @Override
                protected void done() {
                    try {
                        get();
                        displayAssignment(assignmentInfo, test);
                        progressBar.setString("Import abgeschlossen.");
                    } catch (Exception e) {
                        logger.error("Import failed", e);
                        JOptionPane.showMessageDialog(EvaluationPanel.this, "Fehler beim Import: " + e.getMessage(),
                                "Fehler", JOptionPane.ERROR_MESSAGE);
                        progressBar.setString("Import fehlgeschlagen.");
                    }
                    // Keep visible for a moment or use a timer to hide
                    new javax.swing.Timer(3000, e -> {
                        progressBar.setVisible(false);
                        progressBar.setString("Bereit");
                    }).start();
                }
            }.execute();
        }
    }

    private void onTestSelected() {
        // No longer auto-loading assignments here, but we could list recent ones if we
        // wanted.
        // For now, follow the user's specific request for the import flow.
    }

    private void loadAssignment() {
        DatabaseManager.TestInfo test = (DatabaseManager.TestInfo) comboTests.getSelectedItem();
        if (test == null) {
            JOptionPane.showMessageDialog(this, "Bitte zuerst einen Test auswählen!", "Hinweis",
                    JOptionPane.WARNING_MESSAGE);
            return;
        }

        try {
            List<DatabaseManager.AssignmentInfo> assignments = dbManager.getAssignmentsForTest(test.id);
            if (assignments.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Keine Korrekturen für diesen Test gefunden.", "Information",
                        JOptionPane.INFORMATION_MESSAGE);
                return;
            }

            DatabaseManager.AssignmentInfo selected = (DatabaseManager.AssignmentInfo) JOptionPane.showInputDialog(
                    this, "Korrektur auswählen:", "Laden",
                    JOptionPane.PLAIN_MESSAGE, null,
                    assignments.toArray(), assignments.get(0));

            if (selected != null) {
                displayAssignment(selected, test);
            }
        } catch (Exception e) {
            logger.error("Failed to load assignments", e);
            ExceptionMessage.show(this, "Fehler", "Laden fehlgeschlagen", e);
        }
    }

    private void startNewEvaluation() {
        if (currentAssignment != null || !txtClass.getText().trim().isEmpty()) {
            int confirm = JOptionPane.showConfirmDialog(this,
                    "Soll die aktuelle Korrektur-Ansicht wirklich verworfen werden, um eine neue zu starten?",
                    "Bestätigung", JOptionPane.YES_NO_OPTION);
            if (confirm != JOptionPane.YES_OPTION)
                return;
        }
        currentAssignment = null;
        txtClass.setText("");
        listPanel.removeAll();
        activeStudentPanels.clear();
        mainCardLayout.show(mainCardPanel, "LIST");
        revalidate();
        repaint();
    }

    public void displayAssignment(DatabaseManager.AssignmentInfo assignment, DatabaseManager.TestInfo test)
            throws SQLException {
        this.currentAssignment = assignment;
        // Populate fields from assignment
        txtClass.setText(assignment.className);
        if (assignment.title.contains(" - ")) {
            String datePart = assignment.title.substring(0, assignment.title.indexOf(" - "));
            try {
                Date date = new SimpleDateFormat("dd.MM.yyyy").parse(datePart);
                spinDate.setValue(date);
            } catch (Exception e) {
                logger.warn("Could not parse date from assignment title: {}", assignment.title);
            }
        }

        listPanel.removeAll();
        activeStudentPanels.clear();
        singleViewPanel.removeAll();
        mainCardLayout.show(mainCardPanel, "LIST");

        List<DatabaseManager.StudentWorkInfo> workList = dbManager.getStudentWorkForAssignment(assignment.id);

        if (workList.isEmpty()) {
            logger.info("No student works found for assignment '{}'", assignment.title);
            listPanel.revalidate();
            listPanel.repaint();
            return;
        }

        progressBar.setMaximum(workList.size());
        progressBar.setValue(0);
        progressBar.setString("Lade Korrekturen...");
        progressBar.setVisible(true);

        new javax.swing.SwingWorker<Void, EvaluationStudentPanel>() {
            @Override
            protected Void doInBackground() throws Exception {
                for (DatabaseManager.StudentWorkInfo work : workList) {
                    EvaluationStudentPanel panel = new EvaluationStudentPanel(dbManager, work, test);
                    String lang = mapLanguageCode((String) comboLanguage.getSelectedItem());
                    panel.setLanguage(lang);
                    panel.setMaximizeListener(EvaluationPanel.this::handleToggleMaximize);
                    panel.addPropertyChangeListener("isEvaluated", evt -> applyFilter());
                    publish(panel);
                    // Small sleep to make the animation visible if it's too fast
                    Thread.sleep(50);
                }
                return null;
            }

            @Override
            protected void process(List<EvaluationStudentPanel> chunks) {
                for (EvaluationStudentPanel panel : chunks) {
                    listPanel.add(panel);
                    activeStudentPanels.add(panel);
                    listPanel.add(Box.createVerticalStrut(10));
                }
                progressBar.setValue(activeStudentPanels.size());
                progressBar.setString("Geladen: " + activeStudentPanels.size() + " / " + workList.size());
                applyFilter();
                listPanel.revalidate();
                listPanel.repaint();
            }

            @Override
            protected void done() {
                try {
                    get();
                    progressBar.setString("Laden abgeschlossen.");
                    logger.info("Loaded {} student works for assignment '{}'", activeStudentPanels.size(),
                            assignment.title);
                } catch (Exception e) {
                    logger.error("Failed to load student works", e);
                    progressBar.setString("Laden fehlgeschlagen.");
                }
                new javax.swing.Timer(2000, e -> {
                    progressBar.setVisible(false);
                    progressBar.setString("Bereit");
                }).start();
            }
        }.execute();
    }

    private void handleToggleMaximize(EvaluationStudentPanel panel, boolean maximize) {
        if (maximize) {
            // Find current scroll position if we wanted to restore it, but simple focus is
            // usually enough
            singleViewPanel.removeAll();
            singleViewPanel.add(panel, BorderLayout.CENTER);
            mainCardLayout.show(mainCardPanel, "SINGLE");
        } else {
            // Put it back into the list
            // Since it was removed from the listPanel (automatically when added to
            // singleViewPanel),
            // we need to find its original position or just rebuild the list.
            // Rebuilding the list is safer but might lose state if not careful.
            // Actually, Swing allows moving components.
            // We'll just put it back at the right index.

            // To avoid complexity, let's just show the list again.
            // BUT wait, if we added it to singleViewPanel, it's GONE from listPanel.
            // Let's find where it was.

            // Actually, a simpler way is to just refresh the whole list view
            // but we don't want to lose other panels' state.

            // Find the panel in the list of children and restore it if needed.
            // But it's easier to just rebuild or keep a reference.

            // Let's try to find the gap and fill it.
            // Actually, let's just hide the North panel when maximized to give more space?
            // User didn't ask for that, but "Maximized" usually implies full view.

            // Re-adding to list:
            // We need to know which index it was at.
            // For now, let's just revalidate the list.

            // Wait, if it's moved to singleViewPanel, it is removed from listPanel.
            // We must add it back to listPanel.

            // I'll use a more robust way: Re-layout the Whole EvaluationPanel Center.
            restorePanelToList(panel);
        }
    }

    private void restorePanelToList(EvaluationStudentPanel panel) {
        // Find alphabetical or original order?
        // Let's just find the first available slot.
        // Actually, if we just add it back, it goes to the bottom.
        // Better: Keep all panels in a List and rebuild listPanel.

        singleViewPanel.removeAll();
        // Just rebuild the list from the components we still have?
        // No, let's just show the list card.
        // We'll need to re-add the panel to listPanel.

        // Since we don't track indices easily here, let's just add it back to listPanel
        // and hope it's not too jarring. Or better: rebuild the list.

        // Let's try adding it back to the top for now, or just refresh.
        // Actually, I'll just refresh from DB to keep it consistent with other logic,
        // although it's slower.

        // OR: Just keep the panel in the list but change its parent?
        // Component move automatically removes it from old parent.

        // Let's just add it back and revalidate.
        listPanel.add(panel, 0); // Put at top for visibility
        listPanel.add(Box.createVerticalStrut(10), 1);
        mainCardLayout.show(mainCardPanel, "LIST");
        listPanel.revalidate();
        listPanel.repaint();
    }

    private void runBatchAction(String actionType) {
        if (activeStudentPanels.isEmpty()) {
            return;
        }

        progressBar.setMaximum(activeStudentPanels.size());
        progressBar.setValue(0);
        progressBar.setString("Batch: " + actionType + "...");
        progressBar.setVisible(true);

        new Thread(() -> {
            for (int i = 0; i < activeStudentPanels.size(); i++) {
                EvaluationStudentPanel panel = activeStudentPanels.get(i);
                final int index = i;
                try {
                    java.util.concurrent.CompletableFuture<Void> future;
                    switch (actionType) {
                        case "OCR_LOCAL":
                            String language = (String) comboLanguage.getSelectedItem();
                            String langCode = mapLanguageCode(language);
                            future = panel.runLocalRecognition(langCode);
                            break;
                        case "OCR_CLOUD":
                            future = panel.runCloudRecognition();
                            break;
                        case "GRADE_LOCAL":
                            future = panel.runGradingTask(false);
                            break;
                        case "GRADE_CLOUD":
                            future = panel.runGradingTask(true);
                            break;
                        default:
                            continue;
                    }

                    if (future != null) {
                        future.get(); // Wait for this one to finish before moving to next
                    }

                    SwingUtilities.invokeLater(() -> {
                        progressBar.setValue(index + 1);
                        progressBar.setString("Fortschritt: " + (index + 1) + " / " + activeStudentPanels.size());
                    });

                } catch (Exception e) {
                    logger.error("Batch element failed", e);
                }
            }
            SwingUtilities.invokeLater(() -> {
                progressBar.setString("Abgeschlossen.");
                new javax.swing.Timer(3000, event -> {
                    progressBar.setVisible(false);
                    progressBar.setString("Bereit");
                }).start();
            });
        }).start();
    }

    private void searchTest() {
        try {
            List<DatabaseManager.TestInfo> allTests = dbManager.getAllTests();
            if (allTests.isEmpty()) {
                return;
            }
            TestSelectionDialog dialog = new TestSelectionDialog(
                    (java.awt.Frame) SwingUtilities.getWindowAncestor(this),
                    dbManager, allTests, null);
            dialog.setVisible(true);

            DatabaseManager.TestInfo selected = dialog.getSelectedTest();
            if (selected != null) {
                // Find in combo and select
                for (int i = 0; i < comboTests.getItemCount(); i++) {
                    DatabaseManager.TestInfo item = comboTests.getItemAt(i);
                    if (item != null && item.id == selected.id) {
                        comboTests.setSelectedIndex(i);
                        break;
                    }
                }
            }
        } catch (Exception e) {
            logger.error("Search failed", e);
        }
    }

    private void exportToPdf() {
        if (currentAssignment == null || activeStudentPanels.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Bitte zuerst eine Korrektur laden.", "Export",
                    JOptionPane.WARNING_MESSAGE);
            return;
        }

        JFileChooser chooser = new JFileChooser();
        chooser.setSelectedFile(
                new File("Ergebnis_" + currentAssignment.title.replaceAll("[^a-zA-Z0-9]", "_") + ".pdf"));
        if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            File file = chooser.getSelectedFile();
            if (!file.getName().toLowerCase().endsWith(".pdf")) {
                file = new File(file.getAbsolutePath() + ".pdf");
            }

            try {
                List<StudentResult> results = new ArrayList<>();
                for (EvaluationStudentPanel panel : activeStudentPanels) {
                    results.add(new StudentResult(panel.getStudentName(), panel.getFeedback(), panel.getScore()));
                }
                PdfExportUtil.export(file, currentAssignment, results);
                JOptionPane.showMessageDialog(this, "Datei erfolgreich gespeichert unter:\n" + file.getAbsolutePath(),
                        "Export erfolgreich", JOptionPane.INFORMATION_MESSAGE);
            } catch (Exception e) {
                logger.error("PDF Export failed", e);
                JOptionPane.showMessageDialog(this, "Export fehlgeschlagen: " + e.getMessage(), "Fehler",
                        JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void applyFilter() {
        String filter = (String) comboFilter.getSelectedItem();
        for (java.awt.Component c : listPanel.getComponents()) {
            if (c instanceof EvaluationStudentPanel) {
                EvaluationStudentPanel panel = (EvaluationStudentPanel) c;
                boolean visible = true;
                if ("Bewertete anzeigen".equals(filter)) {
                    visible = panel.isEvaluated();
                } else if ("Unbewertete anzeigen".equals(filter)) {
                    visible = !panel.isEvaluated();
                }
                panel.setVisible(visible);

                // Find the following strut
                int idx = -1;
                java.awt.Component[] comps = listPanel.getComponents();
                for (int i = 0; i < comps.length; i++) {
                    if (comps[i] == panel) {
                        idx = i;
                        break;
                    }
                }
                if (idx != -1 && idx + 1 < comps.length && comps[idx + 1] instanceof javax.swing.Box.Filler) {
                    comps[idx + 1].setVisible(visible);
                }
            }
        }
        listPanel.revalidate();
        listPanel.repaint();
    }

    private String mapLanguageCode(String language) {
        if ("Englisch".equals(language))
            return "en";
        if ("Französisch".equals(language))
            return "fr";
        if ("Spanisch".equals(language))
            return "es";
        return "de";
    }
}
