package de.in.flaicheckbot.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Frame;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import de.in.flaicheckbot.db.DatabaseManager;
import de.in.utils.gui.ExceptionMessage;

/**
 * Dialog for searching and selecting tests from the database.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class TestSelectionDialog extends JDialog {
    private static final Logger logger = LogManager.getLogger(TestSelectionDialog.class);
    private final DatabaseManager dbManager;
    private final List<DatabaseManager.TestInfo> allTests;
    private List<DatabaseManager.TestInfo> filteredTests;
    private final JTextField txtSearch;
    private final JList<DatabaseManager.TestInfo> listTests;
    private final DefaultListModel<DatabaseManager.TestInfo> listModel;
    private DatabaseManager.TestInfo selectedTest = null;
    private final Consumer<DatabaseManager.TestInfo> onDeleteCallback;

    public TestSelectionDialog(Frame owner, DatabaseManager dbManager, List<DatabaseManager.TestInfo> allTests,
            Consumer<DatabaseManager.TestInfo> onDeleteCallback) {
        super(owner, "Test laden", true);
        this.dbManager = dbManager;
        this.allTests = allTests;
        this.filteredTests = new ArrayList<>(allTests);
        this.onDeleteCallback = onDeleteCallback;

        setLayout(new BorderLayout(10, 10));
        setSize(500, 400);
        setLocationRelativeTo(owner);

        JPanel topPanel = new JPanel(new BorderLayout(5, 5));
        topPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        topPanel.add(new JLabel("Suchen (Titel, Stufe, Lernabschnitt):"), BorderLayout.NORTH);
        txtSearch = new JTextField();
        topPanel.add(txtSearch, BorderLayout.CENTER);
        add(topPanel, BorderLayout.NORTH);

        listModel = new DefaultListModel<>();
        listTests = new JList<>(listModel);
        listTests.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        updateList();

        JScrollPane scroll = new JScrollPane(listTests);
        scroll.setBorder(BorderFactory.createTitledBorder("Gefundene Tests"));
        add(scroll, BorderLayout.CENTER);

        JPanel buttonPanel = new JPanel();
        buttonPanel.setLayout(new BoxLayout(buttonPanel, BoxLayout.X_AXIS));
        buttonPanel.setBorder(BorderFactory.createEmptyBorder(0, 10, 10, 10));

        JButton btnDelete = new JButton("Löschen");
        btnDelete.setForeground(Color.ORANGE);
        btnDelete.setEnabled(false);

        JButton btnOk = new JButton("Laden");
        JButton btnCancel = new JButton("Abbrechen");

        listTests.addListSelectionListener(e -> {
            btnDelete.setEnabled(listTests.getSelectedValue() != null);
        });

        btnDelete.addActionListener(e -> {
            DatabaseManager.TestInfo selected = listTests.getSelectedValue();
            if (selected != null) {
                int confirm = JOptionPane.showConfirmDialog(this,
                        "Möchten Sie den Test '" + selected.title + "' wirklich unwiderruflich löschen?",
                        "Test löschen",
                        JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);
                if (confirm == JOptionPane.YES_OPTION) {
                    try {
                        if (this.dbManager.isTestInUse(selected.id)) {
                            JOptionPane.showMessageDialog(this,
                                    "Dieser Test kann nicht gelöscht werden, da er bereits in Durchführungen verwendet wird.",
                                    "Löschen nicht möglich", JOptionPane.ERROR_MESSAGE);
                        } else {
                            this.dbManager.deleteTestDefinition(selected.id);
                            if (onDeleteCallback != null) {
                                onDeleteCallback.accept(selected);
                            }
                            allTests.remove(selected);
                            filter();
                            JOptionPane.showMessageDialog(this, "Test erfolgreich gelöscht.", "Erfolg",
                                    JOptionPane.INFORMATION_MESSAGE);
                        }
                    } catch (java.sql.SQLException ex) {
                        logger.error("Failed to delete test", ex);
                        ExceptionMessage.show(this, "Fehler", "Fehler beim Löschen des Tests", ex);
                    }
                }
            }
        });

        btnOk.addActionListener(e -> {
            selectedTest = listTests.getSelectedValue();
            if (selectedTest != null) {
                dispose();
            } else {
                JOptionPane.showMessageDialog(this, "Bitte wählen Sie einen Test aus.", "Information",
                        JOptionPane.INFORMATION_MESSAGE);
            }
        });
        btnCancel.addActionListener(e -> dispose());

        buttonPanel.add(btnDelete);
        buttonPanel.add(Box.createHorizontalGlue());
        buttonPanel.add(btnOk);
        buttonPanel.add(Box.createHorizontalStrut(5));
        buttonPanel.add(btnCancel);
        add(buttonPanel, BorderLayout.SOUTH);

        txtSearch.getDocument().addDocumentListener(new javax.swing.event.DocumentListener() {
            public void insertUpdate(javax.swing.event.DocumentEvent e) {
                filter();
            }

            public void removeUpdate(javax.swing.event.DocumentEvent e) {
                filter();
            }

            public void changedUpdate(javax.swing.event.DocumentEvent e) {
                filter();
            }
        });

        listTests.addMouseListener(new java.awt.event.MouseAdapter() {
            @Override
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                if (evt.getClickCount() == 2) {
                    selectedTest = listTests.getSelectedValue();
                    if (selectedTest != null)
                        dispose();
                }
            }
        });
    }

    private void filter() {
        String query = txtSearch.getText().toLowerCase().trim();
        filteredTests.clear();
        for (DatabaseManager.TestInfo test : allTests) {
            if (test.title.toLowerCase().contains(query) ||
                    (test.gradeLevel != null && test.gradeLevel.toLowerCase().contains(query)) ||
                    (test.learningUnit != null && test.learningUnit.toLowerCase().contains(query))) {
                filteredTests.add(test);
            }
        }
        updateList();
    }

    private void updateList() {
        listModel.clear();
        for (DatabaseManager.TestInfo test : filteredTests) {
            listModel.addElement(test);
        }
    }

    public DatabaseManager.TestInfo getSelectedTest() {
        return selectedTest;
    }
}
