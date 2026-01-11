package de.in.flaicheckbot.ui;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.DefaultCellEditor;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.SwingUtilities;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableCellRenderer;

import de.in.flaicheckbot.ui.DocumentProcessorPanel.RecognitionSegment;

/**
 * Dialog for selecting and correcting segmented training data.
 * Triple sync: Speichern | X (B) | Grafik | X (beides) | Text | X (T)
 */
public class TrainingDataSelectionDialog extends JDialog {
    private final List<RecognitionSegment> imageList = new ArrayList<>();
    private final List<String> textList = new ArrayList<>();
    private final DefaultTableModel tableModel;
    private final JTable table;
    private final JButton btnSave;
    private final JLabel lblStatus;
    private boolean confirmed = false;
    private boolean segmentationFinished = false;

    public TrainingDataSelectionDialog(Component parent, List<RecognitionSegment> initialSegments, String currentText) {
        super(SwingUtilities.getWindowAncestor(parent), "Trainingsdaten auswählen & synchronisieren",
                ModalityType.APPLICATION_MODAL);

        setLayout(new BorderLayout(10, 10));

        if (currentText != null && !currentText.trim().isEmpty()) {
            textList.addAll(Arrays.asList(currentText.split("\\n")));
        }

        // Table Model: Save (0), Del Img (1), Image (2), Del Both (3), Text (4), Del
        // Txt (5)
        tableModel = new DefaultTableModel(
                new Object[] { "Speichern", "X (B)", "Segment Grafik", "X (beides)", "Zugehöriger Text", "X (T)" }, 0) {
            @Override
            public Class<?> getColumnClass(int columnIndex) {
                if (columnIndex == 0)
                    return Boolean.class;
                if (columnIndex == 2)
                    return ImageIcon.class;
                return String.class;
            }

            @Override
            public boolean isCellEditable(int row, int column) {
                return column == 0 || column == 1 || column == 3 || column == 5 || column == 4;
            }
        };

        table = new JTable(tableModel);
        table.setRowHeight(80);
        table.setShowGrid(true);

        // Layout configuration
        table.getColumnModel().getColumn(0).setMinWidth(70);
        table.getColumnModel().getColumn(0).setMaxWidth(70);
        table.getColumnModel().getColumn(0).setResizable(false);

        table.getColumnModel().getColumn(1).setMinWidth(50);
        table.getColumnModel().getColumn(1).setMaxWidth(50);
        table.getColumnModel().getColumn(1).setResizable(false);

        table.getColumnModel().getColumn(2).setPreferredWidth(700);

        table.getColumnModel().getColumn(3).setMinWidth(70);
        table.getColumnModel().getColumn(3).setMaxWidth(70);
        table.getColumnModel().getColumn(3).setResizable(false);

        table.getColumnModel().getColumn(4).setPreferredWidth(300);

        table.getColumnModel().getColumn(5).setMinWidth(50);
        table.getColumnModel().getColumn(5).setMaxWidth(50);
        table.getColumnModel().getColumn(5).setResizable(false);

        // Action Buttons in Table
        table.getColumnModel().getColumn(1).setCellRenderer(new ButtonRenderer("X"));
        table.getColumnModel().getColumn(1)
                .setCellEditor(new ButtonEditor(new JButton("X"), e -> deleteImage(table.getSelectedRow())));

        table.getColumnModel().getColumn(3).setCellRenderer(new ButtonRenderer("X"));
        table.getColumnModel().getColumn(3)
                .setCellEditor(new ButtonEditor(new JButton("X"), e -> deleteBoth(table.getSelectedRow())));

        table.getColumnModel().getColumn(5).setCellRenderer(new ButtonRenderer("X"));
        table.getColumnModel().getColumn(5)
                .setCellEditor(new ButtonEditor(new JButton("X"), e -> deleteText(table.getSelectedRow())));

        tableModel.addTableModelListener(e -> {
            if (e.getColumn() == 4 && e.getFirstRow() >= 0 && e.getFirstRow() < textList.size()) {
                textList.set(e.getFirstRow(), (String) tableModel.getValueAt(e.getFirstRow(), 4));
            }
            updateSaveButtonState();
        });

        JScrollPane scrollPane = new JScrollPane(table);
        add(scrollPane, BorderLayout.CENTER);

        // Bottom
        JPanel bottomPanel = new JPanel(new BorderLayout());
        lblStatus = new JLabel(" ");
        bottomPanel.add(lblStatus, BorderLayout.WEST);

        JPanel actionPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton btnCancel = new JButton("Abbrechen");
        btnCancel.addActionListener(e -> dispose());

        btnSave = new JButton("Speichern");
        btnSave.setEnabled(false);
        btnSave.addActionListener(e -> {
            confirmed = true;
            dispose();
        });

        actionPanel.add(btnCancel);
        actionPanel.add(btnSave);
        bottomPanel.add(actionPanel, BorderLayout.EAST);
        add(bottomPanel, BorderLayout.SOUTH);

        if (initialSegments != null) {
            for (RecognitionSegment seg : initialSegments) {
                if (!seg.rejected) {
                    imageList.add(seg);
                }
            }
            segmentationFinished = true;
        }
        rebuildTable();

        setSize(1200, 600);
        setLocationRelativeTo(parent);
    }

    public void addSegment(RecognitionSegment seg) {
        if (seg.rejected)
            return;
        SwingUtilities.invokeLater(() -> {
            imageList.add(seg);
            lblStatus.setText("Lade Segment " + (imageList.size()) + "...");
            rebuildTable();
        });
    }

    public void setSegmentationFinished() {
        SwingUtilities.invokeLater(() -> {
            segmentationFinished = true;
            lblStatus.setText("Segmentierung abgeschlossen. " + imageList.size() + " Segmente gefunden.");
            updateSaveButtonState();
        });
    }

    private synchronized void rebuildTable() {
        int rows = Math.max(imageList.size(), textList.size());
        int currentRows = tableModel.getRowCount();

        for (int i = 0; i < rows; i++) {
            ImageIcon icon = null;
            if (i < imageList.size()) {
                icon = createIcon(imageList.get(i).base64Image);
            }
            String txt = (i < textList.size()) ? textList.get(i) : "";

            if (i < currentRows) {
                tableModel.setValueAt(icon, i, 2);
                tableModel.setValueAt(txt, i, 4);
            } else {
                tableModel.addRow(new Object[] { true, "X", icon, "X", txt, "X" });
            }
        }

        while (tableModel.getRowCount() > rows) {
            tableModel.removeRow(tableModel.getRowCount() - 1);
        }
        updateSaveButtonState();
    }

    private ImageIcon createIcon(String base64) {
        try {
            byte[] bytes = Base64.getDecoder().decode(base64);
            BufferedImage img = ImageIO.read(new ByteArrayInputStream(bytes));
            if (img != null) {
                int h = 80;
                int w = (int) (img.getWidth() * ((double) h / img.getHeight()));
                if (w > 800) {
                    w = 800;
                    h = (int) (img.getHeight() * ((double) w / img.getWidth()));
                }
                return new ImageIcon(img.getScaledInstance(w, h, Image.SCALE_SMOOTH));
            }
        } catch (Exception e) {
        }
        return null;
    }

    private void deleteImage(int row) {
        if (row >= 0 && row < imageList.size()) {
            imageList.remove(row);
            rebuildTable();
        }
    }

    private void deleteText(int row) {
        if (row >= 0 && row < textList.size()) {
            textList.remove(row);
            rebuildTable();
        }
    }

    private void deleteBoth(int row) {
        boolean changed = false;
        if (row >= 0 && row < imageList.size()) {
            imageList.remove(row);
            changed = true;
        }
        if (row >= 0 && row < textList.size()) {
            textList.remove(row);
            changed = true;
        }
        if (changed)
            rebuildTable();
    }

    private void updateSaveButtonState() {
        if (!segmentationFinished) {
            btnSave.setEnabled(false);
            btnSave.setToolTipText("Warten auf Ende der Segmentierung...");
            return;
        }

        boolean anySelected = false;
        for (int i = 0; i < tableModel.getRowCount(); i++) {
            if (Boolean.TRUE.equals(tableModel.getValueAt(i, 0))) {
                if (i < imageList.size() && i < textList.size()) {
                    String txt = textList.get(i);
                    if (txt != null && !txt.trim().isEmpty()) {
                        anySelected = true;
                        break;
                    }
                }
            }
        }
        btnSave.setEnabled(anySelected);
    }

    public boolean isConfirmed() {
        return confirmed;
    }

    public List<RecognitionSegment> getFinalSegments() {
        List<RecognitionSegment> result = new ArrayList<>();
        int rows = Math.min(imageList.size(), textList.size());
        for (int i = 0; i < rows; i++) {
            if (Boolean.TRUE.equals(tableModel.getValueAt(i, 0))) {
                String correctedText = textList.get(i);
                if (correctedText != null && !correctedText.trim().isEmpty()) {
                    RecognitionSegment orig = imageList.get(i);
                    result.add(new RecognitionSegment(orig.page, orig.index, correctedText, orig.base64Image, false,
                            null));
                }
            }
        }
        return result;
    }

    class ButtonRenderer extends JButton implements TableCellRenderer {
        public ButtonRenderer(String label) {
            setText(label);
        }

        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus,
                int row, int column) {
            return this;
        }
    }

    class ButtonEditor extends DefaultCellEditor {
        private final JButton button;

        public ButtonEditor(JButton button, java.util.function.Consumer<java.awt.event.ActionEvent> action) {
            super(new JCheckBox());
            this.button = button;
            this.button.addActionListener(e -> {
                fireEditingStopped();
                action.accept(e);
            });
        }

        @Override
        public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row,
                int column) {
            return button;
        }
    }
}
