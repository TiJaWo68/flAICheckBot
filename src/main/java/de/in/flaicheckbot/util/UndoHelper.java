package de.in.flaicheckbot.util;

import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;

import javax.swing.AbstractAction;
import javax.swing.KeyStroke;
import javax.swing.text.JTextComponent;
import javax.swing.undo.UndoManager;

/**
 * Utility to add Undo/Redo support to Swing text components.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class UndoHelper {

    public static void addUndoSupport(JTextComponent textComponent) {
        UndoManager undoManager = new UndoManager();

        // Add listener for undoable edits
        textComponent.getDocument().addUndoableEditListener(e -> {
            undoManager.addEdit(e.getEdit());
        });

        // Map Ctrl+Z to Undo
        textComponent.getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_Z, KeyEvent.CTRL_DOWN_MASK), "Undo");
        textComponent.getActionMap().put("Undo", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (undoManager.canUndo()) {
                    undoManager.undo();
                }
            }
        });

        // Map Ctrl+Y to Redo
        textComponent.getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_Y, KeyEvent.CTRL_DOWN_MASK), "Redo");
        textComponent.getActionMap().put("Redo", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (undoManager.canRedo()) {
                    undoManager.redo();
                }
            }
        });
    }
}
