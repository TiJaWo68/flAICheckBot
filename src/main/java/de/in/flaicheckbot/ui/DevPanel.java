package de.in.flaicheckbot.ui;

import de.in.flaicheckbot.db.DatabaseManager;

/**
 * Developer-only panel that extends TrainingPanel but enables visual debug
 * segments.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class DevPanel extends TrainingPanel {

    public DevPanel(DatabaseManager dbManager) {
        super(dbManager);
        this.showDebugSegments = true;
        // Re-initialize UI to pick up the debug segments panel
        removeAll();
        initBaseUI();
        // TrainingPanel adds action buttons in its constructor, we need to replicate
        // that
        // since we just cleared everything with removeAll() and initBaseUI()
        addTrainingActionButtons();
    }
}
