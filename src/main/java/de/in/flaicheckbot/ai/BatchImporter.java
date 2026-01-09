package de.in.flaicheckbot.ai;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import de.in.flaicheckbot.AIEngineClient;
import de.in.flaicheckbot.db.DatabaseManager;

/**
 * CLI Utility for batch importing training samples and resetting the system.
 */
public class BatchImporter {

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Usage: java -cp ... de.in.flaicheckbot.ai.BatchImporter [options]");
            System.out.println("Options:");
            System.out.println("  --reset-db           Nuke all database tables");
            System.out.println("  --reset-ai           Nuke AI LoRA adapters");
            System.out.println("  --import <folder>    Import samples from subfolders (de, en, etc.)");
            return;
        }

        String dbPath = "flaicheckbot.db";
        DatabaseManager dbManager = new DatabaseManager(dbPath);
        AIEngineClient aiClient = new AIEngineClient();
        TrainingManager trainingManager = new TrainingManager();
        AiProcessManager aiManager = new AiProcessManager();

        try {
            // Ensure AI engine is running for batch operations
            if (!aiManager.isEngineRunning()) {
                System.out.println("AI Engine not running, starting...");
                aiManager.startEngine();
                if (!aiManager.waitForEngine(60)) {
                    System.err.println("Fatal: AI Engine failed to start within 60s.");
                    System.exit(1);
                }
            }
            boolean resetDb = false;
            boolean resetAi = false;
            String importFolder = null;

            for (int i = 0; i < args.length; i++) {
                if ("--reset-db".equals(args[i]))
                    resetDb = true;
                else if ("--reset-ai".equals(args[i]))
                    resetAi = true;
                else if ("--import".equals(args[i]) && i + 1 < args.length) {
                    importFolder = args[++i];
                }
            }

            if (resetDb) {
                System.out.println("Resetting database...");
                dbManager.resetDatabase();
            }

            if (resetAi) {
                System.out.println("Resetting AI engine (all languages)...");
                aiClient.resetTraining(null).get(10, TimeUnit.SECONDS);
            }

            if (importFolder != null) {
                File root = new File(importFolder);
                if (!root.exists() || !root.isDirectory()) {
                    System.err.println("Import folder not found: " + importFolder);
                    return;
                }

                List<String> foundLanguages = new ArrayList<>();
                File[] langDirs = root.listFiles(File::isDirectory);
                if (langDirs != null) {
                    for (File langDir : langDirs) {
                        String lang = langDir.getName(); // 'de', 'en', etc.
                        System.out.println("Processing language: " + lang);
                        foundLanguages.add(lang);

                        // Find all PNG and PDF files
                        File[] files = langDir.listFiles((d, name) -> name.toLowerCase().endsWith(".png")
                                || name.toLowerCase().endsWith(".pdf"));
                        if (files == null || files.length == 0) {
                            System.out.println("  No samples found for " + lang);
                            continue;
                        }

                        // Create a training set for this batch
                        int setId = dbManager.createTrainingSet(
                                "Batch Import " + lang + " " + System.currentTimeMillis(), "Batch auto-import", lang);
                        System.out.println("  Created training set ID: " + setId);

                        for (File f : files) {
                            String baseName = f.getName().substring(0, f.getName().lastIndexOf('.'));
                            File txtFile = new File(langDir, baseName + ".txt");
                            if (!txtFile.exists()) {
                                System.err.println("  Warning: Missing TXT for " + f.getName());
                                continue;
                            }

                            String text = Files.readString(txtFile.toPath()).trim();
                            byte[] imageData = Files.readAllBytes(f.toPath());
                            String mime = f.getName().toLowerCase().endsWith(".pdf") ? "application/pdf" : "image/png";
                            dbManager.addTrainingSample(setId, imageData, mime, text);
                            System.out.println("    Imported: " + f.getName());
                        }
                    }
                }

                // Trigger training for each language
                System.out.println("Triggering training for all imported languages...");
                for (String lang : foundLanguages) {
                    System.out.print("  Starting training for " + lang + "... ");
                    String response = trainingManager.startTraining(lang).get(5, TimeUnit.MINUTES);
                    System.out.println("Result: " + response);
                }
            }

            System.out.println("Batch operation completed successfully.");
        } catch (Exception e) {
            System.err.println("Fatal error during batch operation:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
