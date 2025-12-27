package de.in.flaicheckbot.ai;

import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.TimeUnit;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Manages the lifecycle of the local Python AI engine process,
 * including startup, shutdown, and health checks.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class AiProcessManager {
    private static final Logger logger = LogManager.getLogger(AiProcessManager.class);
    private Process process;
    private final String scriptPath;
    private final File aiDir;
    private final PythonEnvironmentManager envManager;
    private String pythonPath;

    public AiProcessManager() {
        this.aiDir = findAiDir();
        this.envManager = new PythonEnvironmentManager(new File("."), new File(aiDir, "requirements.txt"));

        // Use environment variables for overrides, otherwise look for root .venv
        String envPython = System.getenv("FL_KI_PYTHON");
        if (envPython != null) {
            this.pythonPath = envPython;
        } else {
            recalculatePythonPath();
        }
        this.scriptPath = new File(aiDir, "icr_prototype.py").getAbsolutePath();

        logger.info("KI-Engine paths initialized. Home: {}, Python: {}, Script: {}", aiDir.getAbsolutePath(),
                pythonPath, scriptPath);
    }

    public void recalculatePythonPath() {
        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
        String venvSubDir = isWindows ? "Scripts" : "bin";
        String pythonExec = isWindows ? "python.exe" : "python";

        // Priority: 1. Portable Python (Windows only), 2. Root .venv, 3. AI-local venv
        if (isWindows && envManager.isReady()) {
            this.pythonPath = envManager.getPythonPath();
        } else {
            File rootVenv = new File(".venv/" + venvSubDir + "/" + pythonExec);
            if (rootVenv.exists()) {
                this.pythonPath = rootVenv.getAbsolutePath();
            } else {
                File localVenv = new File(aiDir, "venv/" + venvSubDir + "/" + pythonExec);
                if (localVenv.exists()) {
                    this.pythonPath = localVenv.getAbsolutePath();
                } else {
                    // Final fallback: global python
                    this.pythonPath = pythonExec;
                }
            }
        }
    }

    public boolean needsSetup() {
        if (!System.getProperty("os.name").toLowerCase().contains("win"))
            return false;

        // If we only have the global python or nothing, we might need setup
        return pythonPath.equals("python.exe") && !envManager.isReady();
    }

    public void performSetup(PythonEnvironmentManager.ProgressListener listener) throws Exception {
        envManager.setup(listener);
        recalculatePythonPath();
    }

    private File findAiDir() {
        // 1. Check for environment override
        String envHome = System.getenv("FL_KI_HOME");
        if (envHome != null) {
            File f = new File(envHome);
            if (f.exists() && f.isDirectory())
                return f;
        }

        // 2. Check current directory (production/bundle)
        File prodDir = new File("ai");
        if (prodDir.exists() && prodDir.isDirectory())
            return prodDir;

        // 3. Check src directory (development)
        File devDir = new File("src/ai");
        if (devDir.exists() && devDir.isDirectory())
            return devDir;

        // Fallback to current directory as best guess
        logger.warn("Could not reliably locate 'ai' directory! Falling back to current directory.");
        return new File(".");
    }

    public synchronized void startEngine() {
        if (isEngineRunning()) {
            logger.info("AI Engine is already running.");
            return;
        }

        try {
            logger.info("Starting AI Engine: {} {}", pythonPath, scriptPath);
            ProcessBuilder pb = new ProcessBuilder(pythonPath, scriptPath);
            pb.directory(aiDir);

            // Redirect output to log file
            File logDir = new File("log");
            if (!logDir.exists()) {
                logDir.mkdirs();
            }
            File aiLog = new File(logDir, "ai-engine.log");
            pb.redirectOutput(ProcessBuilder.Redirect.appendTo(aiLog));
            pb.redirectError(ProcessBuilder.Redirect.appendTo(aiLog));

            process = pb.start();

            // Shutdown hook to kill the process when JVM exits
            Runtime.getRuntime().addShutdownHook(new Thread(this::stopEngine));

        } catch (IOException e) {
            logger.error("Failed to start AI Engine", e);
        }
    }

    public synchronized void stopEngine() {
        if (process != null && process.isAlive()) {
            logger.info("Stopping AI Engine...");
            process.destroy();
            try {
                if (!process.waitFor(5, TimeUnit.SECONDS)) {
                    process.destroyForcibly();
                }
            } catch (InterruptedException e) {
                process.destroyForcibly();
                Thread.currentThread().interrupt();
            }
            logger.info("AI Engine stopped.");
        }
    }

    public boolean isEngineRunning() {
        try {
            URL url = java.net.URI.create("http://127.0.0.1:8000/docs").toURL();
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(1000);
            connection.setReadTimeout(1000);
            int responseCode = connection.getResponseCode();
            return responseCode == 200;
        } catch (IOException e) {
            return false;
        }
    }

    public boolean waitForEngine(int timeoutSeconds) {
        logger.info("Waiting for AI Engine to initialize...");
        for (int i = 0; i < timeoutSeconds; i++) {
            if (isEngineRunning()) {
                logger.info("AI Engine is ready.");
                return true;
            }
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                logger.warn("Wait for AI Engine interrupted.");
                Thread.currentThread().interrupt();
                return false;
            }
        }
        logger.error("AI Engine failed to start within {}s.", timeoutSeconds);
        return false;
    }
}
