package de.in.flaicheckbot.ai;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.net.URL;
import java.nio.file.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Automates the download and setup of a portable Python environment on Windows.
 * 
 * @author Antigravity
 */
public class PythonEnvironmentManager {
    private static final Logger logger = LogManager.getLogger(PythonEnvironmentManager.class);

    private static final String PYTHON_VERSION = "3.12.3";
    private static final String PYTHON_URL = "https://www.python.org/ftp/python/" + PYTHON_VERSION + "/python-"
            + PYTHON_VERSION + "-embed-amd64.zip";
    private static final String GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py";

    private final File installDir;
    private final File requirementsFile;

    public PythonEnvironmentManager(File baseDir, File requirementsFile) {
        this.installDir = new File(baseDir, ".portable_python");
        this.requirementsFile = requirementsFile;
    }

    public boolean isReady() {
        File pythonExec = new File(installDir, "python.exe");
        return pythonExec.exists();
    }

    public String getPythonPath() {
        return new File(installDir, "python.exe").getAbsolutePath();
    }

    public void setup(ProgressListener listener) throws Exception {
        if (!System.getProperty("os.name").toLowerCase().contains("win")) {
            logger.info("Automatic Python setup only supported on Windows.");
            return;
        }

        if (isReady()) {
            logger.info("Portable Python is already installed at {}", installDir.getAbsolutePath());
            return;
        }

        if (listener != null)
            listener.onProgress("Erstelle Verzeichnis...", 5);
        if (!installDir.exists()) {
            installDir.mkdirs();
        }

        // 1. Download and Extract Python
        File zipFile = new File(installDir, "python_embed.zip");
        if (listener != null)
            listener.onProgress("Lade Python herunter...", 10);
        downloadFile(PYTHON_URL, zipFile);

        if (listener != null)
            listener.onProgress("Entpacke Python...", 30);
        unzip(zipFile, installDir);
        zipFile.delete();

        // 2. Patch ._pth file
        if (listener != null)
            listener.onProgress("Konfiguriere Pfade...", 50);
        patchPthFile();

        // 3. Setup Pip
        File getPipScript = new File(installDir, "get-pip.py");
        if (listener != null)
            listener.onProgress("Initialisiere Pip...", 60);
        downloadFile(GET_PIP_URL, getPipScript);

        runPythonCommand(getPipScript.getAbsolutePath());
        getPipScript.delete();

        // 4. Install Requirements
        if (requirementsFile != null && requirementsFile.exists()) {
            if (listener != null)
                listener.onProgress("Installiere KI-Abhängigkeiten (dies kann dauern)...", 80);
            runPythonCommand("-m", "pip", "install", "-r", requirementsFile.getAbsolutePath());
        }

        if (listener != null)
            listener.onProgress("Konfiguration abgeschlossen.", 100);
        logger.info("Portable Python environment setup complete.");
    }

    private void patchPthFile() throws IOException {
        // Find python312._pth or similar
        File[] pthFiles = installDir.listFiles((dir, name) -> name.endsWith("._pth"));
        if (pthFiles != null && pthFiles.length > 0) {
            File pthFile = pthFiles[0];
            String content = Files.readString(pthFile.toPath());
            if (content.contains("#import site")) {
                content = content.replace("#import site", "import site");
                Files.writeString(pthFile.toPath(), content);
                logger.info("Patched {}", pthFile.getName());
            }
        }
    }

    private void downloadFile(String urlStr, File target) throws IOException {
        logger.info("Downloading {} to {}...", urlStr, target.getAbsolutePath());
        URL url = java.net.URI.create(urlStr).toURL();
        try (InputStream in = url.openStream()) {
            Files.copy(in, target.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private void unzip(File zipFile, File destDir) throws IOException {
        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(zipFile))) {
            ZipEntry entry = zis.getNextEntry();
            while (entry != null) {
                File newFile = new File(destDir, entry.getName());
                if (entry.isDirectory()) {
                    newFile.mkdirs();
                } else {
                    new File(newFile.getParent()).mkdirs();
                    try (FileOutputStream fos = new FileOutputStream(newFile)) {
                        zis.transferTo(fos);
                    }
                }
                zis.closeEntry();
                entry = zis.getNextEntry();
            }
        }
    }

    private void runPythonCommand(String... args) throws IOException, InterruptedException {
        String[] command = new String[args.length + 1];
        command[0] = getPythonPath();
        System.arraycopy(args, 0, command, 1, args.length);

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(installDir);
        pb.inheritIO();
        Process p = pb.start();
        int exitCode = p.waitFor();
        if (exitCode != 0) {
            throw new IOException("Python command failed with exit code " + exitCode);
        }
    }

    public interface ProgressListener {
        void onProgress(String message, int percent);
    }
}
