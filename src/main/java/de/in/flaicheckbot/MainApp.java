package de.in.flaicheckbot;

import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.formdev.flatlaf.FlatDarculaLaf;

import de.in.flaicheckbot.ai.AiProcessManager;
import de.in.flaicheckbot.db.DatabaseManager;
import de.in.flaicheckbot.ui.EvaluationPanel;
import de.in.flaicheckbot.ui.TestDefinitionPanel;
import de.in.flaicheckbot.ui.TrainingPanel;
import de.in.utils.Log4jTools;
import de.in.utils.Version;

/**
 * Central application class that initializes core services (DB, AI Engine) and sets up the main window and navigation.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class MainApp {

	private static final String GROUPID = "de.in.flaicheckbot";
	private static final String ARTIFACTID = "flAICheckBot";

	private static final Logger LOGGER = LogManager.getLogger(MainApp.class);
	private static com.google.auth.Credentials credentials;
	private static String projectId;

	private static final List<Consumer<Boolean>> loginListeners = new ArrayList<>();

	public static void addLoginListener(Consumer<Boolean> listener) {
		loginListeners.add(listener);
		listener.accept(credentials != null);
	}

	public static com.google.auth.Credentials getCredentials() {
		return credentials;
	}

	public static void setCredentials(com.google.auth.Credentials creds) {
		credentials = creds;
		for (Consumer<Boolean> listener : loginListeners) {
			listener.accept(credentials != null);
		}
	}

	public static String getProjectId() {
		return projectId;
	}

	public static void setProjectId(String id) {
		projectId = id;
	}

	public static void main(String[] args) {
		Log4jTools.redirectStdOutErrLog();
		Log4jTools.logEnvironment(LOGGER);
		LOGGER.info("flAICheckBot " + Version.retrieveVersionFromPom(GROUPID, ARTIFACTID) + " started");

		FlatDarculaLaf.setup();

		// UI State Components
		final JLabel lblEngineStatus = new JLabel("KI-Engine: Startet...", SwingConstants.RIGHT);
		lblEngineStatus.setBorder(BorderFactory.createEmptyBorder(2, 5, 2, 10));
		lblEngineStatus.setForeground(Color.GRAY);
		lblEngineStatus.setFont(new Font("SansSerif", Font.PLAIN, 11));

		final JButton btnStart = new JButton("Initialisiere Datenbank...");
		btnStart.setEnabled(false);
		btnStart.setFont(new Font("SansSerif", Font.PLAIN, 18));
		btnStart.setPreferredSize(new Dimension(250, 50));

		// 1. Start AI Engine (Parallel)
		final AiProcessManager aiManager = new AiProcessManager();
		CompletableFuture.runAsync(() -> {
			aiManager.startEngine();
			boolean ready = aiManager.waitForEngine(30);
			SwingUtilities.invokeLater(() -> {
				if (ready) {
					lblEngineStatus.setText("KI-Engine: Bereit ✅");
					lblEngineStatus.setForeground(new Color(0, 150, 0));
				} else {
					lblEngineStatus.setText("KI-Engine: Fehler beim Start ❌");
					lblEngineStatus.setForeground(Color.RED);
				}
			});
		});

		// 2. Start Database (Parallel)
		CompletableFuture<DatabaseManager> dbFuture = CompletableFuture.supplyAsync(() -> {
			LOGGER.info("Initializing Database...");
			return new DatabaseManager("flaicheckbot.db");
		});

		// UI Creation (EDT)
		SwingUtilities.invokeLater(() -> {
			JFrame frame = new JFrame("flAICheckBot - Foreign Language AI Check Bot");
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			frame.setSize(800, 600);
			frame.setLocationRelativeTo(null);

			// App Icon
			try {
				Image icon = new ImageIcon(MainApp.class.getResource("/images/app_icon.png")).getImage();
				frame.setIconImage(icon);
			} catch (Exception e) {
				LOGGER.warn("Could not load app icon", e);
			}

			JPanel mainPanel = new JPanel(new CardLayout());

			// Welcome Page
			BackgroundPanel welcomePanel = new BackgroundPanel("/images/welcome_bg.png");
			JLabel label = new JLabel("Willkommen bei flAICheckBot", SwingConstants.CENTER);
			label.setFont(new Font("SansSerif", Font.BOLD, 36));
			label.setForeground(Color.WHITE);
			// Semi-transparent overlay for better text readability
			label.setBackground(new Color(0, 0, 0, 100));
			label.setOpaque(true);

			welcomePanel.add(label, BorderLayout.CENTER);

			JPanel southPanel = new JPanel();
			southPanel.setOpaque(false);
			southPanel.add(btnStart);
			welcomePanel.add(southPanel, BorderLayout.SOUTH);

			mainPanel.add(welcomePanel, "welcome");

			frame.setLayout(new BorderLayout());
			frame.add(mainPanel, BorderLayout.CENTER);
			frame.add(lblEngineStatus, BorderLayout.SOUTH);

			// Centralized Menu Bar and Title Bar Integration
			frame.getRootPane().putClientProperty("flatlaf.menuBarEmbedded", true);
			javax.swing.JMenuBar menuBar = new javax.swing.JMenuBar();
			javax.swing.JMenu accountMenu = new javax.swing.JMenu("Account");
			javax.swing.JMenuItem loginItem = new javax.swing.JMenuItem("Google Login...");
			loginItem.addActionListener(e -> performLogin(frame));
			accountMenu.add(loginItem);
			menuBar.add(accountMenu);
			frame.setJMenuBar(menuBar);

			frame.setVisible(true);

			// Handle DB completion to setup WorkflowUI
			dbFuture.thenAcceptAsync(dbManager -> {
				SwingUtilities.invokeLater(() -> {
					LOGGER.info("Database ready, initializing UI components...");
					JTabbedPane tabbedPane = new JTabbedPane();
					tabbedPane.addTab("Training", new TrainingPanel(dbManager));
					tabbedPane.addTab("Test-Definition", new TestDefinitionPanel(dbManager));
					tabbedPane.addTab("Bewertung", new EvaluationPanel(dbManager));

					mainPanel.add(tabbedPane, "app");

					btnStart.setText("Anwendung starten");
					btnStart.setEnabled(true);
					btnStart.addActionListener(e -> {
						CardLayout cl = (CardLayout) mainPanel.getLayout();
						cl.show(mainPanel, "app");
						frame.setSize(1080, 800);
						frame.setLocationRelativeTo(null);
					});
				});
			}).exceptionally(ex -> {
				LOGGER.error("Failed to initialize database", ex);
				SwingUtilities.invokeLater(() -> {
					btnStart.setText("Datenbankfehler! ❌");
					btnStart.setForeground(Color.RED);
				});
				return null;
			});
		});
	}

	public static void performLogin(java.awt.Window parent) {
		new Thread(() -> {
			try {
				com.google.auth.Credentials creds = de.in.flaicheckbot.util.GoogleLoginManager.login();
				setCredentials(creds);
				String pid = de.in.flaicheckbot.util.GoogleLoginManager.getProjectId();
				setProjectId(pid);
				LOGGER.info("Login successful. Project ID: {}", pid);
				SwingUtilities.invokeLater(() -> {
					javax.swing.JOptionPane.showMessageDialog(parent, "Erfolgreich bei Google angemeldet!\nProjekt: " + pid, "Login",
							javax.swing.JOptionPane.INFORMATION_MESSAGE);
				});
			} catch (Exception e) {
				LOGGER.error("Login failed", e);
				SwingUtilities.invokeLater(() -> {
					de.in.utils.gui.ExceptionMessage.show(parent, "Fehler", "Login fehlgeschlagen", e);
				});
			}
		}).start();
	}

	private static class BackgroundPanel extends JPanel {
		private Image backgroundImage;

		public BackgroundPanel(String resourcePath) {
			setLayout(new BorderLayout());
			try {
				var res = getClass().getResource(resourcePath);
				if (res != null) {
					backgroundImage = new ImageIcon(res).getImage();
				}
			} catch (Exception e) {
				LOGGER.error("Failed to load background image: " + resourcePath, e);
			}
		}

		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);
			if (backgroundImage != null) {
				g.drawImage(backgroundImage, 0, 0, getWidth(), getHeight(), this);
			}
		}
	}
}
