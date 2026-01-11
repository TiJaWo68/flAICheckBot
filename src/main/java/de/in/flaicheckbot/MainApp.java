package de.in.flaicheckbot;

import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
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
import com.formdev.flatlaf.extras.FlatSVGIcon;

import de.in.flaicheckbot.ai.AiProcessManager;
import de.in.flaicheckbot.db.DatabaseManager;
import de.in.flaicheckbot.ui.EvaluationPanel;
import de.in.flaicheckbot.ui.TestDefinitionPanel;
import de.in.flaicheckbot.ui.TrainingPanel;
import de.in.flaicheckbot.ui.AboutActionWrapper;
import de.in.flaicheckbot.ui.DevPanel;
import de.in.utils.Log4jTools;
import de.in.utils.Version;

/**
 * Central application class that initializes core services (DB, AI Engine) and
 * sets up the main window and navigation.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class MainApp {

	private static final String GROUPID = "de.in.flaicheckbot";
	private static final String ARTIFACTID = "flAICheckBot";

	private static final Logger LOGGER = LogManager.getLogger(MainApp.class);
	private static com.google.auth.Credentials credentials;
	private static String projectId;
	private static String geminiApiKey;
	private static boolean preferApiKey = true; // Default to true for ease of use

	private static final String SETTING_WINDOW_X = "window_x";
	private static final String SETTING_WINDOW_Y = "window_y";
	private static final String SETTING_WINDOW_WIDTH = "window_width";
	private static final String SETTING_WINDOW_HEIGHT = "window_height";
	private static final String SETTING_WINDOW_MAXIMIZED = "window_maximized";
	private static final boolean IS_LINUX = System.getProperty("os.name").toLowerCase().contains("linux");

	private static boolean windowStateLoaded = false;

	public static class AuthStatus {
		public final boolean oauthLoggedIn;
		public final boolean apiKeyProvided;

		public AuthStatus(boolean oauthLoggedIn, boolean apiKeyProvided) {
			this.oauthLoggedIn = oauthLoggedIn;
			this.apiKeyProvided = apiKeyProvided;
		}

		public boolean isAnyAvailable() {
			return oauthLoggedIn || apiKeyProvided;
		}
	}

	private static final List<Consumer<AuthStatus>> authListeners = new ArrayList<>();

	public static void addAuthListener(Consumer<AuthStatus> listener) {
		authListeners.add(listener);
		listener.accept(getAuthStatus());
	}

	private static AuthStatus getAuthStatus() {
		return new AuthStatus(credentials != null, geminiApiKey != null && !geminiApiKey.isEmpty());
	}

	public static com.google.auth.Credentials getCredentials() {
		return credentials;
	}

	public static void setCredentials(com.google.auth.Credentials creds) {
		credentials = creds;
		notifyAuthListeners();
	}

	public static String getGeminiApiKey() {
		return geminiApiKey;
	}

	public static void setGeminiApiKey(String key) {
		geminiApiKey = key;
		notifyAuthListeners();
	}

	public static boolean isPreferApiKey() {
		return preferApiKey;
	}

	public static void setPreferApiKey(boolean prefer) {
		preferApiKey = prefer;
		notifyAuthListeners();
	}

	private static void notifyAuthListeners() {
		AuthStatus status = getAuthStatus();
		for (Consumer<AuthStatus> listener : authListeners) {
			listener.accept(status);
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

		// Parse command line arguments
		String authArg = null;
		boolean devMode = false;
		String devFile = null;
		for (String arg : args) {
			if (arg.startsWith("--auth=")) {
				authArg = arg.substring("--auth=".length());
			} else if (arg.equals("--dev")) {
				devMode = true;
			} else if (arg.startsWith("--dev=")) {
				devMode = true;
				devFile = arg.substring("--dev=".length());
			}
		}

		final String finalAuthArg = authArg;
		final boolean finalDevMode = devMode;
		final String finalDevFile = devFile;

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
		CompletableFuture<Void> setupFuture = CompletableFuture.runAsync(() -> {
			try {
				if (aiManager.needsSetup()) {
					LOGGER.info("Starting automatic Python setup...");
					aiManager.performSetup((msg, pct) -> {
						SwingUtilities.invokeLater(() -> {
							lblEngineStatus.setText("KI-Setup: " + msg + " (" + pct + "%)");
						});
					});
				}
				aiManager.startEngine();
				boolean ready = aiManager.waitForEngine(60);
				SwingUtilities.invokeLater(() -> {
					if (ready) {
						// Fetch detailed status (device info)
						try {
							AIEngineClient client = new AIEngineClient();
							AIEngineClient.EngineStatus status = client.getStatus().get(); // Block briefly as we are in
																							// background thread

							// Version Check
							final String REQUIRED_VERSION = "0.2.2";
							boolean compatible = de.in.flaicheckbot.utils.BackendCompatibilityChecker
									.isCompatible(REQUIRED_VERSION, status.version);

							if (!compatible) {
								String msg = "Inkompatible Backend-Version: " + status.version + " (Erwartet: "
										+ REQUIRED_VERSION + ")";
								LOGGER.error(msg);
								SwingUtilities.invokeLater(() -> {
									lblEngineStatus.setText(msg);
									lblEngineStatus.setForeground(Color.RED);
									// Disable start button essentially (it remains disabled)
								});
								return;
							}

							if (status != null && status.device != null) {
								String deviceText = status.device;
								if ("CUDA".equalsIgnoreCase(status.device)) {
									deviceText = "Powered by CUDA";
								} else if ("MPS".equalsIgnoreCase(status.device)) {
									deviceText = "Powered by Apple Silicon";
								} else {
									// For CPU, maybe show the model or brand
									if (status.deviceName != null && !status.deviceName.isEmpty()) {
										deviceText = status.deviceName;
									} else {
										deviceText = "Powered by CPU";
									}
								}

								// Truncate if too long
								if (deviceText.length() > 40) {
									deviceText = deviceText.substring(0, 37) + "...";
								}

								final String finalText = "KI-Engine: Bereit (" + deviceText + ", v" + status.version
										+ ")";
								final String iconName = status.deviceIcon;

								SwingUtilities.invokeLater(() -> {
									lblEngineStatus.setText(finalText);
									lblEngineStatus.setForeground(new Color(0, 150, 0));

									// Try to load icon
									if (iconName != null) {
										try {
											// Load SVG icon (16x16)
											lblEngineStatus.setIcon(
													new FlatSVGIcon("images/icon_" + iconName + ".svg", 16, 16));
											lblEngineStatus.setHorizontalTextPosition(SwingConstants.LEFT);
											lblEngineStatus.setIconTextGap(8);
										} catch (Exception ex) {
											LOGGER.warn("Could not load device icon: " + iconName, ex);
										}
									}
								});
							} else {
								SwingUtilities.invokeLater(() -> {
									lblEngineStatus.setText("KI-Engine: Bereit ✅");
									lblEngineStatus.setForeground(new Color(0, 150, 0));
								});
							}
						} catch (Exception ex) {
							LOGGER.warn("Failed to fetch detailed status", ex);
							SwingUtilities.invokeLater(() -> {
								lblEngineStatus.setText("KI-Engine: Bereit ✅");
								lblEngineStatus.setForeground(new Color(0, 150, 0));
							});
						}
					} else {
						lblEngineStatus.setText("KI-Engine: Fehler beim Start ❌");
						lblEngineStatus.setForeground(Color.RED);
					}
				});
			} catch (Exception e) {
				LOGGER.error("AI Engine initialization failed", e);
				SwingUtilities.invokeLater(() -> {
					lblEngineStatus.setText("KI-Engine: Setup-Fehler ❌");
					lblEngineStatus.setForeground(Color.RED);
				});
				throw new RuntimeException(e);
			}
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
				Image icon = new ImageIcon(MainApp.class.getResource("/images/app_icon.jpg")).getImage();
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
			// Only disable on Linux to fix multi-monitor clamping
			frame.getRootPane().putClientProperty("flatlaf.menuBarEmbedded", !IS_LINUX);
			javax.swing.JMenuBar menuBar = new javax.swing.JMenuBar();
			javax.swing.JMenu accountMenu = new javax.swing.JMenu("Account");
			javax.swing.JMenuItem loginItem = new javax.swing.JMenuItem("Google Login (OAuth)...");
			loginItem.addActionListener(e -> performLogin(frame));
			accountMenu.add(loginItem);

			javax.swing.JMenuItem apiKeyItem = new javax.swing.JMenuItem("Gemini API Key eingeben...");
			apiKeyItem.addActionListener(e -> performApiKeyEntry(frame, dbFuture.join()));
			accountMenu.add(apiKeyItem);

			accountMenu.addSeparator();

			javax.swing.JCheckBoxMenuItem preferApiKeyItem = new javax.swing.JCheckBoxMenuItem("API Key bevorzugen");
			preferApiKeyItem.setSelected(preferApiKey);
			preferApiKeyItem.addActionListener(e -> {
				setPreferApiKey(preferApiKeyItem.isSelected());
				try {
					dbFuture.join().setSetting("prefer_api_key", String.valueOf(preferApiKeyItem.isSelected()));
				} catch (Exception ex) {
					LOGGER.error("Failed to save preference", ex);
				}
			});
			addAuthListener(status -> {
				SwingUtilities.invokeLater(() -> {
					preferApiKeyItem.setSelected(preferApiKey);
				});
			});
			accountMenu.add(preferApiKeyItem);

			menuBar.add(accountMenu);
			menuBar.add(javax.swing.Box.createHorizontalGlue());

			javax.swing.JMenu helpMenu = new javax.swing.JMenu("Hilfe");
			javax.swing.JMenuItem instructionsItem = new javax.swing.JMenuItem("Anleitung");
			instructionsItem.addActionListener(e -> {
				javax.swing.JOptionPane.showMessageDialog(frame,
						"<html><b>Anleitung flAICheckBot</b><br><br>"
								+ "1. Starten Sie die Anwendung.<br>"
								+ "2. Wählen Sie im Tab 'Training' Dokumente aus.<br>"
								+ "3. Definieren Sie im Tab 'Test-Definition' Ihre Anforderungen.<br>"
								+ "4. Bewerten Sie die Ergebnisse im Tab 'Bewertung'.<br><br>"
								+ "Weitere Informationen finden Sie in der Dokumentation.</html>",
						"Anleitung", javax.swing.JOptionPane.INFORMATION_MESSAGE);
			});
			helpMenu.add(instructionsItem);

			helpMenu.addSeparator();

			AboutActionWrapper aboutAction = new AboutActionWrapper(frame, "TiJaWo68",
					"https://github.com/TiJaWo68/flAICheckBot", "flAiCheckBot", 2025, GROUPID, ARTIFACTID,
					frame.getIconImage());
			javax.swing.JMenuItem aboutItem = new javax.swing.JMenuItem(aboutAction);
			aboutItem.setText("Über flAiCheckBot");
			helpMenu.add(aboutItem);

			menuBar.add(helpMenu);

			frame.setJMenuBar(menuBar);

			frame.setVisible(true);

			// Handle DB and AI completion to setup WorkflowUI
			CompletableFuture.allOf(dbFuture, setupFuture).thenAcceptAsync(v -> {
				DatabaseManager dbManager = dbFuture.join();

				// Save window state on close
				frame.addWindowListener(new WindowAdapter() {
					@Override
					public void windowClosing(WindowEvent e) {
						saveWindowState(frame, dbManager, welcomePanel);
					}
				});

				// Load Gemini API Key from DB
				try {
					String key = dbManager.getSetting("gemini_api_key");
					if (key != null && !key.isEmpty()) {
						setGeminiApiKey(key);
						LOGGER.info("Gemini API Key loaded from database.");
					}

					String prefer = dbManager.getSetting("prefer_api_key");
					if (prefer != null) {
						setPreferApiKey(Boolean.parseBoolean(prefer));
						LOGGER.info("Auth preference loaded from database: preferApiKey={}", prefer);
					}

					// Override with command line arguments if provided
					if ("google".equalsIgnoreCase(finalAuthArg)) {
						LOGGER.info("Command line override: Using Google Login.");
						setPreferApiKey(false);
						// Automatic login will be handled below
					} else if ("apikey".equalsIgnoreCase(finalAuthArg)) {
						LOGGER.info("Command line override: Using Gemini API Key.");
						setPreferApiKey(true);
					}
				} catch (Exception e) {
					LOGGER.warn("Failed to load settings from DB", e);
				}

				SwingUtilities.invokeLater(() -> {
					LOGGER.info("Core systems ready, initializing UI components...");
					JTabbedPane tabbedPane = new JTabbedPane();
					if (finalDevMode) {
						DevPanel devPanel = new DevPanel(dbManager);
						tabbedPane.addTab("Dev", devPanel);
						if (finalDevFile != null) {
							devPanel.loadImage(new java.io.File(finalDevFile));
						}
					}
					tabbedPane.addTab("Training", new TrainingPanel(dbManager));
					tabbedPane.addTab("Test-Definition", new TestDefinitionPanel(dbManager));
					tabbedPane.addTab("Bewertung", new EvaluationPanel(dbManager));

					mainPanel.add(tabbedPane, "app");

					btnStart.setText("Anwendung starten");
					btnStart.setEnabled(true);

					// Automatic login if requested
					if ("google".equalsIgnoreCase(finalAuthArg)) {
						LOGGER.info("Triggering automatic Google Login...");
						performLogin(frame);
					}

					btnStart.addActionListener(e -> {
						CardLayout cl = (CardLayout) mainPanel.getLayout();
						cl.show(mainPanel, "app");
						loadWindowState(frame, dbManager);
						if (!windowStateLoaded) {
							frame.setSize(1080, 800);
							frame.setLocationRelativeTo(null);
						}
					});
				});
			}).exceptionally(ex -> {
				LOGGER.error("Failed to initialize core systems", ex);
				SwingUtilities.invokeLater(() -> {
					btnStart.setText("Initialisierungsfehler! ❌");
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
			} catch (Exception e) {
				LOGGER.error("Login failed", e);
				SwingUtilities.invokeLater(() -> {
					de.in.utils.gui.ExceptionMessage.show(parent, "Fehler", "Login fehlgeschlagen", e);
				});
			}
		}).start();
	}

	public static void performApiKeyEntry(java.awt.Window parent, DatabaseManager dbManager) {
		String current = "";
		try {
			current = dbManager.getSetting("gemini_api_key");
		} catch (Exception e) {
		}

		String key = javax.swing.JOptionPane.showInputDialog(parent,
				"Bitte geben Sie Ihren Gemini API Key ein (von aistudio.google.com):",
				current);

		if (key != null) {
			try {
				dbManager.setSetting("gemini_api_key", key.trim());
				setGeminiApiKey(key.trim());
				javax.swing.JOptionPane.showMessageDialog(parent, "API Key gespeichert.", "Gemini API",
						javax.swing.JOptionPane.INFORMATION_MESSAGE);
			} catch (Exception e) {
				LOGGER.error("Failed to save API Key", e);
				de.in.utils.gui.ExceptionMessage.show(parent, "Fehler", "Speichern fehlgeschlagen", e);
			}
		}
	}

	private static void loadWindowState(JFrame frame, DatabaseManager dbManager) {
		try {
			String xStr = dbManager.getSetting(SETTING_WINDOW_X);
			String yStr = dbManager.getSetting(SETTING_WINDOW_Y);
			String wStr = dbManager.getSetting(SETTING_WINDOW_WIDTH);
			String hStr = dbManager.getSetting(SETTING_WINDOW_HEIGHT);
			String maxStr = dbManager.getSetting(SETTING_WINDOW_MAXIMIZED);

			if (xStr != null && yStr != null && wStr != null && hStr != null) {
				int x = Integer.parseInt(xStr);
				int y = Integer.parseInt(yStr);
				int w = Integer.parseInt(wStr);
				int h = Integer.parseInt(hStr);
				LOGGER.info("Restoring window bounds: x={}, y={}, w={}, h={}, max={}", x, y, w, h, maxStr);

				// Use invokeLater to ensure this happens after the current layout cycle
				SwingUtilities.invokeLater(() -> {
					Dimension oldMin = frame.getMinimumSize();

					if (IS_LINUX) {
						// Aggressive restoration for Linux: temporarily set minimum size to allow large
						// bounds
						frame.setMinimumSize(new Dimension(w, h));
					}

					frame.setBounds(x, y, w, h);

					if (Boolean.parseBoolean(maxStr)) {
						frame.setExtendedState(JFrame.MAXIMIZED_BOTH);
					}

					frame.revalidate();
					frame.repaint();

					if (IS_LINUX) {
						// Revert minimum size after a short delay on Linux
						javax.swing.Timer timer = new javax.swing.Timer(500, ev -> {
							frame.setMinimumSize(oldMin);
						});
						timer.setRepeats(false);
						timer.start();
					}
				});
				windowStateLoaded = true;
			} else {
				LOGGER.info("No saved window state found in database.");
			}
		} catch (Exception e) {
			LOGGER.warn("Failed to load window state", e);
		}
	}

	private static void saveWindowState(JFrame frame, DatabaseManager dbManager, JPanel welcomePanel) {
		if (welcomePanel.isVisible()) {
			LOGGER.info("Not saving window state because Welcome Screen is active.");
			return;
		}
		try {
			boolean maximized = (frame.getExtendedState() & JFrame.MAXIMIZED_BOTH) != 0;
			dbManager.setSetting(SETTING_WINDOW_MAXIMIZED, String.valueOf(maximized));

			// Always save bounds, even if maximized (though they might be the maximized
			// bounds)
			int x = frame.getX();
			int y = frame.getY();
			int w = frame.getWidth();
			int h = frame.getHeight();
			LOGGER.info("Saving window state: x={}, y={}, w={}, h={}, maximized={}", x, y, w, h, maximized);

			dbManager.setSetting(SETTING_WINDOW_X, String.valueOf(x));
			dbManager.setSetting(SETTING_WINDOW_Y, String.valueOf(y));
			dbManager.setSetting(SETTING_WINDOW_WIDTH, String.valueOf(w));
			dbManager.setSetting(SETTING_WINDOW_HEIGHT, String.valueOf(h));
		} catch (Exception e) {
			LOGGER.error("Failed to save window state", e);
		}
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
