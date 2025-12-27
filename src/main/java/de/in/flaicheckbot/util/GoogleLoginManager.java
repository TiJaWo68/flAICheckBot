package de.in.flaicheckbot.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.api.client.auth.oauth2.Credential;
import com.google.api.client.extensions.java6.auth.oauth2.AuthorizationCodeInstalledApp;
import com.google.api.client.extensions.jetty.auth.oauth2.LocalServerReceiver;
import com.google.api.client.googleapis.auth.oauth2.GoogleAuthorizationCodeFlow;
import com.google.api.client.googleapis.auth.oauth2.GoogleClientSecrets;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.client.util.store.FileDataStoreFactory;
import com.google.auth.Credentials;
import com.google.auth.oauth2.UserCredentials;

/**
 * Handles OAuth2 authentication for Google Cloud Vision services.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class GoogleLoginManager {
	private static final Logger logger = LogManager.getLogger(GoogleLoginManager.class);
	private static final JsonFactory JSON_FACTORY = GsonFactory.getDefaultInstance();
	private static final String TOKENS_DIRECTORY_PATH = ".tokens";
	private static final List<String> SCOPES = Collections
			.singletonList("https://www.googleapis.com/auth/cloud-platform");

	// Singleton or simply utility

	/**
	 * Performs login. If a 'client_secret.json' is not found, it prompts the user.
	 * Returns a Credential object compatible with Google Cloud
	 * Libraries.
	 */
	public static Credentials login() throws Exception {
		// 1. Load Client Secrets
		File secretFile = new File("client_secret.json");
		if (!secretFile.exists()) {
			// Try to find it or ask user
			File selected = askUserForClientSecret();
			if (selected == null) {
				throw new IOException("No client_secret.json provided.");
			}
			secretFile = selected;
		}

		final NetHttpTransport HTTP_TRANSPORT = GoogleNetHttpTransport.newTrustedTransport();

		GoogleClientSecrets clientSecrets;
		try (InputStream in = new FileInputStream(secretFile)) {
			clientSecrets = GoogleClientSecrets.load(JSON_FACTORY, new InputStreamReader(in));
		}

		// 2. Build Flow
		GoogleAuthorizationCodeFlow flow = new GoogleAuthorizationCodeFlow.Builder(HTTP_TRANSPORT, JSON_FACTORY,
				clientSecrets, SCOPES)
				.setDataStoreFactory(new FileDataStoreFactory(new java.io.File(TOKENS_DIRECTORY_PATH)))
				.setAccessType("offline").build();

		// 3. Authorize
		LocalServerReceiver receiver = new LocalServerReceiver.Builder().setPort(8888).build();
		Credential credential = new AuthorizationCodeInstalledApp(flow, receiver).authorize("user");

		logger.info("Credentials saved to {}", TOKENS_DIRECTORY_PATH);

		// 4. Adapt to Google Cloud Library Credentials (com.google.auth.Credentials)
		return UserCredentials.newBuilder().setClientId(clientSecrets.getDetails().getClientId())
				.setClientSecret(clientSecrets.getDetails().getClientSecret())
				.setRefreshToken(credential.getRefreshToken()).build();
	}

	/**
	 * Extracts the project_id from client_secret.json.
	 */
	public static String getProjectId() throws IOException {
		File secretFile = new File("client_secret.json");
		if (!secretFile.exists())
			return null;

		try (InputStream in = new FileInputStream(secretFile)) {
			// The project_id is often inside the 'installed' or 'web' object which
			// GoogleClientSecrets doesn't expose directly via Details.
			// Let's use jackson to get it since it's already a dependency.
			com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
			com.fasterxml.jackson.databind.JsonNode root = mapper.readTree(secretFile);
			com.fasterxml.jackson.databind.JsonNode top = root.has("installed") ? root.get("installed")
					: (root.has("web") ? root.get("web") : null);
			if (top != null && top.has("project_id")) {
				return top.get("project_id").asText();
			}
		}
		return null;
	}

	private static File askUserForClientSecret() {
		// This should be called on EDT usually, but login() might be background. Use
		// simple blocking dialog.
		javax.swing.JFileChooser chooser = new javax.swing.JFileChooser();
		chooser.setCurrentDirectory(new File("."));
		chooser.setDialogTitle("Select client_secret.json (OAuth Client ID)");
		chooser.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("JSON Files", "json"));
		int rc = chooser.showOpenDialog(null);
		if (rc == javax.swing.JFileChooser.APPROVE_OPTION) {
			return chooser.getSelectedFile();
		}
		return null;
	}
}
