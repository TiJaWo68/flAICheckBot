package de.in.flaicheckbot.util;

import java.util.ArrayList;
import java.util.List;

/**
 * Centrally manages the available languages and their ISO mappings.
 * 
 * @author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
 */
public class LanguageSelectionProvider {

	public static class LanguageInfo {
		private final String displayName;
		private final String isoCode;

		public LanguageInfo(String displayName, String isoCode) {
			this.displayName = displayName;
			this.isoCode = isoCode;
		}

		public String getDisplayName() {
			return displayName;
		}

		public String getIsoCode() {
			return isoCode;
		}

		@Override
		public String toString() {
			return displayName;
		}
	}

	private static final List<LanguageInfo> LANGUAGES = new ArrayList<>();

	static {
		LANGUAGES.add(new LanguageInfo("Englisch", "en"));
		LANGUAGES.add(new LanguageInfo("Deutsch", "de"));
		LANGUAGES.add(new LanguageInfo("Französisch", "fr"));
		LANGUAGES.add(new LanguageInfo("Spanisch", "es"));
	}

	public static List<LanguageInfo> getLanguages() {
		return new ArrayList<>(LANGUAGES);
	}

	public static String[] getDisplayNames() {
		return LANGUAGES.stream().map(LanguageInfo::getDisplayName).toArray(String[]::new);
	}

	public static String mapToIsoCode(String displayName) {
		if (displayName == null)
			return "de";

		// Handle direct ISO codes too just in case
		if (displayName.length() == 2)
			return displayName;

		for (LanguageInfo info : LANGUAGES) {
			if (info.getDisplayName().equalsIgnoreCase(displayName)) {
				return info.getIsoCode();
			}
		}

		// Fallbacks for common aliases
		if ("English".equalsIgnoreCase(displayName))
			return "en";
		if ("French".equalsIgnoreCase(displayName))
			return "fr";
		if ("Spanish".equalsIgnoreCase(displayName))
			return "es";

		return "de";
	}
}
