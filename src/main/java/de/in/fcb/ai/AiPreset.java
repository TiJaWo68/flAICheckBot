package de.in.fcb.ai;

/**
 * Supported provider presets for the OpenAI-compatible AI endpoint (including
 * Google AI Studio / Gemini).
 */
public enum AiPreset {

    DISABLED("", "", false),
    GEMINI("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash", true),
    OPENAI("https://api.openai.com/v1", "gpt-4o-mini", true),
    OLLAMA("http://localhost:11434/v1", "qwen2.5vl:7b", false),
    CUSTOM("", "", false);

    private final String defaultBaseUrl;
    private final String defaultModel;
    private final boolean apiKeyRequired;

    AiPreset(String defaultBaseUrl, String defaultModel, boolean apiKeyRequired) {
        this.defaultBaseUrl = defaultBaseUrl;
        this.defaultModel = defaultModel;
        this.apiKeyRequired = apiKeyRequired;
    }

    public String defaultBaseUrl() {
        return defaultBaseUrl;
    }

    public String defaultModel() {
        return defaultModel;
    }

    public boolean isApiKeyRequired() {
        return apiKeyRequired;
    }

    public static AiPreset fromKey(String raw) {
        if (raw == null || raw.isBlank()) {
            return DISABLED;
        }
        try {
            return AiPreset.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException exception) {
            return DISABLED;
        }
    }
}
