package de.in.fcb.ai;

/**
 * Runtime configuration for OpenAI-compatible AI model endpoints (including
 * Google AI Studio / Gemini).
 */
public record AiConfig(
        AiPreset preset,
        String baseUrl,
        String model,
        String apiKey) {

    public static final String ENV_API_KEY = "FLAICHECKBOT_AI_API_KEY";

    public static AiConfig disabled() {
        return new AiConfig(AiPreset.DISABLED, "", "", "");
    }

    public boolean canQueryModels() {
        if (preset == null || preset == AiPreset.DISABLED) {
            return false;
        }
        if (normalizedBaseUrl().isBlank()) {
            return false;
        }
        return !preset.isApiKeyRequired() || !effectiveApiKey().isBlank();
    }

    public boolean isConfigured() {
        return canQueryModels() && !normalizedModel().isBlank();
    }

    public String normalizedBaseUrl() {
        if (baseUrl == null) {
            return "";
        }
        String trimmed = baseUrl.trim();
        while (trimmed.endsWith("/")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        return trimmed;
    }

    public String normalizedModel() {
        return model == null ? "" : model.trim();
    }

    public String effectiveApiKey() {
        if (apiKey != null && !apiKey.isBlank()) {
            return apiKey.trim();
        }
        String env = System.getenv(ENV_API_KEY);
        return env == null ? "" : env.trim();
    }
}
