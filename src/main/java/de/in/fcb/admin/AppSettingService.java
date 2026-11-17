package de.in.fcb.admin;

import java.util.Optional;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import de.in.fcb.ai.AiConfig;
import de.in.fcb.ai.AiPreset;

/**
 * Service for accessing and updating global application configuration settings.
 */
@Service
public class AppSettingService {

    public static final String KEY_AI_PRESET = "ai_preset";
    public static final String KEY_AI_BASE_URL = "ai_base_url";
    public static final String KEY_AI_MODEL = "ai_model";
    public static final String KEY_AI_API_KEY = "ai_api_key";

    private final AppSettingRepository settingRepository;

    public AppSettingService(AppSettingRepository settingRepository) {
        this.settingRepository = settingRepository;
    }

    @Transactional(readOnly = true)
    public AiConfig getAiConfig() {
        AiPreset preset = AiPreset.fromKey(getSetting(KEY_AI_PRESET).orElse(null));
        String baseUrl = getSetting(KEY_AI_BASE_URL).orElse(preset.defaultBaseUrl());
        String model = getSetting(KEY_AI_MODEL).orElse(preset.defaultModel());
        String apiKey = getSetting(KEY_AI_API_KEY).orElse("");
        return new AiConfig(preset, baseUrl, model, apiKey);
    }

    @Transactional
    public void setAiConfig(AiConfig config) {
        AiConfig effective = config == null ? AiConfig.disabled() : config;
        AiPreset preset = effective.preset() == null ? AiPreset.DISABLED : effective.preset();
        saveSetting(KEY_AI_PRESET, preset.name(), "Provider preset for AI integration");
        saveSetting(KEY_AI_BASE_URL, effective.normalizedBaseUrl(), "OpenAI-compatible AI API base URL");
        saveSetting(KEY_AI_MODEL, effective.normalizedModel(), "AI model identifier");
        saveSetting(KEY_AI_API_KEY, effective.apiKey() == null ? "" : effective.apiKey().trim(),
                "API key for AI endpoint");
    }

    @Transactional(readOnly = true)
    public Optional<String> getSetting(String key) {
        return settingRepository.findBySettingKey(key).map(AppSetting::getSettingValue);
    }

    @Transactional
    public void saveSetting(String key, String value, String description) {
        if (key == null || key.isBlank()) {
            throw new IllegalArgumentException("Setting key must not be blank");
        }
        if (value == null) {
            throw new IllegalArgumentException("Setting value must not be null");
        }
        AppSetting setting = settingRepository.findBySettingKey(key)
                .orElseGet(() -> new AppSetting(key, value, description));
        setting.setSettingValue(value);
        if (description != null) {
            setting.setDescription(description);
        }
        settingRepository.save(setting);
    }
}
