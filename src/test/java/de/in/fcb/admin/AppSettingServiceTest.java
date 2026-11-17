package de.in.fcb.admin;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.Optional;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import de.in.fcb.ai.AiConfig;
import de.in.fcb.ai.AiPreset;

class AppSettingServiceTest {

    private AppSettingRepository repository;
    private AppSettingService service;

    @BeforeEach
    void setUp() {
        repository = mock(AppSettingRepository.class);
        service = new AppSettingService(repository);
    }

    @Test
    void returnsDisabledAiConfigWhenUnset() {
        when(repository.findBySettingKey(any())).thenReturn(Optional.empty());

        AiConfig config = service.getAiConfig();

        assertThat(config.preset()).isEqualTo(AiPreset.DISABLED);
        assertThat(config.isConfigured()).isFalse();
    }

    @Test
    void returnsStoredAiConfig() {
        when(repository.findBySettingKey(AppSettingService.KEY_AI_PRESET))
                .thenReturn(Optional.of(new AppSetting(AppSettingService.KEY_AI_PRESET, "GEMINI", "")));
        when(repository.findBySettingKey(AppSettingService.KEY_AI_BASE_URL))
                .thenReturn(Optional.of(new AppSetting(AppSettingService.KEY_AI_BASE_URL,
                        "https://generativelanguage.googleapis.com/v1beta/openai", "")));
        when(repository.findBySettingKey(AppSettingService.KEY_AI_MODEL))
                .thenReturn(Optional.of(new AppSetting(AppSettingService.KEY_AI_MODEL, "gemini-2.5-flash", "")));
        when(repository.findBySettingKey(AppSettingService.KEY_AI_API_KEY))
                .thenReturn(Optional.of(new AppSetting(AppSettingService.KEY_AI_API_KEY, "secret-key", "")));

        AiConfig config = service.getAiConfig();

        assertThat(config.preset()).isEqualTo(AiPreset.GEMINI);
        assertThat(config.model()).isEqualTo("gemini-2.5-flash");
        assertThat(config.isConfigured()).isTrue();
    }

    @Test
    void savesAiConfigEntries() {
        when(repository.findBySettingKey(any())).thenReturn(Optional.empty());

        service.setAiConfig(new AiConfig(
                AiPreset.GEMINI,
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                "gemini-2.5-flash",
                " secret "));

        verify(repository, times(4)).save(any(AppSetting.class));
    }

    @Test
    void getSettingAndSaveSetting() {
        when(repository.findBySettingKey("custom.key"))
                .thenReturn(Optional.of(new AppSetting("custom.key", "foo", "bar")));

        Optional<String> val = service.getSetting("custom.key");

        assertThat(val).contains("foo");

        service.saveSetting("custom.key", "baz", "new desc");
        verify(repository).save(any(AppSetting.class));
    }

    @Test
    void rejectsBlankKeyOrNullValue() {
        assertThatThrownBy(() -> service.saveSetting(" ", "val", "desc"))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> service.saveSetting("key", null, "desc"))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
