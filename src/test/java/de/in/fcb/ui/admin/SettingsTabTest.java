package de.in.fcb.ui.admin;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.vaadin.flow.component.UI;

import de.in.fcb.admin.AppSettingService;
import de.in.fcb.ai.AiClient;
import de.in.fcb.ai.AiConfig;
import de.in.fcb.ai.AiPreset;

class SettingsTabTest {

    private UI ui;
    private AppSettingService appSettingService;
    private SettingsTab settingsTab;

    @BeforeEach
    void setUp() {
        ui = new UI();
        UI.setCurrent(ui);
        appSettingService = mock(AppSettingService.class);
        when(appSettingService.getAiConfig()).thenReturn(AiConfig.disabled());
        settingsTab = new SettingsTab(appSettingService);
    }

    @AfterEach
    void tearDown() {
        UI.setCurrent(null);
        ui = null;
    }

    @Test
    void refreshesAiConfigFromService() {
        when(appSettingService.getAiConfig()).thenReturn(new AiConfig(
                AiPreset.GEMINI,
                "https://generativelanguage.googleapis.com/v1beta/openai",
                "gemini-2.5-flash",
                "secret"));

        settingsTab.refresh();

        assertThat(settingsTab.aiPreset.getValue()).isEqualTo(AiPreset.GEMINI);
        assertThat(settingsTab.aiModel.getValue()).isEqualTo("gemini-2.5-flash");
    }

    @Test
    void savesAiConfig() {
        settingsTab.aiPreset.setValue(AiPreset.OLLAMA);
        settingsTab.aiBaseUrl.setValue("http://localhost:11434/v1");
        settingsTab.aiModel.setValue("qwen2.5vl:7b");
        settingsTab.save();

        verify(appSettingService).setAiConfig(
                new AiConfig(AiPreset.OLLAMA, "http://localhost:11434/v1", "qwen2.5vl:7b", ""));
    }

    @Test
    void fetchesModelsIntoComboBoxAndPreservesSelection() {
        AiClient client = mock(AiClient.class);
        when(client.listModels(any())).thenReturn(List.of("gemini-2.5-pro", "gemini-3-flash"));

        SettingsTab tab = new SettingsTab(appSettingService, client);
        tab.aiPreset.setValue(AiPreset.GEMINI);
        tab.aiBaseUrl.setValue("https://generativelanguage.googleapis.com/v1beta/openai");
        tab.aiApiKey.setValue("secret");

        tab.fetchAiModels();

        assertThat(tab.aiModel.getListDataView().getItems().toList())
                .containsExactly("gemini-2.5-pro", "gemini-3-flash");
        assertThat(tab.aiModel.getValue()).isEqualTo("gemini-2.5-pro");
    }

    @Test
    void testConnectionFlagsUnavailableModelWhilePopulatingLiveList() {
        AiClient client = mock(AiClient.class);
        when(client.listModels(any())).thenReturn(List.of("gemini-3-flash", "gemini-3-pro"));

        SettingsTab tab = new SettingsTab(appSettingService, client);
        tab.aiPreset.setValue(AiPreset.GEMINI);
        tab.aiBaseUrl.setValue("https://generativelanguage.googleapis.com/v1beta/openai");
        tab.aiApiKey.setValue("secret");
        tab.setAiModelOptions(List.of("gemini-1.5-flash"), "gemini-1.5-flash");

        tab.testAiConnection();

        assertThat(tab.aiModel.getListDataView().getItems().toList())
                .contains("gemini-3-flash", "gemini-3-pro");
        assertThat(tab.aiModel.isInvalid()).isTrue();
    }
}
