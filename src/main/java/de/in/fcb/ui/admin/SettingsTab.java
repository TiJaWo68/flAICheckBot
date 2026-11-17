package de.in.fcb.ui.admin;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.springframework.beans.factory.annotation.Autowired;

import com.vaadin.flow.component.Component;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.formlayout.FormLayout;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Paragraph;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.PasswordField;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.admin.AppSettingService;
import de.in.fcb.ai.AiClient;
import de.in.fcb.ai.AiConfig;
import de.in.fcb.ai.AiPreset;

/**
 * Administration tab for configuring application-wide settings and AI
 * provider connectivity.
 */
public class SettingsTab extends VerticalLayout implements AdminTab {

    private final AppSettingService appSettingService;
    private final AiClient aiClient;

    final ComboBox<AiPreset> aiPreset = new ComboBox<>();
    final TextField aiBaseUrl = new TextField();
    final ComboBox<String> aiModel = new ComboBox<>();
    final Button fetchModelsButton = new Button();
    final PasswordField aiApiKey = new PasswordField();
    final Button testAiButton = new Button();
    final Button saveButton = new Button();
    private final HorizontalLayout modelRow = new HorizontalLayout();
    private List<String> availableModels = defaultPresetModels();

    public SettingsTab(AppSettingService appSettingService) {
        this(appSettingService, null);
    }

    @Autowired
    public SettingsTab(AppSettingService appSettingService, AiClient aiClient) {
        this.appSettingService = appSettingService;
        this.aiClient = aiClient;

        setSizeFull();
        setPadding(false);
        setSpacing(false);

        add(createHeader(), createForm());
    }

    @Override
    public String getLabelKey() {
        return "admin.tab.settings";
    }

    @Override
    public Component asComponent() {
        return this;
    }

    @Override
    public void refresh() {
        AiConfig aiConfig = appSettingService.getAiConfig();
        if (aiConfig != null) {
            aiPreset.setValue(aiConfig.preset() == null ? AiPreset.DISABLED : aiConfig.preset());
            aiBaseUrl.setValue(aiConfig.baseUrl() == null ? "" : aiConfig.baseUrl());
            setAiModelOptions(availableModels, aiConfig.model());
            aiModel.setInvalid(false);
            aiApiKey.setValue(aiConfig.apiKey() == null ? "" : aiConfig.apiKey());
            updateAiFieldStates(aiPreset.getValue());
        }
    }

    private VerticalLayout createHeader() {
        Span title = new Span(getTranslation("admin.settings.title"));
        title.addClassName("admin-tab-title");

        Paragraph description = new Paragraph(getTranslation("admin.settings.description"));
        description.addClassNames(LumoUtility.TextColor.SECONDARY, LumoUtility.Margin.Top.NONE);

        VerticalLayout header = new VerticalLayout(title, description);
        header.setPadding(false);
        header.setSpacing(false);
        return header;
    }

    private FormLayout createForm() {
        H3 aiHeading = new H3(getTranslation("admin.settings.ai.title"));
        aiHeading.addClassNames(LumoUtility.FontSize.MEDIUM, LumoUtility.Margin.Top.SMALL,
                LumoUtility.Margin.Bottom.XSMALL);

        aiPreset.setLabel(getTranslation("admin.settings.ai.preset"));
        aiPreset.setItems(AiPreset.values());
        aiPreset.setItemLabelGenerator(preset -> getTranslation("admin.settings.ai.preset." + preset.name()));
        aiPreset.setValue(AiPreset.DISABLED);
        aiPreset.setWidth("280px");
        aiPreset.addValueChangeListener(event -> {
            AiPreset selected = event.getValue() == null ? AiPreset.DISABLED : event.getValue();
            if (event.isFromClient() && selected != AiPreset.CUSTOM) {
                aiBaseUrl.setValue(selected.defaultBaseUrl());
                setAiModelOptions(availableModels, selected.defaultModel());
                aiModel.setInvalid(false);
            }
            updateAiFieldStates(selected);
        });

        aiBaseUrl.setLabel(getTranslation("admin.settings.ai.base-url"));
        aiBaseUrl.setWidth("420px");

        aiModel.setLabel(getTranslation("admin.settings.ai.model"));
        aiModel.setAllowCustomValue(true);
        aiModel.setItems(availableModels);
        aiModel.setWidth("320px");
        aiModel.addCustomValueSetListener(event -> {
            String custom = event.getDetail() == null ? "" : event.getDetail().trim();
            if (!custom.isBlank()) {
                setAiModelOptions(availableModels, custom);
                aiModel.setInvalid(false);
            }
        });
        aiModel.addValueChangeListener(event -> {
            if (event.isFromClient()) {
                aiModel.setInvalid(false);
            }
        });

        fetchModelsButton.setText(getTranslation("admin.settings.ai.models.fetch"));
        fetchModelsButton.setIcon(VaadinIcon.REFRESH.create());
        fetchModelsButton.addClickListener(event -> fetchAiModels());
        fetchModelsButton.setVisible(aiClient != null);

        modelRow.add(aiModel, fetchModelsButton);
        modelRow.setDefaultVerticalComponentAlignment(Alignment.BASELINE);

        aiApiKey.setLabel(getTranslation("admin.settings.ai.api-key"));
        aiApiKey.setHelperText(getTranslation("admin.settings.ai.api-key.hint"));
        aiApiKey.setWidth("420px");

        testAiButton.setText(getTranslation("admin.settings.ai.test"));
        testAiButton.setIcon(VaadinIcon.PLUG.create());
        testAiButton.addClickListener(event -> testAiConnection());
        testAiButton.setVisible(aiClient != null);

        saveButton.setText(getTranslation("admin.settings.button.save"));
        saveButton.setIcon(VaadinIcon.CHECK.create());
        saveButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        saveButton.addClickListener(event -> save());

        HorizontalLayout actions = new HorizontalLayout(saveButton, testAiButton);
        actions.setDefaultVerticalComponentAlignment(Alignment.BASELINE);

        FormLayout form = new FormLayout(
                aiHeading,
                aiPreset,
                aiBaseUrl,
                modelRow,
                aiApiKey,
                actions);
        form.setResponsiveSteps(new FormLayout.ResponsiveStep("0", 1));
        updateAiFieldStates(AiPreset.DISABLED);
        return form;
    }

    private void updateAiFieldStates(AiPreset preset) {
        boolean enabled = preset != null && preset != AiPreset.DISABLED;
        aiBaseUrl.setVisible(enabled);
        modelRow.setVisible(enabled);
        aiModel.setVisible(enabled);
        fetchModelsButton.setEnabled(enabled);
        aiApiKey.setVisible(enabled);
        testAiButton.setEnabled(enabled);
        if (!enabled) {
            aiModel.setInvalid(false);
        }
    }

    void setAiModelOptions(List<String> models, String preferredSelection) {
        availableModels = models == null ? List.of() : List.copyOf(models);
        Set<String> items = new LinkedHashSet<>(availableModels);
        String trimmedSelection = preferredSelection == null ? "" : preferredSelection.trim();
        if (!trimmedSelection.isBlank()) {
            items.add(trimmedSelection);
        }
        aiModel.setItems(items);
        if (!trimmedSelection.isBlank()) {
            aiModel.setValue(trimmedSelection);
        } else {
            aiModel.clear();
        }
    }

    void fetchAiModels() {
        if (aiClient == null) {
            return;
        }
        try {
            List<String> models = aiClient.listModels(currentAiConfig());
            if (models.isEmpty()) {
                Notification.show(getTranslation("admin.settings.ai.models.none"), 4000,
                        Notification.Position.MIDDLE);
                return;
            }
            String current = aiModel.getValue() == null ? "" : aiModel.getValue().trim();
            String matched = findMatchingModel(models, current);
            String nextSelection = matched != null ? matched : models.get(0);
            setAiModelOptions(models, nextSelection);
            aiModel.setInvalid(false);
            Notification.show(getTranslation("admin.settings.ai.models.fetched", models.size()), 3000,
                    Notification.Position.MIDDLE);
        } catch (RuntimeException exception) {
            Notification.show(getTranslation("admin.settings.ai.models.error", exception.getMessage()), 5000,
                    Notification.Position.MIDDLE);
        }
    }

    void testAiConnection() {
        if (aiClient == null) {
            return;
        }
        AiConfig config = currentAiConfig();
        try {
            List<String> models = aiClient.listModels(config);
            if (!models.isEmpty()) {
                String currentModel = config.normalizedModel();
                String matchedModel = findMatchingModel(models, currentModel);
                setAiModelOptions(models, matchedModel != null ? matchedModel : currentModel);
                if (matchedModel == null) {
                    String warning = getTranslation("admin.settings.ai.test.model-unavailable",
                            models.size(), currentModel);
                    aiModel.setErrorMessage(warning);
                    aiModel.setInvalid(true);
                    Notification.show(warning, 5000, Notification.Position.MIDDLE);
                    return;
                }
                aiModel.setInvalid(false);
                Notification.show(getTranslation("admin.settings.ai.test.ok.count", models.size()), 3000,
                        Notification.Position.MIDDLE);
                return;
            }
            aiModel.setInvalid(false);
            Notification.show(getTranslation("admin.settings.ai.test.ok"), 3000,
                    Notification.Position.MIDDLE);
        } catch (RuntimeException exception) {
            Notification.show(getTranslation("admin.settings.ai.test.error", exception.getMessage()), 5000,
                    Notification.Position.MIDDLE);
        }
    }

    private static String findMatchingModel(List<String> models, String candidate) {
        if (candidate == null || candidate.isBlank()) {
            return null;
        }
        return models.stream()
                .filter(entry -> entry.equalsIgnoreCase(candidate.trim()))
                .findFirst()
                .orElse(null);
    }

    private static List<String> defaultPresetModels() {
        Set<String> defaults = new LinkedHashSet<>();
        for (AiPreset preset : AiPreset.values()) {
            if (!preset.defaultModel().isBlank()) {
                defaults.add(preset.defaultModel());
            }
        }
        return List.copyOf(defaults);
    }

    private AiConfig currentAiConfig() {
        return new AiConfig(
                aiPreset.getValue() == null ? AiPreset.DISABLED : aiPreset.getValue(),
                aiBaseUrl.getValue(),
                aiModel.getValue(),
                aiApiKey.getValue());
    }

    void save() {
        appSettingService.setAiConfig(currentAiConfig());
        Notification.show(getTranslation("admin.settings.notification.saved"), 3000,
                Notification.Position.MIDDLE);
    }
}
