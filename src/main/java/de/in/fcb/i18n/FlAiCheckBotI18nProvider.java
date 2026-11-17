package de.in.fcb.i18n;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.text.MessageFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.MissingResourceException;
import java.util.PropertyResourceBundle;
import java.util.ResourceBundle;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.stereotype.Component;

import com.vaadin.flow.i18n.I18NProvider;

/**
 * Provides the supported UI locales and resolves translated messages.
 * <p>
 * Falls back to German when a key is missing for the requested locale.
 */
@Component
public class FlAiCheckBotI18nProvider implements I18NProvider {

    private static final String BUNDLE_BASE_NAME = "i18n/messages";

    private static final List<Locale> PROVIDED_LOCALES = List.of(Locale.GERMAN, Locale.ENGLISH);

    private final Map<Locale, ResourceBundle> bundles = new ConcurrentHashMap<>();

    public FlAiCheckBotI18nProvider() {
        PROVIDED_LOCALES.forEach(locale -> bundles.put(locale, loadBundle(locale)));
    }

    @Override
    public List<Locale> getProvidedLocales() {
        return PROVIDED_LOCALES;
    }

    @Override
    public String getTranslation(String key, Locale locale, Object... params) {
        if (key == null) {
            return "";
        }

        String value = resolveValue(key, locale);
        if (value == null) {
            value = resolveValue(key, Locale.GERMAN);
        }
        if (value == null) {
            return key;
        }

        return params.length == 0 ? value : MessageFormat.format(value, params);
    }

    private String resolveValue(String key, Locale locale) {
        ResourceBundle bundle = bundles.get(locale);
        if (bundle == null) {
            return null;
        }

        try {
            return bundle.getString(key);
        } catch (MissingResourceException exception) {
            return null;
        }
    }

    private static ResourceBundle loadBundle(Locale locale) {
        return ResourceBundle.getBundle(BUNDLE_BASE_NAME, locale, UTF8_CONTROL);
    }

    /**
     * Loads property bundles as UTF-8 so that German umlauts are decoded correctly.
     */
    private static final ResourceBundle.Control UTF8_CONTROL = new ResourceBundle.Control() {
        @Override
        public ResourceBundle newBundle(String baseName, Locale locale, String format, ClassLoader loader,
                boolean reload) throws IOException {
            String bundleName = toBundleName(baseName, locale);
            String resourceName = toResourceName(bundleName, "properties");

            try (InputStream stream = loader.getResourceAsStream(resourceName)) {
                if (stream == null) {
                    return null;
                }
                try (Reader reader = new InputStreamReader(stream, StandardCharsets.UTF_8)) {
                    return new PropertyResourceBundle(reader);
                }
            }
        }
    };
}
