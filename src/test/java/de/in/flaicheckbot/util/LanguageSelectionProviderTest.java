package de.in.flaicheckbot.util;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import java.util.List;

public class LanguageSelectionProviderTest {

    @Test
    public void testGetLanguages() {
        List<LanguageSelectionProvider.LanguageInfo> languages = LanguageSelectionProvider.getLanguages();
        assertNotNull(languages);
        assertTrue(languages.size() >= 4);

        boolean foundDe = false;
        for (LanguageSelectionProvider.LanguageInfo info : languages) {
            if ("de".equals(info.getIsoCode()))
                foundDe = true;
        }
        assertTrue(foundDe, "German should be in the list");
    }

    @Test
    public void testMapToIsoCode() {
        assertEquals("de", LanguageSelectionProvider.mapToIsoCode("Deutsch"));
        assertEquals("en", LanguageSelectionProvider.mapToIsoCode("Englisch"));
        assertEquals("fr", LanguageSelectionProvider.mapToIsoCode("Französisch"));
        assertEquals("es", LanguageSelectionProvider.mapToIsoCode("Spanisch"));

        // Case sensitivity
        assertEquals("de", LanguageSelectionProvider.mapToIsoCode("deutsch"));

        // Fallbacks
        assertEquals("en", LanguageSelectionProvider.mapToIsoCode("English"));

        // Default
        assertEquals("de", LanguageSelectionProvider.mapToIsoCode("Unknown"));
        assertEquals("de", LanguageSelectionProvider.mapToIsoCode(null));
    }

    @Test
    public void testGetDisplayNames() {
        String[] names = LanguageSelectionProvider.getDisplayNames();
        assertNotNull(names);
        assertTrue(names.length >= 4);
        assertEquals("Deutsch", names[0]);
    }
}
