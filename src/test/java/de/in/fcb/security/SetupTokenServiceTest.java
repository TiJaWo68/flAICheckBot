package de.in.fcb.security;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import java.util.regex.Pattern;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class SetupTokenServiceTest {

    private static final Pattern TOKEN_PATTERN = Pattern.compile("[A-Za-z2-9]{8}");

    private SetupTokenService service;

    @BeforeEach
    void setUp() {
        service = new SetupTokenService(mock(AppUserRepository.class));
    }

    @Test
    void generatesEightCharacterAlphanumericToken() {
        for (int index = 0; index < 100; index++) {
            String token = service.generateToken();

            assertThat(token).hasSize(8);
            assertThat(TOKEN_PATTERN.matcher(token).matches()).isTrue();
        }
    }

    @Test
    void rejectsAnyTokenBeforeGeneration() {
        assertThat(service.isValid(null)).isFalse();
        assertThat(service.isValid("")).isFalse();
        assertThat(service.isValid("ABC23456")).isFalse();
    }
}
