package de.in.fcb.security;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;

class SecurityConfigTest {

    @Test
    void isAnnotatedAsConfiguration() {
        assertThat(SecurityConfig.class.isAnnotationPresent(Configuration.class)).isTrue();
    }

    @Test
    void providesBcryptPasswordEncoder() {
        SecurityConfig config = new SecurityConfig();

        assertThat(config.passwordEncoder()).isInstanceOf(BCryptPasswordEncoder.class);
    }

    @Test
    void publicEndpointsIncludeStaticAssetsToPreventAuthRedirectLoops() {
        assertThat(SecurityConfig.PUBLIC_ENDPOINTS)
                .as("Static resources must be public so they never trigger a 403 or redirect after login")
                .contains("/icons/**", "/styles/**", "/favicon.ico");
    }
}
