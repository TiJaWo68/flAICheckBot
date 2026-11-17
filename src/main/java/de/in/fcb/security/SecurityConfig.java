package de.in.fcb.security;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;

import com.vaadin.flow.spring.security.VaadinSecurityConfigurer;

import de.in.fcb.ui.LoginView;

/**
 * Application security configuration.
 * <p>
 * Protects all Vaadin views by default and supports local form login as well as
 * optional Google OAuth2/OIDC when {@code GOOGLE_CLIENT_ID} is configured.
 */
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    private static final String SETUP_ROUTE = "/setup";

    static final String[] PUBLIC_ENDPOINTS = {
            "/oauth2/authorization/**",
            "/login/oauth2/code/**",
            "/icons/**",
            "/styles/**",
            "/favicon.ico"
    };

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http,
            CustomOAuth2UserService customOAuth2UserService,
            @Value("${GOOGLE_CLIENT_ID:}") String googleClientId) throws Exception {
        http
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers(PUBLIC_ENDPOINTS).permitAll()
                        .requestMatchers(SETUP_ROUTE).permitAll());

        if (GoogleOAuth2Config.isConfigured(googleClientId)) {
            http.oauth2Login(oauth2 -> oauth2
                    .loginPage("/login")
                    .userInfoEndpoint(userInfo -> userInfo.userService(customOAuth2UserService)));
        }

        http.with(VaadinSecurityConfigurer.vaadin(), configurer -> configurer.loginView(LoginView.class));
        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
