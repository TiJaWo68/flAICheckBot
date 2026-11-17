package de.in.fcb.security;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.concurrent.atomic.AtomicReference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Service;

/**
 * Generates and verifies the one-time token that unlocks the initial setup.
 * <p>
 * As long as no account exists, a cryptographically secure token is generated
 * once at application startup, printed prominently to the server log and kept
 * in memory. The token is invalidated after the initial administrator has been
 * created.
 */
@Service
public class SetupTokenService implements ApplicationRunner {

    private static final Logger LOGGER = LoggerFactory.getLogger(SetupTokenService.class);

    private static final int TOKEN_LENGTH = 8;
    private static final char[] TOKEN_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
            .toCharArray();

    private final AppUserRepository appUserRepository;
    private final AtomicReference<String> token = new AtomicReference<>();

    public SetupTokenService(AppUserRepository appUserRepository) {
        this.appUserRepository = appUserRepository;
    }

    @Override
    public void run(ApplicationArguments args) {
        if (appUserRepository.count() > 0) {
            return;
        }

        String generatedToken = generateToken();
        token.set(generatedToken);
        logToken(generatedToken);
    }

    /**
     * Verifies the submitted token against the generated one.
     */
    public boolean isValid(String candidate) {
        String currentToken = token.get();
        if (currentToken == null || candidate == null) {
            return false;
        }
        return MessageDigest.isEqual(
                currentToken.getBytes(StandardCharsets.UTF_8),
                candidate.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Invalidates the generated token after a successful setup.
     */
    public void invalidate() {
        token.set(null);
    }

    /**
     * Regenerates the setup token on explicit administrator request and prints
     * it to the server log.
     */
    public void renew() {
        String generatedToken = generateToken();
        token.set(generatedToken);
        logToken(generatedToken);
    }

    String generateToken() {
        SecureRandom secureRandom = new SecureRandom();
        StringBuilder tokenBuilder = new StringBuilder(TOKEN_LENGTH);
        for (int index = 0; index < TOKEN_LENGTH; index++) {
            tokenBuilder.append(TOKEN_ALPHABET[secureRandom.nextInt(TOKEN_ALPHABET.length)]);
        }
        return tokenBuilder.toString();
    }

    private void logToken(String generatedToken) {
        LOGGER.info("""
                ==============================================================
                FLAICHECKBOT INITIAL SETUP TOKEN:  [ {} ]
                Bitte diesen Token zur Freischaltung auf /setup eingeben.
                ==============================================================""",
                generatedToken);
    }
}
