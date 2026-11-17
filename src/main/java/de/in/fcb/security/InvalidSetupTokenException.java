package de.in.fcb.security;

/**
 * Signals that the submitted one-time setup token is missing or invalid.
 */
public class InvalidSetupTokenException extends RuntimeException {

    public InvalidSetupTokenException(String message) {
        super(message);
    }
}
