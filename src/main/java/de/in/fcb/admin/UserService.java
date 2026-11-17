package de.in.fcb.admin;

import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import de.in.fcb.security.AppRole;
import de.in.fcb.security.AppUser;
import de.in.fcb.security.AppUserRepository;
import de.in.fcb.security.AuthProvider;

/**
 * Administration service for technical login accounts.
 * <p>
 * Enforces username/email uniqueness, role assignment, local password policy
 * and the safety guards that prevent an administrator from disabling or
 * deleting their own account or the last remaining administrator.
 */
@Service
public class UserService {

    private static final int PASSWORD_MIN_LENGTH = 8;
    private static final int USERNAME_MAX_LENGTH = 50;
    private static final int EMAIL_MAX_LENGTH = 120;

    private static final Pattern USERNAME_PATTERN = Pattern.compile("[a-zA-Z0-9]+");
    private static final Pattern EMAIL_PATTERN = Pattern.compile("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$");

    private final AppUserRepository appUserRepository;
    private final PasswordEncoder passwordEncoder;

    public UserService(AppUserRepository appUserRepository, PasswordEncoder passwordEncoder) {
        this.appUserRepository = appUserRepository;
        this.passwordEncoder = passwordEncoder;
    }

    @Transactional(readOnly = true)
    public List<AppUser> findAll() {
        return appUserRepository.findAll().stream()
                .sorted(Comparator.comparing(AppUser::getUsername, String.CASE_INSENSITIVE_ORDER))
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public AppUser findByUsername(String username) {
        return appUserRepository.findByUsername(username)
                .orElseThrow(() -> new IllegalArgumentException("User not found: " + username));
    }

    @Transactional
    public AppUser create(AppUser user, String initialPassword) {
        user.setId(null);
        validate(user, initialPassword, true);

        if (user.getAuthProvider() == AuthProvider.LOCAL) {
            user.setPasswordHash(passwordEncoder.encode(initialPassword));
        } else {
            user.setPasswordHash(null);
        }
        return appUserRepository.save(user);
    }

    @Transactional
    public AppUser update(AppUser user, String newPassword) {
        AppUser existing = requireUser(user.getId());
        validate(user, newPassword, false);

        if (user.getAuthProvider() == AuthProvider.LOCAL) {
            if (newPassword != null && !newPassword.isBlank()) {
                user.setPasswordHash(passwordEncoder.encode(newPassword));
            } else if (existing.getPasswordHash() == null) {
                throw new IllegalArgumentException("A password is required for local accounts");
            } else {
                user.setPasswordHash(existing.getPasswordHash());
            }
        } else {
            user.setPasswordHash(null);
        }
        return appUserRepository.save(user);
    }

    @Transactional
    public AppUser resetPassword(UUID id, String newPassword) {
        AppUser user = requireUser(id);
        if (user.getAuthProvider() != AuthProvider.LOCAL) {
            throw new IllegalArgumentException("Password reset is only available for local accounts");
        }
        validatePassword(newPassword);
        user.setPasswordHash(passwordEncoder.encode(newPassword));
        return appUserRepository.save(user);
    }

    @Transactional
    public AppUser setEnabled(UUID id, boolean enabled, UUID actingUserId) {
        requireNotSelf(id, actingUserId);
        AppUser user = requireUser(id);
        user.setEnabled(enabled);
        return appUserRepository.save(user);
    }

    @Transactional
    public void delete(UUID id, UUID actingUserId) {
        requireNotSelf(id, actingUserId);
        AppUser user = requireUser(id);
        if (user.hasRole(AppRole.ROLE_ADMIN) && countAdministrators() <= 1) {
            throw new IllegalArgumentException("The last administrator cannot be deleted");
        }
        appUserRepository.delete(user);
    }

    private long countAdministrators() {
        return findAll().stream()
                .filter(user -> user.hasRole(AppRole.ROLE_ADMIN))
                .count();
    }

    private void validate(AppUser user, String password, boolean requirePassword) {
        if (user == null) {
            throw new IllegalArgumentException("User must not be null");
        }
        validateUsername(user);
        validateEmail(user);
        if (user.getRoles() == null || user.getRoles().isEmpty()) {
            throw new IllegalArgumentException("At least one role is required");
        }
        if (user.getAuthProvider() == null) {
            throw new IllegalArgumentException("Authentication method is required");
        }
        if (user.getAuthProvider() == AuthProvider.LOCAL && requirePassword) {
            validatePassword(password);
        }
    }

    private void validateUsername(AppUser user) {
        if (user.getUsername() == null || user.getUsername().isBlank()) {
            throw new IllegalArgumentException("Username is required");
        }
        String username = user.getUsername().trim();
        if (username.length() > USERNAME_MAX_LENGTH) {
            throw new IllegalArgumentException("Username must not exceed " + USERNAME_MAX_LENGTH + " characters");
        }
        if (!USERNAME_PATTERN.matcher(username).matches()) {
            throw new IllegalArgumentException("Username must contain only alphanumeric characters");
        }
        user.setUsername(username);
        appUserRepository.findByUsername(username).ifPresent(existing -> {
            if (!existing.getId().equals(user.getId())) {
                throw new IllegalArgumentException("Username already exists");
            }
        });
    }

    private void validateEmail(AppUser user) {
        if (user.getEmail() == null || user.getEmail().isBlank()) {
            throw new IllegalArgumentException("Email is required");
        }
        String email = user.getEmail().trim();
        if (email.length() > EMAIL_MAX_LENGTH) {
            throw new IllegalArgumentException("Email must not exceed " + EMAIL_MAX_LENGTH + " characters");
        }
        if (!EMAIL_PATTERN.matcher(email).matches()) {
            throw new IllegalArgumentException("Email is not valid");
        }
        user.setEmail(email);
        appUserRepository.findByEmail(email).ifPresent(existing -> {
            if (!existing.getId().equals(user.getId())) {
                throw new IllegalArgumentException("Email already exists");
            }
        });
    }

    private void validatePassword(String password) {
        if (password == null || password.isBlank()) {
            throw new IllegalArgumentException("Password is required");
        }
        if (password.length() < PASSWORD_MIN_LENGTH) {
            throw new IllegalArgumentException("Password must be at least " + PASSWORD_MIN_LENGTH + " characters");
        }
    }

    private AppUser requireUser(UUID id) {
        return appUserRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("User not found: " + id));
    }

    private void requireNotSelf(UUID id, UUID actingUserId) {
        if (id.equals(actingUserId)) {
            throw new IllegalArgumentException("You cannot modify your own account");
        }
    }
}
