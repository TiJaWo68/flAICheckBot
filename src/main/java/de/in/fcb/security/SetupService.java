package de.in.fcb.security;

import java.util.Set;

import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Creates the initial administrator account during the first application setup.
 */
@Service
public class SetupService {

    private final AppUserRepository appUserRepository;
    private final PasswordEncoder passwordEncoder;
    private final SetupTokenService setupTokenService;

    public SetupService(AppUserRepository appUserRepository, PasswordEncoder passwordEncoder,
            SetupTokenService setupTokenService) {
        this.appUserRepository = appUserRepository;
        this.passwordEncoder = passwordEncoder;
        this.setupTokenService = setupTokenService;
    }

    @Transactional
    public AppUser createInitialAdmin(String username, String email, String password, String setupToken) {
        if (appUserRepository.count() > 0) {
            throw new IllegalStateException("Initial setup is unavailable because at least one user already exists");
        }
        if (!setupTokenService.isValid(setupToken)) {
            throw new InvalidSetupTokenException("Invalid or missing setup token");
        }

        AppUser admin = new AppUser();
        admin.setUsername(username);
        admin.setEmail(email);
        admin.setPasswordHash(passwordEncoder.encode(password));
        admin.setRoles(Set.of(AppRole.ROLE_ADMIN));
        admin.setAuthProvider(AuthProvider.LOCAL);
        admin.setEnabled(true);

        AppUser savedAdmin = appUserRepository.save(admin);
        setupTokenService.invalidate();
        return savedAdmin;
    }
}
