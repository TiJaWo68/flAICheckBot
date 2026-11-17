package de.in.fcb.security;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.security.crypto.password.PasswordEncoder;

class SetupServiceTest {

    private static final String VALID_TOKEN = "ABC23456";

    private AppUserRepository repository;
    private PasswordEncoder passwordEncoder;
    private SetupTokenService setupTokenService;
    private SetupService service;

    @BeforeEach
    void setUp() {
        repository = mock(AppUserRepository.class);
        passwordEncoder = mock(PasswordEncoder.class);
        setupTokenService = mock(SetupTokenService.class);
        service = new SetupService(repository, passwordEncoder, setupTokenService);
    }

    @Test
    void createsAdminWithValidTokenAndInvalidatesToken() {
        when(repository.count()).thenReturn(0L);
        when(setupTokenService.isValid(VALID_TOKEN)).thenReturn(true);
        when(passwordEncoder.encode("secret123")).thenReturn("hashed");
        when(repository.save(any(AppUser.class))).thenAnswer(invocation -> invocation.getArgument(0));

        AppUser result = service.createInitialAdmin("admin", "admin@example.com", "secret123", VALID_TOKEN);

        assertThat(result.getUsername()).isEqualTo("admin");
        assertThat(result.getEmail()).isEqualTo("admin@example.com");
        assertThat(result.getPasswordHash()).isEqualTo("hashed");
        assertThat(result.getRoles()).containsExactly(AppRole.ROLE_ADMIN);
        assertThat(result.getAuthProvider()).isEqualTo(AuthProvider.LOCAL);
        assertThat(result.isEnabled()).isTrue();
        verify(repository).save(any(AppUser.class));
        verify(setupTokenService).invalidate();
    }

    @Test
    void rejectsSetupWhenTokenIsInvalid() {
        when(repository.count()).thenReturn(0L);
        when(setupTokenService.isValid("wrong")).thenReturn(false);

        assertThatThrownBy(() -> service.createInitialAdmin("admin", "admin@example.com", "secret123", "wrong"))
                .isInstanceOf(InvalidSetupTokenException.class);

        verify(repository, never()).save(any(AppUser.class));
        verify(passwordEncoder, never()).encode(any(CharSequence.class));
        verify(setupTokenService, never()).invalidate();
    }

    @Test
    void rejectsSetupWhenTokenIsEmpty() {
        when(repository.count()).thenReturn(0L);
        when(setupTokenService.isValid("")).thenReturn(false);

        assertThatThrownBy(() -> service.createInitialAdmin("admin", "admin@example.com", "secret123", ""))
                .isInstanceOf(InvalidSetupTokenException.class);

        verify(repository, never()).save(any(AppUser.class));
        verify(setupTokenService, never()).invalidate();
    }

    @Test
    void rejectsSetupWhenAUserAlreadyExists() {
        when(repository.count()).thenReturn(1L);

        assertThatThrownBy(() -> service.createInitialAdmin("admin", "admin@example.com", "secret123", VALID_TOKEN))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("already exists");

        verify(repository, never()).save(any(AppUser.class));
        verify(passwordEncoder, never()).encode(any(CharSequence.class));
        verify(setupTokenService, never()).invalidate();
    }
}
