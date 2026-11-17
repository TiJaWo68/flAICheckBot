package de.in.fcb.admin;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.security.crypto.password.PasswordEncoder;

import de.in.fcb.security.AppRole;
import de.in.fcb.security.AppUser;
import de.in.fcb.security.AppUserRepository;
import de.in.fcb.security.AuthProvider;

class UserServiceTest {

    private AppUserRepository repository;
    private PasswordEncoder passwordEncoder;
    private UserService service;

    @BeforeEach
    void setUp() {
        repository = mock(AppUserRepository.class);
        passwordEncoder = mock(PasswordEncoder.class);
        service = new UserService(repository, passwordEncoder);
    }

    @Test
    void createHashesInitialPasswordForLocalAccount() {
        when(passwordEncoder.encode("secret123")).thenReturn("hashed");
        when(repository.save(any(AppUser.class))).thenAnswer(invocation -> invocation.getArgument(0));

        AppUser user = user(null, "alice", "alice@example.com", Set.of(AppRole.ROLE_USER), AuthProvider.LOCAL);

        AppUser saved = service.create(user, "secret123");

        assertThat(saved.getPasswordHash()).isEqualTo("hashed");
        verify(repository).save(user);
    }

    @Test
    void createRejectsBlankRoles() {
        AppUser user = user(null, "alice", "alice@example.com", Set.of(), AuthProvider.LOCAL);

        assertThatThrownBy(() -> service.create(user, "secret123"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("role");
    }

    @Test
    void createRejectsNonAlphanumericUsername() {
        AppUser user = user(null, "alice!", "alice@example.com", Set.of(AppRole.ROLE_USER), AuthProvider.LOCAL);

        assertThatThrownBy(() -> service.create(user, "secret123"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("alphanumeric");
    }

    @Test
    void createRejectsDuplicateUsername() {
        AppUser existing = user(UUID.randomUUID(), "alice", "other@example.com", Set.of(AppRole.ROLE_USER),
                AuthProvider.LOCAL);
        when(repository.findByUsername("alice")).thenReturn(Optional.of(existing));

        AppUser user = user(null, "alice", "alice@example.com", Set.of(AppRole.ROLE_USER), AuthProvider.LOCAL);

        assertThatThrownBy(() -> service.create(user, "secret123"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("already exists");
    }

    @Test
    void createRequiresPasswordForLocalAccount() {
        AppUser user = user(null, "alice", "alice@example.com", Set.of(AppRole.ROLE_USER), AuthProvider.LOCAL);

        assertThatThrownBy(() -> service.create(user, "   "))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Password");
    }

    @Test
    void updateKeepsExistingPasswordWhenNewPasswordIsBlank() {
        UUID id = UUID.randomUUID();
        AppUser existing = user(id, "alice", "alice@example.com", Set.of(AppRole.ROLE_USER), AuthProvider.LOCAL);
        existing.setPasswordHash("oldhash");
        when(repository.findById(id)).thenReturn(Optional.of(existing));
        when(repository.save(any(AppUser.class))).thenAnswer(invocation -> invocation.getArgument(0));

        AppUser update = user(id, "alice", "alice@example.com", Set.of(AppRole.ROLE_USER), AuthProvider.LOCAL);

        AppUser saved = service.update(update, "   ");

        assertThat(saved.getPasswordHash()).isEqualTo("oldhash");
    }

    @Test
    void updateEncodesNewPassword() {
        UUID id = UUID.randomUUID();
        AppUser existing = user(id, "alice", "alice@example.com", Set.of(AppRole.ROLE_USER), AuthProvider.LOCAL);
        existing.setPasswordHash("oldhash");
        when(repository.findById(id)).thenReturn(Optional.of(existing));
        when(repository.save(any(AppUser.class))).thenAnswer(invocation -> invocation.getArgument(0));
        when(passwordEncoder.encode("newsecret")).thenReturn("newhash");

        AppUser update = user(id, "alice", "alice@example.com", Set.of(AppRole.ROLE_USER), AuthProvider.LOCAL);

        AppUser saved = service.update(update, "newsecret");

        assertThat(saved.getPasswordHash()).isEqualTo("newhash");
    }

    @Test
    void deleteRejectsSelfDeletion() {
        UUID id = UUID.randomUUID();
        AppUser user = user(id, "alice", "alice@example.com", Set.of(AppRole.ROLE_ADMIN), AuthProvider.LOCAL);

        assertThatThrownBy(() -> service.delete(id, id))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("own account");

        verify(repository, never()).delete(any(AppUser.class));
    }

    @Test
    void deleteRejectsLastAdministrator() {
        UUID id = UUID.randomUUID();
        AppUser user = user(id, "alice", "alice@example.com", Set.of(AppRole.ROLE_ADMIN), AuthProvider.LOCAL);
        when(repository.findById(id)).thenReturn(Optional.of(user));
        when(repository.findAll()).thenReturn(List.of(user));

        assertThatThrownBy(() -> service.delete(id, UUID.randomUUID()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("last administrator");

        verify(repository, never()).delete(any(AppUser.class));
    }

    @Test
    void deleteSucceedsWhenAnotherAdministratorExists() {
        UUID id = UUID.randomUUID();
        AppUser user = user(id, "alice", "alice@example.com", Set.of(AppRole.ROLE_ADMIN), AuthProvider.LOCAL);
        AppUser otherAdmin = user(UUID.randomUUID(), "bob", "bob@example.com", Set.of(AppRole.ROLE_ADMIN),
                AuthProvider.LOCAL);
        when(repository.findById(id)).thenReturn(Optional.of(user));
        when(repository.findAll()).thenReturn(List.of(user, otherAdmin));

        service.delete(id, UUID.randomUUID());

        verify(repository).delete(user);
    }

    @Test
    void resetPasswordRejectsOAuth2Account() {
        UUID id = UUID.randomUUID();
        AppUser user = user(id, "alice", "alice@example.com", Set.of(AppRole.ROLE_USER), AuthProvider.GOOGLE);
        when(repository.findById(id)).thenReturn(Optional.of(user));

        assertThatThrownBy(() -> service.resetPassword(id, "newsecret"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("local");
    }

    private AppUser user(UUID id, String username, String email, Set<AppRole> roles, AuthProvider authProvider) {
        AppUser user = new AppUser();
        user.setId(id);
        user.setUsername(username);
        user.setEmail(email);
        user.setRoles(roles);
        user.setAuthProvider(authProvider);
        return user;
    }
}
