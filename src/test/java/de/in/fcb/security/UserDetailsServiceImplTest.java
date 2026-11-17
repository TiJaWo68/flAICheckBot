package de.in.fcb.security;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.Optional;
import java.util.Set;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;

class UserDetailsServiceImplTest {

    private AppUserRepository repository;
    private UserDetailsServiceImpl service;

    @BeforeEach
    void setUp() {
        repository = mock(AppUserRepository.class);
        service = new UserDetailsServiceImpl(repository);
    }

    @Test
    void loadsEnabledAdminUserWithPasswordHashAndAuthority() {
        AppUser appUser = new AppUser();
        appUser.setUsername("admin");
        appUser.setPasswordHash("$2a$10$hashed");
        appUser.setEnabled(true);
        appUser.setRoles(Set.of(AppRole.ROLE_ADMIN));
        when(repository.findByUsername("admin")).thenReturn(Optional.of(appUser));

        UserDetails result = service.loadUserByUsername("admin");

        assertThat(result.getUsername()).isEqualTo("admin");
        assertThat(result.getPassword()).isEqualTo("$2a$10$hashed");
        assertThat(result.isEnabled()).isTrue();
        assertThat(result.getAuthorities())
                .extracting(GrantedAuthority::getAuthority)
                .containsExactly("ROLE_ADMIN");
    }

    @Test
    void mapsDisabledFlagToUserDetails() {
        AppUser appUser = new AppUser();
        appUser.setUsername("bob");
        appUser.setPasswordHash("hash");
        appUser.setEnabled(false);
        appUser.setRoles(Set.of(AppRole.ROLE_USER));
        when(repository.findByUsername("bob")).thenReturn(Optional.of(appUser));

        assertThat(service.loadUserByUsername("bob").isEnabled()).isFalse();
    }

    @Test
    void loadsOAuth2OnlyAccountWithoutPasswordHash() {
        AppUser appUser = new AppUser();
        appUser.setUsername("google-user");
        appUser.setPasswordHash(null);
        appUser.setEnabled(true);
        appUser.setRoles(Set.of(AppRole.ROLE_USER));
        when(repository.findByUsername("google-user")).thenReturn(Optional.of(appUser));

        UserDetails result = service.loadUserByUsername("google-user");

        assertThat(result.getPassword()).isEmpty();
        assertThat(result.isEnabled()).isTrue();
    }

    @Test
    void throwsWhenUsernameIsUnknown() {
        when(repository.findByUsername("ghost")).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.loadUserByUsername("ghost"))
                .isInstanceOf(UsernameNotFoundException.class)
                .hasMessageContaining("ghost");
    }
}
