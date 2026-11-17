package de.in.fcb.security;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Set;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

/**
 * Persistence integration test that runs Flyway migrations against a real
 * PostgreSQL instance and verifies that an {@link AppUser} with
 * {@link AppRole#ROLE_ADMIN} can be persisted and reloaded.
 */
@SpringBootTest
@Testcontainers
@Tag("integration")
class AppUserRolePersistenceIntegrationTest {

    private static final String POSTGRES_IMAGE = "postgres:16-alpine";

    @Container
    static final PostgreSQLContainer<?> POSTGRES = new PostgreSQLContainer<>(POSTGRES_IMAGE);

    @DynamicPropertySource
    static void registerDatasourceProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", POSTGRES::getJdbcUrl);
        registry.add("spring.datasource.username", POSTGRES::getUsername);
        registry.add("spring.datasource.password", POSTGRES::getPassword);
    }

    @Autowired
    private AppUserRepository appUserRepository;

    @Test
    void persistsAdminWithStringMappedRole() {
        AppUser admin = new AppUser();
        admin.setUsername("admin");
        admin.setEmail("admin@example.com");
        admin.setPasswordHash("hashed");
        admin.setRoles(Set.of(AppRole.ROLE_ADMIN));
        admin.setAuthProvider(AuthProvider.LOCAL);
        admin.setEnabled(true);

        AppUser saved = appUserRepository.save(admin);

        assertThat(saved.getId()).isNotNull();
        assertThat(saved.getRoles()).containsExactly(AppRole.ROLE_ADMIN);

        AppUser reloaded = appUserRepository.findByUsername("admin").orElseThrow();
        assertThat(reloaded.getRoles()).containsExactly(AppRole.ROLE_ADMIN);
    }
}
