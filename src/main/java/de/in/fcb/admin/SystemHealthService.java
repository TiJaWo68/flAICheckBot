package de.in.fcb.admin;

import java.sql.Connection;
import java.util.List;
import java.util.Map;

import javax.sql.DataSource;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import de.in.fcb.security.SetupTokenService;

/**
 * Collects read-only runtime diagnostics for the administrator: application
 * metadata, PostgreSQL connectivity, last Flyway migration and logging
 * configuration.
 */
@Service
public class SystemHealthService {

    private static final int CONNECTION_VALIDATION_SECONDS = 2;

    private final JdbcTemplate jdbcTemplate;
    private final DataSource dataSource;
    private final Environment environment;
    private final SetupTokenService setupTokenService;
    private final String version;
    private final boolean loggingTcpEnabled;
    private final String loggingTcpHost;
    private final int loggingTcpPort;

    public SystemHealthService(JdbcTemplate jdbcTemplate, DataSource dataSource, Environment environment,
            SetupTokenService setupTokenService,
            @Value("${flaicheckbot.version:unknown}") String version,
            @Value("${flaicheckbot.logging.tcp.enabled:false}") boolean loggingTcpEnabled,
            @Value("${flaicheckbot.logging.tcp.host:}") String loggingTcpHost,
            @Value("${flaicheckbot.logging.tcp.port:0}") int loggingTcpPort) {
        this.jdbcTemplate = jdbcTemplate;
        this.dataSource = dataSource;
        this.environment = environment;
        this.setupTokenService = setupTokenService;
        this.version = version;
        this.loggingTcpEnabled = loggingTcpEnabled;
        this.loggingTcpHost = loggingTcpHost;
        this.loggingTcpPort = loggingTcpPort;
    }

    public SystemHealth collect() {
        DatabaseStatus databaseStatus = checkDatabase();
        return new SystemHealth(
                version,
                resolveActiveProfile(),
                System.getProperty("java.vendor") + " " + System.getProperty("java.version"),
                databaseStatus.online(),
                databaseStatus.latencyMs(),
                findLastFlywayMigration(),
                loggingTcpEnabled,
                loggingTcpHost,
                loggingTcpPort);
    }

    /**
     * Regenerates the one-time setup token and prints it to the server log.
     * The token is intentionally not returned to the UI.
     */
    public void renewSetupToken() {
        setupTokenService.renew();
    }

    private String resolveActiveProfile() {
        String[] activeProfiles = environment.getActiveProfiles();
        if (activeProfiles.length == 0) {
            return "default";
        }
        return String.join(", ", activeProfiles);
    }

    private DatabaseStatus checkDatabase() {
        long start = System.nanoTime();
        try (Connection connection = dataSource.getConnection()) {
            boolean valid = connection.isValid(CONNECTION_VALIDATION_SECONDS);
            long latencyMs = (System.nanoTime() - start) / 1_000_000;
            return new DatabaseStatus(valid, latencyMs);
        } catch (Exception exception) {
            return new DatabaseStatus(false, 0);
        }
    }

    private String findLastFlywayMigration() {
        try {
            List<Map<String, Object>> rows = jdbcTemplate.queryForList(
                    "SELECT version, description, installed_on FROM flyway_schema_history "
                            + "WHERE success = TRUE ORDER BY installed_rank DESC LIMIT 1");
            if (rows.isEmpty()) {
                return null;
            }
            Map<String, Object> row = rows.get(0);
            return row.get("version") + " - " + row.get("description") + " (" + row.get("installed_on") + ")";
        } catch (Exception exception) {
            return null;
        }
    }

    private record DatabaseStatus(boolean online, long latencyMs) {
    }
}
