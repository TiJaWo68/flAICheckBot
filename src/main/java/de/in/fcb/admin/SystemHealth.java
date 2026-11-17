package de.in.fcb.admin;

/**
 * Read-only diagnostic snapshot shown in the System Health tab.
 */
public record SystemHealth(
        String version,
        String activeProfile,
        String javaRuntime,
        boolean databaseOnline,
        long databaseLatencyMs,
        String lastFlywayMigration,
        boolean loggingTcpEnabled,
        String loggingTcpHost,
        int loggingTcpPort) {
}
