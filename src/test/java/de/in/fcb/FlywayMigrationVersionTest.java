package de.in.fcb;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.junit.jupiter.api.Test;

class FlywayMigrationVersionTest {

    private static final Pattern VERSION_PATTERN = Pattern.compile("^V(\\d+)__.+\\.sql$");

    @Test
    void allFlywayMigrationsHaveUniqueVersionNumbers() throws IOException {
        Path migrationDir = Path.of("src/main/resources/db/migration");
        List<String> versions;
        try (var files = Files.list(migrationDir)) {
            versions = files
                    .map(path -> path.getFileName().toString())
                    .map(name -> {
                        Matcher matcher = VERSION_PATTERN.matcher(name);
                        assertThat(matcher.matches())
                                .as("Invalid Flyway migration filename: %s", name)
                                .isTrue();
                        return matcher.group(1);
                    })
                    .toList();
        }
        assertThat(versions).doesNotHaveDuplicates();
    }
}
