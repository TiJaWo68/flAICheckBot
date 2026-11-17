package de.in.fcb.exams;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

/**
 * Persistence access for {@link TestDefinition} templates.
 */
public interface TestDefinitionRepository extends JpaRepository<TestDefinition, UUID> {

    List<TestDefinition> findAllByOrderByCreatedAtDesc();

    Optional<TestDefinition> findFirstByTitleIgnoreCase(String title);
}
