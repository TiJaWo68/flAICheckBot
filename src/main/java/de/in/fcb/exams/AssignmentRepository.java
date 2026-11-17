package de.in.fcb.exams;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

/**
 * Persistence access for {@link Assignment} exam executions.
 */
public interface AssignmentRepository extends JpaRepository<Assignment, UUID> {

    List<Assignment> findAllByOrderByExamDateDescCreatedAtDesc();

    Optional<Assignment> findFirstByTitleIgnoreCaseAndSchoolClassId(String title, UUID schoolClassId);

    long countByTestDefinitionId(UUID testDefinitionId);
}
