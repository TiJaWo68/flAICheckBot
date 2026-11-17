package de.in.fcb.exams;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

/**
 * Persistence access for ordered {@link TestTask} items.
 */
public interface TestTaskRepository extends JpaRepository<TestTask, UUID> {

    List<TestTask> findByTestDefinitionIdOrderByPositionAsc(UUID testDefinitionId);

    void deleteByTestDefinitionId(UUID testDefinitionId);

    long countByTestDefinitionId(UUID testDefinitionId);
}
