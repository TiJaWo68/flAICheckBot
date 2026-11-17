package de.in.fcb.exams;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

/**
 * Persistence access for {@link SchoolClass} records.
 */
public interface SchoolClassRepository extends JpaRepository<SchoolClass, UUID> {

    List<SchoolClass> findAllByOrderByNameAsc();

    Optional<SchoolClass> findByNameIgnoreCase(String name);
}
