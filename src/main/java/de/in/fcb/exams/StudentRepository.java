package de.in.fcb.exams;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

/**
 * Persistence access for pseudonymized {@link Student} records.
 */
public interface StudentRepository extends JpaRepository<Student, UUID> {

    List<Student> findBySchoolClassIdOrderByExternalIdAsc(UUID schoolClassId);

    Optional<Student> findBySchoolClassIdAndExternalId(UUID schoolClassId, String externalId);

    long countBySchoolClassId(UUID schoolClassId);
}
