package de.in.fcb.exams;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

/**
 * Persistence access for {@link StudentWork} exam submissions.
 */
public interface StudentWorkRepository extends JpaRepository<StudentWork, UUID> {

    List<StudentWork> findByAssignmentId(UUID assignmentId);

    Optional<StudentWork> findByAssignmentIdAndStudentId(UUID assignmentId, UUID studentId);

    long countByStatusIn(Collection<StudentWorkStatus> statuses);
}
