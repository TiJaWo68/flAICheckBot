package de.in.fcb.exams;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

/**
 * Persistence access for {@link StudentWorkPage} scan files.
 */
public interface StudentWorkPageRepository extends JpaRepository<StudentWorkPage, UUID> {

    List<StudentWorkPage> findByStudentWorkIdOrderByPageNumberAsc(UUID studentWorkId);

    void deleteByStudentWorkId(UUID studentWorkId);

    long countByStudentWorkId(UUID studentWorkId);
}
