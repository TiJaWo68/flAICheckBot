package de.in.fcb.exams;

import java.util.List;
import java.util.UUID;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Service for managing {@link SchoolClass} groups and pseudonymized
 * {@link Student} entries.
 */
@Service
public class SchoolClassService {

    private final SchoolClassRepository classRepository;
    private final StudentRepository studentRepository;

    public SchoolClassService(SchoolClassRepository classRepository, StudentRepository studentRepository) {
        this.classRepository = classRepository;
        this.studentRepository = studentRepository;
    }

    @Transactional(readOnly = true)
    public List<SchoolClass> findAllClasses() {
        return classRepository.findAllByOrderByNameAsc();
    }

    @Transactional(readOnly = true)
    public SchoolClass findClass(UUID id) {
        return classRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Class not found: " + id));
    }

    @Transactional
    public SchoolClass saveClass(SchoolClass schoolClass) {
        if (schoolClass == null) {
            throw new IllegalArgumentException("Class must not be null");
        }
        String name = schoolClass.getName() == null ? "" : schoolClass.getName().trim();
        if (name.isBlank()) {
            throw new IllegalArgumentException("Class name is required");
        }
        String year = schoolClass.getAcademicYear() == null ? "" : schoolClass.getAcademicYear().trim();
        if (year.isBlank()) {
            throw new IllegalArgumentException("Academic year is required");
        }
        schoolClass.setName(name);
        schoolClass.setAcademicYear(year);
        if (schoolClass.getForeignLanguage() == null) {
            schoolClass.setForeignLanguage(ForeignLanguage.EN);
        }

        classRepository.findByNameIgnoreCase(name).ifPresent(existing -> {
            if (!existing.getId().equals(schoolClass.getId())) {
                throw new IllegalArgumentException("Class name already exists: " + name);
            }
        });
        return classRepository.save(schoolClass);
    }

    @Transactional
    public void deleteClass(UUID id) {
        classRepository.deleteById(id);
    }

    @Transactional(readOnly = true)
    public List<Student> findStudents(UUID classId) {
        return studentRepository.findBySchoolClassIdOrderByExternalIdAsc(classId);
    }

    @Transactional(readOnly = true)
    public long countStudents(UUID classId) {
        return studentRepository.countBySchoolClassId(classId);
    }

    @Transactional
    public Student saveStudent(Student student) {
        if (student == null || student.getSchoolClassId() == null) {
            throw new IllegalArgumentException("Student and class ID are required");
        }
        String externalId = student.getExternalId() == null ? "" : student.getExternalId().trim();
        if (externalId.isBlank()) {
            throw new IllegalArgumentException("External ID is required");
        }
        student.setExternalId(externalId);
        if (student.getDisplayAlias() != null) {
            String trimmedAlias = student.getDisplayAlias().trim();
            student.setDisplayAlias(trimmedAlias.isEmpty() ? null : trimmedAlias);
        }

        studentRepository.findBySchoolClassIdAndExternalId(student.getSchoolClassId(), externalId)
                .ifPresent(existing -> {
                    if (!existing.getId().equals(student.getId())) {
                        throw new IllegalArgumentException("Student ID already exists in class: " + externalId);
                    }
                });
        return studentRepository.save(student);
    }

    @Transactional
    public int generateStudentsBatch(UUID classId, String prefix, int fromIndex, int toIndex) {
        if (classId == null) {
            throw new IllegalArgumentException("Class ID is required");
        }
        if (fromIndex < 1 || toIndex < fromIndex || (toIndex - fromIndex) > 100) {
            throw new IllegalArgumentException("Invalid batch range");
        }
        String effectivePrefix = (prefix == null || prefix.isBlank()) ? "S" : prefix.trim();
        int width = Math.max(2, String.valueOf(toIndex).length());

        int created = 0;
        for (int index = fromIndex; index <= toIndex; index++) {
            String extId = effectivePrefix + String.format("%0" + width + "d", index);
            if (studentRepository.findBySchoolClassIdAndExternalId(classId, extId).isEmpty()) {
                Student student = new Student();
                student.setSchoolClassId(classId);
                student.setExternalId(extId);
                studentRepository.save(student);
                created++;
            }
        }
        return created;
    }

    @Transactional
    public void deleteStudent(UUID studentId) {
        studentRepository.deleteById(studentId);
    }
}
