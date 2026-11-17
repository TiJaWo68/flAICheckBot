package de.in.fcb.exams;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.Optional;
import java.util.UUID;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class SchoolClassServiceTest {

    private SchoolClassRepository classRepository;
    private StudentRepository studentRepository;
    private SchoolClassService service;

    @BeforeEach
    void setUp() {
        classRepository = mock(SchoolClassRepository.class);
        studentRepository = mock(StudentRepository.class);
        service = new SchoolClassService(classRepository, studentRepository);
    }

    @Test
    void savesValidClassWithTrimmedName() {
        SchoolClass sc = new SchoolClass();
        sc.setName("  10a  ");
        sc.setAcademicYear("2026/2027");
        when(classRepository.findByNameIgnoreCase("10a")).thenReturn(Optional.empty());
        when(classRepository.save(any(SchoolClass.class))).thenAnswer(inv -> inv.getArgument(0));

        SchoolClass saved = service.saveClass(sc);

        assertThat(saved.getName()).isEqualTo("10a");
        assertThat(saved.getForeignLanguage()).isEqualTo(ForeignLanguage.EN);
    }

    @Test
    void generatesBatchOfPseudonymizedStudentsSkippingExisting() {
        UUID classId = UUID.randomUUID();
        when(studentRepository.findBySchoolClassIdAndExternalId(classId, "S01"))
                .thenReturn(Optional.of(new Student()));
        when(studentRepository.findBySchoolClassIdAndExternalId(classId, "S02"))
                .thenReturn(Optional.empty());
        when(studentRepository.findBySchoolClassIdAndExternalId(classId, "S03"))
                .thenReturn(Optional.empty());

        int created = service.generateStudentsBatch(classId, "S", 1, 3);

        assertThat(created).isEqualTo(2);
        verify(studentRepository, times(2)).save(any(Student.class));
    }

    @Test
    void rejectsDuplicateStudentExternalIdInSameClass() {
        UUID classId = UUID.randomUUID();
        Student existing = new Student();
        existing.setId(UUID.randomUUID());
        existing.setSchoolClassId(classId);
        existing.setExternalId("S05");
        when(studentRepository.findBySchoolClassIdAndExternalId(classId, "S05")).thenReturn(Optional.of(existing));

        Student duplicate = new Student();
        duplicate.setSchoolClassId(classId);
        duplicate.setExternalId("S05");

        assertThatThrownBy(() -> service.saveStudent(duplicate))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("already exists");
    }
}
