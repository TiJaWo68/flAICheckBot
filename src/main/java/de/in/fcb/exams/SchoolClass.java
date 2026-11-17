package de.in.fcb.exams;

import java.time.OffsetDateTime;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

/**
 * School class / learning group containing pseudonymized students.
 */
@Entity
@Table(name = "school_class")
public class SchoolClass {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "name", nullable = false, unique = true, length = 80)
    private String name;

    @Column(name = "academic_year", nullable = false, length = 20)
    private String academicYear;

    @Column(name = "grade_level", length = 20)
    private String gradeLevel;

    @Enumerated(EnumType.STRING)
    @Column(name = "foreign_language", nullable = false, length = 30)
    private ForeignLanguage foreignLanguage = ForeignLanguage.EN;

    @Column(name = "created_at", nullable = false, insertable = false, updatable = false)
    private OffsetDateTime createdAt;

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getAcademicYear() {
        return academicYear;
    }

    public void setAcademicYear(String academicYear) {
        this.academicYear = academicYear;
    }

    public String getGradeLevel() {
        return gradeLevel;
    }

    public void setGradeLevel(String gradeLevel) {
        this.gradeLevel = gradeLevel;
    }

    public ForeignLanguage getForeignLanguage() {
        return foreignLanguage;
    }

    public void setForeignLanguage(ForeignLanguage foreignLanguage) {
        this.foreignLanguage = foreignLanguage;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }
}
