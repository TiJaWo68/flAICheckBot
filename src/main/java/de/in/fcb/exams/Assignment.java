package de.in.fcb.exams;

import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

/**
 * Concrete exam execution linking a {@link SchoolClass} with a {@link TestDefinition}.
 */
@Entity
@Table(name = "assignment")
public class Assignment {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "title", nullable = false, length = 150)
    private String title;

    @Column(name = "school_class_id", nullable = false)
    private UUID schoolClassId;

    @Column(name = "test_definition_id", nullable = false)
    private UUID testDefinitionId;

    @Column(name = "exam_date", nullable = false)
    private LocalDate examDate = LocalDate.now();

    @Column(name = "created_at", nullable = false, insertable = false, updatable = false)
    private OffsetDateTime createdAt;

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public UUID getSchoolClassId() {
        return schoolClassId;
    }

    public void setSchoolClassId(UUID schoolClassId) {
        this.schoolClassId = schoolClassId;
    }

    public UUID getTestDefinitionId() {
        return testDefinitionId;
    }

    public void setTestDefinitionId(UUID testDefinitionId) {
        this.testDefinitionId = testDefinitionId;
    }

    public LocalDate getExamDate() {
        return examDate;
    }

    public void setExamDate(LocalDate examDate) {
        this.examDate = examDate;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }
}
