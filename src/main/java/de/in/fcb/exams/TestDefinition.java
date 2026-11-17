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
 * Reusable exam template containing metadata and ordered {@link TestTask}s.
 */
@Entity
@Table(name = "test_definition")
public class TestDefinition {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "title", nullable = false, length = 150)
    private String title;

    @Enumerated(EnumType.STRING)
    @Column(name = "foreign_language", nullable = false, length = 30)
    private ForeignLanguage foreignLanguage = ForeignLanguage.EN;

    @Column(name = "grade_level", length = 20)
    private String gradeLevel;

    @Column(name = "learning_unit", length = 100)
    private String learningUnit;

    @Column(name = "source_text", columnDefinition = "text")
    private String sourceText;

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

    public ForeignLanguage getForeignLanguage() {
        return foreignLanguage;
    }

    public void setForeignLanguage(ForeignLanguage foreignLanguage) {
        this.foreignLanguage = foreignLanguage;
    }

    public String getGradeLevel() {
        return gradeLevel;
    }

    public void setGradeLevel(String gradeLevel) {
        this.gradeLevel = gradeLevel;
    }

    public String getLearningUnit() {
        return learningUnit;
    }

    public void setLearningUnit(String learningUnit) {
        this.learningUnit = learningUnit;
    }

    public String getSourceText() {
        return sourceText;
    }

    public void setSourceText(String sourceText) {
        this.sourceText = sourceText;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }
}
