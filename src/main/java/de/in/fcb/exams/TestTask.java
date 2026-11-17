package de.in.fcb.exams;

import java.math.BigDecimal;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

/**
 * Single graded task within a {@link TestDefinition}.
 */
@Entity
@Table(name = "test_task")
public class TestTask {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "test_definition_id", nullable = false)
    private UUID testDefinitionId;

    @Column(name = "position", nullable = false)
    private int position;

    @Column(name = "title", nullable = false, length = 120)
    private String title;

    @Column(name = "task_text", nullable = false, columnDefinition = "text")
    private String taskText;

    @Column(name = "reference_text", columnDefinition = "text")
    private String referenceText;

    @Column(name = "grading_hints", columnDefinition = "text")
    private String gradingHints;

    @Column(name = "max_points", nullable = false, precision = 6, scale = 2)
    private BigDecimal maxPoints = new BigDecimal("10.00");

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public UUID getTestDefinitionId() {
        return testDefinitionId;
    }

    public void setTestDefinitionId(UUID testDefinitionId) {
        this.testDefinitionId = testDefinitionId;
    }

    public int getPosition() {
        return position;
    }

    public void setPosition(int position) {
        this.position = position;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getTaskText() {
        return taskText;
    }

    public void setTaskText(String taskText) {
        this.taskText = taskText;
    }

    public String getReferenceText() {
        return referenceText;
    }

    public void setReferenceText(String referenceText) {
        this.referenceText = referenceText;
    }

    public String getGradingHints() {
        return gradingHints;
    }

    public void setGradingHints(String gradingHints) {
        this.gradingHints = gradingHints;
    }

    public BigDecimal getMaxPoints() {
        return maxPoints;
    }

    public void setMaxPoints(BigDecimal maxPoints) {
        this.maxPoints = maxPoints;
    }
}
