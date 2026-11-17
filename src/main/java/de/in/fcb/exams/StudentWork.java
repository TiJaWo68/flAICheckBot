package de.in.fcb.exams;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;

/**
 * Individual student exam submission within an {@link Assignment}, holding the
 * recognized transcription, task-by-task grading JSON, token usage, and review state.
 */
@Entity
@Table(name = "student_work")
public class StudentWork {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "assignment_id", nullable = false)
    private UUID assignmentId;

    @Column(name = "student_id", nullable = false)
    private UUID studentId;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 20)
    private StudentWorkStatus status = StudentWorkStatus.NEW;

    @Column(name = "recognized_text", columnDefinition = "text")
    private String recognizedText;

    @Column(name = "grading_json", columnDefinition = "text")
    private String gradingJson;

    @Column(name = "overall_feedback", columnDefinition = "text")
    private String overallFeedback;

    @Column(name = "achieved_points", nullable = false, precision = 6, scale = 2)
    private BigDecimal achievedPoints = BigDecimal.ZERO;

    @Column(name = "max_points", nullable = false, precision = 6, scale = 2)
    private BigDecimal maxPoints = BigDecimal.ZERO;

    @Column(name = "prompt_tokens", nullable = false)
    private int promptTokens;

    @Column(name = "completion_tokens", nullable = false)
    private int completionTokens;

    @Column(name = "model_used", length = 80)
    private String modelUsed;

    @Column(name = "updated_at", nullable = false)
    private OffsetDateTime updatedAt = OffsetDateTime.now();

    @PrePersist
    @PreUpdate
    void touchUpdatedAt() {
        this.updatedAt = OffsetDateTime.now();
    }

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public UUID getAssignmentId() {
        return assignmentId;
    }

    public void setAssignmentId(UUID assignmentId) {
        this.assignmentId = assignmentId;
    }

    public UUID getStudentId() {
        return studentId;
    }

    public void setStudentId(UUID studentId) {
        this.studentId = studentId;
    }

    public StudentWorkStatus getStatus() {
        return status;
    }

    public void setStatus(StudentWorkStatus status) {
        this.status = status;
    }

    public String getRecognizedText() {
        return recognizedText;
    }

    public void setRecognizedText(String recognizedText) {
        this.recognizedText = recognizedText;
    }

    public String getGradingJson() {
        return gradingJson;
    }

    public void setGradingJson(String gradingJson) {
        this.gradingJson = gradingJson;
    }

    public String getOverallFeedback() {
        return overallFeedback;
    }

    public void setOverallFeedback(String overallFeedback) {
        this.overallFeedback = overallFeedback;
    }

    public BigDecimal getAchievedPoints() {
        return achievedPoints;
    }

    public void setAchievedPoints(BigDecimal achievedPoints) {
        this.achievedPoints = achievedPoints;
    }

    public BigDecimal getMaxPoints() {
        return maxPoints;
    }

    public void setMaxPoints(BigDecimal maxPoints) {
        this.maxPoints = maxPoints;
    }

    public int getPromptTokens() {
        return promptTokens;
    }

    public void setPromptTokens(int promptTokens) {
        this.promptTokens = promptTokens;
    }

    public int getCompletionTokens() {
        return completionTokens;
    }

    public void setCompletionTokens(int completionTokens) {
        this.completionTokens = completionTokens;
    }

    public String getModelUsed() {
        return modelUsed;
    }

    public void setModelUsed(String modelUsed) {
        this.modelUsed = modelUsed;
    }

    public OffsetDateTime getUpdatedAt() {
        return updatedAt;
    }
}
