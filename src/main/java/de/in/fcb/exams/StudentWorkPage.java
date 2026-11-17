package de.in.fcb.exams;

import java.time.OffsetDateTime;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

/**
 * Metadata for an uploaded and optionally redacted scan page stored on the filesystem.
 */
@Entity
@Table(name = "student_work_page")
public class StudentWorkPage {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "student_work_id", nullable = false)
    private UUID studentWorkId;

    @Column(name = "page_number", nullable = false)
    private int pageNumber;

    @Column(name = "storage_path", nullable = false, length = 500)
    private String storagePath;

    @Column(name = "mime_type", nullable = false, length = 60)
    private String mimeType = "image/png";

    @Column(name = "redaction_boxes_json", columnDefinition = "text")
    private String redactionBoxesJson;

    @Column(name = "created_at", nullable = false, insertable = false, updatable = false)
    private OffsetDateTime createdAt;

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public UUID getStudentWorkId() {
        return studentWorkId;
    }

    public void setStudentWorkId(UUID studentWorkId) {
        this.studentWorkId = studentWorkId;
    }

    public int getPageNumber() {
        return pageNumber;
    }

    public void setPageNumber(int pageNumber) {
        this.pageNumber = pageNumber;
    }

    public String getStoragePath() {
        return storagePath;
    }

    public void setStoragePath(String storagePath) {
        this.storagePath = storagePath;
    }

    public String getMimeType() {
        return mimeType;
    }

    public void setMimeType(String mimeType) {
        this.mimeType = mimeType;
    }

    public String getRedactionBoxesJson() {
        return redactionBoxesJson;
    }

    public void setRedactionBoxesJson(String redactionBoxesJson) {
        this.redactionBoxesJson = redactionBoxesJson;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }
}
