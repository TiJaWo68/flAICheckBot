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
 * Pseudonymized student entry belonging to a {@link SchoolClass}.
 * <p>
 * Only {@code externalId} is ever transmitted to external AI endpoints;
 * {@code displayAlias} stays strictly local for teacher reference and export.
 */
@Entity
@Table(name = "student")
public class Student {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "external_id", nullable = false, length = 60)
    private String externalId;

    @Column(name = "display_alias", length = 100)
    private String displayAlias;

    @Column(name = "school_class_id", nullable = false)
    private UUID schoolClassId;

    @Column(name = "created_at", nullable = false, insertable = false, updatable = false)
    private OffsetDateTime createdAt;

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public String getExternalId() {
        return externalId;
    }

    public void setExternalId(String externalId) {
        this.externalId = externalId;
    }

    public String getDisplayAlias() {
        return displayAlias;
    }

    public void setDisplayAlias(String displayAlias) {
        this.displayAlias = displayAlias;
    }

    public UUID getSchoolClassId() {
        return schoolClassId;
    }

    public void setSchoolClassId(UUID schoolClassId) {
        this.schoolClassId = schoolClassId;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public String formatTeacherLabel() {
        if (displayAlias == null || displayAlias.isBlank()) {
            return externalId;
        }
        return externalId + " (" + displayAlias + ")";
    }
}
