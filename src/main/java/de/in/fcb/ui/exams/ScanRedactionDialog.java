package de.in.fcb.ui.exams;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.dialog.Dialog;
import com.vaadin.flow.component.html.Div;
import com.vaadin.flow.component.html.Image;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.NumberField;
import com.vaadin.flow.component.upload.Upload;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.exams.AssignmentService;
import de.in.fcb.exams.ScanStorageService;
import de.in.fcb.exams.Student;
import de.in.fcb.exams.StudentWork;
import de.in.fcb.exams.StudentWorkPage;

/**
 * Dialog for uploading scan pages (PDF or images), previewing them, rotating
 * pages, and permanently burning black redaction rectangles onto stored PNGs
 * prior to sending them to an external AI provider.
 */
public class ScanRedactionDialog extends Dialog {

    private final AssignmentService assignmentService;
    private final ScanStorageService scanStorageService;

    private final Upload upload;
    private final Map<String, ByteArrayOutputStream> buffers = new HashMap<>();

    private final Span pageIndicator = new Span();
    private final Button prevButton = new Button(VaadinIcon.ARROW_LEFT.create());
    private final Button nextButton = new Button(VaadinIcon.ARROW_RIGHT.create());
    private final Button rotateButton = new Button();
    private final Button redactHeaderButton = new Button();
    private final Button deletePageButton = new Button();

    private final NumberField xPercent = new NumberField("X %");
    private final NumberField yPercent = new NumberField("Y %");
    private final NumberField wPercent = new NumberField("Breite %");
    private final NumberField hPercent = new NumberField("Höhe %");
    private final Button redactCustomButton = new Button();

    private final Image previewImage = new Image();
    private final Span emptyLabel = new Span();

    private StudentWork currentWork;
    private List<StudentWorkPage> pages = List.of();
    private int currentIndex;
    private Runnable onChanged;

    public ScanRedactionDialog(AssignmentService assignmentService, ScanStorageService scanStorageService) {
        this.assignmentService = assignmentService;
        this.scanStorageService = scanStorageService;

        setWidth("1080px");
        setMaxWidth("95vw");
        setHeight("88vh");
        setCloseOnEsc(true);
        setCloseOnOutsideClick(false);

        upload = new Upload(this::receiveUpload);
        upload.setAcceptedFileTypes(".pdf", ".png", ".jpg", ".jpeg");
        upload.setMaxFiles(20);
        upload.setUploadButton(new Button(getTranslation("exams.scans.upload.button"), VaadinIcon.UPLOAD.create()));
        upload.setDropLabel(new Span(getTranslation("exams.scans.upload.drop")));
        upload.setWidthFull();
        upload.addSucceededListener(event -> handleUpload(event.getFileName()));

        prevButton.addClickListener(e -> showPage(currentIndex - 1));
        nextButton.addClickListener(e -> showPage(currentIndex + 1));

        rotateButton.setText(getTranslation("exams.scans.button.rotate"));
        rotateButton.setIcon(VaadinIcon.ROTATE_RIGHT.create());
        rotateButton.addClickListener(e -> rotateCurrentPage());

        redactHeaderButton.setText(getTranslation("exams.scans.button.redact-header"));
        redactHeaderButton.setIcon(VaadinIcon.EYE_SLASH.create());
        redactHeaderButton.addThemeVariants(ButtonVariant.LUMO_CONTRAST);
        redactHeaderButton.addClickListener(e -> redactCurrentPage(0.0, 0.0, 1.0, 0.15));

        deletePageButton.setText(getTranslation("exams.scans.button.delete-page"));
        deletePageButton.setIcon(VaadinIcon.TRASH.create());
        deletePageButton.addThemeVariants(ButtonVariant.LUMO_ERROR);
        deletePageButton.addClickListener(e -> deleteCurrentPage());

        HorizontalLayout pageControls = new HorizontalLayout(
                prevButton, pageIndicator, nextButton, rotateButton, redactHeaderButton, deletePageButton);
        pageControls.setWidthFull();
        pageControls.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        pageControls.getStyle().set("flex-wrap", "wrap");

        configureCoordField(xPercent, 0.0);
        configureCoordField(yPercent, 0.0);
        configureCoordField(wPercent, 100.0);
        configureCoordField(hPercent, 12.0);
        redactCustomButton.setText(getTranslation("exams.scans.button.redact-area"));
        redactCustomButton.setIcon(VaadinIcon.CROP.create());
        redactCustomButton.addClickListener(e -> redactCurrentPage(
                valOrZero(xPercent) / 100.0,
                valOrZero(yPercent) / 100.0,
                valOrZero(wPercent) / 100.0,
                valOrZero(hPercent) / 100.0));

        HorizontalLayout customRedactionBar = new HorizontalLayout(
                xPercent, yPercent, wPercent, hPercent, redactCustomButton);
        customRedactionBar.setWidthFull();
        customRedactionBar.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        customRedactionBar.getStyle().set("flex-wrap", "wrap");

        previewImage.setMaxWidth("100%");
        previewImage.getStyle().set("border", "1px solid var(--lumo-contrast-20pct)");

        emptyLabel.setText(getTranslation("exams.scans.empty"));
        emptyLabel.addClassNames(LumoUtility.TextColor.SECONDARY);

        Div previewContainer = new Div(emptyLabel, previewImage);
        previewContainer.setWidthFull();
        previewContainer.getStyle().set("overflow", "auto");
        previewContainer.getStyle().set("flex", "1 1 auto");
        previewContainer.getStyle().set("text-align", "center");

        Button closeButton = new Button(getTranslation("ui.dialog.close"), e -> close());
        closeButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        getFooter().add(closeButton);

        VerticalLayout root = new VerticalLayout(upload, pageControls, customRedactionBar, previewContainer);
        root.setSizeFull();
        root.setPadding(false);
        root.setSpacing(true);
        root.expand(previewContainer);
        add(root);
    }

    public void openForWork(StudentWork work, Student student, Runnable onChanged) {
        this.currentWork = work;
        this.onChanged = onChanged;
        this.currentIndex = 0;
        this.upload.clearFileList();
        this.buffers.clear();
        setHeaderTitle(getTranslation("exams.scans.dialog.title",
                student != null ? student.formatTeacherLabel() : work.getStudentId()));
        reloadPages();
        open();
    }

    private OutputStream receiveUpload(String fileName, String mimeType) {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        buffers.put(fileName, buffer);
        return buffer;
    }

    private void handleUpload(String fileName) {
        ByteArrayOutputStream buffer = buffers.remove(fileName);
        if (buffer == null || currentWork == null) {
            return;
        }
        try {
            assignmentService.uploadPages(currentWork.getId(), fileName, buffer.toByteArray());
            reloadPages();
            if (onChanged != null) {
                onChanged.run();
            }
            Notification.show(getTranslation("exams.scans.upload.success", fileName), 3000,
                    Notification.Position.MIDDLE);
        } catch (Exception ex) {
            Notification.show(getTranslation("admin.notification.error", ex.getMessage()), 5000,
                    Notification.Position.MIDDLE);
        }
    }

    private void reloadPages() {
        pages = currentWork != null ? assignmentService.findPages(currentWork.getId()) : List.of();
        if (currentIndex >= pages.size()) {
            currentIndex = Math.max(0, pages.size() - 1);
        }
        showPage(currentIndex);
    }

    private void showPage(int index) {
        boolean hasPages = !pages.isEmpty();
        prevButton.setEnabled(hasPages && index > 0);
        nextButton.setEnabled(hasPages && index < pages.size() - 1);
        rotateButton.setEnabled(hasPages);
        redactHeaderButton.setEnabled(hasPages);
        redactCustomButton.setEnabled(hasPages);
        deletePageButton.setEnabled(hasPages);
        previewImage.setVisible(hasPages);
        emptyLabel.setVisible(!hasPages);

        if (!hasPages) {
            pageIndicator.setText("0 / 0");
            return;
        }

        currentIndex = Math.max(0, Math.min(index, pages.size() - 1));
        pageIndicator.setText((currentIndex + 1) + " / " + pages.size());

        StudentWorkPage page = pages.get(currentIndex);
        try {
            byte[] bytes = scanStorageService.readImageBytes(page.getStoragePath());
            previewImage.setSrc("data:image/png;base64," + Base64.getEncoder().encodeToString(bytes));
        } catch (IOException ex) {
            previewImage.setSrc("");
            Notification.show(ex.getMessage(), 4000, Notification.Position.MIDDLE);
        }
    }

    private void rotateCurrentPage() {
        if (pages.isEmpty()) {
            return;
        }
        try {
            assignmentService.rotatePage(pages.get(currentIndex).getId(), 90);
            showPage(currentIndex);
        } catch (IOException ex) {
            Notification.show(ex.getMessage(), 4000, Notification.Position.MIDDLE);
        }
    }

    private void redactCurrentPage(double normX, double normY, double normW, double normH) {
        if (pages.isEmpty()) {
            return;
        }
        try {
            assignmentService.redactPage(pages.get(currentIndex).getId(), normX, normY, normW, normH);
            showPage(currentIndex);
            Notification.show(getTranslation("exams.scans.redacted"), 2500, Notification.Position.MIDDLE);
        } catch (Exception ex) {
            Notification.show(ex.getMessage(), 4000, Notification.Position.MIDDLE);
        }
    }

    private void deleteCurrentPage() {
        if (pages.isEmpty()) {
            return;
        }
        assignmentService.deletePage(pages.get(currentIndex).getId());
        reloadPages();
        if (onChanged != null) {
            onChanged.run();
        }
    }

    private static void configureCoordField(NumberField field, double initial) {
        field.setWidth("95px");
        field.setMin(0);
        field.setMax(100);
        field.setStep(1);
        field.setValue(initial);
    }

    private static double valOrZero(NumberField field) {
        return field.getValue() != null ? field.getValue() : 0.0;
    }
}
