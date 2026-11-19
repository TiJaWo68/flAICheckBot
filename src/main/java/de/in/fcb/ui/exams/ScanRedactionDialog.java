package de.in.fcb.ui.exams;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.vaadin.flow.component.ClientCallable;
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
import com.vaadin.flow.component.upload.Upload;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.exams.AssignmentService;
import de.in.fcb.exams.ScanStorageService;
import de.in.fcb.exams.Student;
import de.in.fcb.exams.StudentWork;
import de.in.fcb.exams.StudentWorkPage;

/**
 * Dialog for uploading scan pages (PDF or images), previewing them, rotating
 * pages, interactively drawing black redaction boxes with the mouse, and
 * undoing or resetting redactions without losing the unredacted local backup.
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
    private final Button undoRedactionButton = new Button();
    private final Button clearRedactionsButton = new Button();
    private final Button deletePageButton = new Button();

    private final Span drawHint = new Span();
    private final Image previewImage = new Image();
    private final Div selectionOverlay = new Div();
    private final Div imageStage;
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

        undoRedactionButton.setText(getTranslation("exams.scans.button.undo"));
        undoRedactionButton.setIcon(VaadinIcon.ROTATE_LEFT.create());
        undoRedactionButton.addClickListener(e -> undoLastRedaction());

        clearRedactionsButton.setText(getTranslation("exams.scans.button.reset-redactions"));
        clearRedactionsButton.setIcon(VaadinIcon.CLOSE_CIRCLE_O.create());
        clearRedactionsButton.addThemeVariants(ButtonVariant.LUMO_CONTRAST);
        clearRedactionsButton.addClickListener(e -> clearAllRedactions());

        deletePageButton.setText(getTranslation("exams.scans.button.delete-page"));
        deletePageButton.setIcon(VaadinIcon.TRASH.create());
        deletePageButton.addThemeVariants(ButtonVariant.LUMO_ERROR);
        deletePageButton.addClickListener(e -> deleteCurrentPage());

        HorizontalLayout pageControls = new HorizontalLayout(
                prevButton, pageIndicator, nextButton, rotateButton,
                undoRedactionButton, clearRedactionsButton, deletePageButton);
        pageControls.setWidthFull();
        pageControls.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        pageControls.getStyle().set("flex-wrap", "wrap");

        drawHint.setText(getTranslation("exams.scans.draw-hint"));
        drawHint.addClassNames(LumoUtility.FontSize.SMALL, LumoUtility.TextColor.SECONDARY);

        previewImage.getElement().setAttribute("draggable", "false");
        previewImage.getStyle()
                .set("display", "block")
                .set("max-width", "100%")
                .set("user-select", "none")
                .set("border", "1px solid var(--lumo-contrast-20pct)");

        selectionOverlay.getStyle()
                .set("position", "absolute")
                .set("border", "2px dashed #ff5252")
                .set("background", "rgba(0, 0, 0, 0.45)")
                .set("pointer-events", "none")
                .set("display", "none")
                .set("box-sizing", "border-box");

        imageStage = new Div(previewImage, selectionOverlay);
        imageStage.getStyle()
                .set("position", "relative")
                .set("display", "inline-block")
                .set("user-select", "none")
                .set("cursor", "crosshair")
                .set("line-height", "0");
        installMouseDrawingHandler();

        emptyLabel.setText(getTranslation("exams.scans.empty"));
        emptyLabel.addClassNames(LumoUtility.TextColor.SECONDARY);

        Div previewContainer = new Div(emptyLabel, imageStage);
        previewContainer.setWidthFull();
        previewContainer.getStyle().set("overflow", "auto");
        previewContainer.getStyle().set("flex", "1 1 auto");
        previewContainer.getStyle().set("text-align", "center");

        Button closeButton = new Button(getTranslation("ui.dialog.close"), e -> close());
        closeButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        getFooter().add(closeButton);

        VerticalLayout root = new VerticalLayout(upload, pageControls, drawHint, previewContainer);
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

    @ClientCallable
    public void onDrawRedaction(double normX, double normY, double normWidth, double normHeight) {
        if (normWidth <= 0.0 || normHeight <= 0.0) {
            return;
        }
        redactCurrentPage(normX, normY, normWidth, normHeight);
    }

    private void installMouseDrawingHandler() {
        imageStage.getElement().executeJs("""
                (function(host, stage, img, overlay) {
                  if (stage._redactBound) return;
                  stage._redactBound = true;
                  let dragging = false;
                  let startX = 0;
                  let startY = 0;
                  let rect = null;

                  function clamp(val, min, max) {
                    return Math.max(min, Math.min(max, val));
                  }

                  function updateBox(clientX, clientY) {
                    const curX = clamp(clientX - rect.left, 0, rect.width);
                    const curY = clamp(clientY - rect.top, 0, rect.height);
                    const x = Math.min(startX, curX);
                    const y = Math.min(startY, curY);
                    const w = Math.abs(curX - startX);
                    const h = Math.abs(curY - startY);
                    overlay.style.left = x + 'px';
                    overlay.style.top = y + 'px';
                    overlay.style.width = w + 'px';
                    overlay.style.height = h + 'px';
                    return { x: x, y: y, w: w, h: h };
                  }

                  stage.addEventListener('pointerdown', function(e) {
                    if (e.button !== 0) return;
                    rect = img.getBoundingClientRect();
                    if (!rect || rect.width <= 0 || rect.height <= 0) return;
                    e.preventDefault();
                    dragging = true;
                    startX = clamp(e.clientX - rect.left, 0, rect.width);
                    startY = clamp(e.clientY - rect.top, 0, rect.height);
                    overlay.style.display = 'block';
                    updateBox(e.clientX, e.clientY);
                    if (stage.setPointerCapture) {
                      try { stage.setPointerCapture(e.pointerId); } catch (_) {}
                    }
                  });

                  stage.addEventListener('pointermove', function(e) {
                    if (!dragging) return;
                    e.preventDefault();
                    updateBox(e.clientX, e.clientY);
                  });

                  function finish(e, commit) {
                    if (!dragging) return;
                    dragging = false;
                    const box = updateBox(e.clientX, e.clientY);
                    overlay.style.display = 'none';
                    if (stage.releasePointerCapture) {
                      try { stage.releasePointerCapture(e.pointerId); } catch (_) {}
                    }
                    if (commit && rect && rect.width > 0 && rect.height > 0 && box.w >= 8 && box.h >= 8) {
                      host.$server.onDrawRedaction(
                        box.x / rect.width,
                        box.y / rect.height,
                        box.w / rect.width,
                        box.h / rect.height
                      );
                    }
                  }

                  stage.addEventListener('pointerup', function(e) { finish(e, true); });
                  stage.addEventListener('pointercancel', function(e) { finish(e, false); });
                })($0, $1, $2, $3);
                """, getElement(), imageStage.getElement(), previewImage.getElement(), selectionOverlay.getElement());
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
        deletePageButton.setEnabled(hasPages);
        imageStage.setVisible(hasPages);
        drawHint.setVisible(hasPages);
        emptyLabel.setVisible(!hasPages);

        if (!hasPages) {
            pageIndicator.setText("0 / 0");
            undoRedactionButton.setEnabled(false);
            clearRedactionsButton.setEnabled(false);
            return;
        }

        currentIndex = Math.max(0, Math.min(index, pages.size() - 1));
        pageIndicator.setText((currentIndex + 1) + " / " + pages.size());

        StudentWorkPage page = pages.get(currentIndex);
        boolean hasRedactions = !assignmentService.parseRedactionBoxes(page.getRedactionBoxesJson()).isEmpty();
        undoRedactionButton.setEnabled(hasRedactions);
        clearRedactionsButton.setEnabled(hasRedactions);

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
            reloadPages();
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
            reloadPages();
            Notification.show(getTranslation("exams.scans.redacted"), 2500, Notification.Position.MIDDLE);
        } catch (Exception ex) {
            Notification.show(ex.getMessage(), 4000, Notification.Position.MIDDLE);
        }
    }

    private void undoLastRedaction() {
        if (pages.isEmpty()) {
            return;
        }
        try {
            if (assignmentService.undoLastRedaction(pages.get(currentIndex).getId())) {
                reloadPages();
                Notification.show(getTranslation("exams.scans.undone"), 2500, Notification.Position.MIDDLE);
            }
        } catch (Exception ex) {
            Notification.show(ex.getMessage(), 4000, Notification.Position.MIDDLE);
        }
    }

    private void clearAllRedactions() {
        if (pages.isEmpty()) {
            return;
        }
        try {
            if (assignmentService.clearRedactions(pages.get(currentIndex).getId())) {
                reloadPages();
                Notification.show(getTranslation("exams.scans.reset"), 2500, Notification.Position.MIDDLE);
            }
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
}
