package de.in.fcb.ui.exams;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.dialog.Dialog;
import com.vaadin.flow.component.html.Div;
import com.vaadin.flow.component.html.H4;
import com.vaadin.flow.component.html.Image;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.FlexComponent.JustifyContentMode;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.splitlayout.SplitLayout;
import com.vaadin.flow.component.textfield.BigDecimalField;
import com.vaadin.flow.component.textfield.TextArea;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.theme.lumo.LumoUtility;

import de.in.fcb.exams.AssignmentService;
import de.in.fcb.exams.ScanStorageService;
import de.in.fcb.exams.Student;
import de.in.fcb.exams.StudentWork;
import de.in.fcb.exams.StudentWorkPage;
import de.in.fcb.exams.StudentWorkStatus;
import de.in.fcb.exams.TaskEvaluation;
import de.in.fcb.exams.TestTask;

/**
 * Side-by-side teacher review dialog displaying zoomable scan pages on the
 * left and editable AI transcription, per-task scores, feedback, and review
 * workflow actions on the right.
 */
public class ReviewDialog extends Dialog {

    private final AssignmentService assignmentService;
    private final ScanStorageService scanStorageService;

    private final Span pageIndicator = new Span();
    private final Button prevPageBtn = new Button(VaadinIcon.ARROW_LEFT.create());
    private final Button nextPageBtn = new Button(VaadinIcon.ARROW_RIGHT.create());
    private final Span zoomLabel = new Span("100%");
    private final Image scanImage = new Image();
    private final Span noScanLabel = new Span();

    private final Span statusBadge = new Span();
    private final Span totalScoreBadge = new Span();
    private final Span tokenInfo = new Span();
    private final TextArea recognizedTextArea = new TextArea();
    private final VerticalLayout taskCardsContainer = new VerticalLayout();
    private final TextArea overallFeedbackArea = new TextArea();

    private final List<TaskEditorRow> taskEditorRows = new ArrayList<>();
    private StudentWork currentWork;
    private List<StudentWorkPage> pages = List.of();
    private int pageIndex;
    private int zoomPercent = 100;
    private Runnable onSaved;

    public ReviewDialog(AssignmentService assignmentService, ScanStorageService scanStorageService) {
        this.assignmentService = assignmentService;
        this.scanStorageService = scanStorageService;

        setWidth("96vw");
        setHeight("92vh");
        setCloseOnEsc(true);
        setCloseOnOutsideClick(false);

        VerticalLayout leftPane = createScanViewerPane();
        VerticalLayout rightPane = createEvaluationFormPane();

        SplitLayout split = new SplitLayout(leftPane, rightPane);
        split.setSizeFull();
        split.setSplitterPosition(46);

        Button cancelButton = new Button(getTranslation("admin.button.cancel"), e -> close());
        Button saveDraftButton = new Button(getTranslation("exams.review.button.save-draft"),
                VaadinIcon.CHECK.create(), e -> save(null));
        Button markReviewedButton = new Button(getTranslation("exams.review.button.mark-reviewed"),
                VaadinIcon.CHECK_CIRCLE.create(), e -> save(StudentWorkStatus.REVIEWED));
        markReviewedButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY, ButtonVariant.LUMO_SUCCESS);

        getFooter().add(cancelButton, saveDraftButton, markReviewedButton);
        add(split);
    }

    public void openForReview(
            StudentWork work,
            Student student,
            List<TestTask> definedTasks,
            Runnable onSaved) {
        this.currentWork = work;
        this.onSaved = onSaved;
        this.pageIndex = 0;
        this.zoomPercent = 100;

        setHeaderTitle(getTranslation("exams.review.dialog.title",
                student != null ? student.formatTeacherLabel() : work.getStudentId()));

        pages = assignmentService.findPages(work.getId());
        showPage(0);

        statusBadge.setText(getTranslation("exams.status." + work.getStatus().name()));
        tokenInfo.setText(work.getModelUsed() != null
                ? work.getModelUsed() + " · " + (work.getPromptTokens() + work.getCompletionTokens()) + " Tokens"
                : "");

        recognizedTextArea.setValue(work.getRecognizedText() != null ? work.getRecognizedText() : "");
        overallFeedbackArea.setValue(work.getOverallFeedback() != null ? work.getOverallFeedback() : "");

        populateTaskRows(work, definedTasks);
        updateTotalScore();
        open();
    }

    private VerticalLayout createScanViewerPane() {
        prevPageBtn.addClickListener(e -> showPage(pageIndex - 1));
        nextPageBtn.addClickListener(e -> showPage(pageIndex + 1));

        Button zoomOutBtn = new Button(VaadinIcon.SEARCH_MINUS.create(), e -> adjustZoom(-20));
        Button zoomResetBtn = new Button("100%", e -> setZoom(100));
        Button zoomInBtn = new Button(VaadinIcon.SEARCH_PLUS.create(), e -> adjustZoom(20));

        HorizontalLayout toolbar = new HorizontalLayout(
                prevPageBtn, pageIndicator, nextPageBtn, zoomOutBtn, zoomLabel, zoomResetBtn, zoomInBtn);
        toolbar.setWidthFull();
        toolbar.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        toolbar.getStyle().set("flex-wrap", "wrap");

        noScanLabel.setText(getTranslation("exams.scans.empty"));
        noScanLabel.addClassNames(LumoUtility.TextColor.SECONDARY);

        Div imageBox = new Div(noScanLabel, scanImage);
        imageBox.setWidthFull();
        imageBox.getStyle().set("overflow", "auto");
        imageBox.getStyle().set("flex", "1 1 auto");

        VerticalLayout left = new VerticalLayout(toolbar, imageBox);
        left.setSizeFull();
        left.setPadding(false);
        left.setSpacing(true);
        left.expand(imageBox);
        return left;
    }

    private VerticalLayout createEvaluationFormPane() {
        statusBadge.getElement().getThemeList().add("badge pill");
        totalScoreBadge.getElement().getThemeList().add("badge pill success");
        tokenInfo.addClassNames(LumoUtility.TextColor.SECONDARY, LumoUtility.FontSize.SMALL);

        HorizontalLayout metaBar = new HorizontalLayout(statusBadge, totalScoreBadge, tokenInfo);
        metaBar.setWidthFull();
        metaBar.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        metaBar.setJustifyContentMode(JustifyContentMode.START);
        metaBar.getStyle().set("flex-wrap", "wrap");

        recognizedTextArea.setLabel(getTranslation("exams.review.field.recognized-text"));
        recognizedTextArea.setWidthFull();
        recognizedTextArea.setMinHeight("130px");

        taskCardsContainer.setWidthFull();
        taskCardsContainer.setPadding(false);
        taskCardsContainer.setSpacing(true);

        overallFeedbackArea.setLabel(getTranslation("exams.review.field.overall-feedback"));
        overallFeedbackArea.setWidthFull();
        overallFeedbackArea.setMinHeight("100px");

        VerticalLayout scrollable = new VerticalLayout(
                metaBar, recognizedTextArea, taskCardsContainer, overallFeedbackArea);
        scrollable.setWidthFull();
        scrollable.setPadding(true);
        scrollable.setSpacing(true);
        scrollable.getStyle().set("overflow-y", "auto");
        scrollable.setHeightFull();
        return scrollable;
    }

    private void populateTaskRows(StudentWork work, List<TestTask> definedTasks) {
        taskCardsContainer.removeAll();
        taskEditorRows.clear();

        List<TaskEvaluation> existing = assignmentService.parseTaskEvaluations(work.getGradingJson());
        List<TaskEvaluation> sourceRows = new ArrayList<>();
        if (!existing.isEmpty()) {
            sourceRows.addAll(existing);
        } else {
            for (TestTask task : definedTasks) {
                sourceRows.add(new TaskEvaluation(
                        task.getPosition(),
                        task.getTitle(),
                        "",
                        BigDecimal.ZERO,
                        task.getMaxPoints(),
                        "",
                        ""));
            }
        }

        for (TaskEvaluation eval : sourceRows) {
            TaskEditorRow row = new TaskEditorRow(eval, this::updateTotalScore);
            taskEditorRows.add(row);
            taskCardsContainer.add(row.card);
        }
    }

    private void showPage(int index) {
        boolean hasPages = !pages.isEmpty();
        prevPageBtn.setEnabled(hasPages && index > 0);
        nextPageBtn.setEnabled(hasPages && index < pages.size() - 1);
        scanImage.setVisible(hasPages);
        noScanLabel.setVisible(!hasPages);

        if (!hasPages) {
            pageIndicator.setText("0 / 0");
            return;
        }
        pageIndex = Math.max(0, Math.min(index, pages.size() - 1));
        pageIndicator.setText((pageIndex + 1) + " / " + pages.size());
        setZoom(zoomPercent);

        try {
            byte[] png = scanStorageService.readImageBytes(pages.get(pageIndex).getStoragePath());
            scanImage.setSrc("data:image/png;base64," + Base64.getEncoder().encodeToString(png));
        } catch (IOException ex) {
            scanImage.setSrc("");
        }
    }

    private void adjustZoom(int delta) {
        setZoom(Math.max(40, Math.min(260, zoomPercent + delta)));
    }

    private void setZoom(int percent) {
        this.zoomPercent = percent;
        zoomLabel.setText(percent + "%");
        scanImage.setWidth(percent + "%");
    }

    private void updateTotalScore() {
        BigDecimal achieved = BigDecimal.ZERO;
        BigDecimal max = BigDecimal.ZERO;
        for (TaskEditorRow row : taskEditorRows) {
            TaskEvaluation eval = row.toEvaluation();
            achieved = achieved.add(eval.awardedPoints());
            max = max.add(eval.maxPoints());
        }
        totalScoreBadge.setText(achieved.stripTrailingZeros().toPlainString() + " / "
                + max.stripTrailingZeros().toPlainString() + " Pkt");
    }

    private void save(StudentWorkStatus targetStatus) {
        if (currentWork == null) {
            return;
        }
        List<TaskEvaluation> evaluations = taskEditorRows.stream()
                .map(TaskEditorRow::toEvaluation)
                .toList();
        StudentWorkStatus effectiveStatus = targetStatus != null
                ? targetStatus
                : (currentWork.getStatus() == StudentWorkStatus.NEW
                        || currentWork.getStatus() == StudentWorkStatus.UPLOADED
                        ? StudentWorkStatus.AI_GRADED
                        : currentWork.getStatus());

        assignmentService.updateReview(
                currentWork.getId(),
                recognizedTextArea.getValue(),
                overallFeedbackArea.getValue(),
                evaluations,
                effectiveStatus);
        close();
        if (onSaved != null) {
            onSaved.run();
        }
        Notification.show(getTranslation("admin.notification.saved"), 3000, Notification.Position.MIDDLE);
    }

    private class TaskEditorRow {
        private final int position;
        private final String taskTitle;
        private final BigDecimal maxPoints;
        private final TextArea answerArea = new TextArea();
        private final BigDecimalField pointsField = new BigDecimalField();
        private final TextArea feedbackArea = new TextArea();
        private final TextField annotationsField = new TextField();
        private final VerticalLayout card = new VerticalLayout();

        TaskEditorRow(TaskEvaluation eval, Runnable onScoreChange) {
            this.position = eval.position();
            this.taskTitle = eval.taskTitle();
            this.maxPoints = eval.maxPoints() != null ? eval.maxPoints() : BigDecimal.TEN;

            card.setWidthFull();
            card.setPadding(true);
            card.setSpacing(false);
            card.addClassNames(
                    LumoUtility.Background.CONTRAST_5,
                    LumoUtility.BorderRadius.MEDIUM,
                    LumoUtility.Padding.MEDIUM);

            H4 title = new H4(position + ". " + taskTitle);
            title.addClassNames(LumoUtility.Margin.NONE, LumoUtility.FontSize.MEDIUM);

            pointsField.setLabel(getTranslation("exams.review.field.points"));
            pointsField.setValue(eval.awardedPoints() != null ? eval.awardedPoints() : BigDecimal.ZERO);
            pointsField.setSuffixComponent(new Span("/ " + maxPoints.stripTrailingZeros().toPlainString()));
            pointsField.setWidth("170px");
            pointsField.addValueChangeListener(e -> onScoreChange.run());

            HorizontalLayout top = new HorizontalLayout(title, pointsField);
            top.setWidthFull();
            top.setDefaultVerticalComponentAlignment(Alignment.CENTER);
            top.expand(title);

            answerArea.setLabel(getTranslation("exams.review.field.answer"));
            answerArea.setValue(eval.recognizedAnswer() != null ? eval.recognizedAnswer() : "");
            answerArea.setWidthFull();

            feedbackArea.setLabel(getTranslation("exams.review.field.task-feedback"));
            feedbackArea.setValue(eval.feedback() != null ? eval.feedback() : "");
            feedbackArea.setWidthFull();

            annotationsField.setLabel(getTranslation("exams.review.field.annotations"));
            annotationsField.setValue(eval.errorAnnotations() != null ? eval.errorAnnotations() : "");
            annotationsField.setWidthFull();

            card.add(top, answerArea, feedbackArea, annotationsField);
        }

        TaskEvaluation toEvaluation() {
            BigDecimal pts = pointsField.getValue() != null ? pointsField.getValue() : BigDecimal.ZERO;
            if (pts.signum() < 0) {
                pts = BigDecimal.ZERO;
            }
            if (pts.compareTo(maxPoints) > 0) {
                pts = maxPoints;
            }
            return new TaskEvaluation(
                    position,
                    taskTitle,
                    answerArea.getValue(),
                    pts,
                    maxPoints,
                    feedbackArea.getValue(),
                    annotationsField.getValue());
        }
    }
}
