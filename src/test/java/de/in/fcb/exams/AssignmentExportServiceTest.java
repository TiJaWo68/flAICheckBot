package de.in.fcb.exams;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class AssignmentExportServiceTest {

    private AssignmentService assignmentService;
    private AssignmentExportService exportService;

    @BeforeEach
    void setUp() {
        assignmentService = mock(AssignmentService.class);
        exportService = new AssignmentExportService(assignmentService);
    }

    @Test
    void exportsValidPdfAndCsvWithUtf8Characters() throws Exception {
        UUID studentId = UUID.randomUUID();
        SchoolClass schoolClass = new SchoolClass();
        schoolClass.setName("8b");
        TestDefinition test = new TestDefinition();
        test.setTitle("Französisch Unit 3");
        Assignment assignment = new Assignment();
        assignment.setTitle("Klassenarbeit Französisch");
        assignment.setExamDate(LocalDate.of(2026, 11, 17));

        Student student = new Student();
        student.setId(studentId);
        student.setExternalId("S01");
        student.setDisplayAlias("Max Müller");

        StudentWork work = new StudentWork();
        work.setStudentId(studentId);
        work.setStatus(StudentWorkStatus.REVIEWED);
        work.setAchievedPoints(new BigDecimal("18.00"));
        work.setMaxPoints(new BigDecimal("20.00"));
        work.setOverallFeedback("Très bien! Guter Ausdruck und präzise Grammatik.");
        work.setGradingJson("[{\"position\":1}]");

        when(assignmentService.parseTaskEvaluations(work.getGradingJson())).thenReturn(List.of(
                new TaskEvaluation(1, "Compréhension", "Oui", new BigDecimal("18.00"), new BigDecimal("20.00"),
                        "Sehr gut gelöst.", "")));

        byte[] pdfBytes = exportService.exportPdf(
                assignment, schoolClass, test, List.of(work), Map.of(studentId, student));
        assertThat(pdfBytes).isNotEmpty();
        try (PDDocument pdf = Loader.loadPDF(pdfBytes)) {
            String extracted = new PDFTextStripper().getText(pdf);
            assertThat(extracted).contains("Klassenarbeit");
            assertThat(extracted).contains("S01");
        }

        byte[] csvBytes = exportService.exportCsv(assignment, List.of(work), Map.of(studentId, student));
        String csv = new String(csvBytes, StandardCharsets.UTF_8);
        assertThat(csv).startsWith("\uFEFF");
        assertThat(csv).contains("S01");
        assertThat(csv).contains("Max Müller");
        assertThat(csv).contains("90.0");
    }
}
