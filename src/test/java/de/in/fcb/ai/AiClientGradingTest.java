package de.in.fcb.ai;

import static org.assertj.core.api.Assertions.assertThat;

import java.math.BigDecimal;
import java.util.List;

import org.junit.jupiter.api.Test;

import de.in.fcb.exams.ForeignLanguage;
import de.in.fcb.exams.TestDefinition;
import de.in.fcb.exams.TestTask;

class AiClientGradingTest {

    private final AiClient aiClient = new AiClient();

    @Test
    void parsesExtractedTasksFromFencedJsonResponse() throws Exception {
        String rawResponse = """
                {
                  "choices": [
                    {
                      "message": {
                        "content": "```json\\n{\\"tasks\\":[{\\"position\\":1,\\"title\\":\\"Reading\\",\\"taskText\\":\\"Read text A\\",\\"referenceText\\":\\"Answer A\\",\\"gradingHints\\":\\"Content first\\",\\"maxPoints\\":12.5}]}\\n```"
                      }
                    }
                  ]
                }
                """;

        List<ExtractedTask> tasks = aiClient.parseExtractedTasksResponse(rawResponse);

        assertThat(tasks).hasSize(1);
        assertThat(tasks.get(0).position()).isEqualTo(1);
        assertThat(tasks.get(0).title()).isEqualTo("Reading");
        assertThat(tasks.get(0).maxPoints()).isEqualByComparingTo("12.5");
    }

    @Test
    void buildsMultimodalGradingRequestWithPseudonymizedIdOnly() throws Exception {
        AiConfig config = new AiConfig(AiPreset.GEMINI, "https://example/v1", "gemini-2.5-flash", "key");
        TestDefinition test = new TestDefinition();
        test.setTitle("English Essay");
        test.setForeignLanguage(ForeignLanguage.EN);
        test.setGradeLevel("9");

        TestTask task = new TestTask();
        task.setPosition(1);
        task.setTitle("Essay");
        task.setTaskText("Write a short email.");
        task.setMaxPoints(new BigDecimal("10.00"));

        String json = aiClient.buildGradingRequestJson(
                config, test, List.of(task), "S07", List.of(new byte[] { 1, 2, 3 }));

        assertThat(json).contains("S07");
        assertThat(json).contains("data:image/png;base64,");
        assertThat(json).contains("Write a short email.");
    }

    @Test
    void parsesGradingResponseClampsPointsAndExtractsTokens() throws Exception {
        TestTask task = new TestTask();
        task.setPosition(1);
        task.setTitle("Grammar");
        task.setMaxPoints(new BigDecimal("10.00"));

        String rawResponse = """
                {
                  "usage": {
                    "prompt_tokens": 350,
                    "completion_tokens": 125
                  },
                  "choices": [
                    {
                      "message": {
                        "content": "{\\"recognizedText\\":\\"I went home.\\",\\"overallFeedback\\":\\"Gute Arbeit.\\",\\"tasks\\":[{\\"position\\":1,\\"taskTitle\\":\\"Grammar\\",\\"recognizedAnswer\\":\\"I went home.\\",\\"awardedPoints\\":15.0,\\"maxPoints\\":10.0,\\"feedback\\":\\"Korrekt\\",\\"errorAnnotations\\":\\"\\"}]}"
                      }
                    }
                  ]
                }
                """;

        GradingResult result = aiClient.parseGradingResponse(rawResponse, List.of(task), "gemini-2.5-flash");

        assertThat(result.recognizedText()).isEqualTo("I went home.");
        assertThat(result.overallFeedback()).isEqualTo("Gute Arbeit.");
        assertThat(result.promptTokens()).isEqualTo(350);
        assertThat(result.completionTokens()).isEqualTo(125);
        assertThat(result.modelUsed()).isEqualTo("gemini-2.5-flash");
        assertThat(result.tasks()).hasSize(1);
        assertThat(result.tasks().get(0).awardedPoints()).isEqualByComparingTo("10.00");
    }
}
