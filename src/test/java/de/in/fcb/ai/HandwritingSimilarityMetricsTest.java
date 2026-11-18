package de.in.fcb.ai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

class HandwritingSimilarityMetricsTest {

    @Test
    void computesZeroErrorWhenOnlyWhitespaceDiffers() {
        HandwritingSimilarityMetrics.Metrics metrics = HandwritingSimilarityMetrics.compute(
                "In my opinion\nit is perents duty.",
                "  In   my opinion it is perents duty. ");

        assertThat(metrics.cer()).isEqualTo(0.0);
        assertThat(metrics.wer()).isEqualTo(0.0);
        assertThat(metrics.wordAccuracy()).isEqualTo(1.0);
        assertThat(metrics.illegibleCount()).isZero();
    }

    @Test
    void penalizesAutocorrectedStudentErrorsAndCountsIllegibleMarkers() {
        HandwritingSimilarityMetrics.Metrics metrics = HandwritingSimilarityMetrics.compute(
                "because they rase you up and [something]",
                "because they raise you up and [unleserlich]");

        assertThat(metrics.cer()).isGreaterThan(0.0);
        assertThat(metrics.wer()).isCloseTo(2.0 / 7.0, within(0.001));
        assertThat(metrics.wordAccuracy()).isCloseTo(5.0 / 7.0, within(0.001));
        assertThat(metrics.illegibleCount()).isEqualTo(1);
    }
}
