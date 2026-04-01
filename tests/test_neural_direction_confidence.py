from __future__ import annotations

import unittest

import numpy as np

from pancakebot.domain.models.neural_direction_confidence import (
    apply_temperature_calibrator_to_probs,
    chosen_side_confidence,
    fit_temperature_calibrator_from_probs,
    summarize_confidence_buckets,
)


class NeuralDirectionConfidenceTests(unittest.TestCase):
    def test_temperature_calibrator_preserves_probability_order(self) -> None:
        probs = np.asarray([0.55, 0.60, 0.80, 0.20, 0.40], dtype=np.float32)
        labels = np.asarray([1, 1, 1, 0, 0], dtype=np.int64)
        calibrator = fit_temperature_calibrator_from_probs(
            bull_probs=probs,
            labels=labels,
            max_steps=50,
            learning_rate=0.05,
        )
        calibrated = apply_temperature_calibrator_to_probs(
            bull_probs=probs,
            calibrator=calibrator,
        )
        self.assertEqual(np.argsort(probs).tolist(), np.argsort(calibrated).tolist())
        self.assertGreater(float(calibrator.temperature), 0.0)

    def test_confidence_buckets_select_top_confidence_rows(self) -> None:
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
        preds = np.asarray([1, 0, 0, 0], dtype=np.int64)
        bull_probs = np.asarray([0.90, 0.30, 0.40, 0.10], dtype=np.float32)
        confidence = chosen_side_confidence(
            predicted_labels=preds,
            calibrated_bull_probs=bull_probs,
        )
        buckets = summarize_confidence_buckets(
            labels=labels,
            predicted_labels=preds,
            confidence=confidence,
            coverage_fractions=(1.0, 0.5, 0.25),
        )
        self.assertEqual(len(buckets), 3)
        self.assertEqual(buckets[0].selected_count, 4)
        self.assertEqual(buckets[1].selected_count, 2)
        self.assertEqual(buckets[2].selected_count, 1)
        self.assertEqual(buckets[2].selected_win_rate, 1.0)
        self.assertEqual(
            buckets[2].selected_min_confidence,
            buckets[2].selected_max_confidence,
        )


if __name__ == "__main__":
    unittest.main()
