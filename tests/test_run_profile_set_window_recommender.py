from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from inspection.run_profile_set_model_selector import _load_compare_rows
from inspection.run_profile_set_window_recommender import predict_next_window_recommendation


class ProfileSetWindowRecommenderTests(unittest.TestCase):
    def _write_compare_csv(self) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / "compare.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "tail_offset_rounds",
                    "stageb_per_500",
                    "stageb_bet_rate",
                    "stageg2_bullonly_per_500",
                    "stageg2_bullonly_bet_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "tail_offset_rounds": 432,
                        "stageb_per_500": -0.2,
                        "stageb_bet_rate": 0.07,
                        "stageg2_bullonly_per_500": 0.8,
                        "stageg2_bullonly_bet_rate": 0.05,
                    },
                    {
                        "tail_offset_rounds": 216,
                        "stageb_per_500": 0.3,
                        "stageb_bet_rate": 0.08,
                        "stageg2_bullonly_per_500": -0.4,
                        "stageg2_bullonly_bet_rate": 0.06,
                    },
                    {
                        "tail_offset_rounds": 0,
                        "stageb_per_500": -0.1,
                        "stageb_bet_rate": 0.07,
                        "stageg2_bullonly_per_500": 1.0,
                        "stageg2_bullonly_bet_rate": 0.06,
                    },
                ]
            )
        return path

    def test_prev_winner_with_skip_uses_last_completed_window(self) -> None:
        profiles, rows = _load_compare_rows(self._write_compare_csv())
        self.assertEqual(["stageb", "stageg2_bullonly"], profiles)
        recommendation = predict_next_window_recommendation(
            rows=rows,
            mode="prev_winner_with_skip",
            baseline_profile_name="stageb",
            static_profile_name="stageb",
            lookback=1,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual("stageg2_bullonly", recommendation.chosen_profile)

    def test_trailing_best_vs_stageb_with_skip_can_skip(self) -> None:
        profiles, rows = _load_compare_rows(self._write_compare_csv())
        self.assertEqual(["stageb", "stageg2_bullonly"], profiles)
        recommendation = predict_next_window_recommendation(
            rows=rows,
            mode="trailing_best_vs_stageb_with_skip",
            baseline_profile_name="stageb",
            static_profile_name="stageb",
            lookback=2,
            margin_per_500=0.5,
            skip_threshold_per_500=1.0,
        )
        self.assertEqual("skip", recommendation.chosen_profile)
        self.assertEqual(0.0, recommendation.estimated_per_500)

    def test_trailing_best_vs_stageb_prefers_alternate_on_gap(self) -> None:
        profiles, rows = _load_compare_rows(self._write_compare_csv())
        self.assertEqual(["stageb", "stageg2_bullonly"], profiles)
        recommendation = predict_next_window_recommendation(
            rows=rows,
            mode="trailing_best_vs_stageb",
            baseline_profile_name="stageb",
            static_profile_name="stageb",
            lookback=2,
            margin_per_500=0.1,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual("stageg2_bullonly", recommendation.chosen_profile)


if __name__ == "__main__":
    unittest.main()
