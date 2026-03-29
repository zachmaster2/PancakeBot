from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from inspection.run_profile_set_model_selector import _load_compare_rows
from inspection.run_profile_set_penalty_window_eval import _eval_rows, _summary


class ProfileSetPenaltyWindowEvalTests(unittest.TestCase):
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
                    "flow_bear_loose10_per_500",
                    "flow_bear_loose10_bet_rate",
                    "stageg2_bullonly_per_500",
                    "stageg2_bullonly_bet_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "tail_offset_rounds": 864,
                        "stageb_per_500": 0.6,
                        "stageb_bet_rate": 0.08,
                        "flow_bear_loose10_per_500": -0.2,
                        "flow_bear_loose10_bet_rate": 0.10,
                        "stageg2_bullonly_per_500": -0.1,
                        "stageg2_bullonly_bet_rate": 0.05,
                    },
                    {
                        "tail_offset_rounds": 648,
                        "stageb_per_500": 0.5,
                        "stageb_bet_rate": 0.09,
                        "flow_bear_loose10_per_500": -0.1,
                        "flow_bear_loose10_bet_rate": 0.10,
                        "stageg2_bullonly_per_500": -0.2,
                        "stageg2_bullonly_bet_rate": 0.05,
                    },
                    {
                        "tail_offset_rounds": 432,
                        "stageb_per_500": -0.1,
                        "stageb_bet_rate": 0.07,
                        "flow_bear_loose10_per_500": 0.9,
                        "flow_bear_loose10_bet_rate": 0.12,
                        "stageg2_bullonly_per_500": 0.4,
                        "stageg2_bullonly_bet_rate": 0.06,
                    },
                    {
                        "tail_offset_rounds": 216,
                        "stageb_per_500": -0.2,
                        "stageb_bet_rate": 0.07,
                        "flow_bear_loose10_per_500": 1.0,
                        "flow_bear_loose10_bet_rate": 0.12,
                        "stageg2_bullonly_per_500": 0.5,
                        "stageg2_bullonly_bet_rate": 0.06,
                    },
                    {
                        "tail_offset_rounds": 0,
                        "stageb_per_500": 0.3,
                        "stageb_bet_rate": 0.07,
                        "flow_bear_loose10_per_500": -0.4,
                        "flow_bear_loose10_bet_rate": 0.12,
                        "stageg2_bullonly_per_500": 0.2,
                        "stageg2_bullonly_bet_rate": 0.06,
                    },
                ]
            )
        return path

    def test_eval_rows_produces_window_metrics(self) -> None:
        compare_csv = self._write_compare_csv()
        profiles, rows = _load_compare_rows(compare_csv)
        eval_rows = _eval_rows(
            rows=rows,
            profiles=profiles,
            baseline_profile_name="stageb",
            feature_lookbacks=[1, 2],
            min_train_windows=2,
            min_hold_windows=1,
            cold_start_mode="trailing_best_vs_stageb_with_skip",
            cold_start_lookback=2,
            ridge_alpha=1.0,
            flow_penalty_per_500=0.2,
            stageg2_penalty_per_500=0.0,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual(len(rows), len(eval_rows))
        self.assertIn(eval_rows[-1].chosen_profile, {"stageb", "flow_bear_loose10", "stageg2_bullonly", "skip"})
        self.assertGreaterEqual(eval_rows[-1].oracle_realized_per_500, eval_rows[-1].realized_per_500)
        self.assertFalse(any(row.hold_forced for row in eval_rows))

    def test_eval_rows_honors_min_hold_windows(self) -> None:
        compare_csv = self._write_compare_csv()
        profiles, rows = _load_compare_rows(compare_csv)
        eval_rows = _eval_rows(
            rows=rows,
            profiles=profiles,
            baseline_profile_name="stageb",
            feature_lookbacks=[1, 2],
            min_train_windows=2,
            min_hold_windows=2,
            cold_start_mode="trailing_best_vs_stageb_with_skip",
            cold_start_lookback=2,
            ridge_alpha=1.0,
            flow_penalty_per_500=0.0,
            stageg2_penalty_per_500=0.0,
            margin_per_500=-0.5,
            skip_threshold_per_500=0.0,
        )
        forced_rows = [row for row in eval_rows if row.hold_forced]
        self.assertTrue(forced_rows)
        self.assertEqual(forced_rows[0].chosen_profile, eval_rows[forced_rows[0].window_index - 1].chosen_profile)

    def test_summary_reports_regret_and_baseline_gain(self) -> None:
        compare_csv = self._write_compare_csv()
        profiles, rows = _load_compare_rows(compare_csv)
        eval_rows = _eval_rows(
            rows=rows,
            profiles=profiles,
            baseline_profile_name="stageb",
            feature_lookbacks=[1, 2],
            min_train_windows=2,
            min_hold_windows=1,
            cold_start_mode="trailing_best_vs_stageb_with_skip",
            cold_start_lookback=2,
            ridge_alpha=1.0,
            flow_penalty_per_500=0.2,
            stageg2_penalty_per_500=0.0,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        summary = _summary(
            eval_rows,
            baseline_profile_name="stageb",
            feature_lookbacks=[1, 2],
            min_train_windows=2,
            min_hold_windows=1,
            cold_start_mode="trailing_best_vs_stageb_with_skip",
            cold_start_lookback=2,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
            ridge_alpha=1.0,
            flow_penalty_per_500=0.2,
            stageg2_penalty_per_500=0.0,
        )
        self.assertGreaterEqual(summary.mean_oracle_per_500, summary.mean_realized_per_500)
        self.assertIn("stageb", summary.chosen_profile_counts_json)


if __name__ == "__main__":
    unittest.main()
