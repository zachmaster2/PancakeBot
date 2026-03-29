from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from inspection.run_profile_set_penalty_selector import (
    _evaluate_penalty_selector,
    _legacy_feature_dict,
    _legacy_feature_names,
    _profile_penalty_per_500,
    predict_next_penalty_recommendation,
)
from inspection.run_profile_set_model_selector import _load_compare_rows


class ProfileSetPenaltySelectorTests(unittest.TestCase):
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

    def test_legacy_feature_dict_and_names_match_expected_shape(self) -> None:
        compare_csv = self._write_compare_csv()
        profiles, rows = _load_compare_rows(compare_csv)
        features = _legacy_feature_dict(
            rows=rows,
            idx=3,
            profiles=profiles,
            baseline_profile_name="stageb",
            feature_lookbacks=[1, 2],
        )
        names = _legacy_feature_names(
            profiles=profiles,
            baseline_profile_name="stageb",
            feature_lookbacks=[1, 2],
        )
        self.assertIn("feat_stageb_mean_per500_l1", features)
        self.assertIn("feat_flow_bear_loose10_delta_mean_vs_stageb_l2", features)
        self.assertIn("feat_stageg2_bullonly_delta_last_vs_stageb_l2", names)

    def test_profile_penalty_per_500_uses_flow_and_stageg2_groups(self) -> None:
        self.assertEqual(
            0.3,
            _profile_penalty_per_500(
                profile_name="flow_bear_loose10",
                flow_penalty_per_500=0.3,
                stageg2_penalty_per_500=0.1,
            ),
        )
        self.assertEqual(
            0.1,
            _profile_penalty_per_500(
                profile_name="stageg2_bullonly",
                flow_penalty_per_500=0.3,
                stageg2_penalty_per_500=0.1,
            ),
        )
        self.assertEqual(
            0.0,
            _profile_penalty_per_500(
                profile_name="stageb",
                flow_penalty_per_500=0.3,
                stageg2_penalty_per_500=0.1,
            ),
        )

    def test_evaluate_penalty_selector_returns_positive_result(self) -> None:
        compare_csv = self._write_compare_csv()
        profiles, rows = _load_compare_rows(compare_csv)
        results = _evaluate_penalty_selector(
            rows=rows,
            profiles=profiles,
            baseline_profile_name="stageb",
            feature_lookbacks=[1, 2],
            min_train_windows_list=[2],
            min_hold_windows_list=[1],
            cold_start_modes=["baseline_or_skip", "trailing_best_vs_stageb_with_skip"],
            cold_start_lookbacks=[1, 2],
            margins_per_500=[0.0],
            skip_thresholds_per_500=[0.0],
            ridge_alphas=[1.0],
            flow_penalties_per_500=[0.0, 0.2],
            stageg2_penalties_per_500=[0.0, 0.1],
            min_selected_bet_rate=0.05,
        )
        top = results[0]
        self.assertGreater(float(top.mean_per_500), 0.0)
        self.assertGreaterEqual(float(top.mean_selected_bet_rate), 0.05)
        self.assertIn(top.cold_start_mode, {"baseline_or_skip", "trailing_best_vs_stageb_with_skip"})

    def test_predict_next_penalty_recommendation_returns_valid_profile(self) -> None:
        compare_csv = self._write_compare_csv()
        profiles, rows = _load_compare_rows(compare_csv)
        recommendation = predict_next_penalty_recommendation(
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
        self.assertIn(recommendation.chosen_profile, {"stageb", "flow_bear_loose10", "stageg2_bullonly", "skip"})
        self.assertGreaterEqual(recommendation.training_window_count, 2)


if __name__ == "__main__":
    unittest.main()
