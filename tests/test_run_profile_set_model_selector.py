from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from inspection.run_profile_set_model_selector import (
    _evaluate_model_selectors,
    _feature_dict,
    _feature_names,
    _load_compare_rows,
)


class ProfileSetModelSelectorTests(unittest.TestCase):
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
                        "tail_offset_rounds": 648,
                        "stageb_per_500": 0.6,
                        "stageb_bet_rate": 0.08,
                        "stageg2_bullonly_per_500": -0.2,
                        "stageg2_bullonly_bet_rate": 0.05,
                    },
                    {
                        "tail_offset_rounds": 432,
                        "stageb_per_500": 0.5,
                        "stageb_bet_rate": 0.09,
                        "stageg2_bullonly_per_500": -0.1,
                        "stageg2_bullonly_bet_rate": 0.05,
                    },
                    {
                        "tail_offset_rounds": 216,
                        "stageb_per_500": -0.1,
                        "stageb_bet_rate": 0.07,
                        "stageg2_bullonly_per_500": 0.8,
                        "stageg2_bullonly_bet_rate": 0.06,
                    },
                    {
                        "tail_offset_rounds": 0,
                        "stageb_per_500": -0.2,
                        "stageb_bet_rate": 0.07,
                        "stageg2_bullonly_per_500": 0.9,
                        "stageg2_bullonly_bet_rate": 0.06,
                    },
                ]
            )
        return path

    def test_load_compare_rows_orders_oldest_first(self) -> None:
        compare_csv = self._write_compare_csv()
        profiles, rows = _load_compare_rows(compare_csv)
        self.assertEqual(["stageb", "stageg2_bullonly"], profiles)
        self.assertEqual([648, 432, 216, 0], [row.tail_offset_rounds for row in rows])

    def test_feature_dict_contains_baseline_and_delta_features(self) -> None:
        compare_csv = self._write_compare_csv()
        profiles, rows = _load_compare_rows(compare_csv)
        features = _feature_dict(
            rows=rows,
            idx=2,
            profiles=profiles,
            baseline_profile_name="stageb",
            feature_lookbacks=[1, 2],
        )
        names = _feature_names(
            profiles=profiles,
            baseline_profile_name="stageb",
            feature_lookbacks=[1, 2],
        )
        self.assertIn("feat_stageb_mean_per500_l1", features)
        self.assertIn("feat_stageg2_bullonly_delta_mean_vs_stageb_l2", features)
        self.assertIn("feat_stageg2_bullonly_delta_last_vs_stageb_l1", names)

    def test_evaluate_model_selectors_returns_model_modes(self) -> None:
        compare_csv = self._write_compare_csv()
        profiles, rows = _load_compare_rows(compare_csv)
        results = _evaluate_model_selectors(
            rows=rows,
            profiles=profiles,
            baseline_profile_name="stageb",
            feature_lookbacks=[1, 2],
            min_train_windows_list=[2],
            min_hold_windows_list=[1, 2],
            margins_per_500=[0.0],
            skip_thresholds_per_500=[0.0],
            ridge_alphas=[1.0],
            logistic_c_values=[1.0],
            min_selected_bet_rate=0.05,
        )
        modes = {row.mode for row in results}
        self.assertIn("static_profile", modes)
        self.assertIn("oracle_with_skip", modes)
        self.assertIn("delta_ridge", modes)
        self.assertIn("delta_logistic", modes)
        top = results[0]
        self.assertGreaterEqual(float(top.mean_selected_bet_rate), 0.05)
        self.assertIn(int(top.min_hold_windows), {1, 2})


if __name__ == "__main__":
    unittest.main()
