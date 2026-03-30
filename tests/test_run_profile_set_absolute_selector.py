from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inspection.run_profile_set_absolute_selector import (
    _estimate_profile,
    _evaluate_absolute_selectors,
    _load_compare_rows,
    _pick_absolute_window,
    _resolve_profile_names,
)


class ProfileSetAbsoluteSelectorTests(unittest.TestCase):
    def _write_compare_csv(self) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / "compare.csv"
        path.write_text(
            "\n".join(
                [
                    "tail_offset_rounds,stageb_per_500,stageb_bet_rate,alt_per_500,alt_bet_rate,alt2_per_500,alt2_bet_rate",
                    "400,-0.5,0.05,0.2,0.10,-0.1,0.08",
                    "200,-0.1,0.06,-0.2,0.11,0.3,0.09",
                    "0,-0.2,0.07,0.4,0.12,0.1,0.10",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
        return path

    def test_load_compare_rows_orders_oldest_first(self) -> None:
        profiles, rows = _load_compare_rows(self._write_compare_csv())
        self.assertEqual(["stageb", "alt", "alt2"], profiles)
        self.assertEqual([400, 200, 0], [row.tail_offset_rounds for row in rows])

    def test_resolve_profile_names_defaults_to_all(self) -> None:
        names = _resolve_profile_names("", available_profiles=["stageb", "alt", "alt2"])
        self.assertEqual(["stageb", "alt", "alt2"], names)

    def test_trailing_mean_pick_prefers_positive_best_profile(self) -> None:
        profiles, rows = _load_compare_rows(self._write_compare_csv())
        pick, predicted, estimated_bet_rate = _pick_absolute_window(
            rows=rows,
            idx=2,
            profile_names=profiles,
            cold_start_profile_name="",
            mode="trailing_mean",
            lookback_windows=2,
            min_history_windows=2,
            ewm_alpha=0.7,
            stability_penalty_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual("alt2", pick)
        self.assertAlmostEqual(0.1, predicted)
        self.assertAlmostEqual(0.085, estimated_bet_rate)

    def test_ewm_estimate_weights_recent_history(self) -> None:
        profiles, rows = _load_compare_rows(self._write_compare_csv())
        history = rows[0:2]
        estimate, _ = _estimate_profile(
            history=history,
            profile_name="alt2",
            mode="ewm_mean",
            ewm_alpha=0.5,
            stability_penalty_per_500=0.0,
        )
        self.assertAlmostEqual(((-0.1 * (1.0 / 3.0)) + (0.3 * (2.0 / 3.0))), estimate)

    def test_skip_threshold_can_force_skip(self) -> None:
        profiles, rows = _load_compare_rows(self._write_compare_csv())
        pick, predicted, estimated_bet_rate = _pick_absolute_window(
            rows=rows,
            idx=2,
            profile_names=profiles,
            cold_start_profile_name="",
            mode="trailing_mean",
            lookback_windows=2,
            min_history_windows=2,
            ewm_alpha=0.7,
            stability_penalty_per_500=0.0,
            skip_threshold_per_500=0.15,
        )
        self.assertEqual(("skip", 0.0, 0.0), (pick, predicted, estimated_bet_rate))

    def test_cold_start_profile_can_be_fixed_without_history(self) -> None:
        profiles, rows = _load_compare_rows(self._write_compare_csv())
        pick, predicted, estimated_bet_rate = _pick_absolute_window(
            rows=rows,
            idx=0,
            profile_names=profiles,
            cold_start_profile_name="stageb",
            mode="trailing_mean",
            lookback_windows=2,
            min_history_windows=2,
            ewm_alpha=0.7,
            stability_penalty_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual(("stageb", 0.0, 0.0), (pick, predicted, estimated_bet_rate))

    def test_evaluate_absolute_selectors_includes_static_and_oracle_reference_rows(self) -> None:
        profiles, rows = _load_compare_rows(self._write_compare_csv())
        results, eval_sets = _evaluate_absolute_selectors(
            rows=rows,
            profile_names=profiles,
            cold_start_profile_name="",
            modes=["trailing_mean"],
            lookback_windows_list=[2],
            min_history_windows_list=[2],
            ewm_alphas=[0.7],
            stability_penalties_per_500=[0.0],
            skip_thresholds_per_500=[0.0],
            min_selected_bet_rate=0.05,
        )
        modes = [row.mode for row in results]
        self.assertIn("static_profile", modes)
        self.assertIn("oracle_with_skip", modes)
        self.assertIn("trailing_mean", modes)
        self.assertEqual(len(results), len(eval_sets))


if __name__ == "__main__":
    unittest.main()
