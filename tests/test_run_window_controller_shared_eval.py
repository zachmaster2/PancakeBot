from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inspection.run_window_controller_shared_eval import (
    SharedEvalRow,
    _aggregate_rows,
    _controller_command,
    _parse_nonnegative_int_list,
    _parse_positive_int_list,
    _static_command,
    _summary_metrics,
)


class WindowControllerSharedEvalTests(unittest.TestCase):
    def test_parse_lists_validate_values(self) -> None:
        self.assertEqual([6480, 8640], _parse_positive_int_list("6480,8640"))
        self.assertEqual([0, 216, 432], _parse_nonnegative_int_list("0,216,432"))
        with self.assertRaises(Exception):
            _parse_positive_int_list("0")
        with self.assertRaises(Exception):
            _parse_nonnegative_int_list("-1")

    def test_controller_command_includes_controller_fields(self) -> None:
        cmd = _controller_command(
            python_exe="python",
            config_path="config.toml",
            scenario_name="demo",
            sim_size=6480,
            tail_offset_rounds=216,
            router_mode="selector_max_score",
            controller_mode="trailing_best_vs_baseline_with_skip",
            baseline_profile_name="stageb",
            alternate_profile_name="stageg2",
            window_rounds=216,
            lookback_windows=2,
            margin_per_500=1.0,
            skip_threshold_per_500=0.0,
        )
        self.assertIn("--window-controller-enabled", cmd)
        self.assertIn("--window-controller-mode", cmd)
        self.assertIn("trailing_best_vs_baseline_with_skip", cmd)
        self.assertIn("--window-controller-skip-threshold-per-500", cmd)
        self.assertIn("stageb,stageg2", cmd)

    def test_static_command_uses_only_baseline_profile(self) -> None:
        cmd = _static_command(
            python_exe="python",
            config_path="config.toml",
            scenario_name="demo",
            sim_size=8640,
            tail_offset_rounds=0,
            router_mode="selector_max_score",
            baseline_profile_name="stageb",
        )
        self.assertIn("--active-candidate-names", cmd)
        self.assertIn("stageb", cmd)
        self.assertNotIn("--window-controller-enabled", cmd)

    def test_summary_metrics_reads_backtest_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "backtest_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "num_rounds": 6480,
                        "net_profit_bnb": 1.296,
                        "bet_rate": 0.08,
                    }
                ),
                encoding="utf-8",
            )
            per_500, bet_rate, net = _summary_metrics(path)
        self.assertAlmostEqual(0.1, per_500)
        self.assertAlmostEqual(0.08, bet_rate)
        self.assertAlmostEqual(1.296, net)

    def test_aggregate_rows_computes_lift_summary(self) -> None:
        rows = [
            SharedEvalRow(
                sim_size=6480,
                tail_offset_rounds=0,
                controller_mode="trailing_best_vs_baseline",
                controller_lookback_windows=2,
                controller_margin_per_500=1.0,
                controller_skip_threshold_per_500=0.0,
                controller_per_500=0.10,
                controller_bet_rate=0.08,
                controller_net_profit_bnb=1.0,
                static_stageb_per_500=-0.05,
                static_stageb_bet_rate=0.10,
                static_stageb_net_profit_bnb=-0.5,
                lift_vs_stageb_per_500=0.15,
            ),
            SharedEvalRow(
                sim_size=6480,
                tail_offset_rounds=216,
                controller_mode="trailing_best_vs_baseline",
                controller_lookback_windows=2,
                controller_margin_per_500=1.0,
                controller_skip_threshold_per_500=0.0,
                controller_per_500=0.02,
                controller_bet_rate=0.09,
                controller_net_profit_bnb=0.2,
                static_stageb_per_500=0.01,
                static_stageb_bet_rate=0.11,
                static_stageb_net_profit_bnb=0.1,
                lift_vs_stageb_per_500=0.01,
            ),
        ]
        aggregates = _aggregate_rows(rows)
        self.assertEqual(1, len(aggregates))
        self.assertAlmostEqual(0.06, aggregates[0].controller_mean_per_500)
        self.assertAlmostEqual(0.02, aggregates[0].controller_min_per_500)
        self.assertEqual(2, aggregates[0].lift_wins)


if __name__ == "__main__":
    unittest.main()
