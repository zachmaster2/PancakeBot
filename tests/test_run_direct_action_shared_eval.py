from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inspection.run_direct_action_shared_eval import (
    DirectActionEvalRow,
    _aggregate_rows,
    _eval_command,
    _parse_nonnegative_int_list,
    _parse_positive_int_list,
    _summary_metrics,
)


class DirectActionSharedEvalTests(unittest.TestCase):
    def test_parse_lists_validate_values(self) -> None:
        self.assertEqual([6480, 8640], _parse_positive_int_list("6480,8640"))
        self.assertEqual([0, 216, 432], _parse_nonnegative_int_list("0,216,432"))
        with self.assertRaises(Exception):
            _parse_positive_int_list("0")
        with self.assertRaises(Exception):
            _parse_nonnegative_int_list("-1")

    def test_eval_command_enables_direct_action_and_disables_controller(self) -> None:
        cmd = _eval_command(
            python_exe="python",
            config_path="config.toml",
            scenario_name="demo",
            sim_size=6480,
            tail_offset_rounds=216,
            bundle_path=Path("bundle.pkl.gz"),
        )

        self.assertIn("--direct-action-enabled", cmd)
        self.assertIn("true", cmd)
        self.assertIn("--direct-action-model-bundle-path", cmd)
        self.assertIn("bundle.pkl.gz", cmd)
        self.assertIn("--window-controller-enabled", cmd)
        self.assertIn("false", cmd)

    def test_summary_metrics_reads_backtest_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "backtest_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "num_rounds": 6480,
                        "net_profit_bnb": 1.296,
                        "bet_rate": 0.08,
                        "risk": {"max_drawdown_bnb": 0.42},
                    }
                ),
                encoding="utf-8",
            )
            per_500, bet_rate, net, max_dd = _summary_metrics(path)
        self.assertAlmostEqual(0.1, per_500)
        self.assertAlmostEqual(0.08, bet_rate)
        self.assertAlmostEqual(1.296, net)
        self.assertAlmostEqual(0.42, max_dd)

    def test_aggregate_rows_computes_summary(self) -> None:
        rows = [
            DirectActionEvalRow(
                sim_size=6480,
                tail_offset_rounds=0,
                train_size=15000,
                valid_size=3000,
                random_seed=7,
                required_history_rounds=15061,
                bundle_path="a",
                test_per_500=0.10,
                test_bet_rate=0.08,
                test_net_profit_bnb=1.0,
                max_drawdown_bnb=0.40,
            ),
            DirectActionEvalRow(
                sim_size=6480,
                tail_offset_rounds=216,
                train_size=15000,
                valid_size=3000,
                random_seed=7,
                required_history_rounds=15061,
                bundle_path="b",
                test_per_500=0.02,
                test_bet_rate=0.06,
                test_net_profit_bnb=0.2,
                max_drawdown_bnb=0.35,
            ),
        ]

        aggregates = _aggregate_rows(rows)

        self.assertEqual(1, len(aggregates))
        self.assertAlmostEqual(0.06, aggregates[0].mean_per_500)
        self.assertAlmostEqual(0.02, aggregates[0].min_per_500)
        self.assertAlmostEqual(0.07, aggregates[0].mean_bet_rate)
        self.assertAlmostEqual(0.6, aggregates[0].mean_net_profit_bnb)
        self.assertAlmostEqual(0.40, aggregates[0].max_drawdown_bnb)


if __name__ == "__main__":
    unittest.main()
