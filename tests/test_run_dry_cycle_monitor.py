from __future__ import annotations

import unittest

from inspection.run_dry_cycle_monitor import _summarize


class DryCycleMonitorTests(unittest.TestCase):
    def test_hybrid_allowlists_accept_stageb_and_flow_with_both_sides(self) -> None:
        rows = [
            {
                "action": "BET",
                "selected_strategy": "disloc_stageB_bullonly_recent8pct_v1",
                "bet_side": "Bull",
                "skip_reason": "",
                "expected_profit_bnb": "0.02",
            },
            {
                "action": "BET",
                "selected_strategy": "flow_lgbm_recent_t12k_r1k_regime40_v1",
                "bet_side": "Bear",
                "skip_reason": "",
                "expected_profit_bnb": "0.03",
            },
        ]

        summary = _summarize(
            rows=rows,
            expected_strategies={
                "disloc_stageB_bullonly_recent8pct_v1",
                "flow_lgbm_recent_t12k_r1k_regime40_v1",
            },
            expected_bet_sides={"Bull", "Bear"},
            warn_idle_streak_cycles=240,
            warn_min_cycles_for_rate_check=240,
            warn_total_bet_rate_below=0.02,
        )

        self.assertEqual([], list(summary["anomalies"]))

    def test_allowlists_flag_unexpected_strategy_and_side(self) -> None:
        rows = [
            {
                "action": "BET",
                "selected_strategy": "unexpected_strategy",
                "bet_side": "Bear",
                "skip_reason": "",
                "expected_profit_bnb": "0.01",
            },
        ]

        summary = _summarize(
            rows=rows,
            expected_strategies={"disloc_stageB_bullonly_recent8pct_v1"},
            expected_bet_sides={"Bull"},
            warn_idle_streak_cycles=240,
            warn_min_cycles_for_rate_check=240,
            warn_total_bet_rate_below=0.02,
        )

        anomalies = list(summary["anomalies"])
        self.assertTrue(any("unexpected_selected_strategies" in item for item in anomalies))
        self.assertTrue(any("unexpected_bet_sides" in item for item in anomalies))


if __name__ == "__main__":
    unittest.main()
