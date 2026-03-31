from __future__ import annotations

import unittest

from inspection.run_dry_cycle_monitor import _summarize


class DryCycleMonitorTests(unittest.TestCase):
    def test_contained_allowlists_accept_stageb_bull_only(self) -> None:
        rows = [
            {
                "action": "BET",
                "selected_strategy": "disloc_stageB_bullonly_recent8pct_v1",
                "bet_side": "Bull",
                "skip_reason": "",
                "expected_profit_bnb": "0.02",
            },
        ]

        summary = _summarize(
            rows=rows,
            expected_strategies={"disloc_stageB_bullonly_recent8pct_v1"},
            expected_bet_sides={"Bull"},
            expected_controller_profiles=set(),
            expected_controller_actions=set(),
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
            expected_controller_profiles=set(),
            expected_controller_actions=set(),
            warn_idle_streak_cycles=240,
            warn_min_cycles_for_rate_check=240,
            warn_total_bet_rate_below=0.02,
        )

        anomalies = list(summary["anomalies"])
        self.assertTrue(any("unexpected_selected_strategies" in item for item in anomalies))
        self.assertTrue(any("unexpected_bet_sides" in item for item in anomalies))

    def test_allowlists_flag_unexpected_controller_profile_and_action(self) -> None:
        rows = [
            {
                "action": "SKIP",
                "selected_strategy": "",
                "bet_side": "",
                "skip_reason": "selector_no_candidate",
                "expected_profit_bnb": "",
                "controller_selected_profile": "flow_bear_loose10",
                "controller_selected_action": "profile",
            },
        ]

        summary = _summarize(
            rows=rows,
            expected_strategies={"disloc_stageB_bullonly_recent8pct_v1"},
            expected_bet_sides={"Bull"},
            expected_controller_profiles={"disloc_stageB_bullonly_recent8pct_v1", "disloc_stageG2_bullonly_recent5pct_v1"},
            expected_controller_actions={"skip"},
            warn_idle_streak_cycles=240,
            warn_min_cycles_for_rate_check=240,
            warn_total_bet_rate_below=0.02,
        )

        anomalies = list(summary["anomalies"])
        self.assertTrue(any("unexpected_controller_profiles" in item for item in anomalies))
        self.assertTrue(any("unexpected_controller_actions" in item for item in anomalies))

    def test_summary_counts_controller_profile_runtime_skip_cycles(self) -> None:
        rows = [
            {
                "current_epoch": "10",
                "action": "SKIP",
                "selected_strategy": "",
                "bet_side": "",
                "skip_reason": "selector_no_candidate",
                "expected_profit_bnb": "",
                "controller_mode": "absolute_best_with_skip",
                "controller_window_index": "2",
                "controller_lookback_windows_used": "2",
                "controller_selected_profile": "disloc_stageG2_bullonly_recent5pct_v1",
                "controller_selected_action": "profile",
                "controller_estimated_per_500": "0.05",
                "controller_estimated_selected_bet_rate": "0.04",
            },
            {
                "current_epoch": "11",
                "action": "SKIP",
                "selected_strategy": "",
                "bet_side": "",
                "skip_reason": "window_controller_skip",
                "expected_profit_bnb": "",
                "controller_mode": "absolute_best_with_skip",
                "controller_window_index": "3",
                "controller_lookback_windows_used": "2",
                "controller_selected_profile": "",
                "controller_selected_action": "skip",
                "controller_estimated_per_500": "0.0",
                "controller_estimated_selected_bet_rate": "0.0",
            },
        ]

        summary = _summarize(
            rows=rows,
            expected_strategies=set(),
            expected_bet_sides=set(),
            expected_controller_profiles=set(),
            expected_controller_actions=set(),
            warn_idle_streak_cycles=240,
            warn_min_cycles_for_rate_check=240,
            warn_total_bet_rate_below=0.02,
        )

        self.assertEqual(1, int(summary["controller_profile_cycles"]))
        self.assertEqual(1, int(summary["controller_skip_cycles"]))
        self.assertEqual(1, int(summary["controller_profile_but_runtime_skip_cycles"]))
        latest = dict(summary["latest_controller_state"])
        self.assertEqual("11", latest["current_epoch"])
        self.assertEqual("skip", latest["controller_selected_action"])


if __name__ == "__main__":
    unittest.main()
