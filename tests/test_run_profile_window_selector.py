from __future__ import annotations

import unittest

from inspection.run_profile_window_selector import (
    SelectorResult,
    WindowComparison,
    _evaluate_selectors,
    _select_window_value,
    _window_comparisons,
)


class ProfileWindowSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            WindowComparison(tail_offset_rounds=400, stageb_per_500=-0.5, stageb_bet_rate=0.05, flow_per_500=0.2, flow_bet_rate=0.10),
            WindowComparison(tail_offset_rounds=200, stageb_per_500=0.3, stageb_bet_rate=0.07, flow_per_500=-0.1, flow_bet_rate=0.11),
            WindowComparison(tail_offset_rounds=0, stageb_per_500=-0.2, stageb_bet_rate=0.06, flow_per_500=0.4, flow_bet_rate=0.09),
        ]

    def test_window_comparisons_sort_oldest_to_newest(self) -> None:
        ordered = _window_comparisons(rows=list(reversed(self.rows)))
        self.assertEqual([400, 200, 0], [row.tail_offset_rounds for row in ordered])

    def test_prev_winner_is_causal(self) -> None:
        ordered = _window_comparisons(rows=self.rows)
        pick0, value0, bet_rate0 = _select_window_value(
            rows=ordered,
            idx=0,
            mode="prev_winner",
            lookback=0,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        pick1, value1, bet_rate1 = _select_window_value(
            rows=ordered,
            idx=1,
            mode="prev_winner",
            lookback=0,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        pick2, value2, bet_rate2 = _select_window_value(
            rows=ordered,
            idx=2,
            mode="prev_winner",
            lookback=0,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual(("stageb", -0.5, 0.05), (pick0, value0, bet_rate0))
        self.assertEqual(("flow", -0.1, 0.11), (pick1, value1, bet_rate1))
        self.assertEqual(("stageb", -0.2, 0.06), (pick2, value2, bet_rate2))

    def test_trailing_delta_prefers_flow_when_recent_delta_clears_margin(self) -> None:
        ordered = _window_comparisons(rows=self.rows)
        pick, value, bet_rate = _select_window_value(
            rows=ordered,
            idx=2,
            mode="trailing_delta",
            lookback=2,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual(("flow", 0.4, 0.09), (pick, value, bet_rate))
        pick2, value2, bet_rate2 = _select_window_value(
            rows=ordered,
            idx=2,
            mode="trailing_delta",
            lookback=1,
            margin_per_500=0.5,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual(("stageb", -0.2, 0.06), (pick2, value2, bet_rate2))

    def test_skip_aware_modes_can_skip(self) -> None:
        ordered = _window_comparisons(rows=self.rows)
        pick, value, bet_rate = _select_window_value(
            rows=ordered,
            idx=0,
            mode="prev_winner_with_skip",
            lookback=0,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual(("skip", 0.0, 0.0), (pick, value, bet_rate))
        pick2, value2, bet_rate2 = _select_window_value(
            rows=ordered,
            idx=2,
            mode="trailing_delta_with_skip",
            lookback=2,
            margin_per_500=0.0,
            skip_threshold_per_500=0.5,
        )
        self.assertEqual(("skip", 0.0, 0.0), (pick2, value2, bet_rate2))

    def test_evaluate_selectors_sorts_best_first(self) -> None:
        ordered = _window_comparisons(rows=self.rows)
        results = _evaluate_selectors(
            rows=ordered,
            lookbacks=[1],
            margins_per_500=[0.0],
            skip_thresholds_per_500=[0.0],
            min_selected_bet_rate=0.05,
        )
        self.assertIsInstance(results[0], SelectorResult)
        self.assertGreaterEqual(results[0].mean_per_500, results[-1].mean_per_500)
        modes = [row.mode for row in results]
        self.assertIn("oracle", modes)
        self.assertIn("stageb_only", modes)
        self.assertIn("flow_only", modes)
        self.assertIn("skip_only", modes)
        self.assertIn("trailing_delta_with_skip", modes)


if __name__ == "__main__":
    unittest.main()
