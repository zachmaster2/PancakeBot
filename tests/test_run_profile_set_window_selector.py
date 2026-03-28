from __future__ import annotations

import unittest

from inspection.run_profile_set_window_selector import (
    FlowProfileSpec,
    ProfileMetric,
    SelectorResult,
    WindowRow,
    _evaluate_selectors,
    _ordered_window_rows,
    _parse_flow_profile_spec,
    _pick_window,
)


class ProfileSetWindowSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            WindowRow(
                tail_offset_rounds=400,
                metrics={
                    "stageb": ProfileMetric(per_500=-0.5, bet_rate=0.05),
                    "flow_a": ProfileMetric(per_500=0.2, bet_rate=0.11),
                    "flow_b": ProfileMetric(per_500=-0.1, bet_rate=0.08),
                },
            ),
            WindowRow(
                tail_offset_rounds=200,
                metrics={
                    "stageb": ProfileMetric(per_500=-0.1, bet_rate=0.06),
                    "flow_a": ProfileMetric(per_500=-0.2, bet_rate=0.10),
                    "flow_b": ProfileMetric(per_500=0.3, bet_rate=0.09),
                },
            ),
            WindowRow(
                tail_offset_rounds=0,
                metrics={
                    "stageb": ProfileMetric(per_500=-0.2, bet_rate=0.07),
                    "flow_a": ProfileMetric(per_500=0.4, bet_rate=0.12),
                    "flow_b": ProfileMetric(per_500=0.1, bet_rate=0.09),
                },
            ),
        ]
        self.flow_profiles = [
            FlowProfileSpec(
                name="flow_a",
                train_size=15000,
                val_size=None,
                step_size=None,
                ev_threshold=0.006,
                min_total_pool_c=1.2,
                allowed_sides="bear_only",
                bull_roll_edge_min=0.0,
                bear_roll_edge_min=0.0,
                bull_roll_winrate_min=0.5,
                bear_roll_winrate_min=0.5,
                bull_cooldown_trades=80,
                bear_cooldown_trades=80,
            ),
            FlowProfileSpec(
                name="flow_b",
                train_size=15000,
                val_size=None,
                step_size=None,
                ev_threshold=0.005,
                min_total_pool_c=1.0,
                allowed_sides="bear_only",
                bull_roll_edge_min=0.0,
                bear_roll_edge_min=0.0,
                bull_roll_winrate_min=0.5,
                bear_roll_winrate_min=0.47,
                bull_cooldown_trades=80,
                bear_cooldown_trades=120,
            ),
        ]

    def test_parse_flow_profile_spec_defaults_window_sizes(self) -> None:
        spec = _parse_flow_profile_spec(
            "name=flow_x,train_size=15000,ev_threshold=0.006,min_total_pool_c=1.2,allowed_sides=bear_only",
            window_size_rounds=216,
        )
        self.assertEqual("flow_x", spec.name)
        self.assertEqual(15000, spec.train_size)
        self.assertIsNone(spec.val_size)
        self.assertIsNone(spec.step_size)
        self.assertEqual("bear_only", spec.allowed_sides)

    def test_prev_winner_generalizes_to_multi_profile(self) -> None:
        ordered = _ordered_window_rows(self.rows)
        pick0, value0, bet_rate0 = _pick_window(
            rows=ordered,
            idx=0,
            mode="prev_winner",
            profile_name="",
            lookback=0,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        pick1, value1, bet_rate1 = _pick_window(
            rows=ordered,
            idx=1,
            mode="prev_winner",
            profile_name="",
            lookback=0,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        pick2, value2, bet_rate2 = _pick_window(
            rows=ordered,
            idx=2,
            mode="prev_winner",
            profile_name="",
            lookback=0,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual(("stageb", -0.5, 0.05), (pick0, value0, bet_rate0))
        self.assertEqual(("flow_a", -0.2, 0.10), (pick1, value1, bet_rate1))
        self.assertEqual(("flow_b", 0.1, 0.09), (pick2, value2, bet_rate2))

    def test_trailing_best_vs_stageb_prefers_best_alt(self) -> None:
        ordered = _ordered_window_rows(self.rows)
        pick, value, bet_rate = _pick_window(
            rows=ordered,
            idx=2,
            mode="trailing_best_vs_stageb",
            profile_name="",
            lookback=2,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual(("flow_b", 0.1, 0.09), (pick, value, bet_rate))

    def test_skip_aware_multi_profile_modes_can_skip(self) -> None:
        ordered = _ordered_window_rows(self.rows)
        pick, value, bet_rate = _pick_window(
            rows=ordered,
            idx=0,
            mode="prev_winner_with_skip",
            profile_name="",
            lookback=0,
            margin_per_500=0.0,
            skip_threshold_per_500=0.0,
        )
        self.assertEqual(("skip", 0.0, 0.0), (pick, value, bet_rate))

    def test_evaluate_selectors_includes_static_and_skip_modes(self) -> None:
        results = _evaluate_selectors(
            rows=self.rows,
            flow_profiles=self.flow_profiles,
            lookbacks=[1, 2],
            margins_per_500=[0.0],
            skip_thresholds_per_500=[0.0],
            min_selected_bet_rate=0.05,
        )
        self.assertIsInstance(results[0], SelectorResult)
        modes = [row.mode for row in results]
        self.assertIn("skip_only", modes)
        self.assertIn("static_profile", modes)
        self.assertIn("trailing_best_vs_stageb_with_skip", modes)
        self.assertGreaterEqual(results[0].mean_per_500, results[-1].mean_per_500)


if __name__ == "__main__":
    unittest.main()
