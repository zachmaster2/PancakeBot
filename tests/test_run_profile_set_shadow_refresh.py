from __future__ import annotations

import argparse
import unittest

from inspection.run_profile_set_shadow_refresh import (
    _dislocation_profiles_or_default,
    _flow_profiles_or_default,
    _shadow_argv,
    _window_selector_argv,
)


class ProfileSetShadowRefreshTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            config="config.toml",
            name_prefix="shadow_test",
            window_size_rounds=216,
            num_windows=20,
            source_tail_rounds=30000,
            initial_bankroll_bnb=50.0,
            mode="delta_ridge",
            feature_lookbacks="1,3,5,8",
            min_train_windows=10,
            min_hold_windows=1,
            margin_per_500=-0.2,
            skip_threshold_per_500=0.0,
            ridge_alpha=2.0,
            logistic_c=1.0,
            output_dir="../PancakeBot_var_exp",
            no_resume=False,
        )

    def test_default_profiles_are_populated(self) -> None:
        self.assertGreaterEqual(len(_flow_profiles_or_default([])), 1)
        self.assertEqual(
            ["name=stageg2_bullonly,active_candidate_name=disloc_stageG2_bullonly_recent5pct_v1"],
            _dislocation_profiles_or_default([]),
        )

    def test_window_selector_argv_contains_expected_flags(self) -> None:
        argv = _window_selector_argv(
            python_exe="python",
            args=self._args(),
            flow_profiles=["name=flow_a,train_size=15000,ev_threshold=0.005,min_total_pool_c=1.0,allowed_sides=bear_only,bull_roll_edge_min=0.0,bear_roll_edge_min=0.0,bull_roll_winrate_min=0.5,bear_roll_winrate_min=0.5,bull_cooldown_trades=80,bear_cooldown_trades=80"],
            dislocation_profiles=["name=stageg2_bullonly,active_candidate_name=disloc_stageG2_bullonly_recent5pct_v1"],
        )
        self.assertIn("-m", argv)
        self.assertIn("inspection.run_profile_set_window_selector", argv)
        self.assertIn("--selector-margins-per-500=-0.2,0.0,0.2,0.5", argv)
        self.assertIn("--flow-profile", argv)
        self.assertIn("--dislocation-profile", argv)

    def test_shadow_argv_contains_model_settings(self) -> None:
        argv = _shadow_argv(
            python_exe="python",
            args=self._args(),
            compare_csv="../PancakeBot_var_exp/example_compare.csv",
        )
        self.assertIn("inspection.run_profile_set_shadow_recommender", argv)
        self.assertIn("--feature-lookbacks", argv)
        self.assertIn("--ridge-alpha", argv)


if __name__ == "__main__":
    unittest.main()
