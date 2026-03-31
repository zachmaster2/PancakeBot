from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inspection.write_window_controller_runtime_config import write_runtime_config


class WriteWindowControllerRuntimeConfigTests(unittest.TestCase):
    def test_write_runtime_config_patches_candidates_and_controller(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "config.toml"
            base.write_text(
                "[strategy.dislocation]\n"
                "active_candidate_names = [\n"
                '  "disloc_stageB_bullonly_recent8pct_v1",\n'
                "]\n\n"
                "[strategy.window_controller]\n"
                "enabled = false\n"
                'mode = "absolute_best_with_skip"\n'
                "profile_names = [\n"
                '  "disloc_stageB_bullonly_recent8pct_v1",\n'
                '  "disloc_stageG2_bullonly_recent5pct_v1",\n'
                "]\n"
                'cold_start_profile_name = "disloc_stageB_bullonly_recent8pct_v1"\n'
                "window_rounds = 216\n"
                "lookback_windows = 2\n"
                "min_history_windows = 2\n"
                'estimator_mode = "ewm_mean"\n'
                "ewm_alpha = 0.5\n"
                "stability_penalty_per_500 = 0.0\n"
                "activity_target_bet_rate = 0.0\n"
                "activity_shortfall_penalty_per_500 = 0.0\n"
                "skip_threshold_per_500 = 0.05\n\n"
                "[backtest]\n"
                "simulation_size = 20000\n",
                encoding="utf-8",
            )
            out = write_runtime_config(
                base_config_path=base,
                output_dir=root,
                name_prefix="demo",
                active_candidate_names=[
                    "disloc_stageB_bullonly_recent8pct_v1",
                    "disloc_stageG2_bullonly_recent5pct_v1",
                ],
                enabled="true",
                mode="absolute_best_with_skip",
                profile_names=[
                    "disloc_stageB_bullonly_recent8pct_v1",
                    "disloc_stageG2_bullonly_recent5pct_v1",
                ],
                cold_start_profile_name="disloc_stageB_bullonly_recent8pct_v1",
                window_rounds=216,
                lookback_windows=2,
                min_history_windows=2,
                estimator_mode="ewm_mean",
                ewm_alpha=0.5,
                stability_penalty_per_500=0.0,
                activity_target_bet_rate=0.05,
                activity_shortfall_penalty_per_500=5.0,
                skip_threshold_per_500=0.05,
            )
            text = out.read_text(encoding="utf-8")
        self.assertIn('"disloc_stageG2_bullonly_recent5pct_v1"', text)
        self.assertIn("enabled = true", text)
        self.assertIn("lookback_windows = 2", text)
        self.assertIn('estimator_mode = "ewm_mean"', text)
        self.assertIn("activity_target_bet_rate = 0.05", text)
        self.assertIn("activity_shortfall_penalty_per_500 = 5.0", text)


if __name__ == "__main__":
    unittest.main()
