from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError


class LoadConfigMlAliasTests(unittest.TestCase):
    def test_ml_alias_keys_are_accepted(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = str(base_text).replace("calibrate_size =", "calibration_size =", 1).replace(
            "recalibrate_interval =",
            "recalibration_interval =",
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_ml_aliases.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            cfg = load_app_config(str(cfg_path))

        self.assertEqual(8000, int(cfg.strategy.ml_candidate.train_size))
        self.assertEqual(4000, int(cfg.strategy.ml_candidate.calibrate_size))
        self.assertEqual(500, int(cfg.strategy.ml_candidate.retrain_interval))
        self.assertEqual(250, int(cfg.strategy.ml_candidate.recalibrate_interval))

    def test_ml_alias_conflict_is_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = str(base_text).replace(
            "calibrate_size = 4000",
            "calibrate_size = 4000\ncalibration_size = 3000",
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_ml_alias_conflict.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_predictability_modes_are_accepted(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = str(base_text).replace(
            'predictability_feature_mode = "all_features"',
            'predictability_feature_mode = "arrival_microstructure_plus_regime"',
            1,
        ).replace(
            'predictability_label_mode = "baseline_log_imbalance_side"',
            'predictability_label_mode = "either_side_profitable"',
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_ml_predictability_modes.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            cfg = load_app_config(str(cfg_path))

        self.assertEqual("arrival_microstructure_plus_regime", cfg.strategy.ml_candidate.predictability_feature_mode)
        self.assertEqual("either_side_profitable", cfg.strategy.ml_candidate.predictability_label_mode)

    def test_expected_net_max_is_accepted(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = str(base_text).replace(
            "expected_net_min_bnb = 0.0",
            "expected_net_min_bnb = 0.0\nexpected_net_max_bnb = 0.005",
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_ml_expected_net_max.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            cfg = load_app_config(str(cfg_path))

        self.assertEqual(0.005, float(cfg.strategy.ml_candidate.expected_net_max_bnb))

    def test_expected_net_max_below_min_is_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = str(base_text).replace(
            "expected_net_min_bnb = 0.0",
            "expected_net_min_bnb = 0.01\nexpected_net_max_bnb = 0.005",
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_ml_expected_net_max_invalid.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_predictability_feature_mode_is_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = str(base_text).replace(
            'predictability_feature_mode = "all_features"',
            'predictability_feature_mode = "not_a_mode"',
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_ml_bad_feature_mode.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_predictability_label_mode_is_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = str(base_text).replace(
            'predictability_label_mode = "baseline_log_imbalance_side"',
            'predictability_label_mode = "bad_label_mode"',
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_ml_bad_label_mode.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))


if __name__ == "__main__":
    unittest.main()
