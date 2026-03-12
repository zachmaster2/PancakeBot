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


if __name__ == "__main__":
    unittest.main()
