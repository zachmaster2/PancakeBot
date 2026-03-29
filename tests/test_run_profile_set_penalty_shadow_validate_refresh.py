from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from inspection.run_profile_set_penalty_shadow_validate_refresh import (
    _refresh_argv,
    _validation_argv,
)


class ProfileSetPenaltyShadowValidateRefreshTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            name_prefix="penalty_shadow_validate_refresh_test",
            output_dir="../PancakeBot_var_exp",
            recent_cycles=12,
            refresh_args=["--window-size-rounds", "216", "--num-windows", "20"],
        )

    def test_refresh_argv_includes_refresh_args(self) -> None:
        argv = _refresh_argv(
            python_exe="python",
            args=self._args(),
            output_dir=Path("../PancakeBot_var_exp"),
        )
        self.assertIn("inspection.run_profile_set_penalty_shadow_refresh", argv)
        self.assertIn("--window-size-rounds", argv)
        self.assertIn("--num-windows", argv)

    def test_validation_argv_targets_shadow_validation(self) -> None:
        argv = _validation_argv(
            python_exe="python",
            args=self._args(),
            output_dir=Path("../PancakeBot_var_exp"),
        )
        self.assertIn("inspection.run_profile_set_shadow_validation", argv)
        self.assertIn("--recommendation-json", argv)
        self.assertIn("--recent-cycles", argv)


if __name__ == "__main__":
    unittest.main()
