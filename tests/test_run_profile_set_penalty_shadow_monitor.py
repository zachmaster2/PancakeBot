from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from inspection.run_profile_set_penalty_shadow_monitor import (
    _load_cycle_signature,
    _validate_refresh_argv,
)


class ProfileSetPenaltyShadowMonitorTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            name_prefix="penalty_shadow_monitor_test",
            recent_cycles=12,
            refresh_args=["--window-size-rounds", "216"],
        )

    def test_validate_refresh_argv_includes_passthrough_args(self) -> None:
        argv = _validate_refresh_argv(
            python_exe="python",
            args=self._args(),
            output_dir=Path("../PancakeBot_var_exp"),
        )
        self.assertIn("inspection.run_profile_set_penalty_shadow_validate_refresh", argv)
        self.assertIn("--recent-cycles", argv)
        self.assertIn("--window-size-rounds", argv)

    def test_load_cycle_signature_reads_latest_epoch(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / "dry_cycle_audit.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["current_epoch"])
            writer.writeheader()
            writer.writerow({"current_epoch": "468010"})
            writer.writerow({"current_epoch": "468011"})
        self.assertEqual((2, 468011), _load_cycle_signature(path))


if __name__ == "__main__":
    unittest.main()
