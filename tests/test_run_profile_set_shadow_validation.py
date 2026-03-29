from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from inspection.run_profile_set_shadow_validation import (
    _coherence_status,
    _load_cycle_rows,
    _recent_summary,
)


class ProfileSetShadowValidationTests(unittest.TestCase):
    def test_recent_summary_counts_actions_and_skips(self) -> None:
        rows = [
            {"action": "SKIP", "skip_reason": "selector_no_candidate", "selected_strategy": ""},
            {"action": "BET", "skip_reason": "", "selected_strategy": "disloc_stageB_bullonly_recent8pct_v1"},
            {"action": "SKIP", "skip_reason": "selector_no_candidate", "selected_strategy": ""},
        ]
        summary = _recent_summary(rows, 2)
        self.assertEqual(2, summary["recent_cycle_count"])
        self.assertEqual(1, summary["recent_bet_count"])
        self.assertEqual(1, summary["recent_skip_count"])

    def test_coherence_status_handles_skip_and_stageb_cases(self) -> None:
        cycle_rows = [{"action": "SKIP"}, {"action": "SKIP"}]
        coherence, reason = _coherence_status(
            recommendation={"chosen_profile": "skip"},
            cycle_rows=cycle_rows,
            recent_cycles=12,
        )
        self.assertEqual(("coherent", "shadow_skip_and_recent_dry_is_all_skip"), (coherence, reason))
        coherence, reason = _coherence_status(
            recommendation={"chosen_profile": "stageb"},
            cycle_rows=cycle_rows,
            recent_cycles=12,
        )
        self.assertEqual(("coherent", "shadow_stageb_matches_contained_runtime"), (coherence, reason))

    def test_load_cycle_rows_reads_csv(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / "dry_cycle_audit.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["action", "skip_reason", "selected_strategy"])
            writer.writeheader()
            writer.writerow({"action": "SKIP", "skip_reason": "selector_no_candidate", "selected_strategy": ""})
        rows = _load_cycle_rows(path)
        self.assertEqual(1, len(rows))
        self.assertEqual("SKIP", rows[0]["action"])


if __name__ == "__main__":
    unittest.main()
