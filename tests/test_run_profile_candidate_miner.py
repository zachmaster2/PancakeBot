from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from inspection.run_profile_candidate_miner import (
    _alias_from_candidate_name,
    _current_pool_best_with_skip,
    _load_compare_rows,
    _positive_to_positive_rate,
    _score_candidate,
)


class ProfileCandidateMinerTests(unittest.TestCase):
    def _write_compare_csv(self, rows: list[dict[str, object]]) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / "compare.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_alias_strips_disloc_prefix(self) -> None:
        self.assertEqual("stageG2_bullonly_recent5pct_v1", _alias_from_candidate_name("disloc_stageG2_bullonly_recent5pct_v1"))
        self.assertEqual("other", _alias_from_candidate_name("other"))

    def test_load_compare_rows_reads_profiles_and_offsets(self) -> None:
        path = self._write_compare_csv(
            [
                {
                    "tail_offset_rounds": 216,
                    "stageb_per_500": 0.1,
                    "stageb_bet_rate": 0.08,
                    "alt_per_500": -0.2,
                    "alt_bet_rate": 0.04,
                }
            ]
        )
        profiles, rows, offsets = _load_compare_rows(path)
        self.assertEqual(["stageb", "alt"], profiles)
        self.assertEqual([216], offsets)
        self.assertAlmostEqual(0.1, rows[0]["stageb_per_500"])

    def test_score_candidate_tracks_skip_replacements_and_streaks(self) -> None:
        current_rows = [
            {"stageb_per_500": -0.1, "stageg2_per_500": -0.2},
            {"stageb_per_500": 0.3, "stageg2_per_500": -0.1},
            {"stageb_per_500": -0.2, "stageg2_per_500": -0.3},
        ]
        candidate_rows = [
            {"alt_per_500": 0.4, "alt_bet_rate": 0.06},
            {"alt_per_500": 0.1, "alt_bet_rate": 0.06},
            {"alt_per_500": 0.5, "alt_bet_rate": 0.07},
        ]
        score = _score_candidate(
            profile_name="alt",
            active_candidate_name="disloc_alt",
            candidate_rows=candidate_rows,
            current_rows=current_rows,
            current_profile_names=["stageb", "stageg2"],
        )
        self.assertEqual(2, score.skip_replacement_count)
        self.assertEqual(3, score.positive_window_count)
        self.assertEqual(3, score.max_positive_streak)
        self.assertGreater(score.marginal_oracle_gain_per_500, 0.0)

    def test_positive_to_positive_rate_handles_no_positive(self) -> None:
        self.assertEqual(0.0, _positive_to_positive_rate([-1.0, 0.0, -0.2]))
        self.assertAlmostEqual(0.5, _positive_to_positive_rate([1.0, 1.0, -1.0]))

    def test_current_pool_best_with_skip_uses_zero_floor(self) -> None:
        row = {"a_per_500": -0.1, "b_per_500": -0.3}
        self.assertEqual(0.0, _current_pool_best_with_skip(row, ["a", "b"]))


if __name__ == "__main__":
    unittest.main()
