"""Tests for dry-state file loading in the runtime loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pancakebot.core.errors import InvariantError
from pancakebot.runtime.runtime_loop import _load_dry_bets, _load_dry_settled_epochs


class RuntimeLoopDryStateTests(unittest.TestCase):
    def test_load_dry_bets_rejects_duplicate_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dry_bets.jsonl"
            path.write_text(
                '{"epoch": 1, "bet_bnb": 0.1}\n{"epoch": 1, "bet_bnb": 0.2}\n',
                encoding="utf-8",
            )
            with self.assertRaises(InvariantError):
                _load_dry_bets(str(path))

    def test_load_dry_bets_rejects_bad_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dry_bets.jsonl"
            path.write_text('{"epoch": 1}\nnot-json\n', encoding="utf-8")
            with self.assertRaises(InvariantError):
                _load_dry_bets(str(path))

    def test_load_dry_settled_epochs_rejects_bad_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dry_settled_epochs.txt"
            path.write_text("123\nbad\n", encoding="utf-8")
            with self.assertRaises(InvariantError):
                _load_dry_settled_epochs(str(path))


if __name__ == "__main__":
    unittest.main()
