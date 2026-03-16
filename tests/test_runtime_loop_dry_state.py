"""Tests for dry-state file loading in the runtime loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pancakebot.config.app_config import RuntimeStatePathsConfig
from pancakebot.core.errors import InvariantError
from pancakebot.runtime.runtime_loop import (
    _load_dry_bankroll_state,
    _load_dry_bets,
    _load_dry_settled_epochs,
    _load_dry_settled_epochs_from_audit,
    _resolve_initial_dry_bankroll_state,
    _save_dry_bankroll_state,
)


class _WalletStub:
    def __init__(self, bankroll_bnb: float) -> None:
        self._bankroll_bnb = float(bankroll_bnb)

    def wallet_balance_bnb(self, _wallet_address: str) -> float:
        return float(self._bankroll_bnb)


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

    def test_load_dry_settled_epochs_from_audit_reads_settled_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dry_audit.csv"
            path.write_text(
                "epoch,settled_ts,bankroll_after_settle_bnb\n1,200,50.1\n2,,49.8\n3,400,50.4\n",
                encoding="utf-8",
            )
            loaded = _load_dry_settled_epochs_from_audit(str(path))

        self.assertEqual({1, 3}, loaded)

    def test_dry_bankroll_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dry_bankroll_state.json"
            _save_dry_bankroll_state(
                str(path),
                bankroll_bnb=47.25,
                source="bet",
                epoch=123,
                updated_ts=999,
            )
            state = _load_dry_bankroll_state(str(path))

        self.assertIsNotNone(state)
        assert state is not None
        self.assertAlmostEqual(47.25, float(state.simulated_bankroll_bnb))
        self.assertEqual(999, int(state.updated_ts))
        self.assertEqual("bet", state.source)
        self.assertEqual(123, state.epoch)

    def test_resolve_initial_dry_bankroll_state_recovers_from_latest_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bankroll_state_path = root / "dry_bankroll_state.json"
            dry_bets_path = root / "dry_bets.jsonl"
            dry_audit_path = root / "dry_audit_trades.csv"
            dry_settled_path = root / "dry_settled.txt"

            _save_dry_bankroll_state(
                str(bankroll_state_path),
                bankroll_bnb=50.0,
                source="wallet_init",
                epoch=None,
                updated_ts=100,
            )
            dry_bets_path.write_text(
                (
                    '{"epoch":1,"placed_ts":200,"bankroll_after_bet_bnb":49.5}\n'
                    '{"epoch":2,"placed_ts":400,"bankroll_after_bet_bnb":49.2}\n'
                ),
                encoding="utf-8",
            )
            dry_audit_path.write_text(
                (
                    "epoch,settled_ts,bankroll_after_settle_bnb\n"
                    "1,300,50.1\n"
                ),
                encoding="utf-8",
            )

            cfg = SimpleNamespace(
                runtime_state_paths=RuntimeStatePathsConfig(
                    claim_scan_cursor_path=str(root / "claim.txt"),
                    dry_bets_path=str(dry_bets_path),
                    dry_settled_epochs_path=str(dry_settled_path),
                    dry_audit_trades_path=str(dry_audit_path),
                    dry_bankroll_state_path=str(bankroll_state_path),
                ),
                contract=_WalletStub(55.0),
                wallet_address="0xabc",
            )

            state = _resolve_initial_dry_bankroll_state(cfg)

        self.assertAlmostEqual(49.2, float(state.simulated_bankroll_bnb))
        self.assertEqual("recovered", state.source)
        self.assertEqual(2, state.epoch)

    def test_resolve_initial_dry_bankroll_state_uses_wallet_when_no_state_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = SimpleNamespace(
                runtime_state_paths=RuntimeStatePathsConfig(
                    claim_scan_cursor_path=str(root / "claim.txt"),
                    dry_bets_path=str(root / "dry_bets.jsonl"),
                    dry_settled_epochs_path=str(root / "dry_settled.txt"),
                    dry_audit_trades_path=str(root / "dry_audit.csv"),
                    dry_bankroll_state_path=str(root / "dry_bankroll_state.json"),
                ),
                contract=_WalletStub(61.75),
                wallet_address="0xabc",
            )

            state = _resolve_initial_dry_bankroll_state(cfg)

        self.assertAlmostEqual(61.75, float(state.simulated_bankroll_bnb))
        self.assertEqual("wallet_init", state.source)


if __name__ == "__main__":
    unittest.main()
