"""Tests for dry-state file loading in the runtime loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pancakebot.config.app_config import RuntimeStatePathsConfig
from pancakebot.core.errors import InvariantError, TransientRpcError
from pancakebot.domain.types import Bet, Round
from pancakebot.runtime.runtime_loop import (
    _archive_dry_runtime_state,
    _ensure_dry_cycle_audit_csv,
    _load_dry_bankroll_state,
    _load_dry_bets,
    _load_dry_settled_epochs,
    _load_dry_settled_epochs_from_audit,
    _round_pool_snapshot,
    _resolve_initial_dry_bankroll_state,
    _save_dry_bankroll_state,
)


class _WalletStub:
    def __init__(self, bankroll_bnb: float) -> None:
        self._bankroll_bnb = float(bankroll_bnb)

    def wallet_balance_bnb(self, _wallet_address: str) -> float:
        return float(self._bankroll_bnb)


class _FlakyWalletStub:
    def __init__(self, bankroll_bnb: float, *, failures_before_success: int) -> None:
        self._bankroll_bnb = float(bankroll_bnb)
        self._failures_before_success = int(failures_before_success)
        self.calls = 0

    def wallet_balance_bnb(self, _wallet_address: str) -> float:
        self.calls += 1
        if self._failures_before_success > 0:
            self._failures_before_success -= 1
            raise TransientRpcError("wallet_balance_bnb_failed: simulated_reset")
        return float(self._bankroll_bnb)


class RuntimeLoopDryStateTests(unittest.TestCase):
    def test_archive_dry_runtime_state_moves_files_and_writes_meta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime_dir = root / "var" / "runtime"
            archive_root = root / "exp"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "dry_bets.jsonl").write_text('{"epoch":1}\n', encoding="utf-8")
            (runtime_dir / "dry_cycle_audit.csv").write_text("h\n", encoding="utf-8")
            paths = RuntimeStatePathsConfig(
                claim_scan_cursor_path=str(runtime_dir / "claim.txt"),
                dry_bets_path=str(runtime_dir / "dry_bets.jsonl"),
                dry_settled_epochs_path=str(runtime_dir / "dry_settled_epochs.txt"),
                dry_audit_trades_path=str(runtime_dir / "dry_audit_trades.csv"),
                dry_cycle_audit_path=str(runtime_dir / "dry_cycle_audit.csv"),
                dry_bankroll_state_path=str(runtime_dir / "dry_bankroll_state.json"),
                dry_pipeline_bootstrap_state_path=str(runtime_dir / "dry_pipeline.pkl.gz"),
                live_pipeline_bootstrap_state_path=str(runtime_dir / "live_pipeline.pkl.gz"),
            )

            with patch("pancakebot.runtime.runtime_loop._DRY_RUNTIME_ARCHIVE_ROOT", archive_root):
                archive_dir = _archive_dry_runtime_state(paths, reason="startup_fresh_reset", move_files=True)

            self.assertIsNotNone(archive_dir)
            assert archive_dir is not None
            self.assertFalse((runtime_dir / "dry_bets.jsonl").exists())
            self.assertFalse((runtime_dir / "dry_cycle_audit.csv").exists())
            self.assertTrue((archive_dir / "dry_bets.jsonl").exists())
            self.assertTrue((archive_dir / "dry_cycle_audit.csv").exists())
            meta = (archive_dir / "archive_meta.json").read_text(encoding="utf-8")
            self.assertIn('"reason": "startup_fresh_reset"', meta)

    def test_archive_dry_runtime_state_can_copy_without_removing_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime_dir = root / "var" / "runtime"
            archive_root = root / "exp"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "dry_bankroll_state.json").write_text("{}", encoding="utf-8")
            paths = RuntimeStatePathsConfig(
                claim_scan_cursor_path=str(runtime_dir / "claim.txt"),
                dry_bets_path=str(runtime_dir / "dry_bets.jsonl"),
                dry_settled_epochs_path=str(runtime_dir / "dry_settled_epochs.txt"),
                dry_audit_trades_path=str(runtime_dir / "dry_audit_trades.csv"),
                dry_cycle_audit_path=str(runtime_dir / "dry_cycle_audit.csv"),
                dry_bankroll_state_path=str(runtime_dir / "dry_bankroll_state.json"),
                dry_pipeline_bootstrap_state_path=str(runtime_dir / "dry_pipeline.pkl.gz"),
                live_pipeline_bootstrap_state_path=str(runtime_dir / "live_pipeline.pkl.gz"),
            )

            with patch("pancakebot.runtime.runtime_loop._DRY_RUNTIME_ARCHIVE_ROOT", archive_root):
                archive_dir = _archive_dry_runtime_state(paths, reason="shutdown_snapshot", move_files=False)

            self.assertIsNotNone(archive_dir)
            assert archive_dir is not None
            self.assertTrue((runtime_dir / "dry_bankroll_state.json").exists())
            self.assertTrue((archive_dir / "dry_bankroll_state.json").exists())

    def test_ensure_dry_cycle_audit_csv_resets_stale_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dry_cycle_audit.csv"
            path.write_text("stale\nrow\n", encoding="utf-8")

            cols = _ensure_dry_cycle_audit_csv(str(path), reset=True)
            written = path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(1, len(written))
        self.assertEqual(",".join(cols), written[0])

    def test_round_pool_snapshot_reports_observed_and_cutoff_used_values(self) -> None:
        round_t = Round(
            epoch=1,
            start_at=100,
            lock_at=200,
            close_at=None,
            lock_price=None,
            close_price=None,
            position=None,
            failed=None,
            bets=(
                Bet(wallet_address="0x1", amount_wei=10**17, position="Bull", created_at=180),
                Bet(wallet_address="0x2", amount_wei=2 * 10**17, position="Bear", created_at=183),
                Bet(wallet_address="0x3", amount_wei=3 * 10**17, position="Bull", created_at=185),
            ),
        )

        observed = _round_pool_snapshot(round_t, prefix="observed")
        cutoff_used = _round_pool_snapshot(round_t, prefix="cutoff_used", cutoff_ts=183)

        self.assertAlmostEqual(0.6, float(observed["observed_total_pool_bnb"]))
        self.assertEqual(3, int(observed["observed_total_bets"]))
        self.assertAlmostEqual(0.3, float(cutoff_used["cutoff_used_total_pool_bnb"]))
        self.assertAlmostEqual(0.1, float(cutoff_used["cutoff_used_bull_pool_bnb"]))
        self.assertAlmostEqual(0.2, float(cutoff_used["cutoff_used_bear_pool_bnb"]))
        self.assertEqual(2, int(cutoff_used["cutoff_used_total_bets"]))

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
                    dry_cycle_audit_path=str(root / "dry_cycle_audit.csv"),
                    dry_bankroll_state_path=str(bankroll_state_path),
                    dry_pipeline_bootstrap_state_path=str(root / "dry_pipeline.pkl.gz"),
                    live_pipeline_bootstrap_state_path=str(root / "live_pipeline.pkl.gz"),
                ),
                contract=_WalletStub(55.0),
                wallet_address="0xabc",
                dry_initial_bankroll_bnb=50.0,
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
                    dry_cycle_audit_path=str(root / "dry_cycle_audit.csv"),
                    dry_bankroll_state_path=str(root / "dry_bankroll_state.json"),
                    dry_pipeline_bootstrap_state_path=str(root / "dry_pipeline.pkl.gz"),
                    live_pipeline_bootstrap_state_path=str(root / "live_pipeline.pkl.gz"),
                ),
                contract=_WalletStub(61.75),
                wallet_address="0xabc",
                dry_initial_bankroll_bnb=None,
            )

            state = _resolve_initial_dry_bankroll_state(cfg)

        self.assertAlmostEqual(61.75, float(state.simulated_bankroll_bnb))
        self.assertEqual("wallet_init", state.source)

    def test_resolve_initial_dry_bankroll_state_retries_transient_wallet_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wallet = _FlakyWalletStub(61.75, failures_before_success=1)
            cfg = SimpleNamespace(
                runtime_state_paths=RuntimeStatePathsConfig(
                    claim_scan_cursor_path=str(root / "claim.txt"),
                    dry_bets_path=str(root / "dry_bets.jsonl"),
                    dry_settled_epochs_path=str(root / "dry_settled.txt"),
                    dry_audit_trades_path=str(root / "dry_audit.csv"),
                    dry_cycle_audit_path=str(root / "dry_cycle_audit.csv"),
                    dry_bankroll_state_path=str(root / "dry_bankroll_state.json"),
                    dry_pipeline_bootstrap_state_path=str(root / "dry_pipeline.pkl.gz"),
                    live_pipeline_bootstrap_state_path=str(root / "live_pipeline.pkl.gz"),
                ),
                contract=wallet,
                wallet_address="0xabc",
                dry_initial_bankroll_bnb=None,
            )

            with patch("pancakebot.runtime.runtime_loop.sleep_seconds", return_value=None):
                state = _resolve_initial_dry_bankroll_state(cfg)

        self.assertAlmostEqual(61.75, float(state.simulated_bankroll_bnb))
        self.assertEqual("wallet_init", state.source)
        self.assertEqual(2, wallet.calls)

    def test_resolve_initial_dry_bankroll_state_uses_configured_seed_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = SimpleNamespace(
                runtime_state_paths=RuntimeStatePathsConfig(
                    claim_scan_cursor_path=str(root / "claim.txt"),
                    dry_bets_path=str(root / "dry_bets.jsonl"),
                    dry_settled_epochs_path=str(root / "dry_settled.txt"),
                    dry_audit_trades_path=str(root / "dry_audit.csv"),
                    dry_cycle_audit_path=str(root / "dry_cycle_audit.csv"),
                    dry_bankroll_state_path=str(root / "dry_bankroll_state.json"),
                    dry_pipeline_bootstrap_state_path=str(root / "dry_pipeline.pkl.gz"),
                    live_pipeline_bootstrap_state_path=str(root / "live_pipeline.pkl.gz"),
                ),
                contract=_WalletStub(0.2333411609),
                wallet_address="0xabc",
                dry_initial_bankroll_bnb=50.0,
            )

            state = _resolve_initial_dry_bankroll_state(cfg)

        self.assertAlmostEqual(50.0, float(state.simulated_bankroll_bnb))
        self.assertEqual("configured_init", state.source)

    def test_resolve_initial_dry_bankroll_state_configured_seed_overrides_stale_wallet_init_without_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bankroll_state_path = root / "dry_bankroll_state.json"
            _save_dry_bankroll_state(
                str(bankroll_state_path),
                bankroll_bnb=0.2333411609,
                source="wallet_init",
                epoch=None,
                updated_ts=100,
            )
            cfg = SimpleNamespace(
                runtime_state_paths=RuntimeStatePathsConfig(
                    claim_scan_cursor_path=str(root / "claim.txt"),
                    dry_bets_path=str(root / "dry_bets.jsonl"),
                    dry_settled_epochs_path=str(root / "dry_settled.txt"),
                    dry_audit_trades_path=str(root / "dry_audit.csv"),
                    dry_cycle_audit_path=str(root / "dry_cycle_audit.csv"),
                    dry_bankroll_state_path=str(bankroll_state_path),
                    dry_pipeline_bootstrap_state_path=str(root / "dry_pipeline.pkl.gz"),
                    live_pipeline_bootstrap_state_path=str(root / "live_pipeline.pkl.gz"),
                ),
                contract=_WalletStub(0.2333411609),
                wallet_address="0xabc",
                dry_initial_bankroll_bnb=50.0,
            )

            state = _resolve_initial_dry_bankroll_state(cfg)

        self.assertAlmostEqual(50.0, float(state.simulated_bankroll_bnb))
        self.assertEqual("configured_init", state.source)


if __name__ == "__main__":
    unittest.main()
