from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pancakebot.config.load_config import load_app_config
from pancakebot.config.app_config import RuntimeStatePathsConfig
from pancakebot.domain.closed_rounds_cache import RollingClosedRoundsCache
from pancakebot.domain.types import Round
from pancakebot.runtime.runtime_loop import (
    _ClosedState,
    _bootstrap_strategy_pipeline_from_runtime_snapshot,
)


def _round(*, epoch: int, close_price: float = 600.0) -> Round:
    return Round(
        epoch=int(epoch),
        start_at=1_000_000 + int(epoch) * 300,
        lock_at=1_000_000 + int(epoch) * 300 + 300,
        close_at=1_000_000 + int(epoch) * 300 + 600,
        lock_price=float(close_price),
        close_price=float(close_price),
        position="Bull",
        failed=False,
        bets=(),
    )


class _FakePipeline:
    def __init__(self) -> None:
        self.last_settled_epoch: int | None = None
        self.bootstrap_calls: list[list[int]] = []
        self.imported_states: list[dict[str, object]] = []

    def bootstrap_from_closed_rounds(self, *, rounds: list[Round]) -> None:
        epochs = [int(round_t.epoch) for round_t in rounds]
        self.bootstrap_calls.append(list(epochs))
        if epochs:
            self.last_settled_epoch = int(epochs[-1])

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        self.imported_states.append(dict(state))
        epoch = state.get("last_settled_epoch")
        self.last_settled_epoch = None if epoch is None else int(epoch)

    def export_bootstrap_state(self) -> dict[str, object]:
        return {
            "last_settled_epoch": (
                None if self.last_settled_epoch is None else int(self.last_settled_epoch)
            ),
            "marker": "fake_pipeline_state",
        }


class RuntimePipelineSnapshotTests(unittest.TestCase):
    def _cfg(self, *, root: Path) -> object:
        app_cfg = load_app_config("config.toml")
        return SimpleNamespace(
            dry=True,
            strategy_cfg=app_cfg.strategy,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
            treasury_fee_fraction=0.03,
            round_store=SimpleNamespace(path_jsonl=str(root / "closed_rounds.jsonl")),
            klines_store=SimpleNamespace(path=str(root / "klines.jsonl")),
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
        )

    @staticmethod
    def _closed_state(*, rounds: list[Round], pipeline: _FakePipeline) -> _ClosedState:
        cache = RollingClosedRoundsCache(rounds=list(rounds), capacity=max(2, len(rounds)))
        return _ClosedState(
            cache=cache,
            disk_latest_epoch=int(rounds[-1].epoch),
            klines_cache=object(),
            strategy_pipeline=pipeline,
        )

    def test_runtime_snapshot_reuses_prior_state_and_replays_only_delta_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._cfg(root=root)

            first_pipeline = _FakePipeline()
            first_rounds = [_round(epoch=1), _round(epoch=2), _round(epoch=3)]
            first_closed = self._closed_state(rounds=first_rounds, pipeline=first_pipeline)
            _bootstrap_strategy_pipeline_from_runtime_snapshot(
                cfg=cfg,
                closed=first_closed,
                rounds_all=list(first_rounds),
                warmup_rounds=list(first_rounds),
            )

            self.assertEqual([[1, 2, 3]], first_pipeline.bootstrap_calls)
            self.assertEqual(3, first_closed.pipeline_snapshot_saved_epoch)

            second_pipeline = _FakePipeline()
            second_rounds = [_round(epoch=1), _round(epoch=2), _round(epoch=3), _round(epoch=4), _round(epoch=5)]
            second_warmup = list(second_rounds[-3:])
            second_closed = self._closed_state(rounds=second_warmup, pipeline=second_pipeline)
            _bootstrap_strategy_pipeline_from_runtime_snapshot(
                cfg=cfg,
                closed=second_closed,
                rounds_all=list(second_rounds),
                warmup_rounds=list(second_warmup),
            )

            self.assertEqual(1, len(second_pipeline.imported_states))
            self.assertEqual([[4, 5]], second_pipeline.bootstrap_calls)
            self.assertEqual(5, second_closed.pipeline_snapshot_saved_epoch)

    def test_runtime_snapshot_invalidates_when_last_settled_round_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._cfg(root=root)

            saved_pipeline = _FakePipeline()
            saved_rounds = [_round(epoch=1), _round(epoch=2), _round(epoch=3, close_price=600.0)]
            saved_closed = self._closed_state(rounds=saved_rounds, pipeline=saved_pipeline)
            _bootstrap_strategy_pipeline_from_runtime_snapshot(
                cfg=cfg,
                closed=saved_closed,
                rounds_all=list(saved_rounds),
                warmup_rounds=list(saved_rounds),
            )

            fresh_pipeline = _FakePipeline()
            changed_rounds = [_round(epoch=1), _round(epoch=2), _round(epoch=3, close_price=601.0)]
            fresh_closed = self._closed_state(rounds=changed_rounds, pipeline=fresh_pipeline)
            _bootstrap_strategy_pipeline_from_runtime_snapshot(
                cfg=cfg,
                closed=fresh_closed,
                rounds_all=list(changed_rounds),
                warmup_rounds=list(changed_rounds),
            )

            self.assertEqual([], fresh_pipeline.imported_states)
            self.assertEqual([[1, 2, 3]], fresh_pipeline.bootstrap_calls)


if __name__ == "__main__":
    unittest.main()
