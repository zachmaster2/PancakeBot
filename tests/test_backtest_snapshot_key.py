"""Tests for backtest snapshot-key cache scope behavior."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from pancakebot.backtest.config import BacktestConfig
from pancakebot.backtest.runner import _snapshot_key
from pancakebot.config.load_config import load_app_config
from pancakebot.domain.types import Round


def _round(*, epoch: int, lock_at: int) -> Round:
    return Round(
        epoch=int(epoch),
        start_at=int(lock_at) - 300,
        lock_at=int(lock_at),
        close_at=int(lock_at) + 300,
        lock_price=600.0,
        close_price=601.0,
        position="Bull",
        failed=False,
        bets=(),
    )


class BacktestSnapshotKeyTests(unittest.TestCase):
    def test_continuous_initial_key_ignores_simulation_horizon(self) -> None:
        cfg = load_app_config("config.toml")
        runtime_cfg = SimpleNamespace(
            strategy_cfg=cfg.strategy,
            cutoff_seconds=int(cfg.cutoff_seconds),
            treasury_fee_fraction=0.03,
            buffer_seconds=5,
            round_store=SimpleNamespace(path_jsonl=str(cfg.closed_rounds_path)),
            klines_store=SimpleNamespace(path=str(cfg.klines_path)),
        )

        warmup = [_round(epoch=1, lock_at=1000), _round(epoch=2, lock_at=1300)]
        sim_short = [_round(epoch=3, lock_at=1600)]
        sim_long = [
            _round(epoch=3, lock_at=1600),
            _round(epoch=4, lock_at=1900),
            _round(epoch=5, lock_at=2200),
        ]

        bt_2k = BacktestConfig(
            simulation_size=2000,
            initial_bankroll_bnb=50.0,
            reset_mode="continuous",
            reset_every_rounds=0,
        )
        bt_5k = BacktestConfig(
            simulation_size=5000,
            initial_bankroll_bnb=50.0,
            reset_mode="continuous",
            reset_every_rounds=0,
        )

        key_2k = _snapshot_key(
            runtime_cfg=runtime_cfg,
            backtest_cfg=bt_2k,
            reset_mode="continuous",
            warmup_rounds=list(warmup),
            sim_rounds=list(sim_short),
            phase="continuous_initial",
        )
        key_5k = _snapshot_key(
            runtime_cfg=runtime_cfg,
            backtest_cfg=bt_5k,
            reset_mode="continuous",
            warmup_rounds=list(warmup),
            sim_rounds=list(sim_long),
            phase="continuous_initial",
        )
        self.assertEqual(key_2k, key_5k)

        chunk_key_2k = _snapshot_key(
            runtime_cfg=runtime_cfg,
            backtest_cfg=bt_2k,
            reset_mode="continuous",
            warmup_rounds=list(warmup),
            sim_rounds=list(sim_short),
            phase="chunk_1_of_1",
        )
        chunk_key_5k = _snapshot_key(
            runtime_cfg=runtime_cfg,
            backtest_cfg=bt_5k,
            reset_mode="continuous",
            warmup_rounds=list(warmup),
            sim_rounds=list(sim_long),
            phase="chunk_1_of_1",
        )
        self.assertNotEqual(chunk_key_2k, chunk_key_5k)


if __name__ == "__main__":
    unittest.main()
