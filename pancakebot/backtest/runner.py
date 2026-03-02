from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

from pancakebot.backtest.config import BacktestConfig
from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import info
from pancakebot.domain.strategy.dislocation_engine import (
    build_dislocation_engine_from_config,
)
from pancakebot.domain.strategy.ml_candidate_adapter import MlCandidateAdapter
from pancakebot.domain.strategy.pipeline import StrategyPipeline
from pancakebot.domain.strategy.router import StrategyRouter, StrategyRouterConfig
from pancakebot.domain.types import Kline, Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round


def _tail_rounds(store, *, n: int) -> list[Round]:
    if n <= 0:
        raise InvariantError("tail_rounds_n_invalid")

    dq: Deque[Round] = deque(maxlen=n)
    for r in store.iter_closed_rounds():
        dq.append(r)
    return list(dq)


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _all_klines_from_store(klines_store) -> list[Kline]:
    start = klines_store.earliest_open_time_ms()
    end = klines_store.latest_open_time_ms()
    if start is None or end is None:
        raise InvariantError("backtest_dislocation_klines_store_empty")
    out = klines_store.get_klines_between(
        start_open_time_ms=int(start),
        end_open_time_ms=int(end) + 60_000,
    )
    if not out:
        raise InvariantError("backtest_dislocation_klines_empty")
    return list(out)


def _build_dislocation_engine(*, runtime_cfg, all_klines: list[Kline]):
    engine = build_dislocation_engine_from_config(
        selector_cfg=runtime_cfg.strategy_cfg.dislocation.selector,
        candidate_cfgs=runtime_cfg.strategy_cfg.dislocation.candidates,
        treasury_fee_fraction=float(runtime_cfg.treasury_fee_fraction),
        cutoff_seconds=int(runtime_cfg.cutoff_seconds),
    )
    engine.refresh_klines(list(all_klines))
    return engine


def _build_router(*, runtime_cfg) -> StrategyRouter:
    """Build shared router from strategy config."""

    router_cfg = StrategyRouterConfig(
        mode=str(runtime_cfg.strategy_cfg.router.mode),
        score_threshold_bnb=float(runtime_cfg.strategy_cfg.router.score_threshold_bnb),
        online_warmup_rounds=int(runtime_cfg.strategy_cfg.router.online_warmup_rounds),
        online_num_quantile_bins=int(runtime_cfg.strategy_cfg.router.online_num_quantile_bins),
        online_min_cell_obs=int(runtime_cfg.strategy_cfg.router.online_min_cell_obs),
        online_score_threshold_bnb=float(runtime_cfg.strategy_cfg.router.online_score_threshold_bnb),
        online_use_direction_split=bool(runtime_cfg.strategy_cfg.router.online_use_direction_split),
    )
    return StrategyRouter(config=router_cfg)


def _build_ml_candidate_adapter(*, runtime_cfg) -> MlCandidateAdapter | None:
    """Build optional ML candidate adapter from shared strategy config."""

    ml_cfg = runtime_cfg.strategy_cfg.ml_candidate
    if not bool(ml_cfg.enabled):
        return None
    return MlCandidateAdapter(
        config=ml_cfg,
        cutoff_seconds=int(runtime_cfg.cutoff_seconds),
        treasury_fee_fraction=float(runtime_cfg.treasury_fee_fraction),
        klines_store_like=runtime_cfg.klines_store,
        feature_cache_store=runtime_cfg.feature_cache_store,
    )


def _build_strategy_pipeline(*, runtime_cfg, all_klines: list[Kline]) -> StrategyPipeline:
    """Build shared strategy pipeline for backtest replay."""

    dislocation_engine = _build_dislocation_engine(runtime_cfg=runtime_cfg, all_klines=all_klines)
    router = _build_router(runtime_cfg=runtime_cfg)
    ml_adapter = _build_ml_candidate_adapter(runtime_cfg=runtime_cfg)
    pipeline = StrategyPipeline(
        dislocation_engine=dislocation_engine,
        router=router,
        treasury_fee_fraction=float(runtime_cfg.treasury_fee_fraction),
        ml_candidate_adapter=ml_adapter,
    )
    pipeline.refresh_klines(klines=list(all_klines))
    return pipeline


def _chunk_bootstrap_rounds(
    *,
    closed_rounds: list[Round],
    warmup_rounds: int,
    sim_chunk_start_idx: int,
) -> list[Round]:
    start = int(sim_chunk_start_idx)
    end = int(start) + int(warmup_rounds)
    out = closed_rounds[int(start): int(end)]
    if len(out) != int(warmup_rounds):
        raise InvariantError("backtest_chunk_reset_bootstrap_len_mismatch")
    return list(out)


@dataclass(slots=True)
class _BacktestStats:
    num_bets: int = 0
    num_bets_bull: int = 0
    num_bets_bear: int = 0
    num_wins: int = 0
    num_wins_bull: int = 0
    num_wins_bear: int = 0
    gross_profit_bnb: float = 0.0
    gross_loss_bnb: float = 0.0
    skip_counts_by_reason: dict[str, int] = field(default_factory=dict)

    def count_skip(self, reason: str) -> None:
        key = str(reason).strip()
        if key == "":
            key = "unknown_skip_reason"
        self.skip_counts_by_reason[key] = int(self.skip_counts_by_reason.get(key, 0)) + 1


def _simulate_rounds(
    *,
    pipeline: StrategyPipeline,
    rounds: list[Round],
    runtime_cfg,
    trades_w,
    stats: _BacktestStats,
    bankroll: float,
    rounds_done_before_chunk: int,
    total_sim_rounds: int,
) -> tuple[float, int]:
    rounds_done = int(rounds_done_before_chunk)
    for round_t in rounds:
        decision = pipeline.decide_open_round(
            round_t=round_t,
            bankroll_bnb=float(bankroll),
            allow_oracle_mode=True,
        )

        ev = float(decision.expected_profit_bnb)
        profit = 0.0

        if decision.action == "BET" and float(decision.bet_size_bnb) > 0.0:
            bet_side = str(decision.bet_side)
            if bet_side not in ("Bull", "Bear"):
                raise InvariantError("backtest_dislocation_bet_side_invalid")

            bankroll -= float(decision.bet_size_bnb) + float(GAS_COST_BET_BNB)
            outcome = settle_bet_against_closed_round(
                bet_bnb=float(decision.bet_size_bnb),
                bet_side=str(decision.bet_side),
                round_closed=round_t,
                treasury_fee_fraction=float(runtime_cfg.treasury_fee_fraction),
            )
            credit_bnb = float(outcome.credit_bnb)
            bankroll += float(credit_bnb)
            profit = float(credit_bnb) - float(decision.bet_size_bnb) - float(GAS_COST_BET_BNB)

            stats.num_bets += 1
            if bet_side == "Bull":
                stats.num_bets_bull += 1
            else:
                stats.num_bets_bear += 1

            if str(outcome.outcome) == "win":
                stats.num_wins += 1
                if bet_side == "Bull":
                    stats.num_wins_bull += 1
                else:
                    stats.num_wins_bear += 1

            if float(profit) > 0.0:
                stats.gross_profit_bnb += float(profit)
            elif float(profit) < 0.0:
                stats.gross_loss_bnb += float(-float(profit))
        else:
            stats.count_skip(str(decision.skip_reason or "unknown_skip_reason"))

        p_final = float(decision.p_bull) if decision.p_bull is not None else 0.5
        trades_w.writerow(
            [
                int(round_t.epoch),
                str(decision.action),
                str(decision.skip_reason or ""),
                str(decision.bet_side or ""),
                float(decision.bet_size_bnb),
                float(p_final),
                0.0,
                0.0,
                0.0,
                float(ev),
                float(profit),
                float(bankroll),
                str(decision.selected_strategy or ""),
                str(pipeline.router_mode),
                (
                    float(decision.selector_score_bnb)
                    if decision.selector_score_bnb is not None
                    else ""
                ),
            ]
        )

        pipeline.settle_closed_rounds(rounds=[round_t])
        rounds_done += 1

        if int(rounds_done) % 250 == 0:
            info(
                "BACK",
                "PROG",
                "SIM",
                msg=f"idx={int(rounds_done)}/{int(total_sim_rounds)} bankroll={float(bankroll):.4f} BNB",
            )

    return float(bankroll), int(rounds_done)


def _resolve_reset_settings(backtest_cfg: BacktestConfig) -> tuple[str, int]:
    mode = str(backtest_cfg.reset_mode).strip()
    interval = int(backtest_cfg.reset_every_rounds)
    if mode not in ("continuous", "chunk_reset"):
        raise InvariantError("backtest_reset_mode_not_supported")
    if mode == "chunk_reset" and int(interval) <= 0:
        raise InvariantError("backtest_chunk_reset_every_rounds_must_be_positive")
    if mode == "continuous":
        interval = 0
    return str(mode), int(interval)


def _run_backtest_dislocation(*, runtime_cfg, backtest_cfg: BacktestConfig, out_dir: Path) -> None:
    backtest_cfg.validate()
    simulation_size = int(backtest_cfg.simulation_size)
    reset_mode, reset_every_rounds = _resolve_reset_settings(backtest_cfg)

    warmup_rounds = int(runtime_cfg.strategy_cfg.dislocation.selector.warmup_rounds)
    if int(warmup_rounds) <= 0:
        raise InvariantError("dislocation_warmup_rounds_nonpositive")

    total_n = int(warmup_rounds) + int(simulation_size)
    closed_rounds = _tail_rounds(runtime_cfg.round_store, n=total_n)
    if len(closed_rounds) < int(total_n):
        raise InvariantError("backtest_insufficient_closed_rounds_for_dislocation")

    warmup = closed_rounds[: int(warmup_rounds)]
    sim_rounds = closed_rounds[int(warmup_rounds):]
    if len(sim_rounds) != int(simulation_size):
        raise InvariantError("backtest_dislocation_sim_rounds_len_mismatch")

    all_klines = _all_klines_from_store(runtime_cfg.klines_store)
    router_cfg = runtime_cfg.strategy_cfg.router

    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "backtest_trades.csv"
    summary_path = out_dir / "backtest_summary.json"

    trades_f = open(trades_path, "w", newline="")
    try:
        trades_w = csv.writer(trades_f)
        trades_w.writerow(
            [
                "epoch",
                "action",
                "skip_reason",
                "direction",
                "bet_size_bnb",
                "p_final",
                "final_total_bnb",
                "final_bull_bnb",
                "final_bear_bnb",
                "ev_bnb",
                "profit_bnb",
                "bankroll_bnb",
                "selected_strategy",
                "router_mode",
                "selector_score_bnb",
            ]
        )

        initial_bankroll_bnb = float(backtest_cfg.initial_bankroll_bnb)
        bankroll = float(initial_bankroll_bnb)
        stats = _BacktestStats()
        rounds_done = 0

        if reset_mode == "continuous":
            pipeline = _build_strategy_pipeline(runtime_cfg=runtime_cfg, all_klines=all_klines)
            pipeline.bootstrap_from_closed_rounds(rounds=list(warmup))
            info(
                "BACK",
                "INIT",
                "DISLOC",
                msg=(
                    f"mode={str(reset_mode)} warmup_n={len(warmup)} sim_n={len(sim_rounds)} "
                    f"selector_warmup={int(runtime_cfg.strategy_cfg.dislocation.selector.warmup_rounds)} "
                    f"router_mode={str(router_cfg.mode)} "
                    f"router_score_threshold_bnb={float(router_cfg.score_threshold_bnb):.6f} "
                    f"router_online_warmup_rounds={int(router_cfg.online_warmup_rounds)} "
                    f"ml_candidate_enabled={bool(runtime_cfg.strategy_cfg.ml_candidate.enabled)}"
                ),
            )
            bankroll, rounds_done = _simulate_rounds(
                pipeline=pipeline,
                rounds=sim_rounds,
                runtime_cfg=runtime_cfg,
                trades_w=trades_w,
                stats=stats,
                bankroll=float(bankroll),
                rounds_done_before_chunk=int(rounds_done),
                total_sim_rounds=int(len(sim_rounds)),
            )
        elif reset_mode == "chunk_reset":
            interval = int(reset_every_rounds)
            chunk_count = (int(len(sim_rounds)) + int(interval) - 1) // int(interval)
            info(
                "BACK",
                "INIT",
                "DISLOC",
                msg=(
                    f"mode={str(reset_mode)} reset_every_rounds={int(interval)} "
                    f"warmup_n={len(warmup)} sim_n={len(sim_rounds)} chunks={int(chunk_count)} "
                    f"router_mode={str(router_cfg.mode)} "
                    f"router_score_threshold_bnb={float(router_cfg.score_threshold_bnb):.6f} "
                    f"router_online_warmup_rounds={int(router_cfg.online_warmup_rounds)} "
                    f"ml_candidate_enabled={bool(runtime_cfg.strategy_cfg.ml_candidate.enabled)}"
                ),
            )
            for chunk_index, chunk_start in enumerate(range(0, len(sim_rounds), interval), start=1):
                chunk_end = min(int(chunk_start) + int(interval), len(sim_rounds))
                chunk_rounds = sim_rounds[int(chunk_start): int(chunk_end)]
                chunk_warmup = _chunk_bootstrap_rounds(
                    closed_rounds=closed_rounds,
                    warmup_rounds=int(warmup_rounds),
                    sim_chunk_start_idx=int(chunk_start),
                )

                info(
                    "BACK",
                    "CHUNK",
                    "RESET",
                    msg=(
                        f"chunk={int(chunk_index)}/{int(chunk_count)} sim_idx=[{int(chunk_start)}..{int(chunk_end)-1}] "
                        f"chunk_n={len(chunk_rounds)} warmup_n={len(chunk_warmup)} bankroll_in={float(bankroll):.4f} BNB"
                    ),
                )

                chunk_pipeline = _build_strategy_pipeline(runtime_cfg=runtime_cfg, all_klines=all_klines)
                chunk_pipeline.bootstrap_from_closed_rounds(rounds=list(chunk_warmup))
                bankroll, rounds_done = _simulate_rounds(
                    pipeline=chunk_pipeline,
                    rounds=list(chunk_rounds),
                    runtime_cfg=runtime_cfg,
                    trades_w=trades_w,
                    stats=stats,
                    bankroll=float(bankroll),
                    rounds_done_before_chunk=int(rounds_done),
                    total_sim_rounds=int(len(sim_rounds)),
                )
        else:
            raise InvariantError("backtest_reset_mode_unreachable")

        num_rounds = int(len(sim_rounds))
        num_skips = int(num_rounds) - int(stats.num_bets)
        classified_skips = int(sum(stats.skip_counts_by_reason.values()))
        unclassified_skips = int(num_skips) - int(classified_skips)
        if unclassified_skips > 0:
            stats.count_skip("unclassified_skip")

        summary = {
            "reset_mode": str(reset_mode),
            "reset_every_rounds": int(reset_every_rounds),
            "router_mode": str(router_cfg.mode),
            "router_score_threshold_bnb": float(router_cfg.score_threshold_bnb),
            "router_online_warmup_rounds": int(router_cfg.online_warmup_rounds),
            "ml_candidate_enabled": bool(runtime_cfg.strategy_cfg.ml_candidate.enabled),
            "num_rounds": int(num_rounds),
            "num_bets": int(stats.num_bets),
            "num_skips": int(num_skips),
            "num_wins": int(stats.num_wins),
            "num_bets_bull": int(stats.num_bets_bull),
            "num_bets_bear": int(stats.num_bets_bear),
            "num_wins_bull": int(stats.num_wins_bull),
            "num_wins_bear": int(stats.num_wins_bear),
            "bet_rate": float(_safe_rate(stats.num_bets, num_rounds)),
            "bet_rate_bull": float(_safe_rate(stats.num_bets_bull, stats.num_bets)),
            "bet_rate_bear": float(_safe_rate(stats.num_bets_bear, stats.num_bets)),
            "win_rate": float(_safe_rate(stats.num_wins, stats.num_bets)),
            "win_rate_bull": float(_safe_rate(stats.num_wins_bull, stats.num_bets_bull)),
            "win_rate_bear": float(_safe_rate(stats.num_wins_bear, stats.num_bets_bear)),
            "initial_bankroll_bnb": float(initial_bankroll_bnb),
            "final_bankroll_bnb": float(bankroll),
            "gross_profit_bnb": float(stats.gross_profit_bnb),
            "gross_loss_bnb": float(stats.gross_loss_bnb),
            "net_profit_bnb": float(bankroll - float(initial_bankroll_bnb)),
            "num_skips_by_reason": {str(k): int(v) for k, v in sorted(stats.skip_counts_by_reason.items())},
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        info("BACK", "DONE", "SIM", msg=f"Final bankroll={float(bankroll):.4f} BNB")

    finally:
        trades_f.close()


def run_backtest(*, runtime_cfg, backtest_cfg: BacktestConfig, out_dir: Path) -> None:
    """Run a deterministic replay over the most recent closed rounds.

    Backtest MUST NOT fetch any data. It consumes the on-disk round store and kline store.

    Artifacts are written incrementally to `{out_dir}/`:
      - backtest_trades.csv
      - backtest_summary.json
    """

    _run_backtest_dislocation(runtime_cfg=runtime_cfg, backtest_cfg=backtest_cfg, out_dir=out_dir)
