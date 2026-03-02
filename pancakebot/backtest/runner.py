from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque

from pancakebot.backtest.config import BacktestConfig
from pancakebot.backtest.state_cache import BacktestStateCache, stable_hash
from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import info, warn
from pancakebot.domain.strategy.dislocation_engine import (
    build_dislocation_engine_from_config,
)
from pancakebot.domain.strategy.ml_candidate_adapter import MlCandidateAdapter
from pancakebot.domain.strategy.pipeline import StrategyPipeline
from pancakebot.domain.strategy.router import StrategyRouter, StrategyRouterConfig
from pancakebot.domain.types import Kline, Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round

_BOOTSTRAP_BATCH_ROUNDS = 1000
_BOOTSTRAP_LOG_EVERY_ROUNDS = 5000
_STATE_CACHE_ROOT_DIR = "var/backtest_state_cache"
_STATE_CACHE_VERSION = "backtest_pipeline_state_v1"


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


def _store_file_signature(path: str) -> dict[str, object]:
    p = Path(str(path))
    if not p.exists():
        return {"path": str(path), "exists": False}
    st = p.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _rounds_signature(rounds: list[Round]) -> dict[str, object]:
    if not rounds:
        return {
            "count": 0,
            "first_epoch": None,
            "last_epoch": None,
            "first_lock_at": None,
            "last_lock_at": None,
        }
    first = rounds[0]
    last = rounds[-1]
    return {
        "count": int(len(rounds)),
        "first_epoch": int(first.epoch),
        "last_epoch": int(last.epoch),
        "first_lock_at": int(first.lock_at) if first.lock_at is not None else None,
        "last_lock_at": int(last.lock_at) if last.lock_at is not None else None,
    }


def _snapshot_key(
    *,
    runtime_cfg,
    backtest_cfg: BacktestConfig,
    reset_mode: str,
    warmup_rounds: list[Round],
    sim_rounds: list[Round],
    phase: str,
) -> str:
    strategy_cfg = asdict(runtime_cfg.strategy_cfg)
    payload = {
        "version": str(_STATE_CACHE_VERSION),
        "phase": str(phase),
        "reset_mode": str(reset_mode),
        "cutoff_seconds": int(runtime_cfg.cutoff_seconds),
        "treasury_fee_fraction": float(runtime_cfg.treasury_fee_fraction),
        "buffer_seconds": int(runtime_cfg.buffer_seconds),
        "backtest": {
            "simulation_size": int(backtest_cfg.simulation_size),
            "reset_every_rounds": int(backtest_cfg.reset_every_rounds),
            "initial_bankroll_bnb": float(backtest_cfg.initial_bankroll_bnb),
        },
        "strategy_cfg": strategy_cfg,
        "round_store": _store_file_signature(runtime_cfg.round_store.path_jsonl),
        "kline_store": _store_file_signature(runtime_cfg.klines_store.path),
        "warmup_rounds": _rounds_signature(list(warmup_rounds)),
        "sim_rounds": _rounds_signature(list(sim_rounds)),
    }
    return str(stable_hash(payload))


def _bootstrap_pipeline_with_progress(
    *,
    pipeline: StrategyPipeline,
    warmup_rounds: list[Round],
    phase: str,
) -> None:
    total = int(len(warmup_rounds))
    if int(total) <= 0:
        return
    if int(_BOOTSTRAP_BATCH_ROUNDS) <= 0:
        raise InvariantError("bootstrap_batch_rounds_nonpositive")
    if int(_BOOTSTRAP_LOG_EVERY_ROUNDS) <= 0:
        raise InvariantError("bootstrap_log_every_rounds_nonpositive")

    started = float(time.perf_counter())
    done = 0
    info(
        "BACK",
        "PROG",
        "BOOT",
        msg=f"phase={str(phase)} idx=0/{int(total)}",
    )
    while int(done) < int(total):
        end = min(int(total), int(done) + int(_BOOTSTRAP_BATCH_ROUNDS))
        pipeline.bootstrap_from_closed_rounds(rounds=list(warmup_rounds[int(done): int(end)]))
        done = int(end)

        should_log = (int(done) == int(total)) or (int(done) % int(_BOOTSTRAP_LOG_EVERY_ROUNDS) == 0)
        if not bool(should_log):
            continue

        elapsed = max(1e-6, float(time.perf_counter()) - float(started))
        rate = float(done) / float(elapsed)
        remaining = int(total) - int(done)
        eta = float(remaining) / float(rate) if float(rate) > 0.0 else 0.0
        info(
            "BACK",
            "PROG",
            "BOOT",
            msg=(
                f"phase={str(phase)} idx={int(done)}/{int(total)} "
                f"elapsed={float(elapsed):.1f}s rate={float(rate):.1f}r/s eta={float(eta):.1f}s"
            ),
        )


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
    state_cache = BacktestStateCache(root_dir=str(_STATE_CACHE_ROOT_DIR))

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
            phase = "continuous_initial"
            cache_key = _snapshot_key(
                runtime_cfg=runtime_cfg,
                backtest_cfg=backtest_cfg,
                reset_mode=str(reset_mode),
                warmup_rounds=list(warmup),
                sim_rounds=list(sim_rounds),
                phase=str(phase),
            )
            snapshot_loaded = False
            cached_state = state_cache.load(namespace="pipeline_bootstrap", key=str(cache_key))
            if isinstance(cached_state, dict):
                try:
                    pipeline.import_bootstrap_state(state=cached_state)
                    snapshot_loaded = True
                    info("BACK", "CACHE", "HIT", msg=f"phase={str(phase)} key={str(cache_key)[:16]}")
                except Exception as e:
                    warn("BACK", "CACHE", "LOAD", msg=f"phase={str(phase)} key={str(cache_key)[:16]} err={e}")

            if not bool(snapshot_loaded):
                info("BACK", "CACHE", "MISS", msg=f"phase={str(phase)} key={str(cache_key)[:16]}")
                _bootstrap_pipeline_with_progress(
                    pipeline=pipeline,
                    warmup_rounds=list(warmup),
                    phase=str(phase),
                )
                state_cache.save(
                    namespace="pipeline_bootstrap",
                    key=str(cache_key),
                    value=pipeline.export_bootstrap_state(),
                )
                info("BACK", "CACHE", "SAVE", msg=f"phase={str(phase)} key={str(cache_key)[:16]}")
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
                chunk_phase = f"chunk_{int(chunk_index)}_of_{int(chunk_count)}"
                chunk_cache_key = _snapshot_key(
                    runtime_cfg=runtime_cfg,
                    backtest_cfg=backtest_cfg,
                    reset_mode=str(reset_mode),
                    warmup_rounds=list(chunk_warmup),
                    sim_rounds=list(chunk_rounds),
                    phase=str(chunk_phase),
                )
                chunk_snapshot_loaded = False
                chunk_cached_state = state_cache.load(namespace="pipeline_bootstrap", key=str(chunk_cache_key))
                if isinstance(chunk_cached_state, dict):
                    try:
                        chunk_pipeline.import_bootstrap_state(state=chunk_cached_state)
                        chunk_snapshot_loaded = True
                        info("BACK", "CACHE", "HIT", msg=f"phase={str(chunk_phase)} key={str(chunk_cache_key)[:16]}")
                    except Exception as e:
                        warn(
                            "BACK",
                            "CACHE",
                            "LOAD",
                            msg=f"phase={str(chunk_phase)} key={str(chunk_cache_key)[:16]} err={e}",
                        )

                if not bool(chunk_snapshot_loaded):
                    info("BACK", "CACHE", "MISS", msg=f"phase={str(chunk_phase)} key={str(chunk_cache_key)[:16]}")
                    _bootstrap_pipeline_with_progress(
                        pipeline=chunk_pipeline,
                        warmup_rounds=list(chunk_warmup),
                        phase=str(chunk_phase),
                    )
                    state_cache.save(
                        namespace="pipeline_bootstrap",
                        key=str(chunk_cache_key),
                        value=chunk_pipeline.export_bootstrap_state(),
                    )
                    info("BACK", "CACHE", "SAVE", msg=f"phase={str(chunk_phase)} key={str(chunk_cache_key)[:16]}")
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
        feature_cache_store = getattr(runtime_cfg, "feature_cache_store", None)
        if feature_cache_store is not None:
            try:
                if hasattr(feature_cache_store, "flush"):
                    feature_cache_store.flush()
                if hasattr(feature_cache_store, "close"):
                    feature_cache_store.close()
            except Exception as e:
                warn("BACK", "CACHE", "FLUSH", msg=f"err={e}")


def run_backtest(*, runtime_cfg, backtest_cfg: BacktestConfig, out_dir: Path) -> None:
    """Run a deterministic replay over the most recent closed rounds.

    Backtest MUST NOT fetch any data. It consumes the on-disk round store and kline store.

    Artifacts are written incrementally to `{out_dir}/`:
      - backtest_trades.csv
      - backtest_summary.json
    """

    _run_backtest_dislocation(runtime_cfg=runtime_cfg, backtest_cfg=backtest_cfg, out_dir=out_dir)
