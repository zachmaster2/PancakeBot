"""Step 14 — creative risk-control architecture test.

Four ideas, 3 variants each + 1 baseline ref + stretch goal #8 (combined best).
All at 5 BNB scale with production-faithful pre-decision drawdown check
(inherits Step 13c v2's gate-validated AdaptiveBankrollTracker pattern).

Ideas:
  #1 Graduated re-entry: ramp_linear / ramp_step_4 / ramp_fast
  #3 Shadow betting w/ early exit: shadow_exit_10pos / shadow_exit_20pos / shadow_extend
  #4 Drawdown-velocity breaker: velocity_5pct_10rd / velocity_10pct_30rd / velocity_OR_abs
  #5 Rolling-WR sizing: wr_size_50 / wr_size_55 / wr_size_50_with_breaker

Verification gate FIRST: static dd=0.15 @ 5 BNB must match Step 12b +44.87 ± 1 BNB.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np  # type: ignore

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")
import research.in_process_runner as ipr  # noqa: E402
ipr._EXT_CLOSED_ROUNDS_PATH = EXT_DIR / "closed_rounds.jsonl"
ipr._EXT_BTC_KLINES_PATH = EXT_DIR / "btc_spot_prices.jsonl"
ipr._EXT_ETH_KLINES_PATH = EXT_DIR / "eth_spot_prices.jsonl"
ipr._EXT_SOL_KLINES_PATH = EXT_DIR / "sol_spot_prices.jsonl"

from pancakebot.config import load_strategy_config_from_dict  # noqa: E402
from pancakebot.constants import MAX_GAS_COST_BET_BNB  # noqa: E402
from pancakebot.settlement import settle_bet_against_closed_round  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402
from pancakebot.bankroll_tracker import InMemoryBankrollTracker  # noqa: E402


EPOCH_MIN = 422298
EPOCH_MAX = 484999
CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2
POOL_CUTOFF = 6
TREASURY_FEE = 0.03
MIN_BET = 0.001
COOLDOWN_ROUNDS = 72
DRAWDOWN_PEAK_WINDOW_DAYS = 7
WR_WINDOW = 50

GATE_REFERENCE_PNL = 44.8706
GATE_TOLERANCE = 1.0
INITIAL_BANKROLL = 5.0

COHORT_DEFS = [
    ("extension", 422298, 437561),
    ("cv5", 437562, 474086),
    ("gap_post_cv5_pre_holdout", 474087, 474879),
    ("holdout", 474880, 475311),
    ("ext_v2", 475312, 479952),
    ("fresh_oos", 479953, 483191),
    ("post_fresh", 483192, 999999),
]
COHORT_ORDER = [c[0] for c in COHORT_DEFS]


def cohort_of(epoch: int) -> str:
    for name, lo, hi in COHORT_DEFS:
        if lo <= epoch <= hi:
            return name
    return "unknown"


def empty_cohort_record() -> dict[str, Any]:
    return {c: {
        "n_rounds": 0, "n_bets": 0, "n_wins": 0, "pnl_bnb": 0.0,
        "n_skipped_breaker": 0, "n_skipped_cooldown": 0, "n_skipped_other": 0,
        "n_shadow_rounds": 0, "shadow_pnl_bnb": 0.0,
        "n_wr_size_skip": 0, "n_ramp_skip": 0,
    } for c in COHORT_ORDER}


# ---- Config dataclass for an idea variant ----

@dataclass
class RiskConfig:
    label: str
    # Idea #1 (graduated re-entry): None or callable(cooldown_remaining, cooldown_total) -> float in [0,1]
    ramp_fn: Callable[[int, int], float] | None = None
    # Idea #3 (shadow): None or shadow config
    shadow_mode: str | None = None  # "exit_10pos" / "exit_20pos" / "extend"
    # Idea #4 (velocity): None or velocity config
    velocity_pct: float | None = None  # e.g. 5.0 for 5%
    velocity_rounds: int | None = None  # e.g. 10
    velocity_combined_with_abs: bool = False  # OR with absolute breaker
    # Static absolute breaker: dd_frac (production default 0.15). None disables.
    abs_dd_frac: float | None = 0.15
    # Idea #5 (WR sizing): None or sizing config
    wr_size_fn: Callable[[float], float] | None = None  # rolling_wr -> multiplier


# ---- Step 14 tracker ----

class Step14Tracker(InMemoryBankrollTracker):
    """Single tracker handling all 4 ideas via config flags. Production-faithful
    rolling-7d peak via parent. Pre-decision drawdown check via is_paused override
    (Step 13c v2 pattern with +1 cooldown compensation).
    """

    def __init__(self, *, initial_bankroll: float, drawdown_peak_window_days: int,
                  peak_mode: str, config: RiskConfig,
                  cooldown_rounds: int = COOLDOWN_ROUNDS):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._cfg = config
        self._cd_total = int(cooldown_rounds)
        self._cd_initial_at_fire: int = 0  # cooldown remaining when last fired
        self._initial = float(initial_bankroll)
        # Shadow-bet tracking
        self._shadow_pnl_deque: deque[float] = deque(maxlen=30)
        self._n_shadow_rounds = 0
        self._n_early_exits = 0
        self._n_extensions = 0
        self._cooldown_extensions_used = 0  # max 2 per fire
        # Velocity tracking: list of (start_at, bankroll). Maintained alongside parent.
        self._velocity_entries: deque[tuple[int, float]] = deque()
        # WR sizing: last 50 actual bet outcomes (1=win, 0=loss)
        self._wr_outcomes: deque[int] = deque(maxlen=WR_WINDOW)
        self._n_warmup_bets = 0
        # Stats
        self.n_pauses_fired = 0
        self.n_velocity_fires = 0
        self.n_abs_fires = 0

    # ----- Velocity check helper -----
    def _velocity_drop_breached(self, as_of_start_at: int) -> bool:
        if self._cfg.velocity_pct is None or self._cfg.velocity_rounds is None:
            return False
        window_s = self._cfg.velocity_rounds * 300  # 300s per round
        cutoff_low = as_of_start_at - window_s
        # Find oldest entry within window
        oldest = None
        for ts, br in self._velocity_entries:
            if ts >= cutoff_low:
                oldest = (ts, br)
                break
        if oldest is None:
            return False
        current = self.current_bankroll()
        if oldest[1] <= 0:
            return False
        drop_pct = (oldest[1] - current) / oldest[1] * 100.0
        return drop_pct >= self._cfg.velocity_pct

    # ----- Absolute drawdown check helper -----
    def _abs_dd_breached(self, as_of_start_at: int) -> bool:
        if self._cfg.abs_dd_frac is None:
            return False
        current = self.current_bankroll()
        peak = self.peak_bankroll(as_of_start_at)
        if peak <= 0:
            return False
        dd = (peak - current) / peak
        return dd >= self._cfg.abs_dd_frac

    def is_paused(self, as_of_start_at: int) -> bool:
        # Phase 1: existing cooldown still ticking?
        cd_remain = self._cooldown
        if cd_remain > 0:
            # In cooldown phase
            # Check if shadow mode allows pipeline-through
            if self._cfg.shadow_mode is not None:
                # Shadow mode: don't pause the pipeline; runner handles it
                return False
            # Idea #1 ramp: pure-pause only during initial 0× phase
            if self._cfg.ramp_fn is not None:
                ramp = self._cfg.ramp_fn(cd_remain, self._cd_total)
                if ramp <= 0.0:
                    return True  # still in pure-pause phase
                return False  # in ramp-scaled phase, pipeline runs
            # Default: full pause
            return True

        # Phase 2: not in cooldown — check fresh trigger
        velocity_breached = self._velocity_drop_breached(as_of_start_at)
        abs_breached = self._abs_dd_breached(as_of_start_at)

        # Idea #4 with combined OR: either fires
        # Idea #4 velocity-only: only velocity fires
        # Default (no velocity): only abs fires
        should_fire = False
        if self._cfg.velocity_pct is not None:
            if velocity_breached:
                should_fire = True
                self.n_velocity_fires += 1
            if self._cfg.velocity_combined_with_abs and abs_breached:
                should_fire = True
                self.n_abs_fires += 1
        else:
            # No velocity check; standard abs check
            if abs_breached:
                should_fire = True
                self.n_abs_fires += 1

        if should_fire:
            # +1 cooldown compensation for pipeline's tick_cooldown after is_paused True
            self.set_paused(self._cd_total + 1, as_of_start_at)
            self._cd_initial_at_fire = self._cd_total + 1
            self.n_pauses_fired += 1
            self._cooldown_extensions_used = 0
            # If shadow mode, also reset shadow tracking
            if self._cfg.shadow_mode is not None:
                self._shadow_pnl_deque.clear()
                # Re-evaluate: shadow allows pipeline through, so return False
                return False
            # Idea #1 ramp: if pure-pause phase, return True
            if self._cfg.ramp_fn is not None:
                ramp = self._cfg.ramp_fn(self._cooldown, self._cd_total)
                return ramp <= 0.0
            return True
        return False

    # ----- For runner queries -----
    def is_in_cooldown(self) -> bool:
        return self._cooldown > 0

    def cooldown_progress(self) -> tuple[int, int]:
        """Return (cooldown_remaining, cooldown_total_at_fire) for ramp calc."""
        return (self._cooldown, max(1, self._cd_initial_at_fire))

    def current_ramp_factor(self) -> float:
        if self._cfg.ramp_fn is None:
            return 1.0
        cd_remain = self._cooldown
        if cd_remain <= 0:
            return 1.0
        return self._cfg.ramp_fn(cd_remain, self._cd_total)

    def wr_size_multiplier(self) -> float:
        if self._cfg.wr_size_fn is None:
            return 1.0
        if len(self._wr_outcomes) < WR_WINDOW:
            # Warmup: full size
            return 1.0
        wr = sum(self._wr_outcomes) / len(self._wr_outcomes)
        return self._cfg.wr_size_fn(wr)

    def is_in_warmup(self) -> bool:
        return self._cfg.wr_size_fn is not None and len(self._wr_outcomes) < WR_WINDOW

    def is_shadow_active(self) -> bool:
        return self._cfg.shadow_mode is not None and self._cooldown > 0

    def record_shadow_outcome(self, pnl: float) -> None:
        self._shadow_pnl_deque.append(pnl)
        self._n_shadow_rounds += 1

    def should_exit_cooldown_early(self) -> bool:
        mode = self._cfg.shadow_mode
        if mode is None:
            return False
        if mode == "exit_10pos" and len(self._shadow_pnl_deque) >= 10:
            return sum(list(self._shadow_pnl_deque)[-10:]) > 0
        if mode == "exit_20pos" and len(self._shadow_pnl_deque) >= 20:
            return sum(list(self._shadow_pnl_deque)[-20:]) > 0
        return False

    def should_extend_cooldown(self) -> bool:
        if self._cfg.shadow_mode != "extend":
            return False
        # At end of cooldown (cd_remaining about to hit 0)
        # If last 30 shadow rounds net negative AND extensions < 2
        if self._cooldown != 1:
            return False
        if self._cooldown_extensions_used >= 2:
            return False
        if len(self._shadow_pnl_deque) < 30:
            return False
        return sum(list(self._shadow_pnl_deque)[-30:]) < 0

    def force_clear_cooldown(self) -> None:
        self._cooldown = 0
        self._n_early_exits += 1

    def extend_cooldown(self) -> None:
        self._cooldown = self._cd_total
        self._cooldown_extensions_used += 1
        self._n_extensions += 1

    def record_actual_bet(self, won: bool) -> None:
        if self._cfg.wr_size_fn is not None:
            if len(self._wr_outcomes) < WR_WINDOW:
                self._n_warmup_bets += 1
            self._wr_outcomes.append(1 if won else 0)

    def record_settlement(self, bankroll: float, start_at: int) -> None:
        super().record_settlement(bankroll, start_at)
        # Maintain velocity entries deque
        self._velocity_entries.append((int(start_at), float(bankroll)))
        # Prune entries older than 60 minutes (worst velocity window is 30 rounds = 150 min)
        # Keep 3 hours back to be safe
        prune_cutoff = int(start_at) - 3 * 3600
        while self._velocity_entries and self._velocity_entries[0][0] < prune_cutoff:
            self._velocity_entries.popleft()


# ---- Curve / sizing function library ----

def ramp_linear(cd_remain: int, cd_total: int) -> float:
    elapsed = cd_total - cd_remain
    return max(0.0, min(1.0, elapsed / cd_total))

def ramp_step_4(cd_remain: int, cd_total: int) -> float:
    elapsed = cd_total - cd_remain
    # 18/18/18/18 rounds at 0/0.25/0.5/1.0
    if elapsed < 18: return 0.0
    if elapsed < 36: return 0.25
    if elapsed < 54: return 0.5
    return 1.0

def ramp_fast(cd_remain: int, cd_total: int) -> float:
    elapsed = cd_total - cd_remain
    if elapsed < 12: return 0.0
    if elapsed < 36: return 0.5
    return 1.0


def wr_size_50(rolling_wr: float) -> float:
    return max(0.0, 2.0 * (rolling_wr - 0.50))

def wr_size_55(rolling_wr: float) -> float:
    return max(0.0, 4.0 * (rolling_wr - 0.55))


# ---- Configs to test ----

CONFIGS = [
    # Reference (gate)
    RiskConfig(label="GATE_static_dd15", abs_dd_frac=0.15),

    # #1 Graduated re-entry
    RiskConfig(label="1_ramp_linear", abs_dd_frac=0.15, ramp_fn=ramp_linear),
    RiskConfig(label="1_ramp_step_4", abs_dd_frac=0.15, ramp_fn=ramp_step_4),
    RiskConfig(label="1_ramp_fast", abs_dd_frac=0.15, ramp_fn=ramp_fast),

    # #3 Shadow betting
    RiskConfig(label="3_shadow_exit_10pos", abs_dd_frac=0.15, shadow_mode="exit_10pos"),
    RiskConfig(label="3_shadow_exit_20pos", abs_dd_frac=0.15, shadow_mode="exit_20pos"),
    RiskConfig(label="3_shadow_extend", abs_dd_frac=0.15, shadow_mode="extend"),

    # #4 Velocity breaker
    RiskConfig(label="4_velocity_5pct_10rd", abs_dd_frac=None,
                velocity_pct=5.0, velocity_rounds=10),
    RiskConfig(label="4_velocity_10pct_30rd", abs_dd_frac=None,
                velocity_pct=10.0, velocity_rounds=30),
    RiskConfig(label="4_velocity_OR_abs", abs_dd_frac=0.15,
                velocity_pct=5.0, velocity_rounds=10,
                velocity_combined_with_abs=True),

    # #5 WR sizing
    RiskConfig(label="5_wr_size_50", abs_dd_frac=None, wr_size_fn=wr_size_50),
    RiskConfig(label="5_wr_size_55", abs_dd_frac=None, wr_size_fn=wr_size_55),
    RiskConfig(label="5_wr_size_50_with_breaker", abs_dd_frac=0.15, wr_size_fn=wr_size_50),
]


# ---- Backtest runner ----

def run_step14_backtest(*, config: RiskConfig, all_rounds, btc_klines,
                         eth_klines, sol_klines, earliest_offset: int) -> dict[str, Any]:
    overrides = {
        "gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)},
        # Leave production dd_frac at 0.15; tracker handles adaptive logic
    }
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    tracker = Step14Tracker(
        initial_bankroll=INITIAL_BANKROLL,
        drawdown_peak_window_days=DRAWDOWN_PEAK_WINDOW_DAYS,
        peak_mode="rolling_7d",
        config=config,
        cooldown_rounds=COOLDOWN_ROUNDS,
    )
    # If config disables pipeline-side abs breaker (idea #4 velocity-only / #5 wr-only),
    # set strategy's max_dd to 1.0 to bypass pipeline check. The tracker still fires
    # for velocity if configured.
    if config.abs_dd_frac is None:
        overrides2 = dict(overrides)
        overrides2["risk"] = {"max_drawdown_fraction_from_peak": 1.0}
        sc = load_strategy_config_from_dict(overrides2)

    pipeline = MomentumOnlyPipeline(
        config=gate_cfg, strategy_config=sc, gate=None,
        kline_cutoff_seconds=CANONICAL_CUTOFF, pool_cutoff_seconds=POOL_CUTOFF,
        min_bet_amount_bnb=MIN_BET, treasury_fee_fraction=TREASURY_FEE,
        bankroll_tracker=tracker,
    )
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_klines)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_klines)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_klines)
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    sim_rounds = [r for r in all_rounds if EPOCH_MIN <= r.epoch <= EPOCH_MAX]
    per_cohort = empty_cohort_record()
    bankroll = float(INITIAL_BANKROLL); peak = bankroll; max_dd_frac = 0.0
    bet_sizes: list[float] = []

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        coh = cohort_of(ep)
        per_cohort[coh]["n_rounds"] += 1

        decision = pipeline.decide_open_round(round_t=round_t)

        # If pipeline says SKIP (gate_no_signal, pool_below_min, etc.), respect it
        if decision.action != "BET":
            sr = decision.skip_reason or ""
            if sr == "risk_drawdown_breaker_fired":
                per_cohort[coh]["n_skipped_breaker"] += 1
            elif sr == "risk_cooldown_active":
                per_cohort[coh]["n_skipped_cooldown"] += 1
            else:
                per_cohort[coh]["n_skipped_other"] += 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Pipeline returned BET. Check tracker for special states.
        # If shadow mode active (cooldown going + shadow enabled): compute shadow PnL
        if tracker.is_shadow_active():
            # Compute what bet outcome WOULD have been
            bet_size_shadow = float(decision.bet_size_bnb)
            outcome_shadow = settle_bet_against_closed_round(
                bet_bnb=bet_size_shadow, bet_side=str(decision.bet_side),
                round_closed=round_t, treasury_fee_fraction=TREASURY_FEE,
            )
            shadow_profit = outcome_shadow.credit_bnb - bet_size_shadow - MAX_GAS_COST_BET_BNB
            tracker.record_shadow_outcome(shadow_profit)
            per_cohort[coh]["n_shadow_rounds"] += 1
            per_cohort[coh]["shadow_pnl_bnb"] += shadow_profit
            # Check early exit
            if tracker.should_exit_cooldown_early():
                tracker.force_clear_cooldown()
            # Check extension trigger (only when cooldown about to expire)
            elif tracker.should_extend_cooldown():
                tracker.extend_cooldown()
            # Don't actually settle; tick cooldown manually since pipeline didn't
            tracker.tick_cooldown()
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Apply ramp factor (idea #1)
        ramp = tracker.current_ramp_factor()
        # Apply WR sizing multiplier (idea #5)
        wr_mult = tracker.wr_size_multiplier()
        bet_size_canonical = float(decision.bet_size_bnb)
        bet_size_adj = bet_size_canonical * ramp * wr_mult

        # If multipliers zero out the bet, skip
        if bet_size_adj <= 0.0:
            if ramp <= 0.0:
                per_cohort[coh]["n_ramp_skip"] += 1
            elif wr_mult <= 0.0:
                per_cohort[coh]["n_wr_size_skip"] += 1
            else:
                per_cohort[coh]["n_skipped_other"] += 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Apply min_bet clamp; if below, skip
        if bet_size_adj < MIN_BET:
            per_cohort[coh]["n_skipped_other"] += 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Settle bet
        side = str(decision.bet_side)
        bankroll -= bet_size_adj + MAX_GAS_COST_BET_BNB
        outcome = settle_bet_against_closed_round(
            bet_bnb=bet_size_adj, bet_side=side, round_closed=round_t,
            treasury_fee_fraction=TREASURY_FEE,
        )
        bankroll += outcome.credit_bnb
        profit = outcome.credit_bnb - bet_size_adj - MAX_GAS_COST_BET_BNB

        per_cohort[coh]["n_bets"] += 1
        per_cohort[coh]["pnl_bnb"] += profit
        if outcome.outcome == "win":
            per_cohort[coh]["n_wins"] += 1
        bet_sizes.append(bet_size_adj)
        won = outcome.outcome == "win"
        tracker.record_actual_bet(won)

        if bankroll > peak: peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd_frac: max_dd_frac = dd

        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])

    # Aggregate
    for cd in per_cohort.values():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0

    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())

    # Bet size distribution
    bet_size_stats: dict[str, float] = {}
    if bet_sizes:
        bet_sizes_sorted = sorted(bet_sizes)
        bet_size_stats = {
            "min": min(bet_sizes),
            "mean": statistics.mean(bet_sizes),
            "p25": bet_sizes_sorted[len(bet_sizes_sorted) // 4],
            "p50": statistics.median(bet_sizes),
            "p75": bet_sizes_sorted[3 * len(bet_sizes_sorted) // 4],
            "p95": bet_sizes_sorted[min(len(bet_sizes_sorted) - 1, int(len(bet_sizes_sorted) * 0.95))],
            "max": max(bet_sizes),
        }

    return {
        "label": config.label,
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets else 0.0,
            "net_pnl_bnb": bankroll - INITIAL_BANKROLL,
            "final_bankroll": bankroll,
        },
        "max_drawdown_frac": max_dd_frac,
        "n_pauses_fired": tracker.n_pauses_fired,
        "n_velocity_fires": tracker.n_velocity_fires,
        "n_abs_fires": tracker.n_abs_fires,
        "n_shadow_rounds": tracker._n_shadow_rounds,
        "n_early_exits": tracker._n_early_exits,
        "n_extensions": tracker._n_extensions,
        "n_warmup_bets": tracker._n_warmup_bets,
        "bet_size_stats": bet_size_stats,
        "per_cohort": per_cohort,
    }


def main():
    t_all = time.time()
    print("--- loading rounds + klines ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  {len(all_rounds)} rounds", flush=True)

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1

    t_kl = time.time()
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    print(f"  BTC: {len(btc)} in {time.time()-t_kl:.1f}s", flush=True)
    t_kl = time.time()
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    print(f"  ETH: {len(eth)} in {time.time()-t_kl:.1f}s", flush=True)
    t_kl = time.time()
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  SOL: {len(sol)} in {time.time()-t_kl:.1f}s", flush=True)

    btc_klines = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                  for ep, kl in btc.items()}
    eth_klines = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                  for ep, kl in eth.items()}
    sol_klines = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                  for ep, kl in sol.items()}

    # ----- VERIFICATION GATE -----
    print(f"\n=========================================", flush=True)
    print(f"VERIFICATION GATE: static dd=0.15 @ 5 BNB", flush=True)
    print(f"=========================================", flush=True)
    gate_cfg = CONFIGS[0]
    assert gate_cfg.label == "GATE_static_dd15"
    t = time.time()
    gate_r = run_step14_backtest(
        config=gate_cfg,
        all_rounds=all_rounds, btc_klines=btc_klines,
        eth_klines=eth_klines, sol_klines=sol_klines,
        earliest_offset=earliest_offset,
    )
    gate_pnl = gate_r["summary"]["net_pnl_bnb"]
    gate_bets = gate_r["summary"]["num_bets"]
    delta = gate_pnl - GATE_REFERENCE_PNL
    print(f"GATE RESULT: pnl={gate_pnl:+.4f} BNB / {gate_bets} bets / "
          f"fires={gate_r['n_pauses_fired']} ({time.time()-t:.1f}s)", flush=True)
    print(f"GATE DELTA:  {delta:+.4f} vs Step 12b +{GATE_REFERENCE_PNL:.4f}", flush=True)
    gate_pass = abs(delta) <= GATE_TOLERANCE
    print(f"GATE STATUS: {'PASS' if gate_pass else 'FAIL'}", flush=True)

    results: list[dict[str, Any]] = [gate_r]
    gate_r["elapsed_seconds"] = time.time() - t

    if not gate_pass:
        print(f"\n  HALTING: gate failed.", flush=True)
        out_path = REPO / "var" / "strategy_review" / "creative_risk_step14_data.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "gate_pnl": gate_pnl, "gate_delta_vs_step12b": delta,
                "gate_pass": False, "halted": True,
                "results": results,
                "elapsed_seconds": time.time() - t_all,
            }, f, indent=2, default=float)
        print(f"wrote {out_path}", flush=True)
        return

    # ----- Run all 12 variants -----
    print(f"\n--- Running 12 variant configs ---", flush=True)
    for cfg in CONFIGS[1:]:  # Skip gate
        t = time.time()
        r = run_step14_backtest(
            config=cfg,
            all_rounds=all_rounds, btc_klines=btc_klines,
            eth_klines=eth_klines, sol_klines=sol_klines,
            earliest_offset=earliest_offset,
        )
        r["elapsed_seconds"] = time.time() - t
        results.append(r)
        s = r["summary"]
        extras = ""
        if cfg.shadow_mode is not None:
            extras = f" shadow_rounds={r['n_shadow_rounds']} early_exits={r['n_early_exits']} extensions={r['n_extensions']}"
        if cfg.wr_size_fn is not None:
            extras = f" warmup_bets={r['n_warmup_bets']}"
        print(f"  {cfg.label:38s}: pnl={s['net_pnl_bnb']:+8.4f} bets={s['num_bets']:>4d} "
              f"fires={r['n_pauses_fired']}{extras} ({r['elapsed_seconds']:.1f}s)",
              flush=True)

    # ----- Stretch goal #8: Combined best -----
    # Find best per idea
    best_per_idea = {}
    for r in results[1:]:
        label = r["label"]
        idea = label.split("_")[0]
        if idea not in best_per_idea or r["summary"]["net_pnl_bnb"] > best_per_idea[idea]["summary"]["net_pnl_bnb"]:
            best_per_idea[idea] = r

    print(f"\n--- Best per idea: ---", flush=True)
    for idea, r in sorted(best_per_idea.items()):
        print(f"  idea {idea}: {r['label']} pnl={r['summary']['net_pnl_bnb']:+.4f}", flush=True)

    # Build combined config #8:
    # - From best of #1 or #3: take cooldown-mode (mutually exclusive)
    # - From #4: take velocity setting
    # - From #5: take WR sizing
    print(f"\n--- Stretch goal #8: combined best ---", flush=True)
    best_13 = best_per_idea.get("1") or best_per_idea.get("3")
    best_4 = best_per_idea.get("4")
    best_5 = best_per_idea.get("5")

    # Find the configs by label
    def find_cfg(label: str) -> RiskConfig | None:
        for c in CONFIGS:
            if c.label == label:
                return c
        return None

    # Build combined
    c13 = find_cfg(best_13["label"]) if best_13 else None
    c4 = find_cfg(best_4["label"]) if best_4 else None
    c5 = find_cfg(best_5["label"]) if best_5 else None

    combined_cfg = RiskConfig(
        label="8_combined_best",
        abs_dd_frac=0.15,  # default; may be overridden below
        ramp_fn=c13.ramp_fn if c13 else None,
        shadow_mode=c13.shadow_mode if c13 else None,
        velocity_pct=c4.velocity_pct if c4 else None,
        velocity_rounds=c4.velocity_rounds if c4 else None,
        velocity_combined_with_abs=c4.velocity_combined_with_abs if c4 else False,
        wr_size_fn=c5.wr_size_fn if c5 else None,
    )
    # Reconcile abs_dd_frac: keep 0.15 if any source kept it. Disable if all disabled it.
    if (c4 and c4.abs_dd_frac is None) and (c5 and c5.abs_dd_frac is None):
        combined_cfg.abs_dd_frac = None
    # Don't combine ramp+shadow (mutually exclusive); prefer whichever's best
    if combined_cfg.ramp_fn is not None and combined_cfg.shadow_mode is not None:
        # Prefer the better of #1 vs #3
        b1 = best_per_idea.get("1")
        b3 = best_per_idea.get("3")
        if b1 and b3:
            if b1["summary"]["net_pnl_bnb"] >= b3["summary"]["net_pnl_bnb"]:
                combined_cfg.shadow_mode = None
            else:
                combined_cfg.ramp_fn = None

    print(f"  Combined config: ramp={c13.ramp_fn.__name__ if c13 and c13.ramp_fn else None}, "
          f"shadow={combined_cfg.shadow_mode}, "
          f"velocity={combined_cfg.velocity_pct}/{combined_cfg.velocity_rounds}, "
          f"wr_size={c5.wr_size_fn.__name__ if c5 and c5.wr_size_fn else None}, "
          f"abs_dd={combined_cfg.abs_dd_frac}", flush=True)

    t = time.time()
    r = run_step14_backtest(
        config=combined_cfg,
        all_rounds=all_rounds, btc_klines=btc_klines,
        eth_klines=eth_klines, sol_klines=sol_klines,
        earliest_offset=earliest_offset,
    )
    r["elapsed_seconds"] = time.time() - t
    results.append(r)
    s = r["summary"]
    print(f"  {combined_cfg.label}: pnl={s['net_pnl_bnb']:+8.4f} bets={s['num_bets']} "
          f"fires={r['n_pauses_fired']} ({r['elapsed_seconds']:.1f}s)", flush=True)

    # ----- Final rankings -----
    print(f"\n--- All results ranked by PnL ---", flush=True)
    results_sorted = sorted(results, key=lambda r: -r["summary"]["net_pnl_bnb"])
    for r in results_sorted:
        s = r["summary"]
        print(f"  {r['label']:38s}: pnl={s['net_pnl_bnb']:+8.4f} bets={s['num_bets']:>4d}",
              flush=True)

    out_path = REPO / "var" / "strategy_review" / "creative_risk_step14_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "gate_pnl": gate_pnl, "gate_delta_vs_step12b": delta,
            "gate_pass": gate_pass,
            "config": {
                "initial_bankroll_bnb": INITIAL_BANKROLL,
                "cooldown_rounds": COOLDOWN_ROUNDS,
                "wr_window": WR_WINDOW,
                "cohort_defs": [list(c) for c in COHORT_DEFS],
            },
            "best_per_idea": {idea: r["label"] for idea, r in best_per_idea.items()},
            "combined_config_summary": {
                "ramp_fn": combined_cfg.ramp_fn.__name__ if combined_cfg.ramp_fn else None,
                "shadow_mode": combined_cfg.shadow_mode,
                "velocity_pct": combined_cfg.velocity_pct,
                "velocity_rounds": combined_cfg.velocity_rounds,
                "velocity_combined_with_abs": combined_cfg.velocity_combined_with_abs,
                "wr_size_fn": combined_cfg.wr_size_fn.__name__ if combined_cfg.wr_size_fn else None,
                "abs_dd_frac": combined_cfg.abs_dd_frac,
            },
            "results": results,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
