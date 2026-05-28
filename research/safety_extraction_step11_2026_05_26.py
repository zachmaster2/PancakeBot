"""Step 11 — safety/extraction experiments + most-profitable-window scan.

Three experiments + one bonus query, all at 50 BNB dynamic bankroll across
full 422298..484408 range with canonical (3, 7, 15) cs=2.

A. Sweep max_drawdown_fraction_from_peak ∈ {0.05, 0.08, 0.10, 0.12, 0.15}.
B. Replace fractional drawdown with ABSOLUTE-BNB threshold ∈
   {0.25, 0.50, 1.0, 1.5, 2.0}. Uses a custom BankrollTracker wrapper that
   tracks absolute drawdown internally; the pipeline's fractional check is
   neutered by setting `max_drawdown_fraction_from_peak=1.0` (unreachable).
E. Anti-martingale sizing: bet_size = canonical_bet × multiplier where
   multiplier = min(streak_max, 1 + 0.25 × win_streak). Reset to 1× on loss.
   Sweep streak_max ∈ {1.5, 2.0, 2.5, 3.0}.

Bonus: most profitable contiguous-time window via Kadane's on per-bet
profits from the canonical full backtest. Reported at 5 BNB AND 50 BNB.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
EPOCH_MAX_CONFIG = 484999
CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2
POOL_CUTOFF = 6
TREASURY_FEE = 0.03
MIN_BET = 0.001
INITIAL_BANKROLL = 50.0

# Cohort defs with explicit gap bucket (from Step 10b finding)
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
        "total_bet_size_bnb": 0.0, "max_bet_size_bnb": 0.0,
        "skip_drawdown_breaker": 0, "skip_cooldown": 0,
        "skip_other": 0,
    } for c in COHORT_ORDER}


def parse_trades_to_cohorts(trades_csv: Path) -> tuple[dict[str, dict[str, Any]], float, list[tuple[int, float]]]:
    """Read trades.csv, return per-cohort stats, max drawdown frac, and
    a per-bet (epoch, profit) timeline."""
    per_cohort = empty_cohort_record()
    timeline: list[tuple[int, float]] = []
    peak = INITIAL_BANKROLL
    max_dd_frac = 0.0
    with open(trades_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            coh = cohort_of(epoch)
            per_cohort[coh]["n_rounds"] += 1
            br = float(row["bankroll_bnb"])
            if br > peak:
                peak = br
            if peak > 0:
                dd = (peak - br) / peak
                if dd > max_dd_frac:
                    max_dd_frac = dd
            action = row.get("action")
            if action == "BET":
                profit = float(row["profit_bnb"])
                bet_size = float(row["bet_size_bnb"])
                per_cohort[coh]["n_bets"] += 1
                per_cohort[coh]["pnl_bnb"] += profit
                per_cohort[coh]["total_bet_size_bnb"] += bet_size
                if bet_size > per_cohort[coh]["max_bet_size_bnb"]:
                    per_cohort[coh]["max_bet_size_bnb"] = bet_size
                if profit > 0:
                    per_cohort[coh]["n_wins"] += 1
                timeline.append((epoch, profit))
            else:
                sr = (row.get("skip_reason") or "").strip()
                if sr == "risk_drawdown_breaker_fired":
                    per_cohort[coh]["skip_drawdown_breaker"] += 1
                elif sr == "risk_cooldown_active":
                    per_cohort[coh]["skip_cooldown"] += 1
                else:
                    per_cohort[coh]["skip_other"] += 1
    # Derive WR
    for coh, cd in per_cohort.items():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0
        cd["mean_bet_size_bnb"] = (cd["total_bet_size_bnb"] / cd["n_bets"]
                                    if cd["n_bets"] else 0.0)
    return per_cohort, max_dd_frac, timeline


# ---------------------------------------------------------------------------
# Experiment A: tighter drawdown fraction
# ---------------------------------------------------------------------------

def run_experiment_A(*, all_rounds, btc, eth, sol, earliest_offset, out_root,
                      initial_bankroll: float) -> list[dict[str, Any]]:
    print(f"\n========== Experiment A: max_drawdown_fraction sweep @ {initial_bankroll} BNB ==========")
    fractions = [0.05, 0.08, 0.10, 0.12, 0.15]
    results = []
    for frac in fractions:
        overrides = {
            "gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)},
            "risk": {"max_drawdown_fraction_from_peak": frac},
        }
        sc = load_strategy_config_from_dict(overrides)
        assert sc.risk.max_drawdown_fraction_from_peak == frac
        spec = ipr.FoldSpec(
            name=f"expA_dd{int(frac*100):02d}",
            kline_cutoff_seconds=CANONICAL_CUTOFF,
            epoch_start=EPOCH_MIN, epoch_end=EPOCH_MAX_CONFIG,
            strategy_overrides=overrides,
        )
        t0 = time.time()
        summary = ipr.run_fold(
            spec=spec, strategy_cfg=sc,
            all_rounds=all_rounds, btc_unified=btc, eth_unified=eth, sol_unified=sol,
            earliest_offset=earliest_offset, output_base_dir=out_root,
            initial_bankroll_bnb=initial_bankroll,
            treasury_fee_fraction=TREASURY_FEE, min_bet_amount_bnb=MIN_BET,
        )
        elapsed = time.time() - t0
        trades_csv = out_root / spec.name / "trades.csv"
        per_cohort, max_dd, timeline = parse_trades_to_cohorts(trades_csv)
        results.append({
            "variant_label": f"dd_frac={frac}",
            "max_dd_fraction_param": frac,
            "summary": {k: summary[k] for k in ("num_bets", "num_wins", "win_rate", "net_pnl_bnb") if k in summary},
            "max_drawdown_realized_frac": max_dd,
            "per_cohort": per_cohort,
            "skip_counts": summary.get("skip_counts_by_reason", {}),
            "elapsed_seconds": elapsed,
        })
        print(f"  dd_frac={frac}: bets={summary['num_bets']} WR={summary['win_rate']:.4f} "
              f"pnl={summary['net_pnl_bnb']:+.4f} max_dd={max_dd*100:.2f}% "
              f"({elapsed:.1f}s)")
    return results


# ---------------------------------------------------------------------------
# Experiment B: absolute-BNB drawdown threshold
# ---------------------------------------------------------------------------

class AbsoluteDrawdownTracker:
    """Bankroll tracker that fires drawdown breaker on ABSOLUTE BNB threshold.

    Pipeline's fractional check is neutered by setting `max_drawdown_fraction_from_peak=1.0`
    (unreachable in normal operation). Absolute logic lives here.
    """

    def __init__(self, *, initial_bankroll: float, max_abs_dd_bnb: float,
                 cooldown_rounds: int):
        self._current = float(initial_bankroll)
        self._peak = float(initial_bankroll)
        self._max_abs_dd = float(max_abs_dd_bnb)
        self._cooldown_total = int(cooldown_rounds)
        self._cooldown_remaining = 0
        self._paused = False
        self.n_pauses_fired = 0
        # Diagnostic: track per-epoch event log
        self.events: list[dict[str, Any]] = []

    def current_bankroll(self) -> float:
        return self._current

    def peak_bankroll(self, start_at: int) -> float:  # noqa: ARG002
        return self._peak

    def is_paused(self, start_at: int) -> bool:  # noqa: ARG002
        return self._paused

    def tick_cooldown(self) -> None:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self._paused = False

    def cooldown_remaining(self) -> int:
        return self._cooldown_remaining

    def set_paused(self, rounds: int, start_at: int) -> None:  # noqa: ARG002
        # Pipeline calls this when its fractional check fires, but we
        # neutered that. No-op here; we fire from record_settlement.
        pass

    def record_settlement(self, bankroll: float, start_at: int) -> None:
        self._current = float(bankroll)
        if self._current > self._peak:
            self._peak = self._current
        drawdown_abs = self._peak - self._current
        if not self._paused and drawdown_abs >= self._max_abs_dd:
            self._paused = True
            self._cooldown_remaining = self._cooldown_total
            self.n_pauses_fired += 1


def run_experiment_B_one(*, max_abs_dd: float, all_rounds, btc_klines, eth_klines,
                           sol_klines, earliest_offset, initial_bankroll: float,
                           sc, gate_cfg) -> dict[str, Any]:
    tracker = AbsoluteDrawdownTracker(
        initial_bankroll=initial_bankroll,
        max_abs_dd_bnb=max_abs_dd,
        cooldown_rounds=sc.risk.cooldown_rounds,
    )
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

    sim_rounds = [r for r in all_rounds if EPOCH_MIN <= r.epoch <= EPOCH_MAX_CONFIG]
    per_cohort = empty_cohort_record()
    bankroll = float(initial_bankroll)
    peak = bankroll
    max_dd_frac = 0.0
    timeline: list[tuple[int, float]] = []

    for round_t in sim_rounds:
        coh = cohort_of(int(round_t.epoch))
        per_cohort[coh]["n_rounds"] += 1
        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            sr = decision.skip_reason or ""
            if sr == "risk_drawdown_breaker_fired":
                per_cohort[coh]["skip_drawdown_breaker"] += 1
            elif sr == "risk_cooldown_active":
                per_cohort[coh]["skip_cooldown"] += 1
            else:
                per_cohort[coh]["skip_other"] += 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        bet_size = float(decision.bet_size_bnb)
        side = str(decision.bet_side)
        bankroll -= bet_size + MAX_GAS_COST_BET_BNB
        outcome = settle_bet_against_closed_round(
            bet_bnb=bet_size, bet_side=side, round_closed=round_t,
            treasury_fee_fraction=TREASURY_FEE,
        )
        bankroll += outcome.credit_bnb
        profit = outcome.credit_bnb - bet_size - MAX_GAS_COST_BET_BNB

        per_cohort[coh]["n_bets"] += 1
        per_cohort[coh]["pnl_bnb"] += profit
        per_cohort[coh]["total_bet_size_bnb"] += bet_size
        if bet_size > per_cohort[coh]["max_bet_size_bnb"]:
            per_cohort[coh]["max_bet_size_bnb"] = bet_size
        if outcome.outcome == "win":
            per_cohort[coh]["n_wins"] += 1
        timeline.append((int(round_t.epoch), profit))

        if bankroll > peak:
            peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd_frac:
                max_dd_frac = dd

        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])

    for coh, cd in per_cohort.items():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0
        cd["mean_bet_size_bnb"] = (cd["total_bet_size_bnb"] / cd["n_bets"]
                                    if cd["n_bets"] else 0.0)

    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())
    total_pnl = bankroll - initial_bankroll
    return {
        "variant_label": f"abs_dd={max_abs_dd}",
        "max_abs_dd_param": max_abs_dd,
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets else 0.0,
            "net_pnl_bnb": total_pnl, "final_bankroll": bankroll,
        },
        "max_drawdown_realized_frac": max_dd_frac,
        "n_pauses_fired": tracker.n_pauses_fired,
        "per_cohort": per_cohort,
    }


def run_experiment_B(*, all_rounds, btc, eth, sol, earliest_offset,
                      initial_bankroll: float) -> list[dict[str, Any]]:
    print(f"\n========== Experiment B: absolute-BNB drawdown sweep @ {initial_bankroll} BNB ==========")
    # Set max_dd_fraction_from_peak=1.0 to neuter the pipeline's fractional check
    overrides = {
        "gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)},
        "risk": {"max_drawdown_fraction_from_peak": 1.0},
    }
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    max_lookback = max(CANONICAL_LOOKBACKS)
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

    thresholds = [0.25, 0.50, 1.0, 1.5, 2.0]
    results = []
    for thr in thresholds:
        t0 = time.time()
        r = run_experiment_B_one(
            max_abs_dd=thr, all_rounds=all_rounds,
            btc_klines=btc_klines, eth_klines=eth_klines, sol_klines=sol_klines,
            earliest_offset=earliest_offset,
            initial_bankroll=initial_bankroll, sc=sc, gate_cfg=gate_cfg,
        )
        r["elapsed_seconds"] = time.time() - t0
        results.append(r)
        s = r["summary"]
        print(f"  abs_dd={thr}: bets={s['num_bets']} WR={s['win_rate']:.4f} "
              f"pnl={s['net_pnl_bnb']:+.4f} max_dd={r['max_drawdown_realized_frac']*100:.2f}% "
              f"pauses={r['n_pauses_fired']} ({r['elapsed_seconds']:.1f}s)")
    return results


# ---------------------------------------------------------------------------
# Experiment E: anti-martingale sizing
# ---------------------------------------------------------------------------

def run_experiment_E_one(*, streak_max: float, all_rounds, btc_klines, eth_klines,
                          sol_klines, earliest_offset, initial_bankroll: float,
                          sc, gate_cfg) -> dict[str, Any]:
    tracker = InMemoryBankrollTracker(
        initial_bankroll=initial_bankroll,
        drawdown_peak_window_days=sc.risk.drawdown_peak_window_days,
        peak_mode=sc.risk.drawdown_peak_mode,
    )
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

    sim_rounds = [r for r in all_rounds if EPOCH_MIN <= r.epoch <= EPOCH_MAX_CONFIG]
    per_cohort = empty_cohort_record()
    bankroll = float(initial_bankroll)
    peak = bankroll
    max_dd_frac = 0.0
    win_streak = 0
    max_streak_observed = 0

    for round_t in sim_rounds:
        coh = cohort_of(int(round_t.epoch))
        per_cohort[coh]["n_rounds"] += 1
        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            sr = decision.skip_reason or ""
            if sr == "risk_drawdown_breaker_fired":
                per_cohort[coh]["skip_drawdown_breaker"] += 1
            elif sr == "risk_cooldown_active":
                per_cohort[coh]["skip_cooldown"] += 1
            else:
                per_cohort[coh]["skip_other"] += 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        canonical_bet = float(decision.bet_size_bnb)
        side = str(decision.bet_side)
        multiplier = min(streak_max, 1.0 + 0.25 * win_streak)
        bet_size = canonical_bet * multiplier
        # Bankroll-safety clamp
        safe_max = max(0.0, bankroll - MAX_GAS_COST_BET_BNB - 0.01)
        bet_size = min(bet_size, safe_max)
        bet_size = max(bet_size, MIN_BET)
        if bankroll < bet_size + MAX_GAS_COST_BET_BNB:
            per_cohort[coh]["skip_other"] += 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        bankroll -= bet_size + MAX_GAS_COST_BET_BNB
        outcome = settle_bet_against_closed_round(
            bet_bnb=bet_size, bet_side=side, round_closed=round_t,
            treasury_fee_fraction=TREASURY_FEE,
        )
        bankroll += outcome.credit_bnb
        profit = outcome.credit_bnb - bet_size - MAX_GAS_COST_BET_BNB

        per_cohort[coh]["n_bets"] += 1
        per_cohort[coh]["pnl_bnb"] += profit
        per_cohort[coh]["total_bet_size_bnb"] += bet_size
        if bet_size > per_cohort[coh]["max_bet_size_bnb"]:
            per_cohort[coh]["max_bet_size_bnb"] = bet_size
        if outcome.outcome == "win":
            per_cohort[coh]["n_wins"] += 1
            win_streak += 1
            if win_streak > max_streak_observed:
                max_streak_observed = win_streak
        else:
            win_streak = 0

        if bankroll > peak:
            peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd_frac:
                max_dd_frac = dd

        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])

    for coh, cd in per_cohort.items():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0
        cd["mean_bet_size_bnb"] = (cd["total_bet_size_bnb"] / cd["n_bets"]
                                    if cd["n_bets"] else 0.0)

    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())
    return {
        "variant_label": f"streak_max={streak_max}",
        "streak_max_param": streak_max,
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets else 0.0,
            "net_pnl_bnb": bankroll - initial_bankroll,
            "final_bankroll": bankroll,
        },
        "max_drawdown_realized_frac": max_dd_frac,
        "max_streak_observed": max_streak_observed,
        "per_cohort": per_cohort,
    }


def run_experiment_E(*, all_rounds, btc, eth, sol, earliest_offset,
                      initial_bankroll: float) -> list[dict[str, Any]]:
    print(f"\n========== Experiment E: anti-martingale sweep @ {initial_bankroll} BNB ==========")
    overrides = {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    max_lookback = max(CANONICAL_LOOKBACKS)
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

    streak_maxes = [1.5, 2.0, 2.5, 3.0]
    results = []
    for sm in streak_maxes:
        t0 = time.time()
        r = run_experiment_E_one(
            streak_max=sm, all_rounds=all_rounds,
            btc_klines=btc_klines, eth_klines=eth_klines, sol_klines=sol_klines,
            earliest_offset=earliest_offset,
            initial_bankroll=initial_bankroll, sc=sc, gate_cfg=gate_cfg,
        )
        r["elapsed_seconds"] = time.time() - t0
        results.append(r)
        s = r["summary"]
        print(f"  streak_max={sm}: bets={s['num_bets']} WR={s['win_rate']:.4f} "
              f"pnl={s['net_pnl_bnb']:+.4f} max_dd={r['max_drawdown_realized_frac']*100:.2f}% "
              f"max_streak_obs={r['max_streak_observed']} ({r['elapsed_seconds']:.1f}s)")
    return results


# ---------------------------------------------------------------------------
# Bonus: most profitable contiguous window
# ---------------------------------------------------------------------------

def kadane_max_window(timeline: list[tuple[int, float]]) -> dict[str, Any]:
    """Find the contiguous subsequence of bets with max profit sum.
    Returns start/end epoch + PnL + n_bets.
    """
    if not timeline:
        return {"start_epoch": None, "end_epoch": None, "pnl_bnb": 0.0, "n_bets": 0}
    max_sum = float("-inf")
    cur_sum = 0.0
    start_i = 0
    best_start = 0
    best_end = 0
    cur_start = 0
    for i, (_, profit) in enumerate(timeline):
        if cur_sum + profit < profit:
            cur_sum = profit
            cur_start = i
        else:
            cur_sum += profit
        if cur_sum > max_sum:
            max_sum = cur_sum
            best_start = cur_start
            best_end = i
    bet_window = timeline[best_start:best_end + 1]
    n_bets = len(bet_window)
    n_wins = sum(1 for _, p in bet_window if p > 0)
    return {
        "start_epoch": int(bet_window[0][0]),
        "end_epoch": int(bet_window[-1][0]),
        "pnl_bnb": float(max_sum),
        "n_bets": n_bets,
        "n_wins": n_wins,
        "win_rate": n_wins / n_bets if n_bets else 0.0,
    }


def epoch_to_iso(epoch: int, all_rounds: list) -> str:
    """Map epoch -> ISO date string using closed round's start_at."""
    for r in all_rounds:
        if int(r.epoch) == int(epoch):
            return datetime.fromtimestamp(int(r.start_at), tz=timezone.utc).isoformat()
    return "unknown"


def run_bonus_window_scan(*, all_rounds, btc, eth, sol, earliest_offset,
                           out_root) -> dict[str, Any]:
    print(f"\n========== Bonus: most-profitable-window scan ==========")
    out: dict[str, Any] = {}
    for initial_bankroll in (5.0, 50.0):
        overrides = {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
        sc = load_strategy_config_from_dict(overrides)
        spec = ipr.FoldSpec(
            name=f"bonus_{int(initial_bankroll)}bnb",
            kline_cutoff_seconds=CANONICAL_CUTOFF,
            epoch_start=EPOCH_MIN, epoch_end=EPOCH_MAX_CONFIG,
            strategy_overrides=overrides,
        )
        t0 = time.time()
        summary = ipr.run_fold(
            spec=spec, strategy_cfg=sc,
            all_rounds=all_rounds, btc_unified=btc, eth_unified=eth, sol_unified=sol,
            earliest_offset=earliest_offset, output_base_dir=out_root,
            initial_bankroll_bnb=initial_bankroll,
            treasury_fee_fraction=TREASURY_FEE, min_bet_amount_bnb=MIN_BET,
        )
        trades_csv = out_root / spec.name / "trades.csv"
        _, _, timeline = parse_trades_to_cohorts(trades_csv)
        window = kadane_max_window(timeline)
        if window["start_epoch"] is not None:
            start_iso = epoch_to_iso(window["start_epoch"], all_rounds)
            end_iso = epoch_to_iso(window["end_epoch"], all_rounds)
            window["start_iso"] = start_iso
            window["end_iso"] = end_iso
            try:
                start_dt = datetime.fromisoformat(start_iso)
                end_dt = datetime.fromisoformat(end_iso)
                duration_seconds = (end_dt - start_dt).total_seconds()
                window["duration_days"] = duration_seconds / 86400.0
            except Exception:
                window["duration_days"] = None
        window["scale"] = initial_bankroll
        window["full_range_pnl"] = summary["net_pnl_bnb"]
        out[f"{int(initial_bankroll)}bnb"] = window
        print(f"  @{initial_bankroll} BNB: "
              f"best window [{window['start_epoch']}..{window['end_epoch']}] "
              f"= {window.get('duration_days', 0):.1f}d  "
              f"bets={window['n_bets']} WR={window.get('win_rate', 0):.4f} "
              f"PnL={window['pnl_bnb']:+.4f} BNB "
              f"(vs full range {summary['net_pnl_bnb']:+.4f}) "
              f"[{time.time()-t0:.1f}s]")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_all = time.time()

    print("--- loading rounds (canonical + extended) ---")
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  loaded {len(all_rounds)} rounds; range "
          f"[{all_rounds[0].epoch}..{max(r.epoch for r in all_rounds)}]")

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1

    print("--- loading klines unified ---")
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  BTC={len(btc)} ETH={len(eth)} SOL={len(sol)}")

    out_root = Path(tempfile.mkdtemp(prefix="step11_"))

    exp_A = run_experiment_A(all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
                               earliest_offset=earliest_offset, out_root=out_root,
                               initial_bankroll=INITIAL_BANKROLL)
    exp_B = run_experiment_B(all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
                               earliest_offset=earliest_offset,
                               initial_bankroll=INITIAL_BANKROLL)
    exp_E = run_experiment_E(all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
                               earliest_offset=earliest_offset,
                               initial_bankroll=INITIAL_BANKROLL)
    bonus = run_bonus_window_scan(all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
                                    earliest_offset=earliest_offset, out_root=out_root)

    # Persist
    out_path = REPO / "var" / "strategy_review" / "safety_extraction_step11_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX_CONFIG,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "initial_bankroll_bnb": INITIAL_BANKROLL,
                "cohort_defs": [list(c) for c in COHORT_DEFS],
            },
            "experiment_A_drawdown_fraction_sweep": exp_A,
            "experiment_B_absolute_drawdown_sweep": exp_B,
            "experiment_E_anti_martingale_sweep": exp_E,
            "bonus_most_profitable_window": bonus,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
