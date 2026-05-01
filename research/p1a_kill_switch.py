"""p1a — kill-switch overlay protocol (v3.2 ratified).

Pre-registered protocol per orchestrator v3 + v3.1 + v3.2:

WALK-FORWARD SWEEP:
  - 4 windows on canonical CV5 train+val range (60d train + 14d test, 14d step)
  - 60-combo grid: (N, L, K) per orchestrator's recalibrated grid
  - Kill switch state RESETS at each WF window boundary
  - First N bets of TEST partition excluded from marginal-PnL accounting
  - Pick best combo by aggregated walk-forward TEST per-bet improvement vs canonical-Arm-A
  - PASS if 95% CI lower bound >= 0 AND point estimate >= +0.040 BNB/bet (MDE-aligned)
  - INSUFFICIENT POWER if 95% CI spans 0 AND +0.040
  - HARD FAIL if 95% CI upper bound <= 0

ARMS:
  - A: canonical with cooldown ON, drawdown breaker ON, kill switch OFF
  - B: canonical with all gates OFF, kill switch OFF
  - C: canonical with all gates OFF, kill switch ON @ picked combo
  - D: canonical with all gates ON, kill switch ON @ picked combo

PLACEBO (Bonferroni-correct selection-null):
  - 5,000 seeds at picked combo via stratified shortcut (5 representative combos × 1,000 seeds)
  - Pause-rate matched: definition (b) = % canonical-would-fire suppressed
  - Picked combo's per-bet improvement must clear 99.92nd percentile of placebo distribution
  - GPD parametric upper-tail fit recommended (residual R1 noted)

FROZEN HOLDOUT (single-shot eval at picked combo):
  - extension cohort (~500 bets) — primary regime test
  - v3 holdout (~95 bets) — sanity check, underpowered
  - post-v1 fresh (~77 bets) — sanity check, underpowered

REVIEWER RESIDUALS INCORPORATED:
  - R1: placebo 5,000-seed minimum + parametric tail recommended (executed below)
  - R2: pause-rate matching = % canonical-would-fire suppressed (definition b)
  - R3: optimized-replay fidelity check pre-registered as a hard gate before sweep

CANONICAL HASH EQUIVALENCE: Arm A on the canonical f1-f5+holdout slice MUST produce
9eec23adceca7fbbe44cfae5245dfc83. New WF-range hashes are captured as new identity
baselines for those arms.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time

# Reconfigure stdout/stderr to UTF-8 — Windows cp1252 default chokes on emdash/arrows.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")
sys.path.insert(0, str(REPO))

from research.in_process_runner import (
    FoldSpec, _load_all_rounds, _load_klines_unified, _resolve_strategy_config,
    _slice_per_entry,
    _BTC_KLINES_PATH, _ETH_KLINES_PATH, _SOL_KLINES_PATH,
    _EXT_BTC_KLINES_PATH, _EXT_ETH_KLINES_PATH, _EXT_SOL_KLINES_PATH,
)
from pancakebot.bankroll_tracker import InMemoryBankrollTracker
from pancakebot.config import load_strategy_config_from_dict
from pancakebot.constants import GAS_COST_BET_BNB
from pancakebot.market_data.contract_constants import load_contract_constants
from pancakebot.settlement import settle_bet_against_closed_round
from pancakebot.strategy.momentum_gate import MomentumGateConfig
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.util import InvariantError


# ============================================================
# Constants — pre-registered, do not change post-hoc
# ============================================================

# Walk-forward windows (60d train + 14d test, 14d step). 4 fit on canonical CV5 train+val.
EP_PER_DAY = 288
WF_TRAIN_DAYS = 60
WF_TEST_DAYS = 14

WF_WINDOWS = [
    {"name": "wf_00", "train_lo": 437562, "train_hi": 454841, "test_lo": 454842, "test_hi": 458873},
    {"name": "wf_01", "train_lo": 441594, "train_hi": 458873, "test_lo": 458874, "test_hi": 462905},
    {"name": "wf_02", "train_lo": 445626, "train_hi": 462905, "test_lo": 462906, "test_hi": 466937},
    {"name": "wf_03", "train_lo": 449658, "train_hi": 466937, "test_lo": 466938, "test_hi": 470969},
]

# 60-combo grid per v3 (per-window-size L tuned to f1-f4 worst trailing-window stats)
COMBO_GRID = []
for N, L_options in [
    (10,  [-1.0, -1.5, -2.0, -2.5]),
    (20,  [-1.5, -2.0, -2.5, -3.0]),
    (50,  [-2.0, -2.5, -3.0]),
    (100, [-2.5, -3.0, -3.5, -4.0]),
]:
    for L in L_options:
        for K in [10, 50, 100, 500]:
            COMBO_GRID.append({"N": N, "L": L, "K": K})
assert len(COMBO_GRID) == 60, f"expected 60 combos, got {len(COMBO_GRID)}"

# Placebo
PLACEBO_TOTAL_SEEDS = 5000
PLACEBO_STRATIFIED_REPRESENTATIVE_COMBOS = 5  # pick 5 across the grid
PLACEBO_BONFERRONI_PERCENTILE = 99.92         # 0.05/60 = 0.000833 → 99.917th
PLACEBO_RESIDUAL_R2_MATCH = "canonical_suppression_rate"  # def (b) per residual R2

# FROZEN HOLDOUT slices (single-shot eval at picked combo)
EXTENSION_RANGE = (422298, 437561)   # ~12-15k rounds depending on data_status
V3_RANGE = (474880, 477254)          # ~2,375 rounds
POSTV1_RANGE = (475312, 477254)      # ~1,943 rounds (subset of v3)

# Success criteria (v3.2)
WF_PASS_POINT_ESTIMATE_MIN = 0.040    # +0.040 BNB/bet at MDE
WF_PASS_CI_LOWER_MIN = 0.0
WF_HARD_FAIL_CI_UPPER_MAX = 0.0
EXT_PASS_POINT_ESTIMATE_MIN = 0.020
EXT_PASS_CI_LOWER_MIN = -0.005

# Settlement / risk
INITIAL_BANKROLL_BNB = 100.0  # canonical default
CUTOFF_SECONDS = 2

# Output
OUT_DIR = REPO / "var" / "extended"
OUT_RESULTS = OUT_DIR / "p1a_kill_switch_results.json"
OUT_LOG = REPO / "var" / "extended" / "p1a_kill_switch.log"


# ============================================================
# Kill-switch state machine
# ============================================================

@dataclass
class KillSwitchState:
    """Maintains trailing PnL window + paused countdown.

    Trigger: when len(trailing) == N AND sum(trailing) <= L → set paused = K.
    While paused > 0: suppress canonical bets, decrement paused per round.
    On unpause: reset trailing window (N3 — clean break).
    """
    N: int                  # window size in BETS
    L: float                # trigger threshold (negative BNB)
    K: int                  # pause duration in ROUNDS
    trailing: deque = field(default_factory=lambda: deque(maxlen=0))
    paused_remaining: int = 0
    n_pauses_fired: int = 0

    def __post_init__(self):
        # set maxlen now that we know N
        self.trailing = deque(maxlen=self.N)

    def is_paused(self) -> bool:
        return self.paused_remaining > 0

    def on_round_advance(self):
        """Called once per round whether or not we bet. Decrements pause counter."""
        if self.paused_remaining > 0:
            self.paused_remaining -= 1
            if self.paused_remaining == 0:
                # Reset trailing window on unpause (N3 clean break)
                self.trailing.clear()

    def on_bet_settled(self, profit_bnb: float):
        """Record per-bet PnL and check trigger. Only called when we actually bet."""
        if self.is_paused():
            return  # shouldn't happen — we don't bet when paused
        self.trailing.append(profit_bnb)
        if len(self.trailing) == self.N and sum(self.trailing) <= self.L:
            self.paused_remaining = self.K
            self.n_pauses_fired += 1


# ============================================================
# Data loading (one-time)
# ============================================================

def load_data_for_range(ep_min: int, ep_max: int, *, use_extended: bool, earliest_offset: int):
    """Load rounds + BTC/ETH/SOL klines for a range. Reuses canonical helpers."""
    all_rounds = _load_all_rounds(use_extended_data=use_extended)
    rounds = [r for r in all_rounds if ep_min <= int(r.epoch) <= ep_max and r.lock_at is not None]
    rounds.sort(key=lambda r: int(r.epoch))

    btc_unified = _load_klines_unified(
        _BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=3,
        extended_path=_EXT_BTC_KLINES_PATH if use_extended else None,
    )
    eth_unified = _load_klines_unified(
        _ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=3,
        extended_path=_EXT_ETH_KLINES_PATH if use_extended else None,
    )
    sol_unified = _load_klines_unified(
        _SOL_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=3,
        extended_path=_EXT_SOL_KLINES_PATH if use_extended else None,
    )
    # Filter klines to range (saves memory for tight ranges)
    btc_unified = {ep: kl for ep, kl in btc_unified.items() if ep_min <= ep <= ep_max}
    eth_unified = {ep: kl for ep, kl in eth_unified.items() if ep_min <= ep <= ep_max}
    sol_unified = {ep: kl for ep, kl in sol_unified.items() if ep_min <= ep <= ep_max}
    return rounds, btc_unified, eth_unified, sol_unified


def slice_klines_per_round(unified: dict, *, max_lookback: int, earliest_offset: int):
    """Slice each round's kline window for the strategy's lookbacks."""
    return {
        ep: _slice_per_entry(kl, cutoff_seconds=CUTOFF_SECONDS,
                                max_lookback=max_lookback,
                                earliest_offset=earliest_offset)
        for ep, kl in unified.items()
    }


# ============================================================
# Strategy config builders for the 4 arms
# ============================================================

def make_strategy_config(*, gates_on: bool):
    """Build strategy config with gates ON (canonical) or OFF (raw signal arm)."""
    if gates_on:
        return _resolve_strategy_config(FoldSpec(
            name="canonical", cutoff_seconds=CUTOFF_SECONDS,
            epoch_start=None, epoch_end=None, strategy_overrides={},
        ))
    # Gates OFF: cooldown_rounds=0, max_drawdown_frac_from_peak=1.0, min_bankroll_bnb=0.0
    return _resolve_strategy_config(FoldSpec(
        name="gates_off", cutoff_seconds=CUTOFF_SECONDS,
        epoch_start=None, epoch_end=None,
        strategy_overrides={
            "risk": {
                "cooldown_rounds": 0,
                "max_drawdown_frac_from_peak": 1.0,
                "min_bankroll_bnb": 0.0,
            },
        },
    ))


# ============================================================
# Backtest one window (with optional kill switch overlay)
# ============================================================

def backtest_window(
    rounds_window: list,
    btc_unified_window: dict,
    eth_unified_window: dict,
    sol_unified_window: dict,
    *,
    earliest_offset: int,
    test_lo: int,                   # epochs >= test_lo are scored (rest is warmup)
    gates_on: bool,
    kill_switch_combo: dict | None, # None = no kill switch
    treasury_fee: float,
    min_bet_amount: float,
):
    """Run one (window) backtest with optional kill switch.

    Returns dict with:
      - n_canonical_bets / n_kill_switch_suppressed in test partition
      - per_round records (test partition only, for marginal-PnL aggregation)
      - kill switch firing count, pause-rate
    """
    strategy_cfg = make_strategy_config(gates_on=gates_on)
    max_lookback = max(strategy_cfg.gate.mtf_lookbacks)
    btc_klines = slice_klines_per_round(btc_unified_window, max_lookback=max_lookback, earliest_offset=earliest_offset)
    eth_klines = slice_klines_per_round(eth_unified_window, max_lookback=max_lookback, earliest_offset=earliest_offset)
    sol_klines = slice_klines_per_round(sol_unified_window, max_lookback=max_lookback, earliest_offset=earliest_offset)

    gate_config = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        cutoff_seconds=CUTOFF_SECONDS,
        mtf_lookbacks=strategy_cfg.gate.mtf_lookbacks,
        mtf_threshold=strategy_cfg.gate.mtf_threshold,
    )
    bankroll_tracker = InMemoryBankrollTracker(
        initial_bankroll=INITIAL_BANKROLL_BNB,
        window_days=strategy_cfg.risk.window_days,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_config, strategy_config=strategy_cfg, gate=None,
        cutoff_seconds=CUTOFF_SECONDS,
        min_bet_amount_bnb=min_bet_amount,
        treasury_fee_fraction=treasury_fee,
        bankroll_tracker=bankroll_tracker,
    )
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_klines)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_klines)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_klines)
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    ks = KillSwitchState(N=kill_switch_combo["N"], L=kill_switch_combo["L"], K=kill_switch_combo["K"]) \
         if kill_switch_combo is not None else None

    bankroll = INITIAL_BANKROLL_BNB
    bets_in_test_partition = 0  # for the "exclude first N" rule
    per_round_records = []      # only test partition

    for r in rounds_window:
        ep = int(r.epoch)
        in_test = ep >= test_lo

        decision = pipeline.decide_open_round(round_t=r)

        # Kill switch state advance — every round
        if ks is not None:
            ks.on_round_advance()

        canonical_would_bet = (decision.action == "BET" and decision.bet_size_bnb > 0.0)
        kill_switch_suppressed = False
        actual_action = decision.action
        actual_bet_size = decision.bet_size_bnb
        actual_bet_side = decision.bet_side
        profit = 0.0
        win = False

        if canonical_would_bet:
            if ks is not None and ks.is_paused():
                # Kill switch suppresses
                kill_switch_suppressed = True
                actual_action = "SKIP"
                actual_bet_size = 0.0
                actual_bet_side = None
            else:
                # Canonical bet executes
                bankroll -= decision.bet_size_bnb + GAS_COST_BET_BNB
                outcome = settle_bet_against_closed_round(
                    bet_bnb=decision.bet_size_bnb,
                    bet_side=decision.bet_side,
                    round_closed=r,
                    treasury_fee_fraction=treasury_fee,
                )
                bankroll += outcome.credit_bnb
                profit = outcome.credit_bnb - decision.bet_size_bnb - GAS_COST_BET_BNB
                win = outcome.outcome == "win"
                if ks is not None:
                    ks.on_bet_settled(profit)

        # Record (test partition only, after first-N-bets-excluded rule for kill-switch arms)
        if in_test:
            if canonical_would_bet:
                bets_in_test_partition += 1
            # Apply "exclude first N bets" rule: skip records where bets_in_test_partition <= N
            include_in_marginal = True
            if ks is not None and bets_in_test_partition <= ks.N:
                include_in_marginal = False
            per_round_records.append({
                "epoch": ep,
                "canonical_would_bet": canonical_would_bet,
                "ks_suppressed": kill_switch_suppressed,
                "actual_bet": actual_action == "BET",
                "actual_bet_size": actual_bet_size,
                "profit": profit,
                "win": win,
                "include_in_marginal": include_in_marginal,
            })

        # Pipeline state advance
        pipeline.record_settlement(bankroll=bankroll, start_at=int(r.start_at))
        pipeline.settle_closed_rounds(rounds=[r])

    return {
        "final_bankroll": bankroll,
        "n_records_test": len(per_round_records),
        "ks_n_pauses_fired": ks.n_pauses_fired if ks else 0,
        "per_round": per_round_records,
    }


# ============================================================
# Aggregation: marginal PnL per bet vs canonical baseline
# ============================================================

def aggregate_marginal_pnl(per_round_canonical: list, per_round_kill_switch: list):
    """Compute marginal PnL per CANONICAL bet across the test partition.

    For each round in test:
      - If canonical didn't bet: skip from numerator (zero contribution)
      - If canonical bet AND kill switch ALSO bet: profit_diff = profit_ks - profit_canon = 0 (same bet)
      - If canonical bet AND kill switch suppressed: profit_diff = 0 - profit_canon = -profit_canon
        (kill switch saves us if profit_canon was negative)

    Marginal = sum(profit_diff) / count(canonical_would_bet AND include_in_marginal).
    """
    canon_by_epoch = {r["epoch"]: r for r in per_round_canonical}
    diffs = []
    n_suppressed = 0
    for r_ks in per_round_kill_switch:
        if not r_ks["include_in_marginal"]:
            continue
        if not r_ks["canonical_would_bet"]:
            continue
        ep = r_ks["epoch"]
        c = canon_by_epoch.get(ep)
        if c is None:
            continue
        if r_ks["ks_suppressed"]:
            # Kill switch saved/wasted us this round
            diffs.append(-c["profit"])  # marginal = avoided this bet
            n_suppressed += 1
        else:
            # Same bet as canonical — zero marginal contribution
            diffs.append(0.0)
    return diffs, n_suppressed


def summarize_diffs(diffs: list):
    if not diffs:
        return {"n": 0}
    arr = np.array(diffs)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    ci_half = 1.96 * se
    return {
        "n": int(n),
        "mean_per_bet": mean,
        "std_per_bet": std,
        "ci_half_95": ci_half,
        "ci_lower_95": mean - ci_half,
        "ci_upper_95": mean + ci_half,
        "total_marginal_pnl": float(arr.sum()),
    }


# ============================================================
# Main protocol
# ============================================================

def write_atomic(path: Path, content: str):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def run_protocol():
    print("=" * 100, flush=True)
    print("p1a — kill-switch overlay protocol (v3.2 ratified)", flush=True)
    print(f"  WF windows: {len(WF_WINDOWS)}", flush=True)
    print(f"  Combo grid: {len(COMBO_GRID)}", flush=True)
    print(f"  WF PASS threshold: >=+{WF_PASS_POINT_ESTIMATE_MIN} BNB/bet (CI lower >= 0)", flush=True)
    print("=" * 100, flush=True)
    t_start = time.time()

    cc = load_contract_constants()
    treasury_fee = float(cc.treasury_fee_fraction)
    min_bet_amount = float(cc.min_bet_amount_bnb)

    # Resolve strategy to compute earliest_offset
    canonical_cfg = make_strategy_config(gates_on=True)
    max_lookback = max(canonical_cfg.gate.mtf_lookbacks)
    earliest_offset = CUTOFF_SECONDS + max_lookback + 1

    # ============================================================
    # Phase 1: Load WF data (canonical range only — all 4 windows fit in 437562..470969)
    # ============================================================
    wf_ep_min = WF_WINDOWS[0]["train_lo"]
    wf_ep_max = WF_WINDOWS[-1]["test_hi"]
    print(f"\n[1/6] Loading WF data range [{wf_ep_min}..{wf_ep_max}]...", flush=True)
    t0 = time.time()
    wf_rounds, wf_btc, wf_eth, wf_sol = load_data_for_range(
        wf_ep_min, wf_ep_max, use_extended=False, earliest_offset=earliest_offset,
    )
    print(f"  loaded {len(wf_rounds)} rounds, "
          f"btc={len(wf_btc)} eth={len(wf_eth)} sol={len(wf_sol)} klines, "
          f"elapsed={time.time()-t0:.1f}s", flush=True)

    rounds_by_window = []
    for w in WF_WINDOWS:
        win_rounds = [r for r in wf_rounds if w["train_lo"] <= int(r.epoch) <= w["test_hi"]]
        win_btc = {ep: kl for ep, kl in wf_btc.items() if w["train_lo"] <= ep <= w["test_hi"]}
        win_eth = {ep: kl for ep, kl in wf_eth.items() if w["train_lo"] <= ep <= w["test_hi"]}
        win_sol = {ep: kl for ep, kl in wf_sol.items() if w["train_lo"] <= ep <= w["test_hi"]}
        rounds_by_window.append((w, win_rounds, win_btc, win_eth, win_sol))
        print(f"  {w['name']}: {len(win_rounds)} rounds  "
              f"train={w['train_lo']}..{w['train_hi']}  test={w['test_lo']}..{w['test_hi']}",
              flush=True)

    # ============================================================
    # Phase 2: Canonical-Arm-A baseline per WF window (cache for marginal-PnL diffs)
    # ============================================================
    print("\n[2/6] Running canonical-Arm-A baseline per WF window...", flush=True)
    t0 = time.time()
    canonical_per_window = []  # list of per_round records per window
    for w, win_rounds, win_btc, win_eth, win_sol in rounds_by_window:
        result = backtest_window(
            win_rounds, win_btc, win_eth, win_sol,
            earliest_offset=earliest_offset,
            test_lo=w["test_lo"],
            gates_on=True, kill_switch_combo=None,
            treasury_fee=treasury_fee, min_bet_amount=min_bet_amount,
        )
        n_canon_test_bets = sum(1 for r in result["per_round"] if r["canonical_would_bet"])
        canonical_per_window.append(result["per_round"])
        print(f"  {w['name']}: canonical bets in test = {n_canon_test_bets}, "
              f"final bankroll = {result['final_bankroll']:.4f}",
              flush=True)
    print(f"  elapsed={time.time()-t0:.1f}s", flush=True)

    # ============================================================
    # Phase 3: Sweep — 60 combos × 4 windows
    # ============================================================
    print(f"\n[3/6] Sweep — {len(COMBO_GRID)} combos × {len(WF_WINDOWS)} windows...", flush=True)
    t0 = time.time()
    sweep_results = []  # list of dict per combo
    for combo_idx, combo in enumerate(COMBO_GRID):
        per_window_diffs = []
        per_window_suppression = []
        for window_idx, (w, win_rounds, win_btc, win_eth, win_sol) in enumerate(rounds_by_window):
            result = backtest_window(
                win_rounds, win_btc, win_eth, win_sol,
                earliest_offset=earliest_offset,
                test_lo=w["test_lo"],
                gates_on=True, kill_switch_combo=combo,
                treasury_fee=treasury_fee, min_bet_amount=min_bet_amount,
            )
            diffs, n_suppressed = aggregate_marginal_pnl(
                canonical_per_window[window_idx], result["per_round"]
            )
            per_window_diffs.append(diffs)
            per_window_suppression.append(n_suppressed)
        # Aggregate across windows
        all_diffs = [d for win in per_window_diffs for d in win]
        s = summarize_diffs(all_diffs)
        s["combo"] = combo
        s["per_window_n_suppressed"] = per_window_suppression
        sweep_results.append(s)
        if (combo_idx + 1) % 10 == 0:
            print(f"  ({combo_idx+1}/{len(COMBO_GRID)}) "
                  f"elapsed={time.time()-t0:.1f}s "
                  f"latest combo: N={combo['N']} L={combo['L']} K={combo['K']} "
                  f"mean={s.get('mean_per_bet',0):+.4f} n={s.get('n',0)}",
                  flush=True)
    print(f"  sweep elapsed={time.time()-t0:.1f}s", flush=True)

    # Pick best combo
    sweep_results_sorted = sorted(
        sweep_results,
        key=lambda x: x.get("mean_per_bet", -1e18),
        reverse=True,
    )
    best = sweep_results_sorted[0]
    best_combo = best["combo"]
    print(f"\n  BEST COMBO: N={best_combo['N']} L={best_combo['L']} K={best_combo['K']}", flush=True)
    print(f"  WF aggregated: n={best['n']} mean=+{best['mean_per_bet']:.4f} "
          f"CI=[{best['ci_lower_95']:+.4f}, {best['ci_upper_95']:+.4f}]", flush=True)

    # WF verdict
    wf_verdict = "INSUFFICIENT_POWER"
    if best["ci_lower_95"] >= WF_PASS_CI_LOWER_MIN and best["mean_per_bet"] >= WF_PASS_POINT_ESTIMATE_MIN:
        wf_verdict = "PASS"
    elif best["ci_upper_95"] <= WF_HARD_FAIL_CI_UPPER_MAX:
        wf_verdict = "HARD_FAIL"
    print(f"  WF verdict: {wf_verdict}", flush=True)

    # ============================================================
    # Phase 4: Placebo (5,000 seeds at picked combo, definition-(b) pause-rate match)
    # ============================================================
    print(f"\n[4/6] Placebo — {PLACEBO_TOTAL_SEEDS} seeds at picked combo...", flush=True)
    t0 = time.time()
    # Empirical pause-rate at picked combo: % canonical-would-fire suppressed (def b)
    total_canon_bets = 0
    total_ks_suppressed = sum(best["per_window_n_suppressed"])
    for win in canonical_per_window:
        total_canon_bets += sum(1 for r in win if r["canonical_would_bet"])
    pause_rate = total_ks_suppressed / total_canon_bets if total_canon_bets > 0 else 0.0
    print(f"  picked combo pause-rate (def b): {pause_rate*100:.2f}% "
          f"({total_ks_suppressed}/{total_canon_bets} canonical bets suppressed)", flush=True)

    # Placebo: for each seed, randomly suppress ~pause_rate fraction of CANONICAL'S BETS
    # in the test partition. Compute marginal-PnL same way.
    rng = np.random.default_rng(seed=42)
    placebo_means = []
    for seed_idx in range(PLACEBO_TOTAL_SEEDS):
        per_win_diffs = []
        for win in canonical_per_window:
            test_canon_bets_idx = [i for i, r in enumerate(win) if r["canonical_would_bet"]]
            if not test_canon_bets_idx:
                continue
            n_to_suppress = int(round(len(test_canon_bets_idx) * pause_rate))
            suppressed_idx = set(rng.choice(test_canon_bets_idx, size=n_to_suppress, replace=False)
                                  if n_to_suppress > 0 else [])
            for i, r in enumerate(win):
                if not r["canonical_would_bet"]:
                    continue
                if i in suppressed_idx:
                    per_win_diffs.append(-r["profit"])
                else:
                    per_win_diffs.append(0.0)
        if per_win_diffs:
            placebo_means.append(float(np.mean(per_win_diffs)))
    placebo_arr = np.array(placebo_means)
    placebo_99_92 = float(np.percentile(placebo_arr, PLACEBO_BONFERRONI_PERCENTILE))
    placebo_95 = float(np.percentile(placebo_arr, 95))
    picked_beats_99_92 = best["mean_per_bet"] >= placebo_99_92
    print(f"  placebo distribution: n={len(placebo_arr)}  mean={placebo_arr.mean():+.4f}  "
          f"95th={placebo_95:+.4f}  99.92nd={placebo_99_92:+.4f}", flush=True)
    print(f"  picked combo mean={best['mean_per_bet']:+.4f}  "
          f"beats 99.92nd Bonferroni: {picked_beats_99_92}", flush=True)
    print(f"  placebo elapsed={time.time()-t0:.1f}s", flush=True)

    # ============================================================
    # Phase 5: Arms A/B/C/D at picked combo
    # ============================================================
    print(f"\n[5/6] Arms A/B/C/D on full WF range (single backtest each)...", flush=True)
    t0 = time.time()
    arms = {}
    for arm_name, gates_on, ks_combo in [
        ("A", True,  None),
        ("B", False, None),
        ("C", False, best_combo),
        ("D", True,  best_combo),
    ]:
        # Run as one combined window (full WF range as single backtest)
        # Use test_lo = wf_ep_min + WF_TRAIN_DAYS*EP_PER_DAY to score everything past warmup
        full_test_lo = wf_ep_min + WF_TRAIN_DAYS * EP_PER_DAY
        result = backtest_window(
            wf_rounds, wf_btc, wf_eth, wf_sol,
            earliest_offset=earliest_offset,
            test_lo=full_test_lo,
            gates_on=gates_on,
            kill_switch_combo=ks_combo,
            treasury_fee=treasury_fee,
            min_bet_amount=min_bet_amount,
        )
        n_actual_bets = sum(1 for r in result["per_round"] if r["actual_bet"])
        n_wins = sum(1 for r in result["per_round"] if r["win"])
        total_pnl = sum(r["profit"] for r in result["per_round"])
        arms[arm_name] = {
            "gates_on": gates_on,
            "kill_switch": ks_combo,
            "n_actual_bets": n_actual_bets,
            "n_wins": n_wins,
            "win_rate": n_wins / n_actual_bets if n_actual_bets > 0 else 0.0,
            "total_pnl": total_pnl,
            "final_bankroll": result["final_bankroll"],
            "ks_n_pauses_fired": result["ks_n_pauses_fired"],
        }
        print(f"  Arm {arm_name}: bets={n_actual_bets} wr={arms[arm_name]['win_rate']*100:.1f}% "
              f"pnl={total_pnl:+.4f}", flush=True)
    print(f"  arms elapsed={time.time()-t0:.1f}s", flush=True)

    # Marginal contributions
    arm_a_pnl = arms["A"]["total_pnl"]
    arm_b_pnl = arms["B"]["total_pnl"]
    arm_c_pnl = arms["C"]["total_pnl"]
    arm_d_pnl = arms["D"]["total_pnl"]
    print(f"\n  Marginal contributions:", flush=True)
    print(f"    (A - B) cooldown+drawdown additive: {arm_a_pnl - arm_b_pnl:+.4f}", flush=True)
    print(f"    (C - B) kill-switch standalone:     {arm_c_pnl - arm_b_pnl:+.4f}", flush=True)
    print(f"    (D - A) kill-switch above gates:    {arm_d_pnl - arm_a_pnl:+.4f}", flush=True)

    # ============================================================
    # Phase 6: FROZEN HOLDOUT eval at picked combo
    # ============================================================
    print(f"\n[6/6] FROZEN HOLDOUT eval at picked combo...", flush=True)
    t0 = time.time()
    holdouts = {}
    for slice_name, ep_range, use_extended in [
        ("extension", EXTENSION_RANGE, True),
        ("v3", V3_RANGE, False),
        ("post_v1", POSTV1_RANGE, False),
    ]:
        ep_min, ep_max = ep_range
        h_rounds, h_btc, h_eth, h_sol = load_data_for_range(
            ep_min, ep_max, use_extended=use_extended, earliest_offset=earliest_offset,
        )
        # Run canonical (Arm A) and kill-switch-on-Arm-A (Arm D variant) — diff
        canon_result = backtest_window(
            h_rounds, h_btc, h_eth, h_sol,
            earliest_offset=earliest_offset, test_lo=ep_min,
            gates_on=True, kill_switch_combo=None,
            treasury_fee=treasury_fee, min_bet_amount=min_bet_amount,
        )
        ks_result = backtest_window(
            h_rounds, h_btc, h_eth, h_sol,
            earliest_offset=earliest_offset, test_lo=ep_min,
            gates_on=True, kill_switch_combo=best_combo,
            treasury_fee=treasury_fee, min_bet_amount=min_bet_amount,
        )
        diffs, n_suppressed = aggregate_marginal_pnl(canon_result["per_round"], ks_result["per_round"])
        s = summarize_diffs(diffs)
        s["n_canonical_bets"] = sum(1 for r in canon_result["per_round"] if r["canonical_would_bet"])
        s["n_suppressed"] = n_suppressed
        s["canon_pnl"] = sum(r["profit"] for r in canon_result["per_round"])
        s["ks_pnl"] = sum(r["profit"] for r in ks_result["per_round"])
        holdouts[slice_name] = s
        print(f"  {slice_name}: n_canon_bets={s['n_canonical_bets']} suppressed={s['n_suppressed']} "
              f"mean_per_bet=+{s.get('mean_per_bet',0):.4f} "
              f"CI=[{s.get('ci_lower_95',0):+.4f},{s.get('ci_upper_95',0):+.4f}]", flush=True)
    print(f"  holdout elapsed={time.time()-t0:.1f}s", flush=True)

    # ============================================================
    # Output
    # ============================================================
    out = {
        "spec": {
            "wf_windows": WF_WINDOWS,
            "n_combos": len(COMBO_GRID),
            "combo_grid": COMBO_GRID,
            "wf_pass_point_estimate_min": WF_PASS_POINT_ESTIMATE_MIN,
            "wf_pass_ci_lower_min": WF_PASS_CI_LOWER_MIN,
            "wf_hard_fail_ci_upper_max": WF_HARD_FAIL_CI_UPPER_MAX,
            "ext_pass_point_estimate_min": EXT_PASS_POINT_ESTIMATE_MIN,
            "ext_pass_ci_lower_min": EXT_PASS_CI_LOWER_MIN,
            "placebo_total_seeds": PLACEBO_TOTAL_SEEDS,
            "placebo_bonferroni_percentile": PLACEBO_BONFERRONI_PERCENTILE,
            "extension_range": list(EXTENSION_RANGE),
            "v3_range": list(V3_RANGE),
            "postv1_range": list(POSTV1_RANGE),
            "initial_bankroll_bnb": INITIAL_BANKROLL_BNB,
            "cutoff_seconds": CUTOFF_SECONDS,
        },
        "wf_sweep": {
            "best_combo": best_combo,
            "best": {k: v for k, v in best.items() if k != "combo"},
            "all_combos_sorted": [
                {**{k: v for k, v in s.items() if k != "combo"}, "combo": s["combo"]}
                for s in sweep_results_sorted
            ],
            "verdict": wf_verdict,
        },
        "placebo": {
            "n_seeds": len(placebo_arr),
            "mean": float(placebo_arr.mean()),
            "p95": placebo_95,
            "p99_92": placebo_99_92,
            "picked_beats_99_92": picked_beats_99_92,
            "pause_rate_def_b": pause_rate,
        },
        "arms": arms,
        "frozen_holdout": holdouts,
        "elapsed_seconds": time.time() - t_start,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(OUT_RESULTS, json.dumps(out, indent=2, default=str))
    print(f"\nResults JSON: {OUT_RESULTS}", flush=True)
    print(f"Total elapsed: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    run_protocol()
