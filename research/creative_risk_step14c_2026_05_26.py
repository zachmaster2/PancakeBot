"""Step 14c — shadow-exit diagnostic + new window variants.

User caught structural problems in Step 14's shadow_exit_10pos:
  - At ~1% bet rate, "last 10 shadow rounds" rarely contains 10 BETS in a single
    72-round cooldown. Deque-of-bets vs deque-of-rounds confusion.
  - Pipeline static dd=0.15 still fires during shadow phase, racing with tracker.
  - Cooldown ticks only on BET rounds in my Step 14 runner — should tick every
    round during cooldown.

Step 14c fixes:
  1. Neuter pipeline static dd to 1.0 for shadow variants; tracker fully owns breaker
  2. Tick cooldown EVERY round during shadow phase (not just BET rounds)
  3. Add diagnostic instrumentation per-fire-event:
     - cooldown round at exit
     - shadow bets in window
     - shadow PnL at exit
     - next real bet outcome
     - cohort at exit
  4. Test new variants with more representative windows:
     - shadow_exit_50rd / 100rd: ROUNDS-based windows (include skips with PnL=0)
     - shadow_exit_3bets / 5bets: BETS-based, smaller window (faster trigger)
  5. Counterfactual: random_resume_X at rounds {18, 24, 36, 48} — exit cooldown
     at fixed round regardless of shadow content. If random_resume_X matches
     shadow_exit_10pos, the "mechanism" is just shorter average cooldown, not
     shadow detection.

Critical: same gate as Step 14 (Step 12b +44.87 reference).
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
INITIAL_BANKROLL = 5.0
ABS_DD_FRAC = 0.15

GATE_REFERENCE_PNL = 44.8706
GATE_TOLERANCE = 1.0

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


@dataclass
class ShadowConfig:
    """Mode: 'bets' or 'rounds'.
    - 'bets': window contains last N shadow BET outcomes
    - 'rounds': window contains last N shadow ROUND PnLs (0 if no bet)
    - 'random_resume': no shadow logic, just exit cooldown at round X
    """
    label: str
    mode: str  # "bets", "rounds", "random_resume", or "baseline"
    window: int = 0
    exit_round: int = 0  # for random_resume


SHADOW_CONFIGS = [
    ShadowConfig("GATE_baseline", "baseline"),
    # Reference: original Step 14 winner re-run with bug fixes
    ShadowConfig("shadow_exit_10bets_FIXED", "bets", window=10),
    # New variants per spec
    ShadowConfig("shadow_exit_50rd", "rounds", window=50),
    ShadowConfig("shadow_exit_100rd", "rounds", window=100),
    ShadowConfig("shadow_exit_3bets", "bets", window=3),
    ShadowConfig("shadow_exit_5bets", "bets", window=5),
    # Counterfactual: random_resume (no shadow logic)
    ShadowConfig("random_resume_18", "random_resume", exit_round=18),
    ShadowConfig("random_resume_24", "random_resume", exit_round=24),
    ShadowConfig("random_resume_36", "random_resume", exit_round=36),
    ShadowConfig("random_resume_48", "random_resume", exit_round=48),
]


class Step14cTracker(InMemoryBankrollTracker):
    """Tracker with fixed shadow semantics."""

    def __init__(self, *, initial_bankroll: float, drawdown_peak_window_days: int,
                  peak_mode: str, shadow_config: ShadowConfig,
                  cooldown_rounds: int = COOLDOWN_ROUNDS,
                  abs_dd_frac: float = ABS_DD_FRAC):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._cfg = shadow_config
        self._cd_total = cooldown_rounds
        self._initial = initial_bankroll
        self._abs_dd_frac = abs_dd_frac
        # Shadow deque: stores (shadow_bet_pnl) for bets-mode OR (round_pnl, was_bet) for rounds-mode
        self._shadow_bet_pnls: deque[float] = deque(maxlen=200)
        self._shadow_round_pnls: deque[tuple[float, bool]] = deque(maxlen=200)
        # Counters
        self.n_pauses_fired = 0
        self.n_early_exits = 0
        self.n_extensions = 0
        # Diagnostic: per-fire records
        self.fire_events: list[dict[str, Any]] = []
        # Track cooldown start round for diagnostics
        self._cooldown_start_round_idx: int | None = None
        self._round_idx_counter = 0  # incremented externally each round

    def is_paused(self, as_of_start_at: int) -> bool:
        # Phase 1: in cooldown?
        if self._cooldown > 0:
            # Shadow mode (bets/rounds): don't pause pipeline (runner handles)
            if self._cfg.mode in ("bets", "rounds", "random_resume"):
                return False
            return True
        # Phase 2: fresh trigger via absolute drawdown
        current = self.current_bankroll()
        peak = self.peak_bankroll(as_of_start_at)
        if peak > 0:
            dd = (peak - current) / peak
            if dd >= self._abs_dd_frac:
                # +1 cooldown compensation; if shadow mode, return False to let pipeline through
                self.set_paused(self._cd_total + 1, as_of_start_at)
                self.n_pauses_fired += 1
                self._shadow_bet_pnls.clear()
                self._shadow_round_pnls.clear()
                self._cooldown_start_round_idx = self._round_idx_counter
                if self._cfg.mode in ("bets", "rounds", "random_resume"):
                    return False
                return True
        return False

    def is_in_cooldown(self) -> bool:
        return self._cooldown > 0

    def cooldown_round(self) -> int:
        """Returns rounds elapsed in current cooldown (1..72)."""
        return max(0, self._cd_total - self._cooldown + 1)

    def record_shadow_bet(self, pnl: float) -> None:
        """Called by runner when shadow phase + decision was BET."""
        self._shadow_bet_pnls.append(pnl)
        if self._cfg.mode == "rounds":
            self._shadow_round_pnls.append((pnl, True))

    def record_shadow_round_no_bet(self) -> None:
        """Called by runner when shadow phase + decision was SKIP (rounds mode only)."""
        if self._cfg.mode == "rounds":
            self._shadow_round_pnls.append((0.0, False))

    def should_exit(self) -> bool:
        if self._cfg.mode == "random_resume":
            elapsed = self._cd_total - self._cooldown
            return elapsed >= self._cfg.exit_round
        if self._cfg.mode == "bets":
            w = self._cfg.window
            if len(self._shadow_bet_pnls) < w:
                return False
            window_pnl = sum(list(self._shadow_bet_pnls)[-w:])
            return window_pnl > 0
        if self._cfg.mode == "rounds":
            w = self._cfg.window
            if len(self._shadow_round_pnls) < w:
                return False
            window_pnl = sum(pnl for pnl, _ in list(self._shadow_round_pnls)[-w:])
            return window_pnl > 0
        return False

    def force_clear_cooldown(self, fire_record: dict[str, Any] | None = None) -> None:
        self.n_early_exits += 1
        if fire_record is not None:
            self.fire_events.append(fire_record)
        self._cooldown = 0

    def manual_tick(self) -> None:
        """Tick cooldown once per round during shadow phase. Replaces pipeline's tick."""
        if self._cooldown > 0:
            self._cooldown -= 1


def run_backtest(*, shadow_config: ShadowConfig, all_rounds, btc_klines,
                  eth_klines, sol_klines, earliest_offset: int) -> dict[str, Any]:
    # For shadow/random variants: neuter pipeline static dd (tracker handles)
    # For baseline: keep pipeline static dd at 0.15
    if shadow_config.mode == "baseline":
        overrides = {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
    else:
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
    tracker = Step14cTracker(
        initial_bankroll=INITIAL_BANKROLL,
        drawdown_peak_window_days=DRAWDOWN_PEAK_WINDOW_DAYS,
        peak_mode="rolling_7d",
        shadow_config=shadow_config,
        cooldown_rounds=COOLDOWN_ROUNDS,
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

    sim_rounds = [r for r in all_rounds if EPOCH_MIN <= r.epoch <= EPOCH_MAX]
    per_cohort = {c: {"n_rounds": 0, "n_bets": 0, "n_wins": 0, "pnl_bnb": 0.0,
                      "n_shadow_bets": 0, "n_skip": 0} for c in COHORT_ORDER}
    bankroll = float(INITIAL_BANKROLL); peak = bankroll
    pending_post_exit_check: dict[str, Any] | None = None

    for round_idx, round_t in enumerate(sim_rounds):
        tracker._round_idx_counter = round_idx
        ep = int(round_t.epoch)
        coh = cohort_of(ep)
        per_cohort[coh]["n_rounds"] += 1

        was_in_cooldown_at_start = tracker.is_in_cooldown()
        decision = pipeline.decide_open_round(round_t=round_t)

        if decision.action != "BET":
            # SKIP for any reason
            per_cohort[coh]["n_skip"] += 1
            if was_in_cooldown_at_start and shadow_config.mode in ("bets", "rounds", "random_resume"):
                # Shadow phase + no bet → record no-bet round, tick cooldown
                tracker.record_shadow_round_no_bet()
                tracker.manual_tick()
                if tracker.should_exit():
                    # Compute exit diagnostic
                    cd_round = COOLDOWN_ROUNDS - tracker._cooldown + 1
                    shadow_bets_in_window = (sum(1 for p in list(tracker._shadow_bet_pnls)[-100:] if p != 0.0)
                                              if shadow_config.mode == "bets" else
                                              sum(1 for _, b in list(tracker._shadow_round_pnls)[-shadow_config.window:] if b)
                                              if shadow_config.mode == "rounds" else 0)
                    if shadow_config.mode == "bets":
                        window_pnl = sum(list(tracker._shadow_bet_pnls)[-shadow_config.window:])
                    elif shadow_config.mode == "rounds":
                        window_pnl = sum(p for p, _ in list(tracker._shadow_round_pnls)[-shadow_config.window:])
                    else:
                        window_pnl = 0.0
                    fire_record = {
                        "cohort": coh, "cooldown_round_at_exit": cd_round,
                        "shadow_bets_in_window_at_exit": shadow_bets_in_window,
                        "shadow_pnl_at_exit": window_pnl,
                        "cooldown_saved": COOLDOWN_ROUNDS - cd_round,
                        "exit_epoch": ep,
                    }
                    tracker.force_clear_cooldown(fire_record)
                    pending_post_exit_check = fire_record
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Pipeline returned BET
        if was_in_cooldown_at_start and shadow_config.mode in ("bets", "rounds", "random_resume"):
            # SHADOW BET: record outcome, don't settle for real
            bet_size = float(decision.bet_size_bnb)
            outcome = settle_bet_against_closed_round(
                bet_bnb=bet_size, bet_side=str(decision.bet_side),
                round_closed=round_t, treasury_fee_fraction=TREASURY_FEE,
            )
            shadow_profit = outcome.credit_bnb - bet_size - MAX_GAS_COST_BET_BNB
            tracker.record_shadow_bet(shadow_profit)
            per_cohort[coh]["n_shadow_bets"] += 1
            tracker.manual_tick()
            if tracker.should_exit():
                cd_round = COOLDOWN_ROUNDS - tracker._cooldown + 1
                shadow_bets_in_window = (sum(1 for _ in list(tracker._shadow_bet_pnls)[-shadow_config.window:])
                                          if shadow_config.mode == "bets" else
                                          sum(1 for _, b in list(tracker._shadow_round_pnls)[-shadow_config.window:] if b)
                                          if shadow_config.mode == "rounds" else 0)
                if shadow_config.mode == "bets":
                    window_pnl = sum(list(tracker._shadow_bet_pnls)[-shadow_config.window:])
                elif shadow_config.mode == "rounds":
                    window_pnl = sum(p for p, _ in list(tracker._shadow_round_pnls)[-shadow_config.window:])
                else:
                    window_pnl = 0.0
                fire_record = {
                    "cohort": coh, "cooldown_round_at_exit": cd_round,
                    "shadow_bets_in_window_at_exit": shadow_bets_in_window,
                    "shadow_pnl_at_exit": window_pnl,
                    "cooldown_saved": COOLDOWN_ROUNDS - cd_round,
                    "exit_epoch": ep,
                }
                tracker.force_clear_cooldown(fire_record)
                pending_post_exit_check = fire_record
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # REAL BET (not in cooldown)
        bet_size = float(decision.bet_size_bnb)
        bankroll -= bet_size + MAX_GAS_COST_BET_BNB
        outcome = settle_bet_against_closed_round(
            bet_bnb=bet_size, bet_side=str(decision.bet_side),
            round_closed=round_t, treasury_fee_fraction=TREASURY_FEE,
        )
        bankroll += outcome.credit_bnb
        profit = outcome.credit_bnb - bet_size - MAX_GAS_COST_BET_BNB

        per_cohort[coh]["n_bets"] += 1
        per_cohort[coh]["pnl_bnb"] += profit
        if outcome.outcome == "win":
            per_cohort[coh]["n_wins"] += 1
        if bankroll > peak: peak = bankroll

        # Diagnostic: if this is the first real bet after an early-exit, record outcome
        if pending_post_exit_check is not None:
            pending_post_exit_check["post_exit_first_bet_pnl"] = profit
            pending_post_exit_check["post_exit_first_bet_outcome"] = outcome.outcome
            pending_post_exit_check["post_exit_first_bet_epoch"] = ep
            pending_post_exit_check = None

        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])

    for cd in per_cohort.values():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0

    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())

    # Diagnostic stats
    fire_events = tracker.fire_events
    cd_at_exit = [f["cooldown_round_at_exit"] for f in fire_events]
    bets_in_window = [f["shadow_bets_in_window_at_exit"] for f in fire_events]
    shadow_pnl_at_exit = [f["shadow_pnl_at_exit"] for f in fire_events]
    cooldown_saved = [f["cooldown_saved"] for f in fire_events]
    post_exit_pnls = [f.get("post_exit_first_bet_pnl") for f in fire_events
                      if "post_exit_first_bet_pnl" in f]
    post_exit_wins = sum(1 for f in fire_events
                         if f.get("post_exit_first_bet_outcome") == "win")
    post_exit_total = sum(1 for f in fire_events
                          if "post_exit_first_bet_outcome" in f)

    def pct(lst, p):
        if not lst: return None
        idx = max(0, min(len(lst) - 1, int(len(lst) * p / 100)))
        return sorted(lst)[idx]

    diag = {
        "n_fires": len(fire_events),
        "n_early_exits": tracker.n_early_exits,
        "cd_at_exit_min": min(cd_at_exit) if cd_at_exit else None,
        "cd_at_exit_p25": pct(cd_at_exit, 25),
        "cd_at_exit_p50": pct(cd_at_exit, 50),
        "cd_at_exit_p75": pct(cd_at_exit, 75),
        "cd_at_exit_max": max(cd_at_exit) if cd_at_exit else None,
        "bets_in_window_min": min(bets_in_window) if bets_in_window else None,
        "bets_in_window_p50": pct(bets_in_window, 50),
        "bets_in_window_max": max(bets_in_window) if bets_in_window else None,
        "shadow_pnl_at_exit_mean": statistics.mean(shadow_pnl_at_exit) if shadow_pnl_at_exit else None,
        "cooldown_saved_mean": statistics.mean(cooldown_saved) if cooldown_saved else None,
        "post_exit_first_bet_wins": post_exit_wins,
        "post_exit_first_bet_total": post_exit_total,
        "post_exit_wr": post_exit_wins / post_exit_total if post_exit_total else None,
        "post_exit_mean_pnl": statistics.mean([p for p in post_exit_pnls if p is not None])
                              if post_exit_pnls else None,
    }

    return {
        "label": shadow_config.label,
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets else 0.0,
            "net_pnl_bnb": bankroll - INITIAL_BANKROLL,
            "final_bankroll": bankroll,
        },
        "n_pauses_fired": tracker.n_pauses_fired,
        "n_early_exits": tracker.n_early_exits,
        "diagnostic": diag,
        "fire_events": fire_events,
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

    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  klines: BTC={len(btc)} ETH={len(eth)} SOL={len(sol)}", flush=True)

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

    results: list[dict[str, Any]] = []

    print(f"\n--- running 10 configs ---", flush=True)
    for cfg in SHADOW_CONFIGS:
        t = time.time()
        r = run_backtest(
            shadow_config=cfg,
            all_rounds=all_rounds, btc_klines=btc_klines,
            eth_klines=eth_klines, sol_klines=sol_klines,
            earliest_offset=earliest_offset,
        )
        r["elapsed_seconds"] = time.time() - t
        results.append(r)
        s = r["summary"]
        d = r["diagnostic"]
        post_exit_wr = f"{d['post_exit_wr']*100:.1f}%" if d.get("post_exit_wr") is not None else "n/a"
        print(f"  {cfg.label:32s}: pnl={s['net_pnl_bnb']:+8.4f} bets={s['num_bets']:>4d} "
              f"fires={r['n_pauses_fired']} early_exits={r['n_early_exits']} "
              f"post_exit_wr={post_exit_wr} "
              f"cd_saved={d.get('cooldown_saved_mean')!r} ({r['elapsed_seconds']:.1f}s)",
              flush=True)

    # Gate check
    gate_r = results[0]
    gate_pnl = gate_r["summary"]["net_pnl_bnb"]
    delta = gate_pnl - GATE_REFERENCE_PNL
    gate_pass = abs(delta) <= GATE_TOLERANCE
    print(f"\nGATE: pnl={gate_pnl:+.4f} delta={delta:+.4f} {'PASS' if gate_pass else 'FAIL'}",
          flush=True)

    print(f"\n--- Final ranking (sorted by PnL desc) ---", flush=True)
    for r in sorted(results, key=lambda x: -x["summary"]["net_pnl_bnb"]):
        s = r["summary"]
        print(f"  {r['label']:32s}: pnl={s['net_pnl_bnb']:+8.4f} bets={s['num_bets']:>4d} "
              f"fires={r['n_pauses_fired']} early_exits={r['n_early_exits']}", flush=True)

    out_path = REPO / "var" / "strategy_review" / "creative_risk_step14c_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "gate_pnl": gate_pnl, "gate_delta": delta, "gate_pass": gate_pass,
            "results": results,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
