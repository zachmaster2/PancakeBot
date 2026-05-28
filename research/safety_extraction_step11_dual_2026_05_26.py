"""Step 11 (dual-scale) — A/B/C/E + bonus at BOTH 5 BNB and 50 BNB initial.

Re-runs all four experiments at the small-bankroll scale, since user's actual
deployment bankroll is <5 BNB. Keeps the same per-experiment design:
  A: max_drawdown_fraction_from_peak sweep, standard run_fold
  B: absolute-BNB drawdown threshold via custom AbsoluteDrawdownTracker
  C: sizing-bankroll cap via module-level monkey-patch on _compute_bet_size
  E: anti-martingale via custom pipeline-loop runner

C's sizing_cap sweep is scale-aware:
  @50 BNB: {5, 10, 15, 20, 50}  (50 = baseline-equivalent)
  @ 5 BNB: {3, 5, 10, 20, 50}   (sizing_cap < initial bankroll is meaningful at 5)
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
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
import pancakebot.strategy.momentum_pipeline as _mp_module  # noqa: E402


# ---- Monkey-patch _compute_bet_size for Exp C (kept inactive when cap=inf) ----
_orig_compute_bet_size = _mp_module._compute_bet_size
_SIZING_CAP_BNB: float = float("inf")


def _patched_compute_bet_size(*, current_bankroll, **kwargs):
    capped = current_bankroll
    if current_bankroll is not None and _SIZING_CAP_BNB < current_bankroll:
        capped = _SIZING_CAP_BNB
    return _orig_compute_bet_size(current_bankroll=capped, **kwargs)


_mp_module._compute_bet_size = _patched_compute_bet_size


def set_sizing_cap(cap_bnb: float) -> None:
    global _SIZING_CAP_BNB
    _SIZING_CAP_BNB = float(cap_bnb)


# ---- Constants ----
EPOCH_MIN = 422298
EPOCH_MAX_CONFIG = 484999
CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2
POOL_CUTOFF = 6
TREASURY_FEE = 0.03
MIN_BET = 0.001

SCALES = (5.0,)  # 50 BNB already covered by prior Step 11/11c runs; this fills in 5 BNB

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
        "skip_drawdown_breaker": 0, "skip_cooldown": 0, "skip_other": 0,
    } for c in COHORT_ORDER}


def parse_trades(trades_csv: Path, initial_bankroll: float) -> tuple[dict, float]:
    per_cohort = empty_cohort_record()
    peak = initial_bankroll
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
            else:
                sr = (row.get("skip_reason") or "").strip()
                if sr == "risk_drawdown_breaker_fired":
                    per_cohort[coh]["skip_drawdown_breaker"] += 1
                elif sr == "risk_cooldown_active":
                    per_cohort[coh]["skip_cooldown"] += 1
                else:
                    per_cohort[coh]["skip_other"] += 1
    for cd in per_cohort.values():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0
        cd["mean_bet_size_bnb"] = (cd["total_bet_size_bnb"] / cd["n_bets"]
                                    if cd["n_bets"] else 0.0)
    return per_cohort, max_dd_frac


# ---- Experiment A ----
def exp_A(*, initial_bankroll, all_rounds, btc, eth, sol, earliest_offset, out_root):
    set_sizing_cap(float("inf"))  # disable Exp-C patch
    fractions = [0.05, 0.08, 0.10, 0.12, 0.15]
    results = []
    print(f"\n----- Exp A @ {initial_bankroll} BNB: dd_frac sweep -----")
    for frac in fractions:
        overrides = {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)},
                     "risk": {"max_drawdown_fraction_from_peak": frac}}
        sc = load_strategy_config_from_dict(overrides)
        spec = ipr.FoldSpec(name=f"A_dd{int(frac*100):02d}_{int(initial_bankroll)}bnb",
                             kline_cutoff_seconds=CANONICAL_CUTOFF,
                             epoch_start=EPOCH_MIN, epoch_end=EPOCH_MAX_CONFIG,
                             strategy_overrides=overrides)
        t0 = time.time()
        summary = ipr.run_fold(
            spec=spec, strategy_cfg=sc,
            all_rounds=all_rounds, btc_unified=btc, eth_unified=eth, sol_unified=sol,
            earliest_offset=earliest_offset, output_base_dir=out_root,
            initial_bankroll_bnb=initial_bankroll,
            treasury_fee_fraction=TREASURY_FEE, min_bet_amount_bnb=MIN_BET,
        )
        trades_csv = out_root / spec.name / "trades.csv"
        per_cohort, max_dd = parse_trades(trades_csv, initial_bankroll)
        elapsed = time.time() - t0
        results.append({
            "variant_label": f"dd_frac={frac}",
            "param": frac,
            "summary": {k: summary[k] for k in ("num_bets","num_wins","win_rate","net_pnl_bnb") if k in summary},
            "max_drawdown_realized_frac": max_dd,
            "per_cohort": per_cohort,
            "elapsed_seconds": elapsed,
        })
        print(f"  dd_frac={frac}: bets={summary['num_bets']} WR={summary['win_rate']:.4f} "
              f"pnl={summary['net_pnl_bnb']:+.4f} dd={max_dd*100:.2f}% ({elapsed:.1f}s)")
    return results


# ---- Experiment B ----
class AbsDDTracker:
    def __init__(self, *, initial_bankroll, max_abs_dd_bnb, cooldown_rounds):
        self._cur = float(initial_bankroll); self._peak = float(initial_bankroll)
        self._max_abs = float(max_abs_dd_bnb); self._cd_total = int(cooldown_rounds)
        self._cd_remain = 0; self._paused = False; self.n_pauses_fired = 0
    def current_bankroll(self): return self._cur
    def peak_bankroll(self, start_at): return self._peak
    def is_paused(self, start_at): return self._paused
    def tick_cooldown(self):
        if self._cd_remain > 0:
            self._cd_remain -= 1
            if self._cd_remain == 0: self._paused = False
    def cooldown_remaining(self): return self._cd_remain
    def set_paused(self, rounds, start_at): pass
    def record_settlement(self, bankroll, start_at):
        self._cur = float(bankroll)
        if self._cur > self._peak: self._peak = self._cur
        if not self._paused and (self._peak - self._cur) >= self._max_abs:
            self._paused = True; self._cd_remain = self._cd_total; self.n_pauses_fired += 1


def exp_B(*, initial_bankroll, all_rounds, btc, eth, sol, earliest_offset):
    set_sizing_cap(float("inf"))  # disable C patch
    overrides = {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)},
                 "risk": {"max_drawdown_fraction_from_peak": 1.0}}
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    max_lb = max(CANONICAL_LOOKBACKS)
    btc_k = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                       max_lookback=max_lb, earliest_offset=earliest_offset)
             for ep, kl in btc.items()}
    eth_k = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                       max_lookback=max_lb, earliest_offset=earliest_offset)
             for ep, kl in eth.items()}
    sol_k = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                       max_lookback=max_lb, earliest_offset=earliest_offset)
             for ep, kl in sol.items()}
    sim_rounds = [r for r in all_rounds if EPOCH_MIN <= r.epoch <= EPOCH_MAX_CONFIG]

    thresholds = [0.25, 0.50, 1.0, 1.5, 2.0]
    results = []
    print(f"\n----- Exp B @ {initial_bankroll} BNB: abs_dd sweep -----")
    for thr in thresholds:
        t0 = time.time()
        tracker = AbsDDTracker(initial_bankroll=initial_bankroll, max_abs_dd_bnb=thr,
                                cooldown_rounds=sc.risk.cooldown_rounds)
        pipeline = MomentumOnlyPipeline(
            config=gate_cfg, strategy_config=sc, gate=None,
            kline_cutoff_seconds=CANONICAL_CUTOFF, pool_cutoff_seconds=POOL_CUTOFF,
            min_bet_amount_bnb=MIN_BET, treasury_fee_fraction=TREASURY_FEE,
            bankroll_tracker=tracker,
        )
        pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_k)
        pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_k)
        pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_k)
        pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

        per_cohort = empty_cohort_record()
        bankroll = float(initial_bankroll); peak = bankroll; max_dd_frac = 0.0
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
            bet_size = float(decision.bet_size_bnb); side = str(decision.bet_side)
            bankroll -= bet_size + MAX_GAS_COST_BET_BNB
            outcome = settle_bet_against_closed_round(bet_bnb=bet_size, bet_side=side,
                                                       round_closed=round_t,
                                                       treasury_fee_fraction=TREASURY_FEE)
            bankroll += outcome.credit_bnb
            profit = outcome.credit_bnb - bet_size - MAX_GAS_COST_BET_BNB
            per_cohort[coh]["n_bets"] += 1; per_cohort[coh]["pnl_bnb"] += profit
            per_cohort[coh]["total_bet_size_bnb"] += bet_size
            if bet_size > per_cohort[coh]["max_bet_size_bnb"]:
                per_cohort[coh]["max_bet_size_bnb"] = bet_size
            if outcome.outcome == "win":
                per_cohort[coh]["n_wins"] += 1
            if bankroll > peak: peak = bankroll
            if peak > 0:
                dd = (peak - bankroll) / peak
                if dd > max_dd_frac: max_dd_frac = dd
            pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
            pipeline.settle_closed_rounds(rounds=[round_t])
        for cd in per_cohort.values():
            cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0
            cd["mean_bet_size_bnb"] = (cd["total_bet_size_bnb"] / cd["n_bets"] if cd["n_bets"] else 0.0)
        total_bets = sum(c["n_bets"] for c in per_cohort.values())
        total_wins = sum(c["n_wins"] for c in per_cohort.values())
        elapsed = time.time() - t0
        results.append({
            "variant_label": f"abs_dd={thr}", "param": thr,
            "summary": {"num_bets": total_bets, "num_wins": total_wins,
                        "win_rate": total_wins/total_bets if total_bets else 0.0,
                        "net_pnl_bnb": bankroll - initial_bankroll},
            "max_drawdown_realized_frac": max_dd_frac,
            "n_pauses_fired": tracker.n_pauses_fired,
            "per_cohort": per_cohort,
            "elapsed_seconds": elapsed,
        })
        print(f"  abs_dd={thr}: bets={total_bets} WR={total_wins/total_bets if total_bets else 0:.4f} "
              f"pnl={bankroll-initial_bankroll:+.4f} dd={max_dd_frac*100:.2f}% "
              f"pauses={tracker.n_pauses_fired} ({elapsed:.1f}s)")
    return results


# ---- Experiment C ----
def exp_C(*, initial_bankroll, sizing_caps, all_rounds, btc, eth, sol, earliest_offset, out_root):
    overrides = {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
    sc = load_strategy_config_from_dict(overrides)
    results = []
    print(f"\n----- Exp C @ {initial_bankroll} BNB: sizing_cap sweep "
          f"(caps={sizing_caps}) -----")
    for cap in sizing_caps:
        set_sizing_cap(cap)
        spec = ipr.FoldSpec(name=f"C_cap{int(cap*10):03d}_{int(initial_bankroll)}bnb",
                             kline_cutoff_seconds=CANONICAL_CUTOFF,
                             epoch_start=EPOCH_MIN, epoch_end=EPOCH_MAX_CONFIG,
                             strategy_overrides=overrides)
        t0 = time.time()
        summary = ipr.run_fold(
            spec=spec, strategy_cfg=sc,
            all_rounds=all_rounds, btc_unified=btc, eth_unified=eth, sol_unified=sol,
            earliest_offset=earliest_offset, output_base_dir=out_root,
            initial_bankroll_bnb=initial_bankroll,
            treasury_fee_fraction=TREASURY_FEE, min_bet_amount_bnb=MIN_BET,
        )
        trades_csv = out_root / spec.name / "trades.csv"
        per_cohort, max_dd = parse_trades(trades_csv, initial_bankroll)
        elapsed = time.time() - t0
        results.append({
            "variant_label": f"sizing_cap={cap}", "param": cap,
            "summary": {k: summary[k] for k in ("num_bets","num_wins","win_rate","net_pnl_bnb") if k in summary},
            "max_drawdown_realized_frac": max_dd,
            "per_cohort": per_cohort,
            "elapsed_seconds": elapsed,
        })
        print(f"  cap={cap}: bets={summary['num_bets']} WR={summary['win_rate']:.4f} "
              f"pnl={summary['net_pnl_bnb']:+.4f} dd={max_dd*100:.2f}% ({elapsed:.1f}s)")
    set_sizing_cap(float("inf"))  # reset
    return results


# ---- Experiment E ----
def exp_E(*, initial_bankroll, all_rounds, btc, eth, sol, earliest_offset):
    set_sizing_cap(float("inf"))
    overrides = {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    max_lb = max(CANONICAL_LOOKBACKS)
    btc_k = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                       max_lookback=max_lb, earliest_offset=earliest_offset)
             for ep, kl in btc.items()}
    eth_k = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                       max_lookback=max_lb, earliest_offset=earliest_offset)
             for ep, kl in eth.items()}
    sol_k = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                       max_lookback=max_lb, earliest_offset=earliest_offset)
             for ep, kl in sol.items()}
    sim_rounds = [r for r in all_rounds if EPOCH_MIN <= r.epoch <= EPOCH_MAX_CONFIG]

    streak_maxes = [1.5, 2.0, 2.5, 3.0]
    results = []
    print(f"\n----- Exp E @ {initial_bankroll} BNB: anti-martingale sweep -----")
    for sm in streak_maxes:
        t0 = time.time()
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
        pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_k)
        pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_k)
        pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_k)
        pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

        per_cohort = empty_cohort_record()
        bankroll = float(initial_bankroll); peak = bankroll; max_dd_frac = 0.0
        win_streak = 0; max_streak_obs = 0
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
            canonical_bet = float(decision.bet_size_bnb); side = str(decision.bet_side)
            multiplier = min(sm, 1.0 + 0.25 * win_streak)
            bet_size = canonical_bet * multiplier
            safe_max = max(0.0, bankroll - MAX_GAS_COST_BET_BNB - 0.01)
            bet_size = min(bet_size, safe_max); bet_size = max(bet_size, MIN_BET)
            if bankroll < bet_size + MAX_GAS_COST_BET_BNB:
                per_cohort[coh]["skip_other"] += 1
                pipeline.settle_closed_rounds(rounds=[round_t]); continue
            bankroll -= bet_size + MAX_GAS_COST_BET_BNB
            outcome = settle_bet_against_closed_round(bet_bnb=bet_size, bet_side=side,
                                                       round_closed=round_t,
                                                       treasury_fee_fraction=TREASURY_FEE)
            bankroll += outcome.credit_bnb
            profit = outcome.credit_bnb - bet_size - MAX_GAS_COST_BET_BNB
            per_cohort[coh]["n_bets"] += 1; per_cohort[coh]["pnl_bnb"] += profit
            per_cohort[coh]["total_bet_size_bnb"] += bet_size
            if bet_size > per_cohort[coh]["max_bet_size_bnb"]:
                per_cohort[coh]["max_bet_size_bnb"] = bet_size
            if outcome.outcome == "win":
                per_cohort[coh]["n_wins"] += 1
                win_streak += 1
                if win_streak > max_streak_obs: max_streak_obs = win_streak
            else:
                win_streak = 0
            if bankroll > peak: peak = bankroll
            if peak > 0:
                dd = (peak - bankroll) / peak
                if dd > max_dd_frac: max_dd_frac = dd
            pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
            pipeline.settle_closed_rounds(rounds=[round_t])
        for cd in per_cohort.values():
            cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0
            cd["mean_bet_size_bnb"] = (cd["total_bet_size_bnb"] / cd["n_bets"] if cd["n_bets"] else 0.0)
        total_bets = sum(c["n_bets"] for c in per_cohort.values())
        total_wins = sum(c["n_wins"] for c in per_cohort.values())
        elapsed = time.time() - t0
        results.append({
            "variant_label": f"streak_max={sm}", "param": sm,
            "summary": {"num_bets": total_bets, "num_wins": total_wins,
                        "win_rate": total_wins/total_bets if total_bets else 0.0,
                        "net_pnl_bnb": bankroll - initial_bankroll},
            "max_drawdown_realized_frac": max_dd_frac,
            "max_streak_observed": max_streak_obs,
            "per_cohort": per_cohort,
            "elapsed_seconds": elapsed,
        })
        print(f"  streak_max={sm}: bets={total_bets} WR={total_wins/total_bets if total_bets else 0:.4f} "
              f"pnl={bankroll-initial_bankroll:+.4f} dd={max_dd_frac*100:.2f}% "
              f"max_obs_streak={max_streak_obs} ({elapsed:.1f}s)")
    return results


# ---- Main ----
def main():
    t_all = time.time()
    print("--- loading rounds + klines ---")
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  loaded {len(all_rounds)} rounds; "
          f"max_epoch={max(r.epoch for r in all_rounds)}")
    max_lb = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lb + 1
    latest_offset = CANONICAL_CUTOFF + 1
    t_kl = time.time()
    print(f"  loading BTC klines...", flush=True)
    btc = ipr._load_klines_unified(ipr._BTC_KLINES_PATH,
                                    earliest_offset=earliest_offset, latest_offset=latest_offset,
                                    extended_path=ipr._EXT_BTC_KLINES_PATH)
    print(f"  BTC: {len(btc)} in {time.time()-t_kl:.1f}s", flush=True)
    t_kl = time.time()
    print(f"  loading ETH klines...", flush=True)
    eth = ipr._load_klines_unified(ipr._ETH_KLINES_PATH,
                                    earliest_offset=earliest_offset, latest_offset=latest_offset,
                                    extended_path=ipr._EXT_ETH_KLINES_PATH)
    print(f"  ETH: {len(eth)} in {time.time()-t_kl:.1f}s", flush=True)
    t_kl = time.time()
    print(f"  loading SOL klines...", flush=True)
    sol = ipr._load_klines_unified(ipr._SOL_KLINES_PATH,
                                    earliest_offset=earliest_offset, latest_offset=latest_offset,
                                    extended_path=ipr._EXT_SOL_KLINES_PATH)
    print(f"  SOL: {len(sol)} in {time.time()-t_kl:.1f}s", flush=True)
    out_root = Path(tempfile.mkdtemp(prefix="step11dual_"))

    all_results: dict[str, Any] = {}
    for initial_bankroll in SCALES:
        print(f"\n========================================")
        print(f"========== SCALE: {initial_bankroll} BNB ==========")
        print(f"========================================")
        scale_results = {}
        scale_results["A"] = exp_A(initial_bankroll=initial_bankroll,
                                     all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
                                     earliest_offset=earliest_offset, out_root=out_root)
        scale_results["B"] = exp_B(initial_bankroll=initial_bankroll,
                                     all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
                                     earliest_offset=earliest_offset)
        # Scale-aware C sweep
        if initial_bankroll == 5.0:
            c_caps = [3.0, 5.0, 10.0, 20.0, 50.0]
        else:
            c_caps = [5.0, 10.0, 15.0, 20.0, 50.0]
        scale_results["C"] = exp_C(initial_bankroll=initial_bankroll, sizing_caps=c_caps,
                                     all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
                                     earliest_offset=earliest_offset, out_root=out_root)
        scale_results["E"] = exp_E(initial_bankroll=initial_bankroll,
                                     all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
                                     earliest_offset=earliest_offset)
        all_results[f"{int(initial_bankroll)}bnb"] = scale_results

    out_path = REPO / "var" / "strategy_review" / "safety_extraction_step11_dual_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX_CONFIG,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "scales": list(SCALES),
                "cohort_defs": [list(c) for c in COHORT_DEFS],
            },
            "results": all_results,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
