"""Step 13b — fix peak-semantics + expand adaptive dd_frac curve set.

BUG FIX from Step 13:
  Step 13's AdaptiveBankrollTracker used all-time-max peak instead of the
  production rolling-7d peak. Numbers were systematically lower than
  production-tracker comparable.

  Step 13b subclasses InMemoryBankrollTracker directly, inheriting the
  production rolling-7d peak logic verbatim. Only the adaptive dd_frac
  check is added.

VERIFICATION GATE (first action):
  Run static dd=0.15 @ 5 BNB. Confirm it matches Step 12b's +44.87 BNB.
  If gate fails, halt — the tracker is still wrong.

EXPANDED CURVE SET (9 curves; original 5 + 4 new "relax-on-dip"):

  Original (Step 13):
    1. linear:        max(0.08, 0.15 - (br-5)/45 * 0.07)
    2. piecewise:     <10→0.15; <20→0.12; <30→0.10; >=30→0.08
    3. log:           max(0.08, 0.15 - log10(br/5) * 0.07)
    4. inv_sqrt:      max(0.08, 0.15 / sqrt(br/5))
    5. aggressive_early: <7→0.15; <15→0.10; >=15→0.08

  New (Step 13b — relax-on-dip family):
    6. relax_linear:    0.15 if br<=init else max(0.08, 0.15 - (br-init)/45 * 0.07)
    7. relax_aggressive: 0.20 if br<0.7*init; 0.15 if br<=init; else 0.08
    8. symmetric:       clamp(0.05, 0.15 - (br/init - 1) * 0.05, 0.25)
    9. no_breaker_on_dip: 1.0 if br<init else 0.08

  Curve 10 (high-water reset per cohort) skipped — needs cohort knowledge
  inside the tracker, structurally awkward.

Total: 9 curves × {vol on/off} × 2 scales = 36 + 4 static references = 40.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import time
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
VOL_LOOKBACK_HOURS = 24
VOL_THRESHOLD_PCT = 30.0
COOLDOWN_ROUNDS = 72
DRAWDOWN_PEAK_WINDOW_DAYS = 7

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
        "total_bet_size_bnb": 0.0, "n_vol_vetoed": 0,
        "skip_drawdown_breaker": 0, "skip_cooldown": 0, "skip_other": 0,
        "sum_dd_frac_in_effect": 0.0,
        "n_dd_observations": 0,
    } for c in COHORT_ORDER}


# ---- Adaptive curves (current_bankroll, initial_bankroll) -> dd_frac ----

def curve_linear(br: float, init: float) -> float:  # noqa: ARG001
    val = 0.15 - max(0.0, br - 5) / 45.0 * 0.07
    return max(0.08, min(0.15, val))

def curve_piecewise_step(br: float, init: float) -> float:  # noqa: ARG001
    if br < 10: return 0.15
    if br < 20: return 0.12
    if br < 30: return 0.10
    return 0.08

def curve_log(br: float, init: float) -> float:  # noqa: ARG001
    if br <= 0: return 0.15
    val = 0.15 - math.log10(br / 5.0) * 0.07 if br >= 5 else 0.15
    return max(0.08, min(0.15, val))

def curve_inv_sqrt(br: float, init: float) -> float:  # noqa: ARG001
    if br <= 0: return 0.15
    val = 0.15 / math.sqrt(br / 5.0) if br >= 5 else 0.15
    return max(0.08, min(0.15, val))

def curve_aggressive_early(br: float, init: float) -> float:  # noqa: ARG001
    if br < 7: return 0.15
    if br < 15: return 0.10
    return 0.08

def curve_relax_linear(br: float, init: float) -> float:
    if br <= init: return 0.15
    return max(0.08, 0.15 - (br - init) / 45.0 * 0.07)

def curve_relax_aggressive(br: float, init: float) -> float:
    if br < 0.7 * init: return 0.20
    if br <= init: return 0.15
    return 0.08

def curve_symmetric(br: float, init: float) -> float:
    if init <= 0: return 0.15
    val = 0.15 - (br / init - 1.0) * 0.05
    return max(0.05, min(0.25, val))

def curve_no_breaker_on_dip(br: float, init: float) -> float:
    if br < init: return 1.0  # effectively disable breaker
    return 0.08


def curve_static_factory(value: float) -> Callable[[float, float], float]:
    def f(br: float, init: float) -> float:  # noqa: ARG001
        return value
    return f


ADAPTIVE_CURVES = {
    "linear":            curve_linear,
    "piecewise_step":    curve_piecewise_step,
    "log":               curve_log,
    "inv_sqrt":          curve_inv_sqrt,
    "aggressive_early":  curve_aggressive_early,
    "relax_linear":      curve_relax_linear,
    "relax_aggressive":  curve_relax_aggressive,
    "symmetric":         curve_symmetric,
    "no_breaker_on_dip": curve_no_breaker_on_dip,
}


# ---- AdaptiveBankrollTracker — subclasses InMemoryBankrollTracker ----

class AdaptiveBankrollTracker(InMemoryBankrollTracker):
    """Inherits production rolling-7d peak logic. Adds adaptive dd_frac
    breaker check in record_settlement.
    """

    def __init__(self, *, initial_bankroll: float, drawdown_peak_window_days: int,
                  peak_mode: str, dd_frac_curve: Callable[[float, float], float],
                  cooldown_rounds: int = COOLDOWN_ROUNDS):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._curve = dd_frac_curve
        self._cd_total = int(cooldown_rounds)
        self._initial_bankroll = float(initial_bankroll)
        self.n_pauses_fired = 0
        self.last_dd_frac = dd_frac_curve(initial_bankroll, initial_bankroll)

    def record_settlement(self, bankroll: float, start_at: int) -> None:
        super().record_settlement(bankroll, start_at)
        current = self.current_bankroll()
        peak = self.peak_bankroll(start_at)
        self.last_dd_frac = self._curve(current, self._initial_bankroll)
        if peak > 0 and not self.is_paused(start_at):
            dd = (peak - current) / peak
            if dd >= self.last_dd_frac:
                self.set_paused(self._cd_total, start_at)
                self.n_pauses_fired += 1


def compute_vol_cache(btc_timeline: list[tuple[int, int, float]],
                       lookback_seconds: int) -> dict[int, float]:
    if not btc_timeline:
        return {}
    epochs = np.array([r[0] for r in btc_timeline])
    ts = np.array([r[1] for r in btc_timeline])
    closes = np.array([r[2] for r in btc_timeline])
    n = len(ts)
    log_closes = np.log(closes)
    log_returns = np.diff(log_closes)
    end_ts_of_returns = ts[1:]
    out: dict[int, float] = {}
    PER_YEAR_5MIN = 288 * 365
    for i in range(n):
        target_ts = ts[i]
        cutoff_low = target_ts - lookback_seconds
        cutoff_high = target_ts - 2
        idx_lo = np.searchsorted(end_ts_of_returns, cutoff_low, side="left")
        idx_hi = np.searchsorted(end_ts_of_returns, cutoff_high, side="right")
        if idx_hi - idx_lo < 3:
            continue
        window = log_returns[idx_lo:idx_hi]
        sd = float(np.std(window, ddof=1))
        vol_ann_pct = sd * math.sqrt(PER_YEAR_5MIN) * 100.0
        out[int(epochs[i])] = vol_ann_pct
    return out


def run_backtest(*, initial_bankroll: float,
                  dd_frac_curve: Callable[[float, float], float],
                  vol_cache: dict[int, float] | None, vol_threshold: float,
                  all_rounds, btc_klines, eth_klines, sol_klines,
                  earliest_offset: int, label: str) -> dict[str, Any]:
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
    tracker = AdaptiveBankrollTracker(
        initial_bankroll=initial_bankroll,
        drawdown_peak_window_days=DRAWDOWN_PEAK_WINDOW_DAYS,
        peak_mode="rolling_7d",
        dd_frac_curve=dd_frac_curve,
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
    per_cohort = empty_cohort_record()
    bankroll = float(initial_bankroll); peak = bankroll; max_dd_frac = 0.0
    n_vol_vetoes = 0

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        coh = cohort_of(ep)
        per_cohort[coh]["n_rounds"] += 1
        per_cohort[coh]["sum_dd_frac_in_effect"] += tracker.last_dd_frac
        per_cohort[coh]["n_dd_observations"] += 1

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

        if vol_cache is not None:
            vol = vol_cache.get(ep)
            if vol is None or vol < vol_threshold:
                n_vol_vetoes += 1
                per_cohort[coh]["n_vol_vetoed"] += 1
                pipeline.settle_closed_rounds(rounds=[round_t])
                continue

        bet_size = float(decision.bet_size_bnb); side = str(decision.bet_side)
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
        cd["mean_bet_size_bnb"] = (cd["total_bet_size_bnb"] / cd["n_bets"]
                                    if cd["n_bets"] else 0.0)
        cd["mean_dd_frac_in_effect"] = (cd["sum_dd_frac_in_effect"] / cd["n_dd_observations"]
                                          if cd["n_dd_observations"] else 0.0)
    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())
    return {
        "label": label,
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins/total_bets if total_bets else 0.0,
            "net_pnl_bnb": bankroll - initial_bankroll,
            "final_bankroll": bankroll,
        },
        "max_drawdown_frac": max_dd_frac,
        "n_vol_vetoes": n_vol_vetoes,
        "n_breaker_fires": tracker.n_pauses_fired,
        "per_cohort": per_cohort,
    }


def main():
    t_all = time.time()
    print("--- loading rounds + klines ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  {len(all_rounds)} rounds; epoch range "
          f"[{all_rounds[0].epoch}..{max(r.epoch for r in all_rounds)}]", flush=True)

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1

    for name, path, ext in [
        ("BTC", ipr._BTC_KLINES_PATH, ipr._EXT_BTC_KLINES_PATH),
        ("ETH", ipr._ETH_KLINES_PATH, ipr._EXT_ETH_KLINES_PATH),
        ("SOL", ipr._SOL_KLINES_PATH, ipr._EXT_SOL_KLINES_PATH),
    ]:
        pass

    t_kl = time.time()
    print("  loading BTC klines...", flush=True)
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    print(f"  BTC: {len(btc)} in {time.time()-t_kl:.1f}s", flush=True)
    t_kl = time.time()
    print("  loading ETH klines...", flush=True)
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    print(f"  ETH: {len(eth)} in {time.time()-t_kl:.1f}s", flush=True)
    t_kl = time.time()
    print("  loading SOL klines...", flush=True)
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

    # Vol cache (24h)
    print("\n--- building 24h vol cache ---", flush=True)
    btc_timeline: list[tuple[int, int, float]] = []
    for ep, kl in btc.items():
        if not kl: continue
        last_candle = kl[-1]
        ts_ms = int(last_candle[0]); last_close = float(last_candle[4])
        if last_close > 0:
            btc_timeline.append((int(ep), ts_ms // 1000, last_close))
    btc_timeline.sort(key=lambda x: x[1])
    t_v = time.time()
    vol_cache = compute_vol_cache(btc_timeline, VOL_LOOKBACK_HOURS * 3600)
    print(f"  vol_24h: {len(vol_cache)} epochs ({time.time()-t_v:.1f}s)", flush=True)

    # ----- VERIFICATION GATE -----
    print(f"\n=========================================", flush=True)
    print(f"VERIFICATION GATE: static dd=0.15 @ 5 BNB", flush=True)
    print(f"Step 12b reference: +44.87 BNB / 1634 bets", flush=True)
    print(f"=========================================", flush=True)
    t = time.time()
    gate_r = run_backtest(
        initial_bankroll=5.0, dd_frac_curve=curve_static_factory(0.15),
        vol_cache=None, vol_threshold=0.0,
        all_rounds=all_rounds, btc_klines=btc_klines,
        eth_klines=eth_klines, sol_klines=sol_klines,
        earliest_offset=earliest_offset, label="gate_static_dd15_5bnb",
    )
    s = gate_r["summary"]
    gate_pnl = s["net_pnl_bnb"]
    gate_bets = s["num_bets"]
    delta = gate_pnl - 44.87
    print(f"GATE RESULT: pnl={gate_pnl:+.4f} BNB / {gate_bets} bets / "
          f"fires={gate_r['n_breaker_fires']} ({time.time()-t:.1f}s)", flush=True)
    print(f"GATE DELTA: {delta:+.4f} vs Step 12b +44.87", flush=True)
    gate_pass = abs(delta) < 1.0  # Within 1 BNB
    print(f"GATE STATUS: {'PASS' if gate_pass else 'FAIL'}", flush=True)
    if not gate_pass:
        print(f"  WARNING: gate failed; continuing but absolute numbers may not match Step 12b", flush=True)

    results: list[dict[str, Any]] = [gate_r]
    gate_r["scale"] = 5.0; gate_r["curve"] = "static_dd15_GATE"; gate_r["has_vol_filter"] = False
    gate_r["elapsed_seconds"] = time.time() - t

    # ----- Static refs (other than gate) -----
    for scale in (5.0, 50.0):
        for dd in (0.15, 0.08):
            if scale == 5.0 and dd == 0.15:
                continue  # already ran as gate
            label = f"ref_static_dd{int(dd*100):02d}_{int(scale)}bnb"
            t = time.time()
            r = run_backtest(
                initial_bankroll=scale, dd_frac_curve=curve_static_factory(dd),
                vol_cache=None, vol_threshold=0.0,
                all_rounds=all_rounds, btc_klines=btc_klines,
                eth_klines=eth_klines, sol_klines=sol_klines,
                earliest_offset=earliest_offset, label=label,
            )
            r["scale"] = scale; r["curve"] = f"static_{dd}"
            r["has_vol_filter"] = False
            r["elapsed_seconds"] = time.time() - t
            results.append(r)
            s = r["summary"]
            print(f"  {label}: bets={s['num_bets']} WR={s['win_rate']:.4f} "
                  f"pnl={s['net_pnl_bnb']:+.4f} fires={r['n_breaker_fires']} "
                  f"({r['elapsed_seconds']:.1f}s)", flush=True)

    # ----- Adaptive curves -----
    for scale in (5.0, 50.0):
        for curve_name, curve_fn in ADAPTIVE_CURVES.items():
            for filter_on in (False, True):
                label = f"adaptive_{curve_name}_{'volON' if filter_on else 'volOFF'}_{int(scale)}bnb"
                t = time.time()
                r = run_backtest(
                    initial_bankroll=scale, dd_frac_curve=curve_fn,
                    vol_cache=(vol_cache if filter_on else None),
                    vol_threshold=VOL_THRESHOLD_PCT,
                    all_rounds=all_rounds, btc_klines=btc_klines,
                    eth_klines=eth_klines, sol_klines=sol_klines,
                    earliest_offset=earliest_offset, label=label,
                )
                r["scale"] = scale; r["curve"] = curve_name
                r["has_vol_filter"] = filter_on
                r["elapsed_seconds"] = time.time() - t
                results.append(r)
                s = r["summary"]
                print(f"  {label}: bets={s['num_bets']} WR={s['win_rate']:.4f} "
                      f"pnl={s['net_pnl_bnb']:+.4f} fires={r['n_breaker_fires']} "
                      f"vetoes={r['n_vol_vetoes']} "
                      f"({r['elapsed_seconds']:.1f}s)", flush=True)

    # Curve samples for plotting
    curve_samples_5: dict[str, list[tuple[float, float]]] = {}
    curve_samples_50: dict[str, list[tuple[float, float]]] = {}
    sample_brs = (1.0, 2.0, 3.5, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                  35.0, 40.0, 50.0, 65.0, 75.0, 100.0)
    for name, fn in ADAPTIVE_CURVES.items():
        curve_samples_5[name] = [(br, fn(br, 5.0)) for br in sample_brs]
        curve_samples_50[name] = [(br, fn(br, 50.0)) for br in sample_brs]

    out_path = REPO / "var" / "strategy_review" / "adaptive_dd_step13b_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "vol_lookback_hours": VOL_LOOKBACK_HOURS,
                "vol_threshold_pct": VOL_THRESHOLD_PCT,
                "cooldown_rounds": COOLDOWN_ROUNDS,
                "drawdown_peak_window_days": DRAWDOWN_PEAK_WINDOW_DAYS,
                "cohort_defs": [list(c) for c in COHORT_DEFS],
                "curve_samples_5bnb_init": curve_samples_5,
                "curve_samples_50bnb_init": curve_samples_50,
            },
            "gate_pnl": gate_pnl,
            "gate_delta_vs_step12b": delta,
            "gate_pass": gate_pass,
            "results": results,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
