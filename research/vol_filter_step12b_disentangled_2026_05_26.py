"""Step 12b — disentangled vol-filter analysis.

Step 12 conflated (a) pure vol-filter effect on raw edge with (b) bankroll-
trajectory feedback into the drawdown breaker. The −9.43 BNB "perfect
filter hurts at 5 BNB" finding was a (b) effect — risk gates were tuned
for the unfiltered trajectory.

This rerun separates the effects into three layers:

  L1: Static bankroll (no compounding, no risk gates).
      Pure sum-of-per-bet PnL — measures raw edge.

  L2: Dynamic bankroll, drawdown breaker DISABLED (max_dd=1.0).
      Compounding active, no risk-gate veto — measures filter + compounding.

  L3: Dynamic + breaker re-tuned per filter (dd_frac ∈ {0.08, 0.15, 0.25}).
      Apples-to-apples: each filter gets its own optimal breaker setting.

Filter configs per layer:
  - baseline: no filter
  - vol_24h_thr30: vol filter (Step 12 best variant)
  - perfect_no_ext: extension cohort fully excluded

Scales: 5 BNB and 50 BNB.

Total: 6 (L1) + 6 (L2) + 18 (L3 = 3 configs × 2 scales × 3 dd_frac) = 30 backtests.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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
SCALES = (5.0, 50.0)
L3_DD_FRACS = (0.08, 0.15, 0.25)

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
    } for c in COHORT_ORDER}


# StaticBankrollTracker from Step 8 — freezes reported bankroll, no risk gates fire
class StaticBankrollTracker:
    def __init__(self, static_bankroll: float) -> None:
        self._static = float(static_bankroll)
    def current_bankroll(self) -> float: return self._static
    def peak_bankroll(self, start_at: int) -> float: return self._static  # noqa: ARG002
    def is_paused(self, start_at: int) -> bool: return False  # noqa: ARG002
    def tick_cooldown(self) -> None: pass
    def cooldown_remaining(self) -> int: return 0
    def set_paused(self, rounds: int, start_at: int) -> None: pass  # noqa: ARG002
    def record_settlement(self, bankroll: float, start_at: int) -> None: pass  # noqa: ARG002


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


def run_backtest(*, initial_bankroll: float, vol_cache: dict[int, float] | None,
                  vol_threshold: float, exclude_extension: bool,
                  use_static_tracker: bool, dd_frac_override: float | None,
                  all_rounds, btc_klines, eth_klines, sol_klines,
                  earliest_offset: int, label: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
    if dd_frac_override is not None:
        overrides["risk"] = {"max_drawdown_fraction_from_peak": float(dd_frac_override)}
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    if use_static_tracker:
        tracker: Any = StaticBankrollTracker(static_bankroll=initial_bankroll)
    else:
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

    sim_rounds = [r for r in all_rounds if EPOCH_MIN <= r.epoch <= EPOCH_MAX]
    per_cohort = empty_cohort_record()
    bankroll = float(initial_bankroll); peak = bankroll; max_dd_frac = 0.0
    n_vol_vetoes = 0

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        coh = cohort_of(ep)
        per_cohort[coh]["n_rounds"] += 1

        if exclude_extension and coh == "extension":
            per_cohort[coh]["skip_other"] += 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

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

        if not use_static_tracker:
            pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])

    for cd in per_cohort.values():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0
        cd["mean_bet_size_bnb"] = (cd["total_bet_size_bnb"] / cd["n_bets"]
                                    if cd["n_bets"] else 0.0)
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
        "per_cohort": per_cohort,
        "use_static_tracker": use_static_tracker,
        "dd_frac_override": dd_frac_override,
        "exclude_extension": exclude_extension,
        "has_vol_filter": vol_cache is not None,
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

    # Build vol cache
    print("\n--- building BTC last-close timeline + 24h vol cache ---", flush=True)
    btc_timeline: list[tuple[int, int, float]] = []
    for ep, kl in btc.items():
        if not kl:
            continue
        last_candle = kl[-1]
        ts_ms = int(last_candle[0])
        last_close = float(last_candle[4])
        if last_close > 0:
            btc_timeline.append((int(ep), ts_ms // 1000, last_close))
    btc_timeline.sort(key=lambda x: x[1])
    t_v = time.time()
    vol_cache = compute_vol_cache(btc_timeline, VOL_LOOKBACK_HOURS * 3600)
    print(f"  vol_{VOL_LOOKBACK_HOURS}h: {len(vol_cache)} epochs ({time.time()-t_v:.1f}s)", flush=True)

    # Filter configs:
    #   baseline:        no vol filter, no extension exclusion
    #   vol_24h_thr30:   vol filter active
    #   perfect_no_ext:  extension excluded
    filter_configs = [
        ("baseline", None, 0.0, False),
        ("vol_24h_thr30", vol_cache, VOL_THRESHOLD_PCT, False),
        ("perfect_no_ext", None, 0.0, True),
    ]

    results: list[dict[str, Any]] = []

    # ----- LAYER 1: Static bankroll, no risk gates -----
    print(f"\n========== LAYER 1: STATIC BANKROLL (pure edge) ==========", flush=True)
    for scale in SCALES:
        for cfg_name, vc, vt, exc in filter_configs:
            t = time.time()
            label = f"L1_static_{cfg_name}_{int(scale)}bnb"
            r = run_backtest(
                initial_bankroll=scale, vol_cache=vc, vol_threshold=vt,
                exclude_extension=exc, use_static_tracker=True,
                dd_frac_override=None,
                all_rounds=all_rounds, btc_klines=btc_klines,
                eth_klines=eth_klines, sol_klines=sol_klines,
                earliest_offset=earliest_offset, label=label,
            )
            r["layer"] = "L1"; r["scale"] = scale; r["config"] = cfg_name
            r["elapsed_seconds"] = time.time() - t
            results.append(r)
            s = r["summary"]
            print(f"  L1 {cfg_name} @ {scale}: bets={s['num_bets']} WR={s['win_rate']:.4f} "
                  f"pnl={s['net_pnl_bnb']:+.4f} vetoes={r['n_vol_vetoes']} "
                  f"({r['elapsed_seconds']:.1f}s)", flush=True)

    # ----- LAYER 2: Dynamic bankroll, drawdown breaker disabled (dd_frac=1.0) -----
    print(f"\n========== LAYER 2: DYNAMIC, BREAKER DISABLED ==========", flush=True)
    for scale in SCALES:
        for cfg_name, vc, vt, exc in filter_configs:
            t = time.time()
            label = f"L2_dyn_no_breaker_{cfg_name}_{int(scale)}bnb"
            r = run_backtest(
                initial_bankroll=scale, vol_cache=vc, vol_threshold=vt,
                exclude_extension=exc, use_static_tracker=False,
                dd_frac_override=1.0,
                all_rounds=all_rounds, btc_klines=btc_klines,
                eth_klines=eth_klines, sol_klines=sol_klines,
                earliest_offset=earliest_offset, label=label,
            )
            r["layer"] = "L2"; r["scale"] = scale; r["config"] = cfg_name
            r["elapsed_seconds"] = time.time() - t
            results.append(r)
            s = r["summary"]
            print(f"  L2 {cfg_name} @ {scale}: bets={s['num_bets']} WR={s['win_rate']:.4f} "
                  f"pnl={s['net_pnl_bnb']:+.4f} vetoes={r['n_vol_vetoes']} "
                  f"({r['elapsed_seconds']:.1f}s)", flush=True)

    # ----- LAYER 3: Dynamic, breaker dd_frac swept per config -----
    print(f"\n========== LAYER 3: DYNAMIC, BREAKER TUNED ==========", flush=True)
    for scale in SCALES:
        for cfg_name, vc, vt, exc in filter_configs:
            for dd in L3_DD_FRACS:
                t = time.time()
                label = f"L3_dd{int(dd*100):02d}_{cfg_name}_{int(scale)}bnb"
                r = run_backtest(
                    initial_bankroll=scale, vol_cache=vc, vol_threshold=vt,
                    exclude_extension=exc, use_static_tracker=False,
                    dd_frac_override=dd,
                    all_rounds=all_rounds, btc_klines=btc_klines,
                    eth_klines=eth_klines, sol_klines=sol_klines,
                    earliest_offset=earliest_offset, label=label,
                )
                r["layer"] = "L3"; r["scale"] = scale; r["config"] = cfg_name
                r["dd_frac"] = dd
                r["elapsed_seconds"] = time.time() - t
                results.append(r)
                s = r["summary"]
                print(f"  L3 dd={dd} {cfg_name} @ {scale}: bets={s['num_bets']} "
                      f"WR={s['win_rate']:.4f} pnl={s['net_pnl_bnb']:+.4f} "
                      f"vetoes={r['n_vol_vetoes']} ({r['elapsed_seconds']:.1f}s)",
                      flush=True)

    out_path = REPO / "var" / "strategy_review" / "vol_filter_step12b_disentangled_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "vol_lookback_hours": VOL_LOOKBACK_HOURS,
                "vol_threshold_pct": VOL_THRESHOLD_PCT,
                "scales": list(SCALES),
                "l3_dd_fracs": list(L3_DD_FRACS),
                "cohort_defs": [list(c) for c in COHORT_DEFS],
            },
            "results": results,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
