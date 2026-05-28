"""Step 12 — volatility filter backtest.

Vol metric: rolling realized vol on BTC last-close prices (one per 5-min
round), computed per-round at point-in-time (no lookahead). Annualized
via sqrt(288 × 365).

Sweep:
  lookback  ∈ {6h, 12h, 24h}      (288/round = 1 day; lookback is hours)
  threshold ∈ {30, 40, 50, 60, 70, 80}  (annualized vol %)
  scale     ∈ {5 BNB, 50 BNB}

Filter logic: at each round's bet decision, if the pipeline says BET and
rolling vol < threshold, veto → skip. Otherwise take the bet.

Reference rows: baseline (no filter) + perfect filter (extension-excluded).

Per-cohort breakdown via 7-bucket scheme including gap_post_cv5_pre_holdout.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import time
from datetime import datetime, timezone
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

LOOKBACK_HOURS = (6, 12, 24)
THRESHOLDS = (30.0, 40.0, 50.0, 60.0, 70.0, 80.0)  # annualized vol %
SCALES = (5.0, 50.0)

# 7-bucket cohort scheme
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


def stream_last_close_with_ts(path: Path) -> dict[int, tuple[int, float]]:
    """Stream-read JSONL, extract {epoch: (ts_last_candle, last_close_price)}."""
    out: dict[int, tuple[int, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("error") or rec.get("klines_1s") is None:
                continue
            kl = rec["klines_1s"]
            if not kl:
                continue
            last_candle = kl[-1]
            ts_ms = int(last_candle[0])
            last_close = float(last_candle[4])
            if last_close <= 0:
                continue
            out[int(rec["epoch"])] = (ts_ms // 1000, last_close)
    return out


def compute_vol_cache(btc_timeline: list[tuple[int, int, float]],
                       lookback_seconds: int) -> dict[int, float]:
    """Per-epoch annualized rolling vol from BTC last-close timeline.
    timeline: sorted list of (epoch, ts, close).

    For each entry i at ts_i, find all earlier entries with ts in
    [ts_i - lookback_seconds, ts_i - cutoff_seconds_buffer], compute log
    returns std, annualize.

    Returns {epoch: vol_pct} (None/missing if insufficient samples).
    """
    if not btc_timeline:
        return {}
    epochs = np.array([r[0] for r in btc_timeline])
    ts = np.array([r[1] for r in btc_timeline])
    closes = np.array([r[2] for r in btc_timeline])
    n = len(ts)
    log_closes = np.log(closes)
    # log returns
    log_returns = np.diff(log_closes)
    # Each log_returns[i] corresponds to the transition from ts[i] to ts[i+1].
    # For vol at round i (lock at ts[i]), we want log_returns up to (but not
    # including) the transition INTO ts[i] (which is log_returns[i-1]).
    # Actually we want returns BETWEEN consecutive prior rounds; use returns
    # with end-ts < ts[i] - cutoff. Use end-ts of log_returns[j] = ts[j+1].
    end_ts_of_returns = ts[1:]

    out: dict[int, float] = {}
    PER_YEAR_5MIN = 288 * 365  # 288 5-min periods/day × 365
    for i in range(n):
        target_ts = ts[i]
        cutoff_low = target_ts - lookback_seconds
        cutoff_high = target_ts - 2  # cutoff_seconds=2; exclude returns AT/AFTER lock
        # Find returns with end_ts in [cutoff_low, cutoff_high]
        idx_lo = np.searchsorted(end_ts_of_returns, cutoff_low, side="left")
        idx_hi = np.searchsorted(end_ts_of_returns, cutoff_high, side="right")
        if idx_hi - idx_lo < 3:
            continue  # need at least 3 returns
        window = log_returns[idx_lo:idx_hi]
        sd = float(np.std(window, ddof=1))
        vol_ann_pct = sd * math.sqrt(PER_YEAR_5MIN) * 100.0
        out[int(epochs[i])] = vol_ann_pct
    return out


def run_backtest_with_vol_filter(*, initial_bankroll: float, vol_cache: dict[int, float] | None,
                                    threshold: float, all_rounds, btc_klines, eth_klines,
                                    sol_klines, earliest_offset: int,
                                    exclude_extension: bool = False) -> dict[str, Any]:
    """Run canonical (3,7,15) cs=2 with optional vol-filter veto.

    If vol_cache is None, no filter (baseline).
    If exclude_extension=True, additionally skip all extension-cohort epochs
    (perfect-filter ceiling).
    """
    overrides = {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
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
    n_filter_vetoes = 0

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        coh = cohort_of(ep)
        per_cohort[coh]["n_rounds"] += 1

        # Perfect-filter shortcut: skip all extension epochs
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

        # Vol filter veto check
        if vol_cache is not None:
            vol = vol_cache.get(ep)
            if vol is None or vol < threshold:
                n_filter_vetoes += 1
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
    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())
    return {
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins/total_bets if total_bets else 0.0,
            "net_pnl_bnb": bankroll - initial_bankroll,
            "final_bankroll": bankroll,
        },
        "max_drawdown_frac": max_dd_frac,
        "n_vol_vetoes": n_filter_vetoes,
        "per_cohort": per_cohort,
    }


def main():
    t_all = time.time()

    print("--- loading rounds + klines ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  {len(all_rounds)} rounds; epoch range "
          f"[{all_rounds[0].epoch}..{max(r.epoch for r in all_rounds)}]", flush=True)
    epoch_to_ts = {int(r.epoch): int(r.start_at) for r in all_rounds}

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1

    t_kl = time.time()
    print("  loading BTC klines (unified)...", flush=True)
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

    # Slice klines per-entry for pipeline
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

    # Build BTC last-close timeline from the unified arrays
    print("\n--- building BTC last-close timeline + vol caches ---", flush=True)
    t_v = time.time()
    btc_timeline: list[tuple[int, int, float]] = []  # (epoch, ts, last_close)
    for ep, kl in btc.items():
        if not kl:
            continue
        last_candle = kl[-1]
        ts_ms = int(last_candle[0])
        last_close = float(last_candle[4])
        if last_close > 0:
            btc_timeline.append((int(ep), ts_ms // 1000, last_close))
    btc_timeline.sort(key=lambda x: x[1])
    print(f"  BTC timeline: {len(btc_timeline)} entries", flush=True)

    vol_caches: dict[int, dict[int, float]] = {}
    for hours in LOOKBACK_HOURS:
        t_v_l = time.time()
        vol_caches[hours] = compute_vol_cache(btc_timeline, hours * 3600)
        print(f"  vol_{hours}h: {len(vol_caches[hours])} epochs covered "
              f"({time.time()-t_v_l:.1f}s)", flush=True)
    print(f"  vol precompute total: {time.time()-t_v:.1f}s", flush=True)

    # Run backtests
    results: list[dict[str, Any]] = []

    for scale in SCALES:
        print(f"\n========== SCALE: {scale} BNB ==========", flush=True)
        # Baseline (no filter)
        t = time.time()
        r = run_backtest_with_vol_filter(
            initial_bankroll=scale, vol_cache=None, threshold=0.0,
            all_rounds=all_rounds, btc_klines=btc_klines, eth_klines=eth_klines,
            sol_klines=sol_klines, earliest_offset=earliest_offset,
        )
        r["variant_label"] = "baseline_no_filter"
        r["scale"] = scale; r["lookback"] = None; r["threshold"] = None
        r["elapsed_seconds"] = time.time() - t
        results.append(r)
        print(f"  baseline (no filter): bets={r['summary']['num_bets']} "
              f"WR={r['summary']['win_rate']:.4f} "
              f"pnl={r['summary']['net_pnl_bnb']:+.4f} "
              f"({r['elapsed_seconds']:.1f}s)", flush=True)
        # Perfect filter (extension excluded)
        t = time.time()
        r = run_backtest_with_vol_filter(
            initial_bankroll=scale, vol_cache=None, threshold=0.0,
            all_rounds=all_rounds, btc_klines=btc_klines, eth_klines=eth_klines,
            sol_klines=sol_klines, earliest_offset=earliest_offset,
            exclude_extension=True,
        )
        r["variant_label"] = "perfect_filter_no_extension"
        r["scale"] = scale; r["lookback"] = None; r["threshold"] = None
        r["elapsed_seconds"] = time.time() - t
        results.append(r)
        print(f"  perfect filter (no ext): bets={r['summary']['num_bets']} "
              f"WR={r['summary']['win_rate']:.4f} "
              f"pnl={r['summary']['net_pnl_bnb']:+.4f} "
              f"({r['elapsed_seconds']:.1f}s)", flush=True)

        # Vol filter sweep
        for lookback in LOOKBACK_HOURS:
            for thr in THRESHOLDS:
                t = time.time()
                r = run_backtest_with_vol_filter(
                    initial_bankroll=scale, vol_cache=vol_caches[lookback],
                    threshold=thr,
                    all_rounds=all_rounds, btc_klines=btc_klines, eth_klines=eth_klines,
                    sol_klines=sol_klines, earliest_offset=earliest_offset,
                )
                r["variant_label"] = f"vol_{lookback}h_thr{int(thr)}"
                r["scale"] = scale; r["lookback"] = lookback; r["threshold"] = thr
                r["elapsed_seconds"] = time.time() - t
                results.append(r)
                veto_rate = (r["n_vol_vetoes"] /
                             (r["summary"]["num_bets"] + r["n_vol_vetoes"])
                             if (r["summary"]["num_bets"] + r["n_vol_vetoes"]) > 0
                             else 0.0)
                print(f"  vol_{lookback}h thr={thr}: bets={r['summary']['num_bets']} "
                      f"WR={r['summary']['win_rate']:.4f} "
                      f"pnl={r['summary']['net_pnl_bnb']:+.4f} "
                      f"vetoes={r['n_vol_vetoes']} ({veto_rate*100:.1f}%) "
                      f"({r['elapsed_seconds']:.1f}s)", flush=True)

    out_path = REPO / "var" / "strategy_review" / "vol_filter_step12_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "lookback_hours": list(LOOKBACK_HOURS),
                "thresholds_pct": list(THRESHOLDS),
                "scales": list(SCALES),
                "cohort_defs": [list(c) for c in COHORT_DEFS],
            },
            "results": results,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
