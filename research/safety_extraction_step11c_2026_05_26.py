"""Step 11c — Experiment C: cap sizing-bankroll while letting actual compound.

Mechanism:
  - Actual bankroll compounds normally (real PnL feeds back).
  - Drawdown breaker uses ACTUAL bankroll vs ACTUAL peak (unchanged).
  - For SIZING ONLY, the bankroll cap component uses min(actual, sizing_cap_bnb).
  - Pool-fraction cap + absolute cap still apply normally.

Implementation:
  Monkey-patch `pancakebot.strategy.momentum_pipeline._compute_bet_size` at
  module level to cap the `current_bankroll` argument before delegating to
  the original. The pipeline's drawdown / min-bankroll checks query
  `tracker.current_bankroll()` separately and are unaffected — they still
  see the real, compounding bankroll.

Sweep: sizing_cap ∈ {5, 10, 15, 20, 50}. 50 = baseline (since min(50, 50)=50
at initial bankroll, and actual bankroll grows slower than the cap is
binding). 5 = tightest cap (mirrors dyn5's sizing exactly while letting
actual bankroll compound for the drawdown breaker).

All runs at INITIAL_BANKROLL = 50 BNB. Real impact-aware settlement,
canonical (3, 7, 15) cs=2 across full 422298..484408.
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
import pancakebot.strategy.momentum_pipeline as _mp_module  # noqa: E402


# -- Monkey-patch _compute_bet_size to cap current_bankroll for sizing ----
_orig_compute_bet_size = _mp_module._compute_bet_size
_SIZING_CAP_BNB: float = float("inf")


def _patched_compute_bet_size(*, current_bankroll: float | None, **kwargs: Any) -> float:
    capped = current_bankroll
    if current_bankroll is not None and _SIZING_CAP_BNB < current_bankroll:
        capped = _SIZING_CAP_BNB
    return _orig_compute_bet_size(current_bankroll=capped, **kwargs)


_mp_module._compute_bet_size = _patched_compute_bet_size  # install patch


def set_sizing_cap(cap_bnb: float) -> None:
    global _SIZING_CAP_BNB
    _SIZING_CAP_BNB = float(cap_bnb)


# -- Config ---------------------------------------------------------------

EPOCH_MIN = 422298
EPOCH_MAX_CONFIG = 484999
CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2
INITIAL_BANKROLL = 50.0
TREASURY_FEE = 0.03
MIN_BET = 0.001

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


def parse_trades_to_cohorts(trades_csv: Path, initial_bankroll: float) -> tuple[dict, float]:
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
    for coh, cd in per_cohort.items():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0
        cd["mean_bet_size_bnb"] = (cd["total_bet_size_bnb"] / cd["n_bets"]
                                    if cd["n_bets"] else 0.0)
    return per_cohort, max_dd_frac


def run_variant(*, sizing_cap: float, all_rounds, btc, eth, sol,
                 earliest_offset, out_root) -> dict[str, Any]:
    set_sizing_cap(sizing_cap)
    overrides = {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
    sc = load_strategy_config_from_dict(overrides)
    spec = ipr.FoldSpec(
        name=f"expC_cap{int(sizing_cap)}",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        epoch_start=EPOCH_MIN, epoch_end=EPOCH_MAX_CONFIG,
        strategy_overrides=overrides,
    )
    t0 = time.time()
    summary = ipr.run_fold(
        spec=spec, strategy_cfg=sc,
        all_rounds=all_rounds, btc_unified=btc, eth_unified=eth, sol_unified=sol,
        earliest_offset=earliest_offset, output_base_dir=out_root,
        initial_bankroll_bnb=INITIAL_BANKROLL,
        treasury_fee_fraction=TREASURY_FEE, min_bet_amount_bnb=MIN_BET,
    )
    elapsed = time.time() - t0
    trades_csv = out_root / spec.name / "trades.csv"
    per_cohort, max_dd = parse_trades_to_cohorts(trades_csv, INITIAL_BANKROLL)
    print(f"  sizing_cap={sizing_cap}: bets={summary['num_bets']} "
          f"WR={summary['win_rate']:.4f} pnl={summary['net_pnl_bnb']:+.4f} "
          f"max_dd={max_dd*100:.2f}% ({elapsed:.1f}s)")
    return {
        "variant_label": f"sizing_cap={sizing_cap}",
        "sizing_cap_param": sizing_cap,
        "summary": {k: summary[k] for k in
                    ("num_bets", "num_wins", "win_rate", "net_pnl_bnb",
                     "final_bankroll_bnb")
                    if k in summary},
        "max_drawdown_realized_frac": max_dd,
        "per_cohort": per_cohort,
        "skip_counts": summary.get("skip_counts_by_reason", {}),
        "elapsed_seconds": elapsed,
    }


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
    print(f"  BTC={len(btc)} ETH={len(eth)} SOL={len(sol)}")

    out_root = Path(tempfile.mkdtemp(prefix="step11c_"))

    print(f"\n========== Experiment C: sizing_cap sweep @ {INITIAL_BANKROLL} BNB ==========")
    print(f"  sizing-bankroll = min(actual, sizing_cap); drawdown uses actual; "
          f"PnL compounds normally")
    sizing_caps = [5.0, 10.0, 15.0, 20.0, 50.0]
    results = []
    for cap in sizing_caps:
        r = run_variant(sizing_cap=cap, all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
                         earliest_offset=earliest_offset, out_root=out_root)
        results.append(r)

    out_path = REPO / "var" / "strategy_review" / "safety_extraction_step11c_data.json"
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
            "experiment_C_sizing_cap_sweep": results,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
