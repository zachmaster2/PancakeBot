"""Cutoff=3 re-tune candidate: lookbacks=(2,6,14).

Each canonical lookback reduced by 1 (3->2, 7->6, 15->14). Tests whether
the cutoff=3 architecture (deeper data window) recovers from the
catastrophic decline shown in cutoff_3_preliminary.py when given
slightly shorter lookbacks that better match the new anchor's recent
volatility regime.

Comparison frame: canonical baseline cutoff=2 + (3,7,15).

Usage::
    python research/cutoff_3_retune_2_6_14.py

Writes per-fold trades.csv + summary.json under
``var/cutoff_3_artifacts/{c2,c3_2_6_14}/<fold>/`` (gitignored).
Aggregated comparison emitted to stdout + comparison_2_6_14.json.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.in_process_runner import FoldSpec, run_experiment  # noqa: E402


_FOLDS = [
    {"name": "f1", "epoch_start": 437562, "epoch_end": 444866},
    {"name": "f2", "epoch_start": 444867, "epoch_end": 452171},
    {"name": "f3", "epoch_start": 452172, "epoch_end": 459476},
    {"name": "f4", "epoch_start": 459477, "epoch_end": 466781},
    {"name": "f5", "epoch_start": 466782, "epoch_end": 474086},
]
_HOLDOUT = {"name": "holdout", "epoch_start": 474880, "epoch_end": 475311}
_ALL_SPECS = _FOLDS + [_HOLDOUT]


def _build_specs(*, label: str, cutoff: int,
                 mtf_lookbacks: tuple[int, ...] | None = None) -> list[FoldSpec]:
    overrides: dict = {}
    if mtf_lookbacks is not None:
        overrides = {"gate": {"mtf_lookbacks": list(mtf_lookbacks)}}
    return [
        FoldSpec(
            name=f"{label}/{fold['name']}",
            kline_cutoff_seconds=cutoff,
            epoch_start=fold["epoch_start"],
            epoch_end=fold["epoch_end"],
            strategy_overrides=overrides,
            plot=False,
        )
        for fold in _ALL_SPECS
    ]


def _read_trades(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(row["epoch"])] = {
                "action": row["action"],
                "skip_reason": row["skip_reason"],
                "direction": row["direction"],
                "bet_size_bnb": float(row["bet_size_bnb"]),
                "profit_bnb": float(row["profit_bnb"]),
            }
    return out


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _diff_fold(c2: dict[int, dict], rt: dict[int, dict]) -> dict:
    common = sorted(set(c2.keys()) & set(rt.keys()))
    same_action_same_side = 0
    same_action_diff_side = 0
    c2_bet_rt_skip = 0
    c2_skip_rt_bet = 0
    pnl_c2_bet_rt_skip = 0.0
    pnl_c2_skip_rt_bet = 0.0
    pnl_side_flip = 0.0
    for ep in common:
        a = c2[ep]
        b = rt[ep]
        if a["action"] == "BET" and b["action"] == "BET":
            if a["direction"] == b["direction"]:
                same_action_same_side += 1
            else:
                same_action_diff_side += 1
                pnl_side_flip += b["profit_bnb"] - a["profit_bnb"]
        elif a["action"] == "BET" and b["action"] == "SKIP":
            c2_bet_rt_skip += 1
            pnl_c2_bet_rt_skip += a["profit_bnb"]
        elif a["action"] == "SKIP" and b["action"] == "BET":
            c2_skip_rt_bet += 1
            pnl_c2_skip_rt_bet += b["profit_bnb"]
        else:
            same_action_same_side += 1
    return {
        "common_round_count": len(common),
        "same_action_same_side": same_action_same_side,
        "same_action_diff_side": same_action_diff_side,
        "c2_bet_retune_skip": c2_bet_rt_skip,
        "c2_skip_retune_bet": c2_skip_rt_bet,
        "agreement_rate": same_action_same_side / len(common) if common else 0.0,
        "pnl_c2_bet_retune_skip_bnb": pnl_c2_bet_rt_skip,
        "pnl_c2_skip_retune_bet_bnb": pnl_c2_skip_rt_bet,
        "pnl_side_flip_delta_bnb": pnl_side_flip,
    }


def main() -> int:
    out_root = _REPO_ROOT / "var" / "cutoff_3_artifacts"
    out_root.mkdir(parents=True, exist_ok=True)

    print("Running cutoff=2 baseline (canonical lookbacks)...", flush=True)
    c2_specs = _build_specs(label="c2", cutoff=2)
    run_experiment(experiment_specs=c2_specs, output_base_dir=out_root)

    print("Running cutoff=3 with re-tuned lookbacks (2,6,14)...", flush=True)
    c3_specs = _build_specs(label="c3_2_6_14", cutoff=3, mtf_lookbacks=(2, 6, 14))
    run_experiment(experiment_specs=c3_specs, output_base_dir=out_root)

    by_fold: dict[str, dict] = {}
    for fold in _ALL_SPECS:
        name = fold["name"]
        c2_dir = out_root / "c2" / name
        rt_dir = out_root / "c3_2_6_14" / name
        c2_summary = _read_summary(c2_dir / "summary.json")
        rt_summary = _read_summary(rt_dir / "summary.json")
        c2_trades = _read_trades(c2_dir / "trades.csv")
        rt_trades = _read_trades(rt_dir / "trades.csv")
        diff = _diff_fold(c2_trades, rt_trades)
        by_fold[name] = {
            "c2": {
                "total_rounds": int(c2_summary["backtest_round_count"]),
                "num_bets": int(c2_summary["num_bets"]),
                "num_wins": int(c2_summary["num_wins"]),
                "win_rate": float(c2_summary["win_rate"]),
                "bet_rate": float(c2_summary["bet_rate"]),
                "net_pnl_bnb": float(c2_summary["net_pnl_bnb"]),
                "skip_counts_by_reason": dict(c2_summary["skip_counts_by_reason"]),
            },
            "retune": {
                "total_rounds": int(rt_summary["backtest_round_count"]),
                "num_bets": int(rt_summary["num_bets"]),
                "num_wins": int(rt_summary["num_wins"]),
                "win_rate": float(rt_summary["win_rate"]),
                "bet_rate": float(rt_summary["bet_rate"]),
                "net_pnl_bnb": float(rt_summary["net_pnl_bnb"]),
                "skip_counts_by_reason": dict(rt_summary["skip_counts_by_reason"]),
            },
            "diff": diff,
        }

    cv5 = {"c2_bets": 0, "c2_wins": 0, "c2_pnl": 0.0,
           "rt_bets": 0, "rt_wins": 0, "rt_pnl": 0.0,
           "rt_total_rounds": 0,
           "agreement_count": 0, "common_count": 0,
           "c2_bet_rt_skip": 0, "c2_skip_rt_bet": 0, "side_flip": 0,
           "pnl_c2_bet_rt_skip": 0.0, "pnl_c2_skip_rt_bet": 0.0,
           "pnl_side_flip": 0.0}
    for name in ("f1", "f2", "f3", "f4", "f5"):
        f = by_fold[name]
        cv5["c2_bets"] += f["c2"]["num_bets"]
        cv5["c2_wins"] += f["c2"]["num_wins"]
        cv5["c2_pnl"] += f["c2"]["net_pnl_bnb"]
        cv5["rt_bets"] += f["retune"]["num_bets"]
        cv5["rt_wins"] += f["retune"]["num_wins"]
        cv5["rt_pnl"] += f["retune"]["net_pnl_bnb"]
        cv5["rt_total_rounds"] += f["retune"]["total_rounds"]
        cv5["agreement_count"] += f["diff"]["same_action_same_side"]
        cv5["common_count"] += f["diff"]["common_round_count"]
        cv5["c2_bet_rt_skip"] += f["diff"]["c2_bet_retune_skip"]
        cv5["c2_skip_rt_bet"] += f["diff"]["c2_skip_retune_bet"]
        cv5["side_flip"] += f["diff"]["same_action_diff_side"]
        cv5["pnl_c2_bet_rt_skip"] += f["diff"]["pnl_c2_bet_retune_skip_bnb"]
        cv5["pnl_c2_skip_rt_bet"] += f["diff"]["pnl_c2_skip_retune_bet_bnb"]
        cv5["pnl_side_flip"] += f["diff"]["pnl_side_flip_delta_bnb"]

    print()
    print("=" * 100)
    print("CUTOFF=3 + (2,6,14) vs CANONICAL CUTOFF=2 + (3,7,15)")
    print("=" * 100)
    print()
    print(f"{'fold':>10}  {'cutoff=2 (3,7,15) [canonical]':>34}  {'cutoff=3 (2,6,14) [re-tune]':>32}")
    print("-" * 100)
    for name in ("f1", "f2", "f3", "f4", "f5", "holdout"):
        f = by_fold[name]
        c2 = f["c2"]
        rt = f["retune"]
        c2_str = (f"{c2['num_bets']:>5}/{c2['num_wins']:>4} "
                  f"{c2['net_pnl_bnb']:>+9.4f} ({c2['bet_rate']*100:>4.1f}%)")
        rt_str = (f"{rt['num_bets']:>5}/{rt['num_wins']:>4} "
                  f"{rt['net_pnl_bnb']:>+9.4f} ({rt['bet_rate']*100:>4.1f}%)")
        delta = rt['net_pnl_bnb'] - c2['net_pnl_bnb']
        print(f"{name:>10}  {c2_str:>34}  {rt_str:>32}   d={delta:>+8.4f}")
    print("-" * 100)
    print(f"{'CV5':>10}  {cv5['c2_bets']:>5}/{cv5['c2_wins']:>4} {cv5['c2_pnl']:>+9.4f}                    "
          f"{cv5['rt_bets']:>5}/{cv5['rt_wins']:>4} {cv5['rt_pnl']:>+9.4f}            "
          f"d={cv5['rt_pnl']-cv5['c2_pnl']:>+8.4f}")
    print()

    cv5_wr_c2 = cv5["c2_wins"] / cv5["c2_bets"] if cv5["c2_bets"] > 0 else 0.0
    cv5_wr_rt = cv5["rt_wins"] / cv5["rt_bets"] if cv5["rt_bets"] > 0 else 0.0
    print(f"CV5 win-rate: c2={cv5_wr_c2:.4f} ({cv5['c2_wins']}/{cv5['c2_bets']}), "
          f"retune={cv5_wr_rt:.4f} ({cv5['rt_wins']}/{cv5['rt_bets']})")
    folds_positive_c2 = sum(1 for n in ("f1","f2","f3","f4","f5")
                             if by_fold[n]['c2']['net_pnl_bnb'] > 0)
    folds_positive_rt = sum(1 for n in ("f1","f2","f3","f4","f5")
                             if by_fold[n]['retune']['net_pnl_bnb'] > 0)
    print(f"Folds positive: c2={folds_positive_c2}/5, retune={folds_positive_rt}/5")
    print()

    print("Per-round decision diff (CV5 aggregated):")
    print(f"  total common rounds:                          {cv5['common_count']:>6}")
    print(f"  c2 == retune (same action, same side):        {cv5['agreement_count']:>6}  "
          f"({cv5['agreement_count']/cv5['common_count']*100:.2f}%)")
    print(f"  c2 BET, retune SKIP:                          {cv5['c2_bet_rt_skip']:>6}  "
          f"PnL c2 made on those: {cv5['pnl_c2_bet_rt_skip']:+.4f} BNB")
    print(f"  c2 SKIP, retune BET:                          {cv5['c2_skip_rt_bet']:>6}  "
          f"PnL retune made on those: {cv5['pnl_c2_skip_rt_bet']:+.4f} BNB")
    print(f"  side flip (BET-Bull <-> BET-Bear):            {cv5['side_flip']:>6}  "
          f"net PnL delta: {cv5['pnl_side_flip']:+.4f} BNB")
    print()

    print("CV5 skip-reason distribution (counts):")
    c2_skips: Counter = Counter()
    rt_skips: Counter = Counter()
    for name in ("f1", "f2", "f3", "f4", "f5"):
        for k, v in by_fold[name]["c2"]["skip_counts_by_reason"].items():
            c2_skips[k] += v
        for k, v in by_fold[name]["retune"]["skip_counts_by_reason"].items():
            rt_skips[k] += v
    all_reasons = sorted(set(c2_skips) | set(rt_skips))
    print(f"  {'reason':>40}  {'cutoff=2':>10}  {'retune':>10}")
    print(f"  {'':>40}  {'-' * 10}  {'-' * 10}")
    for r in all_reasons:
        print(f"  {r:>40}  {c2_skips[r]:>10}  {rt_skips[r]:>10}")

    out_obj = {
        "lookback_variant": [2, 6, 14],
        "kline_cutoff_seconds": 3,
        "by_fold": by_fold,
        "cv5_aggregate": cv5,
        "cv5_skip_distribution": {
            "c2_canonical": dict(c2_skips),
            "retune_2_6_14": dict(rt_skips),
        },
    }
    out_json = out_root / "comparison_2_6_14.json"
    out_json.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print()
    print(f"Full JSON: {out_json.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
