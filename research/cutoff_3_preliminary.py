"""Cutoff=3 vs cutoff=2 preliminary backtest with per-round decision diff.

Runs the canonical 5-fold + holdout twice (cutoff=2 baseline + cutoff=3
with the same (3,7,15) lookbacks) and emits a side-by-side comparison
covering:

- Per-fold totals (bets/wins/PnL) — the headline.
- Signal-fire distribution (BET vs SKIP rate; skip-reason breakdown).
- Per-round decision diff: for every round, did c2 and c3 agree on
  action+side, or did they diverge (and how did the divergence outcome)?

Usage::
    python research/cutoff_3_preliminary.py

Writes JSON output + a Markdown table to stdout. Per-fold trades.csv
files land under ``var/cutoff_3_artifacts/<run>/<fold>/`` (gitignored).
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


def _build_specs(out_dir: Path, *, label: str, cutoff: int) -> list[FoldSpec]:
    return [
        FoldSpec(
            name=f"{label}/{fold['name']}",
            kline_cutoff_seconds=cutoff,
            epoch_start=fold["epoch_start"],
            epoch_end=fold["epoch_end"],
            strategy_overrides={},
            plot=False,
        )
        for fold in _ALL_SPECS
    ]


def _read_trades(path: Path) -> dict[int, dict]:
    """Map epoch -> {action, skip_reason, direction, bet_size_bnb, profit_bnb}."""
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            out[epoch] = {
                "action": row["action"],
                "skip_reason": row["skip_reason"],
                "direction": row["direction"],
                "bet_size_bnb": float(row["bet_size_bnb"]),
                "profit_bnb": float(row["profit_bnb"]),
            }
    return out


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _decision_label(rec: dict) -> str:
    """Compact label per round: BET-Bull / BET-Bear / SKIP-<reason>."""
    if rec["action"] == "BET":
        return f"BET-{rec['direction']}"
    return f"SKIP-{rec['skip_reason']}"


def _diff_fold(c2: dict[int, dict], c3: dict[int, dict]) -> dict:
    """Compute per-round decision diff between two trades dicts."""
    common = sorted(set(c2.keys()) & set(c3.keys()))
    if len(common) != len(c2) or len(common) != len(c3):
        # Should be equal — both runs use the same fold range.
        # Surface mismatched epochs as a sanity check.
        only_c2 = sorted(set(c2.keys()) - set(c3.keys()))
        only_c3 = sorted(set(c3.keys()) - set(c2.keys()))
        if only_c2 or only_c3:
            print(f"  WARN: epoch mismatch only_c2={len(only_c2)} only_c3={len(only_c3)}",
                  file=sys.stderr)

    same_action_same_side = 0      # both BET-Bull, both BET-Bear, or both SKIP-anything
    same_action_diff_side = 0       # c2 BET-Bull, c3 BET-Bear (or reverse)
    c2_bet_c3_skip = 0              # c2 fired, c3 stayed out
    c2_skip_c3_bet = 0              # c2 stayed out, c3 fired
    same_skip_diff_reason = 0       # both SKIP but different reason

    pnl_when_disagree_c2_bet_c3_skip = 0.0
    pnl_when_disagree_c2_skip_c3_bet = 0.0
    pnl_when_disagree_side_flip = 0.0  # net (c3.profit - c2.profit)

    for epoch in common:
        a = c2[epoch]
        b = c3[epoch]
        if a["action"] == "BET" and b["action"] == "BET":
            if a["direction"] == b["direction"]:
                same_action_same_side += 1
            else:
                same_action_diff_side += 1
                pnl_when_disagree_side_flip += b["profit_bnb"] - a["profit_bnb"]
        elif a["action"] == "BET" and b["action"] == "SKIP":
            c2_bet_c3_skip += 1
            pnl_when_disagree_c2_bet_c3_skip += a["profit_bnb"]  # what c2 earned that c3 skipped
        elif a["action"] == "SKIP" and b["action"] == "BET":
            c2_skip_c3_bet += 1
            pnl_when_disagree_c2_skip_c3_bet += b["profit_bnb"]  # what c3 earned that c2 skipped
        else:
            # both SKIP
            if a["skip_reason"] == b["skip_reason"]:
                same_action_same_side += 1
            else:
                same_skip_diff_reason += 1

    return {
        "common_round_count": len(common),
        "same_action_same_side": same_action_same_side,
        "same_action_diff_side": same_action_diff_side,
        "c2_bet_c3_skip": c2_bet_c3_skip,
        "c2_skip_c3_bet": c2_skip_c3_bet,
        "same_skip_diff_reason": same_skip_diff_reason,
        "agreement_rate": (
            same_action_same_side / len(common) if common else 0.0
        ),
        "pnl_c2_bet_c3_skip_bnb": pnl_when_disagree_c2_bet_c3_skip,
        "pnl_c2_skip_c3_bet_bnb": pnl_when_disagree_c2_skip_c3_bet,
        "pnl_side_flip_delta_bnb": pnl_when_disagree_side_flip,
    }


def main() -> int:
    out_root = _REPO_ROOT / "var" / "cutoff_3_artifacts"
    out_root.mkdir(parents=True, exist_ok=True)

    print("Running cutoff=2 baseline...", flush=True)
    c2_specs = _build_specs(out_root, label="c2", cutoff=2)
    run_experiment(experiment_specs=c2_specs, output_base_dir=out_root)

    print("Running cutoff=3 preliminary...", flush=True)
    c3_specs = _build_specs(out_root, label="c3", cutoff=3)
    run_experiment(experiment_specs=c3_specs, output_base_dir=out_root)

    # Aggregate per fold.
    by_fold: dict[str, dict] = {}
    for fold in _ALL_SPECS:
        name = fold["name"]
        c2_dir = out_root / "c2" / name
        c3_dir = out_root / "c3" / name
        c2_summary = _read_summary(c2_dir / "summary.json")
        c3_summary = _read_summary(c3_dir / "summary.json")
        c2_trades = _read_trades(c2_dir / "trades.csv")
        c3_trades = _read_trades(c3_dir / "trades.csv")
        diff = _diff_fold(c2_trades, c3_trades)

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
            "c3": {
                "total_rounds": int(c3_summary["backtest_round_count"]),
                "num_bets": int(c3_summary["num_bets"]),
                "num_wins": int(c3_summary["num_wins"]),
                "win_rate": float(c3_summary["win_rate"]),
                "bet_rate": float(c3_summary["bet_rate"]),
                "net_pnl_bnb": float(c3_summary["net_pnl_bnb"]),
                "skip_counts_by_reason": dict(c3_summary["skip_counts_by_reason"]),
            },
            "diff": diff,
        }

    # CV5 aggregate.
    cv5 = {"c2_bets": 0, "c2_wins": 0, "c2_pnl": 0.0, "c2_total_rounds": 0,
           "c3_bets": 0, "c3_wins": 0, "c3_pnl": 0.0, "c3_total_rounds": 0,
           "agreement_count": 0, "common_count": 0,
           "c2_bet_c3_skip": 0, "c2_skip_c3_bet": 0, "side_flip": 0,
           "pnl_c2_bet_c3_skip": 0.0, "pnl_c2_skip_c3_bet": 0.0,
           "pnl_side_flip_delta": 0.0}
    for name in ("f1", "f2", "f3", "f4", "f5"):
        f = by_fold[name]
        cv5["c2_bets"] += f["c2"]["num_bets"]
        cv5["c2_wins"] += f["c2"]["num_wins"]
        cv5["c2_pnl"] += f["c2"]["net_pnl_bnb"]
        cv5["c2_total_rounds"] += f["c2"]["total_rounds"]
        cv5["c3_bets"] += f["c3"]["num_bets"]
        cv5["c3_wins"] += f["c3"]["num_wins"]
        cv5["c3_pnl"] += f["c3"]["net_pnl_bnb"]
        cv5["c3_total_rounds"] += f["c3"]["total_rounds"]
        cv5["agreement_count"] += f["diff"]["same_action_same_side"]
        cv5["common_count"] += f["diff"]["common_round_count"]
        cv5["c2_bet_c3_skip"] += f["diff"]["c2_bet_c3_skip"]
        cv5["c2_skip_c3_bet"] += f["diff"]["c2_skip_c3_bet"]
        cv5["side_flip"] += f["diff"]["same_action_diff_side"]
        cv5["pnl_c2_bet_c3_skip"] += f["diff"]["pnl_c2_bet_c3_skip_bnb"]
        cv5["pnl_c2_skip_c3_bet"] += f["diff"]["pnl_c2_skip_c3_bet_bnb"]
        cv5["pnl_side_flip_delta"] += f["diff"]["pnl_side_flip_delta_bnb"]

    print()
    print("=" * 100)
    print("CUTOFF=3 vs CUTOFF=2 -- per-round decision diff over canonical CV5 + holdout")
    print("=" * 100)
    print()

    # Headline table.
    print("Per-fold (bets / wins / PnL / bet-rate):")
    print(f"{'fold':>10}  {'cutoff=2':>40}  {'cutoff=3':>40}")
    print("-" * 100)
    for name in ("f1", "f2", "f3", "f4", "f5", "holdout"):
        f = by_fold[name]
        c2 = f["c2"]
        c3 = f["c3"]
        c2_str = (f"{c2['num_bets']:>5}/{c2['num_wins']:>4} "
                  f"{c2['net_pnl_bnb']:>+9.4f} BNB "
                  f"({c2['bet_rate']*100:>5.2f}%/{c2['total_rounds']})")
        c3_str = (f"{c3['num_bets']:>5}/{c3['num_wins']:>4} "
                  f"{c3['net_pnl_bnb']:>+9.4f} BNB "
                  f"({c3['bet_rate']*100:>5.2f}%/{c3['total_rounds']})")
        print(f"{name:>10}  {c2_str:>40}  {c3_str:>40}")

    print()
    print("Per-round decision diff (CV5 aggregated):")
    print(f"  total common rounds:                          {cv5['common_count']:>6}")
    print(f"  c2 == c3 (same action, same side):            {cv5['agreement_count']:>6}  "
          f"({cv5['agreement_count']/cv5['common_count']*100:.2f}%)")
    print(f"  c2 BET, c3 SKIP (c3 stayed out):              {cv5['c2_bet_c3_skip']:>6}  "
          f"PnL c2 made on those: {cv5['pnl_c2_bet_c3_skip']:+.4f} BNB")
    print(f"  c2 SKIP, c3 BET (c3 fired):                   {cv5['c2_skip_c3_bet']:>6}  "
          f"PnL c3 made on those: {cv5['pnl_c2_skip_c3_bet']:+.4f} BNB")
    print(f"  c2 BET-Bull, c3 BET-Bear (or reverse) flip:   {cv5['side_flip']:>6}  "
          f"net PnL delta (c3-c2): {cv5['pnl_side_flip_delta']:+.4f} BNB")
    print()

    # Skip-reason breakdown.
    print("CV5 skip-reason distribution (counts):")
    c2_skips: Counter = Counter()
    c3_skips: Counter = Counter()
    for name in ("f1", "f2", "f3", "f4", "f5"):
        for k, v in by_fold[name]["c2"]["skip_counts_by_reason"].items():
            c2_skips[k] += v
        for k, v in by_fold[name]["c3"]["skip_counts_by_reason"].items():
            c3_skips[k] += v
    all_reasons = sorted(set(c2_skips) | set(c3_skips))
    print(f"  {'reason':>40}  {'cutoff=2':>10}  {'cutoff=3':>10}")
    print(f"  {'':>40}  {'-' * 10}  {'-' * 10}")
    for r in all_reasons:
        print(f"  {r:>40}  {c2_skips[r]:>10}  {c3_skips[r]:>10}")

    # JSON dump.
    out = {
        "by_fold": by_fold,
        "cv5_aggregate": cv5,
        "cv5_skip_distribution": {
            "c2": dict(c2_skips),
            "c3": dict(c3_skips),
        },
    }
    out_json = out_root / "comparison.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"Full JSON written to {out_json.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
