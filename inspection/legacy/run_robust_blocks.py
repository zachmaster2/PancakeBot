from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: str
    columns_csv: str


_FULL_FEATURES_TOKEN = "__FULL__"


def _parse_candidate(spec: str) -> CandidateSpec:
    raw = str(spec).strip()
    if "=" not in raw:
        raise ValueError("candidate_must_be_name_equals_columns_csv")
    name, cols = raw.split("=", 1)
    name = str(name).strip()
    cols = str(cols).strip()
    if not name:
        raise ValueError("candidate_name_empty")
    if not cols:
        raise ValueError("candidate_columns_empty")
    return CandidateSpec(name=name, columns_csv=cols)


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument("--train-size", type=int, required=True)
    p.add_argument("--calibrate-size", type=int, required=True)
    p.add_argument("--rw-floor", type=float, required=True)
    p.add_argument("--rw-power", type=float, required=True)
    p.add_argument("--block-size", type=int, default=2000)
    p.add_argument("--num-blocks", type=int, default=5)
    p.add_argument("--skip-most-recent-blocks", type=int, default=0)
    p.add_argument("--direction-model-type", type=str, choices=("lgbm", "logistic"), default="lgbm")
    p.add_argument("--calibration-mode", type=str, choices=("isotonic", "raw", "platt"), default="raw")
    p.add_argument("--window-order", type=str, choices=("cal_train", "train_cal", "train_in_sample"), default="cal_train")
    p.add_argument("--direction-filter-mode", type=str, choices=("none", "bull_only", "bear_only", "both_sides"), default="both_sides")
    p.add_argument("--direction-threshold-mode", type=str, choices=("fixed", "quantile"), default="quantile")
    p.add_argument("--direction-threshold-bull", type=float, default=0.5)
    p.add_argument("--direction-threshold-bear", type=float, default=0.5)
    p.add_argument("--direction-target-bull-rate", type=float, default=0.04)
    p.add_argument("--direction-target-bear-rate", type=float, default=0.01)
    p.add_argument("--direction-threshold-window", type=int, default=300)
    p.add_argument("--direction-center-mode", type=str, choices=("fixed_0p5", "rolling_median", "rolling_mean"), default="rolling_median")
    p.add_argument("--direction-center-window", type=int, default=300)
    p.add_argument("--direction-edge-floor-pp", type=float, default=0.001)
    p.add_argument("--force-no-positive-ev", action="store_true", default=False)
    p.add_argument("--no-positive-ev-floor-bnb", type=float, default=None)
    p.add_argument("--ev-reliability-window", type=int, default=0)
    p.add_argument("--ev-reliability-min-bets", type=int, default=20)
    p.add_argument("--ev-reliability-quantile", type=float, default=0.70)
    p.add_argument("--ev-reliability-min-mean-profit", type=float, default=0.0)
    p.add_argument("--regime-filter", type=str, choices=("none", "pool_imbalance"), default="none")
    p.add_argument("--regime-min-imbalance", type=float, default=0.12)
    p.add_argument("--direction-viability-hard-fail", action="store_true", default=False)
    p.add_argument("--fixed-bet-bnb", type=float, default=None)
    p.add_argument("--winrate-only", action="store_true", default=False)
    p.add_argument("--candidate", action="append", default=[])
    return p


def _run_one(*, args, scenario_name: str, columns_csv: str, sim_offset_rounds: int) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "inspection.run_backtest_scenario",
        "--config",
        str(args.config),
        "--name",
        str(scenario_name),
        "--train-size",
        str(int(args.train_size)),
        "--calibrate-size",
        str(int(args.calibrate_size)),
        "--rw-floor",
        str(float(args.rw_floor)),
        "--rw-power",
        str(float(args.rw_power)),
        "--sim-size",
        str(int(args.block_size)),
        "--sim-offset-rounds",
        str(int(sim_offset_rounds)),
        "--direction-model-type",
        str(args.direction_model_type),
        "--calibration-mode",
        str(args.calibration_mode),
        "--window-order",
        str(args.window_order),
        "--direction-filter-mode",
        str(args.direction_filter_mode),
        "--direction-threshold-mode",
        str(args.direction_threshold_mode),
        "--direction-threshold-bull",
        str(float(args.direction_threshold_bull)),
        "--direction-threshold-bear",
        str(float(args.direction_threshold_bear)),
        "--direction-target-bull-rate",
        str(float(args.direction_target_bull_rate)),
        "--direction-target-bear-rate",
        str(float(args.direction_target_bear_rate)),
        "--direction-threshold-window",
        str(int(args.direction_threshold_window)),
        "--direction-center-mode",
        str(args.direction_center_mode),
        "--direction-center-window",
        str(int(args.direction_center_window)),
        "--direction-edge-floor-pp",
        str(float(args.direction_edge_floor_pp)),
        "--ev-reliability-window",
        str(int(args.ev_reliability_window)),
        "--ev-reliability-min-bets",
        str(int(args.ev_reliability_min_bets)),
        "--ev-reliability-quantile",
        str(float(args.ev_reliability_quantile)),
        "--ev-reliability-min-mean-profit",
        str(float(args.ev_reliability_min_mean_profit)),
        "--regime-filter",
        str(args.regime_filter),
        "--regime-min-imbalance",
        str(float(args.regime_min_imbalance)),
    ]
    if bool(args.force_no_positive_ev):
        cmd.append("--force-no-positive-ev")
    if args.no_positive_ev_floor_bnb is not None:
        cmd.extend(["--no-positive-ev-floor-bnb", str(float(args.no_positive_ev_floor_bnb))])
    if bool(args.direction_viability_hard_fail):
        cmd.append("--direction-viability-hard-fail")
    if str(columns_csv).strip() != _FULL_FEATURES_TOKEN:
        cmd.extend(["--sparse-probe-columns", str(columns_csv)])
    if args.fixed_bet_bnb is not None:
        cmd.extend(["--fixed-bet-bnb", str(float(args.fixed_bet_bnb))])
    if bool(args.winrate_only):
        cmd.append("--winrate-only")

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"scenario_failed: {scenario_name}")

    summary_path = Path("var/exp") / str(scenario_name) / "backtest_summary.json"
    trades_path = Path("var/exp") / str(scenario_name) / "backtest_trades.csv"
    summary = json.loads(summary_path.read_text())

    first_epoch = None
    last_epoch = None
    with trades_path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            epoch = int(row["epoch"])
            if first_epoch is None:
                first_epoch = int(epoch)
            last_epoch = int(epoch)

    return {
        "scenario": str(scenario_name),
        "sim_offset_rounds": int(sim_offset_rounds),
        "epoch_first": first_epoch if first_epoch is not None else -1,
        "epoch_last": last_epoch if last_epoch is not None else -1,
        "net": float(summary["net_profit_bnb"]),
        "bets": int(summary["num_bets"]),
        "wins": int(summary["num_wins"]),
        "bet_rate": float(summary["bet_rate"]),
        "win_rate": float(summary["win_rate"]),
        "final_bankroll": float(summary["final_bankroll_bnb"]),
        "stable": bool(summary.get("chunk_stability", {}).get("stability_ok", False)),
        "max_drawdown_bnb": float(summary.get("risk", {}).get("max_drawdown_bnb", 0.0)),
        "no_signal": int(summary.get("skip_reason_groups", {}).get("direction_filter_no_signal", 0)),
        "insufficient_edge": int(summary.get("skip_reason_groups", {}).get("insufficient_edge", 0)),
        "no_ev": int(summary.get("skip_reason_groups", {}).get("no_positive_ev", 0)),
        "side_mismatch": int(summary.get("skip_reason_groups", {}).get("direction_filter_side_mismatch", 0)),
    }


def _aggregate(*, rows: list[dict]) -> dict:
    nets = [float(r["net"]) for r in rows]
    bets = [int(r["bets"]) for r in rows]
    wins = [int(r["wins"]) for r in rows]
    bet_rates = [float(r["bet_rate"]) for r in rows]
    mdds = [float(r["max_drawdown_bnb"]) for r in rows]
    return {
        "blocks": int(len(rows)),
        "net_total": float(sum(nets)),
        "net_mean": float(sum(nets) / len(nets)),
        "net_median": float(statistics.median(nets)),
        "net_worst": float(min(nets)),
        "net_best": float(max(nets)),
        "positive_blocks": int(sum(1 for n in nets if float(n) > 0.0)),
        "positive_block_frac": float(sum(1 for n in nets if float(n) > 0.0) / len(nets)),
        "bets_total": int(sum(bets)),
        "wins_total": int(sum(wins)),
        "win_rate_weighted": float(_safe_rate(sum(wins), sum(bets))),
        "bet_rate_mean": float(sum(bet_rates) / len(bet_rates)),
        "max_drawdown_worst": float(max(mdds)),
        "stability_all_blocks": bool(all(bool(r["stable"]) for r in rows)),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if int(args.block_size) <= 0:
        raise ValueError("block_size_must_be_positive")
    if int(args.num_blocks) <= 0:
        raise ValueError("num_blocks_must_be_positive")
    if int(args.skip_most_recent_blocks) < 0:
        raise ValueError("skip_most_recent_blocks_must_be_nonnegative")
    if not args.candidate:
        raise ValueError("at_least_one_candidate_required")

    candidates = [_parse_candidate(spec) for spec in args.candidate]
    skip = int(args.skip_most_recent_blocks)
    offsets = [int(args.block_size) * i for i in range(int(args.num_blocks) + int(skip) - 1, int(skip) - 1, -1)]

    out_dir = Path("var/exp")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_block_rows: list[dict] = []
    agg_rows: list[dict] = []

    for cand in candidates:
        cand_rows: list[dict] = []
        for block_idx, offset in enumerate(offsets, start=1):
            scenario_name = (
                f"{args.name_prefix}_{cand.name}_"
                f"b{int(block_idx)}of{int(args.num_blocks)}_off{int(offset)}"
            )
            row = _run_one(
                args=args,
                scenario_name=str(scenario_name),
                columns_csv=str(cand.columns_csv),
                sim_offset_rounds=int(offset),
            )
            row["candidate"] = str(cand.name)
            row["features"] = (
                "(full_feature_schema)"
                if str(cand.columns_csv).strip() == _FULL_FEATURES_TOKEN
                else str(cand.columns_csv)
            )
            row["block_index"] = int(block_idx)
            cand_rows.append(row)
            all_block_rows.append(dict(row))
            print(
                "BLOCK_DONE "
                + f"candidate={cand.name} block={block_idx}/{args.num_blocks} "
                + f"offset={offset} net={row['net']} bets={row['bets']} win={row['win_rate']}"
            )

        agg = _aggregate(rows=cand_rows)
        agg["candidate"] = str(cand.name)
        agg["features"] = (
            "(full_feature_schema)"
            if str(cand.columns_csv).strip() == _FULL_FEATURES_TOKEN
            else str(cand.columns_csv)
        )
        agg_rows.append(agg)

    agg_rows_sorted = sorted(
        agg_rows,
        key=lambda r: (
            float(r["net_median"]),
            float(r["positive_block_frac"]),
            float(r["net_worst"]),
            float(r["net_total"]),
        ),
        reverse=True,
    )

    blocks_csv = out_dir / f"{args.name_prefix}_blocks.csv"
    agg_csv = out_dir / f"{args.name_prefix}_aggregate.csv"
    agg_json = out_dir / f"{args.name_prefix}_aggregate.json"

    with blocks_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "features",
                "block_index",
                "sim_offset_rounds",
                "epoch_first",
                "epoch_last",
                "scenario",
                "net",
                "bets",
                "wins",
                "bet_rate",
                "win_rate",
                "final_bankroll",
                "stable",
                "max_drawdown_bnb",
                "no_signal",
                "insufficient_edge",
                "no_ev",
                "side_mismatch",
            ],
        )
        w.writeheader()
        for row in all_block_rows:
            w.writerow(row)

    with agg_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "features",
                "blocks",
                "net_total",
                "net_mean",
                "net_median",
                "net_worst",
                "net_best",
                "positive_blocks",
                "positive_block_frac",
                "bets_total",
                "wins_total",
                "win_rate_weighted",
                "bet_rate_mean",
                "max_drawdown_worst",
                "stability_all_blocks",
            ],
        )
        w.writeheader()
        for row in agg_rows_sorted:
            w.writerow(row)

    agg_json.write_text(json.dumps({"aggregate": agg_rows_sorted, "blocks": all_block_rows}, indent=2, sort_keys=True))

    print(f"BLOCKS_CSV={blocks_csv}")
    print(f"AGG_CSV={agg_csv}")
    print(f"AGG_JSON={agg_json}")
    for idx, row in enumerate(agg_rows_sorted, start=1):
        print(
            "RANK "
            + f"{idx} candidate={row['candidate']} "
            + f"median_net={row['net_median']} "
            + f"positive_frac={row['positive_block_frac']} "
            + f"worst_net={row['net_worst']} "
            + f"total_net={row['net_total']}"
        )


if __name__ == "__main__":
    main()
