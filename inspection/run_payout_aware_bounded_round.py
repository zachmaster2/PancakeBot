from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pancakebot.core.errors import InvariantError

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class EvalRef:
    summary_json_path: str
    rows_csv_path: str
    trace_csv_path: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument(
        "--manifest-csv",
        type=str,
        default="../PancakeBot_var_exp/direction_ensemble_longstream_manifest_20260403.csv",
    )
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument(
        "--direction-sources",
        type=str,
        default="mlp,catboost,lightgbm,tcn,soft_mean_all,mean2_mlp_catboost",
    )
    parser.add_argument("--source-train-sizes", type=str, default="100000,200000,400000")
    parser.add_argument("--stake-sizes", type=str, default="0.05,0.10,0.30,0.50")
    parser.add_argument("--sim-size", type=int, default=50000)
    parser.add_argument("--valid-size", type=int, default=3000)
    parser.add_argument("--tail-offset-rounds", type=int, default=0)
    parser.add_argument("--robustness-offsets", type=str, default="0,5000,10000,15000")
    parser.add_argument("--payout-model-types", type=str, default="catboost")
    parser.add_argument("--target-mode", type=str, default="win_profit_residual")
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default="-0.020,-0.010,-0.005,0.000,0.001,0.0025,0.005,0.010,0.020",
    )
    parser.add_argument("--valid-min-bet-rate", type=float, default=0.005)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    parser.add_argument("--current-bar-net-bnb", type=float, default=3.7952054686396153)
    parser.add_argument("--current-bar-per500", type=float, default=0.037952054686396154)
    parser.add_argument("--current-bar-max-dd-bnb", type=float, default=3.216061216538158)
    return parser


def _parse_str_list(raw: str) -> list[str]:
    values = [str(token).strip() for token in str(raw).split(",") if str(token).strip() != ""]
    if not values:
        raise InvariantError("payout_bounded_round_str_list_empty")
    return values


def _parse_positive_int_list(raw: str) -> list[int]:
    out = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = int(text)
        if int(value) <= 0:
            raise InvariantError("payout_bounded_round_nonpositive_int")
        out.append(int(value))
    if not out:
        raise InvariantError("payout_bounded_round_int_list_empty")
    return out


def _parse_nonnegative_int_list(raw: str) -> list[int]:
    out = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = int(text)
        if int(value) < 0:
            raise InvariantError("payout_bounded_round_negative_int")
        out.append(int(value))
    if not out:
        raise InvariantError("payout_bounded_round_nonnegative_int_list_empty")
    return out


def _parse_float_list(raw: str) -> list[float]:
    out = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        out.append(float(text))
    if not out:
        raise InvariantError("payout_bounded_round_float_list_empty")
    return out


def _run_eval(
    *,
    python_exe: Path,
    cwd: Path,
    config_path: Path,
    manifest_csv: Path,
    output_dir: Path,
    name_prefix: str,
    direction_source: str,
    train_sizes: list[int],
    bet_size_bnb: float,
    tail_offset_rounds: int,
    payout_model_types: str,
    target_mode: str,
    sim_size: int,
    valid_size: int,
    threshold_grid: str,
    valid_min_bet_rate: float,
) -> EvalRef:
    summary_json_path = (output_dir / f"{name_prefix}_payout_aware_policy_summary.json").resolve()
    rows_csv_path = (output_dir / f"{name_prefix}_payout_aware_policy_rows.csv").resolve()
    trace_csv_path = (output_dir / f"{name_prefix}_payout_aware_policy_trace_rows.csv").resolve()
    if summary_json_path.exists() and rows_csv_path.exists() and trace_csv_path.exists():
        return EvalRef(
            summary_json_path=str(summary_json_path),
            rows_csv_path=str(rows_csv_path),
            trace_csv_path=str(trace_csv_path),
        )
    cmd = [
        str(python_exe),
        "-m",
        "inspection.run_payout_aware_policy_eval",
        "--config",
        str(config_path),
        "--name-prefix",
        str(name_prefix),
        "--manifest-csv",
        str(manifest_csv),
        "--payout-model-types",
        str(payout_model_types),
        "--target-mode",
        str(target_mode),
        "--direction-source",
        str(direction_source),
        "--train-sizes",
        ",".join(str(int(value)) for value in train_sizes),
        "--sim-size",
        str(int(sim_size)),
        "--valid-size",
        str(int(valid_size)),
        "--tail-offset-rounds",
        str(int(tail_offset_rounds)),
        "--bet-size-bnb",
        str(float(bet_size_bnb)),
        f"--threshold-grid={str(threshold_grid)}",
        "--valid-min-bet-rate",
        str(float(valid_min_bet_rate)),
    ]
    subprocess.run(cmd, cwd=str(cwd), check=True)
    return EvalRef(
        summary_json_path=str(summary_json_path),
        rows_csv_path=str(rows_csv_path),
        trace_csv_path=str(trace_csv_path),
    )


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def _row_label(row: dict[str, object]) -> str:
    return (
        f"{row['payout_model_type']} "
        f"{row['target_mode']} "
        f"{row['direction_source']} "
        f"train{row['train_size']} "
        f"bet{float(row['bet_size_bnb']):.2f}"
    )


def _select_trace_rows(*, trace_csv_path: str | Path, row: dict[str, object]) -> pd.DataFrame:
    df = pd.read_csv(Path(trace_csv_path).resolve())
    filt = (
        (df["payout_model_type"].astype(str) == str(row["payout_model_type"]))
        & (df["target_mode"].astype(str) == str(row["target_mode"]))
        & (df["direction_source"].astype(str) == str(row["direction_source"]))
        & (pd.to_numeric(df["train_size"], errors="coerce").fillna(-1).astype(int) == int(row["train_size"]))
    )
    out = df.loc[filt].copy()
    if out.empty:
        raise InvariantError("payout_bounded_round_trace_empty")
    out["epoch"] = pd.to_numeric(out["target_epoch"], errors="coerce").fillna(0).astype(int)
    out["cumulative_profit_bnb"] = pd.to_numeric(out["cumulative_profit_bnb"], errors="coerce").fillna(0.0)
    out["realized_profit_bnb"] = pd.to_numeric(out["realized_profit_bnb"], errors="coerce").fillna(0.0)
    out["bet_flag"] = out["action"].astype(str).str.startswith("bet_").astype(int)
    out = out.sort_values("epoch").reset_index(drop=True)
    return out


def _plot_cumulative_overlay(
    *,
    traces_by_label: dict[str, pd.DataFrame],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(14, 8))
    for label, trace in traces_by_label.items():
        plt.plot(
            np.arange(1, int(len(trace)) + 1),
            trace["cumulative_profit_bnb"].to_numpy(dtype=np.float32),
            label=label,
            linewidth=2,
        )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Held-Out Round Index")
    plt.ylabel("Cumulative Profit (BNB)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_rolling_overlay(
    *,
    traces_by_label: dict[str, pd.DataFrame],
    output_path: Path,
    title: str,
    window_rounds: int = 2000,
) -> None:
    plt.figure(figsize=(14, 8))
    for label, trace in traces_by_label.items():
        realized = trace["realized_profit_bnb"].to_numpy(dtype=np.float32)
        if int(len(realized)) < int(window_rounds):
            continue
        rolled = np.convolve(realized, np.ones(int(window_rounds), dtype=np.float32), mode="valid")
        ys = rolled * 500.0 / float(window_rounds)
        xs = np.arange(int(window_rounds), int(window_rounds) + int(len(ys)))
        plt.plot(xs, ys, label=label, linewidth=1.6)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Held-Out Round Index")
    plt.ylabel(f"Rolling Net / 500 (window={int(window_rounds)})")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _decision(
    *,
    best_row: dict[str, object],
    robustness_rows: list[dict[str, object]],
    current_bar_net_bnb: float,
    current_bar_max_dd_bnb: float,
) -> tuple[str, str]:
    best_net = float(best_row["net_profit_bnb"])
    best_dd = float(best_row["max_drawdown_bnb"])
    robust_nets = [float(row["net_profit_bnb"]) for row in robustness_rows]
    robust_per500 = [float(row["profit_per_500_bnb"]) for row in robustness_rows]
    robust_dd = [float(row["max_drawdown_bnb"]) for row in robustness_rows]
    positive_count = sum(1 for value in robust_nets if float(value) > 0.0)
    mean_per500 = float(sum(robust_per500) / len(robust_per500))
    worst_per500 = float(min(robust_per500))
    best_beats_bar = float(best_net) > float(current_bar_net_bnb) or (
        float(best_net) >= float(current_bar_net_bnb) * 0.9 and float(best_dd) < float(current_bar_max_dd_bnb) * 0.75
    )
    robust_enough = (
        int(positive_count) >= max(3, len(robust_nets) - 1)
        and float(mean_per500) > 0.0
        and float(worst_per500) > -0.01
        and float(max(robust_dd)) <= 5.0
    )
    if bool(best_beats_bar) and bool(robust_enough):
        return "dry-run", "best latest-tail branch beat or materially stabilized the current bar and stayed positive across the robustness offsets"
    return "quit", "bounded round failed to produce a branch that both beats/stabilizes the current bar and holds up across adjacent latest-tail offsets"


def main() -> None:
    args = _build_parser().parse_args()
    cwd = Path.cwd()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(str(args.config)).resolve()
    manifest_csv = Path(str(args.manifest_csv)).resolve()
    python_exe = Path(sys.executable)

    direction_sources = _parse_str_list(args.direction_sources)
    source_train_sizes = _parse_positive_int_list(args.source_train_sizes)
    stake_sizes = _parse_float_list(args.stake_sizes)
    robustness_offsets = _parse_nonnegative_int_list(args.robustness_offsets)
    if 0 not in robustness_offsets:
        robustness_offsets = [0] + list(robustness_offsets)

    source_stage_rows: list[dict[str, object]] = []
    source_stage_refs: dict[str, EvalRef] = {}
    best_source_row_by_name: dict[str, dict[str, object]] = {}

    for direction_source in direction_sources:
        run_name = f"{args.name_prefix}_source_{direction_source}_bet005"
        eval_ref = _run_eval(
            python_exe=python_exe,
            cwd=cwd,
            config_path=config_path,
            manifest_csv=manifest_csv,
            output_dir=output_dir,
            name_prefix=run_name,
            direction_source=str(direction_source),
            train_sizes=list(source_train_sizes),
            bet_size_bnb=0.05,
            tail_offset_rounds=int(args.tail_offset_rounds),
            payout_model_types=str(args.payout_model_types),
            target_mode=str(args.target_mode),
            sim_size=int(args.sim_size),
            valid_size=int(args.valid_size),
            threshold_grid=str(args.threshold_grid),
            valid_min_bet_rate=float(args.valid_min_bet_rate),
        )
        summary = _load_json(eval_ref.summary_json_path)
        rows = [dict(row) for row in summary["rows"]]
        if not rows:
            raise InvariantError("payout_bounded_round_source_rows_empty")
        source_stage_refs[str(direction_source)] = eval_ref
        source_stage_rows.extend(rows)
        best_source_row_by_name[str(direction_source)] = max(
            rows,
            key=lambda row: (
                float(row["net_profit_bnb"]),
                -float(row["max_drawdown_bnb"]),
                float(row["profit_per_500_bnb"]),
            ),
        )

    final_rows: list[dict[str, object]] = []
    trace_by_label: dict[str, pd.DataFrame] = {}

    for direction_source in direction_sources:
        base_row = dict(best_source_row_by_name[str(direction_source)])
        eval_ref = source_stage_refs[str(direction_source)]
        final_rows.append(base_row)
        trace_by_label[_row_label(base_row)] = _select_trace_rows(
            trace_csv_path=eval_ref.trace_csv_path,
            row=base_row,
        )
        for stake_size in stake_sizes:
            if math.isclose(float(stake_size), 0.05, abs_tol=1e-9):
                continue
            run_name = f"{args.name_prefix}_stake_{direction_source}_bet{str(float(stake_size)).replace('.', '')}"
            stake_eval_ref = _run_eval(
                python_exe=python_exe,
                cwd=cwd,
                config_path=config_path,
                manifest_csv=manifest_csv,
                output_dir=output_dir,
                name_prefix=run_name,
                direction_source=str(direction_source),
                train_sizes=[int(base_row["train_size"])],
                bet_size_bnb=float(stake_size),
                tail_offset_rounds=int(args.tail_offset_rounds),
                payout_model_types=str(args.payout_model_types),
                target_mode=str(args.target_mode),
                sim_size=int(args.sim_size),
                valid_size=int(args.valid_size),
                threshold_grid=str(args.threshold_grid),
                valid_min_bet_rate=float(args.valid_min_bet_rate),
            )
            summary = _load_json(stake_eval_ref.summary_json_path)
            rows = [dict(row) for row in summary["rows"]]
            if len(rows) != 1:
                raise InvariantError("payout_bounded_round_stake_rows_count_invalid")
            row = dict(rows[0])
            final_rows.append(row)
            trace_by_label[_row_label(row)] = _select_trace_rows(
                trace_csv_path=stake_eval_ref.trace_csv_path,
                row=row,
            )

    best_row = max(
        final_rows,
        key=lambda row: (
            float(row["net_profit_bnb"]),
            -float(row["max_drawdown_bnb"]),
            float(row["profit_per_500_bnb"]),
        ),
    )
    best_label = _row_label(best_row)

    robustness_rows: list[dict[str, object]] = []
    robustness_traces: dict[str, pd.DataFrame] = {}
    for offset in robustness_offsets:
        run_name = (
            f"{args.name_prefix}_robust_{str(best_row['direction_source'])}_"
            f"train{int(best_row['train_size'])}_bet{str(float(best_row['bet_size_bnb'])).replace('.', '')}_"
            f"off{int(offset):05d}"
        )
        eval_ref = _run_eval(
            python_exe=python_exe,
            cwd=cwd,
            config_path=config_path,
            manifest_csv=manifest_csv,
            output_dir=output_dir,
            name_prefix=run_name,
            direction_source=str(best_row["direction_source"]),
            train_sizes=[int(best_row["train_size"])],
            bet_size_bnb=float(best_row["bet_size_bnb"]),
            tail_offset_rounds=int(offset),
            payout_model_types=str(best_row["payout_model_type"]),
            target_mode=str(best_row["target_mode"]),
            sim_size=int(args.sim_size),
            valid_size=int(args.valid_size),
            threshold_grid=str(args.threshold_grid),
            valid_min_bet_rate=float(args.valid_min_bet_rate),
        )
        summary = _load_json(eval_ref.summary_json_path)
        rows = [dict(row) for row in summary["rows"]]
        if len(rows) != 1:
            raise InvariantError("payout_bounded_round_robust_rows_count_invalid")
        row = dict(rows[0])
        row["robustness_offset_rounds"] = int(offset)
        robustness_rows.append(row)
        robustness_traces[f"offset{int(offset)}"] = _select_trace_rows(
            trace_csv_path=eval_ref.trace_csv_path,
            row=row,
        )

    decision, decision_reason = _decision(
        best_row=best_row,
        robustness_rows=robustness_rows,
        current_bar_net_bnb=float(args.current_bar_net_bnb),
        current_bar_max_dd_bnb=float(args.current_bar_max_dd_bnb),
    )

    source_best_rows = [
        max(
            [dict(row) for row in final_rows if str(row["direction_source"]) == str(source)],
            key=lambda row: (
                float(row["net_profit_bnb"]),
                -float(row["max_drawdown_bnb"]),
                float(row["profit_per_500_bnb"]),
            ),
        )
        for source in direction_sources
    ]

    report_path = output_dir / f"{args.name_prefix}_bounded_round_report.md"
    summary_path = output_dir / f"{args.name_prefix}_bounded_round_summary.json"
    source_overlay_plot = output_dir / f"{args.name_prefix}_bounded_round_source_best_cumulative.png"
    source_roll_plot = output_dir / f"{args.name_prefix}_bounded_round_source_best_rolling.png"
    top_overlay_plot = output_dir / f"{args.name_prefix}_bounded_round_top_cumulative.png"
    robust_overlay_plot = output_dir / f"{args.name_prefix}_bounded_round_best_robustness_cumulative.png"

    traces_source_best = {_row_label(row): trace_by_label[_row_label(row)] for row in source_best_rows}
    _plot_cumulative_overlay(
        traces_by_label=traces_source_best,
        output_path=source_overlay_plot,
        title="Source Best Latest-Tail Cumulative BNB",
    )
    _plot_rolling_overlay(
        traces_by_label=traces_source_best,
        output_path=source_roll_plot,
        title="Source Best Latest-Tail Rolling Net / 500",
    )
    top_rows = sorted(final_rows, key=lambda row: float(row["net_profit_bnb"]), reverse=True)[:6]
    traces_top = {_row_label(row): trace_by_label[_row_label(row)] for row in top_rows}
    _plot_cumulative_overlay(
        traces_by_label=traces_top,
        output_path=top_overlay_plot,
        title="Top Bounded-Round Latest-Tail Cumulative BNB",
    )
    _plot_cumulative_overlay(
        traces_by_label=robustness_traces,
        output_path=robust_overlay_plot,
        title=f"Robustness Offsets for Best Config: {best_label}",
    )

    summary_payload = {
        "best_row": best_row,
        "decision": decision,
        "decision_reason": decision_reason,
        "current_bar": {
            "net_profit_bnb": float(args.current_bar_net_bnb),
            "profit_per_500_bnb": float(args.current_bar_per500),
            "max_drawdown_bnb": float(args.current_bar_max_dd_bnb),
        },
        "source_stage_rows": source_stage_rows,
        "final_rows": final_rows,
        "source_best_rows": source_best_rows,
        "robustness_rows": robustness_rows,
        "plots": {
            "source_best_cumulative": str(source_overlay_plot),
            "source_best_rolling": str(source_roll_plot),
            "top_cumulative": str(top_overlay_plot),
            "best_robustness_cumulative": str(robust_overlay_plot),
        },
        "report_path": str(report_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    lines: list[str] = []
    lines.append("# Payout-Aware Bounded Round")
    lines.append("")
    lines.append("## Standard")
    lines.append("")
    lines.append("- synced first")
    lines.append(f"- latest contiguous held-out stream: `{int(args.sim_size)}` valid rounds")
    lines.append(f"- payout model types tested: `{args.payout_model_types}`")
    lines.append(f"- target mode: `{args.target_mode}`")
    lines.append(f"- direction sources: `{', '.join(direction_sources)}`")
    lines.append(f"- fixed stakes: `{', '.join(f'{float(v):.2f}' for v in stake_sizes)}`")
    lines.append("")
    lines.append("## Final Decision")
    lines.append("")
    lines.append(f"- decision: `{decision}`")
    lines.append(f"- reason: {decision_reason}")
    lines.append("")
    lines.append("## Current Bar")
    lines.append("")
    lines.append(f"- net profit: `{float(args.current_bar_net_bnb):.6f}` BNB")
    lines.append(f"- profit per 500: `{float(args.current_bar_per500):.6f}`")
    lines.append(f"- max drawdown: `{float(args.current_bar_max_dd_bnb):.6f}` BNB")
    lines.append("")
    lines.append("## Best Bounded-Round Row")
    lines.append("")
    lines.append(f"- config: `{best_label}`")
    lines.append(f"- net profit: `{float(best_row['net_profit_bnb']):.6f}` BNB")
    lines.append(f"- profit per 500: `{float(best_row['profit_per_500_bnb']):.6f}`")
    lines.append(f"- bet rate: `{100.0 * float(best_row['bet_rate']):.3f}%`")
    lines.append(f"- win rate: `{100.0 * float(best_row['win_rate']):.3f}%`")
    lines.append(f"- max drawdown: `{float(best_row['max_drawdown_bnb']):.6f}` BNB")
    lines.append(f"- bull threshold: `{float(best_row['bull_threshold']):.6f}`")
    lines.append(f"- bear threshold: `{float(best_row['bear_threshold']):.6f}`")
    lines.append("")
    lines.append("## Source Best Rows")
    lines.append("")
    lines.append("| Direction Source | Train | Stake | Net BNB | Net / 500 | Bet rate | Win rate | Max DD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(source_best_rows, key=lambda item: float(item["net_profit_bnb"]), reverse=True):
        lines.append(
            "| "
            + f"{row['direction_source']} | "
            + f"{int(row['train_size'])} | "
            + f"{float(row['bet_size_bnb']):.2f} | "
            + f"{float(row['net_profit_bnb']):.6f} | "
            + f"{float(row['profit_per_500_bnb']):.6f} | "
            + f"{100.0 * float(row['bet_rate']):.3f}% | "
            + f"{100.0 * float(row['win_rate']):.3f}% | "
            + f"{float(row['max_drawdown_bnb']):.6f} |"
        )
    lines.append("")
    lines.append("## All Final Rows")
    lines.append("")
    lines.append("| Config | Net BNB | Net / 500 | Bet rate | Win rate | Max DD |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted(final_rows, key=lambda item: float(item["net_profit_bnb"]), reverse=True):
        lines.append(
            "| "
            + f"{_row_label(row)} | "
            + f"{float(row['net_profit_bnb']):.6f} | "
            + f"{float(row['profit_per_500_bnb']):.6f} | "
            + f"{100.0 * float(row['bet_rate']):.3f}% | "
            + f"{100.0 * float(row['win_rate']):.3f}% | "
            + f"{float(row['max_drawdown_bnb']):.6f} |"
        )
    lines.append("")
    lines.append("## Robustness Of Best Config")
    lines.append("")
    lines.append("| Offset | Net BNB | Net / 500 | Bet rate | Win rate | Max DD |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in sorted(robustness_rows, key=lambda item: int(item["robustness_offset_rounds"])):
        lines.append(
            "| "
            + f"{int(row['robustness_offset_rounds'])} | "
            + f"{float(row['net_profit_bnb']):.6f} | "
            + f"{float(row['profit_per_500_bnb']):.6f} | "
            + f"{100.0 * float(row['bet_rate']):.3f}% | "
            + f"{100.0 * float(row['win_rate']):.3f}% | "
            + f"{float(row['max_drawdown_bnb']):.6f} |"
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"- [source best cumulative]({source_overlay_plot})")
    lines.append(f"- [source best rolling]({source_roll_plot})")
    lines.append(f"- [top cumulative]({top_overlay_plot})")
    lines.append(f"- [best robustness cumulative]({robust_overlay_plot})")
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"SUMMARY_JSON={summary_path}")
    print(f"REPORT_MD={report_path}")
    print(f"DECISION={decision}")


if __name__ == "__main__":
    import math

    main()
