from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--report-path", type=str, required=True)
    parser.add_argument("--rolling-window-rounds", type=int, default=2000)
    return parser


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _label_for_row(row: dict[str, object]) -> str:
    return (
        f"{row['payout_model_type']} "
        f"{row['target_mode']} "
        f"{row['direction_source']} "
        f"train{row['train_size']}"
    )


def _plot_cumulative_bnb(
    *,
    traces_by_label: dict[str, list[dict[str, str]]],
    output_path: Path,
    title: str,
    highlight_label: str | None = None,
) -> None:
    plt.figure(figsize=(12, 6))
    for label, rows in traces_by_label.items():
        xs = np.arange(1, int(len(rows)) + 1, dtype=np.int64)
        ys = np.asarray([float(row["cumulative_profit_bnb"]) for row in rows], dtype=np.float32)
        alpha = 1.0 if str(label) == str(highlight_label) or highlight_label is None else 0.35
        linewidth = 2.4 if str(label) == str(highlight_label) else 1.4
        plt.plot(xs, ys, label=label, alpha=alpha, linewidth=linewidth)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.title(title)
    plt.xlabel("Held-Out Round Index")
    plt.ylabel("Cumulative Profit (BNB)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_rolling_profit_per_500(
    *,
    rows: list[dict[str, str]],
    output_path: Path,
    title: str,
    window: int,
) -> None:
    realized = np.asarray([float(row["realized_profit_bnb"]) for row in rows], dtype=np.float32)
    kernel = np.ones(int(window), dtype=np.float32)
    rolled = np.convolve(realized, kernel, mode="valid")
    ys = rolled * 500.0 / float(window)
    xs = np.arange(int(window), int(window) + int(len(ys)), dtype=np.int64)
    plt.figure(figsize=(12, 6))
    plt.plot(xs, ys, color="#1f77b4", linewidth=1.6)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.title(title)
    plt.xlabel("Held-Out Round Index")
    plt.ylabel(f"Rolling Net / 500 (window={int(window)})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_cumulative_bet_count(
    *,
    rows: list[dict[str, str]],
    output_path: Path,
    title: str,
) -> None:
    bet_flags = np.asarray(
        [1 if str(row["action"]).startswith("bet_") else 0 for row in rows],
        dtype=np.int64,
    )
    xs = np.arange(1, int(len(rows)) + 1, dtype=np.int64)
    ys = np.cumsum(bet_flags)
    plt.figure(figsize=(12, 6))
    plt.plot(xs, ys, color="#d62728", linewidth=1.6)
    plt.title(title)
    plt.xlabel("Held-Out Round Index")
    plt.ylabel("Cumulative Bets")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = _build_parser().parse_args()
    summary_path = Path(str(args.summary_json)).resolve()
    report_path = Path(str(args.report_path)).resolve()
    summary = _load_summary(summary_path)
    trace_rows = _load_trace_rows(Path(str(summary["trace_csv_path"])).resolve())
    rows = list(summary["rows"])
    label_by_key = {
        (
            str(row["payout_model_type"]),
            str(row["target_mode"]),
            str(row["direction_source"]),
            int(row["train_size"]),
        ): _label_for_row(row)
        for row in rows
    }
    traces_by_label: dict[str, list[dict[str, str]]] = {}
    for row in trace_rows:
        key = (
            str(row["payout_model_type"]),
            str(row["target_mode"]),
            str(row["direction_source"]),
            int(row["train_size"]),
        )
        traces_by_label.setdefault(label_by_key[key], []).append(row)
    for value in traces_by_label.values():
        value.sort(key=lambda row: int(row["target_epoch"]))

    best_row = dict(summary["best_row"])
    best_label = _label_for_row(best_row)
    best_traces = traces_by_label[best_label]
    cumulative_values = np.asarray([float(row["cumulative_profit_bnb"]) for row in best_traces], dtype=np.float32)
    quarter_idx = int(len(best_traces) * 0.25)
    half_idx = int(len(best_traces) * 0.50)
    three_quarter_idx = int(len(best_traces) * 0.75)

    base_name = report_path.stem
    plot_dir = report_path.parent
    all_plot = (plot_dir / f"{base_name}_cumulative_bnb_all.png").resolve()
    best_plot = (plot_dir / f"{base_name}_cumulative_bnb_best.png").resolve()
    rolling_plot = (plot_dir / f"{base_name}_rolling_profit_per500_best.png").resolve()
    bet_plot = (plot_dir / f"{base_name}_cumulative_bets_best.png").resolve()

    _plot_cumulative_bnb(
        traces_by_label=traces_by_label,
        output_path=all_plot,
        title="Payout-Aware Longstream Cumulative BNB",
        highlight_label=best_label,
    )
    _plot_cumulative_bnb(
        traces_by_label={best_label: best_traces},
        output_path=best_plot,
        title=f"Best Longstream Cumulative BNB: {best_label}",
    )
    _plot_rolling_profit_per_500(
        rows=best_traces,
        output_path=rolling_plot,
        title=f"Best Longstream Rolling Net / 500: {best_label}",
        window=int(args.rolling_window_rounds),
    )
    _plot_cumulative_bet_count(
        rows=best_traces,
        output_path=bet_plot,
        title=f"Best Longstream Cumulative Bets: {best_label}",
    )

    report_lines = [
        "# Payout-Aware Longstream Report",
        "",
        f"Summary source: `{summary_path}`",
        "",
        "## Scope",
        "",
        "- latest contiguous held-out stream",
        f"- sim size: `{summary['sim_size']}` valid target rounds",
        f"- target mode: `{summary['target_mode']}`",
        f"- direction source: `{summary['direction_source']}`",
        f"- bet size: `{summary['bet_size_bnb']}` BNB",
        "",
        "## Best Result",
        "",
        f"- config: `{best_label}`",
        f"- final net profit: `{float(best_row['net_profit_bnb']):.6f}` BNB",
        f"- net per 500: `{float(best_row['profit_per_500_bnb']):.6f}`",
        f"- bet rate: `{100.0 * float(best_row['bet_rate']):.3f}%`",
        f"- win rate: `{100.0 * float(best_row['win_rate']):.3f}%`",
        f"- max drawdown: `{float(best_row['max_drawdown_bnb']):.6f}` BNB",
        f"- thresholds: bull `{float(best_row['bull_threshold']):.6f}`, bear `{float(best_row['bear_threshold']):.6f}`",
        "",
        "## Best Curve Diagnostics",
        "",
        f"- cumulative min: `{float(np.min(cumulative_values)):.6f}` BNB",
        f"- cumulative max: `{float(np.max(cumulative_values)):.6f}` BNB",
        f"- cumulative at 25% stream: `{float(cumulative_values[max(quarter_idx - 1, 0)]):.6f}` BNB",
        f"- cumulative at 50% stream: `{float(cumulative_values[max(half_idx - 1, 0)]):.6f}` BNB",
        f"- cumulative at 75% stream: `{float(cumulative_values[max(three_quarter_idx - 1, 0)]):.6f}` BNB",
        f"- cumulative at end: `{float(cumulative_values[-1]):.6f}` BNB",
        f"- share of held-out rounds at nonnegative cumulative profit: `{100.0 * float(np.mean(cumulative_values >= 0.0)):.3f}%`",
        "",
        "## Config Table",
        "",
        "| Config | Net BNB | Net / 500 | Bet rate | Win rate | Max DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: float(item["net_profit_bnb"]), reverse=True):
        label = _label_for_row(row)
        report_lines.append(
            "| "
            + f"{label} | "
            + f"{float(row['net_profit_bnb']):.6f} | "
            + f"{float(row['profit_per_500_bnb']):.6f} | "
            + f"{100.0 * float(row['bet_rate']):.3f}% | "
            + f"{100.0 * float(row['win_rate']):.3f}% | "
            + f"{float(row['max_drawdown_bnb']):.6f} |"
        )
    report_lines += [
        "",
        "## Plots",
        "",
        f"- [{all_plot.name}]({all_plot})",
        f"- [{best_plot.name}]({best_plot})",
        f"- [{rolling_plot.name}]({rolling_plot})",
        f"- [{bet_plot.name}]({bet_plot})",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
