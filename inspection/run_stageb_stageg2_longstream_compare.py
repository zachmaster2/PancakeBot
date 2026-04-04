from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

from inspection.run_profile_set_window_selector import (
    ProfileMetric,
    WindowRow,
    _ordered_window_rows,
    _pick_window,
)
from pancakebot.core.errors import InvariantError

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"
_PATH_LINE_PATTERN = re.compile(r'^(?P<key>[A-Za-z0-9_]+)\s*=\s*"(?P<value>[^"]+)"\s*$')


@dataclass(frozen=True, slots=True)
class LegacyHeuristicConfig:
    mode: str
    lookback: int
    margin_per_500: float
    skip_threshold_per_500: float


@dataclass(frozen=True, slots=True)
class LongstreamSummary:
    num_windows: int
    selected_windows: int
    stageb_windows: int
    stageg2_windows: int
    skip_windows: int
    num_rounds: int
    num_bets: int
    bet_rate: float
    net_profit_bnb: float
    profit_per_500_bnb: float
    max_drawdown_bnb: float
    start_epoch: int
    end_epoch: int


_LEGACY_HEURISTIC = LegacyHeuristicConfig(
    mode="trailing_best_vs_stageb_with_skip",
    lookback=1,
    margin_per_500=0.5,
    skip_threshold_per_500=0.0,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--window-size-rounds", type=int, default=216)
    parser.add_argument("--num-windows", type=int, default=231)
    parser.add_argument("--source-tail-rounds", type=int, default=65061)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    parser.add_argument(
        "--payout-trace-csv",
        type=str,
        default="../PancakeBot_var_exp/payoutaware_residual_cat_mlp_20260403_payout_aware_policy_trace_rows.csv",
    )
    parser.add_argument(
        "--payout-summary-json",
        type=str,
        default="../PancakeBot_var_exp/payoutaware_residual_cat_mlp_20260403_payout_aware_policy_summary.json",
    )
    return parser


def _run_command(*, args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=str(cwd), check=True)


def _selector_command(*, python_exe: str, config_path: Path, name_prefix: str, window_size_rounds: int, num_windows: int, source_tail_rounds: int) -> list[str]:
    return [
        str(python_exe),
        "-m",
        "inspection.run_profile_set_window_selector",
        "--config",
        str(config_path),
        "--name-prefix",
        str(name_prefix),
        "--window-size-rounds",
        str(int(window_size_rounds)),
        "--num-windows",
        str(int(num_windows)),
        "--source-tail-rounds",
        str(int(source_tail_rounds)),
        "--dislocation-profile",
        "name=stageg2_bullonly,active_candidate_name=disloc_stageG2_bullonly_recent5pct_v1",
        "--selector-lookbacks",
        "1,2,3,4,5",
        "--selector-margins-per-500=-0.2,0.0,0.2,0.5,1.0",
        "--selector-skip-thresholds-per-500=0.0,0.05,0.1",
        "--min-selected-bet-rate",
        "0.01",
    ]


def _materialize_legacy_base_config(*, config_path: Path, output_dir: Path, name_prefix: str) -> Path:
    text = config_path.read_text(encoding="utf-8")
    marker = "[strategy.window_controller]"
    start = text.find(marker)
    if int(start) < 0:
        raise InvariantError("stageb_stageg2_window_controller_section_missing")
    next_section = text.find("\n[", int(start) + len(marker))
    end = len(text) if int(next_section) < 0 else int(next_section) + 1
    section = text[int(start) : int(end)]
    enabled_line = "enabled = true"
    if enabled_line not in section and "enabled = false" not in section:
        raise InvariantError("stageb_stageg2_window_controller_enabled_missing")
    section = section.replace("enabled = true", "enabled = false", 1)
    patched = text[: int(start)] + section + text[int(end) :]
    patched_lines: list[str] = []
    for raw_line in patched.splitlines():
        match = _PATH_LINE_PATTERN.match(raw_line.strip())
        if match is None:
            patched_lines.append(raw_line)
            continue
        key = str(match.group("key"))
        value = str(match.group("value"))
        if not (str(key).endswith("_path") or str(key).endswith("_dir")):
            patched_lines.append(raw_line)
            continue
        resolved = Path(value)
        if not resolved.is_absolute():
            resolved = (config_path.parent / resolved).resolve()
        patched_lines.append(f'{key} = "{resolved.as_posix()}"')
    patched = "\n".join(patched_lines) + "\n"
    out_path = (output_dir / f"{name_prefix}_legacy_base.toml").resolve()
    out_path.write_text(patched, encoding="utf-8", newline="\n")
    return out_path


def _load_compare_rows(compare_csv: Path) -> list[WindowRow]:
    df = pd.read_csv(compare_csv)
    if df.empty:
        raise InvariantError("stageb_stageg2_longstream_compare_empty")
    rows: list[WindowRow] = []
    for _, raw in df.iterrows():
        rows.append(
            WindowRow(
                tail_offset_rounds=int(raw["tail_offset_rounds"]),
                metrics={
                    "stageb": ProfileMetric(
                        per_500=float(raw["stageb_per_500"]),
                        bet_rate=float(raw["stageb_bet_rate"]),
                    ),
                    "stageg2_bullonly": ProfileMetric(
                        per_500=float(raw["stageg2_bullonly_per_500"]),
                        bet_rate=float(raw["stageg2_bullonly_bet_rate"]),
                    ),
                },
            )
        )
    return _ordered_window_rows(rows)


def _load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").fillna(0).astype(int)
    df["profit_bnb"] = pd.to_numeric(df["profit_bnb"], errors="coerce").fillna(0.0)
    return df


def _stitch_legacy_trace(*, output_dir: Path, name_prefix: str, rows: list[WindowRow], cfg: LegacyHeuristicConfig) -> tuple[pd.DataFrame, LongstreamSummary]:
    traces: list[pd.DataFrame] = []
    stageb_windows = 0
    stageg2_windows = 0
    skip_windows = 0
    for idx, row in enumerate(rows):
        pick, _, _ = _pick_window(
            rows=rows,
            idx=int(idx),
            mode=str(cfg.mode),
            profile_name="",
            lookback=int(cfg.lookback),
            margin_per_500=float(cfg.margin_per_500),
            skip_threshold_per_500=float(cfg.skip_threshold_per_500),
        )
        if str(pick) == "skip":
            skip_windows += 1
            run_name = f"{name_prefix}_stageb_off{int(row.tail_offset_rounds):05d}"
        elif str(pick) == "stageg2_bullonly":
            stageg2_windows += 1
            run_name = f"{name_prefix}_{str(pick)}_off{int(row.tail_offset_rounds):05d}"
        else:
            stageb_windows += 1
            run_name = f"{name_prefix}_{str(pick)}_off{int(row.tail_offset_rounds):05d}"
        trades = _load_trades((output_dir / run_name / "backtest_trades.csv").resolve())
        trades = trades.sort_values("epoch").reset_index(drop=True)
        if str(pick) == "skip":
            trades = trades.copy()
            trades["action"] = "SKIP"
            trades["skip_reason"] = "legacy_skip_controller_window"
            trades["direction"] = ""
            trades["bet_size_bnb"] = 0.0
            trades["p_final"] = 0.5
            trades["final_total_bnb"] = 0.0
            trades["final_bull_bnb"] = 0.0
            trades["final_bear_bnb"] = 0.0
            trades["ev_bnb"] = 0.0
            trades["profit_bnb"] = 0.0
            trades["selected_strategy"] = ""
            trades["selector_score_bnb"] = ""
        trades["controller_pick"] = str(pick)
        trades["window_tail_offset_rounds"] = int(row.tail_offset_rounds)
        traces.append(trades)
    if not traces:
        raise InvariantError("stageb_stageg2_longstream_no_traces")
    merged = pd.concat(traces, ignore_index=True)
    merged = merged.sort_values("epoch").reset_index(drop=True)
    merged["cumulative_profit_bnb"] = merged["profit_bnb"].cumsum()
    merged["bet_flag"] = (merged["action"].astype(str) == "BET").astype(int)
    running_peak = merged["cumulative_profit_bnb"].cummax()
    drawdown = running_peak - merged["cumulative_profit_bnb"]
    num_rounds = int(len(merged))
    num_bets = int(merged["bet_flag"].sum())
    net_profit = float(merged["profit_bnb"].sum())
    summary = LongstreamSummary(
        num_windows=int(len(rows)),
        selected_windows=int(stageb_windows + stageg2_windows),
        stageb_windows=int(stageb_windows),
        stageg2_windows=int(stageg2_windows),
        skip_windows=int(skip_windows),
        num_rounds=int(num_rounds),
        num_bets=int(num_bets),
        bet_rate=float(num_bets / float(num_rounds)) if int(num_rounds) > 0 else 0.0,
        net_profit_bnb=float(net_profit),
        profit_per_500_bnb=float(net_profit * 500.0 / float(num_rounds)) if int(num_rounds) > 0 else 0.0,
        max_drawdown_bnb=float(drawdown.max()) if len(drawdown) else 0.0,
        start_epoch=int(merged["epoch"].iloc[0]),
        end_epoch=int(merged["epoch"].iloc[-1]),
    )
    return merged, summary


def _load_best_payout_trace(*, payout_trace_csv: Path, payout_summary_json: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    summary = json.loads(payout_summary_json.read_text(encoding="utf-8"))
    best = dict(summary["best_row"])
    df = pd.read_csv(payout_trace_csv)
    df = df[
        (df["payout_model_type"].astype(str) == str(best["payout_model_type"]))
        & (df["target_mode"].astype(str) == str(best["target_mode"]))
        & (df["direction_source"].astype(str) == str(best["direction_source"]))
        & (pd.to_numeric(df["train_size"], errors="coerce").fillna(-1).astype(int) == int(best["train_size"]))
    ].copy()
    if df.empty:
        raise InvariantError("stageb_stageg2_payout_trace_empty")
    df["epoch"] = pd.to_numeric(df["target_epoch"], errors="coerce").fillna(0).astype(int)
    df["cumulative_profit_bnb"] = pd.to_numeric(df["cumulative_profit_bnb"], errors="coerce").fillna(0.0)
    df["realized_profit_bnb"] = pd.to_numeric(df["realized_profit_bnb"], errors="coerce").fillna(0.0)
    df["bet_flag"] = df["action"].astype(str).str.startswith("bet_").astype(int)
    df = df.sort_values("epoch").reset_index(drop=True)
    return df, best


def _plot_cumulative(*, legacy_trace: pd.DataFrame, payout_trace: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(14, 8))
    plt.plot(legacy_trace["epoch"], legacy_trace["cumulative_profit_bnb"], label="legacy_skip_stageb_vs_stageg2", linewidth=2)
    plt.plot(payout_trace["epoch"], payout_trace["cumulative_profit_bnb"], label="payout_aware_best", linewidth=2)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Profit (BNB)")
    plt.title("Latest-Tail Cumulative Profit: Legacy Skip Controller vs Payout-Aware")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_rolling(*, legacy_trace: pd.DataFrame, payout_trace: pd.DataFrame, output_path: Path, window_rounds: int = 500) -> None:
    plt.figure(figsize=(14, 8))
    legacy_roll = legacy_trace["profit_bnb"].rolling(int(window_rounds), min_periods=int(window_rounds)).sum() * 500.0 / float(window_rounds)
    payout_roll = payout_trace["realized_profit_bnb"].rolling(int(window_rounds), min_periods=int(window_rounds)).sum() * 500.0 / float(window_rounds)
    plt.plot(legacy_trace["epoch"], legacy_roll, label="legacy_skip_stageb_vs_stageg2", linewidth=1.5)
    plt.plot(payout_trace["epoch"], payout_roll, label="payout_aware_best", linewidth=1.5)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Rolling Profit per 500")
    plt.title(f"Latest-Tail Rolling Profit per 500 ({int(window_rounds)} rounds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = _build_parser().parse_args()
    cwd = Path.cwd()
    output_dir = Path(os.environ.get("PANCAKEBOT_EXP_DIR", str(args.output_dir))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(str(args.config)).resolve()
    python_exe = Path(sys.executable)
    active_config_path = _materialize_legacy_base_config(
        config_path=config_path,
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
    )

    compare_csv = output_dir / f"{args.name_prefix}_profile_set_window_compare.csv"
    if not compare_csv.exists():
        cmd = _selector_command(
            python_exe=str(python_exe),
            config_path=active_config_path,
            name_prefix=str(args.name_prefix),
            window_size_rounds=int(args.window_size_rounds),
            num_windows=int(args.num_windows),
            source_tail_rounds=int(args.source_tail_rounds),
        )
        _run_command(args=cmd, cwd=cwd)

    rows = _load_compare_rows(compare_csv)
    legacy_trace, legacy_summary = _stitch_legacy_trace(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        rows=rows,
        cfg=_LEGACY_HEURISTIC,
    )
    legacy_trace_path = output_dir / f"{args.name_prefix}_legacy_longstream_trace.csv"
    legacy_trace.to_csv(legacy_trace_path, index=False)

    payout_trace, payout_best = _load_best_payout_trace(
        payout_trace_csv=Path(str(args.payout_trace_csv)).resolve(),
        payout_summary_json=Path(str(args.payout_summary_json)).resolve(),
    )

    cumulative_plot = output_dir / f"{args.name_prefix}_legacy_vs_payout_cumulative_bnb.png"
    rolling_plot = output_dir / f"{args.name_prefix}_legacy_vs_payout_rolling_profit_per500.png"
    report_path = output_dir / f"{args.name_prefix}_legacy_vs_payout_report.md"
    summary_path = output_dir / f"{args.name_prefix}_legacy_vs_payout_summary.json"

    _plot_cumulative(legacy_trace=legacy_trace, payout_trace=payout_trace, output_path=cumulative_plot)
    _plot_rolling(legacy_trace=legacy_trace, payout_trace=payout_trace, output_path=rolling_plot)

    payout_summary = {
        "num_rounds": int(len(payout_trace)),
        "num_bets": int(payout_trace["bet_flag"].sum()),
        "bet_rate": float(float(payout_trace["bet_flag"].sum()) / float(len(payout_trace))),
        "net_profit_bnb": float(payout_trace["cumulative_profit_bnb"].iloc[-1]),
        "profit_per_500_bnb": float(float(payout_trace["cumulative_profit_bnb"].iloc[-1]) * 500.0 / float(len(payout_trace))),
        "start_epoch": int(payout_trace["epoch"].iloc[0]),
        "end_epoch": int(payout_trace["epoch"].iloc[-1]),
        "best_row": payout_best,
    }
    summary_payload = {
        "legacy_summary": asdict(legacy_summary),
        "payout_summary": payout_summary,
        "legacy_trace_csv": str(legacy_trace_path),
        "payout_trace_csv": str(Path(str(args.payout_trace_csv)).resolve()),
        "cumulative_plot_path": str(cumulative_plot),
        "rolling_plot_path": str(rolling_plot),
        "report_path": str(report_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    lines = [
        "# Legacy Skip Controller vs Payout-Aware Longstream",
        "",
        "## Legacy Controller",
        "",
        f"- mode: `{_LEGACY_HEURISTIC.mode}`",
        f"- lookback: `{_LEGACY_HEURISTIC.lookback}`",
        f"- margin per 500: `{_LEGACY_HEURISTIC.margin_per_500}`",
        f"- skip threshold per 500: `{_LEGACY_HEURISTIC.skip_threshold_per_500}`",
        f"- windows: `{legacy_summary.num_windows}` of `{int(args.window_size_rounds)}` rounds",
        f"- selected windows: `{legacy_summary.selected_windows}`",
        f"- picks: `stageB={legacy_summary.stageb_windows}` `stageG2={legacy_summary.stageg2_windows}` `skip={legacy_summary.skip_windows}`",
        f"- rounds covered: `{legacy_summary.num_rounds}`",
        f"- bets: `{legacy_summary.num_bets}`",
        f"- bet rate: `{legacy_summary.bet_rate:.4%}`",
        f"- net profit: `{legacy_summary.net_profit_bnb:.6f}` BNB",
        f"- profit per 500: `{legacy_summary.profit_per_500_bnb:.6f}`",
        f"- max drawdown: `{legacy_summary.max_drawdown_bnb:.6f}` BNB",
        "",
        "## Payout-Aware Reference",
        "",
        f"- config: `{payout_best['payout_model_type']}` `{payout_best['target_mode']}` direction `{payout_best['direction_source']}` train `{payout_best['train_size']}`",
        f"- rounds: `{payout_summary['num_rounds']}`",
        f"- bets: `{payout_summary['num_bets']}`",
        f"- bet rate: `{payout_summary['bet_rate']:.4%}`",
        f"- net profit: `{payout_summary['net_profit_bnb']:.6f}` BNB",
        f"- profit per 500: `{payout_summary['profit_per_500_bnb']:.6f}`",
        "",
        "## Plots",
        "",
        f"- cumulative BNB: `{cumulative_plot}`",
        f"- rolling profit per 500: `{rolling_plot}`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"SUMMARY_JSON={summary_path}")
    print(f"REPORT_MD={report_path}")
    print(f"PLOT_CUMULATIVE={cumulative_plot}")
    print(f"PLOT_ROLLING={rolling_plot}")


if __name__ == "__main__":
    main()
