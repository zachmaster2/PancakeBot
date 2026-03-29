from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys

from pancakebot.core.errors import InvariantError

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class SharedEvalRow:
    sim_size: int
    tail_offset_rounds: int
    controller_mode: str
    controller_lookback_windows: int
    controller_margin_per_500: float
    controller_skip_threshold_per_500: float
    controller_per_500: float
    controller_bet_rate: float
    controller_net_profit_bnb: float
    static_stageb_per_500: float
    static_stageb_bet_rate: float
    static_stageb_net_profit_bnb: float
    lift_vs_stageb_per_500: float


@dataclass(frozen=True, slots=True)
class SharedEvalAggregateRow:
    sim_size: int
    controller_mode: str
    controller_lookback_windows: int
    controller_margin_per_500: float
    controller_skip_threshold_per_500: float
    num_offsets: int
    controller_mean_per_500: float
    controller_min_per_500: float
    controller_mean_bet_rate: float
    static_stageb_mean_per_500: float
    static_stageb_min_per_500: float
    static_stageb_mean_bet_rate: float
    mean_lift_vs_stageb_per_500: float
    min_lift_vs_stageb_per_500: float
    lift_wins: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--sim-sizes", type=str, default="6480,8640,10800")
    parser.add_argument("--tail-offset-rounds", type=str, default="0,216,432,648,864")
    parser.add_argument("--router-mode", type=str, default="selector_max_score")
    parser.add_argument(
        "--controller-mode",
        type=str,
        choices=("trailing_best_vs_baseline", "trailing_best_vs_baseline_with_skip"),
        required=True,
    )
    parser.add_argument("--baseline-profile-name", type=str, default="disloc_stageB_bullonly_recent8pct_v1")
    parser.add_argument("--alternate-profile-name", type=str, default="disloc_cons_20260227_x80")
    parser.add_argument("--window-rounds", type=int, default=216)
    parser.add_argument("--lookback-windows", type=int, required=True)
    parser.add_argument("--margin-per-500", type=float, required=True)
    parser.add_argument("--skip-threshold-per-500", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser


def _parse_positive_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = int(text)
        if int(value) <= 0:
            raise InvariantError("window_controller_shared_eval_nonpositive_int")
        out.append(int(value))
    if not out:
        raise InvariantError("window_controller_shared_eval_empty_int_list")
    return out


def _parse_nonnegative_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = int(text)
        if int(value) < 0:
            raise InvariantError("window_controller_shared_eval_negative_offset")
        out.append(int(value))
    if not out:
        raise InvariantError("window_controller_shared_eval_empty_offset_list")
    return out


def _summary_metrics(summary_path: Path) -> tuple[float, float, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    num_rounds = int(summary["num_rounds"])
    if int(num_rounds) <= 0:
        raise InvariantError("window_controller_shared_eval_num_rounds_nonpositive")
    net_profit_bnb = float(summary["net_profit_bnb"])
    per_500 = float(net_profit_bnb) * 500.0 / float(num_rounds)
    bet_rate = float(summary["bet_rate"])
    return float(per_500), float(bet_rate), float(net_profit_bnb)


def _scenario_summary_path(*, output_dir: Path, scenario_name: str) -> Path:
    return (output_dir / str(scenario_name) / "backtest_summary.json").resolve()


def _controller_scenario_name(
    *,
    name_prefix: str,
    sim_size: int,
    tail_offset_rounds: int,
) -> str:
    return f"{name_prefix}_tail{int(sim_size)}_off{int(tail_offset_rounds):05d}"


def _static_scenario_name(
    *,
    name_prefix: str,
    sim_size: int,
    tail_offset_rounds: int,
) -> str:
    return f"{name_prefix}_stageb_tail{int(sim_size)}_off{int(tail_offset_rounds):05d}"


def _run_command(*, args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=str(cwd), check=True)


def _controller_command(
    *,
    python_exe: str,
    config_path: str,
    scenario_name: str,
    sim_size: int,
    tail_offset_rounds: int,
    router_mode: str,
    controller_mode: str,
    baseline_profile_name: str,
    alternate_profile_name: str,
    window_rounds: int,
    lookback_windows: int,
    margin_per_500: float,
    skip_threshold_per_500: float,
) -> list[str]:
    cmd = [
        str(python_exe),
        "-m",
        "inspection.run_backtest_scenario",
        "--config",
        str(config_path),
        "--name",
        str(scenario_name),
        "--sim-size",
        str(int(sim_size)),
        "--tail-offset-rounds",
        str(int(tail_offset_rounds)),
        "--router-mode",
        str(router_mode),
        "--active-candidate-names",
        f"{str(baseline_profile_name)},{str(alternate_profile_name)}",
        "--window-controller-enabled",
        "true",
        "--window-controller-mode",
        str(controller_mode),
        "--window-controller-baseline-profile-name",
        str(baseline_profile_name),
        "--window-controller-alternate-profile-name",
        str(alternate_profile_name),
        "--window-controller-window-rounds",
        str(int(window_rounds)),
        "--window-controller-lookback-windows",
        str(int(lookback_windows)),
        "--window-controller-margin-per-500",
        str(float(margin_per_500)),
    ]
    if str(controller_mode) == "trailing_best_vs_baseline_with_skip":
        cmd.extend(
            [
                "--window-controller-skip-threshold-per-500",
                str(float(skip_threshold_per_500)),
            ]
        )
    return cmd


def _static_command(
    *,
    python_exe: str,
    config_path: str,
    scenario_name: str,
    sim_size: int,
    tail_offset_rounds: int,
    router_mode: str,
    baseline_profile_name: str,
) -> list[str]:
    return [
        str(python_exe),
        "-m",
        "inspection.run_backtest_scenario",
        "--config",
        str(config_path),
        "--name",
        str(scenario_name),
        "--sim-size",
        str(int(sim_size)),
        "--tail-offset-rounds",
        str(int(tail_offset_rounds)),
        "--router-mode",
        str(router_mode),
        "--active-candidate-names",
        str(baseline_profile_name),
    ]


def _aggregate_rows(rows: list[SharedEvalRow]) -> list[SharedEvalAggregateRow]:
    groups: dict[tuple[int, str, int, float, float], list[SharedEvalRow]] = {}
    for row in rows:
        key = (
            int(row.sim_size),
            str(row.controller_mode),
            int(row.controller_lookback_windows),
            float(row.controller_margin_per_500),
            float(row.controller_skip_threshold_per_500),
        )
        groups.setdefault(key, []).append(row)
    out: list[SharedEvalAggregateRow] = []
    for key, bucket in sorted(groups.items()):
        lifts = [float(row.lift_vs_stageb_per_500) for row in bucket]
        controller_vals = [float(row.controller_per_500) for row in bucket]
        controller_bet_rates = [float(row.controller_bet_rate) for row in bucket]
        baseline_vals = [float(row.static_stageb_per_500) for row in bucket]
        baseline_bet_rates = [float(row.static_stageb_bet_rate) for row in bucket]
        out.append(
            SharedEvalAggregateRow(
                sim_size=int(key[0]),
                controller_mode=str(key[1]),
                controller_lookback_windows=int(key[2]),
                controller_margin_per_500=float(key[3]),
                controller_skip_threshold_per_500=float(key[4]),
                num_offsets=int(len(bucket)),
                controller_mean_per_500=float(sum(controller_vals) / float(len(controller_vals))),
                controller_min_per_500=float(min(controller_vals)),
                controller_mean_bet_rate=float(sum(controller_bet_rates) / float(len(controller_bet_rates))),
                static_stageb_mean_per_500=float(sum(baseline_vals) / float(len(baseline_vals))),
                static_stageb_min_per_500=float(min(baseline_vals)),
                static_stageb_mean_bet_rate=float(sum(baseline_bet_rates) / float(len(baseline_bet_rates))),
                mean_lift_vs_stageb_per_500=float(sum(lifts) / float(len(lifts))),
                min_lift_vs_stageb_per_500=float(min(lifts)),
                lift_wins=int(sum(1 for value in lifts if float(value) > 0.0)),
            )
        )
    return out


def _write_rows_csv(*, path: Path, rows: list[object]) -> None:
    if not rows:
        raise InvariantError("window_controller_shared_eval_rows_empty")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = _build_parser().parse_args()
    sim_sizes = _parse_positive_int_list(str(args.sim_sizes))
    tail_offsets = _parse_nonnegative_int_list(str(args.tail_offset_rounds))
    root = Path.cwd().resolve()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    python_exe = str(Path(sys.executable).resolve())
    rows: list[SharedEvalRow] = []
    for sim_size in sim_sizes:
        for tail_offset_rounds in tail_offsets:
            controller_name = _controller_scenario_name(
                name_prefix=str(args.name_prefix),
                sim_size=int(sim_size),
                tail_offset_rounds=int(tail_offset_rounds),
            )
            controller_summary_path = _scenario_summary_path(output_dir=output_dir, scenario_name=str(controller_name))
            if not (bool(args.reuse_existing) and controller_summary_path.exists()):
                _run_command(
                    args=_controller_command(
                        python_exe=python_exe,
                        config_path=str(args.config),
                        scenario_name=str(controller_name),
                        sim_size=int(sim_size),
                        tail_offset_rounds=int(tail_offset_rounds),
                        router_mode=str(args.router_mode),
                        controller_mode=str(args.controller_mode),
                        baseline_profile_name=str(args.baseline_profile_name),
                        alternate_profile_name=str(args.alternate_profile_name),
                        window_rounds=int(args.window_rounds),
                        lookback_windows=int(args.lookback_windows),
                        margin_per_500=float(args.margin_per_500),
                        skip_threshold_per_500=float(args.skip_threshold_per_500),
                    ),
                    cwd=root,
                )

            static_name = _static_scenario_name(
                name_prefix=str(args.name_prefix),
                sim_size=int(sim_size),
                tail_offset_rounds=int(tail_offset_rounds),
            )
            static_summary_path = _scenario_summary_path(output_dir=output_dir, scenario_name=str(static_name))
            if not (bool(args.reuse_existing) and static_summary_path.exists()):
                _run_command(
                    args=_static_command(
                        python_exe=python_exe,
                        config_path=str(args.config),
                        scenario_name=str(static_name),
                        sim_size=int(sim_size),
                        tail_offset_rounds=int(tail_offset_rounds),
                        router_mode=str(args.router_mode),
                        baseline_profile_name=str(args.baseline_profile_name),
                    ),
                    cwd=root,
                )

            controller_per_500, controller_bet_rate, controller_net_profit = _summary_metrics(controller_summary_path)
            static_per_500, static_bet_rate, static_net_profit = _summary_metrics(static_summary_path)
            rows.append(
                SharedEvalRow(
                    sim_size=int(sim_size),
                    tail_offset_rounds=int(tail_offset_rounds),
                    controller_mode=str(args.controller_mode),
                    controller_lookback_windows=int(args.lookback_windows),
                    controller_margin_per_500=float(args.margin_per_500),
                    controller_skip_threshold_per_500=float(args.skip_threshold_per_500),
                    controller_per_500=float(controller_per_500),
                    controller_bet_rate=float(controller_bet_rate),
                    controller_net_profit_bnb=float(controller_net_profit),
                    static_stageb_per_500=float(static_per_500),
                    static_stageb_bet_rate=float(static_bet_rate),
                    static_stageb_net_profit_bnb=float(static_net_profit),
                    lift_vs_stageb_per_500=float(controller_per_500 - static_per_500),
                )
            )

    aggregate_rows = _aggregate_rows(rows)
    csv_path = output_dir / f"{args.name_prefix}_window_controller_shared_eval.csv"
    summary_path = output_dir / f"{args.name_prefix}_window_controller_shared_eval_summary.json"
    _write_rows_csv(path=csv_path, rows=rows)
    summary_path.write_text(
        json.dumps(
            {
                "controller_mode": str(args.controller_mode),
                "lookback_windows": int(args.lookback_windows),
                "margin_per_500": float(args.margin_per_500),
                "skip_threshold_per_500": float(args.skip_threshold_per_500),
                "rows": [asdict(row) for row in rows],
                "aggregates": [asdict(row) for row in aggregate_rows],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
