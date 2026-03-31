from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from pancakebot.core.errors import InvariantError

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"
_CLOSED_ROUNDS_PATTERN = re.compile(r'(^closed_rounds_path\s*=\s*")[^"]+(")', re.MULTILINE)


@dataclass(frozen=True, slots=True)
class WindowComparison:
    tail_offset_rounds: int
    stageb_per_500: float
    stageb_bet_rate: float
    flow_per_500: float
    flow_bet_rate: float


@dataclass(frozen=True, slots=True)
class SelectorResult:
    mode: str
    lookback: int
    margin_per_500: float
    skip_threshold_per_500: float
    mean_per_500: float
    mean_selected_bet_rate: float
    stageb_picks: int
    flow_picks: int
    skip_picks: int
    meets_min_selected_bet_rate: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--window-size-rounds", type=int, required=True)
    parser.add_argument("--num-windows", type=int, default=10)
    parser.add_argument("--tail-offset-rounds", type=str, default=None)
    parser.add_argument("--source-tail-rounds", type=int, default=None)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=None)
    parser.add_argument("--flow-train-size", type=int, required=True)
    parser.add_argument("--flow-val-size", type=int, default=None)
    parser.add_argument("--flow-step-size", type=int, default=None)
    parser.add_argument("--flow-ev-threshold", type=float, required=True)
    parser.add_argument("--flow-min-total-pool-c", type=float, required=True)
    parser.add_argument("--flow-allowed-sides", type=str, choices=("both", "bull_only", "bear_only"), default="bear_only")
    parser.add_argument("--flow-bull-roll-edge-min", type=float, default=0.0)
    parser.add_argument("--flow-bear-roll-edge-min", type=float, default=0.0)
    parser.add_argument("--flow-bull-roll-winrate-min", type=float, default=0.50)
    parser.add_argument("--flow-bear-roll-winrate-min", type=float, default=0.50)
    parser.add_argument("--flow-bull-cooldown-trades", type=int, default=80)
    parser.add_argument("--flow-bear-cooldown-trades", type=int, default=80)
    parser.add_argument("--selector-lookbacks", type=str, default="1,2,3,4")
    parser.add_argument("--selector-margins-per-500", type=str, default="-0.2,0.0,0.2,0.5")
    parser.add_argument("--selector-skip-thresholds-per-500", type=str, default="0.0")
    parser.add_argument("--min-selected-bet-rate", type=float, default=0.01)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _parse_int_list(raw: str | None, *, default_window_size: int, num_windows: int) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return [int(i) * int(default_window_size) for i in range(int(num_windows))]
    return [int(token.strip()) for token in str(raw).split(",") if token.strip() != ""]


def _parse_float_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in str(raw).split(",") if token.strip() != ""]


def _parse_positive_int_list(raw: str) -> list[int]:
    values = [int(token.strip()) for token in str(raw).split(",") if token.strip() != ""]
    if any(int(value) <= 0 for value in values):
        raise InvariantError("profile_window_selector_lookback_nonpositive")
    return values


def _write_last_n_lines(*, src: Path, dest: Path, n_lines: int) -> None:
    if int(n_lines) <= 0:
        raise InvariantError("profile_window_selector_source_tail_nonpositive")
    buffer: deque[str] = deque(maxlen=int(n_lines))
    with src.open("r", encoding="utf-8-sig") as f:
        for line in f:
            buffer.append(line.rstrip("\n"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="\n") as f:
        for line in buffer:
            f.write(f"{line}\n")


def _materialize_tail_config(
    *,
    base_config_path: Path,
    source_tail_rounds: int,
    exp_root: Path,
    name_prefix: str,
) -> Path:
    cfg_text = base_config_path.read_text(encoding="utf-8")
    match = _CLOSED_ROUNDS_PATTERN.search(cfg_text)
    if match is None:
        raise InvariantError("profile_window_selector_closed_rounds_path_missing")
    current_path = match.group(0)
    _ = current_path
    src_line = match.group(0)
    src_value_match = re.search(r'"([^"]+)"', src_line)
    if src_value_match is None:
        raise InvariantError("profile_window_selector_closed_rounds_path_parse_failed")
    source_closed_rounds = Path(base_config_path.parent / src_value_match.group(1)).resolve()
    tail_jsonl = (exp_root / f"{name_prefix}_closed_rounds_tail{int(source_tail_rounds)}.jsonl").resolve()
    _write_last_n_lines(src=source_closed_rounds, dest=tail_jsonl, n_lines=int(source_tail_rounds))
    patched_text = _CLOSED_ROUNDS_PATTERN.sub(
        rf'\1{tail_jsonl.as_posix()}\2',
        cfg_text,
        count=1,
    )
    out_cfg = (exp_root / f"{name_prefix}_tail{int(source_tail_rounds)}.toml").resolve()
    out_cfg.write_text(patched_text, encoding="utf-8", newline="\n")
    return out_cfg


def _run_subprocess(*, args: list[str], cwd: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".log", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                args,
                cwd=str(cwd),
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if int(proc.returncode) != 0:
            log_tail = tmp_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise InvariantError(
                "profile_window_selector_subprocess_failed: "
                f"cmd={' '.join(args)} output={log_tail}"
            )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_stageb_window(
    *,
    cwd: Path,
    exp_root: Path,
    config_path: Path,
    run_name: str,
    window_size_rounds: int,
    tail_offset_rounds: int,
    initial_bankroll_bnb: float | None,
    resume: bool,
) -> dict[str, object]:
    summary_path = exp_root / run_name / "backtest_summary.json"
    if not bool(resume) or not summary_path.exists():
        cmd = [
            sys.executable,
            "-m",
            "inspection.run_backtest_scenario",
            "--config",
            str(config_path),
            "--name",
            str(run_name),
            "--sim-size",
            str(int(window_size_rounds)),
            "--tail-offset-rounds",
            str(int(tail_offset_rounds)),
            "--reset-mode",
            "continuous",
        ]
        if initial_bankroll_bnb is not None:
            cmd.extend(["--initial-bankroll-bnb", str(float(initial_bankroll_bnb))])
        _run_subprocess(args=cmd, cwd=cwd)
    return _load_json(summary_path)


def _ensure_flow_window(
    *,
    cwd: Path,
    exp_root: Path,
    config_path: Path,
    run_name: str,
    window_size_rounds: int,
    tail_offset_rounds: int,
    initial_bankroll_bnb: float | None,
    flow_train_size: int,
    flow_val_size: int,
    flow_step_size: int,
    flow_ev_threshold: float,
    flow_min_total_pool_c: float,
    flow_allowed_sides: str,
    flow_bull_roll_edge_min: float,
    flow_bear_roll_edge_min: float,
    flow_bull_roll_winrate_min: float,
    flow_bear_roll_winrate_min: float,
    flow_bull_cooldown_trades: int,
    flow_bear_cooldown_trades: int,
    resume: bool,
) -> dict[str, object]:
    summary_path = exp_root / run_name / "backtest_summary.json"
    if not bool(resume) or not summary_path.exists():
        cmd = [
            sys.executable,
            "-m",
            "inspection.run_flow_backtest_scenario",
            "--config",
            str(config_path),
            "--name",
            str(run_name),
            "--sim-size",
            str(int(flow_train_size + int(window_size_rounds))),
            "--tail-offset-rounds",
            str(int(tail_offset_rounds)),
            "--train-size",
            str(int(flow_train_size)),
            "--val-size",
            str(int(flow_val_size)),
            "--step-size",
            str(int(flow_step_size)),
            "--ev-threshold",
            str(float(flow_ev_threshold)),
            "--min-total-pool-c",
            str(float(flow_min_total_pool_c)),
            "--allowed-sides",
            str(flow_allowed_sides),
            "--bull-roll-edge-min",
            str(float(flow_bull_roll_edge_min)),
            "--bear-roll-edge-min",
            str(float(flow_bear_roll_edge_min)),
            "--bull-roll-winrate-min",
            str(float(flow_bull_roll_winrate_min)),
            "--bear-roll-winrate-min",
            str(float(flow_bear_roll_winrate_min)),
            "--bull-cooldown-trades",
            str(int(flow_bull_cooldown_trades)),
            "--bear-cooldown-trades",
            str(int(flow_bear_cooldown_trades)),
        ]
        if initial_bankroll_bnb is not None:
            cmd.extend(["--initial-bankroll-bnb", str(float(initial_bankroll_bnb))])
        _run_subprocess(args=cmd, cwd=cwd)
    return _load_json(summary_path)


def _window_comparisons(*, rows: list[WindowComparison]) -> list[WindowComparison]:
    return sorted(rows, key=lambda row: int(row.tail_offset_rounds), reverse=True)


def _select_window_value(
    *,
    rows: list[WindowComparison],
    idx: int,
    mode: str,
    lookback: int,
    margin_per_500: float,
    skip_threshold_per_500: float,
) -> tuple[str, float, float]:
    if str(mode) == "stageb_only":
        return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
    if str(mode) == "flow_only":
        return "flow", float(rows[idx].flow_per_500), float(rows[idx].flow_bet_rate)
    if str(mode) == "skip_only":
        return "skip", 0.0, 0.0
    if str(mode) == "oracle":
        if float(rows[idx].flow_per_500) > float(rows[idx].stageb_per_500):
            return "flow", float(rows[idx].flow_per_500), float(rows[idx].flow_bet_rate)
        return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
    if str(mode) == "oracle_with_skip":
        if float(rows[idx].stageb_per_500) <= float(skip_threshold_per_500) and float(rows[idx].flow_per_500) <= float(
            skip_threshold_per_500
        ):
            return "skip", 0.0, 0.0
        if float(rows[idx].flow_per_500) > float(rows[idx].stageb_per_500):
            return "flow", float(rows[idx].flow_per_500), float(rows[idx].flow_bet_rate)
        return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
    if str(mode) == "prev_winner":
        if int(idx) == 0:
            return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
        prev = rows[int(idx) - 1]
        if float(prev.flow_per_500) > float(prev.stageb_per_500):
            return "flow", float(rows[idx].flow_per_500), float(rows[idx].flow_bet_rate)
        return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
    if str(mode) == "prev_winner_with_skip":
        if int(idx) == 0:
            if float(rows[idx].stageb_per_500) <= float(skip_threshold_per_500):
                return "skip", 0.0, 0.0
            return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
        prev = rows[int(idx) - 1]
        if float(prev.stageb_per_500) <= float(skip_threshold_per_500) and float(prev.flow_per_500) <= float(
            skip_threshold_per_500
        ):
            return "skip", 0.0, 0.0
        if float(prev.flow_per_500) > float(prev.stageb_per_500):
            return "flow", float(rows[idx].flow_per_500), float(rows[idx].flow_bet_rate)
        return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
    if str(mode) == "trailing_delta":
        if int(idx) < int(lookback):
            return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
        hist = rows[int(idx) - int(lookback) : int(idx)]
        mean_flow = sum(float(row.flow_per_500) for row in hist) / float(len(hist))
        mean_stageb = sum(float(row.stageb_per_500) for row in hist) / float(len(hist))
        if float(mean_flow - mean_stageb) > float(margin_per_500):
            return "flow", float(rows[idx].flow_per_500), float(rows[idx].flow_bet_rate)
        return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
    if str(mode) == "trailing_delta_with_skip":
        if int(idx) < int(lookback):
            if float(rows[idx].stageb_per_500) <= float(skip_threshold_per_500):
                return "skip", 0.0, 0.0
            return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
        hist = rows[int(idx) - int(lookback) : int(idx)]
        mean_flow = sum(float(row.flow_per_500) for row in hist) / float(len(hist))
        mean_stageb = sum(float(row.stageb_per_500) for row in hist) / float(len(hist))
        if max(float(mean_flow), float(mean_stageb)) <= float(skip_threshold_per_500):
            return "skip", 0.0, 0.0
        if float(mean_flow - mean_stageb) > float(margin_per_500):
            return "flow", float(rows[idx].flow_per_500), float(rows[idx].flow_bet_rate)
        return "stageb", float(rows[idx].stageb_per_500), float(rows[idx].stageb_bet_rate)
    raise InvariantError("profile_window_selector_mode_invalid")


def _evaluate_selectors(
    *,
    rows: list[WindowComparison],
    lookbacks: list[int],
    margins_per_500: list[float],
    skip_thresholds_per_500: list[float],
    min_selected_bet_rate: float,
) -> list[SelectorResult]:
    ordered = _window_comparisons(rows=rows)
    modes: list[tuple[str, int, float, float]] = [
        ("stageb_only", 0, 0.0, 0.0),
        ("flow_only", 0, 0.0, 0.0),
        ("skip_only", 0, 0.0, 0.0),
        ("oracle", 0, 0.0, 0.0),
        ("prev_winner", 0, 0.0, 0.0),
    ]
    for lookback in lookbacks:
        for margin in margins_per_500:
            modes.append(("trailing_delta", int(lookback), float(margin), 0.0))
    for skip_threshold in skip_thresholds_per_500:
        modes.append(("oracle_with_skip", 0, 0.0, float(skip_threshold)))
        modes.append(("prev_winner_with_skip", 0, 0.0, float(skip_threshold)))
        for lookback in lookbacks:
            for margin in margins_per_500:
                modes.append(
                    (
                        "trailing_delta_with_skip",
                        int(lookback),
                        float(margin),
                        float(skip_threshold),
                    )
                )
    results: list[SelectorResult] = []
    for mode, lookback, margin, skip_threshold in modes:
        total = 0.0
        total_selected_bet_rate = 0.0
        stageb_picks = 0
        flow_picks = 0
        skip_picks = 0
        for idx in range(len(ordered)):
            pick, value, bet_rate = _select_window_value(
                rows=ordered,
                idx=int(idx),
                mode=str(mode),
                lookback=int(lookback),
                margin_per_500=float(margin),
                skip_threshold_per_500=float(skip_threshold),
            )
            total += float(value)
            total_selected_bet_rate += float(bet_rate)
            if str(pick) == "flow":
                flow_picks += 1
            elif str(pick) == "skip":
                skip_picks += 1
            else:
                stageb_picks += 1
        results.append(
            SelectorResult(
                mode=str(mode),
                lookback=int(lookback),
                margin_per_500=float(margin),
                skip_threshold_per_500=float(skip_threshold),
                mean_per_500=float(total / float(len(ordered))) if ordered else 0.0,
                mean_selected_bet_rate=float(total_selected_bet_rate / float(len(ordered))) if ordered else 0.0,
                stageb_picks=int(stageb_picks),
                flow_picks=int(flow_picks),
                skip_picks=int(skip_picks),
                meets_min_selected_bet_rate=(
                    float(total_selected_bet_rate / float(len(ordered))) >= float(min_selected_bet_rate)
                    if ordered
                    else False
                ),
            )
        )
    return sorted(
        results,
        key=lambda row: (
            1 if bool(row.meets_min_selected_bet_rate) else 0,
            float(row.mean_per_500),
            float(row.mean_selected_bet_rate),
        ),
        reverse=True,
    )


def main() -> None:
    args = _build_parser().parse_args()
    cwd = Path.cwd()
    exp_root = Path(os.environ.get("PANCAKEBOT_EXP_DIR", _DEFAULT_EXP_ROOT)).resolve()
    exp_root.mkdir(parents=True, exist_ok=True)

    if int(args.window_size_rounds) <= 0:
        raise InvariantError("profile_window_selector_window_size_nonpositive")
    if int(args.num_windows) <= 0:
        raise InvariantError("profile_window_selector_num_windows_nonpositive")
    if int(args.flow_train_size) <= 0:
        raise InvariantError("profile_window_selector_flow_train_size_nonpositive")

    offsets = _parse_int_list(
        args.tail_offset_rounds,
        default_window_size=int(args.window_size_rounds),
        num_windows=int(args.num_windows),
    )
    if any(int(offset) < 0 for offset in offsets):
        raise InvariantError("profile_window_selector_tail_offset_negative")
    lookbacks = _parse_positive_int_list(args.selector_lookbacks)
    margins_per_500 = _parse_float_list(args.selector_margins_per_500)
    skip_thresholds_per_500 = _parse_float_list(args.selector_skip_thresholds_per_500)
    flow_val_size = int(args.window_size_rounds if args.flow_val_size is None else args.flow_val_size)
    flow_step_size = int(args.window_size_rounds if args.flow_step_size is None else args.flow_step_size)
    if float(args.min_selected_bet_rate) < 0.0:
        raise InvariantError("profile_window_selector_min_selected_bet_rate_negative")

    config_path = Path(str(args.config)).resolve()
    active_config_path = config_path
    if args.source_tail_rounds is not None:
        active_config_path = _materialize_tail_config(
            base_config_path=config_path,
            source_tail_rounds=int(args.source_tail_rounds),
            exp_root=exp_root,
            name_prefix=str(args.name_prefix),
        )

    rows: list[WindowComparison] = []
    resume = not bool(args.no_resume)
    for tail_offset_rounds in offsets:
        stage_name = f"{args.name_prefix}_stageb_w{int(args.window_size_rounds)}_off{int(tail_offset_rounds)}"
        flow_name = (
            f"{args.name_prefix}_flow_w{int(args.window_size_rounds)}_"
            f"train{int(args.flow_train_size)}_off{int(tail_offset_rounds)}"
        )
        stage_summary = _ensure_stageb_window(
            cwd=cwd,
            exp_root=exp_root,
            config_path=active_config_path,
            run_name=stage_name,
            window_size_rounds=int(args.window_size_rounds),
            tail_offset_rounds=int(tail_offset_rounds),
            initial_bankroll_bnb=args.initial_bankroll_bnb,
            resume=bool(resume),
        )
        flow_summary = _ensure_flow_window(
            cwd=cwd,
            exp_root=exp_root,
            config_path=active_config_path,
            run_name=flow_name,
            window_size_rounds=int(args.window_size_rounds),
            tail_offset_rounds=int(tail_offset_rounds),
            initial_bankroll_bnb=args.initial_bankroll_bnb,
            flow_train_size=int(args.flow_train_size),
            flow_val_size=int(flow_val_size),
            flow_step_size=int(flow_step_size),
            flow_ev_threshold=float(args.flow_ev_threshold),
            flow_min_total_pool_c=float(args.flow_min_total_pool_c),
            flow_allowed_sides=str(args.flow_allowed_sides),
            flow_bull_roll_edge_min=float(args.flow_bull_roll_edge_min),
            flow_bear_roll_edge_min=float(args.flow_bear_roll_edge_min),
            flow_bull_roll_winrate_min=float(args.flow_bull_roll_winrate_min),
            flow_bear_roll_winrate_min=float(args.flow_bear_roll_winrate_min),
            flow_bull_cooldown_trades=int(args.flow_bull_cooldown_trades),
            flow_bear_cooldown_trades=int(args.flow_bear_cooldown_trades),
            resume=bool(resume),
        )
        rows.append(
            WindowComparison(
                tail_offset_rounds=int(tail_offset_rounds),
                stageb_per_500=float(stage_summary["net_profit_bnb"]) * 500.0 / float(int(args.window_size_rounds)),
                stageb_bet_rate=float(stage_summary["bet_rate"]),
                flow_per_500=float(flow_summary["per_500"]),
                flow_bet_rate=float(flow_summary["bet_rate"]),
            )
        )

    ordered_rows = _window_comparisons(rows=rows)
    selector_rows = _evaluate_selectors(
        rows=ordered_rows,
        lookbacks=lookbacks,
        margins_per_500=margins_per_500,
        skip_thresholds_per_500=skip_thresholds_per_500,
        min_selected_bet_rate=float(args.min_selected_bet_rate),
    )

    compare_csv = exp_root / f"{args.name_prefix}_profile_window_compare.csv"
    compare_json = exp_root / f"{args.name_prefix}_profile_window_compare.json"
    selector_csv = exp_root / f"{args.name_prefix}_profile_window_selectors.csv"
    selector_json = exp_root / f"{args.name_prefix}_profile_window_selectors.json"

    compare_csv.write_text("", encoding="utf-8")
    with compare_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("tail_offset_rounds,stageb_per_500,stageb_bet_rate,flow_per_500,flow_bet_rate,better\n")
        for row in ordered_rows:
            better = "flow" if float(row.flow_per_500) > float(row.stageb_per_500) else "stageb"
            f.write(
                f"{int(row.tail_offset_rounds)},{float(row.stageb_per_500)},{float(row.stageb_bet_rate)},"
                f"{float(row.flow_per_500)},{float(row.flow_bet_rate)},{better}\n"
            )
    compare_json.write_text(
        json.dumps([asdict(row) for row in ordered_rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    selector_csv.write_text("", encoding="utf-8")
    with selector_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "mode,lookback,margin_per_500,skip_threshold_per_500,mean_per_500,"
            "mean_selected_bet_rate,meets_min_selected_bet_rate,stageb_picks,flow_picks,skip_picks\n"
        )
        for row in selector_rows:
            f.write(
                f"{row.mode},{int(row.lookback)},{float(row.margin_per_500)},{float(row.skip_threshold_per_500)},"
                f"{float(row.mean_per_500)},{float(row.mean_selected_bet_rate)},{bool(row.meets_min_selected_bet_rate)},"
                f"{int(row.stageb_picks)},{int(row.flow_picks)},{int(row.skip_picks)}\n"
            )
    selector_json.write_text(
        json.dumps([asdict(row) for row in selector_rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"COMPARE_CSV={compare_csv}")
    print(f"COMPARE_JSON={compare_json}")
    print(f"SELECTOR_CSV={selector_csv}")
    print(f"SELECTOR_JSON={selector_json}")
    if selector_rows:
        best = selector_rows[0]
        print(
            "BEST_SELECTOR="
            f"{best.mode} lookback={int(best.lookback)} margin_per_500={float(best.margin_per_500)} "
            f"mean_per_500={float(best.mean_per_500)}"
        )


if __name__ == "__main__":
    main()
