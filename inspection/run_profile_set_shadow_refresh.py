from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"
_DEFAULT_FLOW_PROFILES = (
    "name=flow_bear_base,train_size=15000,ev_threshold=0.006,min_total_pool_c=1.2,"
    "allowed_sides=bear_only,bull_roll_edge_min=0.0,bear_roll_edge_min=0.0,"
    "bull_roll_winrate_min=0.5,bear_roll_winrate_min=0.5,"
    "bull_cooldown_trades=80,bear_cooldown_trades=80",
    "name=flow_bear_loose12,train_size=15000,ev_threshold=0.005,min_total_pool_c=1.2,"
    "allowed_sides=bear_only,bull_roll_edge_min=0.0,bear_roll_edge_min=-0.002,"
    "bull_roll_winrate_min=0.5,bear_roll_winrate_min=0.47,"
    "bull_cooldown_trades=80,bear_cooldown_trades=120",
    "name=flow_bear_loose10,train_size=15000,ev_threshold=0.005,min_total_pool_c=1.0,"
    "allowed_sides=bear_only,bull_roll_edge_min=0.0,bear_roll_edge_min=-0.002,"
    "bull_roll_winrate_min=0.5,bear_roll_winrate_min=0.47,"
    "bull_cooldown_trades=80,bear_cooldown_trades=120",
    "name=flow_bear_strict15,train_size=15000,ev_threshold=0.005,min_total_pool_c=1.5,"
    "allowed_sides=bear_only,bull_roll_edge_min=0.0,bear_roll_edge_min=-0.002,"
    "bull_roll_winrate_min=0.5,bear_roll_winrate_min=0.47,"
    "bull_cooldown_trades=80,bear_cooldown_trades=120",
)
_DEFAULT_DISLOCATION_PROFILES = (
    "name=stageg2_bullonly,active_candidate_name=disloc_stageG2_bullonly_recent5pct_v1",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--window-size-rounds", type=int, default=216)
    parser.add_argument("--num-windows", type=int, default=20)
    parser.add_argument("--source-tail-rounds", type=int, default=30000)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=50.0)
    parser.add_argument("--flow-profile", action="append", default=[])
    parser.add_argument("--dislocation-profile", action="append", default=[])
    parser.add_argument("--mode", type=str, choices=("delta_ridge", "delta_logistic"), default="delta_ridge")
    parser.add_argument("--feature-lookbacks", type=str, default="1,3,5,8")
    parser.add_argument("--min-train-windows", type=int, default=10)
    parser.add_argument("--min-hold-windows", type=int, default=1)
    parser.add_argument("--margin-per-500", type=float, default=-0.2)
    parser.add_argument("--skip-threshold-per-500", type=float, default=0.0)
    parser.add_argument("--ridge-alpha", type=float, default=2.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _flow_profiles_or_default(raw: list[str]) -> list[str]:
    return list(raw) if raw else [str(x) for x in _DEFAULT_FLOW_PROFILES]


def _dislocation_profiles_or_default(raw: list[str]) -> list[str]:
    return list(raw) if raw else [str(x) for x in _DEFAULT_DISLOCATION_PROFILES]


def _run_subprocess(*, argv: list[str], cwd: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".log", delete=False) as handle:
        log_path = Path(handle.name)
        with log_path.open("wb") as log_handle:
            completed = subprocess.run(
                argv,
                cwd=str(cwd),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    if int(completed.returncode) == 0:
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass
        return
    try:
        tail_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        tail_text = ""
    raise RuntimeError(f"profile_set_shadow_refresh_subprocess_failed:\n{tail_text[-4000:]}")


def _window_selector_argv(
    *,
    python_exe: str,
    args: argparse.Namespace,
    flow_profiles: list[str],
    dislocation_profiles: list[str],
) -> list[str]:
    argv = [
        str(python_exe),
        "-m",
        "inspection.run_profile_set_window_selector",
        "--config",
        str(args.config),
        "--name-prefix",
        str(args.name_prefix),
        "--window-size-rounds",
        str(int(args.window_size_rounds)),
        "--num-windows",
        str(int(args.num_windows)),
        "--source-tail-rounds",
        str(int(args.source_tail_rounds)),
        "--initial-bankroll-bnb",
        str(float(args.initial_bankroll_bnb)),
        "--selector-lookbacks",
        "1,2,3,4,5",
        "--selector-margins-per-500=-0.2,0.0,0.2,0.5",
        "--selector-skip-thresholds-per-500",
        "0.0,0.05,0.1",
        "--min-selected-bet-rate",
        "0.05",
    ]
    if bool(args.no_resume):
        argv.append("--no-resume")
    for spec in flow_profiles:
        argv.extend(["--flow-profile", str(spec)])
    for spec in dislocation_profiles:
        argv.extend(["--dislocation-profile", str(spec)])
    return argv


def _shadow_argv(
    *,
    python_exe: str,
    args: argparse.Namespace,
    compare_csv: Path,
) -> list[str]:
    return [
        str(python_exe),
        "-m",
        "inspection.run_profile_set_shadow_recommender",
        "--compare-csv",
        str(compare_csv),
        "--name-prefix",
        str(args.name_prefix),
        "--mode",
        str(args.mode),
        "--feature-lookbacks",
        str(args.feature_lookbacks),
        "--min-train-windows",
        str(int(args.min_train_windows)),
        "--min-hold-windows",
        str(int(args.min_hold_windows)),
        "--margin-per-500",
        str(float(args.margin_per_500)),
        "--skip-threshold-per-500",
        str(float(args.skip_threshold_per_500)),
        "--ridge-alpha",
        str(float(args.ridge_alpha)),
        "--logistic-c",
        str(float(args.logistic_c)),
        "--output-dir",
        str(args.output_dir),
    ]


def main() -> None:
    args = _build_parser().parse_args()
    cwd = Path.cwd().resolve()
    python_exe = Path.cwd().resolve() / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        raise FileNotFoundError(f"profile_set_shadow_refresh_python_missing: {python_exe}")
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_profiles = _flow_profiles_or_default(list(args.flow_profile))
    dislocation_profiles = _dislocation_profiles_or_default(list(args.dislocation_profile))
    compare_csv = output_dir / f"{args.name_prefix}_profile_set_window_compare.csv"
    recommendation_json = output_dir / f"{args.name_prefix}_profile_set_shadow_recommendation.json"

    _run_subprocess(
        argv=_window_selector_argv(
            python_exe=str(python_exe),
            args=args,
            flow_profiles=flow_profiles,
            dislocation_profiles=dislocation_profiles,
        ),
        cwd=cwd,
    )
    _run_subprocess(
        argv=_shadow_argv(
            python_exe=str(python_exe),
            args=args,
            compare_csv=compare_csv,
        ),
        cwd=cwd,
    )
    recommendation = json.loads(recommendation_json.read_text(encoding="utf-8"))
    summary = {
        "name_prefix": str(args.name_prefix),
        "compare_csv": str(compare_csv),
        "recommendation_json": str(recommendation_json),
        "chosen_profile": str(recommendation.get("chosen_profile", "")),
        "chosen_predicted_per_500": float(recommendation.get("chosen_predicted_per_500", 0.0)),
        "estimated_selected_bet_rate": float(recommendation.get("estimated_selected_bet_rate", 0.0)),
    }
    summary_path = output_dir / f"{args.name_prefix}_profile_set_shadow_refresh_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
