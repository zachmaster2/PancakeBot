from __future__ import annotations

import argparse
from pathlib import Path
import re

from pancakebot.core.errors import InvariantError

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"
_ACTIVE_CANDIDATES_PATTERN = re.compile(
    r"^active_candidate_names\s*=\s*\[(?:.|\n)*?^\]",
    re.MULTILINE,
)
_WINDOW_CONTROLLER_SECTION_PATTERN = re.compile(
    r"(?ms)^\[strategy\.window_controller\]\n(?:.*?\n)*(?=^\[|\Z)"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument(
        "--active-candidate-names",
        type=str,
        default="disloc_stageB_bullonly_recent8pct_v1,disloc_stageG2_bullonly_recent5pct_v1,disloc_altB_20260227_x80",
    )
    parser.add_argument("--window-controller-enabled", type=str, default="true")
    parser.add_argument(
        "--window-controller-mode",
        type=str,
        choices=("absolute_best_with_skip",),
        default="absolute_best_with_skip",
    )
    parser.add_argument(
        "--window-controller-profile-names",
        type=str,
        default="disloc_stageB_bullonly_recent8pct_v1,disloc_stageG2_bullonly_recent5pct_v1,disloc_altB_20260227_x80",
    )
    parser.add_argument(
        "--window-controller-cold-start-profile-name",
        type=str,
        default="disloc_stageB_bullonly_recent8pct_v1",
    )
    parser.add_argument("--window-controller-window-rounds", type=int, default=216)
    parser.add_argument("--window-controller-lookback-windows", type=int, default=2)
    parser.add_argument("--window-controller-min-history-windows", type=int, default=2)
    parser.add_argument(
        "--window-controller-estimator-mode",
        type=str,
        choices=("trailing_mean", "ewm_mean"),
        default="ewm_mean",
    )
    parser.add_argument("--window-controller-ewm-alpha", type=float, default=0.85)
    parser.add_argument("--window-controller-stability-penalty-per-500", type=float, default=0.0)
    parser.add_argument("--window-controller-activity-target-bet-rate", type=float, default=0.0)
    parser.add_argument("--window-controller-activity-shortfall-penalty-per-500", type=float, default=0.0)
    parser.add_argument("--window-controller-skip-threshold-per-500", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _parse_names(raw: str) -> list[str]:
    names = [str(token).strip() for token in str(raw).split(",") if str(token).strip()]
    if not names:
        raise InvariantError("window_controller_runtime_config_active_candidates_empty")
    return names


def _bool_token(raw: str) -> str:
    token = str(raw).strip().lower()
    if token in {"true", "1", "yes", "on"}:
        return "true"
    if token in {"false", "0", "no", "off"}:
        return "false"
    raise InvariantError("window_controller_runtime_config_bool_invalid")


def _replace_active_candidates(*, config_text: str, active_candidate_names: list[str]) -> str:
    if _ACTIVE_CANDIDATES_PATTERN.search(config_text) is None:
        raise InvariantError("window_controller_runtime_config_active_candidate_block_missing")
    replacement = "active_candidate_names = [\n" + "".join(
        f'  "{str(name)}",\n' for name in active_candidate_names
    ) + "]"
    return _ACTIVE_CANDIDATES_PATTERN.sub(replacement, config_text, count=1)


def _replace_window_controller_section(
    *,
    config_text: str,
    enabled: str,
    mode: str,
    profile_names: list[str],
    cold_start_profile_name: str,
    window_rounds: int,
    lookback_windows: int,
    min_history_windows: int,
    estimator_mode: str,
    ewm_alpha: float,
    stability_penalty_per_500: float,
    activity_target_bet_rate: float,
    activity_shortfall_penalty_per_500: float,
    skip_threshold_per_500: float,
) -> str:
    if _WINDOW_CONTROLLER_SECTION_PATTERN.search(config_text) is None:
        raise InvariantError("window_controller_runtime_config_section_missing")
    replacement = (
        "[strategy.window_controller]\n"
        "# Research-only window controller. Disabled by default until it clears\n"
        "# broader causal backtests and a controlled dry rollout.\n"
        f"enabled = {str(enabled)}\n"
        f'mode = "{str(mode)}"\n'
        "profile_names = [\n"
        + "".join(f'  "{str(name)}",\n' for name in profile_names)
        + "]\n"
        f'cold_start_profile_name = "{str(cold_start_profile_name)}"\n'
        f"window_rounds = {int(window_rounds)}\n"
        f"lookback_windows = {int(lookback_windows)}\n"
        f"min_history_windows = {int(min_history_windows)}\n"
        f'estimator_mode = "{str(estimator_mode)}"\n'
        f"ewm_alpha = {float(ewm_alpha)}\n"
        f"stability_penalty_per_500 = {float(stability_penalty_per_500)}\n"
        f"activity_target_bet_rate = {float(activity_target_bet_rate)}\n"
        f"activity_shortfall_penalty_per_500 = {float(activity_shortfall_penalty_per_500)}\n"
        f"skip_threshold_per_500 = {float(skip_threshold_per_500)}\n\n"
    )
    return _WINDOW_CONTROLLER_SECTION_PATTERN.sub(replacement, config_text, count=1)


def write_runtime_config(
    *,
    base_config_path: Path,
    output_dir: Path,
    name_prefix: str,
    active_candidate_names: list[str],
    enabled: str,
    mode: str,
    profile_names: list[str],
    cold_start_profile_name: str,
    window_rounds: int,
    lookback_windows: int,
    min_history_windows: int,
    estimator_mode: str,
    ewm_alpha: float,
    stability_penalty_per_500: float,
    activity_target_bet_rate: float,
    activity_shortfall_penalty_per_500: float,
    skip_threshold_per_500: float,
) -> Path:
    config_text = base_config_path.read_text(encoding="utf-8")
    patched = _replace_active_candidates(
        config_text=config_text,
        active_candidate_names=active_candidate_names,
    )
    patched = _replace_window_controller_section(
        config_text=patched,
        enabled=str(enabled),
        mode=str(mode),
        profile_names=list(profile_names),
        cold_start_profile_name=str(cold_start_profile_name),
        window_rounds=int(window_rounds),
        lookback_windows=int(lookback_windows),
        min_history_windows=int(min_history_windows),
        estimator_mode=str(estimator_mode),
        ewm_alpha=float(ewm_alpha),
        stability_penalty_per_500=float(stability_penalty_per_500),
        activity_target_bet_rate=float(activity_target_bet_rate),
        activity_shortfall_penalty_per_500=float(activity_shortfall_penalty_per_500),
        skip_threshold_per_500=float(skip_threshold_per_500),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (output_dir / f"{name_prefix}_window_controller_runtime.toml").resolve()
    out_path.write_text(patched, encoding="utf-8", newline="\n")
    return out_path


def main() -> None:
    args = _build_parser().parse_args()
    names = _parse_names(str(args.active_candidate_names))
    profile_names = _parse_names(str(args.window_controller_profile_names))
    enabled = _bool_token(str(args.window_controller_enabled))
    out_path = write_runtime_config(
        base_config_path=Path(str(args.base_config)).resolve(),
        output_dir=Path(str(args.output_dir)).resolve(),
        name_prefix=str(args.name_prefix),
        active_candidate_names=names,
        enabled=str(enabled),
        mode=str(args.window_controller_mode),
        profile_names=profile_names,
        cold_start_profile_name=str(args.window_controller_cold_start_profile_name),
        window_rounds=int(args.window_controller_window_rounds),
        lookback_windows=int(args.window_controller_lookback_windows),
        min_history_windows=int(args.window_controller_min_history_windows),
        estimator_mode=str(args.window_controller_estimator_mode),
        ewm_alpha=float(args.window_controller_ewm_alpha),
        stability_penalty_per_500=float(args.window_controller_stability_penalty_per_500),
        activity_target_bet_rate=float(args.window_controller_activity_target_bet_rate),
        activity_shortfall_penalty_per_500=float(
            args.window_controller_activity_shortfall_penalty_per_500
        ),
        skip_threshold_per_500=float(args.window_controller_skip_threshold_per_500),
    )
    print(str(out_path))


if __name__ == "__main__":
    main()
