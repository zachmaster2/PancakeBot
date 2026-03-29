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
        default="disloc_stageB_bullonly_recent8pct_v1,disloc_cons_20260227_x80",
    )
    parser.add_argument("--window-controller-enabled", type=str, default="true")
    parser.add_argument(
        "--window-controller-mode",
        type=str,
        choices=("trailing_best_vs_baseline", "trailing_best_vs_baseline_with_skip"),
        default="trailing_best_vs_baseline",
    )
    parser.add_argument("--window-controller-baseline-profile-name", type=str, default="disloc_stageB_bullonly_recent8pct_v1")
    parser.add_argument("--window-controller-alternate-profile-name", type=str, default="disloc_cons_20260227_x80")
    parser.add_argument("--window-controller-window-rounds", type=int, default=216)
    parser.add_argument("--window-controller-lookback-windows", type=int, default=3)
    parser.add_argument("--window-controller-margin-per-500", type=float, default=1.0)
    parser.add_argument("--window-controller-skip-threshold-per-500", type=float, default=0.0)
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
    baseline_profile_name: str,
    alternate_profile_name: str,
    window_rounds: int,
    lookback_windows: int,
    margin_per_500: float,
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
        f'baseline_profile_name = "{str(baseline_profile_name)}"\n'
        f'alternate_profile_name = "{str(alternate_profile_name)}"\n'
        f"window_rounds = {int(window_rounds)}\n"
        f"lookback_windows = {int(lookback_windows)}\n"
        f"margin_per_500 = {float(margin_per_500)}\n"
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
    baseline_profile_name: str,
    alternate_profile_name: str,
    window_rounds: int,
    lookback_windows: int,
    margin_per_500: float,
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
        baseline_profile_name=str(baseline_profile_name),
        alternate_profile_name=str(alternate_profile_name),
        window_rounds=int(window_rounds),
        lookback_windows=int(lookback_windows),
        margin_per_500=float(margin_per_500),
        skip_threshold_per_500=float(skip_threshold_per_500),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (output_dir / f"{name_prefix}_window_controller_runtime.toml").resolve()
    out_path.write_text(patched, encoding="utf-8", newline="\n")
    return out_path


def main() -> None:
    args = _build_parser().parse_args()
    names = _parse_names(str(args.active_candidate_names))
    enabled = _bool_token(str(args.window_controller_enabled))
    out_path = write_runtime_config(
        base_config_path=Path(str(args.base_config)).resolve(),
        output_dir=Path(str(args.output_dir)).resolve(),
        name_prefix=str(args.name_prefix),
        active_candidate_names=names,
        enabled=str(enabled),
        mode=str(args.window_controller_mode),
        baseline_profile_name=str(args.window_controller_baseline_profile_name),
        alternate_profile_name=str(args.window_controller_alternate_profile_name),
        window_rounds=int(args.window_controller_window_rounds),
        lookback_windows=int(args.window_controller_lookback_windows),
        margin_per_500=float(args.window_controller_margin_per_500),
        skip_threshold_per_500=float(args.window_controller_skip_threshold_per_500),
    )
    print(str(out_path))


if __name__ == "__main__":
    main()
