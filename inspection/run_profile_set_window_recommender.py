from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from pancakebot.core.errors import InvariantError
from inspection.run_profile_set_model_selector import CompareWindowRow, _load_compare_rows

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"
_VALID_MODES = (
    "skip_only",
    "static_profile",
    "prev_winner",
    "prev_winner_with_skip",
    "trailing_best_vs_stageb",
    "trailing_best_vs_stageb_with_skip",
)


@dataclass(frozen=True, slots=True)
class WindowHeuristicRecommendation:
    mode: str
    baseline_profile_name: str
    lookback: int
    margin_per_500: float
    skip_threshold_per_500: float
    training_window_count: int
    chosen_profile: str
    estimated_per_500: float
    estimated_selected_bet_rate: float
    trailing_per_500_json: str
    trailing_bet_rate_json: str
    latest_completed_tail_offset_rounds: int
    latest_completed_per_500_json: str
    latest_completed_bet_rate_json: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-csv", type=str, required=True)
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--baseline-profile-name", type=str, default="stageb")
    parser.add_argument("--static-profile-name", type=str, default="stageb")
    parser.add_argument("--lookback", type=int, default=1)
    parser.add_argument("--margin-per-500", type=float, default=0.0)
    parser.add_argument("--skip-threshold-per-500", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _best_profile(row: CompareWindowRow) -> tuple[str, float]:
    best_name = ""
    best_value = 0.0
    for name, value in row.per_500.items():
        if best_name == "" or float(value) > float(best_value):
            best_name = str(name)
            best_value = float(value)
    if best_name == "":
        raise InvariantError("profile_set_window_recommender_best_profile_missing")
    return str(best_name), float(best_value)


def _trailing_means(rows: list[CompareWindowRow], lookback: int) -> tuple[dict[str, float], dict[str, float]]:
    if not rows:
        return {}, {}
    hist = rows[-max(1, int(lookback)) :]
    profile_names = sorted(hist[0].per_500.keys())
    per_500 = {
        str(name): float(sum(float(row.per_500[str(name)]) for row in hist) / len(hist))
        for name in profile_names
    }
    bet_rate = {
        str(name): float(sum(float(row.bet_rate[str(name)]) for row in hist) / len(hist))
        for name in profile_names
    }
    return per_500, bet_rate


def predict_next_window_recommendation(
    *,
    rows: list[CompareWindowRow],
    mode: str,
    baseline_profile_name: str,
    static_profile_name: str,
    lookback: int,
    margin_per_500: float,
    skip_threshold_per_500: float,
) -> WindowHeuristicRecommendation:
    if str(mode) not in _VALID_MODES:
        raise InvariantError("profile_set_window_recommender_mode_invalid")
    if not rows:
        raise InvariantError("profile_set_window_recommender_rows_empty")
    baseline_name = str(baseline_profile_name)
    if baseline_name not in rows[0].per_500:
        raise InvariantError("profile_set_window_recommender_baseline_invalid")
    latest_completed = rows[-1]
    trailing_per_500, trailing_bet_rate = _trailing_means(rows, int(lookback))
    chosen_profile = "skip"
    estimated_per_500 = 0.0
    estimated_selected_bet_rate = 0.0
    if str(mode) == "skip_only":
        pass
    elif str(mode) == "static_profile":
        chosen_profile = str(static_profile_name)
        estimated_per_500 = float(trailing_per_500.get(str(chosen_profile), 0.0))
        estimated_selected_bet_rate = float(trailing_bet_rate.get(str(chosen_profile), 0.0))
    elif str(mode) == "prev_winner":
        chosen_profile, _ = _best_profile(latest_completed)
        estimated_per_500 = float(trailing_per_500.get(str(chosen_profile), 0.0))
        estimated_selected_bet_rate = float(trailing_bet_rate.get(str(chosen_profile), 0.0))
    elif str(mode) == "prev_winner_with_skip":
        winner, value = _best_profile(latest_completed)
        if float(value) > float(skip_threshold_per_500):
            chosen_profile = str(winner)
            estimated_per_500 = float(trailing_per_500.get(str(chosen_profile), 0.0))
            estimated_selected_bet_rate = float(trailing_bet_rate.get(str(chosen_profile), 0.0))
    else:
        baseline_mean = float(trailing_per_500.get(str(baseline_name), 0.0))
        best_profile = str(baseline_name)
        best_mean = float(baseline_mean)
        for profile_name, profile_mean in trailing_per_500.items():
            if str(profile_name) == str(baseline_name):
                continue
            if float(profile_mean) > float(best_mean):
                best_profile = str(profile_name)
                best_mean = float(profile_mean)
        if str(mode) == "trailing_best_vs_stageb_with_skip" and max(float(baseline_mean), float(best_mean)) <= float(skip_threshold_per_500):
            chosen_profile = "skip"
        elif str(best_profile) != str(baseline_name) and float(best_mean - baseline_mean) > float(margin_per_500):
            chosen_profile = str(best_profile)
        else:
            chosen_profile = str(baseline_name)
        if str(chosen_profile) != "skip":
            estimated_per_500 = float(trailing_per_500.get(str(chosen_profile), 0.0))
            estimated_selected_bet_rate = float(trailing_bet_rate.get(str(chosen_profile), 0.0))
    return WindowHeuristicRecommendation(
        mode=str(mode),
        baseline_profile_name=str(baseline_name),
        lookback=int(lookback),
        margin_per_500=float(margin_per_500),
        skip_threshold_per_500=float(skip_threshold_per_500),
        training_window_count=int(len(rows)),
        chosen_profile=str(chosen_profile),
        estimated_per_500=float(estimated_per_500),
        estimated_selected_bet_rate=float(estimated_selected_bet_rate),
        trailing_per_500_json=json.dumps(dict(sorted(trailing_per_500.items())), sort_keys=True, separators=(",", ":")),
        trailing_bet_rate_json=json.dumps(dict(sorted(trailing_bet_rate.items())), sort_keys=True, separators=(",", ":")),
        latest_completed_tail_offset_rounds=int(latest_completed.tail_offset_rounds),
        latest_completed_per_500_json=json.dumps(
            {str(k): float(v) for k, v in sorted(latest_completed.per_500.items())},
            sort_keys=True,
            separators=(",", ":"),
        ),
        latest_completed_bet_rate_json=json.dumps(
            {str(k): float(v) for k, v in sorted(latest_completed.bet_rate.items())},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def main() -> None:
    args = _build_parser().parse_args()
    profiles, rows = _load_compare_rows(Path(str(args.compare_csv)).resolve())
    if str(args.baseline_profile_name).strip() not in profiles:
        raise InvariantError("profile_set_window_recommender_baseline_invalid")
    if int(args.lookback) <= 0:
        raise InvariantError("profile_set_window_recommender_lookback_nonpositive")
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recommendation = predict_next_window_recommendation(
        rows=rows,
        mode=str(args.mode),
        baseline_profile_name=str(args.baseline_profile_name),
        static_profile_name=str(args.static_profile_name),
        lookback=int(args.lookback),
        margin_per_500=float(args.margin_per_500),
        skip_threshold_per_500=float(args.skip_threshold_per_500),
    )
    json_path = output_dir / f"{args.name_prefix}_profile_set_window_recommendation.json"
    json_path.write_text(
        json.dumps(asdict(recommendation), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
