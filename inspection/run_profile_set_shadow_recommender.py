from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from pancakebot.core.errors import InvariantError
from inspection.run_profile_set_model_selector import (
    _load_compare_rows,
    _parse_float_list,
    _parse_positive_int_list,
    _predict_next_recommendation,
)

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-csv", type=str, required=True)
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=("delta_ridge", "delta_logistic"), default="delta_ridge")
    parser.add_argument("--baseline-profile-name", type=str, default="stageb")
    parser.add_argument("--feature-lookbacks", type=str, default="1,3,5,8")
    parser.add_argument("--min-train-windows", type=int, default=10)
    parser.add_argument("--min-hold-windows", type=int, default=1)
    parser.add_argument("--margin-per-500", type=float, default=-0.2)
    parser.add_argument("--skip-threshold-per-500", type=float, default=0.0)
    parser.add_argument("--ridge-alpha", type=float, default=2.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.min_train_windows) <= 0:
        raise InvariantError("profile_set_shadow_min_train_windows_nonpositive")
    if int(args.min_hold_windows) <= 0:
        raise InvariantError("profile_set_shadow_min_hold_windows_nonpositive")
    profiles, rows = _load_compare_rows(Path(str(args.compare_csv)).resolve())
    baseline_profile_name = str(args.baseline_profile_name).strip()
    if baseline_profile_name == "" or str(baseline_profile_name) not in profiles:
        raise InvariantError("profile_set_shadow_baseline_profile_invalid")
    feature_lookbacks = _parse_positive_int_list(str(args.feature_lookbacks))
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    recommendation = _predict_next_recommendation(
        mode=str(args.mode),
        rows=rows,
        profiles=profiles,
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
        min_train_windows=int(args.min_train_windows),
        min_hold_windows=int(args.min_hold_windows),
        ridge_alpha=float(args.ridge_alpha),
        logistic_c=float(args.logistic_c),
        margin_per_500=float(args.margin_per_500),
        skip_threshold_per_500=float(args.skip_threshold_per_500),
    )
    latest_completed = rows[-1]
    out = dict(asdict(recommendation))
    out["profile_names"] = [str(x) for x in profiles]
    out["latest_completed_tail_offset_rounds"] = int(latest_completed.tail_offset_rounds)
    out["latest_completed_per_500_json"] = json.dumps(
        {str(k): float(v) for k, v in sorted(latest_completed.per_500.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    out["latest_completed_bet_rate_json"] = json.dumps(
        {str(k): float(v) for k, v in sorted(latest_completed.bet_rate.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    json_path = output_dir / f"{args.name_prefix}_profile_set_shadow_recommendation.json"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
