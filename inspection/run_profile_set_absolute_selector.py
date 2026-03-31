from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from pancakebot.core.errors import InvariantError
from inspection.run_profile_window_selector import _parse_float_list, _parse_positive_int_list

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"
_ABSOLUTE_MODES = ("trailing_mean", "ewm_mean")


@dataclass(frozen=True, slots=True)
class CompareWindowRow:
    tail_offset_rounds: int
    per_500: dict[str, float]
    bet_rate: dict[str, float]


@dataclass(frozen=True, slots=True)
class AbsoluteSelectorResult:
    mode: str
    profile_names_json: str
    cold_start_profile_name: str
    lookback_windows: int
    min_history_windows: int
    ewm_alpha: float
    stability_penalty_per_500: float
    skip_threshold_per_500: float
    mean_predicted_per_500: float
    mean_realized_per_500: float
    mean_prediction_error_per_500: float
    mean_regret_vs_oracle_per_500: float
    mean_selected_bet_rate: float
    meets_min_selected_bet_rate: bool
    pick_counts_json: str


@dataclass(frozen=True, slots=True)
class AbsoluteWindowEvalRow:
    selector_rank: int
    selector_mode: str
    tail_offset_rounds: int
    window_index: int
    chosen_action: str
    predicted_per_500: float
    realized_chosen_per_500: float
    realized_chosen_bet_rate: float
    realized_stageb_per_500: float
    realized_oracle_with_skip_per_500: float
    regret_vs_oracle_per_500: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-csv", type=str, required=True)
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--profile-names", type=str, default="")
    parser.add_argument("--modes", type=str, default="trailing_mean,ewm_mean")
    parser.add_argument("--cold-start-profile-name", type=str, default="")
    parser.add_argument("--lookback-windows", type=str, default="2,3,5,8")
    parser.add_argument("--min-history-windows", type=str, default="2,3,5")
    parser.add_argument("--ewm-alphas", type=str, default="0.5,0.7,0.85")
    parser.add_argument("--stability-penalties-per-500", type=str, default="0.0,0.25,0.5")
    parser.add_argument("--skip-thresholds-per-500", type=str, default="0.0,0.05,0.1")
    parser.add_argument("--min-selected-bet-rate", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    parser.add_argument("--write-top-n-window-evals", type=int, default=3)
    return parser


def _safe_float(raw: str | None) -> float:
    text = str(raw).strip() if raw is not None else ""
    if text == "":
        raise InvariantError("profile_set_absolute_float_missing")
    value = float(text)
    if not math.isfinite(value):
        raise InvariantError("profile_set_absolute_float_nonfinite")
    return float(value)


def _parse_mode_list(raw: str) -> list[str]:
    values = [str(token).strip() for token in str(raw).split(",") if str(token).strip() != ""]
    if not values:
        raise InvariantError("profile_set_absolute_modes_empty")
    invalid = [str(value) for value in values if str(value) not in _ABSOLUTE_MODES]
    if invalid:
        raise InvariantError(f"profile_set_absolute_modes_invalid: {','.join(invalid)}")
    return values


def _load_compare_rows(compare_csv: Path) -> tuple[list[str], list[CompareWindowRow]]:
    if not compare_csv.exists():
        raise FileNotFoundError(f"profile_set_absolute_compare_missing: {compare_csv}")
    rows: list[CompareWindowRow] = []
    with compare_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InvariantError("profile_set_absolute_compare_header_missing")
        fieldnames = [str(x) for x in reader.fieldnames]
        profile_names = [
            str(name[: -len("_per_500")])
            for name in fieldnames
            if str(name).endswith("_per_500") and str(name) != "tail_offset_rounds"
        ]
        if not profile_names:
            raise InvariantError("profile_set_absolute_profile_columns_missing")
        for raw in reader:
            per_500 = {str(name): _safe_float(raw.get(f"{name}_per_500")) for name in profile_names}
            bet_rate = {str(name): _safe_float(raw.get(f"{name}_bet_rate")) for name in profile_names}
            rows.append(
                CompareWindowRow(
                    tail_offset_rounds=int(raw["tail_offset_rounds"]),
                    per_500=per_500,
                    bet_rate=bet_rate,
                )
            )
    if not rows:
        raise InvariantError("profile_set_absolute_compare_rows_empty")
    return [str(name) for name in profile_names], sorted(rows, key=lambda row: int(row.tail_offset_rounds), reverse=True)


def _resolve_profile_names(raw: str, *, available_profiles: list[str]) -> list[str]:
    requested = [str(token).strip() for token in str(raw).split(",") if str(token).strip() != ""]
    if not requested:
        return list(available_profiles)
    missing = [str(name) for name in requested if str(name) not in set(available_profiles)]
    if missing:
        raise InvariantError(f"profile_set_absolute_profiles_missing: {','.join(missing)}")
    return requested


def _history_slice(*, rows: list[CompareWindowRow], idx: int, lookback_windows: int) -> list[CompareWindowRow]:
    return rows[max(0, int(idx) - int(lookback_windows)) : int(idx)]


def _ewm_weights(length: int, alpha: float) -> list[float]:
    weights = [float(alpha) ** float(length - 1 - idx) for idx in range(int(length))]
    total = float(sum(weights))
    if float(total) <= 0.0:
        raise InvariantError("profile_set_absolute_ewm_weights_nonpositive")
    return [float(weight / total) for weight in weights]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    return float(sum(float(value) * float(weight) for value, weight in zip(values, weights)))


def _std(values: list[float]) -> float:
    mean_value = _mean(values)
    return math.sqrt(float(sum((float(value) - mean_value) ** 2 for value in values) / len(values)))


def _estimate_profile(
    *,
    history: list[CompareWindowRow],
    profile_name: str,
    mode: str,
    ewm_alpha: float,
    stability_penalty_per_500: float,
) -> tuple[float, float]:
    values = [float(row.per_500[str(profile_name)]) for row in history]
    bet_rates = [float(row.bet_rate[str(profile_name)]) for row in history]
    if str(mode) == "trailing_mean":
        raw_estimate = _mean(values)
        bet_rate_estimate = _mean(bet_rates)
    elif str(mode) == "ewm_mean":
        weights = _ewm_weights(len(values), float(ewm_alpha))
        raw_estimate = _weighted_mean(values, weights)
        bet_rate_estimate = _weighted_mean(bet_rates, weights)
    else:
        raise InvariantError("profile_set_absolute_mode_invalid")
    stability_penalty = float(stability_penalty_per_500) * _std(values) if len(values) > 1 else 0.0
    return float(raw_estimate - stability_penalty), float(bet_rate_estimate)


def _pick_absolute_window(
    *,
    rows: list[CompareWindowRow],
    idx: int,
    profile_names: list[str],
    cold_start_profile_name: str,
    mode: str,
    lookback_windows: int,
    min_history_windows: int,
    ewm_alpha: float,
    stability_penalty_per_500: float,
    skip_threshold_per_500: float,
) -> tuple[str, float, float]:
    history = _history_slice(rows=rows, idx=int(idx), lookback_windows=int(lookback_windows))
    if len(history) < int(min_history_windows):
        if str(cold_start_profile_name) != "":
            return str(cold_start_profile_name), 0.0, 0.0
        return "skip", 0.0, 0.0
    best_profile = "skip"
    best_estimate = 0.0
    best_bet_rate_estimate = 0.0
    for profile_name in profile_names:
        estimate, bet_rate_estimate = _estimate_profile(
            history=history,
            profile_name=str(profile_name),
            mode=str(mode),
            ewm_alpha=float(ewm_alpha),
            stability_penalty_per_500=float(stability_penalty_per_500),
        )
        if float(estimate) > float(best_estimate):
            best_profile = str(profile_name)
            best_estimate = float(estimate)
            best_bet_rate_estimate = float(bet_rate_estimate)
    if float(best_estimate) <= float(skip_threshold_per_500):
        return "skip", 0.0, 0.0
    return str(best_profile), float(best_estimate), float(best_bet_rate_estimate)


def _evaluate_static_profile(
    *,
    rows: list[CompareWindowRow],
    profile_name: str,
) -> tuple[AbsoluteSelectorResult, list[AbsoluteWindowEvalRow]]:
    total_predicted = 0.0
    total_realized = 0.0
    total_error = 0.0
    total_regret = 0.0
    total_bet_rate = 0.0
    picks = Counter()
    eval_rows: list[AbsoluteWindowEvalRow] = []
    for idx, row in enumerate(rows):
        realized = float(row.per_500[str(profile_name)])
        bet_rate = float(row.bet_rate[str(profile_name)])
        oracle_with_skip = max(0.0, *(float(value) for value in row.per_500.values()))
        total_predicted += float(realized)
        total_realized += float(realized)
        total_error += 0.0
        total_regret += float(oracle_with_skip - realized)
        total_bet_rate += float(bet_rate)
        picks[str(profile_name)] += 1
        eval_rows.append(
            AbsoluteWindowEvalRow(
                selector_rank=0,
                selector_mode="static_profile",
                tail_offset_rounds=int(row.tail_offset_rounds),
                window_index=int(idx),
                chosen_action=str(profile_name),
                predicted_per_500=float(realized),
                realized_chosen_per_500=float(realized),
                realized_chosen_bet_rate=float(bet_rate),
                realized_stageb_per_500=float(row.per_500.get("stageb", realized)),
                realized_oracle_with_skip_per_500=float(oracle_with_skip),
                regret_vs_oracle_per_500=float(oracle_with_skip - realized),
            )
        )
    count = len(rows)
    return (
        AbsoluteSelectorResult(
            mode="static_profile",
            profile_names_json=json.dumps([str(profile_name)], separators=(",", ":")),
            cold_start_profile_name="",
            lookback_windows=0,
            min_history_windows=0,
            ewm_alpha=0.0,
            stability_penalty_per_500=0.0,
            skip_threshold_per_500=0.0,
            mean_predicted_per_500=float(total_predicted / count),
            mean_realized_per_500=float(total_realized / count),
            mean_prediction_error_per_500=float(total_error / count),
            mean_regret_vs_oracle_per_500=float(total_regret / count),
            mean_selected_bet_rate=float(total_bet_rate / count),
            meets_min_selected_bet_rate=False,
            pick_counts_json=json.dumps(dict(sorted(picks.items())), separators=(",", ":"), sort_keys=True),
        ),
        eval_rows,
    )


def _evaluate_oracle_with_skip(
    *,
    rows: list[CompareWindowRow],
    profile_names: list[str],
) -> tuple[AbsoluteSelectorResult, list[AbsoluteWindowEvalRow]]:
    total_bet_rate = 0.0
    picks = Counter()
    eval_rows: list[AbsoluteWindowEvalRow] = []
    for idx, row in enumerate(rows):
        best_profile = "skip"
        best_value = 0.0
        best_bet_rate = 0.0
        for profile_name in profile_names:
            realized = float(row.per_500[str(profile_name)])
            if float(realized) > float(best_value):
                best_profile = str(profile_name)
                best_value = float(realized)
                best_bet_rate = float(row.bet_rate[str(profile_name)])
        total_bet_rate += float(best_bet_rate)
        picks[str(best_profile)] += 1
        eval_rows.append(
            AbsoluteWindowEvalRow(
                selector_rank=0,
                selector_mode="oracle_with_skip",
                tail_offset_rounds=int(row.tail_offset_rounds),
                window_index=int(idx),
                chosen_action=str(best_profile),
                predicted_per_500=float(best_value),
                realized_chosen_per_500=float(best_value),
                realized_chosen_bet_rate=float(best_bet_rate),
                realized_stageb_per_500=float(row.per_500.get("stageb", 0.0)),
                realized_oracle_with_skip_per_500=float(best_value),
                regret_vs_oracle_per_500=0.0,
            )
        )
    count = len(rows)
    total = float(sum(float(eval_row.realized_chosen_per_500) for eval_row in eval_rows))
    return (
        AbsoluteSelectorResult(
            mode="oracle_with_skip",
            profile_names_json=json.dumps(list(profile_names), separators=(",", ":")),
            cold_start_profile_name="",
            lookback_windows=0,
            min_history_windows=0,
            ewm_alpha=0.0,
            stability_penalty_per_500=0.0,
            skip_threshold_per_500=0.0,
            mean_predicted_per_500=float(total / count),
            mean_realized_per_500=float(total / count),
            mean_prediction_error_per_500=0.0,
            mean_regret_vs_oracle_per_500=0.0,
            mean_selected_bet_rate=float(total_bet_rate / count),
            meets_min_selected_bet_rate=False,
            pick_counts_json=json.dumps(dict(sorted(picks.items())), separators=(",", ":"), sort_keys=True),
        ),
        eval_rows,
    )


def _evaluate_absolute_selectors(
    *,
    rows: list[CompareWindowRow],
    profile_names: list[str],
    cold_start_profile_name: str,
    modes: list[str],
    lookback_windows_list: list[int],
    min_history_windows_list: list[int],
    ewm_alphas: list[float],
    stability_penalties_per_500: list[float],
    skip_thresholds_per_500: list[float],
    min_selected_bet_rate: float,
) -> tuple[list[AbsoluteSelectorResult], list[list[AbsoluteWindowEvalRow]]]:
    results: list[AbsoluteSelectorResult] = []
    eval_sets: list[list[AbsoluteWindowEvalRow]] = []

    for profile_name in profile_names:
        static_result, static_evals = _evaluate_static_profile(rows=rows, profile_name=str(profile_name))
        results.append(
            AbsoluteSelectorResult(
                **{
                    **asdict(static_result),
                    "meets_min_selected_bet_rate": float(static_result.mean_selected_bet_rate) >= float(min_selected_bet_rate),
                }
            )
        )
        eval_sets.append(static_evals)
    oracle_result, oracle_evals = _evaluate_oracle_with_skip(rows=rows, profile_names=profile_names)
    results.append(
        AbsoluteSelectorResult(
            **{
                **asdict(oracle_result),
                "meets_min_selected_bet_rate": float(oracle_result.mean_selected_bet_rate) >= float(min_selected_bet_rate),
            }
        )
    )
    eval_sets.append(oracle_evals)

    for mode in modes:
        active_ewm_alphas = ewm_alphas if str(mode) == "ewm_mean" else [0.0]
        for lookback_windows in lookback_windows_list:
            for min_history_windows in min_history_windows_list:
                if int(min_history_windows) > int(lookback_windows):
                    continue
                for ewm_alpha in active_ewm_alphas:
                    for stability_penalty_per_500 in stability_penalties_per_500:
                        for skip_threshold_per_500 in skip_thresholds_per_500:
                            total_predicted = 0.0
                            total_realized = 0.0
                            total_error = 0.0
                            total_regret = 0.0
                            total_bet_rate = 0.0
                            picks = Counter()
                            eval_rows: list[AbsoluteWindowEvalRow] = []
                            for idx, row in enumerate(rows):
                                chosen_action, predicted, _predicted_bet_rate = _pick_absolute_window(
                                    rows=rows,
                                    idx=int(idx),
                                    profile_names=profile_names,
                                    cold_start_profile_name=str(cold_start_profile_name),
                                    mode=str(mode),
                                    lookback_windows=int(lookback_windows),
                                    min_history_windows=int(min_history_windows),
                                    ewm_alpha=float(ewm_alpha),
                                    stability_penalty_per_500=float(stability_penalty_per_500),
                                    skip_threshold_per_500=float(skip_threshold_per_500),
                                )
                                realized = 0.0 if str(chosen_action) == "skip" else float(row.per_500[str(chosen_action)])
                                realized_bet_rate = 0.0 if str(chosen_action) == "skip" else float(row.bet_rate[str(chosen_action)])
                                oracle_with_skip = max(0.0, *(float(value) for value in row.per_500.values()))
                                total_predicted += float(predicted)
                                total_realized += float(realized)
                                total_error += float(predicted - realized)
                                total_regret += float(oracle_with_skip - realized)
                                total_bet_rate += float(realized_bet_rate)
                                picks[str(chosen_action)] += 1
                                eval_rows.append(
                                    AbsoluteWindowEvalRow(
                                        selector_rank=0,
                                        selector_mode=str(mode),
                                        tail_offset_rounds=int(row.tail_offset_rounds),
                                        window_index=int(idx),
                                        chosen_action=str(chosen_action),
                                        predicted_per_500=float(predicted),
                                        realized_chosen_per_500=float(realized),
                                        realized_chosen_bet_rate=float(realized_bet_rate),
                                        realized_stageb_per_500=float(row.per_500.get("stageb", 0.0)),
                                        realized_oracle_with_skip_per_500=float(oracle_with_skip),
                                        regret_vs_oracle_per_500=float(oracle_with_skip - realized),
                                    )
                                )
                            count = len(rows)
                            results.append(
                                AbsoluteSelectorResult(
                                    mode=str(mode),
                                    profile_names_json=json.dumps(list(profile_names), separators=(",", ":")),
                                    cold_start_profile_name=str(cold_start_profile_name),
                                    lookback_windows=int(lookback_windows),
                                    min_history_windows=int(min_history_windows),
                                    ewm_alpha=float(ewm_alpha),
                                    stability_penalty_per_500=float(stability_penalty_per_500),
                                    skip_threshold_per_500=float(skip_threshold_per_500),
                                    mean_predicted_per_500=float(total_predicted / count),
                                    mean_realized_per_500=float(total_realized / count),
                                    mean_prediction_error_per_500=float(total_error / count),
                                    mean_regret_vs_oracle_per_500=float(total_regret / count),
                                    mean_selected_bet_rate=float(total_bet_rate / count),
                                    meets_min_selected_bet_rate=float(total_bet_rate / count) >= float(min_selected_bet_rate),
                                    pick_counts_json=json.dumps(dict(sorted(picks.items())), separators=(",", ":"), sort_keys=True),
                                )
                            )
                            eval_sets.append(eval_rows)

    ranked = sorted(
        enumerate(results),
        key=lambda item: (
            1 if bool(item[1].meets_min_selected_bet_rate) else 0,
            float(item[1].mean_realized_per_500),
            -float(item[1].mean_regret_vs_oracle_per_500),
            float(item[1].mean_selected_bet_rate),
        ),
        reverse=True,
    )
    ranked_results: list[AbsoluteSelectorResult] = []
    ranked_eval_sets: list[list[AbsoluteWindowEvalRow]] = []
    for rank, (original_index, result) in enumerate(ranked, start=1):
        ranked_results.append(result)
        ranked_eval_sets.append(
            [
                AbsoluteWindowEvalRow(**{**asdict(eval_row), "selector_rank": int(rank)})
                for eval_row in eval_sets[int(original_index)]
            ]
        )
    return ranked_results, ranked_eval_sets


def main() -> None:
    args = _build_parser().parse_args()
    if float(args.min_selected_bet_rate) < 0.0:
        raise InvariantError("profile_set_absolute_min_selected_bet_rate_negative")
    if int(args.write_top_n_window_evals) < 0:
        raise InvariantError("profile_set_absolute_write_top_n_negative")

    available_profiles, rows = _load_compare_rows(Path(str(args.compare_csv)).resolve())
    profile_names = _resolve_profile_names(str(args.profile_names), available_profiles=available_profiles)
    cold_start_profile_name = str(args.cold_start_profile_name).strip()
    if str(cold_start_profile_name) != "" and str(cold_start_profile_name) not in set(profile_names):
        raise InvariantError("profile_set_absolute_cold_start_profile_invalid")
    modes = _parse_mode_list(str(args.modes))
    lookback_windows_list = _parse_positive_int_list(str(args.lookback_windows))
    min_history_windows_list = _parse_positive_int_list(str(args.min_history_windows))
    ewm_alphas = _parse_float_list(str(args.ewm_alphas))
    stability_penalties_per_500 = _parse_float_list(str(args.stability_penalties_per_500))
    skip_thresholds_per_500 = _parse_float_list(str(args.skip_thresholds_per_500))

    for alpha in ewm_alphas:
        if float(alpha) <= 0.0 or float(alpha) > 1.0:
            raise InvariantError("profile_set_absolute_ewm_alpha_invalid")

    results, ranked_eval_sets = _evaluate_absolute_selectors(
        rows=rows,
        profile_names=profile_names,
        cold_start_profile_name=str(cold_start_profile_name),
        modes=modes,
        lookback_windows_list=lookback_windows_list,
        min_history_windows_list=min_history_windows_list,
        ewm_alphas=ewm_alphas,
        stability_penalties_per_500=stability_penalties_per_500,
        skip_thresholds_per_500=skip_thresholds_per_500,
        min_selected_bet_rate=float(args.min_selected_bet_rate),
    )

    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.name_prefix}_profile_set_absolute_selectors.csv"
    json_path = output_dir / f"{args.name_prefix}_profile_set_absolute_selectors.json"
    best_eval_csv_path = output_dir / f"{args.name_prefix}_profile_set_absolute_best_window_eval.csv"
    best_eval_json_path = output_dir / f"{args.name_prefix}_profile_set_absolute_best_window_eval.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    json_path.write_text(
        json.dumps([asdict(row) for row in results], indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )

    best_eval_rows: list[AbsoluteWindowEvalRow] = []
    for eval_rows in ranked_eval_sets[: int(args.write_top_n_window_evals)]:
        best_eval_rows.extend(eval_rows)
    if best_eval_rows:
        with best_eval_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(best_eval_rows[0]).keys()))
            writer.writeheader()
            for row in best_eval_rows:
                writer.writerow(asdict(row))
        best_eval_json_path.write_text(
            json.dumps([asdict(row) for row in best_eval_rows], indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
    else:
        best_eval_csv_path.write_text("", encoding="utf-8", newline="\n")
        best_eval_json_path.write_text("[]\n", encoding="utf-8")

    top = results[0]
    print(
        json.dumps(
            {
                "best_mode": str(top.mode),
                "best_mean_realized_per_500": float(top.mean_realized_per_500),
                "best_mean_selected_bet_rate": float(top.mean_selected_bet_rate),
                "meets_min_selected_bet_rate": bool(top.meets_min_selected_bet_rate),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
