"""Score candidates by marginal pocket value relative to a reference set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from inspection.strategy_router_common import parse_strategy_prefixes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--dataset-csv", type=str, required=True)
    parser.add_argument("--dataset-meta", type=str, required=True)
    parser.add_argument("--reference-strategy-names", type=str, required=True)
    parser.add_argument("--candidate-strategy-names", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="../PancakeBot_var_exp")
    return parser


def _load_meta(path: Path) -> tuple[list[str], dict[str, str]]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    key_map = {str(k): str(v) for k, v in meta.get("strategy_column_keys", {}).items()}
    if not key_map:
        raise ValueError("candidate_pocket_report_key_map_missing")
    return list(key_map.keys()), key_map


def _streak_lengths(mask: list[bool]) -> list[int]:
    out: list[int] = []
    cur = 0
    for flag in mask:
        if bool(flag):
            cur += 1
        elif int(cur) > 0:
            out.append(int(cur))
            cur = 0
    if int(cur) > 0:
        out.append(int(cur))
    return out


def _streak_total(values: list[float], mask: list[bool], min_len: int) -> float:
    total = 0.0
    cur_values: list[float] = []
    for value, flag in zip(values, mask, strict=False):
        if bool(flag):
            cur_values.append(float(value))
        elif cur_values:
            if len(cur_values) >= int(min_len):
                total += float(sum(cur_values))
            cur_values = []
    if cur_values and len(cur_values) >= int(min_len):
        total += float(sum(cur_values))
    return float(total)


def main() -> None:
    args = _build_parser().parse_args()
    all_strategies, key_map = _load_meta(Path(str(args.dataset_meta)))
    reference_names = parse_strategy_prefixes(str(args.reference_strategy_names))
    for name in reference_names:
        if str(name) not in all_strategies:
            raise ValueError(f"candidate_pocket_report_reference_unknown: {name}")
    candidate_names = (
        parse_strategy_prefixes(str(args.candidate_strategy_names))
        if str(args.candidate_strategy_names).strip() != ""
        else [name for name in all_strategies if str(name) not in reference_names]
    )
    for name in candidate_names:
        if str(name) not in all_strategies:
            raise ValueError(f"candidate_pocket_report_candidate_unknown: {name}")

    target_rounds_total = 0
    rows: list[dict[str, Any]] = []
    with Path(str(args.dataset_csv)).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            target_rounds_total += int(raw["target_num_rounds"])
            profits = {
                str(name): float(raw[f"label_{key_map[str(name)]}_next_block_profit_bnb"])
                for name in all_strategies
            }
            rows.append(
                {
                    "target_block_index": int(raw["target_block_index"]),
                    "profits": profits,
                }
            )
    if not rows:
        raise ValueError("candidate_pocket_report_dataset_empty")

    ranked_rows: list[dict[str, Any]] = []
    for candidate_name in candidate_names:
        candidate_profit_series: list[float] = []
        marginal_gain_series: list[float] = []
        distinct_mask: list[bool] = []
        oracle_positive_blocks = 0
        beat_baseline_blocks = 0
        positive_blocks = 0
        reference_positive_overlap = 0

        for row in rows:
            profits = dict(row["profits"])
            reference_best = max([0.0] + [float(profits[str(name)]) for name in reference_names])
            candidate_profit = float(profits[str(candidate_name)])
            if float(candidate_profit) > 0.0:
                positive_blocks += 1
            if float(candidate_profit) > float(profits[str(reference_names[0])]):
                beat_baseline_blocks += 1
            if float(candidate_profit) > 0.0 and float(reference_best) > 0.0:
                reference_positive_overlap += 1
            marginal_gain = float(max(float(reference_best), float(candidate_profit), 0.0) - float(reference_best))
            distinct = bool(float(candidate_profit) > float(reference_best) and float(candidate_profit) > 0.0)
            if distinct:
                oracle_positive_blocks += 1
            candidate_profit_series.append(float(candidate_profit))
            marginal_gain_series.append(float(marginal_gain))
            distinct_mask.append(bool(distinct))

        streaks = _streak_lengths(distinct_mask)
        distinct_count = int(sum(1 for flag in distinct_mask if bool(flag)))
        transition_den = int(sum(1 for idx in range(len(distinct_mask) - 1) if bool(distinct_mask[idx])))
        transition_num = int(
            sum(
                1
                for idx in range(len(distinct_mask) - 1)
                if bool(distinct_mask[idx]) and bool(distinct_mask[idx + 1])
            )
        )
        ranked_rows.append(
            {
                "candidate_name": str(candidate_name),
                "marginal_gain_bnb": float(sum(marginal_gain_series)),
                "marginal_gain_per_500_rounds": float(sum(marginal_gain_series) / float(target_rounds_total) * 500.0),
                "distinct_profitable_blocks": int(distinct_count),
                "distinct_block_fraction": float(distinct_count / len(rows)),
                "positive_blocks": int(positive_blocks),
                "beat_baseline_blocks": int(beat_baseline_blocks),
                "reference_positive_overlap_blocks": int(reference_positive_overlap),
                "distinct_max_streak_len": int(max(streaks) if streaks else 0),
                "distinct_streak_count_ge2": int(sum(1 for x in streaks if int(x) >= 2)),
                "distinct_streak_count_ge3": int(sum(1 for x in streaks if int(x) >= 3)),
                "distinct_streak_marginal_gain_bnb_ge2": float(_streak_total(marginal_gain_series, distinct_mask, 2)),
                "distinct_streak_marginal_gain_bnb_ge3": float(_streak_total(marginal_gain_series, distinct_mask, 3)),
                "continuation_prob_next_distinct_given_distinct": (
                    0.0 if int(transition_den) <= 0 else float(transition_num) / float(transition_den)
                ),
                "oracle_positive_blocks_against_reference": int(oracle_positive_blocks),
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            float(row["marginal_gain_per_500_rounds"]),
            float(row["distinct_streak_marginal_gain_bnb_ge2"]),
            int(row["distinct_max_streak_len"]),
        ),
        reverse=True,
    )

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.name_prefix}_candidate_pocket_report.csv"
    json_path = output_dir / f"{args.name_prefix}_candidate_pocket_report.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranked_rows)
    json_path.write_text(
        json.dumps(
            {
                "reference_strategies": [str(x) for x in reference_names],
                "candidate_strategies": [str(x) for x in candidate_names],
                "rows": ranked_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"REPORT_CSV={csv_path}")
    print(f"REPORT_JSON={json_path}")
    if ranked_rows:
        print(f"BEST_CANDIDATE={ranked_rows[0]['candidate_name']}")
        print(f"BEST_MARGINAL_PER_500={ranked_rows[0]['marginal_gain_per_500_rounds']}")


if __name__ == "__main__":
    main()
