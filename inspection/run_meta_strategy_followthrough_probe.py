"""Run continuation-style baseline overlay probes on a meta-strategy dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from inspection.strategy_router_common import parse_strategy_prefixes


_SCORE_MODES = (
    "followthrough",
    "last_delta",
    "positive_mean",
    "streak_weighted",
)


@dataclass(frozen=True, slots=True)
class FollowthroughRow:
    target_block_index: int
    target_sim_offset_rounds: int
    target_epoch_start: int
    target_epoch_end: int
    target_num_rounds: int
    profits_bnb: dict[str, float]
    num_bets: dict[str, int]
    bet_rates: dict[str, float]
    oracle_profit_bnb: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--dataset-csv", type=str, required=True)
    parser.add_argument("--dataset-meta", type=str, required=True)
    parser.add_argument("--baseline-strategy-name", type=str, required=True)
    parser.add_argument("--active-strategy-names", type=str, required=True)
    parser.add_argument("--score-mode", choices=_SCORE_MODES, default="followthrough")
    parser.add_argument("--required-streak-len", type=int, default=2)
    parser.add_argument("--min-transition-prob", type=float, default=0.25)
    parser.add_argument("--min-hold-blocks", type=int, default=4)
    parser.add_argument("--min-train-rows", type=int, default=12)
    parser.add_argument("--starting-bankroll-bnb", type=float, default=50.0)
    parser.add_argument("--output-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--write-decisions", action="store_true", default=False)
    return parser


def _tail_positive_streak_len(values: list[float]) -> int:
    count = 0
    for value in reversed(values):
        if float(value) <= 0.0:
            break
        count += 1
    return int(count)


def _positive_transition_prob(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    den = 0
    num = 0
    for first, second in zip(values, values[1:], strict=False):
        if float(first) > 0.0:
            den += 1
            if float(second) > 0.0:
                num += 1
    if int(den) <= 0:
        return None
    return float(num) / float(den)


def _positive_next_mean(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    next_values = [float(second) for first, second in zip(values, values[1:], strict=False) if float(first) > 0.0]
    if not next_values:
        return None
    return float(sum(next_values) / len(next_values))


def _positive_mean(values: list[float]) -> float | None:
    positive_values = [float(value) for value in values if float(value) > 0.0]
    if not positive_values:
        return None
    return float(sum(positive_values) / len(positive_values))


def _score_candidate(
    *,
    score_mode: str,
    delta_history: list[float],
) -> tuple[float | None, int, float | None, float | None]:
    streak_len = _tail_positive_streak_len(delta_history)
    transition_prob = _positive_transition_prob(delta_history)
    next_mean = _positive_next_mean(delta_history)
    positive_mean = _positive_mean(delta_history)

    if transition_prob is None or next_mean is None or float(next_mean) <= 0.0:
        return None, int(streak_len), transition_prob, next_mean

    if str(score_mode) == "followthrough":
        return float(transition_prob) * float(next_mean), int(streak_len), transition_prob, next_mean
    if str(score_mode) == "last_delta":
        return float(delta_history[-1]), int(streak_len), transition_prob, next_mean
    if str(score_mode) == "positive_mean":
        if positive_mean is None:
            return None, int(streak_len), transition_prob, next_mean
        return float(positive_mean), int(streak_len), transition_prob, next_mean
    if str(score_mode) == "streak_weighted":
        return (
            float(transition_prob) * float(next_mean) * float(min(int(streak_len), 3)),
            int(streak_len),
            transition_prob,
            next_mean,
        )
    raise ValueError("meta_strategy_followthrough_score_mode_unreachable")


def _load_meta(path: Path) -> tuple[list[str], dict[str, str]]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    key_map = {str(k): str(v) for k, v in meta.get("strategy_column_keys", {}).items()}
    if not key_map:
        raise ValueError("meta_strategy_followthrough_key_map_missing")
    return list(key_map.keys()), key_map


def _load_rows(dataset_csv: Path, key_map: dict[str, str]) -> list[FollowthroughRow]:
    rows: list[FollowthroughRow] = []
    with dataset_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            profits_bnb: dict[str, float] = {}
            num_bets: dict[str, int] = {}
            bet_rates: dict[str, float] = {}
            for strategy_name, column_key in key_map.items():
                profits_bnb[str(strategy_name)] = float(raw[f"label_{column_key}_next_block_profit_bnb"])
                num_bets[str(strategy_name)] = int(raw[f"label_{column_key}_next_block_num_bets"])
                bet_rates[str(strategy_name)] = float(raw[f"label_{column_key}_next_block_bet_rate"])
            rows.append(
                FollowthroughRow(
                    target_block_index=int(raw["target_block_index"]),
                    target_sim_offset_rounds=int(raw["target_sim_offset_rounds"]),
                    target_epoch_start=int(raw["target_epoch_start"]),
                    target_epoch_end=int(raw["target_epoch_end"]),
                    target_num_rounds=int(raw["target_num_rounds"]),
                    profits_bnb=profits_bnb,
                    num_bets=num_bets,
                    bet_rates=bet_rates,
                    oracle_profit_bnb=float(raw["label_oracle_profit_bnb"]),
                )
            )
    if not rows:
        raise ValueError("meta_strategy_followthrough_dataset_empty")
    return rows


def _run_probe(
    *,
    rows: list[FollowthroughRow],
    baseline_strategy_name: str,
    active_candidates: list[str],
    score_mode: str,
    required_streak_len: int,
    min_transition_prob: float,
    min_hold_blocks: int,
    min_train_rows: int,
    starting_bankroll_bnb: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    total_rounds = 0
    total_selected_bets = 0
    total_oracle_bnb = 0.0
    cumulative_net_bnb = 0.0
    peak_net_bnb = 0.0
    max_drawdown_bnb = 0.0
    min_bankroll_bnb = float(starting_bankroll_bnb)
    num_switches = 0
    incumbent_pick = str(baseline_strategy_name)
    hold_remaining = 0
    picks_by_strategy: Counter[str] = Counter()
    pick_reasons: Counter[str] = Counter()
    decision_rows: list[dict[str, object]] = []

    for idx, row in enumerate(rows):
        chosen_score: float | None = None
        chosen_streak_len: int | None = None
        chosen_transition_prob: float | None = None
        chosen_next_mean_bnb: float | None = None

        if str(incumbent_pick) != str(baseline_strategy_name) and int(hold_remaining) > 0:
            pick = str(incumbent_pick)
            hold_remaining -= 1
            pick_reason = "hold_interval"
        else:
            hist = rows[:idx]
            best_pick = str(baseline_strategy_name)
            best_score = float("-inf")
            best_streak_len = 0
            best_transition_prob = None
            best_next_mean_bnb = None
            if len(hist) >= int(min_train_rows):
                for candidate_name in active_candidates:
                    delta_history = [
                        float(hist_row.profits_bnb[str(candidate_name)])
                        - float(hist_row.profits_bnb[str(baseline_strategy_name)])
                        for hist_row in hist
                    ]
                    score, streak_len, transition_prob, next_mean_bnb = _score_candidate(
                        score_mode=str(score_mode),
                        delta_history=delta_history,
                    )
                    if int(streak_len) < int(required_streak_len):
                        continue
                    if transition_prob is None or float(transition_prob) < float(min_transition_prob):
                        continue
                    if score is None or float(score) <= 0.0:
                        continue
                    if float(score) > float(best_score):
                        best_score = float(score)
                        best_pick = str(candidate_name)
                        best_streak_len = int(streak_len)
                        best_transition_prob = transition_prob
                        best_next_mean_bnb = next_mean_bnb
            pick = str(best_pick)
            chosen_score = None if float(best_score) == float("-inf") else float(best_score)
            chosen_streak_len = int(best_streak_len) if str(pick) != str(baseline_strategy_name) else None
            chosen_transition_prob = best_transition_prob if str(pick) != str(baseline_strategy_name) else None
            chosen_next_mean_bnb = best_next_mean_bnb if str(pick) != str(baseline_strategy_name) else None
            if str(pick) != str(baseline_strategy_name):
                pick_reason = (
                    "switch_from_baseline"
                    if str(incumbent_pick) == str(baseline_strategy_name)
                    else "switch_between_candidates"
                )
                hold_remaining = max(0, int(min_hold_blocks) - 1)
            else:
                pick_reason = "stay_on_baseline"

        realized_profit_bnb = float(row.profits_bnb[str(pick)])
        realized_num_bets = int(row.num_bets[str(pick)])
        total_rounds += int(row.target_num_rounds)
        total_selected_bets += int(realized_num_bets)
        total_oracle_bnb += float(row.oracle_profit_bnb)
        cumulative_net_bnb += float(realized_profit_bnb)
        peak_net_bnb = max(float(peak_net_bnb), float(cumulative_net_bnb))
        max_drawdown_bnb = max(float(max_drawdown_bnb), float(peak_net_bnb) - float(cumulative_net_bnb))
        bankroll_bnb = float(starting_bankroll_bnb) + float(cumulative_net_bnb)
        min_bankroll_bnb = min(float(min_bankroll_bnb), float(bankroll_bnb))
        if idx > 0 and str(pick) != str(incumbent_pick):
            num_switches += 1
        incumbent_pick = str(pick)
        picks_by_strategy[str(pick)] += 1
        pick_reasons[str(pick_reason)] += 1

        decision_rows.append(
            {
                "target_block_index": int(row.target_block_index),
                "target_sim_offset_rounds": int(row.target_sim_offset_rounds),
                "target_epoch_start": int(row.target_epoch_start),
                "target_epoch_end": int(row.target_epoch_end),
                "pick": str(pick),
                "pick_reason": str(pick_reason),
                "pick_score": "" if chosen_score is None else float(chosen_score),
                "pick_tail_positive_streak_len": "" if chosen_streak_len is None else int(chosen_streak_len),
                "pick_transition_prob": "" if chosen_transition_prob is None else float(chosen_transition_prob),
                "pick_positive_next_mean_bnb": "" if chosen_next_mean_bnb is None else float(chosen_next_mean_bnb),
                "realized_profit_bnb": float(realized_profit_bnb),
                "realized_num_bets": int(realized_num_bets),
                "realized_bet_rate": float(row.bet_rates[str(pick)]),
                "baseline_profit_bnb": float(row.profits_bnb[str(baseline_strategy_name)]),
                "baseline_num_bets": int(row.num_bets[str(baseline_strategy_name)]),
                "baseline_bet_rate": float(row.bet_rates[str(baseline_strategy_name)]),
                "lift_vs_baseline_bnb": (
                    float(realized_profit_bnb) - float(row.profits_bnb[str(baseline_strategy_name)])
                ),
                "cum_net_bnb": float(cumulative_net_bnb),
                "drawdown_bnb": float(peak_net_bnb) - float(cumulative_net_bnb),
            }
        )

    summary = {
        "num_blocks": int(len(rows)),
        "num_rounds": int(total_rounds),
        "net_profit_bnb": float(cumulative_net_bnb),
        "net_profit_per_500_rounds": (
            float(cumulative_net_bnb) / float(total_rounds) * 500.0 if int(total_rounds) > 0 else 0.0
        ),
        "oracle_profit_bnb": float(total_oracle_bnb),
        "oracle_profit_per_500_rounds": (
            float(total_oracle_bnb) / float(total_rounds) * 500.0 if int(total_rounds) > 0 else 0.0
        ),
        "capture_ratio_vs_oracle": (
            float(cumulative_net_bnb) / float(total_oracle_bnb) if float(total_oracle_bnb) > 0.0 else 0.0
        ),
        "total_selected_bets": int(total_selected_bets),
        "selected_bet_rate": (
            float(total_selected_bets) / float(total_rounds) if int(total_rounds) > 0 else 0.0
        ),
        "max_drawdown_bnb": float(max_drawdown_bnb),
        "starting_bankroll_bnb": float(starting_bankroll_bnb),
        "min_bankroll_bnb": float(min_bankroll_bnb),
        "loss_from_start_to_min_bnb": float(float(starting_bankroll_bnb) - float(min_bankroll_bnb)),
        "num_switches": int(num_switches),
        "picks_by_strategy": {str(name): int(count) for name, count in sorted(picks_by_strategy.items())},
        "pick_reasons": {str(name): int(count) for name, count in sorted(pick_reasons.items())},
    }
    return summary, decision_rows


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.required_streak_len) <= 0:
        raise ValueError("meta_strategy_followthrough_required_streak_len_nonpositive")
    if float(args.min_transition_prob) < 0.0 or float(args.min_transition_prob) > 1.0:
        raise ValueError("meta_strategy_followthrough_min_transition_prob_out_of_range")
    if int(args.min_hold_blocks) <= 0:
        raise ValueError("meta_strategy_followthrough_min_hold_blocks_nonpositive")
    if int(args.min_train_rows) <= 0:
        raise ValueError("meta_strategy_followthrough_min_train_rows_nonpositive")

    all_strategies, key_map = _load_meta(Path(str(args.dataset_meta)))
    baseline_strategy_name = str(args.baseline_strategy_name).strip()
    if baseline_strategy_name not in all_strategies:
        raise ValueError(f"meta_strategy_followthrough_baseline_unknown: {baseline_strategy_name}")

    active_candidates = parse_strategy_prefixes(str(args.active_strategy_names))
    for candidate_name in active_candidates:
        if str(candidate_name) not in all_strategies:
            raise ValueError(f"meta_strategy_followthrough_candidate_unknown: {candidate_name}")
        if str(candidate_name) == str(baseline_strategy_name):
            raise ValueError("meta_strategy_followthrough_baseline_not_allowed_in_candidates")

    rows = _load_rows(Path(str(args.dataset_csv)), key_map)
    summary, decision_rows = _run_probe(
        rows=rows,
        baseline_strategy_name=str(baseline_strategy_name),
        active_candidates=[str(name) for name in active_candidates],
        score_mode=str(args.score_mode),
        required_streak_len=int(args.required_streak_len),
        min_transition_prob=float(args.min_transition_prob),
        min_hold_blocks=int(args.min_hold_blocks),
        min_train_rows=int(args.min_train_rows),
        starting_bankroll_bnb=float(args.starting_bankroll_bnb),
    )

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.name_prefix}_meta_followthrough_probe_summary.json"
    decisions_path = output_dir / f"{args.name_prefix}_meta_followthrough_probe_decisions.csv"

    summary_payload = {
        "probe": {
            "name_prefix": str(args.name_prefix),
            "dataset_csv": str(args.dataset_csv),
            "dataset_meta": str(args.dataset_meta),
            "baseline_strategy_name": str(args.baseline_strategy_name),
            "active_strategy_names": [str(name) for name in active_candidates],
            "score_mode": str(args.score_mode),
            "required_streak_len": int(args.required_streak_len),
            "min_transition_prob": float(args.min_transition_prob),
            "min_hold_blocks": int(args.min_hold_blocks),
            "min_train_rows": int(args.min_train_rows),
            "starting_bankroll_bnb": float(args.starting_bankroll_bnb),
        },
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    if bool(args.write_decisions):
        with decisions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0].keys()))
            writer.writeheader()
            writer.writerows(decision_rows)

    print(f"SUMMARY={summary_path}")
    if bool(args.write_decisions):
        print(f"DECISIONS={decisions_path}")
    print(f"NET={summary['net_profit_bnb']}")
    print(f"NET_PER_500={summary['net_profit_per_500_rounds']}")
    print(f"SELECTED_BET_RATE={summary['selected_bet_rate']}")


if __name__ == "__main__":
    main()
