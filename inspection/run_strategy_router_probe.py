"""Run router policy probes against a prebuilt strategy-router dataset.

This probe is inspection-only. It does not execute production strategy logic.
It consumes `*_router_dataset.csv` outputs and evaluates routing modes in a
strict walk-forward fashion.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspection.strategy_router_common import parse_strategy_prefixes

_ROUTER_MODES = ("cash_only", "oracle_cash", "expected_net_max", "online_cellmean")


@dataclass(frozen=True, slots=True)
class RoundCandidate:
    """Per-round data for one strategy from the router dataset."""

    bet_available: bool
    direction_idx: int
    expected_net_selected_bnb: float | None
    abs_dislocation_bull: float | None
    profit_bnb: float


@dataclass(frozen=True, slots=True)
class RouterRound:
    """One aligned round with all candidate strategy rows."""

    block_index: int
    sim_offset_rounds: int
    epoch: int
    candidates: dict[str, RoundCandidate]
    oracle_profit_bnb: float


def _safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        value = float(text)
    except Exception:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return float(value)


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _quantile_edges(values: list[float], n_bins: int) -> list[float]:
    if int(n_bins) <= 1:
        raise ValueError("router_num_quantile_bins_invalid")
    if not values:
        return [0.0 for _ in range(int(n_bins) + 1)]
    ordered = sorted(float(v) for v in values)
    edges: list[float] = []
    for i in range(int(n_bins) + 1):
        q = float(i) / float(n_bins)
        idx = int(round((len(ordered) - 1) * q))
        idx = max(0, min(len(ordered) - 1, idx))
        edges.append(float(ordered[idx]))
    return edges


def _bin_index(value: float, edges: list[float]) -> int:
    bins = int(len(edges) - 1)
    if int(bins) <= 1:
        return 0
    for i in range(int(bins)):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if float(value) >= float(lo) and (float(value) < float(hi) or int(i) == int(bins - 1)):
            return int(i)
    return int(bins - 1)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for router probes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--dataset-csv", type=str, required=True)
    parser.add_argument("--dataset-meta", type=str, required=True)
    parser.add_argument("--router-mode", type=str, choices=_ROUTER_MODES, required=True)
    parser.add_argument("--expected-net-threshold-bnb", type=float, default=0.0)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--num-quantile-bins", type=int, default=12)
    parser.add_argument("--min-cell-obs", type=int, default=5)
    parser.add_argument("--score-threshold-bnb", type=float, default=0.0)
    parser.add_argument("--use-direction-split", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="var/exp")
    parser.add_argument("--write-trades", action="store_true", default=False)
    return parser


def _load_meta(meta_path: Path) -> tuple[list[str], dict[str, str]]:
    if not meta_path.exists():
        raise FileNotFoundError(f"router_meta_missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    strategies = [str(x) for x in meta.get("strategy_prefixes", [])]
    key_map = {str(k): str(v) for k, v in meta.get("strategy_column_keys", {}).items()}
    if not strategies:
        raise ValueError("router_meta_strategy_prefixes_missing")
    if not key_map:
        raise ValueError("router_meta_strategy_key_map_missing")
    return strategies, key_map


def _load_rounds(dataset_csv: Path, strategies: list[str], key_map: dict[str, str]) -> list[RouterRound]:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"router_dataset_missing: {dataset_csv}")

    rounds: list[RouterRound] = []
    with dataset_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            candidates: dict[str, RoundCandidate] = {}
            for strategy in strategies:
                key = key_map[str(strategy)]
                candidates[str(strategy)] = RoundCandidate(
                    bet_available=bool(int(raw.get(f"feat_{key}_bet_available", "0"))),
                    direction_idx=int(raw.get(f"feat_{key}_direction_idx", "-1")),
                    expected_net_selected_bnb=_safe_float(raw.get(f"feat_{key}_expected_net_selected_bnb")),
                    abs_dislocation_bull=_safe_float(raw.get(f"feat_{key}_abs_dislocation_bull")),
                    profit_bnb=float(raw.get(f"label_{key}_profit_bnb", "0.0")),
                )
            rounds.append(
                RouterRound(
                    block_index=int(raw["block_index"]),
                    sim_offset_rounds=int(raw["sim_offset_rounds"]),
                    epoch=int(raw["epoch"]),
                    candidates=candidates,
                    oracle_profit_bnb=float(raw.get("label_oracle_profit_bnb", "0.0")),
                )
            )

    if not rounds:
        raise ValueError("router_dataset_empty")
    return rounds


def _pick_expected_net_max(
    *,
    round_row: RouterRound,
    strategies: list[str],
    expected_net_threshold_bnb: float,
) -> str:
    best_strategy = "CASH"
    best_expected = float("-inf")
    for strategy in strategies:
        candidate = round_row.candidates[str(strategy)]
        if not bool(candidate.bet_available):
            continue
        if candidate.expected_net_selected_bnb is None:
            continue
        expected = float(candidate.expected_net_selected_bnb)
        if float(expected) < float(expected_net_threshold_bnb):
            continue
        if float(expected) > float(best_expected):
            best_expected = float(expected)
            best_strategy = str(strategy)
    return str(best_strategy)


def _fit_cell_edges(
    *,
    rounds: list[RouterRound],
    strategies: list[str],
    warmup_rounds: int,
    num_quantile_bins: int,
) -> dict[str, tuple[list[float], list[float]]]:
    edges_by_strategy: dict[str, tuple[list[float], list[float]]] = {}
    warmup_rows = rounds[: int(warmup_rounds)]
    for strategy in strategies:
        expected_values: list[float] = []
        dislocation_values: list[float] = []
        for round_row in warmup_rows:
            candidate = round_row.candidates[str(strategy)]
            if not bool(candidate.bet_available):
                continue
            if candidate.expected_net_selected_bnb is None:
                continue
            if candidate.abs_dislocation_bull is None:
                continue
            expected_values.append(float(candidate.expected_net_selected_bnb))
            dislocation_values.append(float(candidate.abs_dislocation_bull))
        edges_by_strategy[str(strategy)] = (
            _quantile_edges(expected_values, int(num_quantile_bins)),
            _quantile_edges(dislocation_values, int(num_quantile_bins)),
        )
    return edges_by_strategy


def _cell_key(
    *,
    candidate: RoundCandidate,
    expected_edges: list[float],
    dislocation_edges: list[float],
    use_direction_split: bool,
) -> tuple[int, int, int]:
    if candidate.expected_net_selected_bnb is None:
        raise ValueError("router_cell_key_expected_missing")
    if candidate.abs_dislocation_bull is None:
        raise ValueError("router_cell_key_dislocation_missing")
    expected_bin = _bin_index(float(candidate.expected_net_selected_bnb), expected_edges)
    dislocation_bin = _bin_index(float(candidate.abs_dislocation_bull), dislocation_edges)
    side_bin = int(candidate.direction_idx) if bool(use_direction_split) else 0
    return int(expected_bin), int(dislocation_bin), int(side_bin)


def _run_probe(
    *,
    rounds: list[RouterRound],
    strategies: list[str],
    router_mode: str,
    expected_net_threshold_bnb: float,
    warmup_rounds: int,
    num_quantile_bins: int,
    min_cell_obs: int,
    score_threshold_bnb: float,
    use_direction_split: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if str(router_mode) not in _ROUTER_MODES:
        raise ValueError("router_mode_invalid")

    if int(warmup_rounds) < 0:
        raise ValueError("router_warmup_rounds_negative")
    if str(router_mode) == "online_cellmean":
        if int(warmup_rounds) <= 0:
            raise ValueError("router_online_cellmean_warmup_rounds_must_be_positive")
        if int(warmup_rounds) >= int(len(rounds)):
            raise ValueError("router_online_cellmean_warmup_rounds_too_large")
        if int(min_cell_obs) <= 0:
            raise ValueError("router_online_cellmean_min_cell_obs_must_be_positive")

    cumulative_net_bnb = 0.0
    peak_net_bnb = 0.0
    max_drawdown_bnb = 0.0
    num_bets = 0
    num_wins = 0
    num_cash = 0
    oracle_total_bnb = 0.0
    picks_by_strategy: dict[str, int] = {str(strategy): 0 for strategy in strategies}
    picks_by_strategy["CASH"] = 0
    trade_rows: list[dict[str, Any]] = []

    sum_profit_by_cell: dict[str, dict[tuple[int, int, int], float]] = {
        str(strategy): {} for strategy in strategies
    }
    count_by_cell: dict[str, dict[tuple[int, int, int], int]] = {
        str(strategy): {} for strategy in strategies
    }
    edges_by_strategy: dict[str, tuple[list[float], list[float]]] = {}

    if str(router_mode) == "online_cellmean":
        edges_by_strategy = _fit_cell_edges(
            rounds=rounds,
            strategies=strategies,
            warmup_rounds=int(warmup_rounds),
            num_quantile_bins=int(num_quantile_bins),
        )

        # Warmup rows populate cell statistics but do not place live bets.
        for round_row in rounds[: int(warmup_rounds)]:
            for strategy in strategies:
                candidate = round_row.candidates[str(strategy)]
                if not bool(candidate.bet_available):
                    continue
                if candidate.expected_net_selected_bnb is None or candidate.abs_dislocation_bull is None:
                    continue
                expected_edges, dislocation_edges = edges_by_strategy[str(strategy)]
                key = _cell_key(
                    candidate=candidate,
                    expected_edges=expected_edges,
                    dislocation_edges=dislocation_edges,
                    use_direction_split=bool(use_direction_split),
                )
                sum_profit_by_cell[str(strategy)][key] = float(
                    sum_profit_by_cell[str(strategy)].get(key, 0.0) + float(candidate.profit_bnb)
                )
                count_by_cell[str(strategy)][key] = int(count_by_cell[str(strategy)].get(key, 0) + 1)

    for idx, round_row in enumerate(rounds):
        oracle_total_bnb += float(round_row.oracle_profit_bnb)
        chosen = "CASH"
        chosen_profit_bnb = 0.0
        chosen_score = ""

        is_warmup = str(router_mode) == "online_cellmean" and int(idx) < int(warmup_rounds)

        if str(router_mode) == "cash_only":
            chosen = "CASH"
        elif str(router_mode) == "oracle_cash":
            best_profit = 0.0
            best_strategy = "CASH"
            for strategy in strategies:
                candidate = round_row.candidates[str(strategy)]
                if not bool(candidate.bet_available):
                    continue
                if float(candidate.profit_bnb) > float(best_profit):
                    best_profit = float(candidate.profit_bnb)
                    best_strategy = str(strategy)
            chosen = str(best_strategy)
            chosen_profit_bnb = float(best_profit)
        elif str(router_mode) == "expected_net_max":
            chosen = _pick_expected_net_max(
                round_row=round_row,
                strategies=strategies,
                expected_net_threshold_bnb=float(expected_net_threshold_bnb),
            )
            if str(chosen) != "CASH":
                chosen_profit_bnb = float(round_row.candidates[str(chosen)].profit_bnb)
                chosen_score = round_row.candidates[str(chosen)].expected_net_selected_bnb
        elif str(router_mode) == "online_cellmean":
            if not bool(is_warmup):
                best_estimated = float("-inf")
                best_strategy = "CASH"
                for strategy in strategies:
                    candidate = round_row.candidates[str(strategy)]
                    if not bool(candidate.bet_available):
                        continue
                    if candidate.expected_net_selected_bnb is None or candidate.abs_dislocation_bull is None:
                        continue
                    expected_edges, dislocation_edges = edges_by_strategy[str(strategy)]
                    key = _cell_key(
                        candidate=candidate,
                        expected_edges=expected_edges,
                        dislocation_edges=dislocation_edges,
                        use_direction_split=bool(use_direction_split),
                    )
                    count = int(count_by_cell[str(strategy)].get(key, 0))
                    if int(count) < int(min_cell_obs):
                        continue
                    estimated = float(sum_profit_by_cell[str(strategy)][key]) / float(count)
                    if float(estimated) < float(score_threshold_bnb):
                        continue
                    if float(estimated) > float(best_estimated):
                        best_estimated = float(estimated)
                        best_strategy = str(strategy)
                chosen = str(best_strategy)
                if str(chosen) != "CASH":
                    chosen_profit_bnb = float(round_row.candidates[str(chosen)].profit_bnb)
                    chosen_score = float(best_estimated)
        else:
            raise ValueError("router_mode_unreachable")

        # Online shadow update after each round closes.
        # Warmup rows are already seeded before this loop, so skip re-adding.
        if str(router_mode) == "online_cellmean" and not bool(is_warmup):
            for strategy in strategies:
                candidate = round_row.candidates[str(strategy)]
                if not bool(candidate.bet_available):
                    continue
                if candidate.expected_net_selected_bnb is None or candidate.abs_dislocation_bull is None:
                    continue
                expected_edges, dislocation_edges = edges_by_strategy[str(strategy)]
                key = _cell_key(
                    candidate=candidate,
                    expected_edges=expected_edges,
                    dislocation_edges=dislocation_edges,
                    use_direction_split=bool(use_direction_split),
                )
                sum_profit_by_cell[str(strategy)][key] = float(
                    sum_profit_by_cell[str(strategy)].get(key, 0.0) + float(candidate.profit_bnb)
                )
                count_by_cell[str(strategy)][key] = int(count_by_cell[str(strategy)].get(key, 0) + 1)

        if str(chosen) == "CASH":
            num_cash += 1
            picks_by_strategy["CASH"] += 1
            chosen_profit_bnb = 0.0
        else:
            num_bets += 1
            picks_by_strategy[str(chosen)] += 1
            if float(chosen_profit_bnb) > 0.0:
                num_wins += 1

        cumulative_net_bnb += float(chosen_profit_bnb)
        if float(cumulative_net_bnb) > float(peak_net_bnb):
            peak_net_bnb = float(cumulative_net_bnb)
        drawdown = float(peak_net_bnb) - float(cumulative_net_bnb)
        if float(drawdown) > float(max_drawdown_bnb):
            max_drawdown_bnb = float(drawdown)

        trade_rows.append(
            {
                "block_index": int(round_row.block_index),
                "sim_offset_rounds": int(round_row.sim_offset_rounds),
                "epoch": int(round_row.epoch),
                "pick": str(chosen),
                "pick_score": chosen_score,
                "profit_bnb": float(chosen_profit_bnb),
                "oracle_profit_bnb": float(round_row.oracle_profit_bnb),
                "cum_net_bnb": float(cumulative_net_bnb),
                "regret_to_oracle_bnb": float(round_row.oracle_profit_bnb - chosen_profit_bnb),
            }
        )

    rounds_total = int(len(rounds))
    summary = {
        "num_rounds": int(rounds_total),
        "num_bets": int(num_bets),
        "num_cash": int(num_cash),
        "num_wins": int(num_wins),
        "bet_rate": float(_safe_rate(num_bets, rounds_total)),
        "win_rate_on_bets": float(_safe_rate(num_wins, num_bets)),
        "net_profit_bnb": float(cumulative_net_bnb),
        "net_profit_per_500_rounds": float(cumulative_net_bnb / rounds_total * 500.0),
        "oracle_profit_bnb": float(oracle_total_bnb),
        "oracle_profit_per_500_rounds": float(oracle_total_bnb / rounds_total * 500.0),
        "capture_ratio_vs_oracle": float(cumulative_net_bnb / oracle_total_bnb)
        if float(oracle_total_bnb) > 0.0
        else 0.0,
        "max_drawdown_bnb": float(max_drawdown_bnb),
        "picks_by_strategy": {str(k): int(v) for k, v in picks_by_strategy.items()},
    }
    return summary, trade_rows


def main() -> None:
    """Run one router probe and write summary artifacts."""

    args = _build_parser().parse_args()
    strategies, key_map = _load_meta(Path(str(args.dataset_meta)))

    # Validate strategy-prefix parser behavior consistently with dataset builder.
    parse_strategy_prefixes(",".join(strategies))

    rounds = _load_rounds(
        dataset_csv=Path(str(args.dataset_csv)),
        strategies=strategies,
        key_map=key_map,
    )
    summary, trades = _run_probe(
        rounds=rounds,
        strategies=strategies,
        router_mode=str(args.router_mode),
        expected_net_threshold_bnb=float(args.expected_net_threshold_bnb),
        warmup_rounds=int(args.warmup_rounds),
        num_quantile_bins=int(args.num_quantile_bins),
        min_cell_obs=int(args.min_cell_obs),
        score_threshold_bnb=float(args.score_threshold_bnb),
        use_direction_split=bool(args.use_direction_split),
    )

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.name_prefix}_router_probe_summary.json"
    summary_payload = {
        "probe": {
            "name_prefix": str(args.name_prefix),
            "dataset_csv": str(args.dataset_csv),
            "dataset_meta": str(args.dataset_meta),
            "router_mode": str(args.router_mode),
            "expected_net_threshold_bnb": float(args.expected_net_threshold_bnb),
            "warmup_rounds": int(args.warmup_rounds),
            "num_quantile_bins": int(args.num_quantile_bins),
            "min_cell_obs": int(args.min_cell_obs),
            "score_threshold_bnb": float(args.score_threshold_bnb),
            "use_direction_split": bool(args.use_direction_split),
        },
        "strategies": [str(x) for x in strategies],
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    trades_path = output_dir / f"{args.name_prefix}_router_probe_trades.csv"
    if bool(args.write_trades):
        with trades_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "block_index",
                    "sim_offset_rounds",
                    "epoch",
                    "pick",
                    "pick_score",
                    "profit_bnb",
                    "oracle_profit_bnb",
                    "cum_net_bnb",
                    "regret_to_oracle_bnb",
                ],
            )
            writer.writeheader()
            writer.writerows(trades)

    print(f"SUMMARY={summary_path}")
    if bool(args.write_trades):
        print(f"TRADES={trades_path}")
    print(f"MODE={args.router_mode}")
    print(f"NET={summary['net_profit_bnb']}")
    print(f"NET_PER_500={summary['net_profit_per_500_rounds']}")
    print(f"CAPTURE={summary['capture_ratio_vs_oracle']}")


if __name__ == "__main__":
    main()
