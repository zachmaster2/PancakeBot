"""Generate exact opposite-side candidate block artifacts from existing trades.

This inspection tool reuses historical block artifacts and flips every BET
direction while preserving the original bet timing and stake. Settlement is
recomputed exactly from closed-round data, so the derived candidate is an
honest "same rounds, opposite side" series.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from inspection.strategy_router_common import (
    StrategyTradeRow,
    build_block_offsets,
    build_scenario_name,
    load_strategy_trade_rows,
    parse_strategy_prefixes,
)
from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.runtime.contract_constants_cache import load_contract_constants
from pancakebot.runtime.settlement import settle_bet_against_closed_round


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-strategy-prefixes", type=str, required=True)
    parser.add_argument("--name-suffix", type=str, default="opp")
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--num-blocks", type=int, default=80)
    parser.add_argument("--skip-most-recent-blocks", type=int, default=0)
    parser.add_argument("--base-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--output-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--trades-filename", type=str, default="dislocation_trades.csv")
    parser.add_argument("--closed-rounds-path", type=str, default="var/closed_rounds.jsonl")
    return parser


def _flip_direction(direction: str) -> str:
    direct = str(direction).strip().upper()
    if direct == "BULL":
        return "BEAR"
    if direct == "BEAR":
        return "BULL"
    raise InvariantError("opposite_candidate_direction_invalid")


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _write_trade_rows(path: Path, rows: list[StrategyTradeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "action",
                "direction",
                "expected_net_selected",
                "dislocation_bull",
                "p_nowcast_bull",
                "p_market_bull",
                "bet_size_bnb",
                "profit_bnb",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    int(row.epoch),
                    str(row.action),
                    str(row.direction),
                    "" if row.expected_net_selected_bnb is None else float(row.expected_net_selected_bnb),
                    "" if row.dislocation_bull is None else float(row.dislocation_bull),
                    "" if row.p_nowcast_bull is None else float(row.p_nowcast_bull),
                    "" if row.p_market_bull is None else float(row.p_market_bull),
                    "" if row.bet_size_bnb is None else float(row.bet_size_bnb),
                    float(row.profit_bnb),
                ]
            )


def main() -> None:
    args = _build_parser().parse_args()
    source_prefixes = parse_strategy_prefixes(str(args.source_strategy_prefixes))
    suffix = str(args.name_suffix).strip()
    if suffix == "":
        raise ValueError("opposite_candidate_name_suffix_empty")
    if int(args.block_size) <= 0 or int(args.num_blocks) <= 0:
        raise ValueError("opposite_candidate_block_shape_invalid")
    if int(args.skip_most_recent_blocks) < 0:
        raise ValueError("opposite_candidate_skip_negative")

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(str(args.base_dir))

    constants = load_contract_constants()
    rounds_by_epoch = {
        int(round_t.epoch): round_t
        for round_t in ClosedRoundsStore(str(args.closed_rounds_path)).iter_closed_rounds()
    }
    if not rounds_by_epoch:
        raise ValueError("opposite_candidate_closed_rounds_empty")

    offsets = build_block_offsets(
        block_size=int(args.block_size),
        num_blocks=int(args.num_blocks),
        skip_most_recent_blocks=int(args.skip_most_recent_blocks),
    )

    for source_prefix in source_prefixes:
        derived_prefix = f"{str(source_prefix)}_{suffix}"
        block_rows: list[dict[str, Any]] = []
        block_nets: list[float] = []
        bets_total = 0
        wins_total = 0

        for block_index, sim_offset_rounds in enumerate(offsets, start=1):
            source_scenario = build_scenario_name(
                strategy_prefix=str(source_prefix),
                block_index=int(block_index),
                num_blocks=int(args.num_blocks),
                sim_offset_rounds=int(sim_offset_rounds),
            )
            derived_scenario = build_scenario_name(
                strategy_prefix=str(derived_prefix),
                block_index=int(block_index),
                num_blocks=int(args.num_blocks),
                sim_offset_rounds=int(sim_offset_rounds),
            )
            source_trades_path = base_dir / source_scenario / str(args.trades_filename)
            source_rows = load_strategy_trade_rows(source_trades_path)
            derived_rows: list[StrategyTradeRow] = []

            block_net = 0.0
            num_bets = 0
            num_wins = 0
            for epoch in sorted(source_rows):
                row = source_rows[int(epoch)]
                if str(row.action) == "SKIP":
                    derived = StrategyTradeRow(
                        epoch=int(row.epoch),
                        action="SKIP",
                        direction="",
                        expected_net_selected_bnb=None,
                        dislocation_bull=row.dislocation_bull,
                        p_nowcast_bull=row.p_nowcast_bull,
                        p_market_bull=row.p_market_bull,
                        bet_size_bnb=0.0,
                        profit_bnb=0.0,
                    )
                else:
                    if row.bet_size_bnb is None or float(row.bet_size_bnb) <= 0.0:
                        raise InvariantError("opposite_candidate_bet_size_missing")
                    round_closed = rounds_by_epoch.get(int(row.epoch))
                    if round_closed is None:
                        raise InvariantError(f"opposite_candidate_round_missing: {int(row.epoch)}")
                    flipped_direction = _flip_direction(str(row.direction))
                    settle = settle_bet_against_closed_round(
                        bet_bnb=float(row.bet_size_bnb),
                        bet_side=str(flipped_direction),
                        round_closed=round_closed,
                        treasury_fee_fraction=float(constants.treasury_fee_fraction),
                    )
                    profit_bnb = float(settle.credit_bnb) - float(row.bet_size_bnb) - float(GAS_COST_BET_BNB)
                    derived = StrategyTradeRow(
                        epoch=int(row.epoch),
                        action="BET",
                        direction=str(flipped_direction),
                        expected_net_selected_bnb=None,
                        dislocation_bull=row.dislocation_bull,
                        p_nowcast_bull=row.p_nowcast_bull,
                        p_market_bull=row.p_market_bull,
                        bet_size_bnb=float(row.bet_size_bnb),
                        profit_bnb=float(profit_bnb),
                    )
                    num_bets += 1
                    if str(settle.outcome) == "win":
                        num_wins += 1
                block_net += float(derived.profit_bnb)
                derived_rows.append(derived)

            scenario_dir = output_dir / derived_scenario
            _write_trade_rows(scenario_dir / str(args.trades_filename), derived_rows)
            summary = {
                "scenario": {
                    "name": str(derived_scenario),
                    "source_strategy_prefix": str(source_prefix),
                    "derived_strategy_prefix": str(derived_prefix),
                    "block_index": int(block_index),
                    "sim_offset_rounds": int(sim_offset_rounds),
                    "block_size": int(len(derived_rows)),
                    "strategy_family": "opposite_side_clone",
                },
                "net_profit_bnb": float(block_net),
                "num_rounds": int(len(derived_rows)),
                "num_bets": int(num_bets),
                "num_wins": int(num_wins),
                "bet_rate": float(_safe_rate(num_bets, len(derived_rows))),
                "win_rate": float(_safe_rate(num_wins, num_bets)),
            }
            (scenario_dir / "dislocation_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            block_rows.append(
                {
                    "scenario": str(derived_scenario),
                    "block_index": int(block_index),
                    "sim_offset_rounds": int(sim_offset_rounds),
                    "net": float(block_net),
                    "bets": int(num_bets),
                    "wins": int(num_wins),
                    "bet_rate": float(_safe_rate(num_bets, len(derived_rows))),
                    "win_rate": float(_safe_rate(num_wins, num_bets)),
                }
            )
            block_nets.append(float(block_net))
            bets_total += int(num_bets)
            wins_total += int(num_wins)

        agg = {
            "blocks": int(len(block_rows)),
            "net_total": float(sum(block_nets)),
            "net_mean": float(sum(block_nets) / len(block_nets)),
            "net_median": float(statistics.median(block_nets)),
            "net_worst": float(min(block_nets)),
            "net_best": float(max(block_nets)),
            "positive_blocks": int(sum(1 for x in block_nets if float(x) > 0.0)),
            "positive_block_frac": float(sum(1 for x in block_nets if float(x) > 0.0) / len(block_nets)),
            "net_per_500": float(sum(block_nets) / float(int(args.block_size) * int(args.num_blocks)) * 500.0),
            "bets_total": int(bets_total),
            "win_rate_weighted": float(_safe_rate(wins_total, bets_total)),
        }
        agg_path = output_dir / f"{derived_prefix}_aggregate.json"
        agg_path.write_text(
            json.dumps(
                {
                    "name_prefix": str(derived_prefix),
                    "source_strategy_prefix": str(source_prefix),
                    "rows": block_rows,
                    "aggregate": agg,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"AGG={agg_path}")
        print(f"NET_PER_500={agg['net_per_500']}")


if __name__ == "__main__":
    main()
