"""Generate canonical block artifacts for one custom dislocation candidate.

This inspection tool replays the production dislocation signal path for a
single candidate over aligned recent-history blocks and writes
`dislocation_trades.csv` + `dislocation_summary.json` per block so the
existing meta-strategy and router tooling can ingest the candidate directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from inspection.backtest_harness_common import (
    _apply_backtest_gas_profile,
    _resolve_gas_price_wei_override,
    build_runtime_cfg,
    load_cfg,
    resolve_exp_root,
)
from inspection.run_alta_single_idea import _candidate_pool, _parse_set_overrides
from inspection.strategy_router_common import build_block_offsets, build_scenario_name
from pancakebot.backtest.runner import _all_klines_from_store, _build_strategy_pipeline, _tail_rounds
from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.runtime.settlement import settle_bet_against_closed_round


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--candidate-name", type=str, required=True)
    parser.add_argument("--candidate-source", choices=("active", "all_config"), default="all_config")
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--num-blocks", type=int, default=80)
    parser.add_argument("--skip-most-recent-blocks", type=int, default=0)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--gas-price-wei-override", type=int, default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--no-resume", action="store_true", default=False)
    return parser


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _block_slice(*, rounds: list[Any], block_size: int, sim_offset_rounds: int) -> tuple[int, int]:
    end = int(len(rounds)) - int(sim_offset_rounds)
    start = int(end) - int(block_size)
    if int(start) < 0 or int(end) > int(len(rounds)):
        raise InvariantError("dislocation_candidate_blocks_block_slice_invalid")
    return int(start), int(end)


def _trade_rows_path(*, scenario_dir: Path) -> Path:
    return Path(scenario_dir) / "dislocation_trades.csv"


def _summary_path(*, scenario_dir: Path) -> Path:
    return Path(scenario_dir) / "dislocation_summary.json"


def _existing_block_row(*, scenario_dir: Path) -> dict[str, Any] | None:
    summary_path = _summary_path(scenario_dir=scenario_dir)
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(summary, dict):
        return None
    scenario = summary.get("scenario", {})
    if not isinstance(scenario, dict):
        scenario = {}
    return {
        "scenario": str(scenario.get("name", scenario_dir.name)),
        "block_index": int(scenario.get("block_index", 0)),
        "sim_offset_rounds": int(scenario.get("sim_offset_rounds", 0)),
        "epoch_first": int(summary.get("epoch_first", 0)),
        "epoch_last": int(summary.get("epoch_last", 0)),
        "net": float(summary.get("net_profit_bnb", 0.0)),
        "bets": int(summary.get("num_bets", 0)),
        "wins": int(summary.get("num_wins", 0)),
        "bet_rate": float(summary.get("bet_rate", 0.0)),
        "win_rate": float(summary.get("win_rate", 0.0)),
    }


def _write_trades(*, path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.block_size) <= 0:
        raise InvariantError("dislocation_candidate_blocks_block_size_nonpositive")
    if int(args.num_blocks) <= 0:
        raise InvariantError("dislocation_candidate_blocks_num_blocks_nonpositive")
    if int(args.skip_most_recent_blocks) < 0:
        raise InvariantError("dislocation_candidate_blocks_skip_most_recent_blocks_negative")
    if float(args.initial_bankroll_bnb) <= 0.0:
        raise InvariantError("dislocation_candidate_blocks_initial_bankroll_nonpositive")

    cfg = load_cfg(config_path=str(args.config))
    candidate_pool = _candidate_pool(
        cfg=cfg,
        config_path=str(args.config),
        candidate_source=str(args.candidate_source),
    )
    candidate_map = {str(candidate.name): candidate for candidate in candidate_pool}
    candidate_name = str(args.candidate_name)
    if str(candidate_name) not in candidate_map:
        raise InvariantError(f"dislocation_candidate_blocks_candidate_missing: {candidate_name}")

    candidate_overrides = _parse_set_overrides(list(args.set))
    tuned_candidate = replace(candidate_map[str(candidate_name)], **candidate_overrides)
    strategy_cfg = replace(
        cfg.strategy,
        dislocation=replace(cfg.strategy.dislocation, candidates=(tuned_candidate,)),
        ml_candidate=replace(cfg.strategy.ml_candidate, enabled=False),
    )

    resolved_gas_override = _resolve_gas_price_wei_override(
        gas_price_wei_override=args.gas_price_wei_override
    )
    gas_profile = _apply_backtest_gas_profile(gas_price_wei_override=resolved_gas_override)
    runtime_cfg = build_runtime_cfg(
        cfg=cfg,
        strategy_cfg=strategy_cfg,
        gas_price_wei_override=resolved_gas_override,
    )

    warmup_rounds = int(cfg.strategy.dislocation.selector.warmup_rounds)
    tail_n = int(warmup_rounds) + int(args.block_size) * (
        int(args.num_blocks) + int(args.skip_most_recent_blocks)
    )
    closed_rounds = _tail_rounds(runtime_cfg.round_store, n=int(tail_n))
    if len(closed_rounds) != int(tail_n):
        raise InvariantError("dislocation_candidate_blocks_tail_rounds_len_mismatch")
    all_klines = _all_klines_from_store(runtime_cfg.klines_store)

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = build_block_offsets(
        block_size=int(args.block_size),
        num_blocks=int(args.num_blocks),
        skip_most_recent_blocks=int(args.skip_most_recent_blocks),
    )

    block_rows: list[dict[str, Any]] = []
    block_nets: list[float] = []
    bets_total = 0
    wins_total = 0
    resume = not bool(args.no_resume)

    if not bool(args.aggregate_only):
        for block_idx, offset in enumerate(offsets, start=1):
            scenario_name = build_scenario_name(
                strategy_prefix=str(args.name_prefix),
                block_index=int(block_idx),
                num_blocks=int(args.num_blocks),
                sim_offset_rounds=int(offset),
            )
            scenario_dir = output_dir / str(scenario_name)
            existing_row = _existing_block_row(scenario_dir=scenario_dir) if bool(resume) else None
            if existing_row is not None:
                if int(args.progress_every) > 0 and (
                    int(block_idx) == 1
                    or int(block_idx) == int(args.num_blocks)
                    or int(block_idx) % int(args.progress_every) == 0
                ):
                    print(
                        "BLOCK_SKIP "
                        + f"block={int(block_idx)}/{int(args.num_blocks)} offset={int(offset)} "
                        + "reason=resume"
                    )
                continue

            start, end = _block_slice(
                rounds=closed_rounds,
                block_size=int(args.block_size),
                sim_offset_rounds=int(offset),
            )
            warmup_start = int(start) - int(warmup_rounds)
            if int(warmup_start) < 0:
                raise InvariantError("dislocation_candidate_blocks_warmup_slice_invalid")
            warmup_slice = closed_rounds[int(warmup_start): int(start)]
            block_rounds = closed_rounds[int(start): int(end)]
            if len(warmup_slice) != int(warmup_rounds):
                raise InvariantError("dislocation_candidate_blocks_warmup_len_mismatch")
            if len(block_rounds) != int(args.block_size):
                raise InvariantError("dislocation_candidate_blocks_block_len_mismatch")

            pipeline = _build_strategy_pipeline(runtime_cfg=runtime_cfg, all_klines=list(all_klines))
            pipeline.bootstrap_from_closed_rounds(rounds=list(warmup_slice))

            bankroll_bnb = float(args.initial_bankroll_bnb)
            block_net = 0.0
            num_bets = 0
            num_wins = 0
            skip_counts: dict[str, int] = {}
            trade_rows: list[list[Any]] = [
                [
                    "epoch",
                    "action",
                    "skip_reason",
                    "direction",
                    "p_nowcast_bull",
                    "p_market_bull",
                    "dislocation_bull",
                    "expected_net_bull",
                    "expected_net_bear",
                    "expected_net_selected",
                    "pool_total_bnb_cutoff",
                    "bet_size_bnb",
                    "profit_bnb",
                    "bankroll_bnb",
                ]
            ]

            for round_t in block_rounds:
                candidate_signals = pipeline.candidate_signals_for_open_round(round_t=round_t)
                signal = candidate_signals.get(str(tuned_candidate.name))
                if signal is None:
                    raise InvariantError("dislocation_candidate_blocks_signal_missing")

                direction = ""
                profit_bnb = 0.0
                if str(signal.action) == "BET" and float(signal.bet_size_bnb) > 0.0:
                    bet_side = str(signal.bet_side or "")
                    if bet_side not in ("Bull", "Bear"):
                        raise InvariantError("dislocation_candidate_blocks_bet_side_invalid")
                    settle = settle_bet_against_closed_round(
                        bet_bnb=float(signal.bet_size_bnb),
                        bet_side=str(bet_side),
                        round_closed=round_t,
                        treasury_fee_fraction=float(runtime_cfg.treasury_fee_fraction),
                    )
                    profit_bnb = float(settle.credit_bnb) - float(signal.bet_size_bnb) - float(GAS_COST_BET_BNB)
                    bankroll_bnb += float(profit_bnb)
                    block_net += float(profit_bnb)
                    num_bets += 1
                    if str(settle.outcome) == "win":
                        num_wins += 1
                    direction = "BULL" if str(bet_side) == "Bull" else "BEAR"
                else:
                    reason = str(signal.skip_reason or "unknown_skip_reason")
                    skip_counts[reason] = int(skip_counts.get(reason, 0) + 1)

                trade_rows.append(
                    [
                        int(round_t.epoch),
                        str(signal.action),
                        str(signal.skip_reason or ""),
                        str(direction),
                        "" if signal.p_bull is None else float(signal.p_bull),
                        "",
                        "" if signal.dislocation_bull is None else float(signal.dislocation_bull),
                        "",
                        "",
                        "" if signal.expected_profit_bnb is None else float(signal.expected_profit_bnb),
                        "",
                        float(signal.bet_size_bnb),
                        float(profit_bnb),
                        float(bankroll_bnb),
                    ]
                )
                pipeline.settle_closed_rounds(rounds=[round_t])

            _write_trades(path=_trade_rows_path(scenario_dir=scenario_dir), rows=trade_rows)
            summary = {
                "scenario": {
                    "name": str(scenario_name),
                    "block_index": int(block_idx),
                    "sim_offset_rounds": int(offset),
                    "block_size": int(args.block_size),
                    "strategy_family": "dislocation_single_candidate",
                },
                "config": {
                    "candidate": asdict(tuned_candidate),
                    "gas_profile": dict(gas_profile),
                },
                "initial_bankroll_bnb": float(args.initial_bankroll_bnb),
                "final_bankroll_bnb": float(bankroll_bnb),
                "net_profit_bnb": float(block_net),
                "num_rounds": int(args.block_size),
                "num_bets": int(num_bets),
                "num_wins": int(num_wins),
                "bet_rate": float(_safe_rate(num_bets, int(args.block_size))),
                "win_rate": float(_safe_rate(num_wins, num_bets)),
                "num_skips_by_reason": {str(k): int(v) for k, v in sorted(skip_counts.items())},
                "epoch_first": int(block_rounds[0].epoch),
                "epoch_last": int(block_rounds[-1].epoch),
            }
            _summary_path(scenario_dir=scenario_dir).write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if int(args.progress_every) > 0 and (
                int(block_idx) == 1
                or int(block_idx) == int(args.num_blocks)
                or int(block_idx) % int(args.progress_every) == 0
            ):
                print(
                    "BLOCK_DONE "
                    + f"block={int(block_idx)}/{int(args.num_blocks)} offset={int(offset)} "
                    + f"net={float(block_net):.6f} bets={int(num_bets)} "
                    + f"win={float(_safe_rate(num_wins, num_bets)):.4f}"
                )

    for block_idx, offset in enumerate(offsets, start=1):
        scenario_name = build_scenario_name(
            strategy_prefix=str(args.name_prefix),
            block_index=int(block_idx),
            num_blocks=int(args.num_blocks),
            sim_offset_rounds=int(offset),
        )
        scenario_dir = output_dir / str(scenario_name)
        row = _existing_block_row(scenario_dir=scenario_dir)
        if row is None:
            raise InvariantError("dislocation_candidate_blocks_missing_block_summary")
        block_rows.append(dict(row))
        block_nets.append(float(row["net"]))
        bets_total += int(row["bets"])
        wins_total += int(row["wins"])

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
        "net_per_500_rounds": float(
            sum(block_nets) / float(int(args.block_size) * int(args.num_blocks)) * 500.0
        ),
        "bets_total": int(bets_total),
        "win_rate_weighted": float(_safe_rate(wins_total, bets_total)),
    }
    aggregate_path = output_dir / f"{str(args.name_prefix)}_aggregate.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "name_prefix": str(args.name_prefix),
                "block_size": int(args.block_size),
                "num_blocks": int(args.num_blocks),
                "skip_most_recent_blocks": int(args.skip_most_recent_blocks),
                "candidate_name": str(tuned_candidate.name),
                "candidate_overrides": dict(candidate_overrides),
                "config": {
                    "candidate": asdict(tuned_candidate),
                    "gas_profile": dict(gas_profile),
                },
                "rows": block_rows,
                "aggregate": agg,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"AGG={aggregate_path}")
    print(f"NET_PER_500={agg['net_per_500']}")


if __name__ == "__main__":
    main()
