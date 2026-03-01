"""Shared utilities for strategy-router inspection tooling.

This module is intentionally inspection-only and does not execute production
strategy logic. It reads historical strategy trade artifacts and exposes a
typed, deterministic view for router experiments.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_VALID_ACTIONS = ("BET", "SKIP")
_VALID_DIRECTIONS = ("BULL", "BEAR", "")
_CASH_STRATEGY_NAME = "CASH"


@dataclass(frozen=True, slots=True)
class StrategyTradeRow:
    """One strategy's trade decision/outcome for a single round."""

    epoch: int
    action: str
    direction: str
    expected_net_selected_bnb: float | None
    dislocation_bull: float | None
    p_nowcast_bull: float | None
    p_market_bull: float | None
    bet_size_bnb: float | None
    profit_bnb: float


@dataclass(frozen=True, slots=True)
class BlockRoundSnapshot:
    """All strategy rows aligned to one epoch inside one block."""

    block_index: int
    sim_offset_rounds: int
    epoch: int
    rows_by_strategy: dict[str, StrategyTradeRow | None]


def parse_strategy_prefixes(raw: str) -> list[str]:
    """Parse and validate comma-separated strategy prefixes."""

    prefixes = [str(x).strip() for x in str(raw).split(",") if str(x).strip()]
    if not prefixes:
        raise ValueError("strategy_prefixes_empty")

    seen: set[str] = set()
    unique: list[str] = []
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        unique.append(prefix)
    return unique


def build_block_offsets(
    *,
    block_size: int,
    num_blocks: int,
    skip_most_recent_blocks: int,
) -> list[int]:
    """Build block offsets ordered from oldest block to newest block."""

    if int(block_size) <= 0:
        raise ValueError("block_size_must_be_positive")
    if int(num_blocks) <= 0:
        raise ValueError("num_blocks_must_be_positive")
    if int(skip_most_recent_blocks) < 0:
        raise ValueError("skip_most_recent_blocks_negative")

    return [
        int(block_size) * i
        for i in range(
            int(num_blocks) + int(skip_most_recent_blocks) - 1,
            int(skip_most_recent_blocks) - 1,
            -1,
        )
    ]


def build_scenario_name(
    *,
    strategy_prefix: str,
    block_index: int,
    num_blocks: int,
    sim_offset_rounds: int,
) -> str:
    """Return the canonical block-scenario directory name."""

    return (
        f"{str(strategy_prefix)}_b{int(block_index)}of{int(num_blocks)}_"
        f"off{int(sim_offset_rounds)}"
    )


def _safe_float(value: Any) -> float | None:
    """Parse a finite float from CSV text; return None for missing/invalid."""

    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        parsed = float(raw)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _parse_action(raw: str) -> str:
    action = str(raw).strip().upper()
    if action not in _VALID_ACTIONS:
        raise ValueError(f"strategy_trade_action_invalid: {raw}")
    return action


def _parse_direction(*, action: str, raw: str) -> str:
    direction = str(raw).strip().upper()
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"strategy_trade_direction_invalid: {raw}")
    if str(action) == "BET" and str(direction) not in ("BULL", "BEAR"):
        raise ValueError("strategy_trade_direction_missing_for_bet")
    return direction


def load_strategy_trade_rows(trades_csv_path: Path) -> dict[int, StrategyTradeRow]:
    """Load one strategy's block trade CSV into epoch-indexed rows."""

    if not trades_csv_path.exists():
        raise FileNotFoundError(f"missing_strategy_trades_csv: {trades_csv_path}")

    rows: dict[int, StrategyTradeRow] = {}
    with trades_csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            epoch = int(raw["epoch"])
            action = _parse_action(str(raw.get("action", "")))
            direction = _parse_direction(
                action=action,
                raw=str(raw.get("direction", "")),
            )
            profit_bnb = _safe_float(raw.get("profit_bnb"))
            if profit_bnb is None:
                raise ValueError("strategy_trade_profit_missing")

            rows[int(epoch)] = StrategyTradeRow(
                epoch=int(epoch),
                action=str(action),
                direction=str(direction),
                expected_net_selected_bnb=_safe_float(raw.get("expected_net_selected")),
                dislocation_bull=_safe_float(raw.get("dislocation_bull")),
                p_nowcast_bull=_safe_float(raw.get("p_nowcast_bull")),
                p_market_bull=_safe_float(raw.get("p_market_bull")),
                bet_size_bnb=_safe_float(raw.get("bet_size_bnb")),
                profit_bnb=float(profit_bnb),
            )

    if not rows:
        raise ValueError(f"strategy_trades_csv_empty: {trades_csv_path}")
    return rows


def load_block_round_snapshots(
    *,
    strategy_prefixes: list[str],
    block_size: int,
    num_blocks: int,
    skip_most_recent_blocks: int,
    base_dir: Path,
    trades_filename: str,
) -> list[BlockRoundSnapshot]:
    """Load aligned per-round snapshots across all strategies and blocks."""

    offsets = build_block_offsets(
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        skip_most_recent_blocks=int(skip_most_recent_blocks),
    )

    snapshots: list[BlockRoundSnapshot] = []
    for block_index, sim_offset_rounds in enumerate(offsets, start=1):
        rows_by_strategy: dict[str, dict[int, StrategyTradeRow]] = {}
        all_epochs: set[int] = set()

        for strategy_prefix in strategy_prefixes:
            scenario_name = build_scenario_name(
                strategy_prefix=str(strategy_prefix),
                block_index=int(block_index),
                num_blocks=int(num_blocks),
                sim_offset_rounds=int(sim_offset_rounds),
            )
            trades_path = base_dir / scenario_name / str(trades_filename)
            rows = load_strategy_trade_rows(trades_path)
            rows_by_strategy[str(strategy_prefix)] = rows
            all_epochs |= set(rows.keys())

        if not all_epochs:
            raise ValueError("router_block_has_no_epochs")

        for epoch in sorted(all_epochs):
            per_strategy_row: dict[str, StrategyTradeRow | None] = {}
            for strategy_prefix in strategy_prefixes:
                per_strategy_row[str(strategy_prefix)] = rows_by_strategy[str(strategy_prefix)].get(
                    int(epoch)
                )
            snapshots.append(
                BlockRoundSnapshot(
                    block_index=int(block_index),
                    sim_offset_rounds=int(sim_offset_rounds),
                    epoch=int(epoch),
                    rows_by_strategy=per_strategy_row,
                )
            )
    return snapshots


def to_column_key_map(strategy_prefixes: list[str]) -> dict[str, str]:
    """Map strategy prefix -> unique lowercase ASCII-safe column key."""

    out: dict[str, str] = {}
    used: set[str] = set()
    for strategy_prefix in strategy_prefixes:
        raw = re.sub(r"[^a-zA-Z0-9]+", "_", str(strategy_prefix).lower()).strip("_")
        key = raw if raw else "strategy"
        suffix = 2
        while key in used:
            key = f"{raw}_{suffix}"
            suffix += 1
        used.add(key)
        out[str(strategy_prefix)] = str(key)
    return out


def direction_to_idx(direction: str) -> int:
    """Encode direction string to index for model tables."""

    direct = str(direction).strip().upper()
    if direct == "BULL":
        return 1
    if direct == "BEAR":
        return 0
    return -1


def oracle_cash_pick(
    rows_by_strategy: dict[str, StrategyTradeRow | None],
) -> tuple[str, float]:
    """Return hindsight best strategy-or-cash choice for one round.

    `CASH` yields exactly `0.0` BNB and is selected whenever no strategy has a
    strictly positive realized profit.
    """

    best_strategy = _CASH_STRATEGY_NAME
    best_profit_bnb = 0.0
    for strategy_prefix, row in rows_by_strategy.items():
        if row is None or str(row.action) != "BET":
            continue
        if float(row.profit_bnb) > float(best_profit_bnb):
            best_profit_bnb = float(row.profit_bnb)
            best_strategy = str(strategy_prefix)
    return str(best_strategy), float(best_profit_bnb)

