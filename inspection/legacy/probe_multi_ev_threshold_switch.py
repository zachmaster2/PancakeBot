from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TradeRow:
    epoch: int
    action: str
    expected_net_selected: float | None
    profit_bnb: float


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _load_trades(path: Path) -> dict[int, TradeRow]:
    if not path.exists():
        raise FileNotFoundError(f"missing_trades_csv: {path}")
    out: dict[int, TradeRow] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for raw in rd:
            ep = int(raw["epoch"])
            action = str(raw.get("action", "")).strip().upper()
            if action not in ("BET", "SKIP"):
                raise ValueError("trade_row_action_invalid")
            ev = _safe_float(raw.get("expected_net_selected"))
            p = _safe_float(raw.get("profit_bnb"))
            if p is None:
                raise ValueError("trade_row_profit_missing")
            out[ep] = TradeRow(epoch=ep, action=action, expected_net_selected=ev, profit_bnb=float(p))
    if not out:
        raise ValueError(f"empty_trades_csv: {path}")
    return out


def _offsets(*, block_size: int, num_blocks: int, skip_most_recent_blocks: int) -> list[int]:
    return [
        int(block_size) * i
        for i in range(int(num_blocks) + int(skip_most_recent_blocks) - 1, int(skip_most_recent_blocks) - 1, -1)
    ]


def _scenario_name(prefix: str, idx: int, num_blocks: int, offset: int) -> str:
    return f"{prefix}_b{idx}of{num_blocks}_off{offset}"


def _load_block(
    *,
    strategy_prefixes: list[str],
    block_idx: int,
    num_blocks: int,
    offset: int,
) -> tuple[list[int], dict[str, dict[int, TradeRow]]]:
    out_dir = Path("var/exp")
    rows_by_strategy: dict[str, dict[int, TradeRow]] = {}
    epochs: set[int] = set()
    for s in strategy_prefixes:
        name = _scenario_name(str(s), int(block_idx), int(num_blocks), int(offset))
        rows = _load_trades(out_dir / name / "dislocation_trades.csv")
        rows_by_strategy[s] = rows
        epochs |= set(rows.keys())
    return sorted(epochs), rows_by_strategy


def _load_dataset(
    *,
    strategy_prefixes: list[str],
    block_size: int,
    num_blocks: int,
    skip_most_recent_blocks: int,
) -> list[tuple[list[int], dict[str, dict[int, TradeRow]]]]:
    offsets = _offsets(
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        skip_most_recent_blocks=int(skip_most_recent_blocks),
    )
    out: list[tuple[list[int], dict[str, dict[int, TradeRow]]]] = []
    for block_idx, offset in enumerate(offsets, start=1):
        out.append(
            _load_block(
                strategy_prefixes=strategy_prefixes,
                block_idx=int(block_idx),
                num_blocks=int(num_blocks),
                offset=int(offset),
            )
        )
    return out


def _eval_cfg(
    *,
    block_size: int,
    num_blocks: int,
    strategy_prefixes: list[str],
    blocks_data: list[tuple[list[int], dict[str, dict[int, TradeRow]]]],
    thresholds: dict[str, float],
    biases: dict[str, float],
    score_floor: float,
    margin_delta: float,
) -> dict[str, Any]:
    block_nets: list[float] = []
    bets_total = 0
    wins_total = 0
    picks_total: dict[str, int] = {s: 0 for s in strategy_prefixes}

    for epochs, rows in blocks_data:
        net = 0.0
        for ep in epochs:
            scored: list[tuple[float, str, float]] = []
            for s in strategy_prefixes:
                r = rows[s].get(int(ep))
                if r is None or str(r.action) != "BET":
                    continue
                ev = float(r.expected_net_selected) if r.expected_net_selected is not None else 0.0
                if float(ev) < float(thresholds[s]):
                    continue
                score = float(ev) + float(biases[s])
                if float(score) < float(score_floor):
                    continue
                scored.append((float(score), str(s), float(r.profit_bnb)))

            if not scored:
                continue
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[0]
            second = scored[1][0] if len(scored) > 1 else float("-inf")
            if len(scored) > 1 and (float(top[0]) - float(second)) < float(margin_delta):
                continue
            net += float(top[2])
            bets_total += 1
            if float(top[2]) > 0.0:
                wins_total += 1
            picks_total[str(top[1])] += 1
        block_nets.append(float(net))

    total_rounds = int(block_size) * int(num_blocks)
    net_total = float(sum(block_nets))
    return {
        "net_per_500": float(net_total / float(total_rounds) * 500.0),
        "net_total": float(net_total),
        "net_median": float(statistics.median(block_nets)),
        "positive_block_frac": float(sum(1 for x in block_nets if float(x) > 0.0) / len(block_nets)),
        "bets_total": int(bets_total),
        "win_rate_weighted": float(float(wins_total) / float(bets_total)) if int(bets_total) > 0 else 0.0,
        "bet_rate": float(float(bets_total) / float(total_rounds)),
        "picks_total": {str(k): int(v) for k, v in picks_total.items()},
    }


def _run(args: argparse.Namespace) -> None:
    random.seed(int(args.seed))
    strategy_prefixes = [str(x).strip() for x in str(args.strategy_prefixes).split(",") if str(x).strip()]
    if len(strategy_prefixes) < 2:
        raise ValueError("strategy_prefixes_requires_at_least_two")
    blocks_data = _load_dataset(
        strategy_prefixes=strategy_prefixes,
        block_size=int(args.block_size),
        num_blocks=int(args.num_blocks),
        skip_most_recent_blocks=int(args.skip_most_recent_blocks),
    )

    th_vals = [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    bias_vals = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
    floor_vals = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    margin_vals = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    while len(rows) < int(args.num_samples):
        thresholds = {s: float(random.choice(th_vals)) for s in strategy_prefixes}
        biases = {s: float(random.choice(bias_vals)) for s in strategy_prefixes}
        score_floor = float(random.choice(floor_vals))
        margin_delta = float(random.choice(margin_vals))
        key = "|".join(
            [str(thresholds[s]) for s in strategy_prefixes]
            + [str(biases[s]) for s in strategy_prefixes]
            + [str(score_floor), str(margin_delta)]
        )
        if key in seen:
            continue
        seen.add(key)

        m = _eval_cfg(
            block_size=int(args.block_size),
            num_blocks=int(args.num_blocks),
            strategy_prefixes=strategy_prefixes,
            blocks_data=blocks_data,
            thresholds=thresholds,
            biases=biases,
            score_floor=float(score_floor),
            margin_delta=float(margin_delta),
        )
        r = {
            "thresholds": {str(k): float(v) for k, v in thresholds.items()},
            "biases": {str(k): float(v) for k, v in biases.items()},
            "score_floor": float(score_floor),
            "margin_delta": float(margin_delta),
            **m,
        }
        rows.append(r)
        if len(rows) % 100 == 0:
            best = max(rows, key=lambda x: float(x["net_per_500"]))
            print(f"SEARCH_PROGRESS n={len(rows)} best={best['net_per_500']}")

    rows_sorted = sorted(rows, key=lambda x: float(x["net_per_500"]), reverse=True)
    out_dir = Path("var/exp")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.name_prefix}.json"
    json_path.write_text(
        json.dumps(
            {
                "name_prefix": str(args.name_prefix),
                "strategy_prefixes": strategy_prefixes,
                "block_size": int(args.block_size),
                "num_blocks": int(args.num_blocks),
                "skip_most_recent_blocks": int(args.skip_most_recent_blocks),
                "num_samples": int(args.num_samples),
                "results": rows_sorted,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"JSON={json_path}")
    for i, r in enumerate(rows_sorted[:20], start=1):
        print(
            "TOP "
            + f"rank={i} net500={r['net_per_500']} bets={r['bets_total']} "
            + f"floor={r['score_floor']} delta={r['margin_delta']} "
            + f"thresholds={r['thresholds']} biases={r['biases']}"
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--strategy-prefixes", type=str, required=True)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--num-blocks", type=int, default=40)
    p.add_argument("--skip-most-recent-blocks", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    _run(args)


if __name__ == "__main__":
    main()
