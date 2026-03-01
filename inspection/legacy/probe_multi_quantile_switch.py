from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import statistics
from collections import deque
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


def _epochs_for_block(
    *,
    strategy_prefixes: list[str],
    block_idx: int,
    num_blocks: int,
    offset: int,
) -> tuple[list[int], dict[str, dict[int, TradeRow]]]:
    out_dir = Path("var/exp")
    rows_by_strategy: dict[str, dict[int, TradeRow]] = {}
    epochs: set[int] = set()
    for sp in strategy_prefixes:
        name = _scenario_name(str(sp), int(block_idx), int(num_blocks), int(offset))
        path = out_dir / name / "dislocation_trades.csv"
        rows = _load_trades(path)
        rows_by_strategy[str(sp)] = rows
        epochs |= set(rows.keys())
    return sorted(epochs), rows_by_strategy


def _quantile_from_sorted(sorted_vals: list[float], x: float) -> float:
    n = int(len(sorted_vals))
    if int(n) <= 0:
        return 0.0
    pos = bisect.bisect_right(sorted_vals, float(x))
    return float(pos) / float(n)


def _eval_config(
    *,
    strategy_prefixes: list[str],
    block_size: int,
    num_blocks: int,
    skip_most_recent_blocks: int,
    q_window: int,
    q_min: float,
    w_q: float,
    w_win: float,
    w_mean: float,
    perf_window: int,
    score_threshold: float,
) -> dict[str, Any]:
    offsets = _offsets(
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        skip_most_recent_blocks=int(skip_most_recent_blocks),
    )

    # Per-strategy rolling state
    ev_hist: dict[str, deque[float]] = {s: deque(maxlen=int(q_window)) for s in strategy_prefixes}
    profit_hist: dict[str, deque[float]] = {s: deque(maxlen=int(perf_window)) for s in strategy_prefixes}
    win_hist: dict[str, deque[int]] = {s: deque(maxlen=int(perf_window)) for s in strategy_prefixes}

    block_nets: list[float] = []
    bets_total = 0
    wins_total = 0
    picks_total: dict[str, int] = {s: 0 for s in strategy_prefixes}

    for block_idx, offset in enumerate(offsets, start=1):
        epochs, rows_by_strategy = _epochs_for_block(
            strategy_prefixes=strategy_prefixes,
            block_idx=int(block_idx),
            num_blocks=int(num_blocks),
            offset=int(offset),
        )
        block_net = 0.0

        for ep in epochs:
            best_s: str | None = None
            best_score = float("-inf")
            best_profit = 0.0
            any_eligible = False

            for s in strategy_prefixes:
                row = rows_by_strategy[s].get(int(ep))
                if row is None or str(row.action) != "BET":
                    continue
                ev_now = float(row.expected_net_selected) if row.expected_net_selected is not None else 0.0

                q = 0.0
                if len(ev_hist[s]) > 0:
                    q = _quantile_from_sorted(sorted(float(v) for v in ev_hist[s]), float(ev_now))

                # If no history yet, use neutral quantile=0.5 so early blocks can still bet.
                if len(ev_hist[s]) == 0:
                    q = 0.5

                win_rate = 0.0
                if len(win_hist[s]) > 0:
                    win_rate = float(sum(win_hist[s])) / float(len(win_hist[s]))
                mean_profit = 0.0
                if len(profit_hist[s]) > 0:
                    mean_profit = float(sum(profit_hist[s])) / float(len(profit_hist[s]))

                score = (
                    float(w_q) * float(q)
                    + float(w_win) * float(win_rate - 0.5)
                    + float(w_mean) * float(mean_profit)
                )
                if float(q) < float(q_min):
                    continue
                if float(score) < float(score_threshold):
                    continue

                any_eligible = True
                if float(score) > float(best_score):
                    best_score = float(score)
                    best_s = str(s)
                    best_profit = float(row.profit_bnb)

            if best_s is not None:
                block_net += float(best_profit)
                bets_total += 1
                if float(best_profit) > 0.0:
                    wins_total += 1
                picks_total[best_s] += 1

            # Shadow updates for all strategies after round closes.
            for s in strategy_prefixes:
                row = rows_by_strategy[s].get(int(ep))
                if row is None or str(row.action) != "BET":
                    continue
                ev_now = float(row.expected_net_selected) if row.expected_net_selected is not None else 0.0
                ev_hist[s].append(float(ev_now))
                profit_hist[s].append(float(row.profit_bnb))
                win_hist[s].append(1 if float(row.profit_bnb) > 0.0 else 0)

        block_nets.append(float(block_net))

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


def _run_search(args: argparse.Namespace) -> None:
    random.seed(int(args.seed))

    strategy_prefixes = [str(x).strip() for x in str(args.strategy_prefixes).split(",") if str(x).strip()]
    if len(strategy_prefixes) < 2:
        raise ValueError("strategy_prefixes_requires_at_least_two")

    q_window_vals = [20, 30, 40, 60, 90, 120, 180, 240, 320, 480]
    q_min_vals = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    w_q_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    w_win_vals = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
    w_mean_vals = [-20.0, -10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    perf_window_vals = [20, 30, 40, 60, 90, 120, 180, 240]
    score_thr_vals = [-2.0, -1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]

    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    while len(results) < int(args.num_samples):
        q_window = int(random.choice(q_window_vals))
        q_min = float(random.choice(q_min_vals))
        w_q = float(random.choice(w_q_vals))
        w_win = float(random.choice(w_win_vals))
        w_mean = float(random.choice(w_mean_vals))
        perf_window = int(random.choice(perf_window_vals))
        score_threshold = float(random.choice(score_thr_vals))
        k = "|".join(
            str(x)
            for x in (q_window, q_min, w_q, w_win, w_mean, perf_window, score_threshold)
        )
        if k in seen:
            continue
        seen.add(k)

        metrics = _eval_config(
            strategy_prefixes=strategy_prefixes,
            block_size=int(args.block_size),
            num_blocks=int(args.num_blocks),
            skip_most_recent_blocks=int(args.skip_most_recent_blocks),
            q_window=int(q_window),
            q_min=float(q_min),
            w_q=float(w_q),
            w_win=float(w_win),
            w_mean=float(w_mean),
            perf_window=int(perf_window),
            score_threshold=float(score_threshold),
        )
        row = {
            "q_window": int(q_window),
            "q_min": float(q_min),
            "w_q": float(w_q),
            "w_win": float(w_win),
            "w_mean": float(w_mean),
            "perf_window": int(perf_window),
            "score_threshold": float(score_threshold),
            **metrics,
        }
        results.append(row)
        if len(results) % 50 == 0:
            best = max(results, key=lambda r: float(r["net_per_500"]))
            print(f"SEARCH_PROGRESS n={len(results)} best={best['net_per_500']}")

    results_sorted = sorted(results, key=lambda r: float(r["net_per_500"]), reverse=True)
    out_dir = Path("var/exp")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.name_prefix}.csv"
    json_path = out_dir / f"{args.name_prefix}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "q_window",
            "q_min",
            "w_q",
            "w_win",
            "w_mean",
            "perf_window",
            "score_threshold",
            "net_per_500",
            "net_total",
            "net_median",
            "positive_block_frac",
            "bets_total",
            "win_rate_weighted",
            "bet_rate",
            "picks_total",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results_sorted:
            w.writerow(r)

    json_path.write_text(
        json.dumps(
            {
                "name_prefix": str(args.name_prefix),
                "strategy_prefixes": strategy_prefixes,
                "block_size": int(args.block_size),
                "num_blocks": int(args.num_blocks),
                "skip_most_recent_blocks": int(args.skip_most_recent_blocks),
                "num_samples": int(args.num_samples),
                "results": results_sorted,
            },
            indent=2,
            sort_keys=True,
        )
    )

    top = results_sorted[:20]
    print(f"CSV={csv_path}")
    print(f"JSON={json_path}")
    for idx, r in enumerate(top, start=1):
        print(
            "TOP "
            + f"rank={idx} "
            + f"net500={r['net_per_500']} "
            + f"q_window={r['q_window']} "
            + f"q_min={r['q_min']} "
            + f"wq={r['w_q']} "
            + f"wwin={r['w_win']} "
            + f"wmean={r['w_mean']} "
            + f"pwin={r['perf_window']} "
            + f"thr={r['score_threshold']} "
            + f"bets={r['bets_total']}"
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--strategy-prefixes", type=str, required=True)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--num-blocks", type=int, default=40)
    p.add_argument("--skip-most-recent-blocks", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    _run_search(args)


if __name__ == "__main__":
    main()

