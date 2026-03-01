from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TradeRow:
    epoch: int
    action: str
    direction: str
    expected_net_selected: float | None
    dislocation_bull: float | None
    profit_bnb: float


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _parse_trade_row(raw: dict[str, str]) -> TradeRow:
    action = str(raw.get("action", "")).strip().upper()
    if action not in ("BET", "SKIP"):
        raise ValueError("trade_row_action_invalid")
    epoch = int(raw["epoch"])
    direction = str(raw.get("direction", "")).strip().upper()
    if action == "BET" and direction not in ("BULL", "BEAR"):
        raise ValueError("trade_row_direction_invalid")
    expected_net = _safe_float(raw.get("expected_net_selected"))
    disloc = _safe_float(raw.get("dislocation_bull"))
    profit = _safe_float(raw.get("profit_bnb"))
    if profit is None:
        raise ValueError("trade_row_profit_missing")
    return TradeRow(
        epoch=int(epoch),
        action=str(action),
        direction=str(direction),
        expected_net_selected=expected_net,
        dislocation_bull=disloc,
        profit_bnb=float(profit),
    )


def _load_trades(path: Path) -> dict[int, TradeRow]:
    if not path.exists():
        raise FileNotFoundError(f"missing_trades_csv: {path}")
    out: dict[int, TradeRow] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for raw in rd:
            row = _parse_trade_row(raw)
            out[int(row.epoch)] = row
    if not out:
        raise ValueError(f"empty_trades_csv: {path}")
    return out


def _offsets(*, block_size: int, num_blocks: int, skip_most_recent_blocks: int) -> list[int]:
    return [
        int(block_size) * i
        for i in range(int(num_blocks) + int(skip_most_recent_blocks) - 1, int(skip_most_recent_blocks) - 1, -1)
    ]


def _build_scenario_name(name_prefix: str, block_idx: int, num_blocks: int, offset: int) -> str:
    return f"{name_prefix}_b{int(block_idx)}of{int(num_blocks)}_off{int(offset)}"


def _score_candidate(
    *,
    row: TradeRow | None,
    history_profit: deque[float],
    history_win: deque[int],
    alpha_expected: float,
    beta_profit_mean: float,
    gamma_win_rate: float,
    eta_dislocation_abs: float,
    score_bias: float,
    min_history: int,
) -> float:
    if row is None or str(row.action) != "BET":
        return float("-inf")

    expected = float(row.expected_net_selected) if row.expected_net_selected is not None else 0.0
    disloc_abs = abs(float(row.dislocation_bull)) if row.dislocation_bull is not None else 0.0
    score = (
        float(alpha_expected) * float(expected)
        + float(eta_dislocation_abs) * float(disloc_abs)
        + float(score_bias)
    )
    if len(history_profit) >= int(min_history) and len(history_win) >= int(min_history):
        mean_profit = float(sum(history_profit) / len(history_profit))
        win_rate = float(sum(history_win) / len(history_win))
        score += float(beta_profit_mean) * float(mean_profit)
        score += float(gamma_win_rate) * float(win_rate - 0.5)
    return float(score)


def _simulate_block(
    *,
    strategy_order: list[str],
    strategy_rows: dict[str, dict[int, TradeRow]],
    score_window: int,
    min_history: int,
    alpha_expected: float,
    beta_profit_mean: float,
    gamma_win_rate: float,
    eta_dislocation_abs: float,
    score_biases: dict[str, float],
    cash_score_threshold: float,
    write_trades_path: Path | None,
) -> dict[str, Any]:
    epochs_all: set[int] = set()
    for rows in strategy_rows.values():
        epochs_all |= set(rows.keys())
    epochs = sorted(epochs_all)
    if not epochs:
        raise ValueError("multi_switch_block_epochs_empty")

    profit_hist: dict[str, deque[float]] = {
        s: deque(maxlen=int(score_window)) for s in strategy_order
    }
    win_hist: dict[str, deque[int]] = {
        s: deque(maxlen=int(score_window)) for s in strategy_order
    }

    net_profit = 0.0
    num_bets = 0
    num_wins = 0
    num_cash = 0
    picks_by_strategy: dict[str, int] = {s: 0 for s in strategy_order}

    trades_rows: list[list[Any]] = []
    if write_trades_path is not None:
        header = ["epoch", "action", "pick", "pick_score", "profit_bnb", "cum_net_bnb"]
        for s in strategy_order:
            header.append(f"score_{s}")
            header.append(f"expected_{s}")
            header.append(f"dislocation_{s}")
        trades_rows.append(header)

    for ep in epochs:
        scores: dict[str, float] = {}
        for s in strategy_order:
            row = strategy_rows[s].get(int(ep))
            scores[s] = _score_candidate(
                row=row,
                history_profit=profit_hist[s],
                history_win=win_hist[s],
                alpha_expected=float(alpha_expected),
                beta_profit_mean=float(beta_profit_mean),
                gamma_win_rate=float(gamma_win_rate),
                eta_dislocation_abs=float(eta_dislocation_abs),
                score_bias=float(score_biases.get(s, 0.0)),
                min_history=int(min_history),
            )

        pick = "CASH"
        pick_score = float("-inf")
        profit = 0.0
        for s in strategy_order:
            sc = float(scores[s])
            if float(sc) > float(pick_score):
                pick_score = float(sc)
                pick = str(s)
        if not math.isfinite(float(pick_score)) or float(pick_score) < float(cash_score_threshold):
            pick = "CASH"
            pick_score = float("-inf")
        if pick != "CASH":
            row_pick = strategy_rows[pick].get(int(ep))
            if row_pick is None or str(row_pick.action) != "BET":
                pick = "CASH"
                pick_score = float("-inf")
            else:
                profit = float(row_pick.profit_bnb)
                num_bets += 1
                if profit > 0.0:
                    num_wins += 1
                picks_by_strategy[pick] += 1
        if pick == "CASH":
            num_cash += 1

        net_profit += float(profit)

        if trades_rows:
            out = [
                int(ep),
                "BET" if pick != "CASH" else "SKIP",
                str(pick),
                float(pick_score) if math.isfinite(float(pick_score)) else "",
                float(profit),
                float(net_profit),
            ]
            for s in strategy_order:
                row = strategy_rows[s].get(int(ep))
                sc = scores[s]
                out.append(float(sc) if math.isfinite(float(sc)) else "")
                out.append(None if row is None else row.expected_net_selected)
                out.append(None if row is None else row.dislocation_bull)
            trades_rows.append(out)

        # Update all strategy shadow histories with realized outcomes at this epoch.
        for s in strategy_order:
            row = strategy_rows[s].get(int(ep))
            if row is not None and str(row.action) == "BET":
                profit_hist[s].append(float(row.profit_bnb))
                win_hist[s].append(1 if float(row.profit_bnb) > 0.0 else 0)

    if write_trades_path is not None:
        write_trades_path.parent.mkdir(parents=True, exist_ok=True)
        with write_trades_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(trades_rows)

    n_rounds = int(len(epochs))
    return {
        "num_rounds": int(n_rounds),
        "num_bets": int(num_bets),
        "num_wins": int(num_wins),
        "num_cash": int(num_cash),
        "bet_rate": float(_safe_rate(int(num_bets), int(n_rounds))),
        "win_rate": float(_safe_rate(int(num_wins), int(num_bets))),
        "net_profit_bnb": float(net_profit),
        "picks_by_strategy": {str(k): int(v) for k, v in picks_by_strategy.items()},
    }


def _parse_csv_list(s: str) -> list[str]:
    return [str(x).strip() for x in str(s).split(",") if str(x).strip()]


def _parse_biases(s: str, strategy_order: list[str]) -> dict[str, float]:
    vals = _parse_csv_list(str(s))
    if not vals:
        return {k: 0.0 for k in strategy_order}
    if len(vals) != len(strategy_order):
        raise ValueError("score_biases_length_mismatch")
    out: dict[str, float] = {}
    for k, v in zip(strategy_order, vals):
        out[str(k)] = float(v)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--strategy-prefixes", type=str, required=True)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--num-blocks", type=int, default=20)
    p.add_argument("--skip-most-recent-blocks", type=int, default=0)
    p.add_argument("--score-window", type=int, default=120)
    p.add_argument("--min-history", type=int, default=60)
    p.add_argument("--alpha-expected", type=float, default=1.0)
    p.add_argument("--beta-profit-mean", type=float, default=1.0)
    p.add_argument("--gamma-win-rate", type=float, default=0.5)
    p.add_argument("--eta-dislocation-abs", type=float, default=0.0)
    p.add_argument("--score-biases", type=str, default="")
    p.add_argument("--cash-score-threshold", type=float, default=0.0)
    p.add_argument("--write-trades", action="store_true", default=False)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if int(args.block_size) <= 10:
        raise ValueError("block_size_too_small")
    if int(args.num_blocks) <= 0:
        raise ValueError("num_blocks_must_be_positive")
    if int(args.skip_most_recent_blocks) < 0:
        raise ValueError("skip_most_recent_blocks_negative")
    if int(args.score_window) <= 0:
        raise ValueError("score_window_must_be_positive")
    if int(args.min_history) < 0:
        raise ValueError("min_history_negative")
    if int(args.min_history) > int(args.score_window):
        raise ValueError("min_history_exceeds_score_window")

    strategy_order = _parse_csv_list(str(args.strategy_prefixes))
    if len(strategy_order) < 2:
        raise ValueError("strategy_prefixes_requires_at_least_two")
    score_biases = _parse_biases(str(args.score_biases), strategy_order=strategy_order)

    offsets = _offsets(
        block_size=int(args.block_size),
        num_blocks=int(args.num_blocks),
        skip_most_recent_blocks=int(args.skip_most_recent_blocks),
    )

    out_dir = Path("var/exp")
    out_dir.mkdir(parents=True, exist_ok=True)

    block_rows: list[dict[str, Any]] = []
    nets: list[float] = []
    bets_total = 0
    wins_total = 0
    picks_totals: dict[str, int] = {s: 0 for s in strategy_order}
    cash_total = 0

    for block_idx, offset in enumerate(offsets, start=1):
        strategy_rows: dict[str, dict[int, TradeRow]] = {}
        for sp in strategy_order:
            s_name = _build_scenario_name(
                name_prefix=str(sp),
                block_idx=int(block_idx),
                num_blocks=int(args.num_blocks),
                offset=int(offset),
            )
            s_path = out_dir / s_name / "dislocation_trades.csv"
            strategy_rows[str(sp)] = _load_trades(s_path)

        scenario_name = _build_scenario_name(
            name_prefix=str(args.name_prefix),
            block_idx=int(block_idx),
            num_blocks=int(args.num_blocks),
            offset=int(offset),
        )
        write_trades_path = (out_dir / scenario_name / "switch_trades.csv") if bool(args.write_trades) else None

        summary = _simulate_block(
            strategy_order=strategy_order,
            strategy_rows=strategy_rows,
            score_window=int(args.score_window),
            min_history=int(args.min_history),
            alpha_expected=float(args.alpha_expected),
            beta_profit_mean=float(args.beta_profit_mean),
            gamma_win_rate=float(args.gamma_win_rate),
            eta_dislocation_abs=float(args.eta_dislocation_abs),
            score_biases=score_biases,
            cash_score_threshold=float(args.cash_score_threshold),
            write_trades_path=write_trades_path,
        )

        row = {
            "scenario": str(scenario_name),
            "block_index": int(block_idx),
            "sim_offset_rounds": int(offset),
            "net": float(summary["net_profit_bnb"]),
            "bets": int(summary["num_bets"]),
            "wins": int(summary["num_wins"]),
            "bet_rate": float(summary["bet_rate"]),
            "win_rate": float(summary["win_rate"]),
            "pick_cash": int(summary["num_cash"]),
        }
        for s in strategy_order:
            row[f"pick_{s}"] = int(summary["picks_by_strategy"].get(s, 0))
        block_rows.append(row)

        nets.append(float(summary["net_profit_bnb"]))
        bets_total += int(summary["num_bets"])
        wins_total += int(summary["num_wins"])
        cash_total += int(summary["num_cash"])
        for s in strategy_order:
            picks_totals[s] += int(summary["picks_by_strategy"].get(s, 0))

        scenario_out = out_dir / scenario_name / "switch_summary.json"
        scenario_out.parent.mkdir(parents=True, exist_ok=True)
        scenario_out.write_text(
            json.dumps(
                {
                    "scenario": {
                        "name": str(scenario_name),
                        "strategy_prefixes": [str(x) for x in strategy_order],
                        "score_window": int(args.score_window),
                        "min_history": int(args.min_history),
                        "alpha_expected": float(args.alpha_expected),
                        "beta_profit_mean": float(args.beta_profit_mean),
                        "gamma_win_rate": float(args.gamma_win_rate),
                        "eta_dislocation_abs": float(args.eta_dislocation_abs),
                        "score_biases": {str(k): float(v) for k, v in score_biases.items()},
                        "cash_score_threshold": float(args.cash_score_threshold),
                        "block_size": int(args.block_size),
                        "sim_offset_rounds": int(offset),
                    },
                    **summary,
                },
                indent=2,
                sort_keys=True,
            )
        )

        print(
            "BLOCK_DONE "
            + f"block={block_idx}/{args.num_blocks} "
            + f"offset={offset} "
            + f"net={row['net']} bets={row['bets']} win={row['win_rate']}"
        )

    agg = {
        "blocks": int(len(block_rows)),
        "net_total": float(sum(nets)),
        "net_mean": float(sum(nets) / len(nets)),
        "net_median": float(statistics.median(nets)),
        "net_worst": float(min(nets)),
        "net_best": float(max(nets)),
        "positive_blocks": int(sum(1 for x in nets if float(x) > 0.0)),
        "positive_block_frac": float(sum(1 for x in nets if float(x) > 0.0) / len(nets)),
        "bets_total": int(bets_total),
        "wins_total": int(wins_total),
        "cash_total": int(cash_total),
        "win_rate_weighted": float(_safe_rate(int(wins_total), int(bets_total))),
        "bet_rate_mean": float(sum(float(r["bet_rate"]) for r in block_rows) / len(block_rows)),
        "net_per_500_rounds": float((sum(nets) / (int(args.block_size) * len(block_rows))) * 500.0),
        "strategy_prefixes": ",".join(str(x) for x in strategy_order),
        "score_window": int(args.score_window),
        "min_history": int(args.min_history),
        "alpha_expected": float(args.alpha_expected),
        "beta_profit_mean": float(args.beta_profit_mean),
        "gamma_win_rate": float(args.gamma_win_rate),
        "eta_dislocation_abs": float(args.eta_dislocation_abs),
        "cash_score_threshold": float(args.cash_score_threshold),
    }
    for s in strategy_order:
        agg[f"picks_total_{s}"] = int(picks_totals[s])
    for s in strategy_order:
        agg[f"score_bias_{s}"] = float(score_biases.get(s, 0.0))

    blocks_csv = out_dir / f"{args.name_prefix}_blocks.csv"
    agg_csv = out_dir / f"{args.name_prefix}_aggregate.csv"
    agg_json = out_dir / f"{args.name_prefix}_aggregate.json"

    block_fields = [
        "block_index",
        "sim_offset_rounds",
        "scenario",
        "net",
        "bets",
        "wins",
        "bet_rate",
        "win_rate",
        "pick_cash",
    ] + [f"pick_{s}" for s in strategy_order]
    with blocks_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=block_fields)
        w.writeheader()
        for row in block_rows:
            w.writerow(row)

    agg_fields = [
        "strategy_prefixes",
        "score_window",
        "min_history",
        "alpha_expected",
        "beta_profit_mean",
        "gamma_win_rate",
        "eta_dislocation_abs",
        "cash_score_threshold",
        "blocks",
        "net_total",
        "net_mean",
        "net_median",
        "net_worst",
        "net_best",
        "positive_blocks",
        "positive_block_frac",
        "bets_total",
        "wins_total",
        "cash_total",
        "win_rate_weighted",
        "bet_rate_mean",
        "net_per_500_rounds",
    ] + [f"picks_total_{s}" for s in strategy_order] + [f"score_bias_{s}" for s in strategy_order]
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerow(agg)

    agg_json.write_text(json.dumps({"aggregate": agg, "blocks": block_rows}, indent=2, sort_keys=True))

    print(f"BLOCKS_CSV={blocks_csv}")
    print(f"AGG_CSV={agg_csv}")
    print(f"AGG_JSON={agg_json}")
    print(
        "SUMMARY "
        + f"net_total={agg['net_total']} "
        + f"net_median={agg['net_median']} "
        + f"positive_frac={agg['positive_block_frac']} "
        + f"net_per_500={agg['net_per_500_rounds']}"
    )


if __name__ == "__main__":
    main()

