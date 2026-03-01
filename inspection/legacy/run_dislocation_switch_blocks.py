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
    pool_total_bnb_cutoff: float | None
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
    pool_total = _safe_float(raw.get("pool_total_bnb_cutoff"))
    profit = _safe_float(raw.get("profit_bnb"))
    if profit is None:
        raise ValueError("trade_row_profit_missing")

    return TradeRow(
        epoch=int(epoch),
        action=str(action),
        direction=str(direction),
        expected_net_selected=expected_net,
        dislocation_bull=disloc,
        pool_total_bnb_cutoff=pool_total,
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
    a_rows: dict[int, TradeRow],
    b_rows: dict[int, TradeRow],
    score_window: int,
    min_history: int,
    alpha_expected_a: float,
    alpha_expected_b: float,
    beta_profit_mean: float,
    gamma_win_rate: float,
    eta_dislocation_abs_a: float,
    eta_dislocation_abs_b: float,
    score_bias_a: float,
    score_bias_b: float,
    cash_score_threshold: float,
    write_trades_path: Path | None,
) -> dict[str, Any]:
    epochs = sorted(set(a_rows.keys()) | set(b_rows.keys()))
    if not epochs:
        raise ValueError("switch_block_epochs_empty")

    a_profit_hist: deque[float] = deque(maxlen=int(score_window))
    b_profit_hist: deque[float] = deque(maxlen=int(score_window))
    a_win_hist: deque[int] = deque(maxlen=int(score_window))
    b_win_hist: deque[int] = deque(maxlen=int(score_window))

    net_profit = 0.0
    num_bets = 0
    num_wins = 0
    num_pick_a = 0
    num_pick_b = 0
    num_cash = 0

    trades_rows: list[list[Any]] = []
    if write_trades_path is not None:
        trades_rows.append(
            [
                "epoch",
                "action",
                "pick",
                "score_a",
                "score_b",
                "score_pick",
                "profit_bnb",
                "cum_net_bnb",
                "a_expected_net",
                "b_expected_net",
                "a_dislocation_bull",
                "b_dislocation_bull",
            ]
        )

    for ep in epochs:
        a_row = a_rows.get(int(ep))
        b_row = b_rows.get(int(ep))

        score_a = _score_candidate(
            row=a_row,
            history_profit=a_profit_hist,
            history_win=a_win_hist,
            alpha_expected=float(alpha_expected_a),
            beta_profit_mean=float(beta_profit_mean),
            gamma_win_rate=float(gamma_win_rate),
            eta_dislocation_abs=float(eta_dislocation_abs_a),
            score_bias=float(score_bias_a),
            min_history=int(min_history),
        )
        score_b = _score_candidate(
            row=b_row,
            history_profit=b_profit_hist,
            history_win=b_win_hist,
            alpha_expected=float(alpha_expected_b),
            beta_profit_mean=float(beta_profit_mean),
            gamma_win_rate=float(gamma_win_rate),
            eta_dislocation_abs=float(eta_dislocation_abs_b),
            score_bias=float(score_bias_b),
            min_history=int(min_history),
        )

        pick = "CASH"
        profit = 0.0
        pick_score = float("-inf")
        if float(score_a) >= float(score_b):
            pick = "A"
            pick_score = float(score_a)
            if pick_score >= float(cash_score_threshold) and a_row is not None and str(a_row.action) == "BET":
                profit = float(a_row.profit_bnb)
            else:
                pick = "CASH"
        else:
            pick = "B"
            pick_score = float(score_b)
            if pick_score >= float(cash_score_threshold) and b_row is not None and str(b_row.action) == "BET":
                profit = float(b_row.profit_bnb)
            else:
                pick = "CASH"

        if pick == "A":
            num_pick_a += 1
            num_bets += 1
            if profit > 0.0:
                num_wins += 1
        elif pick == "B":
            num_pick_b += 1
            num_bets += 1
            if profit > 0.0:
                num_wins += 1
        else:
            num_cash += 1

        net_profit += float(profit)

        if trades_rows:
            trades_rows.append(
                [
                    int(ep),
                    "BET" if pick in ("A", "B") else "SKIP",
                    str(pick),
                    float(score_a) if math.isfinite(score_a) else "",
                    float(score_b) if math.isfinite(score_b) else "",
                    float(pick_score) if math.isfinite(pick_score) else "",
                    float(profit),
                    float(net_profit),
                    None if a_row is None else a_row.expected_net_selected,
                    None if b_row is None else b_row.expected_net_selected,
                    None if a_row is None else a_row.dislocation_bull,
                    None if b_row is None else b_row.dislocation_bull,
                ]
            )

        # Update shadow histories with realized outcomes after the epoch closes.
        if a_row is not None and str(a_row.action) == "BET":
            a_profit_hist.append(float(a_row.profit_bnb))
            a_win_hist.append(1 if float(a_row.profit_bnb) > 0.0 else 0)
        if b_row is not None and str(b_row.action) == "BET":
            b_profit_hist.append(float(b_row.profit_bnb))
            b_win_hist.append(1 if float(b_row.profit_bnb) > 0.0 else 0)

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
        "num_pick_a": int(num_pick_a),
        "num_pick_b": int(num_pick_b),
        "num_cash": int(num_cash),
        "bet_rate": float(_safe_rate(int(num_bets), int(n_rounds))),
        "win_rate": float(_safe_rate(int(num_wins), int(num_bets))),
        "net_profit_bnb": float(net_profit),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--strategy-a-prefix", type=str, required=True)
    p.add_argument("--strategy-b-prefix", type=str, required=True)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--num-blocks", type=int, default=20)
    p.add_argument("--skip-most-recent-blocks", type=int, default=0)

    p.add_argument("--score-window", type=int, default=120)
    p.add_argument("--min-history", type=int, default=60)
    p.add_argument("--alpha-expected", type=float, default=1.0)
    p.add_argument("--alpha-expected-a", type=float, default=None)
    p.add_argument("--alpha-expected-b", type=float, default=None)
    p.add_argument("--beta-profit-mean", type=float, default=1.0)
    p.add_argument("--gamma-win-rate", type=float, default=0.5)
    p.add_argument("--eta-dislocation-abs", type=float, default=0.0)
    p.add_argument("--eta-dislocation-abs-a", type=float, default=None)
    p.add_argument("--eta-dislocation-abs-b", type=float, default=None)
    p.add_argument("--score-bias-a", type=float, default=0.0)
    p.add_argument("--score-bias-b", type=float, default=0.0)
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

    alpha_expected_a = (
        float(args.alpha_expected_a) if args.alpha_expected_a is not None else float(args.alpha_expected)
    )
    alpha_expected_b = (
        float(args.alpha_expected_b) if args.alpha_expected_b is not None else float(args.alpha_expected)
    )
    eta_dislocation_abs_a = (
        float(args.eta_dislocation_abs_a)
        if args.eta_dislocation_abs_a is not None
        else float(args.eta_dislocation_abs)
    )
    eta_dislocation_abs_b = (
        float(args.eta_dislocation_abs_b)
        if args.eta_dislocation_abs_b is not None
        else float(args.eta_dislocation_abs)
    )

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

    for block_idx, offset in enumerate(offsets, start=1):
        a_name = _build_scenario_name(
            name_prefix=str(args.strategy_a_prefix),
            block_idx=int(block_idx),
            num_blocks=int(args.num_blocks),
            offset=int(offset),
        )
        b_name = _build_scenario_name(
            name_prefix=str(args.strategy_b_prefix),
            block_idx=int(block_idx),
            num_blocks=int(args.num_blocks),
            offset=int(offset),
        )

        a_trades_path = out_dir / a_name / "dislocation_trades.csv"
        b_trades_path = out_dir / b_name / "dislocation_trades.csv"
        a_rows = _load_trades(a_trades_path)
        b_rows = _load_trades(b_trades_path)

        scenario_name = _build_scenario_name(
            name_prefix=str(args.name_prefix),
            block_idx=int(block_idx),
            num_blocks=int(args.num_blocks),
            offset=int(offset),
        )
        write_trades_path = (out_dir / scenario_name / "switch_trades.csv") if bool(args.write_trades) else None

        summary = _simulate_block(
            a_rows=a_rows,
            b_rows=b_rows,
            score_window=int(args.score_window),
            min_history=int(args.min_history),
            alpha_expected_a=float(alpha_expected_a),
            alpha_expected_b=float(alpha_expected_b),
            beta_profit_mean=float(args.beta_profit_mean),
            gamma_win_rate=float(args.gamma_win_rate),
            eta_dislocation_abs_a=float(eta_dislocation_abs_a),
            eta_dislocation_abs_b=float(eta_dislocation_abs_b),
            score_bias_a=float(args.score_bias_a),
            score_bias_b=float(args.score_bias_b),
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
            "pick_a": int(summary["num_pick_a"]),
            "pick_b": int(summary["num_pick_b"]),
            "pick_cash": int(summary["num_cash"]),
        }
        block_rows.append(row)

        nets.append(float(summary["net_profit_bnb"]))
        bets_total += int(summary["num_bets"])
        wins_total += int(summary["num_wins"])

        scenario_out = out_dir / scenario_name / "switch_summary.json"
        scenario_out.parent.mkdir(parents=True, exist_ok=True)
        scenario_out.write_text(
            json.dumps(
                {
                    "scenario": {
                        "name": str(scenario_name),
                        "strategy_a_prefix": str(args.strategy_a_prefix),
                        "strategy_b_prefix": str(args.strategy_b_prefix),
                        "score_window": int(args.score_window),
                        "min_history": int(args.min_history),
                        "alpha_expected": float(args.alpha_expected),
                        "alpha_expected_a": float(alpha_expected_a),
                        "alpha_expected_b": float(alpha_expected_b),
                        "beta_profit_mean": float(args.beta_profit_mean),
                        "gamma_win_rate": float(args.gamma_win_rate),
                        "eta_dislocation_abs": float(args.eta_dislocation_abs),
                        "eta_dislocation_abs_a": float(eta_dislocation_abs_a),
                        "eta_dislocation_abs_b": float(eta_dislocation_abs_b),
                        "score_bias_a": float(args.score_bias_a),
                        "score_bias_b": float(args.score_bias_b),
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
        "win_rate_weighted": float(_safe_rate(int(wins_total), int(bets_total))),
        "bet_rate_mean": float(sum(float(r["bet_rate"]) for r in block_rows) / len(block_rows)),
        "net_per_500_rounds": float((sum(nets) / (int(args.block_size) * len(block_rows))) * 500.0),
        "strategy_a_prefix": str(args.strategy_a_prefix),
        "strategy_b_prefix": str(args.strategy_b_prefix),
        "score_window": int(args.score_window),
        "min_history": int(args.min_history),
        "alpha_expected": float(args.alpha_expected),
        "alpha_expected_a": float(alpha_expected_a),
        "alpha_expected_b": float(alpha_expected_b),
        "beta_profit_mean": float(args.beta_profit_mean),
        "gamma_win_rate": float(args.gamma_win_rate),
        "eta_dislocation_abs": float(args.eta_dislocation_abs),
        "eta_dislocation_abs_a": float(eta_dislocation_abs_a),
        "eta_dislocation_abs_b": float(eta_dislocation_abs_b),
        "score_bias_a": float(args.score_bias_a),
        "score_bias_b": float(args.score_bias_b),
        "cash_score_threshold": float(args.cash_score_threshold),
    }

    blocks_csv = out_dir / f"{args.name_prefix}_blocks.csv"
    agg_csv = out_dir / f"{args.name_prefix}_aggregate.csv"
    agg_json = out_dir / f"{args.name_prefix}_aggregate.json"

    with blocks_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "block_index",
                "sim_offset_rounds",
                "scenario",
                "net",
                "bets",
                "wins",
                "bet_rate",
                "win_rate",
                "pick_a",
                "pick_b",
                "pick_cash",
            ],
        )
        w.writeheader()
        for row in block_rows:
            w.writerow(row)

    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "strategy_a_prefix",
                "strategy_b_prefix",
                "score_window",
                "min_history",
                "alpha_expected",
                "alpha_expected_a",
                "alpha_expected_b",
                "beta_profit_mean",
                "gamma_win_rate",
                "eta_dislocation_abs",
                "eta_dislocation_abs_a",
                "eta_dislocation_abs_b",
                "score_bias_a",
                "score_bias_b",
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
                "win_rate_weighted",
                "bet_rate_mean",
                "net_per_500_rounds",
            ],
        )
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
