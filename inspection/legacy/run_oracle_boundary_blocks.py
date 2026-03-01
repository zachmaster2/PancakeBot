from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.types import Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round


@dataclass(frozen=True, slots=True)
class KlineIndex:
    close_times_ms: list[int]
    close_prices: list[float]

    def spot_at_or_before(self, ts_ms: int) -> float | None:
        idx = bisect.bisect_right(self.close_times_ms, int(ts_ms)) - 1
        if int(idx) < 0:
            return None
        return float(self.close_prices[int(idx)])


@dataclass(frozen=True, slots=True)
class Decision:
    side: str | None
    reason: str
    boundary_dev: float | None
    flow_imbalance: float | None


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _load_rounds(path: Path) -> list[Round]:
    if not path.exists():
        raise FileNotFoundError(f"missing_rounds_jsonl: {path}")
    out: list[Round] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = str(line).strip()
            if not s:
                continue
            r = Round.from_json(json.loads(s))
            if bool(r.failed):
                continue
            if r.lock_at is None or r.close_at is None or r.lock_price is None or r.close_price is None:
                continue
            if float(r.lock_price) <= 0.0 or float(r.close_price) <= 0.0:
                continue
            out.append(r)
    if not out:
        raise InvariantError("oracle_boundary_rounds_empty")
    prev = int(out[0].epoch)
    for r in out[1:]:
        e = int(r.epoch)
        if int(e) <= int(prev):
            raise InvariantError("oracle_boundary_rounds_not_strictly_ascending")
        prev = int(e)
    return out


def _load_kline_index(path: Path) -> KlineIndex:
    if not path.exists():
        raise FileNotFoundError(f"missing_klines_jsonl: {path}")
    times: list[int] = []
    prices: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = str(line).strip()
            if not s:
                continue
            obj = json.loads(s)
            ct = int(obj["close_time_ms"])
            cp = float(obj["close_price"])
            if times and int(ct) <= int(times[-1]):
                raise InvariantError("oracle_boundary_klines_not_strictly_ascending")
            times.append(int(ct))
            prices.append(float(cp))
    if not times:
        raise InvariantError("oracle_boundary_klines_empty")
    return KlineIndex(close_times_ms=times, close_prices=prices)


def _late_flow_imbalance(round_t: Round, *, flow_window_seconds: int) -> float | None:
    if int(flow_window_seconds) <= 0:
        raise InvariantError("flow_window_seconds_must_be_positive")
    if round_t.lock_at is None:
        return None
    lock_ts = int(round_t.lock_at)
    start_ts = int(lock_ts) - int(flow_window_seconds)
    bull = 0
    bear = 0
    for b in round_t.bets:
        t = int(b.created_at)
        if int(t) > int(lock_ts) or int(t) < int(start_ts):
            continue
        if str(b.position) == "Bull":
            bull += int(b.amount_wei)
        elif str(b.position) == "Bear":
            bear += int(b.amount_wei)
    tot = int(bull) + int(bear)
    if int(tot) <= 0:
        return None
    return float((int(bull) - int(bear)) / float(tot))


def _direction_from_sign(sign: int) -> str:
    if int(sign) > 0:
        return "BULL"
    if int(sign) < 0:
        return "BEAR"
    raise InvariantError("sign_zero_has_no_direction")


def _decide(
    *,
    prev_round: Round,
    round_t: Round,
    kidx: KlineIndex,
    cutoff_seconds: int,
    boundary_threshold: float,
    flow_window_seconds: int,
    flow_min_imbalance: float,
    rule: str,
) -> Decision:
    if prev_round.lock_price is None:
        return Decision(side=None, reason="prev_lock_price_missing", boundary_dev=None, flow_imbalance=None)
    if round_t.lock_at is None:
        return Decision(side=None, reason="round_lock_at_missing", boundary_dev=None, flow_imbalance=None)

    anchor = float(prev_round.lock_price)
    if float(anchor) <= 0.0:
        return Decision(side=None, reason="prev_lock_price_nonpositive", boundary_dev=None, flow_imbalance=None)

    cutoff_ms = int(int(round_t.lock_at) - int(cutoff_seconds)) * 1000
    spot = kidx.spot_at_or_before(int(cutoff_ms))
    if spot is None:
        return Decision(side=None, reason="no_boundary_spot", boundary_dev=None, flow_imbalance=None)

    dev = (float(spot) - float(anchor)) / float(anchor)
    dev_sign = 1 if float(dev) > 0.0 else (-1 if float(dev) < 0.0 else 0)
    dev_mag = abs(float(dev))

    flow_imb = _late_flow_imbalance(round_t, flow_window_seconds=int(flow_window_seconds))
    flow_sign = 0 if flow_imb is None else (1 if float(flow_imb) > 0.0 else (-1 if float(flow_imb) < 0.0 else 0))
    flow_mag = 0.0 if flow_imb is None else abs(float(flow_imb))

    needs_dev = bool(
        rule
        in (
            "dev_follow",
            "dev_contra",
            "flow_follow_and_dev_gate",
            "flow_contra_and_dev_gate",
            "flow_follow_when_agree_dev",
            "flow_follow_when_opp_dev",
        )
    )
    needs_flow = bool(
        rule
        in (
            "flow_follow",
            "flow_contra",
            "flow_follow_and_dev_gate",
            "flow_contra_and_dev_gate",
            "flow_follow_when_agree_dev",
            "flow_follow_when_opp_dev",
        )
    )

    if needs_dev and float(dev_mag) < float(boundary_threshold):
        return Decision(side=None, reason="boundary_below_threshold", boundary_dev=float(dev), flow_imbalance=flow_imb)
    if needs_flow and int(flow_sign) == 0:
        return Decision(side=None, reason="flow_unavailable", boundary_dev=float(dev), flow_imbalance=flow_imb)
    if needs_flow and float(flow_mag) < float(flow_min_imbalance):
        return Decision(side=None, reason="flow_below_min_imbalance", boundary_dev=float(dev), flow_imbalance=flow_imb)

    if str(rule) == "flow_follow":
        return Decision(side=_direction_from_sign(int(flow_sign)), reason="bet", boundary_dev=float(dev), flow_imbalance=flow_imb)
    if str(rule) == "flow_contra":
        return Decision(side=_direction_from_sign(-int(flow_sign)), reason="bet", boundary_dev=float(dev), flow_imbalance=flow_imb)
    if str(rule) == "dev_follow":
        if int(dev_sign) == 0:
            return Decision(side=None, reason="dev_zero", boundary_dev=float(dev), flow_imbalance=flow_imb)
        return Decision(side=_direction_from_sign(int(dev_sign)), reason="bet", boundary_dev=float(dev), flow_imbalance=flow_imb)
    if str(rule) == "dev_contra":
        if int(dev_sign) == 0:
            return Decision(side=None, reason="dev_zero", boundary_dev=float(dev), flow_imbalance=flow_imb)
        return Decision(side=_direction_from_sign(-int(dev_sign)), reason="bet", boundary_dev=float(dev), flow_imbalance=flow_imb)
    if str(rule) == "flow_follow_and_dev_gate":
        return Decision(side=_direction_from_sign(int(flow_sign)), reason="bet", boundary_dev=float(dev), flow_imbalance=flow_imb)
    if str(rule) == "flow_contra_and_dev_gate":
        return Decision(side=_direction_from_sign(-int(flow_sign)), reason="bet", boundary_dev=float(dev), flow_imbalance=flow_imb)
    if str(rule) == "flow_follow_when_agree_dev":
        if int(flow_sign) != int(dev_sign):
            return Decision(side=None, reason="signal_relation_mismatch", boundary_dev=float(dev), flow_imbalance=flow_imb)
        return Decision(side=_direction_from_sign(int(flow_sign)), reason="bet", boundary_dev=float(dev), flow_imbalance=flow_imb)
    if str(rule) == "flow_follow_when_opp_dev":
        if int(flow_sign) == int(dev_sign):
            return Decision(side=None, reason="signal_relation_mismatch", boundary_dev=float(dev), flow_imbalance=flow_imb)
        return Decision(side=_direction_from_sign(int(flow_sign)), reason="bet", boundary_dev=float(dev), flow_imbalance=flow_imb)

    raise InvariantError("oracle_boundary_rule_unknown")


def _slice_with_offset(*, rounds: list[Round], block_size: int, offset_rounds: int) -> list[Round]:
    if int(block_size) <= 0:
        raise InvariantError("block_size_must_be_positive")
    if int(offset_rounds) < 0:
        raise InvariantError("offset_rounds_negative")
    required = int(block_size) + int(offset_rounds)
    if len(rounds) < int(required):
        raise InvariantError("insufficient_rounds_for_offset_slice")
    tail = list(rounds[-int(required):])
    if int(offset_rounds) > 0:
        tail = tail[:-int(offset_rounds)]
    out = tail[-int(block_size):]
    if len(out) != int(block_size):
        raise InvariantError("offset_slice_len_mismatch")
    return out


def _simulate_block(
    *,
    rounds_block: list[Round],
    kidx: KlineIndex,
    cutoff_seconds: int,
    boundary_threshold: float,
    flow_window_seconds: int,
    flow_min_imbalance: float,
    rule: str,
    fixed_bet_bnb: float,
    initial_bankroll_bnb: float,
    treasury_fee_fraction: float,
    write_trades_path: Path | None,
) -> dict[str, Any]:
    bankroll = float(initial_bankroll_bnb)
    wins = 0
    bets = 0
    bets_bull = 0
    bets_bear = 0
    gross_profit = 0.0
    gross_loss = 0.0
    skip_counts: dict[str, int] = {}

    trades_rows: list[list[Any]] = []
    if write_trades_path is not None:
        trades_rows.append(
            [
                "epoch",
                "action",
                "skip_reason",
                "direction",
                "boundary_dev",
                "flow_imbalance",
                "bet_size_bnb",
                "profit_bnb",
                "bankroll_bnb",
            ]
        )

    for idx in range(1, len(rounds_block)):
        prev_r = rounds_block[idx - 1]
        r = rounds_block[idx]

        dec = _decide(
            prev_round=prev_r,
            round_t=r,
            kidx=kidx,
            cutoff_seconds=int(cutoff_seconds),
            boundary_threshold=float(boundary_threshold),
            flow_window_seconds=int(flow_window_seconds),
            flow_min_imbalance=float(flow_min_imbalance),
            rule=str(rule),
        )

        if dec.side is None:
            key = str(dec.reason)
            skip_counts[key] = int(skip_counts.get(key, 0)) + 1
            if trades_rows:
                trades_rows.append(
                    [
                        int(r.epoch),
                        "SKIP",
                        str(dec.reason),
                        "",
                        None if dec.boundary_dev is None else float(dec.boundary_dev),
                        None if dec.flow_imbalance is None else float(dec.flow_imbalance),
                        0.0,
                        0.0,
                        float(bankroll),
                    ]
                )
            continue

        total_cost = float(fixed_bet_bnb) + float(GAS_COST_BET_BNB)
        if float(bankroll) < float(total_cost):
            skip_counts["insufficient_bankroll"] = int(skip_counts.get("insufficient_bankroll", 0)) + 1
            if trades_rows:
                trades_rows.append(
                    [
                        int(r.epoch),
                        "SKIP",
                        "insufficient_bankroll",
                        str(dec.side),
                        None if dec.boundary_dev is None else float(dec.boundary_dev),
                        None if dec.flow_imbalance is None else float(dec.flow_imbalance),
                        0.0,
                        0.0,
                        float(bankroll),
                    ]
                )
            continue

        bankroll -= float(total_cost)
        outcome = settle_bet_against_closed_round(
            bet_bnb=float(fixed_bet_bnb),
            bet_side=str(dec.side),
            round_closed=r,
            treasury_fee_fraction=float(treasury_fee_fraction),
        )
        bankroll += float(outcome.credit_bnb)

        profit = -float(total_cost) + float(outcome.credit_bnb)
        if float(profit) >= 0.0:
            gross_profit += float(profit)
        else:
            gross_loss += -float(profit)

        bets += 1
        if str(dec.side) == "BULL":
            bets_bull += 1
        else:
            bets_bear += 1
        if str(outcome.outcome) == "win":
            wins += 1

        if trades_rows:
            trades_rows.append(
                [
                    int(r.epoch),
                    "BET",
                    "",
                    str(dec.side),
                    None if dec.boundary_dev is None else float(dec.boundary_dev),
                    None if dec.flow_imbalance is None else float(dec.flow_imbalance),
                    float(fixed_bet_bnb),
                    float(profit),
                    float(bankroll),
                ]
            )

    if write_trades_path is not None:
        write_trades_path.parent.mkdir(parents=True, exist_ok=True)
        with write_trades_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(trades_rows)

    n_rounds = int(len(rounds_block))
    n_eval = int(max(0, n_rounds - 1))
    net = float(bankroll - float(initial_bankroll_bnb))
    return {
        "num_rounds": int(n_rounds),
        "num_eval_rounds": int(n_eval),
        "num_bets": int(bets),
        "num_wins": int(wins),
        "num_bets_bull": int(bets_bull),
        "num_bets_bear": int(bets_bear),
        "bet_rate": float(_safe_rate(int(bets), int(n_eval))),
        "win_rate": float(_safe_rate(int(wins), int(bets))),
        "gross_profit_bnb": float(gross_profit),
        "gross_loss_bnb": float(gross_loss),
        "net_profit_bnb": float(net),
        "initial_bankroll_bnb": float(initial_bankroll_bnb),
        "final_bankroll_bnb": float(bankroll),
        "skip_reason_counts": {str(k): int(v) for k, v in sorted(skip_counts.items())},
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--closed-rounds-path", type=str, default="var/closed_rounds.jsonl")
    p.add_argument("--klines-path", type=str, default="var/klines.jsonl")
    p.add_argument("--block-size", type=int, default=10000)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--skip-most-recent-blocks", type=int, default=0)
    p.add_argument("--cutoff-seconds", type=int, default=30)
    p.add_argument("--boundary-threshold-bps", type=float, default=10.0)
    p.add_argument("--flow-window-seconds", type=int, default=60)
    p.add_argument("--flow-min-imbalance", type=float, default=0.60)
    p.add_argument(
        "--rule",
        type=str,
        choices=(
            "flow_follow",
            "flow_contra",
            "dev_follow",
            "dev_contra",
            "flow_follow_and_dev_gate",
            "flow_contra_and_dev_gate",
            "flow_follow_when_agree_dev",
            "flow_follow_when_opp_dev",
        ),
        default="flow_follow_when_opp_dev",
    )
    p.add_argument("--fixed-bet-bnb", type=float, default=0.05)
    p.add_argument("--initial-bankroll-bnb", type=float, default=0.5)
    p.add_argument("--treasury-fee-fraction", type=float, default=0.03)
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
    if int(args.cutoff_seconds) < 0:
        raise ValueError("cutoff_seconds_negative")
    if float(args.boundary_threshold_bps) < 0.0:
        raise ValueError("boundary_threshold_bps_negative")
    if int(args.flow_window_seconds) <= 0:
        raise ValueError("flow_window_seconds_must_be_positive")
    if not (0.0 <= float(args.flow_min_imbalance) <= 1.0):
        raise ValueError("flow_min_imbalance_out_of_range")
    if float(args.fixed_bet_bnb) <= 0.0:
        raise ValueError("fixed_bet_bnb_must_be_positive")
    if float(args.initial_bankroll_bnb) <= 0.0:
        raise ValueError("initial_bankroll_bnb_must_be_positive")
    if not (0.0 <= float(args.treasury_fee_fraction) < 1.0):
        raise ValueError("treasury_fee_fraction_out_of_range")

    rounds = _load_rounds(Path(str(args.closed_rounds_path)))
    kidx = _load_kline_index(Path(str(args.klines_path)))

    boundary_threshold = float(args.boundary_threshold_bps) / 10000.0
    skip = int(args.skip_most_recent_blocks)
    offsets = [
        int(args.block_size) * i
        for i in range(int(args.num_blocks) + int(skip) - 1, int(skip) - 1, -1)
    ]

    out_dir = Path("var/exp")
    out_dir.mkdir(parents=True, exist_ok=True)

    block_rows: list[dict[str, Any]] = []
    nets: list[float] = []
    bets_total = 0
    wins_total = 0

    for block_idx, offset in enumerate(offsets, start=1):
        scenario_name = f"{args.name_prefix}_b{int(block_idx)}of{int(args.num_blocks)}_off{int(offset)}"
        block = _slice_with_offset(rounds=rounds, block_size=int(args.block_size), offset_rounds=int(offset))
        trades_path = (out_dir / scenario_name / "oracle_boundary_trades.csv") if bool(args.write_trades) else None
        summary = _simulate_block(
            rounds_block=block,
            kidx=kidx,
            cutoff_seconds=int(args.cutoff_seconds),
            boundary_threshold=float(boundary_threshold),
            flow_window_seconds=int(args.flow_window_seconds),
            flow_min_imbalance=float(args.flow_min_imbalance),
            rule=str(args.rule),
            fixed_bet_bnb=float(args.fixed_bet_bnb),
            initial_bankroll_bnb=float(args.initial_bankroll_bnb),
            treasury_fee_fraction=float(args.treasury_fee_fraction),
            write_trades_path=trades_path,
        )
        row = {
            "scenario": str(scenario_name),
            "block_index": int(block_idx),
            "sim_offset_rounds": int(offset),
            "epoch_first": int(block[0].epoch),
            "epoch_last": int(block[-1].epoch),
            "net": float(summary["net_profit_bnb"]),
            "bets": int(summary["num_bets"]),
            "wins": int(summary["num_wins"]),
            "bet_rate": float(summary["bet_rate"]),
            "win_rate": float(summary["win_rate"]),
        }
        block_rows.append(row)
        nets.append(float(summary["net_profit_bnb"]))
        bets_total += int(summary["num_bets"])
        wins_total += int(summary["num_wins"])

        scenario_out = out_dir / scenario_name / "oracle_boundary_summary.json"
        scenario_out.parent.mkdir(parents=True, exist_ok=True)
        scenario_out.write_text(
            json.dumps(
                {
                    "scenario": {
                        "name": str(scenario_name),
                        "rule": str(args.rule),
                        "cutoff_seconds": int(args.cutoff_seconds),
                        "boundary_threshold_bps": float(args.boundary_threshold_bps),
                        "flow_window_seconds": int(args.flow_window_seconds),
                        "flow_min_imbalance": float(args.flow_min_imbalance),
                        "fixed_bet_bnb": float(args.fixed_bet_bnb),
                        "initial_bankroll_bnb": float(args.initial_bankroll_bnb),
                        "treasury_fee_fraction": float(args.treasury_fee_fraction),
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
        "rule": str(args.rule),
        "cutoff_seconds": int(args.cutoff_seconds),
        "boundary_threshold_bps": float(args.boundary_threshold_bps),
        "flow_window_seconds": int(args.flow_window_seconds),
        "flow_min_imbalance": float(args.flow_min_imbalance),
        "fixed_bet_bnb": float(args.fixed_bet_bnb),
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
                "epoch_first",
                "epoch_last",
                "scenario",
                "net",
                "bets",
                "wins",
                "bet_rate",
                "win_rate",
            ],
        )
        w.writeheader()
        for row in block_rows:
            w.writerow(row)

    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rule",
                "cutoff_seconds",
                "boundary_threshold_bps",
                "flow_window_seconds",
                "flow_min_imbalance",
                "fixed_bet_bnb",
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
