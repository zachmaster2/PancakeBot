from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path


def _parse_allowlist(raw: str) -> set[str]:
    text = str(raw).strip()
    if text == "":
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle-audit-csv", type=str, default="var/runtime/dry_cycle_audit.csv")
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument(
        "--expected-strategies",
        type=str,
        default="disloc_stageB_bullonly_recent8pct_v1,flow_lgbm_recent_t12k_r1k_regime40_v1",
    )
    parser.add_argument("--expected-bet-sides", type=str, default="Bull,Bear")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--duration-seconds", type=int, default=43_200)
    parser.add_argument("--warn-idle-streak-cycles", type=int, default=240)
    parser.add_argument("--warn-min-cycles-for-rate-check", type=int, default=240)
    parser.add_argument("--warn-total-bet-rate-below", type=float, default=0.02)
    return parser


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _idle_streak(rows: list[dict[str, str]]) -> int:
    streak = 0
    for row in reversed(rows):
        if str(row.get("action", "")).strip() == "BET":
            break
        streak += 1
    return int(streak)


def _summarize(
    *,
    rows: list[dict[str, str]],
    expected_strategies: set[str],
    expected_bet_sides: set[str],
    warn_idle_streak_cycles: int,
    warn_min_cycles_for_rate_check: int,
    warn_total_bet_rate_below: float,
) -> dict[str, object]:
    total_cycles = int(len(rows))
    bet_rows = [row for row in rows if str(row.get("action", "")).strip() == "BET"]
    total_bets = int(len(bet_rows))
    total_bet_rate = 0.0 if total_cycles <= 0 else float(total_bets) / float(total_cycles)
    recent_60 = rows[-60:]
    recent_120 = rows[-120:]
    recent_240 = rows[-240:]

    def _bet_rate(window_rows: list[dict[str, str]]) -> float:
        if not window_rows:
            return 0.0
        bets = sum(1 for row in window_rows if str(row.get("action", "")).strip() == "BET")
        return float(bets) / float(len(window_rows))

    skip_counts = Counter(
        str(row.get("skip_reason", "")).strip()
        for row in rows
        if str(row.get("skip_reason", "")).strip()
    )
    strategy_counts = Counter(
        str(row.get("selected_strategy", "")).strip()
        for row in bet_rows
        if str(row.get("selected_strategy", "")).strip()
    )
    side_counts = Counter(
        str(row.get("bet_side", "")).strip()
        for row in bet_rows
        if str(row.get("bet_side", "")).strip()
    )
    recent_expected_profit = [
        value
        for value in (
            _safe_float(row.get("expected_profit_bnb"))
            for row in rows[-20:]
        )
        if value is not None
    ]

    anomalies: list[str] = []
    unexpected_strategies = (
        sorted(name for name in strategy_counts if str(name) not in expected_strategies)
        if expected_strategies
        else []
    )
    if expected_strategies and unexpected_strategies:
        anomalies.append(f"unexpected_selected_strategies={unexpected_strategies}")
    unexpected_sides = (
        sorted(name for name in side_counts if str(name) not in expected_bet_sides)
        if expected_bet_sides
        else []
    )
    if expected_bet_sides and unexpected_sides:
        anomalies.append(f"unexpected_bet_sides={unexpected_sides}")
    current_idle_streak = _idle_streak(rows)
    if (
        int(total_cycles) >= int(warn_min_cycles_for_rate_check)
        and float(total_bet_rate) < float(warn_total_bet_rate_below)
    ):
        anomalies.append(
            "total_bet_rate_below_threshold:"
            f"bet_rate={float(total_bet_rate):.6f}"
            f":threshold={float(warn_total_bet_rate_below):.6f}"
        )
    if int(current_idle_streak) >= int(warn_idle_streak_cycles):
        anomalies.append(
            f"idle_streak_ge_threshold:{int(current_idle_streak)}"
            f":threshold={int(warn_idle_streak_cycles)}"
        )

    return {
        "monitor_ts": int(time.time()),
        "total_cycles": int(total_cycles),
        "total_bets": int(total_bets),
        "total_bet_rate": float(total_bet_rate),
        "recent_60_bet_rate": float(_bet_rate(recent_60)),
        "recent_120_bet_rate": float(_bet_rate(recent_120)),
        "recent_240_bet_rate": float(_bet_rate(recent_240)),
        "current_idle_streak_cycles": int(current_idle_streak),
        "top_skip_reasons": dict(skip_counts.most_common(5)),
        "selected_strategy_counts": dict(strategy_counts),
        "bet_side_counts": dict(side_counts),
        "recent_mean_expected_profit_bnb": (
            0.0
            if not recent_expected_profit
            else float(sum(recent_expected_profit) / float(len(recent_expected_profit)))
        ),
        "anomalies": list(anomalies),
    }


def main() -> None:
    args = _build_parser().parse_args()
    cycle_audit_csv = Path(str(args.cycle_audit_csv))
    output_jsonl = Path(str(args.output_jsonl))
    summary_json = Path(str(args.summary_json))
    expected_strategies = _parse_allowlist(str(args.expected_strategies))
    expected_bet_sides = _parse_allowlist(str(args.expected_bet_sides))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    start_ts = int(time.time())
    deadline_ts = int(start_ts) + int(args.duration_seconds)
    last_signature: tuple[int, int | None] | None = None

    while int(time.time()) < int(deadline_ts):
        rows = _load_rows(cycle_audit_csv)
        latest_epoch = None
        if rows:
            latest_epoch_raw = str(rows[-1].get("current_epoch", "")).strip()
            latest_epoch = int(latest_epoch_raw) if latest_epoch_raw else None
        signature = (int(len(rows)), latest_epoch)
        if signature != last_signature:
            summary = _summarize(
                rows=rows,
                expected_strategies=set(expected_strategies),
                expected_bet_sides=set(expected_bet_sides),
                warn_idle_streak_cycles=int(args.warn_idle_streak_cycles),
                warn_min_cycles_for_rate_check=int(args.warn_min_cycles_for_rate_check),
                warn_total_bet_rate_below=float(args.warn_total_bet_rate_below),
            )
            with output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary, sort_keys=True))
                f.write("\n")
            summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            last_signature = signature
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
