"""Local hot-window scan: max-PnL contiguous time windows of fixed lengths
(7d, 14d, 30d) at both 5 BNB and 50 BNB scales.

Uses cached trades.csv from Step 11 bonus runs. No backtest re-run needed —
just joins per-bet (epoch, profit) with round.start_at for timestamps and
slides a fixed-length window across the bet sequence.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.in_process_runner as ipr

BONUS_TRADES_5BNB = Path(r"C:\Users\zking\AppData\Local\Temp\step11_k3ub14zf\bonus_5bnb\trades.csv")
BONUS_TRADES_50BNB = Path(r"C:\Users\zking\AppData\Local\Temp\step11_k3ub14zf\bonus_50bnb\trades.csv")

WINDOW_LENGTHS_DAYS = (7, 14, 30)


def load_bets_with_timestamps(trades_csv: Path, epoch_to_ts: dict[int, int]) -> list[dict]:
    bets = []
    with open(trades_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("action") != "BET":
                continue
            epoch = int(row["epoch"])
            ts = epoch_to_ts.get(epoch)
            if ts is None:
                continue
            bets.append({
                "epoch": epoch, "ts": ts,
                "profit": float(row["profit_bnb"]),
                "win": float(row["profit_bnb"]) > 0,
            })
    bets.sort(key=lambda b: b["ts"])
    return bets


def find_best_window(bets: list[dict], window_seconds: int) -> dict:
    """Sliding window: for each starting bet i, find max j s.t.
    bets[j].ts - bets[i].ts <= window_seconds. Track the sum-profit
    over [i..j]; return the global max.
    """
    n = len(bets)
    if n == 0:
        return {"start_idx": None}
    best_sum = float("-inf")
    best_i = 0
    best_j = 0
    j = 0
    cur_sum = 0.0
    for i in range(n):
        # Extend j as long as we're within the window
        while j < n and bets[j]["ts"] - bets[i]["ts"] <= window_seconds:
            cur_sum += bets[j]["profit"]
            j += 1
        # Window is [i..j-1]
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_i = i
            best_j = j - 1
        # Remove bets[i] before advancing i
        cur_sum -= bets[i]["profit"]
    window = bets[best_i:best_j + 1]
    n_bets = len(window)
    n_wins = sum(1 for b in window if b["win"])
    return {
        "start_idx": best_i, "end_idx": best_j,
        "start_epoch": window[0]["epoch"], "end_epoch": window[-1]["epoch"],
        "start_ts": window[0]["ts"], "end_ts": window[-1]["ts"],
        "n_bets": n_bets, "n_wins": n_wins,
        "win_rate": n_wins / n_bets if n_bets else 0.0,
        "pnl_bnb": best_sum,
        "actual_span_days": (window[-1]["ts"] - window[0]["ts"]) / 86400.0,
    }


def fmt_dt(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    print("--- loading rounds for epoch -> start_at lookup ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    epoch_to_ts = {int(r.epoch): int(r.start_at) for r in all_rounds}
    print(f"  {len(epoch_to_ts)} rounds loaded", flush=True)

    print("\n--- loading bet timelines from cached trades.csv ---", flush=True)
    bets_5 = load_bets_with_timestamps(BONUS_TRADES_5BNB, epoch_to_ts)
    bets_50 = load_bets_with_timestamps(BONUS_TRADES_50BNB, epoch_to_ts)
    print(f"  5 BNB:  {len(bets_5)} bets", flush=True)
    print(f"  50 BNB: {len(bets_50)} bets", flush=True)

    results = {}
    for length_days in WINDOW_LENGTHS_DAYS:
        win_s = length_days * 86400
        for scale, bets in (("5bnb", bets_5), ("50bnb", bets_50)):
            r = find_best_window(bets, win_s)
            r["scale"] = scale
            r["length_days"] = length_days
            results[(scale, length_days)] = r

    # Print table
    print("\n=== HOT-WINDOW TABLE ===")
    print(f"{'Length':>8s} {'Scale':>6s}  {'Start (UTC)':>17s} {'End (UTC)':>17s} "
          f"{'Span':>5s}  {'Bets':>5s} {'WR':>7s} {'PnL':>10s}")
    for length_days in WINDOW_LENGTHS_DAYS:
        for scale in ("5bnb", "50bnb"):
            r = results[(scale, length_days)]
            print(f"{length_days:>5d}d   {scale:>6s}  "
                  f"{fmt_dt(r['start_ts']):>17s} {fmt_dt(r['end_ts']):>17s} "
                  f"{r['actual_span_days']:>5.2f}  "
                  f"{r['n_bets']:>5d} {r['win_rate']:>7.4f} "
                  f"{r['pnl_bnb']:>+10.4f}")


if __name__ == "__main__":
    main()
