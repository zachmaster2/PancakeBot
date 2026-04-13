"""Compare dry mode CSV pools (event/RPC-observed) vs backtest pools.

Run AFTER dry mode has run with the event watcher AND sync has fetched
closed round data for those epochs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import BNB_WEI
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore


def main():
    csv_path = "var/runtime/dry_cycle_audit.csv"
    rounds_path = "var/closed_rounds.jsonl"

    csv_pools = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["current_epoch"])
            bull = row.get("observed_bull_pool_bnb", "")
            bear = row.get("observed_bear_pool_bnb", "")
            cutoff = int(row.get("cutoff_ts", 0))

            if not bull or not bear or float(bull) == 0.0:
                continue

            csv_pools[epoch] = {
                "bull": float(bull),
                "bear": float(bear),
                "total": float(bull) + float(bear),
                "cutoff_ts": cutoff,
            }

    if not csv_pools:
        print("No non-zero pool data in CSV. Run dry mode with pool logging first.")
        return

    print(f"CSV epochs with pool data: {len(csv_pools)}")

    store = ClosedRoundsStore(rounds_path)
    all_rounds = {int(r.epoch): r for r in store.iter_closed_rounds()}

    matched = 0
    mismatched = 0

    print(f"\n{'Epoch':>8s}  {'CSV Bull':>10s}  {'BT Bull':>10s}  {'d_Bull':>8s}  "
          f"{'CSV Bear':>10s}  {'BT Bear':>10s}  {'d_Bear':>8s}  {'d_Total%':>8s}")
    print("-" * 90)

    for epoch in sorted(csv_pools.keys()):
        rnd = all_rounds.get(epoch)
        if rnd is None:
            print(f"  epoch {epoch}: NOT IN STORED ROUNDS (run sync first)")
            continue

        csv_data = csv_pools[epoch]
        cutoff_ts = csv_data["cutoff_ts"]

        bt_bull_wei = 0
        bt_bear_wei = 0
        for bet in rnd.bets:
            if int(bet.created_at) > cutoff_ts:
                continue
            if bet.position == "Bull":
                bt_bull_wei += int(bet.amount_wei)
            else:
                bt_bear_wei += int(bet.amount_wei)
        bt_bull = bt_bull_wei / 1e18
        bt_bear = bt_bear_wei / 1e18
        bt_total = bt_bull + bt_bear

        d_bull = csv_data["bull"] - bt_bull
        d_bear = csv_data["bear"] - bt_bear
        d_total = csv_data["total"] - bt_total
        d_pct = abs(d_total) / max(0.001, bt_total) * 100

        flag = "" if d_pct < 1.0 else " ***" if d_pct > 10.0 else " *"

        print(f"{epoch:>8d}  {csv_data['bull']:10.4f}  {bt_bull:10.4f}  {d_bull:+8.4f}  "
              f"{csv_data['bear']:10.4f}  {bt_bear:10.4f}  {d_bear:+8.4f}  {d_pct:7.1f}%{flag}")

        if d_pct < 1.0:
            matched += 1
        else:
            mismatched += 1

    print(f"\nMatched (<1% diff): {matched}")
    print(f"Mismatched (>=1%):  {mismatched}")
    if matched + mismatched > 0:
        print(f"Match rate: {matched/(matched+mismatched)*100:.0f}%")


if __name__ == "__main__":
    main()
