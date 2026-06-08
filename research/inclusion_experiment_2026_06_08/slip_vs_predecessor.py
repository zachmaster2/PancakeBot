"""Slip vs broadcast-offset RELATIVE TO PREDECESSOR SEAL (deadline-normalized).

Zach's point: slip is driven by broadcast time vs the predecessor block's seal,
not vs lock. Lock-relative offset doesn't normalize for the predecessor's phase
(it varies round to round across the 9 positions). The dynamic deadline IS
anchored to the predecessor, so (deadline_ms - broadcast_ms) is the right
normalization.

For each post-fix bet: reconstruct the ACTUAL predecessor block (block before the
lock block) from chain, then:
  pred_off  = lock_ms - predecessor_ms              (predecessor phase before lock)
  bcast_off = kline_fire_offset - decision_latency  (broadcast before lock)
  slack     = predecessor_ms - broadcast_ms = bcast_off - pred_off  (lead before seal)
  off_vs_dl = compute_submit_deadline_ms(pred, lock) - broadcast_ms (SSOT-normalized)
  slip      = incl_lat > 550
Test: do slipped bets broadcast CLOSER to the deadline (smaller off_vs_dl / slack)?
"""
from __future__ import annotations

import csv
import json
import urllib.request as R

from pancakebot.chain.rpc_poller import (
    READ_PATH_HEDGED_ENDPOINTS, compute_milli_ts, compute_submit_deadline_ms,
)

FFDD = "2026-06-06T07:49"
AUDIT = "/root/pancakebot/var/live/cycle_audit.csv"
BETS = "/root/pancakebot/var/live/bets.jsonl"


def rpc(method, params):
    for e in READ_PATH_HEDGED_ENDPOINTS:
        try:
            req = R.Request(e, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                            "params": params}).encode(), headers={"Content-Type": "application/json"})
            with R.urlopen(req, timeout=5) as r:
                return json.load(r).get("result")
        except Exception:  # noqa: BLE001
            continue
    return None


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    lockts, kfire, declat = {}, {}, {}
    for row in csv.DictReader(open(AUDIT)):
        try:
            ce = int(row["current_epoch"])
        except Exception:  # noqa: BLE001
            continue
        try:
            lockts[ce] = int(float(row["lock_ts"]))
        except Exception:  # noqa: BLE001
            pass
        if row.get("action") == "BET":
            try:
                kfire[ce] = float(row["kline_fire_offset_before_lock_ms"])
                declat[ce] = float(row["decision_latency_ms"])
            except Exception:  # noqa: BLE001
                pass

    bets = {}
    for line in open(BETS):
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if d.get("ts", "") < FFDD:
            continue
        if d.get("status") in ("CONFIRMED", "LATE") and d.get("included_block_number"):
            bets[d["epoch"]] = (d["status"], int(d["included_block_number"]))

    bcache = {}

    def bms(bn):
        if bn not in bcache:
            b = rpc("eth_getBlockByNumber", [hex(bn), False])
            bcache[bn] = compute_milli_ts(b) if b else None
        return bcache[bn]

    rows = []
    print(f"{'epoch':>7} {'st':>4} {'bcast_off':>9} {'pred_off':>8} {'slack':>6} "
          f"{'off_vs_dl':>9} {'incl':>5} {'slip':>5}")
    for ep, (st, blk) in sorted(bets.items()):
        lt = lockts.get(ep)
        if lt is None or ep not in kfire:
            continue
        lock_ms = lt * 1000
        bcast_off = kfire[ep] - declat[ep]
        bcast_ms = lock_ms - bcast_off
        lock_bn = None                               # first block with ms >= lock_ms
        for bn in range(blk - 4, blk + 4):
            m = bms(bn)
            if m is not None and m >= lock_ms:
                lock_bn = bn
                break
        if lock_bn is None:
            continue
        pred_ms = bms(lock_bn - 1)
        incl_ms = bms(blk)
        if pred_ms is None or incl_ms is None:
            continue
        incl_lat = (incl_ms - lock_ms) + bcast_off
        slip = incl_lat > 550
        pred_off = lock_ms - pred_ms
        slack = pred_ms - bcast_ms                   # = bcast_off - pred_off
        deadline_ms = compute_submit_deadline_ms(
            predicted_predecessor_milli_ts=pred_ms, lock_ms=lock_ms)
        off_vs_dl = deadline_ms - bcast_ms
        rows.append((ep, st, bcast_off, pred_off, slack, off_vs_dl, incl_lat, slip))
        print(f"{ep:>7} {st[:4]:>4} {bcast_off:>9.0f} {pred_off:>8.0f} {slack:>6.0f} "
              f"{off_vs_dl:>9.0f} {incl_lat:>5.0f} {'SLIP' if slip else '':>5}")

    sl = [r for r in rows if r[7]]
    ns = [r for r in rows if not r[7]]
    print(f"\nn={len(rows)}  slip={len(sl)}  on-time={len(ns)}")
    print(f"  bcast_off (lock-relative):   slip mean {mean([r[2] for r in sl]):.0f}  "
          f"on-time {mean([r[2] for r in ns]):.0f}   <- the old (un-normalized) view")
    print(f"  pred_off (predecessor phase): slip mean {mean([r[3] for r in sl]):.0f}  "
          f"on-time {mean([r[3] for r in ns]):.0f}")
    print(f"  slack (predecessor - bcast): slip mean {mean([r[4] for r in sl]):.0f}  "
          f"on-time {mean([r[4] for r in ns]):.0f}   <- PREDECESSOR-normalized")
    print(f"  off_vs_deadline (SSOT):      slip mean {mean([r[5] for r in sl]):.0f}  "
          f"on-time {mean([r[5] for r in ns]):.0f}   <- DEADLINE-normalized")
    print("\n  slip rate by off_vs_deadline bin:")
    for lo, hi in [(-1e9, 0), (0, 100), (100, 200), (200, 300), (300, 400), (400, 1e9)]:
        b = [r for r in rows if lo <= r[5] < hi]
        if b:
            ns_ = sum(1 for r in b if r[7])
            lbl = f"[{lo:.0f},{hi:.0f})" if lo > -1e8 else f"(<0, past dl)"
            print(f"    {lbl:>14}: n={len(b):2d} slip={ns_} ({100 * ns_ / len(b):.0f}%)")
    print("\n  predecessor phases hit (pred_off, the 9 positions):")
    from collections import Counter
    c = Counter(int(round(r[3] / 50) * 50) for r in rows)
    for k in sorted(c):
        print(f"    lock-{k:>3}: {c[k]}")


if __name__ == "__main__":
    main()
