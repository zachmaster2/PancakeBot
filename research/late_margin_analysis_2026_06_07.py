"""Retrospective LATE margin analysis (read-only; run ON the VM).

For each post-cache-fix bet (ffdd4be onward, 2026-06-06T07:49+):
  margin = included_block_BEP520_ms - lock_ts*1000
  margin < 0  => block produced BEFORE the lock second  => on-time
  margin >= 0 => block produced AT/AFTER the lock second => LATE (contract needs
                 block.timestamp < lockTimestamp, whole-second granularity)

Buffer model: shifting the submit deadline Δ ms earlier shifts the whole bet
timeline (broadcast -> included block) ~Δ earlier, so margin -> margin - Δ.
Resulting LATE rate after a Δ buffer = P(margin >= Δ). The buffer curve makes
the tradeoff concrete; the OKX-publish floor (fetch can't fire much before
~lock-850ms without kline-publish failures) bounds how large Δ can usefully be.

Run: cd /root/pancakebot && PYTHONPATH=/root/pancakebot ./.venv/bin/python research/late_margin_analysis_2026_06_07.py
"""
from __future__ import annotations

import csv
import json
import urllib.request as R

from pancakebot.chain.rpc_poller import READ_PATH_HEDGED_ENDPOINTS, compute_milli_ts

FFDD = "2026-06-06T07:49"  # pre-cache-fix deploy boundary
BETS = "/root/pancakebot/var/live/bets.jsonl"
AUDIT = "/root/pancakebot/var/live/cycle_audit.csv"


def rpc(method, params):
    for e in READ_PATH_HEDGED_ENDPOINTS:
        try:
            req = R.Request(
                e, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                                    "params": params}).encode(),
                headers={"Content-Type": "application/json"})
            with R.urlopen(req, timeout=5) as r:
                return json.load(r).get("result")
        except Exception:  # noqa: BLE001
            continue
    return None


def main():
    # lock_ts per epoch from cycle_audit. NOTE: match by current_epoch, NOT
    # locked_epoch — in a given cycle, lock_ts is the lock of the round being
    # bet/about-to-lock (current_epoch); locked_epoch is the prior, already-
    # locked round, whose lock_ts would be ~1 round (300s) too late.
    lockts = {}
    with open(AUDIT) as f:
        for row in csv.DictReader(f):
            try:
                lockts[int(row["current_epoch"])] = int(float(row["lock_ts"]))
            except Exception:  # noqa: BLE001
                pass

    # post-fix bets with an included block
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

    rows = []
    for ep, (st, blk) in sorted(bets.items()):
        lt = lockts.get(ep)
        if lt is None:
            continue
        b = rpc("eth_getBlockByNumber", [hex(blk), False])
        if not b:
            continue
        bms = compute_milli_ts(b)
        rows.append((ep, st, bms - lt * 1000))

    LATE = "LATE"
    # Q1-3
    since = [r for r in rows if r[0] > 487722]
    nlate_all = sum(1 for r in rows if r[1] == LATE)
    print("=== Q1-3 ===")
    print(f"bets since 487722: {len(since)}  LATE among them: {sum(1 for r in since if r[1]==LATE)}")
    print(f"post-fix total: {len(rows)}  LATE: {nlate_all}  "
          f"rate: {100*nlate_all/len(rows):.0f}%  (was 2/12=17%)")

    print("\n=== per-bet margin (ms; <0 on-time, >=0 LATE) ===")
    for ep, st, m in rows:
        flag = "  <-- LATE" if st == LATE else ""
        print(f"  {ep}  {st:9s}  margin={m:+5d}ms{flag}")

    ms = sorted(m for _, _, m in rows)
    n = len(ms)

    def pct(q):
        return ms[min(n - 1, int(n * q))]

    print(f"\n=== margin distribution (all post-fix, n={n}) ===")
    for q in (.5, .75, .9, .95, .99):
        print(f"  p{int(q*100):2d}: {pct(q):+5d}ms")
    print(f"  min={min(ms):+d}  max={max(ms):+d}")
    late_ms = sorted(m for _, st, m in rows if st == LATE)
    if late_ms:
        print(f"  LATE-only margins: {late_ms}")

    print("\n=== buffer curve: shift submit Δ earlier -> LATE rate = P(margin >= Δ) ===")
    for d in (0, 25, 50, 75, 100, 150, 200, 300, 450):
        late = sum(1 for m in ms if m >= d)
        print(f"  Δ={d:4d}ms  ->  LATE {late}/{n} = {100*late/n:4.0f}%   "
              f"(fetch would fire ~lock-{776+d}ms)")


if __name__ == "__main__":
    main()
