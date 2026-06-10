"""Measured broadcast->included-block inclusion latency (read-only, run ON VM).

Broadcast offset comes from cycle_audit: kline_fire_offset_before_lock_ms -
decision_latency_ms (validated within ~5ms against the LATE alerts'
submit_offset_ms, which is the bot's own broadcast-vs-lock measurement).

Per post-cache-fix bet (ffdd4be onward):
  bcast_off = kline_fire_offset - decision_latency   (ms before lock)
  margin    = included_block_BEP520_ms - lock_ts*1000 (<0 on-time, >=0 LATE)
  incl_lat  = margin + bcast_off = included_ms - broadcast_ms (broadcast->block)
"""
from __future__ import annotations

import csv
import json
import urllib.request as R

from pancakebot.chain.rpc_poller import READ_PATH_HEDGED_ENDPOINTS, compute_milli_ts

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


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


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

    rows = []
    print(f"{'epoch':>7} {'st':>4} {'bcast_off':>9} {'margin':>7} {'incl_lat':>8} {'slip?':>6}")
    for ep, (st, blk) in sorted(bets.items()):
        lt = lockts.get(ep)
        if lt is None or ep not in kfire:
            continue
        b = rpc("eth_getBlockByNumber", [hex(blk), False])
        if not b:
            continue
        ims = compute_milli_ts(b)
        margin = ims - lt * 1000
        bcoff = kfire[ep] - declat[ep]
        incl = margin + bcoff
        slip = "SLIP" if incl > 550 else ""
        rows.append((ep, st, bcoff, margin, incl))
        print(f"{ep:>7} {st[:4]:>4} {bcoff:>9.0f} {margin:>+7.0f} {incl:>8.0f} {slip:>6}")

    incl_all = [r[4] for r in rows]
    incl_ok = [r[4] for r in rows if r[1] == "CONFIRMED"]
    incl_late = [r[4] for r in rows if r[1] == "LATE"]

    print(f"\n=== inclusion latency: broadcast -> included block (ms), n={len(rows)} ===")
    for lbl, xs in (("ALL", incl_all), ("on-time", incl_ok), ("LATE", incl_late)):
        if xs:
            print(f"  {lbl:>8}: n={len(xs):2d}  p50={pct(xs,.5):.0f}  p90={pct(xs,.9):.0f}  "
                  f"p95={pct(xs,.95):.0f}  p99={pct(xs,.99):.0f}  min={min(xs):.0f} max={max(xs):.0f}")
    made = [x for x in incl_all if x <= 550]
    slipped = [x for x in incl_all if x > 550]
    print(f"\n  bimodal split: made-next-block (<=550ms): n={len(made)} mean={sum(made)/len(made):.0f}ms"
          f"  |  slipped-one-block (>550ms): n={len(slipped)} mean={sum(slipped)/len(slipped):.0f}ms"
          f"  |  gap ~= 1 block (450ms)")
    bc = [r[2] for r in rows]
    print(f"\n  broadcast offset (lock - broadcast): p50={pct(bc,.5):.0f} min={min(bc):.0f} max={max(bc):.0f}ms")
    print(f"  -> a bet lands LATE iff it SLIPS (incl ~760) AND broadcast_offset < incl (too late to absorb)")


if __name__ == "__main__":
    main()
