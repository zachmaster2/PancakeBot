"""Pin the BEP-520 sub-second block grid around the 4 VM-live epochs to settle
the on-time-vs-LATE paradox: did 487387 land in a different block position
relative to lock than the 3 LATE bets? Read-only chain queries.

Run on the VM:  /root/pancakebot/.venv/bin/python research/probe_block_grid_late_2026_06_06.py
"""
import json
import urllib.request

from pancakebot.chain.rpc_poller import compute_milli_ts, READ_PATH_HEDGED_ENDPOINTS

RPC = READ_PATH_HEDGED_ENDPOINTS[0]

# (epoch, included_block_number, lock_ts_seconds, verdict, broadcast_offset_ms_reconstructed)
CASES = [
    (487387, 102542963, 1780701590, "CONFIRMED", 563),
    (487395, 102548413, 1780704050, "LATE", 162),
    (487408, 102557283, 1780708046, "LATE", 217),
    (487410, 102558651, 1780708664, "LATE", 360),
]


def get_block(n):
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber",
        "params": [hex(n), False],
    }
    req = urllib.request.Request(
        RPC, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)["result"]


for epoch, inc, lock_ts, verdict, bcast_off in CASES:
    lock_ms = lock_ts * 1000
    print(f"\n===== epoch {epoch}  ({verdict})  lock_ts={lock_ts} (={lock_ms} ms)  "
          f"included_block={inc}  broadcast~lock-{bcast_off}ms =====")
    rows = []
    for n in range(inc - 4, inc + 3):
        b = get_block(n)
        sec = int(b["timestamp"], 16)
        milli = compute_milli_ts(b)  # sec*1000 + mixHash_ms
        rows.append((n, sec, milli))
    # lock block = first block whose SECOND timestamp >= lock_ts (contract granularity)
    lock_block = next((n for n, sec, _ in rows if sec >= lock_ts), None)
    for n, sec, milli in rows:
        tag = ""
        if n == inc:
            tag += " <== INCLUDED(our TX)"
        if n == lock_block:
            tag += " <== LOCK block (first sec>=lock_ts)"
        ms_part = milli - sec * 1000 if milli is not None else None
        off_from_lock = (milli - lock_ms) if milli is not None else None
        late_sec = "LATE-sec" if sec >= lock_ts else "ok-sec"
        print(f"  blk {n}  sec={sec}  ms={ms_part:>4}  milli_off_from_lock={off_from_lock:+6} "
              f"[{late_sec}]{tag}")
