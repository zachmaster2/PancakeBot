#!/usr/bin/env python3
"""Parity v2: eth_getLogs (bloXroute) vs eth_getBlockReceipts (current 3
endpoints) for PancakeSwap Prediction V2 Bet events, over one pinned window.

v1 (parity_getlogs_vs_receipts) returned getLogs=0 because the 3 production
endpoints BLOCK eth_getLogs (-32005 "limit exceeded" / 403) and the script
swallowed the error. The endpoint survey found bsc.rpc.blxrbdn.com serves
getLogs with no practical range cap. This re-runs the parity with bloXroute on
the getLogs path, RAISING on any RPC error (no silent swallow), and asserts a
byte-identical bet-event set. Read-only; production mainnet.
  python parity_getlogs_v2_blxr_2026_06_09.py [n_blocks=2000]
"""
import json
import sys
import time
import urllib.request as R

RECEIPT_ENDPOINTS = ["https://bsc-dataseed1.binance.org", "https://bsc-dataseed1.defibit.io",
                     "https://bsc-rpc.publicnode.com"]
GETLOGS_ENDPOINT = "https://bsc.rpc.blxrbdn.com"
CONTRACT = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA".lower()
BET_BULL = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
BET_BEAR = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"
RECEIPT_BATCH = 20
GETLOGS_CHUNK = 1000


def post_hedged(payload, timeout=60):
    body = json.dumps(payload).encode()
    last = None
    for _ in range(4):
        for e in RECEIPT_ENDPOINTS:
            try:
                req = R.Request(e, data=body, headers={"Content-Type": "application/json"})
                with R.urlopen(req, timeout=timeout) as r:
                    raw = r.read()
                return json.loads(raw), len(raw)
            except Exception as ex:  # noqa: BLE001
                last = ex
        time.sleep(0.5)
    raise RuntimeError(f"receipt endpoints failed: {last}")


def post_getlogs(frm, to, timeout=30):
    """Single getLogs call to bloXroute. RAISES on any RPC error (no swallow)."""
    flt = {"fromBlock": hex(frm), "toBlock": hex(to), "address": CONTRACT,
           "topics": [[BET_BULL, BET_BEAR]]}
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs", "params": [flt]}).encode()
    req = R.Request(GETLOGS_ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    with R.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    p = json.loads(raw)
    if "error" in p:
        raise RuntimeError(f"getLogs error {frm}..{to}: {p['error']}")
    return p.get("result") or [], len(raw)


def extract_bet(log):
    if (log.get("address") or "").lower() != CONTRACT:
        return None
    topics = log.get("topics") or []
    if len(topics) < 3:
        return None
    t0 = topics[0]
    if t0 != BET_BULL and t0 != BET_BEAR:
        return None
    try:
        return (int(log["blockNumber"], 16), log["transactionHash"], int(log["logIndex"], 16),
                "Bull" if t0 == BET_BULL else "Bear", int(topics[2], 16),
                int(log.get("data", "0x0"), 16), topics[1])
    except (ValueError, KeyError, IndexError):
        return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    p, _ = post_hedged({"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []})
    h = int(p["result"], 16)
    lo, hi = h - n + 1, h
    print(f"window: blocks {lo}..{hi}  ({n} blocks)  head={h}", flush=True)

    # OLD path: per-block eth_getBlockReceipts (batched), client-side filter
    t0 = time.time()
    old, old_bytes = set(), 0
    bns = list(range(lo, hi + 1))
    for i in range(0, len(bns), RECEIPT_BATCH):
        chunk = bns[i:i + RECEIPT_BATCH]
        payload = [{"jsonrpc": "2.0", "id": j, "method": "eth_getBlockReceipts", "params": [hex(bn)]}
                   for j, bn in enumerate(chunk)]
        resp, nbytes = post_hedged(payload)
        old_bytes += nbytes
        byid = {r.get("id"): r for r in resp}
        for j in range(len(chunk)):
            r = byid.get(j, {})
            for rcpt in (r.get("result") or []):
                if not isinstance(rcpt, dict):
                    continue
                for log in (rcpt.get("logs") or []):
                    b = extract_bet(log)
                    if b:
                        old.add(b)
        if (i // RECEIPT_BATCH) % 20 == 0:
            print(f"  ...receipts {i+len(chunk)}/{n} blocks  ({old_bytes/1e6:.0f} MB)", flush=True)
    old_dt = time.time() - t0

    # NEW path: eth_getLogs range (chunked) on bloXroute, server-side filter
    t1 = time.time()
    new, new_bytes = set(), 0
    b = lo
    while b <= hi:
        to = min(b + GETLOGS_CHUNK - 1, hi)
        logs, nbytes = post_getlogs(b, to)
        new_bytes += nbytes
        for log in logs:
            bet = extract_bet(log)
            if bet:
                new.add(bet)
        b = to + 1
    new_dt = time.time() - t1

    # ---- parity ----
    print("\n=== PARITY ===")
    print(f"bet events: receipts={len(old)}  getLogs={len(new)}")
    only_old, only_new = old - new, new - old
    print(f"IDENTICAL: {old == new}")
    for label, s in (("in receipts but MISSING from getLogs", only_old),
                     ("in getLogs but NOT in receipts", only_new)):
        if s:
            print(f"  *** {len(s)} {label}: ***")
            for x in sorted(s)[:15]:
                print(f"     blk={x[0]} tx={x[1][:14]} logIdx={x[2]} {x[3]} ep={x[4]} amt={x[5]}")

    # ---- volume ----
    print(f"\n=== DATA VOLUME ({n}-block window) ===")
    print(f"  getBlockReceipts: {old_bytes/1e6:8.1f} MB   ({old_bytes/n/1024:6.0f} KB/block)  fetch={old_dt:.0f}s")
    print(f"  getLogs (blxr):   {new_bytes/1024:8.1f} KB   ({new_bytes/n:6.1f} bytes/block)  fetch={new_dt:.1f}s")
    print(f"  reduction: {old_bytes/max(1,new_bytes):.0f}x smaller")
    polls_day = 86400 / 8
    print("  extrapolated @ 18-block/8s polls:")
    print(f"    receipts: {old_bytes/n*18*polls_day/1e9:6.1f} GB/day")
    print(f"    getLogs:  {new_bytes/n*18*polls_day/1e6:6.1f} MB/day")


if __name__ == "__main__":
    main()
