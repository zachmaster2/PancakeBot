#!/usr/bin/env python3
"""Parity + volume check: eth_getLogs vs eth_getBlockReceipts for PancakeSwap
Prediction V2 Bet events. Proves the getLogs migration extracts a BYTE-IDENTICAL
bet-event set from the same block window, and measures the data-volume reduction.

Run on production mainnet (the real contract, all bettors). Regression-keepable.
  python parity_getlogs_vs_receipts_2026_06_09.py [n_blocks=2000]
"""
import json
import sys
import time
import urllib.request as R

ENDPOINTS = ["https://bsc-dataseed1.binance.org", "https://bsc-dataseed1.defibit.io",
             "https://bsc-rpc.publicnode.com"]
CONTRACT = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA".lower()
BET_BULL = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
BET_BEAR = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"
RECEIPT_BATCH = 20            # match the live bot's batch size (~6 MB/batch)
GETLOGS_CHUNK = 1000          # getLogs range per call


def post(payload, timeout=60):
    """Hedged POST with a small retry across endpoints. Returns (json, byte_len)."""
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(4):
        for e in ENDPOINTS:
            try:
                req = R.Request(e, data=body, headers={"Content-Type": "application/json"})
                with R.urlopen(req, timeout=timeout) as r:
                    raw = r.read()
                    return json.loads(raw), len(raw)
            except Exception as ex:  # noqa: BLE001
                last = ex
        time.sleep(0.5)
    raise RuntimeError(f"all endpoints failed after retries: {last}")


def head():
    p, _ = post({"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []})
    return int(p["result"], 16)


def extract_bet(log):
    """Mirror the bot's _process_receipts_for_block extraction exactly."""
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
    h = head()
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
        resp, nbytes = post(payload)
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

    # NEW path: eth_getLogs range (chunked), server-side filter
    t1 = time.time()
    new, new_bytes = set(), 0
    b = lo
    while b <= hi:
        to = min(b + GETLOGS_CHUNK - 1, hi)
        flt = {"fromBlock": hex(b), "toBlock": hex(to), "address": CONTRACT,
               "topics": [[BET_BULL, BET_BEAR]]}
        resp, nbytes = post({"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs", "params": [flt]})
        new_bytes += nbytes
        for log in (resp.get("result") or []):
            bet = extract_bet(log)
            if bet:
                new.add(bet)
        b = to + 1
    new_dt = time.time() - t1

    # ---- parity ----
    print(f"\n=== PARITY ===")
    print(f"bet events: receipts={len(old)}  getLogs={len(new)}")
    only_old, only_new = old - new, new - old
    print(f"IDENTICAL: {old == new}")
    if only_old:
        print(f"  *** {len(only_old)} in receipts but MISSING from getLogs: ***")
        for x in sorted(only_old)[:15]:
            print(f"     blk={x[0]} tx={x[1][:14]} logIdx={x[2]} {x[3]} ep={x[4]} amt={x[5]}")
    if only_new:
        print(f"  *** {len(only_new)} in getLogs but NOT in receipts: ***")
        for x in sorted(only_new)[:15]:
            print(f"     blk={x[0]} tx={x[1][:14]} logIdx={x[2]} {x[3]} ep={x[4]} amt={x[5]}")

    # ---- volume ----
    print(f"\n=== DATA VOLUME ({n}-block window) ===")
    print(f"  getBlockReceipts: {old_bytes/1e6:8.1f} MB   ({old_bytes/n/1024:6.0f} KB/block)  fetch={old_dt:.0f}s")
    print(f"  getLogs:          {new_bytes/1024:8.1f} KB   ({new_bytes/n:6.1f} bytes/block)  fetch={new_dt:.0f}s")
    print(f"  reduction: {old_bytes/max(1,new_bytes):.0f}x smaller")
    polls_day = 86400 / 8
    rcpt_day = old_bytes / n * 18 * polls_day
    logs_day = new_bytes / n * 18 * polls_day
    print(f"  extrapolated @ 18-block/8s polls:")
    print(f"    receipts: {rcpt_day/1e9:6.1f} GB/day")
    print(f"    getLogs:  {logs_day/1e6:6.1f} MB/day")


if __name__ == "__main__":
    main()
