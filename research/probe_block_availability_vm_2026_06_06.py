"""Re-baseline RPC_BLOCK_AVAILABILITY_DELAY_P99_MS on the VM (read-only, gentle).

Measures, per production READ endpoint, the delay from a block's BEP-520
production time (mixHash ms) to the first successful eth_getBlockReceipts for
it — i.e. how long after a block is produced its receipts are fetchable. Polls
a FUTURE block (head+1) from before it exists, so the measurement excludes
client-side block-detection lag.

GENTLE: 100ms between poll attempts (the betting bot shares these endpoints; a
tight loop trips their rate-limiter -> 403). 403/HTTP errors are counted and
the sample skipped, never spun on. Hard attempt cap guarantees termination.

Run on VM:  cd /root/pancakebot && PYTHONPATH=/root/pancakebot \
    ./.venv/bin/python research/probe_block_availability_vm_2026_06_06.py --n 25
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

from pancakebot.chain.rpc_poller import READ_PATH_HEDGED_ENDPOINTS, compute_milli_ts


def _rpc(url, method, params, timeout=5):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r).get("result")


def _pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    eps = list(READ_PATH_HEDGED_ENDPOINTS)

    # Phase 0: which endpoints even serve eth_getBlockReceipts (single call each).
    serving = []
    for e in eps:
        try:
            h = int(_rpc(e, "eth_blockNumber", []), 16)
            rc = _rpc(e, "eth_getBlockReceipts", [hex(h - 2)])
            ok = isinstance(rc, list)
            print(f"endpoint {e}: getBlockReceipts {'OK' if ok else 'NON-LIST'} "
                  f"(n={len(rc) if ok else rc})", flush=True)
            if ok:
                serving.append(e)
        except Exception as ex:
            print(f"endpoint {e}: FAIL {type(ex).__name__} {str(ex)[:70]}", flush=True)
    if not serving:
        print("no serving endpoints — abort"); return

    per = {e: [] for e in serving}
    err403 = {e: 0 for e in serving}
    pooled = []
    got = 0
    attempts = 0
    cap = args.n * 4
    while got < args.n and attempts < cap:
        attempts += 1
        ep = serving[attempts % len(serving)]
        try:
            head = int(_rpc(serving[0], "eth_blockNumber", []), 16)
        except Exception:
            continue
        target = head + 1
        deadline = time.time() + 3.0
        t_success = None
        broke_on_err = False
        while time.time() < deadline:
            try:
                rc = _rpc(ep, "eth_getBlockReceipts", [hex(target)])
                if rc is not None:
                    t_success = time.time() * 1000.0
                    break
            except urllib.error.HTTPError as he:
                if he.code == 403:
                    err403[ep] += 1
                broke_on_err = True
                break
            except Exception:
                broke_on_err = True
                break
            time.sleep(0.1)
        if t_success is None:
            if broke_on_err:
                continue
            continue
        try:
            blk = _rpc(serving[0], "eth_getBlockByNumber", [hex(target), False])
            milli = compute_milli_ts(blk)
        except Exception:
            milli = None
        if milli is None:
            continue
        avail = t_success - milli
        if avail < 0 or avail > 5000:
            continue
        per[ep].append(avail)
        pooled.append(avail)
        got += 1

    print(f"\ncurrent RPC_BLOCK_AVAILABILITY_DELAY_P99_MS = 600 (home/drpc, 2026-05-07)")
    print(f"attempts={attempts} samples={got} 403s={err403}")
    for ep in serving:
        xs = per[ep]
        if not xs:
            print(f"  {ep}: n=0"); continue
        print(f"  {ep}: n={len(xs)} p50={_pct(xs,.5):.0f} p95={_pct(xs,.95):.0f} "
              f"p99={_pct(xs,.99):.0f} max={max(xs):.0f}")
    if pooled:
        print(f"  POOLED: n={len(pooled)} p50={_pct(pooled,.5):.0f} "
              f"p95={_pct(pooled,.95):.0f} p99={_pct(pooled,.99):.0f} max={max(pooled):.0f}")


if __name__ == "__main__":
    main()
