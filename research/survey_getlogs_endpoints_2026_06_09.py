#!/usr/bin/env python3
"""Survey free public BSC RPC endpoints for eth_getLogs capability.

The 3 production endpoints (binance/defibit/publicnode) BLOCK eth_getLogs
(-32005 "limit exceeded" / 403). The getLogs migration needs a read endpoint
that actually serves the contract's Bet logs. This probes a broad set of
no-key public BSC RPCs and reports, per endpoint: liveness (eth_blockNumber),
then eth_getLogs over escalating ranges ending near head — OK/n-logs/bytes/ms,
the range cap, or the exact rejection. Read-only; production mainnet.
  python survey_getlogs_endpoints_2026_06_09.py
"""
import json
import time
import urllib.request as R

CONTRACT = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA".lower()
BET_BULL = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
BET_BEAR = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"

CANDIDATES = [
    "https://bsc-dataseed.bnbchain.org",
    "https://bsc-dataseed1.ninicoin.io",
    "https://bsc.publicnode.com",
    "https://1rpc.io/bnb",
    "https://bsc.drpc.org",
    "https://bsc.meowrpc.com",
    "https://binance.llamarpc.com",
    "https://bsc.blockpi.network/v1/rpc/public",
    "https://rpc.ankr.com/bsc",
    "https://bsc-mainnet.public.blastapi.io",
    "https://endpoints.omniatech.io/v1/bsc/mainnet/public",
    "https://bsc-pokt.nodies.app",
    "https://bsc.rpc.blxrbdn.com",
    "https://koge-rpc-bsc.48.club",
    "https://bscrpc.com",
    "https://bsc.4everland.org",
]


def rpc(endpoint, method, params, timeout=15):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = R.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    t = time.time()
    with R.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw), len(raw), int((time.time() - t) * 1000)


def head_from_binance():
    p, _, _ = rpc("https://bsc-dataseed1.binance.org", "eth_blockNumber", [])
    return int(p["result"], 16)


def getlogs(endpoint, frm, to):
    flt = {"fromBlock": hex(frm), "toBlock": hex(to), "address": CONTRACT,
           "topics": [[BET_BULL, BET_BEAR]]}
    try:
        p, nbytes, dt = rpc(endpoint, "eth_getLogs", [flt])
        if "error" in p:
            return f"ERR {str(p['error'].get('message', p['error']))[:44]} ({dt}ms)"
        return f"OK {len(p.get('result') or [])}logs {nbytes}b ({dt}ms)"
    except Exception as ex:  # noqa: BLE001
        return f"EXC {type(ex).__name__}:{str(ex)[:40]}"


anchor = head_from_binance() - 100   # settled, recent — representative of the poll window
print(f"anchor block (head-100) = {anchor}\n", flush=True)

for ep in CANDIDATES:
    try:
        p, _, dt = rpc(ep, "eth_blockNumber", [])
        lag = anchor + 100 - int(p["result"], 16)
        alive = f"alive bn={int(p['result'], 16)} (lag {lag}, {dt}ms)"
    except Exception as ex:  # noqa: BLE001
        print(f"=== {ep}\n    DEAD: {type(ex).__name__}:{str(ex)[:60]}", flush=True)
        continue
    print(f"=== {ep}\n    {alive}", flush=True)
    for span in (1, 20, 50, 200, 1000, 5000):
        print(f"      {span:>4}blk: {getlogs(ep, anchor - span + 1, anchor)}", flush=True)
