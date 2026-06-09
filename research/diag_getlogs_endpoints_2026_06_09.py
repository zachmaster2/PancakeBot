#!/usr/bin/env python3
"""Diagnose why eth_getLogs returned 0 in the parity check. Prints the RAW
response (error object or result count + bytes) per endpoint, for a single
known-bet block and escalating ranges, with and without the topic filter.

Pinpoints whether the 3 production endpoints reject/limit getLogs (and the
exact error), vs a filter-shape problem. Read-only; production mainnet.
  python diag_getlogs_endpoints_2026_06_09.py
"""
import json
import urllib.request as R

ENDPOINTS = ["https://bsc-dataseed1.binance.org", "https://bsc-dataseed1.defibit.io",
             "https://bsc-rpc.publicnode.com"]
CONTRACT = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA".lower()
BET_BULL = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
BET_BEAR = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"
KNOWN = 103154279  # parity run found a Bull bet here (ep488284)


def call(endpoint, frm, to, *, with_topics=True, with_addr=True, timeout=30):
    flt = {"fromBlock": hex(frm), "toBlock": hex(to)}
    if with_addr:
        flt["address"] = CONTRACT
    if with_topics:
        flt["topics"] = [[BET_BULL, BET_BEAR]]
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs",
                       "params": [flt]}).encode()
    req = R.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    try:
        with R.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        p = json.loads(raw)
        if "error" in p:
            return f"ERROR {json.dumps(p['error'])[:160]}"
        n = len(p.get("result") or [])
        return f"OK {n} logs ({len(raw)} bytes)"
    except Exception as ex:  # noqa: BLE001
        return f"EXC {type(ex).__name__}: {str(ex)[:120]}"


for ep in ENDPOINTS:
    print(f"\n=== {ep} ===", flush=True)
    print(f"  1blk[{KNOWN}] +addr +topics:  {call(ep, KNOWN, KNOWN)}", flush=True)
    print(f"  1blk[{KNOWN}] +addr -topics:  {call(ep, KNOWN, KNOWN, with_topics=False)}", flush=True)
    print(f"  1blk[{KNOWN}] -addr +topics:  {call(ep, KNOWN, KNOWN, with_addr=False)}", flush=True)
    for span in (5, 50, 200, 500, 1000, 2000):
        print(f"  {span:>4}blk +addr +topics:  {call(ep, KNOWN - span + 1, KNOWN)}", flush=True)
