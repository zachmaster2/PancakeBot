#!/usr/bin/env python3
"""Paired getLogs host comparison + capability probe. READ-ONLY.

Runs in its own process; touches no bot state, no config, no systemd.
Replicates production's filter exactly (rpc_poller._bet_events_for_range):
    {fromBlock, toBlock, address=<prediction v2>, topics=[[BULL, BEAR]]}

Design notes that matter:
  * VARIED QUERY: each sample targets a FRESH recent block (head - LAG),
    never repeated, so provider caches cannot be measured instead of the
    provider.
  * PAIRED + INTERLEAVED: both hosts get the identical range back-to-back,
    with the order coin-flipped per sample.
  * GENEROUS TIMEOUT (not production's 250ms): censoring at the production
    bound would hide the tail we are trying to see. We record true latency
    and separately count how many samples WOULD have breached 250ms.
  * PARITY: log count + sorted tx-hash set compared per sample.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
import urllib.request

CONTRACT = "0x18b2a687610328590bc8f2e5fedde3b582a49cda"
BULL = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
BEAR = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"

PROD_TIMEOUT_MS = 250          # what production allows; used for censoring stats
BENCH_TIMEOUT_S = 5.0          # generous, to measure the real tail
HEAD_LAG = 5                   # blocks behind head, so laggy hosts still have it

PRIMARY = "https://bsc.rpc.blxrbdn.com"          # bloXroute (head/anchor today)
SECONDARY = "https://bsc-rpc.publicnode.com"     # publicnode (getLogs today)

CANDIDATES = [
    ("bloXroute",        "https://bsc.rpc.blxrbdn.com"),
    ("publicnode",       "https://bsc-rpc.publicnode.com"),
    ("bnbchain-dataseed", "https://bsc-dataseed.bnbchain.org"),
    ("defibit1",         "https://bsc-dataseed1.defibit.io"),
    ("defibit2",         "https://bsc-dataseed2.defibit.io"),
    ("ninicoin1",        "https://bsc-dataseed1.ninicoin.io"),
    ("ankr",             "https://rpc.ankr.com/bsc"),
    ("1rpc",             "https://1rpc.io/bnb"),
    ("drpc",             "https://bsc.drpc.org"),
    ("llama",            "https://binance.llamarpc.com"),
    ("blast",            "https://bsc-mainnet.public.blastapi.io"),
    ("meowrpc",          "https://bsc.meowrpc.com"),
    ("nodereal-public",  "https://bsc-mainnet.nodereal.io/v1/64a9df0874fb4a93b9d0a3849de012d3"),
    ("omnia",            "https://endpoints.omniatech.io/v1/bsc/mainnet/public"),
]


def rpc(url: str, method: str, params: list, *, timeout: float):
    body = json.dumps({"jsonrpc": "2.0", "id": 1,
                       "method": method, "params": params}).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "pancakebot-bench/1.0"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    ms = (time.perf_counter() - t0) * 1000.0
    obj = json.loads(raw)
    if isinstance(obj, dict) and obj.get("error"):
        raise RuntimeError("rpc_error:%s" % json.dumps(obj["error"])[:120])
    return obj.get("result"), ms


def logs_filter(block: int) -> dict:
    return {"fromBlock": hex(block), "toBlock": hex(block),
            "address": CONTRACT, "topics": [[BULL, BEAR]]}


def fingerprint(logs) -> tuple:
    if logs is None:
        return (-1, ())
    return (len(logs),
            tuple(sorted((l.get("transactionHash", ""), l.get("logIndex", ""))
                         for l in logs)))


def head(url: str) -> int:
    res, _ = rpc(url, "eth_blockNumber", [], timeout=BENCH_TIMEOUT_S)
    return int(res, 16)


# ---------------------------------------------------------------- probe ----
def capability_probe() -> dict:
    print("=" * 74)
    print("CAPABILITY PROBE — can each host serve a filtered eth_getLogs?")
    print("=" * 74)
    try:
        block = head(SECONDARY) - HEAD_LAG
    except Exception:
        block = head(PRIMARY) - HEAD_LAG
    print("probe block: %d (head - %d)\n" % (block, HEAD_LAG))

    ref = None
    results = {}
    for name, url in CANDIDATES:
        entry = {"url": url}
        try:
            logs, ms = rpc(url, "eth_getLogs", [logs_filter(block)],
                           timeout=BENCH_TIMEOUT_S)
            fp = fingerprint(logs)
            entry.update(ok=True, ms=round(ms, 1), n_logs=fp[0], fp=fp)
            if ref is None and name in ("bloXroute", "publicnode"):
                ref = fp
        except Exception as e:
            entry.update(ok=False, error="%s: %s" % (type(e).__name__, str(e)[:90]))
        results[name] = entry

    for name, e in results.items():
        if not e["ok"]:
            print("  %-18s FAIL   %s" % (name, e["error"]))
        else:
            match = "" if ref is None else (
                " parity=OK" if e["fp"] == ref else " parity=MISMATCH")
            print("  %-18s PASS   %6.1fms  logs=%d%s"
                  % (name, e["ms"], e["n_logs"], match))
    passers = [n for n, e in results.items() if e["ok"]]
    matching = [n for n, e in results.items()
                if e["ok"] and (ref is None or e["fp"] == ref)]
    print("\n  PASS (served getLogs): %d -> %s" % (len(passers), passers))
    print("  PASS AND byte-parity : %d -> %s" % (len(matching), matching))
    return {n: {k: v for k, v in e.items() if k != "fp"}
            for n, e in results.items()}


# ---------------------------------------------------------------- bench ----
def paired_bench(minutes: float, interval_s: float) -> dict:
    print("\n" + "=" * 74)
    print("PAIRED LATENCY — %s vs %s" % (PRIMARY, SECONDARY))
    print("=" * 74)
    deadline = time.time() + minutes * 60.0
    rows = {"bloXroute": [], "publicnode": []}
    errs = {"bloXroute": [], "publicnode": []}
    parity_ok = parity_bad = parity_skip = 0
    parity_ok_logbearing = parity_bad_logbearing = 0
    log_blocks: list[int] = []
    seen_blocks = set()
    n = 0
    while time.time() < deadline:
        try:
            h = head(SECONDARY)
        except Exception:
            try:
                h = head(PRIMARY)
            except Exception:
                time.sleep(interval_s)
                continue
        block = h - HEAD_LAG
        if block in seen_blocks:          # never repeat a range
            time.sleep(1.0)
            continue
        seen_blocks.add(block)
        flt = [logs_filter(block)]

        order = [("bloXroute", PRIMARY), ("publicnode", SECONDARY)]
        if random.random() < 0.5:
            order.reverse()

        fps = {}
        for label, url in order:
            try:
                logs, ms = rpc(url, "eth_getLogs", flt, timeout=BENCH_TIMEOUT_S)
                rows[label].append(ms)
                fps[label] = fingerprint(logs)
            except Exception as e:
                errs[label].append("%s: %s" % (type(e).__name__, str(e)[:70]))

        # INSTRUMENTATION GAP, fixed 2026-08-25: the first version counted
        # parity across ALL samples without recording per-sample log counts,
        # so "216 identical" was dominated by EMPTY blocks and supported no
        # fidelity claim at all. A parity number is only meaningful over
        # blocks that actually carry Bull/Bear logs, so track that subset.
        n_logs = fps.get("publicnode", fps.get("bloXroute", (0,)))[0]
        if n_logs > 0:
            log_blocks.append(block)
            if len(fps) == 2:
                if fps["bloXroute"] == fps["publicnode"]:
                    parity_ok_logbearing += 1
                else:
                    parity_bad_logbearing += 1

        if len(fps) == 2:
            if fps["bloXroute"] == fps["publicnode"]:
                parity_ok += 1
            else:
                parity_bad += 1
                print("  PARITY MISMATCH block %d: blx=%s pub=%s"
                      % (block, fps["bloXroute"][0], fps["publicnode"][0]))
        else:
            parity_skip += 1

        n += 1
        if n % 25 == 0:
            print("  ... %d paired samples, %.0f min elapsed"
                  % (n, (time.time() - (deadline - minutes * 60.0)) / 60.0),
                  flush=True)
        time.sleep(interval_s)

    def stats(v):
        if not v:
            return None
        v = sorted(v)
        q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
        return dict(n=len(v), p50=round(q(.5), 1), p90=round(q(.9), 1),
                    p99=round(q(.99), 1), max=round(v[-1], 1),
                    over_prod_timeout=sum(1 for x in v if x > PROD_TIMEOUT_MS))

    out = {"samples": n, "parity_ok": parity_ok, "parity_mismatch": parity_bad,
           "parity_skipped": parity_skip,
           "log_bearing_blocks": len(log_blocks),
           "parity_ok_logbearing": parity_ok_logbearing,
           "parity_mismatch_logbearing": parity_bad_logbearing,
           "hosts": {}}
    print("\n  %-12s %5s %8s %8s %8s %9s %14s %7s"
          % ("host", "n", "p50", "p90", "p99", "max", ">250ms(prod)", "errors"))
    for label in ("bloXroute", "publicnode"):
        s = stats(rows[label])
        out["hosts"][label] = {"latency": s, "errors": len(errs[label]),
                               "error_samples": errs[label][:5]}
        if s:
            print("  %-12s %5d %7.1fms %7.1fms %7.1fms %8.1fms %8d (%4.1f%%) %7d"
                  % (label, s["n"], s["p50"], s["p90"], s["p99"], s["max"],
                     s["over_prod_timeout"],
                     100.0 * s["over_prod_timeout"] / s["n"], len(errs[label])))
        else:
            print("  %-12s   NO SUCCESSFUL SAMPLES   errors=%d"
                  % (label, len(errs[label])))
    print("\n  parity: %d identical, %d MISMATCH, %d skipped (a host errored)"
          % (parity_ok, parity_bad, parity_skip))
    for label in ("bloXroute", "publicnode"):
        if errs[label]:
            print("  %s error sample: %s" % (label, errs[label][0]))
    return out


if __name__ == "__main__":
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 9.0
    started = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print("started %s | target %.0f min, one paired sample / %.0fs\n"
          % (started, minutes, interval))
    probe = capability_probe()
    bench = paired_bench(minutes, interval)
    doc = {"started_utc": started,
           "finished_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "probe": probe, "bench": bench}
    with open("/tmp/getlogs_bench_result.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, default=str)
    print("\nwrote /tmp/getlogs_bench_result.json")
