"""Measure fire-to-all-pool P99 RTT across the bot's production batch sizes.

The bot now fires every JSON-RPC call to every endpoint in
``READ_PATH_HEDGED_ENDPOINTS`` in parallel and takes the first response.
This probe replicates that exact pattern (urllib3 PoolManager +
ThreadPoolExecutor with max_workers=3*len(pool), FIRST_COMPLETED wait)
and measures per-batch winning-RTT distributions at the batch sizes
the bot actually uses:

  - batch=10 → used by EXPECTED_FINAL_POLL_BATCH_SIZE (wake offset)
  - batch=15 → used by EXPECTED_RAMP_POLL_{1,2}_BATCH_SIZE (wake offsets)
  - batch=20 → production cap, used by _estimated_catchup_ms (feasibility)
  - batch=2, 5 → smaller end of the existing table, for monotonicity

Output: per-batch p50/p90/p95/p99 in milliseconds + a JSON dump for
the memo. Numbers feed directly into
``RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE`` in timing_constants.py to
replace the stale 2026-05-07 single-publicnode measurements.

Note: this probe runs alongside the bot, so endpoints see ~2x normal
load. RPC providers handle thousands of req/s; tail latency should
not be materially affected, but be aware when comparing to truly
isolated runs.
"""
from __future__ import annotations

import concurrent.futures
import json
import statistics
import sys
import time

import urllib3

ENDPOINTS = [
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-dataseed1.ninicoin.io",
    "https://bsc-dataseed1.binance.org",
    "https://bsc-dataseed3.binance.org",
    "https://bsc-rpc.publicnode.com",
    "https://bsc.rpc.blxrbdn.com",
]

UA = "pancakebot-rpc-poller/1.0"
TIMEOUT_SECONDS = 5

# Match the bot's executor sizing (3 * len(pool) for straggler headroom).
EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=3 * len(ENDPOINTS), thread_name_prefix="probe-fta",
)

POOL = urllib3.PoolManager(
    num_pools=len(ENDPOINTS),
    maxsize=len(ENDPOINTS),
    headers={"User-Agent": UA, "Content-Type": "application/json"},
)


def get_head() -> int | None:
    body = json.dumps({"jsonrpc": "2.0", "id": 1,
                       "method": "eth_blockNumber", "params": []}).encode()
    for ep in ENDPOINTS:
        try:
            resp = POOL.request(
                "POST", ep, body=body,
                timeout=urllib3.Timeout(connect=TIMEOUT_SECONDS, read=TIMEOUT_SECONDS),
                retries=False,
            )
            if resp.status != 200:
                continue
            return int(json.loads(resp.data)["result"], 16)
        except Exception:
            continue
    return None


def rpc_post(url: str, body: bytes) -> bytes:
    resp = POOL.request(
        "POST", url, body=body,
        timeout=urllib3.Timeout(connect=TIMEOUT_SECONDS, read=TIMEOUT_SECONDS),
        retries=False,
    )
    if resp.status != 200:
        raise urllib3.exceptions.HTTPError(f"http_{resp.status}")
    return resp.data


def fire_to_all(body: bytes) -> tuple[str | None, float]:
    """Fire to every endpoint; return (winner, rtt_ms) or (None, deadline_ms)."""
    fut_to_ep = {EXECUTOR.submit(rpc_post, ep, body): ep for ep in ENDPOINTS}
    pending = set(fut_to_ep.keys())
    deadline = time.monotonic() + TIMEOUT_SECONDS
    t0 = time.monotonic()
    while pending:
        remaining = max(0.001, deadline - time.monotonic())
        done, pending = concurrent.futures.wait(
            pending, timeout=remaining,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            return None, (time.monotonic() - t0) * 1000
        for fut in done:
            try:
                _resp = fut.result()
                return fut_to_ep[fut], (time.monotonic() - t0) * 1000
            except Exception:
                continue
    return None, (time.monotonic() - t0) * 1000


def measure(batch_size: int, n_samples: int, head: int) -> dict:
    print(f"\n=== batch_size={batch_size} n={n_samples} ===", flush=True)
    rtts: list[float] = []
    failures = 0
    # Bot makes fire-to-all calls at most every ~3s in critical-path
    # mode (ramp/final/critical) and every 30s in periodic-poll mode.
    # Sleep 3s between probe calls so abandoned-future stragglers from
    # the prior call have time to clear before the next; otherwise the
    # 18-worker executor saturates with pinned-for-5s urllib3 sockets
    # and the measurement reflects executor-queueing artifacts, not
    # endpoint latency. Trade-off: 200 calls take ~10-15 min wall.
    INTER_CALL_SLEEP_S = 3.0
    for i in range(n_samples):
        offset = (i * batch_size) % 1000
        first = head - offset - batch_size + 1
        batch = [
            {"jsonrpc": "2.0", "id": j, "method": "eth_getBlockReceipts",
             "params": [hex(first + j)]}
            for j in range(batch_size)
        ]
        body = json.dumps(batch).encode()
        winner, rtt = fire_to_all(body)
        if winner is None:
            failures += 1
        else:
            rtts.append(rtt)
        if (i + 1) % 25 == 0:
            print(f"  progress: {i+1}/{n_samples} (failures={failures})",
                  flush=True)
        time.sleep(INTER_CALL_SLEEP_S)

    if not rtts:
        return {"batch_size": batch_size, "n_samples": n_samples,
                "successes": 0, "failures": failures,
                "p50": None, "p90": None, "p95": None, "p99": None}

    s = sorted(rtts)
    def pct(p):
        return s[max(0, min(len(s) - 1, int(p / 100.0 * (len(s) - 1))))]

    summary = {
        "batch_size": batch_size,
        "n_samples": n_samples,
        "successes": len(rtts),
        "failures": failures,
        "min": s[0], "max": s[-1],
        "mean": statistics.mean(rtts),
        "p50": pct(50), "p90": pct(90), "p95": pct(95), "p99": pct(99),
    }
    print(f"  successes={len(rtts)}/{n_samples} failures={failures}", flush=True)
    print(f"  p50={summary['p50']:.0f} p90={summary['p90']:.0f} "
          f"p95={summary['p95']:.0f} p99={summary['p99']:.0f} "
          f"max={summary['max']:.0f}", flush=True)
    return summary


def main():
    head = get_head()
    if head is None:
        print("could not fetch head block", flush=True)
        sys.exit(1)
    print(f"head_block={head}", flush=True)
    print(f"executor max_workers={3 * len(ENDPOINTS)} "
          f"pool maxsize={len(ENDPOINTS)} timeout={TIMEOUT_SECONDS}s",
          flush=True)

    # Sample sizes per batch size. p99 stability requires ~50+ samples;
    # batch=20 (production cap; used by feasibility math) gets 2x for
    # tighter tail estimate. Combined with the 3s inter-call sleep this
    # is ~15 min total wallclock.
    schedule = [
        (2, 50),
        (5, 50),
        (10, 50),
        (15, 50),
        (20, 100),
    ]
    results = []
    for batch_size, n in schedule:
        results.append(measure(batch_size, n, head))

    # Persist for the memo.
    out_path = (
        "var/incident_reports/2026_05_11_fire_to_all_p99_measurement.json"
    )
    payload = {
        "probe": "research/probe_fire_to_all_p99_2026_05_11.py",
        "endpoints": ENDPOINTS,
        "executor_max_workers": 3 * len(ENDPOINTS),
        "pool_maxsize": len(ENDPOINTS),
        "timeout_seconds": TIMEOUT_SECONDS,
        "head_block": head,
        "results": results,
    }
    import os
    os.makedirs("var/incident_reports", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    # Headline table for direct copy into timing_constants.py
    print("\n=== Headline P99 table (to paste) ===", flush=True)
    print("RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE: dict[int, int] = {", flush=True)
    for r in results:
        if r["p99"] is not None:
            print(f"    {r['batch_size']}: {int(round(r['p99']))},  # p99 (n={r['n_samples']}, fire-to-all, urllib3)",
                  flush=True)
    print("}", flush=True)

    EXECUTOR.shutdown(wait=False)


if __name__ == "__main__":
    main()
