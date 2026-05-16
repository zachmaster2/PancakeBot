"""Diagnose fire-to-all P99: explicit warmup + diverse-only pool variant.

Two methodology questions from the user re: the 2026-05-11 fire-to-all
P99 measurement (which produced 1319ms at batch=20):

1. Was the urllib3 PoolManager properly warm before measurements started?
   The prior probe's `get_head()` returns on first success, warming only
   the first endpoint in the list. Subsequent fire_to_all samples 2+
   benefit from cumulative warmup via prior calls' stragglers, but sample
   1 hits 5 cold connections. Need explicit per-endpoint warmup to
   eliminate this bias.

2. Is the 4 BSC-dataseed-family + 2 distinct-provider pool actually
   slower than a leaner 1-per-provider pool? If dataseed endpoints
   share infrastructure with correlated tails, having 4 of them in the
   pool might not help vs 1 dataseed + publicnode + bloXroute. Theory
   of fire-to-all says "more racers = faster" only if endpoints are
   independent.

This probe runs two variants back-to-back, both with explicit warmup:

  VARIANT A: 6 endpoints (current bot pool)
  VARIANT B: 3 endpoints (defibit + publicnode + bloXroute = 1 per provider)

WARMUP: before each variant's timed measurements, fire 3 sequential
requests per endpoint and wait for completion. This populates the
urllib3 PoolManager's per-host connection pool AND warms OS DNS cache,
TCP connect, TLS handshake state.

Bot stays running (PID 23924 on master 0045483) — absolute values
may include bot-contention inflation but the A-vs-B COMPARISON is
the point. The 2026-05-11 bot-off probe gave 1319ms p99 for the
6-endpoint pool; this run will measure A again (with bot ON, so
expect higher) AND B (3 endpoints, with bot ON).
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import time

import urllib3

POOL_6 = [
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-dataseed1.ninicoin.io",
    "https://bsc-dataseed1.binance.org",
    "https://bsc-dataseed3.binance.org",
    "https://bsc-rpc.publicnode.com",
    "https://bsc.rpc.blxrbdn.com",
]
POOL_3 = [
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-rpc.publicnode.com",
    "https://bsc.rpc.blxrbdn.com",
]

UA = "pancakebot-rpc-poller/1.0"
TIMEOUT_SECONDS = 5
BATCH_SIZE = 20
N_SAMPLES = 30
INTER_CALL_SLEEP_S = 30.0
WARMUP_PER_ENDPOINT = 3


def make_pool(pool_size: int) -> urllib3.PoolManager:
    return urllib3.PoolManager(
        num_pools=pool_size,
        maxsize=pool_size,
        headers={"User-Agent": UA, "Content-Type": "application/json"},
    )


def get_head(pool: urllib3.PoolManager, endpoints: list[str]) -> int | None:
    body = json.dumps({"jsonrpc": "2.0", "id": 1,
                       "method": "eth_blockNumber", "params": []}).encode()
    for ep in endpoints:
        try:
            resp = pool.request(
                "POST", ep, body=body,
                timeout=urllib3.Timeout(connect=TIMEOUT_SECONDS, read=TIMEOUT_SECONDS),
                retries=False,
            )
            if resp.status == 200:
                return int(json.loads(resp.data)["result"], 16)
        except Exception:
            continue
    return None


def rpc_post_timed(pool, url: str, body: bytes) -> tuple[bool, float]:
    t0 = time.monotonic()
    try:
        resp = pool.request(
            "POST", url, body=body,
            timeout=urllib3.Timeout(connect=TIMEOUT_SECONDS, read=TIMEOUT_SECONDS),
            retries=False,
        )
        rtt = (time.monotonic() - t0) * 1000
        return resp.status == 200, rtt
    except Exception:
        return False, (time.monotonic() - t0) * 1000


def warmup_pool(pool: urllib3.PoolManager, endpoints: list[str]) -> dict:
    """Explicit per-endpoint warmup: fire WARMUP_PER_ENDPOINT sequential
    requests to each endpoint, all completing before next endpoint, so the
    PoolManager has a warm connection cached for every endpoint when
    fire_to_all begins.

    Returns per-endpoint warmup stats (rtts) for diagnostic purposes —
    the FIRST request to each endpoint is the cold one; subsequent
    requests should be much faster (DNS cached, TCP/TLS reused).
    """
    print(f"  Warming up {len(endpoints)} endpoints ({WARMUP_PER_ENDPOINT} "
          f"requests each, sequential)...", flush=True)
    body = json.dumps({"jsonrpc": "2.0", "id": 1,
                       "method": "eth_blockNumber", "params": []}).encode()
    out = {}
    for ep in endpoints:
        rtts = []
        for _ in range(WARMUP_PER_ENDPOINT):
            ok, rtt = rpc_post_timed(pool, ep, body)
            rtts.append(rtt)
            time.sleep(0.1)
        out[ep] = rtts
        cold = rtts[0]
        warm = sum(rtts[1:]) / max(1, len(rtts) - 1) if len(rtts) > 1 else 0
        print(f"    {ep:42s}  cold={cold:5.0f}ms  warm_avg={warm:5.0f}ms",
              flush=True)
    print(f"  Warmup complete.", flush=True)
    return out


def pct(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    return s[max(0, min(len(s) - 1, int(p / 100.0 * (len(s) - 1))))]


def run_variant(
    label: str, endpoints: list[str], head: int,
) -> dict:
    """Run a full variant: warmup + N timed fire-to-all samples."""
    print(f"\n{'='*60}", flush=True)
    print(f"VARIANT {label}: {len(endpoints)}-endpoint pool", flush=True)
    print(f"{'='*60}", flush=True)
    for ep in endpoints:
        print(f"  - {ep}", flush=True)

    pool = make_pool(len(endpoints))
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=3 * len(endpoints),
        thread_name_prefix=f"probe-{label}",
    )

    warmup_stats = warmup_pool(pool, endpoints)

    print(f"\n  Measuring fire-to-all (n={N_SAMPLES}, batch={BATCH_SIZE}, "
          f"spacing={INTER_CALL_SLEEP_S}s)...", flush=True)

    winner_rtts: list[float] = []
    failures = 0

    for i in range(N_SAMPLES):
        offset = (i * BATCH_SIZE) % 1000
        first = head - offset - BATCH_SIZE + 1
        batch = [
            {"jsonrpc": "2.0", "id": j, "method": "eth_getBlockReceipts",
             "params": [hex(first + j)]}
            for j in range(BATCH_SIZE)
        ]
        body = json.dumps(batch).encode()

        t_start = time.monotonic()
        fut_to_ep = {
            executor.submit(rpc_post_timed, pool, ep, body): ep
            for ep in endpoints
        }
        pending = set(fut_to_ep.keys())
        deadline = t_start + TIMEOUT_SECONDS
        winner_rtt: float | None = None

        while pending:
            remaining = max(0.001, deadline - time.monotonic())
            done, pending = concurrent.futures.wait(
                pending, timeout=remaining,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                break
            for fut in done:
                try:
                    ok, _ = fut.result()
                    if ok and winner_rtt is None:
                        winner_rtt = (time.monotonic() - t_start) * 1000
                except Exception:
                    pass
            if winner_rtt is not None:
                break

        if winner_rtt is not None:
            winner_rtts.append(winner_rtt)
            print(f"    [{i+1:2d}/{N_SAMPLES}] winner={winner_rtt:.0f}ms",
                  flush=True)
        else:
            failures += 1
            print(f"    [{i+1:2d}/{N_SAMPLES}] FAIL_ALL", flush=True)

        if i + 1 < N_SAMPLES:
            time.sleep(INTER_CALL_SLEEP_S)

    summary = {
        "label": label,
        "endpoints": endpoints,
        "n": N_SAMPLES,
        "successes": len(winner_rtts),
        "failures": failures,
        "p50": pct(winner_rtts, 50),
        "p90": pct(winner_rtts, 90),
        "p95": pct(winner_rtts, 95),
        "p99": pct(winner_rtts, 99),
        "max": max(winner_rtts) if winner_rtts else 0,
    }
    print(f"\n  {label} results: successes={len(winner_rtts)}/{N_SAMPLES} "
          f"failures={failures}", flush=True)
    print(f"    p50={summary['p50']:.0f} p90={summary['p90']:.0f} "
          f"p95={summary['p95']:.0f} p99={summary['p99']:.0f} "
          f"max={summary['max']:.0f}", flush=True)

    executor.shutdown(wait=False)
    pool.clear()
    return summary


def main():
    print("Probe: fire-to-all warmup + diversity diagnostic", flush=True)
    print(f"  BATCH_SIZE={BATCH_SIZE}  N_SAMPLES={N_SAMPLES}  "
          f"spacing={INTER_CALL_SLEEP_S}s  warmup={WARMUP_PER_ENDPOINT}/endpoint",
          flush=True)
    print(f"  Bot status: assumed RUNNING (PID 23924 on master 0045483).",
          flush=True)
    print(f"  Expected wallclock: ~{2 * INTER_CALL_SLEEP_S * N_SAMPLES / 60:.0f} min",
          flush=True)

    # Get head once via a quick lookup (don't pollute warmup stats).
    bootstrap_pool = make_pool(1)
    head = get_head(bootstrap_pool, POOL_6)
    bootstrap_pool.clear()
    if head is None:
        print("could not fetch head", flush=True)
        sys.exit(1)
    print(f"  head_block={head}", flush=True)

    a = run_variant("A_6_endpoints", POOL_6, head)
    b = run_variant("B_3_endpoints", POOL_3, head)

    print(f"\n{'='*60}", flush=True)
    print(f"COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Metric    A (6ep)   B (3ep)   delta", flush=True)
    for k in ["p50", "p90", "p95", "p99", "max"]:
        a_v = a[k]
        b_v = b[k]
        delta = b_v - a_v
        sign = "+" if delta >= 0 else ""
        print(f"  {k:6s}     {a_v:5.0f}    {b_v:5.0f}    "
              f"{sign}{delta:.0f}ms", flush=True)
    print(f"  failures   {a['failures']:5d}    {b['failures']:5d}",
          flush=True)
    print(f"\n  Interpretation:", flush=True)
    if b["p99"] < a["p99"] * 0.8:
        print(f"    B (3ep diverse) is materially faster at p99 — "
              f"dataseed-family correlation hypothesis SUPPORTED.", flush=True)
    elif b["p99"] > a["p99"] * 1.2:
        print(f"    B (3ep) is slower at p99 — fewer racers loses the "
              f"min-of-N benefit; diverse hypothesis NOT supported.",
              flush=True)
    else:
        print(f"    A and B p99 within ~20% — dataseed-family correlation "
              f"is NOT the dominant factor in fire-to-all tail.",
              flush=True)


if __name__ == "__main__":
    main()
