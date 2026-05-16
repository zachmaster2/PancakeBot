"""Verify the fire-to-all probe methodology.

Side-by-side comparison:
  1. SINGLE-ENDPOINT: time each endpoint sequentially with urllib3 PoolManager.
     Per-endpoint p50/p99.
  2. FIRE-TO-ALL (instrumented): fire to all 6 in parallel, record per-future
     completion time (not just winner). Compute:
       - measured winner time (what the probe reports)
       - actual min(per-future) (what fire-to-all SHOULD be measuring)
       - per-endpoint distribution under parallel load

If measured-winner-time == min(per-future), methodology is correct and the
measurement reflects real network behavior (correlated tail). If they differ,
there's a measurement bug.

Sanity check: fire-to-all p99 should be <= best-single-endpoint p99. If not,
either methodology is broken OR endpoints are heavily correlated under
concurrent load.

Run alongside bot (PID 5780) — same conditions as the prior probe.
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
BATCH_SIZE = 20
N_SAMPLES = 30
INTER_CALL_SLEEP = 3.0

EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=3 * len(ENDPOINTS), thread_name_prefix="verify-fta",
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


def rpc_post_timed(url: str, body: bytes) -> tuple[bool, float]:
    """Returns (success, rtt_ms)."""
    t0 = time.monotonic()
    try:
        resp = POOL.request(
            "POST", url, body=body,
            timeout=urllib3.Timeout(connect=TIMEOUT_SECONDS, read=TIMEOUT_SECONDS),
            retries=False,
        )
        rtt = (time.monotonic() - t0) * 1000
        return resp.status == 200, rtt
    except Exception:
        return False, (time.monotonic() - t0) * 1000


def pct(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    return s[max(0, min(len(s) - 1, int(p / 100.0 * (len(s) - 1))))]


def measure_single(head: int, batch_size: int, n: int) -> dict:
    """Sequential per-endpoint measurement — establishes the SINGLE baseline."""
    print(f"\n=== SCENARIO A: sequential per-endpoint (batch={batch_size}, n={n} each) ===",
          flush=True)
    out = {}
    for ep in ENDPOINTS:
        rtts: list[float] = []
        for i in range(n):
            offset = (i * batch_size) % 1000
            first = head - offset - batch_size + 1
            batch = [
                {"jsonrpc": "2.0", "id": j, "method": "eth_getBlockReceipts",
                 "params": [hex(first + j)]}
                for j in range(batch_size)
            ]
            body = json.dumps(batch).encode()
            ok, rtt = rpc_post_timed(ep, body)
            if ok:
                rtts.append(rtt)
            time.sleep(0.3)  # gentle spacing
        out[ep] = {
            "n": len(rtts), "min": min(rtts) if rtts else 0,
            "p50": pct(rtts, 50), "p90": pct(rtts, 90),
            "p99": pct(rtts, 99), "max": max(rtts) if rtts else 0,
        }
        s = out[ep]
        print(f"  {ep:42s}  n={s['n']:2d}  p50={s['p50']:5.0f}  "
              f"p99={s['p99']:5.0f}  max={s['max']:5.0f}", flush=True)
    return out


def measure_fire_to_all_instrumented(head: int, batch_size: int, n: int) -> dict:
    """Fire-to-all with per-future timing capture. Reveals if measured
    winner-time differs from min(per-future) — that would indicate a
    measurement bug."""
    print(f"\n=== SCENARIO B: fire-to-all instrumented (batch={batch_size}, n={n}) ===",
          flush=True)
    winner_rtts: list[float] = []
    actual_min_rtts: list[float] = []
    per_ep_rtts: dict[str, list[float]] = {ep: [] for ep in ENDPOINTS}
    measurement_gaps: list[float] = []  # winner_time - actual_min

    for i in range(n):
        offset = (i * batch_size) % 1000
        first = head - offset - batch_size + 1
        batch = [
            {"jsonrpc": "2.0", "id": j, "method": "eth_getBlockReceipts",
             "params": [hex(first + j)]}
            for j in range(batch_size)
        ]
        body = json.dumps(batch).encode()

        # Fire-to-all with per-future timing.
        t_start = time.monotonic()
        fut_to_ep = {}
        for ep in ENDPOINTS:
            fut = EXECUTOR.submit(rpc_post_timed, ep, body)
            fut_to_ep[fut] = ep
        t_submitted = time.monotonic()

        # Wait FIRST_COMPLETED until success.
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
                    ok, fut_rtt = fut.result()
                    if ok and winner_rtt is None:
                        winner_rtt = (time.monotonic() - t_start) * 1000
                        # capture the per-future-internal rtt for the winner
                        per_ep_rtts[fut_to_ep[fut]].append(fut_rtt)
                    elif ok:
                        # Recorded sibling: per-future rtt
                        per_ep_rtts[fut_to_ep[fut]].append(fut_rtt)
                except Exception:
                    pass
            if winner_rtt is not None:
                break

        # Drain remaining pending: record their per-future rtts as they finish,
        # up to deadline. Important to NOT skip — we want per-future stats.
        # But don't block longer than deadline.
        if pending:
            try:
                more_done = concurrent.futures.wait(
                    pending, timeout=max(0, deadline - time.monotonic()),
                    return_when=concurrent.futures.ALL_COMPLETED,
                )
                for fut in more_done.done:
                    try:
                        ok, fut_rtt = fut.result()
                        if ok:
                            per_ep_rtts[fut_to_ep[fut]].append(fut_rtt)
                    except Exception:
                        pass
            except Exception:
                pass

        if winner_rtt is not None:
            winner_rtts.append(winner_rtt)
            # actual min across per-future rtts that completed by now
            this_cycle_rtts = []
            for ep in ENDPOINTS:
                if per_ep_rtts[ep]:
                    this_cycle_rtts.append(per_ep_rtts[ep][-1])
            if this_cycle_rtts:
                min_rtt = min(this_cycle_rtts)
                actual_min_rtts.append(min_rtt)
                measurement_gaps.append(winner_rtt - min_rtt)

        if (i + 1) % 5 == 0:
            print(f"  progress: {i+1}/{n}", flush=True)
        time.sleep(INTER_CALL_SLEEP)

    print(f"\n  Winner times (probe's reported metric):", flush=True)
    print(f"    n={len(winner_rtts)}  p50={pct(winner_rtts,50):.0f}  "
          f"p99={pct(winner_rtts,99):.0f}  max={max(winner_rtts) if winner_rtts else 0:.0f}",
          flush=True)
    print(f"  Actual min(per-future) (what fire-to-all SHOULD measure):", flush=True)
    print(f"    n={len(actual_min_rtts)}  p50={pct(actual_min_rtts,50):.0f}  "
          f"p99={pct(actual_min_rtts,99):.0f}  max={max(actual_min_rtts) if actual_min_rtts else 0:.0f}",
          flush=True)
    print(f"  Measurement gap (winner_time - actual_min):", flush=True)
    print(f"    n={len(measurement_gaps)}  p50={pct(measurement_gaps,50):.0f}  "
          f"p99={pct(measurement_gaps,99):.0f}  max={max(measurement_gaps) if measurement_gaps else 0:.0f}",
          flush=True)
    print(f"\n  Per-endpoint completion times under parallel load:", flush=True)
    for ep in ENDPOINTS:
        r = per_ep_rtts[ep]
        if r:
            print(f"    {ep:42s}  n={len(r):2d}  p50={pct(r,50):5.0f}  "
                  f"p99={pct(r,99):5.0f}", flush=True)
        else:
            print(f"    {ep:42s}  no completions", flush=True)
    return {
        "winner_rtts": winner_rtts,
        "actual_min_rtts": actual_min_rtts,
        "measurement_gaps": measurement_gaps,
        "per_ep_rtts": per_ep_rtts,
    }


def main():
    head = get_head()
    if head is None:
        print("could not fetch head", flush=True)
        sys.exit(1)
    print(f"head={head} batch_size={BATCH_SIZE} n_samples={N_SAMPLES}", flush=True)

    # Scenario A: sequential single
    single = measure_single(head, BATCH_SIZE, N_SAMPLES)

    # Scenario B: fire-to-all instrumented
    fta = measure_fire_to_all_instrumented(head, BATCH_SIZE, N_SAMPLES)

    # Sanity check
    print(f"\n=== Sanity check ===", flush=True)
    best_single_p99 = min(s["p99"] for s in single.values())
    print(f"  Best single-endpoint p99: {best_single_p99:.0f}ms", flush=True)
    fta_winner_p99 = pct(fta["winner_rtts"], 99)
    print(f"  Fire-to-all winner-time p99: {fta_winner_p99:.0f}ms", flush=True)
    if fta_winner_p99 > best_single_p99:
        print(f"  *** ANOMALY: fire-to-all is SLOWER than best single endpoint by "
              f"{fta_winner_p99 - best_single_p99:.0f}ms", flush=True)
        print(f"  *** This shouldn't happen if methodology is correct.", flush=True)
    else:
        print(f"  OK: fire-to-all is faster than best single endpoint by "
              f"{best_single_p99 - fta_winner_p99:.0f}ms", flush=True)

    EXECUTOR.shutdown(wait=False)


if __name__ == "__main__":
    main()
