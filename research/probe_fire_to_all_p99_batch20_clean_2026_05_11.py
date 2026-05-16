"""Clean-spacing re-probe: fire-to-all p99 at batch=20, free of executor-queue artifact.

The 2026-05-11 verification probe (probe_methodology_verify_2026_05_11.py)
revealed that the original probe's wall-clock measurement included an
executor-queue-delay tail (winner_time p99 = 4435ms vs actual_min p99 =
2024ms, gap p99 = 1371ms). Cause: 3s inter-call spacing < urllib3's 5s
read timeout, so abandoned futures from previous calls pinned executor
workers, biasing main-thread observation lag.

The bot doesn't suffer this in production because its fire-to-all calls
are at least 1.4s apart (final → critical_path) and typically 30s apart
(periodic poll cadence) — abandoned futures clear cleanly between calls.

This probe replicates the bot's spacing exactly: 30s inter-call. Only
batch=20 (the production cap that feeds `_estimated_catchup_ms`).
Instrumented to record both winner-time AND per-future RTT so we can
confirm the gap has collapsed (validating the methodology).

Sample size: 30 calls × 30s spacing = 15 min wall.
"""
from __future__ import annotations

import concurrent.futures
import json
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
INTER_CALL_SLEEP_S = 30.0

EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=3 * len(ENDPOINTS), thread_name_prefix="probe-clean",
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
    """Returns (success, internal_rtt_ms) measured inside the worker thread."""
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


def main():
    head = get_head()
    if head is None:
        print("could not fetch head", flush=True)
        sys.exit(1)
    print(f"head={head} batch_size={BATCH_SIZE} n={N_SAMPLES} "
          f"inter_call={INTER_CALL_SLEEP_S}s", flush=True)
    print(f"executor max_workers={3*len(ENDPOINTS)} pool maxsize={len(ENDPOINTS)}",
          flush=True)
    print(f"Expected wallclock: ~{INTER_CALL_SLEEP_S * N_SAMPLES / 60:.0f} min",
          flush=True)

    winner_rtts: list[float] = []
    actual_min_rtts: list[float] = []
    measurement_gaps: list[float] = []
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

        # Fire-to-all
        t_start = time.monotonic()
        fut_to_ep = {}
        for ep in ENDPOINTS:
            fut = EXECUTOR.submit(rpc_post_timed, ep, body)
            fut_to_ep[fut] = ep

        pending = set(fut_to_ep.keys())
        deadline = t_start + TIMEOUT_SECONDS
        winner_rtt: float | None = None
        cycle_per_future_rtts: list[float] = []
        cycle_winner_internal_rtt: float | None = None

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
                    if ok:
                        cycle_per_future_rtts.append(fut_rtt)
                        if winner_rtt is None:
                            winner_rtt = (time.monotonic() - t_start) * 1000
                            cycle_winner_internal_rtt = fut_rtt
                except Exception:
                    pass
            if winner_rtt is not None:
                break

        # Drain remaining pending to capture per-future stats (best-effort,
        # bounded by deadline).
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
                            cycle_per_future_rtts.append(fut_rtt)
                    except Exception:
                        pass
            except Exception:
                pass

        if winner_rtt is not None:
            winner_rtts.append(winner_rtt)
            if cycle_per_future_rtts:
                actual_min = min(cycle_per_future_rtts)
                actual_min_rtts.append(actual_min)
                measurement_gaps.append(winner_rtt - actual_min)
        else:
            failures += 1

        # Per-call progress line (every call, since 30s spacing makes
        # this readable).
        if winner_rtt is not None:
            print(f"  [{i+1:2d}/{N_SAMPLES}] winner={winner_rtt:.0f}ms "
                  f"(internal={cycle_winner_internal_rtt:.0f}ms, "
                  f"min={actual_min_rtts[-1]:.0f}ms, "
                  f"gap={measurement_gaps[-1]:.0f}ms)", flush=True)
        else:
            print(f"  [{i+1:2d}/{N_SAMPLES}] FAIL_ALL "
                  f"(timeouts on every endpoint)", flush=True)

        if i + 1 < N_SAMPLES:
            time.sleep(INTER_CALL_SLEEP_S)

    print(f"\n=== Results (n={N_SAMPLES}, batch_size={BATCH_SIZE}, "
          f"spacing={INTER_CALL_SLEEP_S}s) ===", flush=True)
    print(f"successes: {len(winner_rtts)}/{N_SAMPLES}, failures: {failures}",
          flush=True)
    print(f"\nWinner-time (probe's wallclock):", flush=True)
    print(f"  p50={pct(winner_rtts,50):.0f} p90={pct(winner_rtts,90):.0f} "
          f"p95={pct(winner_rtts,95):.0f} p99={pct(winner_rtts,99):.0f} "
          f"max={max(winner_rtts) if winner_rtts else 0:.0f}", flush=True)
    print(f"\nActual min(per-future) (artifact-free):", flush=True)
    print(f"  p50={pct(actual_min_rtts,50):.0f} p90={pct(actual_min_rtts,90):.0f} "
          f"p95={pct(actual_min_rtts,95):.0f} p99={pct(actual_min_rtts,99):.0f} "
          f"max={max(actual_min_rtts) if actual_min_rtts else 0:.0f}", flush=True)
    print(f"\nMeasurement gap (winner_time - actual_min) — should be small if "
          f"executor not saturated:", flush=True)
    print(f"  p50={pct(measurement_gaps,50):.0f} p90={pct(measurement_gaps,90):.0f} "
          f"p99={pct(measurement_gaps,99):.0f} "
          f"max={max(measurement_gaps) if measurement_gaps else 0:.0f}",
          flush=True)

    EXECUTOR.shutdown(wait=False)


if __name__ == "__main__":
    main()
