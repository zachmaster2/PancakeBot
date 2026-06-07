"""Re-measure RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE under the CURRENT
3-endpoint hedged pool (READ_PATH_HEDGED_ENDPOINTS), n>=100 per bucket.

The table in pancakebot/timing_constants.py (~line 322) was measured under
the OLD 6-endpoint pool at n=30 (pre-Bundle-6). Bundle 6 trimmed the pool
to 3 endpoints, so production now runs min-of-3; p99 is expected HIGHER.
The table feeds `_estimated_catchup_ms` (catch-up feasibility gate).

Transport matches production exactly:
  - urllib3 PoolManager, fire-to-all hedged (first 200 wins, rest abandoned)
  - receipts-only eth_getBlockReceipts batches
  - the live 3-endpoint READ_PATH_HEDGED_ENDPOINTS pool

Spacing: 5s inter-call (safe per probe_batch_size_2026_05_14.py; the
executor-queue artifact documented in probe_fire_to_all_p99_batch20_clean
only appears below 5s, where 3s < urllib3's 5s read timeout biased p99
high). 5s keeps total runtime reasonable while not degrading the live bot
(PID ~48208) which polls these endpoints sparsely (8s periodic).

Methodology guards:
  - We record BOTH the winner wallclock (what the bot observes) AND the
    per-future internal RTT so we can confirm the measurement gap stays
    small (artifact-free) and report per-endpoint p99.
  - Round-robin across [2, 5, 10, 15, 20]; n>=100 per size, prioritizing
    20 and 15 (the load-bearing buckets).
  - 5s read timeout matches production RPC_HTTP_BATCH_TIMEOUT_SECONDS.

Output: research/probe_batch_receipts_p99_3ep_2026_06_03.jsonl (raw)
        research/probe_batch_receipts_p99_3ep_2026_06_03_summary.json
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

import urllib3

# CURRENT live 3-endpoint pool (READ_PATH_HEDGED_ENDPOINTS in
# pancakebot/chain/rpc_poller.py, verified 2026-06-03).
ENDPOINTS = [
    "https://bsc-dataseed1.binance.org",
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-rpc.publicnode.com",
]

UA = "pancakebot-rpc-poller/1.0"
TIMEOUT_SECONDS = 5  # matches RPC_HTTP_BATCH_TIMEOUT_SECONDS
BATCH_SIZES = [2, 5, 10, 15, 20]
N_PER_SIZE = 100
INTER_CALL_SLEEP_S = 5.0

HERE = Path(__file__).resolve().parent
JSONL_PATH = HERE / "probe_batch_receipts_p99_3ep_2026_06_03.jsonl"
SUMMARY_PATH = HERE / "probe_batch_receipts_p99_3ep_2026_06_03_summary.json"

# Back-off guard: if we see this many consecutive all-fail cycles, widen
# spacing so we don't degrade the live bot.
MAX_CONSEC_ALLFAIL = 3

EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=3 * len(ENDPOINTS), thread_name_prefix="probe-3ep",
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


def build_body(head: int, sample_idx: int, size: int) -> bytes:
    # Spread start blocks so we don't repeatedly hit the same cache slot.
    offset = 200 + (sample_idx * 23) % 4000
    first = head - offset
    batch = [
        {"jsonrpc": "2.0", "id": j, "method": "eth_getBlockReceipts",
         "params": [hex(first + j)]}
        for j in range(size)
    ]
    return json.dumps(batch).encode()


def hedged_call(body: bytes):
    """Fire-to-all; return (winner_wallclock_ms, winner_ep, winner_internal_ms,
    per_ep_rtts: dict[ep -> (ok, rtt_ms)])."""
    t_start = time.monotonic()
    fut_to_ep = {EXECUTOR.submit(rpc_post_timed, ep, body): ep for ep in ENDPOINTS}
    pending = set(fut_to_ep.keys())
    deadline = t_start + TIMEOUT_SECONDS
    winner_wall: float | None = None
    winner_ep: str | None = None
    winner_internal: float | None = None
    per_ep: dict[str, tuple[bool, float]] = {}

    while pending:
        remaining = max(0.001, deadline - time.monotonic())
        done, pending = concurrent.futures.wait(
            pending, timeout=remaining,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            break
        for fut in done:
            ep = fut_to_ep[fut]
            try:
                ok, fut_rtt = fut.result()
                per_ep[ep] = (ok, fut_rtt)
                if ok and winner_wall is None:
                    winner_wall = (time.monotonic() - t_start) * 1000
                    winner_ep = ep
                    winner_internal = fut_rtt
            except Exception:
                per_ep[ep] = (False, (time.monotonic() - t_start) * 1000)
        if winner_wall is not None:
            break

    # Drain remaining to capture per-endpoint stats (best-effort, bounded).
    if pending:
        try:
            res = concurrent.futures.wait(
                pending, timeout=max(0.0, deadline - time.monotonic()),
                return_when=concurrent.futures.ALL_COMPLETED,
            )
            for fut in res.done:
                ep = fut_to_ep[fut]
                try:
                    ok, fut_rtt = fut.result()
                    per_ep[ep] = (ok, fut_rtt)
                except Exception:
                    pass
        except Exception:
            pass

    return winner_wall, winner_ep, winner_internal, per_ep


CURRENT_TABLE = {2: 421, 5: 771, 10: 910, 15: 1213, 20: 1319}


def main() -> int:
    head = get_head()
    if head is None:
        print("could not fetch head", flush=True)
        return 1

    total = N_PER_SIZE * len(BATCH_SIZES)
    print(f"head={head} sizes={BATCH_SIZES} n_per_size={N_PER_SIZE} "
          f"total={total} spacing={INTER_CALL_SLEEP_S}s", flush=True)
    print(f"endpoints={ENDPOINTS}", flush=True)
    print(f"expected wallclock ~{total * INTER_CALL_SLEEP_S / 60:.0f} min",
          flush=True)

    winner_wall: dict[int, list[float]] = {bs: [] for bs in BATCH_SIZES}
    winner_internal: dict[int, list[float]] = {bs: [] for bs in BATCH_SIZES}
    per_ep_rtts: dict[str, list[float]] = {ep: [] for ep in ENDPOINTS}
    failures: dict[int, int] = {bs: 0 for bs in BATCH_SIZES}
    consec_allfail = 0
    spacing = INTER_CALL_SLEEP_S

    f = JSONL_PATH.open("w", encoding="utf-8")
    try:
        for i in range(total):
            bs = BATCH_SIZES[i % len(BATCH_SIZES)]
            body = build_body(head, i, bs)
            wall, ep, internal, per_ep = hedged_call(body)

            rec = {
                "ts": time.time(), "idx": i, "batch_size": bs,
                "winner_wall_ms": wall, "winner_ep": ep,
                "winner_internal_ms": internal,
                "per_ep": {e: {"ok": ok, "rtt_ms": r}
                           for e, (ok, r) in per_ep.items()},
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

            for e, (ok, r) in per_ep.items():
                if ok:
                    per_ep_rtts[e].append(r)

            if wall is not None:
                winner_wall[bs].append(wall)
                if internal is not None:
                    winner_internal[bs].append(internal)
                consec_allfail = 0
            else:
                failures[bs] += 1
                consec_allfail += 1

            done_for_bs = len(winner_wall[bs]) + failures[bs]
            if (i + 1) % 5 == 0 or wall is None:
                tag = "FAIL_ALL" if wall is None else f"{wall:.0f}ms ep={ep}"
                print(f"  [{i+1:3d}/{total}] bs={bs:2d} "
                      f"(#{done_for_bs} for this size) {tag}", flush=True)

            # Adaptive back-off if the live bot is being stressed.
            if consec_allfail >= MAX_CONSEC_ALLFAIL:
                spacing = min(30.0, spacing * 2)
                print(f"  !! {consec_allfail} consecutive all-fail -> "
                      f"widening spacing to {spacing}s", flush=True)
                consec_allfail = 0

            if i + 1 < total:
                time.sleep(spacing)
    finally:
        f.close()

    # Build summary.
    summary: dict = {
        "endpoints": ENDPOINTS,
        "spacing_s": INTER_CALL_SLEEP_S,
        "timeout_s": TIMEOUT_SECONDS,
        "current_table": CURRENT_TABLE,
        "per_size": {},
        "per_endpoint": {},
    }
    print(f"\n=== Results (3-endpoint pool, spacing={INTER_CALL_SLEEP_S}s) ===",
          flush=True)
    print(f"{'sz':>3} {'cur':>5} {'n':>4} {'p50':>6} {'p90':>6} {'p95':>6} "
          f"{'p99':>6} {'max':>6} {'fail':>4} | {'rec':>5} {'delta':>7}",
          flush=True)
    for bs in BATCH_SIZES:
        ws = winner_wall[bs]
        n = len(ws)
        p99 = pct(ws, 99)
        cur = CURRENT_TABLE[bs]
        # Recommendation: round p99 up to nearest int; flag drift > 15%.
        rec_val = int(round(p99)) if n else None
        delta = (rec_val - cur) if rec_val is not None else None
        drift = (p99 > cur * 1.15) if n else False
        summary["per_size"][bs] = {
            "n": n, "n_fail": failures[bs],
            "p50_ms": pct(ws, 50), "p90_ms": pct(ws, 90),
            "p95_ms": pct(ws, 95), "p99_ms": p99,
            "max_ms": max(ws) if ws else 0,
            "current_table_ms": cur,
            "recommended_ms": rec_val,
            "delta_ms": delta,
            "material_drift_gt_15pct": drift,
            "winner_internal_p99_ms": pct(winner_internal[bs], 99),
        }
        print(f"{bs:>3} {cur:>5} {n:>4} {pct(ws,50):>6.0f} {pct(ws,90):>6.0f} "
              f"{pct(ws,95):>6.0f} {p99:>6.0f} {max(ws) if ws else 0:>6.0f} "
              f"{failures[bs]:>4} | "
              f"{rec_val if rec_val is not None else 0:>5} "
              f"{delta if delta is not None else 0:>+7}", flush=True)

    print(f"\n=== Per-endpoint RTT (all successful per-future responses) ===",
          flush=True)
    for ep in ENDPOINTS:
        rs = per_ep_rtts[ep]
        summary["per_endpoint"][ep] = {
            "n_success": len(rs),
            "p50_ms": pct(rs, 50), "p95_ms": pct(rs, 95),
            "p99_ms": pct(rs, 99), "max_ms": max(rs) if rs else 0,
        }
        print(f"  {ep:>42} n={len(rs):>4} p50={pct(rs,50):>6.0f} "
              f"p95={pct(rs,95):>6.0f} p99={pct(rs,99):>6.0f} "
              f"max={max(rs) if rs else 0:>6.0f}", flush=True)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {SUMMARY_PATH}", flush=True)
    print(f"wrote {JSONL_PATH}", flush=True)

    EXECUTOR.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
