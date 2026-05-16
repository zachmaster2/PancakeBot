"""Spike: validate eth_getBlockReceipts batched-RPC viability for the
WSS->RPC polling architecture pivot (2026-05-07).

Tests, on each candidate free RPC (drpc.org, publicnode):

1. Batching support: send batched JSON-RPC arrays of size 200, 100,
   50, 25, and 10 with eth_getBlockReceipts sub-calls. Verify each
   endpoint returns valid responses (no truncation, no out-of-order
   ids). Identify the highest reliable batch size.

2. RPC_BATCH_RECEIPTS_RTT_P99_MS: time 200 batched calls (each
   contains the locked batch size). p99 latency.

3. RPC_BLOCK_AVAILABILITY_DELAY_P99_MS: subscribe to newHeads via
   WSS; immediately on newHead arrival, query
   eth_getBlockReceipts(blockHash=X) over HTTP. Measure delay
   newhead_arrival -> first successful receipt fetch.

4. Poll-consistency sanity: poll same block range twice (10s apart),
   verify identical bet-event sets returned.

5. Reorg-frequency sanity: sliding window of (block_number,
   block_hash); periodically re-fetch eth_getBlockByNumber(N) and
   compare hashes. Document observed reorg events over the spike.

Pass criteria:
  - >=99% batch success at the locked size
  - 100% poll consistency
  - p99 latency < 2s (otherwise wake derivation gets uncomfortable)

Usage: python research/probe_rpc_polling.py
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
import urllib.error
import urllib.request as _urllib_req

PREDICTION_V2 = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA".lower()
BET_BULL_TOPIC = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
BET_BEAR_TOPIC = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"

ENDPOINTS_HTTP = [
    ("drpc.org", "https://bsc.drpc.org"),
    ("publicnode", "https://bsc-rpc.publicnode.com"),
]
ENDPOINTS_WSS = [
    ("drpc.org", "wss://bsc.drpc.org"),
    ("publicnode", "wss://bsc.publicnode.com"),
]

HTTP_TIMEOUT = 30
USER_AGENT = "pancakebot-spike/0.2"


def rpc_call_single(rpc: str, method: str, params: list, *, timeout: int = HTTP_TIMEOUT):
    """Single JSON-RPC call. Returns (latency_ms, result_or_None, error_str)."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    t0 = time.time()
    try:
        resp = _urllib_req.urlopen(
            _urllib_req.Request(
                rpc, data=body,
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            ),
            timeout=timeout,
        )
        payload = json.loads(resp.read())
        latency_ms = (time.time() - t0) * 1000
        if "error" in payload:
            return latency_ms, None, f"rpc_error:{payload['error']}"
        return latency_ms, payload.get("result"), None
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        return latency_ms, None, f"{type(e).__name__}:{e}"


def rpc_call_batched(rpc: str, calls: list[tuple[str, list]], *, timeout: int = HTTP_TIMEOUT):
    """Batched JSON-RPC. Returns (latency_ms, list_of_(result,error), top_level_error_or_None).

    Each element of returned list is parallel to input calls."""
    batch = [
        {"jsonrpc": "2.0", "id": i, "method": method, "params": params}
        for i, (method, params) in enumerate(calls)
    ]
    body = json.dumps(batch).encode()
    t0 = time.time()
    try:
        resp = _urllib_req.urlopen(
            _urllib_req.Request(
                rpc, data=body,
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            ),
            timeout=timeout,
        )
        payload = json.loads(resp.read())
        latency_ms = (time.time() - t0) * 1000
        if not isinstance(payload, list):
            return latency_ms, [], f"non_list_response:type={type(payload).__name__}"
        # Verify all expected ids returned, no truncation, no extras
        ids_returned = sorted(r.get("id", -1) for r in payload)
        ids_expected = list(range(len(calls)))
        if ids_returned != ids_expected:
            missing = set(ids_expected) - set(ids_returned)
            extras = set(ids_returned) - set(ids_expected)
            return latency_ms, [], (
                f"id_mismatch: missing={sorted(missing)} extras={sorted(extras)} "
                f"len_returned={len(payload)} expected={len(calls)}"
            )
        # Build aligned result list (sorted by id)
        by_id = {r["id"]: r for r in payload}
        results: list[tuple[object, str | None]] = []
        for i in range(len(calls)):
            r = by_id[i]
            if "error" in r:
                results.append((None, f"rpc_error:{r['error']}"))
            else:
                results.append((r.get("result"), None))
        return latency_ms, results, None
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        return latency_ms, [], f"{type(e).__name__}:{e}"


def collect_block_hashes(rpc: str, n: int) -> list[tuple[int, str]]:
    """Return [(block_number, block_hash), ...] for n recent blocks."""
    _, head_hex, err = rpc_call_single(rpc, "eth_blockNumber", [])
    if err is not None or not isinstance(head_hex, str):
        print(f"FATAL: {rpc} eth_blockNumber failed: {err}")
        sys.exit(1)
    head = int(head_hex, 16)
    print(f"[xref] head={head} on {rpc}")
    out: list[tuple[int, str]] = []
    # Take blocks from head-5 backwards. Be generous: try 2x the requested
    # count to absorb endpoint-side block-fetch failures.
    candidates = list(range(head - 2 * n - 5, head - 5))
    for bn in candidates:
        if len(out) >= n:
            break
        _, blk, err = rpc_call_single(rpc, "eth_getBlockByNumber", [hex(bn), False])
        if err is None and isinstance(blk, dict) and "hash" in blk:
            out.append((bn, blk["hash"]))
    return out


# ---------------------------------------------------------------------------
# Test 1: batching support
# ---------------------------------------------------------------------------

def test_batching_support(label: str, rpc: str, blocks: list[tuple[int, str]]) -> dict:
    """For batch sizes 200, 100, 50, 25, 10 — send N batches each, verify
    success rate and id integrity. Returns summary dict."""
    print(f"\n[{label}] Test 1: batching support")
    sizes = [200, 100, 50, 25, 10]
    results = {}
    for size in sizes:
        if len(blocks) < size:
            print(f"  size={size:3d}: SKIP (not enough blocks: {len(blocks)})")
            continue
        # Send 5 trial batches of this size, drawing from front of blocks list
        n_trials = 5
        latencies = []
        successes = 0
        errors = []
        for trial in range(n_trials):
            chunk = blocks[trial * size: (trial + 1) * size] if size * (trial + 1) <= len(blocks) else blocks[:size]
            calls = [("eth_getBlockReceipts", [bh]) for _, bh in chunk]
            latency_ms, batch_results, top_err = rpc_call_batched(rpc, calls)
            if top_err is not None:
                errors.append(top_err)
                continue
            # Each sub-call should have either a list-of-receipts or an error.
            sub_errors = [e for _, e in batch_results if e is not None]
            if sub_errors:
                errors.append(f"sub_errors={len(sub_errors)}/{len(batch_results)}: {sub_errors[:2]}")
                continue
            successes += 1
            latencies.append(latency_ms)
        success_rate = successes / n_trials if n_trials else 0
        p50 = statistics.median(latencies) if latencies else 0
        p_max = max(latencies) if latencies else 0
        print(f"  size={size:3d}: success={success_rate:.0%} ({successes}/{n_trials})  "
              f"p50={p50:.0f}ms  max={p_max:.0f}ms  errors={errors[:1]}")
        results[size] = {
            "success_rate": success_rate,
            "p50_ms": p50,
            "max_ms": p_max,
            "errors": errors,
        }
    return results


# ---------------------------------------------------------------------------
# Test 2: RTT p99 at the locked batch size
# ---------------------------------------------------------------------------

def test_rtt_p99(label: str, rpc: str, blocks: list[tuple[int, str]],
                 batch_size: int, n_batches: int = 200) -> dict:
    """Run n_batches batched calls of batch_size each. Report p50/p95/p99."""
    print(f"\n[{label}] Test 2: RTT p99 at batch_size={batch_size} over {n_batches} batches")
    # Cycle through blocks for variety
    if len(blocks) < batch_size:
        print(f"  not enough blocks ({len(blocks)} < {batch_size}); SKIP")
        return {}
    latencies = []
    failures = 0
    for trial in range(n_batches):
        # Pick batch_size blocks starting at offset
        offset = (trial * 7) % max(1, len(blocks) - batch_size)
        chunk = blocks[offset: offset + batch_size]
        calls = [("eth_getBlockReceipts", [bh]) for _, bh in chunk]
        latency_ms, batch_results, top_err = rpc_call_batched(rpc, calls)
        if top_err is not None or any(e for _, e in batch_results):
            failures += 1
            continue
        latencies.append(latency_ms)
        if (trial + 1) % 50 == 0:
            print(f"  progress: {trial + 1}/{n_batches}")
    if not latencies:
        print(f"  no successful batches; cannot compute p99")
        return {"failures": failures, "n_batches": n_batches}
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
    print(f"  batches: {len(latencies)}/{n_batches}  failures: {failures}")
    print(f"  latency p50/p95/p99: {p50:.0f}ms / {p95:.0f}ms / {p99:.0f}ms")
    return {
        "batch_size": batch_size,
        "n_batches": n_batches,
        "n_success": len(latencies),
        "n_fail": failures,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


# ---------------------------------------------------------------------------
# Test 2b: RTT curve across batch sizes [2, 5, 10, 15, 20]
# ---------------------------------------------------------------------------

def test_rtt_curve(label: str, rpc: str, blocks: list[tuple[int, str]],
                   n_samples_per_size: int = 50) -> dict:
    """Measure RTT p50/p95/p99 for batch sizes 2, 5, 10, 15, 20.

    Drives the final-poll wake derivation: smaller batches (final poll
    needs only the few blocks since last periodic poll) have smaller
    RTT and tighter budget.
    """
    print(f"\n[{label}] Test 2b: RTT curve across batch sizes (n={n_samples_per_size} per size)")
    sizes = [2, 5, 10, 15, 20]
    curve: dict[int, dict] = {}
    for size in sizes:
        if len(blocks) < size:
            print(f"  size={size:2d}: SKIP (not enough blocks)")
            continue
        latencies = []
        failures = 0
        for trial in range(n_samples_per_size):
            offset = (trial * 7) % max(1, len(blocks) - size)
            chunk = blocks[offset: offset + size]
            calls = [("eth_getBlockReceipts", [bh]) for _, bh in chunk]
            latency_ms, batch_results, top_err = rpc_call_batched(rpc, calls)
            if top_err is not None or any(e for _, e in batch_results):
                failures += 1
                continue
            latencies.append(latency_ms)
        if not latencies:
            print(f"  size={size:2d}: ALL FAILED ({failures}/{n_samples_per_size})")
            curve[size] = {"failed": True, "failures": failures}
            continue
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
        print(f"  size={size:2d}: p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms  "
              f"({len(latencies)}/{n_samples_per_size} ok)")
        curve[size] = {
            "n_success": len(latencies),
            "n_fail": failures,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
        }
    return curve


# ---------------------------------------------------------------------------
# Test 3: block availability delay (newhead -> receipt) using WSS
# ---------------------------------------------------------------------------

async def test_block_availability_delay(label_wss: str, url_wss: str,
                                         label_http: str, url_http: str,
                                         duration_s: int = 90) -> dict:
    """Subscribe newHeads on WSS. On each arrival, time how long until
    eth_getBlockReceipts(blockHash) returns a non-error result via HTTP.
    Returns p50/p95/p99 of those delays."""
    import websockets

    print(f"\n[{label_wss}] Test 3: block availability delay ({duration_s}s)")
    delays_ms = []
    n_newheads = 0
    n_misses = 0  # newheads where receipt fetch failed even after retry

    async with websockets.connect(
        url_wss, ping_interval=30, ping_timeout=10, open_timeout=15,
        max_size=None, max_queue=None,
    ) as ws:
        await ws.send(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "eth_subscribe", "params": ["newHeads"],
        }))
        sub = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if "result" not in sub:
            print(f"  newHeads subscribe failed: {sub}")
            return {}
        deadline = time.time() + duration_s
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=duration_s)
            except asyncio.TimeoutError:
                break
            t_arrival = time.time()
            msg = json.loads(raw)
            params = msg.get("params") or {}
            result = params.get("result")
            if not isinstance(result, dict) or "hash" not in result:
                continue
            block_hash = result["hash"]
            n_newheads += 1
            # Synchronously poll receipt up to 5 retries (200ms apart)
            success = False
            for attempt in range(5):
                _, receipts, err = rpc_call_single(url_http, "eth_getBlockReceipts", [block_hash])
                if err is None and isinstance(receipts, list):
                    delay_ms = (time.time() - t_arrival) * 1000
                    delays_ms.append(delay_ms)
                    success = True
                    break
                await asyncio.sleep(0.2)
            if not success:
                n_misses += 1
            if n_newheads % 20 == 0:
                print(f"  progress: {n_newheads} newheads, {len(delays_ms)} hits, {n_misses} misses")

    if not delays_ms:
        print(f"  no successful availability measurements")
        return {"newheads": n_newheads, "misses": n_misses}
    p50 = statistics.median(delays_ms)
    p95 = statistics.quantiles(delays_ms, n=20)[18] if len(delays_ms) >= 20 else max(delays_ms)
    p99 = statistics.quantiles(delays_ms, n=100)[98] if len(delays_ms) >= 100 else max(delays_ms)
    print(f"  newheads={n_newheads} hits={len(delays_ms)} misses={n_misses}")
    print(f"  delay p50/p95/p99: {p50:.0f}ms / {p95:.0f}ms / {p99:.0f}ms")
    return {
        "newheads": n_newheads,
        "hits": len(delays_ms),
        "misses": n_misses,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


# ---------------------------------------------------------------------------
# Test 4: poll-consistency (sanity)
# ---------------------------------------------------------------------------

def test_poll_consistency(label: str, rpc: str, blocks: list[tuple[int, str]],
                           batch_size: int) -> dict:
    """Poll same block range twice (10s apart). Same bet-event set both times?"""
    print(f"\n[{label}] Test 4: poll-consistency")
    if len(blocks) < batch_size:
        print(f"  insufficient blocks; SKIP")
        return {}
    chunk = blocks[:batch_size]
    calls = [("eth_getBlockReceipts", [bh]) for _, bh in chunk]
    _, results_a, err_a = rpc_call_batched(rpc, calls)
    if err_a is not None:
        print(f"  poll1 failed: {err_a}")
        return {}
    print(f"  poll1 done. waiting 10s before poll2...")
    time.sleep(10)
    _, results_b, err_b = rpc_call_batched(rpc, calls)
    if err_b is not None:
        print(f"  poll2 failed: {err_b}")
        return {}
    keys_a = set()
    keys_b = set()
    for receipts, _ in results_a:
        if not isinstance(receipts, list):
            continue
        for r in receipts:
            for log in r.get("logs", []) or []:
                if (log.get("address") or "").lower() != PREDICTION_V2:
                    continue
                topics = log.get("topics") or []
                if not topics:
                    continue
                if topics[0] in (BET_BULL_TOPIC, BET_BEAR_TOPIC):
                    keys_a.add((r["transactionHash"], int(log["logIndex"], 16)))
    for receipts, _ in results_b:
        if not isinstance(receipts, list):
            continue
        for r in receipts:
            for log in r.get("logs", []) or []:
                if (log.get("address") or "").lower() != PREDICTION_V2:
                    continue
                topics = log.get("topics") or []
                if not topics:
                    continue
                if topics[0] in (BET_BULL_TOPIC, BET_BEAR_TOPIC):
                    keys_b.add((r["transactionHash"], int(log["logIndex"], 16)))
    consistent = keys_a == keys_b
    print(f"  poll1={len(keys_a)} bet events; poll2={len(keys_b)}; consistent={consistent}")
    return {"poll1_count": len(keys_a), "poll2_count": len(keys_b), "consistent": consistent}


# ---------------------------------------------------------------------------
# Test 5: reorg detection (sanity)
# ---------------------------------------------------------------------------

def test_reorg_detection(label: str, rpc: str, duration_s: int = 60) -> dict:
    """Maintain sliding window of (block_number, block_hash) for last 10 blocks.
    Every 10s, refetch eth_getBlockByNumber(N) and compare hashes."""
    print(f"\n[{label}] Test 5: reorg detection ({duration_s}s)")
    window: dict[int, str] = {}
    reorg_events = []
    deadline = time.time() + duration_s
    polls = 0
    while time.time() < deadline:
        # Fetch latest 10 blocks
        _, head_hex, err = rpc_call_single(rpc, "eth_blockNumber", [])
        if err is not None or not isinstance(head_hex, str):
            print(f"  poll failed: {err}")
            time.sleep(10)
            continue
        head = int(head_hex, 16)
        polls += 1
        for bn in range(head - 9, head + 1):
            _, blk, err = rpc_call_single(rpc, "eth_getBlockByNumber", [hex(bn), False])
            if err is not None or not isinstance(blk, dict):
                continue
            new_hash = blk["hash"]
            old_hash = window.get(bn)
            if old_hash and old_hash != new_hash:
                reorg_events.append({
                    "block": bn,
                    "old_hash": old_hash,
                    "new_hash": new_hash,
                })
                print(f"  REORG detected at block {bn}: {old_hash[:10]}... -> {new_hash[:10]}...")
            window[bn] = new_hash
        # Trim window to latest 10
        if len(window) > 30:
            for k in sorted(window.keys())[:-30]:
                del window[k]
        time.sleep(10)
    print(f"  polls={polls} reorgs={len(reorg_events)}")
    return {"polls": polls, "reorg_events": reorg_events}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print(f"=== Phase 0 spike: RPC polling viability ===")
    summaries: dict[str, dict] = {}

    for (label_h, rpc_http), (label_w, url_wss) in zip(ENDPOINTS_HTTP, ENDPOINTS_WSS):
        assert label_h == label_w
        print(f"\n{'='*60}\n{label_h} ({rpc_http} / {url_wss})\n{'='*60}")
        # Collect a big block list once
        blocks = collect_block_hashes(rpc_http, n=300)
        print(f"  collected {len(blocks)} block hashes")
        if len(blocks) < 150:
            print(f"  insufficient block hashes (<150); skipping endpoint")
            continue

        s_batching = test_batching_support(label_h, rpc_http, blocks)
        # Lock batch_size: largest size where success_rate == 100%
        locked_batch_size = 0
        for size in sorted(s_batching.keys(), reverse=True):
            if s_batching[size]["success_rate"] >= 1.0:
                locked_batch_size = size
                break
        print(f"\n[{label_h}] LOCKED batch_size={locked_batch_size}")

        s_rtt = {}
        s_curve = {}
        s_consistency = {}
        if locked_batch_size > 0:
            s_rtt = test_rtt_p99(label_h, rpc_http, blocks, locked_batch_size, n_batches=100)
            s_curve = test_rtt_curve(label_h, rpc_http, blocks, n_samples_per_size=50)
            s_consistency = test_poll_consistency(label_h, rpc_http, blocks, locked_batch_size)

        s_availability = await test_block_availability_delay(
            label_w, url_wss, label_h, rpc_http, duration_s=60,
        )
        s_reorg = test_reorg_detection(label_h, rpc_http, duration_s=60)

        summaries[label_h] = {
            "batching": s_batching,
            "locked_batch_size": locked_batch_size,
            "rtt": s_rtt,
            "rtt_curve": s_curve,
            "consistency": s_consistency,
            "availability": s_availability,
            "reorg": s_reorg,
        }

    # Final report
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for label, s in summaries.items():
        print(f"\n{label}:")
        print(f"  batch_size_locked: {s['locked_batch_size']}")
        if s["rtt"]:
            print(f"  rtt p50/p95/p99: {s['rtt']['p50_ms']:.0f}/{s['rtt']['p95_ms']:.0f}/{s['rtt']['p99_ms']:.0f}ms")
        if s["consistency"]:
            print(f"  poll consistency: {s['consistency']['consistent']} "
                  f"(poll1={s['consistency']['poll1_count']}, poll2={s['consistency']['poll2_count']})")
        if s["availability"]:
            a = s["availability"]
            print(f"  availability newheads={a.get('newheads')} hits={a.get('hits')} misses={a.get('misses')}")
            if "p99_ms" in a:
                print(f"  availability p50/p95/p99: {a['p50_ms']:.0f}/{a['p95_ms']:.0f}/{a['p99_ms']:.0f}ms")
        print(f"  reorgs in 60s: {len(s['reorg'].get('reorg_events', []))} "
              f"(over {s['reorg'].get('polls', 0)} polls)")

    # RTT curve table for quick reading
    print(f"\n{'='*60}\nRTT curve (p99 ms by batch size)\n{'='*60}")
    print(f"{'endpoint':12s} | {'sz=2':>8s} | {'sz=5':>8s} | {'sz=10':>8s} | {'sz=15':>8s} | {'sz=20':>8s}")
    for label, s in summaries.items():
        curve = s.get("rtt_curve", {})
        cells = [f"{label:12s}"]
        for size in [2, 5, 10, 15, 20]:
            entry = curve.get(size, {})
            if "p99_ms" in entry:
                cells.append(f"{entry['p99_ms']:>8.0f}")
            else:
                cells.append(f"{'FAIL':>8s}")
        print(" | ".join(cells))

    # Persist JSON
    out_path = "/tmp/spike_rpc_polling.json"
    try:
        with open(out_path, "w") as f:
            json.dump(summaries, f, indent=2, default=str)
        print(f"\nSaved JSON: {out_path}")
    except Exception as e:
        print(f"  could not save JSON: {e}")


if __name__ == "__main__":
    asyncio.run(main())
