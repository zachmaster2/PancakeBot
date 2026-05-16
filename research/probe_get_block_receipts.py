"""Spike script: validate eth_getBlockReceipts reliability on free BSC RPCs.

Tests the assumption underpinning Approach A' in
var/design/proactive_wss_reconciliation_2026_05_06.md (rev 3): that
eth_getBlockReceipts works reliably on drpc.org and publicnode and
returns complete bet-event coverage for PredictionV2.

Method:
  1. Pick N recent blocks (default 100).
  2. For each block, on each endpoint:
     - Time the eth_getBlockReceipts call.
     - Decode bet-event count from receipts.
  3. Cross-reference with eth_getBlockByNumber(full_txs=True) +
     per-tx eth_getTransactionReceipt (the proven-reliable backfill
     primitive) for ground-truth bet-event count.
  4. Report per-endpoint: success rate, latency p50/p95/p99,
     log-count match-rate.

Pass criteria:
  - >99% success on each endpoint
  - 100% bet-event count match against the cross-reference

Usage: python research/probe_get_block_receipts.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request as _urllib_req

PREDICTION_V2 = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA".lower()
BET_BULL_TOPIC = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
BET_BEAR_TOPIC = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"

ENDPOINTS = [
    ("drpc.org", "https://bsc.drpc.org"),
    ("publicnode", "https://bsc-rpc.publicnode.com"),
]

# Cross-reference endpoint (the existing backfill stack uses these).
XREF = "https://bsc-dataseed1.defibit.io"

N_BLOCKS = 200
HTTP_TIMEOUT = 15


def rpc_call(rpc: str, method: str, params: list, *, timeout: int = HTTP_TIMEOUT):
    """Single JSON-RPC call. Returns (latency_ms, result_or_None, error_str)."""
    body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    ).encode()
    t0 = time.time()
    try:
        resp = _urllib_req.urlopen(
            _urllib_req.Request(
                rpc, data=body, headers={
                    "Content-Type": "application/json",
                    "User-Agent": "pancakebot-spike/0.1",
                }
            ),
            timeout=timeout,
        )
        payload = json.loads(resp.read())
        latency_ms = (time.time() - t0) * 1000
        if "error" in payload:
            return latency_ms, None, f"rpc_error:{payload['error']}"
        return latency_ms, payload.get("result"), None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        latency_ms = (time.time() - t0) * 1000
        return latency_ms, None, f"{type(e).__name__}:{e}"
    except Exception as e:  # noqa: BLE001
        latency_ms = (time.time() - t0) * 1000
        return latency_ms, None, f"{type(e).__name__}:{e}"


def rpc_batch(rpc: str, calls: list[tuple[str, list]], *, timeout: int = 30):
    """Batch JSON-RPC. Returns list of (result, error_str) parallel to calls."""
    batch = [
        {"jsonrpc": "2.0", "id": i, "method": method, "params": params}
        for i, (method, params) in enumerate(calls)
    ]
    body = json.dumps(batch).encode()
    try:
        resp = _urllib_req.urlopen(
            _urllib_req.Request(
                rpc, data=body, headers={
                    "Content-Type": "application/json",
                    "User-Agent": "pancakebot-spike/0.1",
                }
            ),
            timeout=timeout,
        )
        out = json.loads(resp.read())
        if not isinstance(out, list):
            return [(None, f"non_list_response")] * len(calls)
        by_id: dict[int, tuple] = {}
        for r in out:
            rid = r.get("id")
            if "error" in r:
                by_id[rid] = (None, f"rpc_error:{r['error']}")
            else:
                by_id[rid] = (r.get("result"), None)
        return [by_id.get(i, (None, "missing_id")) for i in range(len(calls))]
    except Exception as e:  # noqa: BLE001
        return [(None, f"{type(e).__name__}:{e}")] * len(calls)


def count_bet_events_in_receipts(receipts: list[dict]) -> int:
    """Count BetBull/BetBear logs from a list of tx receipts."""
    if not isinstance(receipts, list):
        return -1
    total = 0
    for r in receipts:
        if not isinstance(r, dict):
            continue
        for log in r.get("logs", []) or []:
            if (log.get("address") or "").lower() != PREDICTION_V2:
                continue
            topics = log.get("topics") or []
            if not topics:
                continue
            if topics[0] in (BET_BULL_TOPIC, BET_BEAR_TOPIC):
                total += 1
    return total


def count_bet_events_via_block_full_txs(rpc: str, block_number_hex: str) -> tuple[int, str | None]:
    """Cross-reference path: eth_getBlockByNumber(full_txs=True) +
    per-tx eth_getTransactionReceipt on PredictionV2-targeted txs.
    Returns (count, error_or_None)."""
    _, blk, err = rpc_call(rpc, "eth_getBlockByNumber", [block_number_hex, True])
    if err is not None or not isinstance(blk, dict):
        return -1, err or "block_fetch_failed"
    txs = blk.get("transactions") or []
    candidates = []
    for tx in txs:
        if not isinstance(tx, dict):
            continue
        to_addr = (tx.get("to") or "").lower()
        if to_addr != PREDICTION_V2:
            continue
        # value > 0 is the live filter; for cross-reference completeness
        # we still need to receipt-check zero-value txs (if any) since
        # the audit will see logs regardless of value. Include all.
        candidates.append(tx.get("hash"))
    if not candidates:
        return 0, None
    rcpt_calls = [("eth_getTransactionReceipt", [h]) for h in candidates]
    results = rpc_batch(rpc, rcpt_calls)
    # noinspection PyBroadException
    try:
        receipts = [r for r, e in results if e is None and isinstance(r, dict)]
    except Exception:
        return -1, "receipt_batch_parse_failed"
    return count_bet_events_in_receipts(receipts), None


def fetch_block_hashes(rpc: str, n: int) -> list[tuple[str, str]]:
    """Pick n random recent blocks. Returns [(block_number_hex, block_hash), ...]."""
    _, blk_num_hex, err = rpc_call(rpc, "eth_blockNumber", [])
    if err is not None or not isinstance(blk_num_hex, str):
        print(f"FATAL: cannot fetch current block from {rpc}: {err}")
        sys.exit(1)
    head = int(blk_num_hex, 16)
    # Pick from the last ~1200 blocks (covers ~2 rounds, 10 minutes); recent
    # enough that all endpoints should have it. Two rounds captures multiple
    # high-density bet windows before lock.
    candidates = random.sample(range(head - 1200, head - 50), n)
    out: list[tuple[str, str]] = []
    print(f"[xref] fetching {n} block hashes from {rpc} (head={head})")
    for i, bn in enumerate(candidates):
        _, blk, err = rpc_call(rpc, "eth_getBlockByNumber", [hex(bn), False])
        if err is not None or not isinstance(blk, dict):
            print(f"  skip block {bn}: {err}")
            continue
        out.append((hex(bn), blk["hash"]))
        if (i + 1) % 25 == 0:
            print(f"  progress: {i + 1}/{n}")
    return out


def probe_endpoint(label: str, rpc: str, blocks: list[tuple[str, str]]) -> dict:
    """Test eth_getBlockReceipts on `rpc` for each (bn_hex, hash) in blocks.
    Returns a dict with per-block results + summary stats."""
    results = []
    print(f"\n[{label}] probing {len(blocks)} blocks via {rpc}")
    for i, (bn_hex, blk_hash) in enumerate(blocks):
        latency_ms, receipts, err = rpc_call(rpc, "eth_getBlockReceipts", [blk_hash])
        n_bets = -1
        if err is None and receipts is not None:
            n_bets = count_bet_events_in_receipts(receipts)
        results.append(
            {
                "block_hex": bn_hex,
                "block_hash": blk_hash,
                "latency_ms": latency_ms,
                "error": err,
                "n_bets_receipts": n_bets,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  progress: {i + 1}/{len(blocks)}")
    return results


def summarize(label: str, results: list[dict], xref_counts: dict[str, int]) -> dict:
    n = len(results)
    successes = [r for r in results if r["error"] is None and r["n_bets_receipts"] >= 0]
    fails = n - len(successes)
    success_rate = len(successes) / n if n else 0.0

    latencies = [r["latency_ms"] for r in successes]
    latency_p50 = statistics.median(latencies) if latencies else 0.0
    latency_p95 = (
        statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0)
    )
    latency_p99 = (
        statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies, default=0.0)
    )

    matches = 0
    mismatches = []
    for r in successes:
        bn = r["block_hex"]
        xref = xref_counts.get(bn, -1)
        if xref < 0:
            continue
        if r["n_bets_receipts"] == xref:
            matches += 1
        else:
            mismatches.append(
                {"block_hex": bn, "endpoint_count": r["n_bets_receipts"], "xref_count": xref}
            )

    match_rate = matches / len(successes) if successes else 0.0
    error_samples = [r["error"] for r in results if r["error"] is not None][:5]

    print(f"\n=== {label} summary ===")
    print(f"  blocks tested:    {n}")
    print(f"  success rate:     {success_rate:.1%}  ({len(successes)}/{n})")
    print(f"  failures:         {fails}")
    print(f"  match rate:       {match_rate:.1%}  ({matches}/{len(successes)})")
    print(f"  mismatches:       {len(mismatches)}")
    print(f"  latency p50/p95/p99 ms: {latency_p50:.0f} / {latency_p95:.0f} / {latency_p99:.0f}")
    if error_samples:
        print(f"  error samples:    {error_samples}")
    if mismatches[:3]:
        print(f"  mismatch samples: {mismatches[:3]}")

    return {
        "label": label,
        "n_blocks": n,
        "n_success": len(successes),
        "n_fail": fails,
        "success_rate": success_rate,
        "match_rate": match_rate,
        "n_mismatch": len(mismatches),
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "latency_p99_ms": latency_p99,
        "error_samples": error_samples,
        "mismatch_samples": mismatches[:5],
    }


def main() -> None:
    random.seed(0xBADCAFE)  # deterministic

    print(f"=== Phase 0a spike: eth_getBlockReceipts viability ({N_BLOCKS} blocks) ===")
    print(f"xref endpoint: {XREF}")
    print(f"probe endpoints: {[e[1] for e in ENDPOINTS]}")

    # 1. Pick blocks via xref endpoint.
    blocks = fetch_block_hashes(XREF, N_BLOCKS)
    if len(blocks) < N_BLOCKS // 2:
        print(f"FATAL: only got {len(blocks)} blocks; xref endpoint failing")
        sys.exit(1)
    print(f"[xref] {len(blocks)} blocks selected")

    # 2. Build cross-reference bet counts (the gold standard).
    print(f"\n[xref] computing ground-truth bet counts via getBlockByNumber+receipts...")
    xref_counts: dict[str, int] = {}
    for i, (bn_hex, _) in enumerate(blocks):
        n_bets, err = count_bet_events_via_block_full_txs(XREF, bn_hex)
        if err is None:
            xref_counts[bn_hex] = n_bets
        if (i + 1) % 25 == 0:
            print(f"  progress: {i + 1}/{len(blocks)}")
    n_with_xref = len(xref_counts)
    n_with_bets = sum(1 for v in xref_counts.values() if v > 0)
    total_xref_bets = sum(xref_counts.values())
    print(f"\n[xref] {n_with_xref}/{len(blocks)} blocks have valid xref counts")
    print(f"[xref] {n_with_bets} blocks contain bet events ({total_xref_bets} total bets)")

    # 3. Probe each candidate endpoint.
    summaries = []
    for label, rpc in ENDPOINTS:
        results = probe_endpoint(label, rpc, blocks)
        s = summarize(label, results, xref_counts)
        summaries.append(s)

    # 4. Final verdict.
    print(f"\n=== VERDICT ===")
    pass_threshold_success = 0.99
    pass_threshold_match = 1.00
    overall_pass = True
    for s in summaries:
        ok = s["success_rate"] >= pass_threshold_success and s["match_rate"] >= pass_threshold_match
        verdict = "PASS" if ok else "FAIL"
        print(
            f"  {s['label']:12s}  success={s['success_rate']:.1%}  match={s['match_rate']:.1%}"
            f"  {verdict}"
        )
        overall_pass = overall_pass and ok

    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        sys.exit(2)


if __name__ == "__main__":
    main()
