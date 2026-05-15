"""I3 probe — empirical comparison of batch shapes.

Compares per-batch RTT for two shapes under fire-to-all-pool transport:

  A) RECEIPTS_ONLY: 20 × eth_getBlockReceipts  (= 20 sub-calls)
  B) RECEIPTS_PLUS_HEADERS: 20 × eth_getBlockReceipts + 20 × eth_getBlockByNumber  (= 40 sub-calls)

Methodology:
- Same endpoint pool + transport as production (urllib3 PoolManager, fire-to-all,
  first-success-wins).
- Spaced calls (5s inter-call) to stay clean of rate-limiting (the original
  2026-05-11 probe used 30s; we use 5s because the production bot already
  polls every 8s, so 5s sampling is in the same regime).
- Alternate A/B/A/B... so any temporal drift in endpoint performance
  affects both shapes equally.
- Sample size: 20 per shape (40 total samples, ~3.5 min runtime).
- Reports p50/p95/p99/max for each shape.

Output: ``var/i3_batch_shape_compare_2026_05_14.json`` + JSONL of every call.

Caveats:
- The production bot is also polling these endpoints every 8s. Some overlap is
  unavoidable. The 5s sampling cadence means the probe and bot interleave
  at random phase.
- This is an ISOLATED-call measurement (one batch every 5s, not back-to-back
  burst). It does NOT capture the burst regime that gave the 3122 ms/batch
  cold-start. For that, we'd need to stop the bot and run back-to-back.
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

import urllib3


_ENDPOINTS: tuple[str, ...] = (
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-dataseed1.ninicoin.io",
    "https://bsc-dataseed1.binance.org",
    "https://bsc-dataseed3.binance.org",
    "https://bsc-rpc.publicnode.com",
    "https://bsc.rpc.blxrbdn.com",
)

_BATCH_SIZE = 20
_N_PER_SHAPE = 20
_INTERVAL_S = 5.0
_TIMEOUT_S = 5.0

_pool = urllib3.PoolManager(
    num_pools=len(_ENDPOINTS),
    maxsize=len(_ENDPOINTS),
    headers={
        "User-Agent": "probe-batch-shape/1.0",
        "Content-Type": "application/json",
    },
)
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=3 * len(_ENDPOINTS),
    thread_name_prefix="probe-hedge",
)


def _rpc_post(url: str, body: bytes) -> bytes:
    resp = _pool.request("POST", url, body=body, timeout=_TIMEOUT_S, retries=False)
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}")
    return resp.data


def _hedged_batch(body: bytes) -> tuple[str, float]:
    """Returns (winner_endpoint, rtt_seconds). Raises on all-failed."""
    t_start = time.monotonic()
    fut_to_ep = {_executor.submit(_rpc_post, ep, body): ep for ep in _ENDPOINTS}
    pending = set(fut_to_ep.keys())
    errors: list[tuple[str, str]] = []
    deadline = t_start + _TIMEOUT_S
    while pending:
        remaining = max(0.001, deadline - time.monotonic())
        done, pending = concurrent.futures.wait(
            pending, timeout=remaining,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            raise RuntimeError(f"hedged_timeout; errors={errors}")
        for fut in done:
            ep = fut_to_ep[fut]
            try:
                fut.result()
                rtt = time.monotonic() - t_start
                return ep, rtt
            except Exception as e:
                errors.append((ep, f"{type(e).__name__}: {e}"))
    raise RuntimeError(f"all_failed; errors={errors}")


def _build_receipts_only(start_block: int) -> bytes:
    batch = [
        {"jsonrpc": "2.0", "id": j, "method": "eth_getBlockReceipts",
         "params": [hex(start_block + j)]}
        for j in range(_BATCH_SIZE)
    ]
    return json.dumps(batch).encode()


def _build_receipts_plus_headers(start_block: int) -> bytes:
    batch: list[dict] = []
    next_id = 0
    for j in range(_BATCH_SIZE):
        bn = start_block + j
        batch.append({
            "jsonrpc": "2.0", "id": next_id, "method": "eth_getBlockReceipts",
            "params": [hex(bn)],
        })
        next_id += 1
        batch.append({
            "jsonrpc": "2.0", "id": next_id, "method": "eth_getBlockByNumber",
            "params": [hex(bn), False],
        })
        next_id += 1
    return json.dumps(batch).encode()


def _get_latest_block() -> int:
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": [],
    }).encode()
    _ep, _rtt = _hedged_batch(body)
    # _hedged_batch returns on first success; we need the block number.
    # Re-fire a single call to get the result.
    for ep in _ENDPOINTS:
        try:
            resp = _rpc_post(ep, body)
            payload = json.loads(resp)
            return int(payload["result"], 16)
        except Exception:
            continue
    raise RuntimeError("could not fetch head")


def main() -> int:
    out_dir = Path("var")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "i3_batch_shape_compare_2026_05_14.jsonl"
    summary_path = out_dir / "i3_batch_shape_compare_2026_05_14.json"

    head = _get_latest_block()
    print(f"head={head}; running {_N_PER_SHAPE} samples per shape, "
          f"{_INTERVAL_S}s interval", flush=True)

    samples_a: list[float] = []  # receipts-only
    samples_b: list[float] = []  # receipts+headers
    records: list[dict] = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(2 * _N_PER_SHAPE):
            shape = "A" if (i % 2 == 0) else "B"
            # Slide the block range so we don't re-hit the same blocks
            # (provider caches identical responses).
            start = head - 200 - i * _BATCH_SIZE
            if shape == "A":
                body = _build_receipts_only(start)
            else:
                body = _build_receipts_plus_headers(start)
            t0 = time.monotonic()
            try:
                ep, rtt = _hedged_batch(body)
                rtt_ms = rtt * 1000.0
                rec = {
                    "ts": time.time(), "shape": shape, "start_block": start,
                    "ok": True, "rtt_ms": rtt_ms, "winner": ep,
                }
                (samples_a if shape == "A" else samples_b).append(rtt_ms)
                print(f"[{i+1}/{2*_N_PER_SHAPE}] shape={shape} rtt={rtt_ms:.0f}ms ep={ep}",
                      flush=True)
            except Exception as e:
                rec = {
                    "ts": time.time(), "shape": shape, "start_block": start,
                    "ok": False, "err": f"{type(e).__name__}: {e}",
                }
                print(f"[{i+1}/{2*_N_PER_SHAPE}] shape={shape} ERR {rec['err']}",
                      flush=True)
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            # Pace.
            spent = time.monotonic() - t0
            time.sleep(max(0.0, _INTERVAL_S - spent))

    def _pct(xs: list[float], p: float) -> float:
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        if n == 0:
            return 0.0
        return xs_sorted[min(n - 1, int(n * p))]

    def _summary(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean_ms": sum(xs) / len(xs),
            "p50_ms": _pct(xs, 0.50),
            "p95_ms": _pct(xs, 0.95),
            "p99_ms": _pct(xs, 0.99),
            "max_ms": max(xs),
        }

    summary = {
        "n_per_shape": _N_PER_SHAPE,
        "interval_s": _INTERVAL_S,
        "batch_size": _BATCH_SIZE,
        "receipts_only_A": _summary(samples_a),
        "receipts_plus_headers_B": _summary(samples_b),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
