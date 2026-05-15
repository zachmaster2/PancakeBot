"""I4 probe — empirical comparison of batch_size for receipts-only batches.

Compares per-batch RTT for batch_size in {20, 50, 100} with receipts-only
shape (the I3-recommended layout). Goal: find the optimal batch_size for
cold-start backfill.

Methodology:
- Same endpoint pool + transport as production (urllib3 PoolManager, fire-to-all).
- Spaced calls (5s inter-call). Conservative pacing so we don't trigger
  rate limits that would affect the running bot.
- Round-robin across batch sizes (20, 50, 100, 20, 50, 100, ...).
- N=10 per size = 30 samples, ~2.5 min runtime.
- Tracks blocks-per-second (blocks_in_batch / rtt_seconds) — the true
  catchup throughput metric.

Output: ``var/i4_batch_size_2026_05_14.json`` summary + JSONL per call.

Caveats:
- Bigger batches stress endpoints more; some endpoints may reject >50
  sub-calls. The probe records errors so we know which endpoints
  fail at which size.
- 5s pacing avoids burst-regime measurement; cold-start burst is
  separate (would require stopping the bot).
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

_BATCH_SIZES = (20, 50, 100)
_N_PER_SIZE = 10
_INTERVAL_S = 5.0
_TIMEOUT_S = 10.0  # bigger batches need more headroom

_pool = urllib3.PoolManager(
    num_pools=len(_ENDPOINTS),
    maxsize=len(_ENDPOINTS),
    headers={
        "User-Agent": "probe-batch-size/1.0",
        "Content-Type": "application/json",
    },
)
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=3 * len(_ENDPOINTS),
    thread_name_prefix="probe-bs-hedge",
)


def _rpc_post(url: str, body: bytes) -> bytes:
    resp = _pool.request("POST", url, body=body, timeout=_TIMEOUT_S, retries=False)
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}")
    return resp.data


def _hedged_batch(body: bytes) -> tuple[str, float, dict[str, str]]:
    t_start = time.monotonic()
    fut_to_ep = {_executor.submit(_rpc_post, ep, body): ep for ep in _ENDPOINTS}
    pending = set(fut_to_ep.keys())
    per_ep_errors: dict[str, str] = {}
    deadline = t_start + _TIMEOUT_S
    while pending:
        remaining = max(0.001, deadline - time.monotonic())
        done, pending = concurrent.futures.wait(
            pending, timeout=remaining,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            for fut in pending:
                per_ep_errors[fut_to_ep[fut]] = "timeout"
            raise RuntimeError(f"hedged_timeout; per_ep_errors={per_ep_errors}")
        for fut in done:
            ep = fut_to_ep[fut]
            try:
                fut.result()
                rtt = time.monotonic() - t_start
                return ep, rtt, per_ep_errors
            except Exception as e:
                per_ep_errors[ep] = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"all_failed; per_ep_errors={per_ep_errors}")


def _build_receipts_only(start_block: int, n: int) -> bytes:
    batch = [
        {"jsonrpc": "2.0", "id": j, "method": "eth_getBlockReceipts",
         "params": [hex(start_block + j)]}
        for j in range(n)
    ]
    return json.dumps(batch).encode()


def _get_latest_block() -> int:
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": [],
    }).encode()
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
    jsonl_path = out_dir / "i4_batch_size_2026_05_14.jsonl"
    summary_path = out_dir / "i4_batch_size_2026_05_14.json"

    head = _get_latest_block()
    print(f"head={head}; sizes={_BATCH_SIZES}; N={_N_PER_SIZE} per size",
          flush=True)

    samples: dict[int, list[float]] = {bs: [] for bs in _BATCH_SIZES}
    failures: dict[int, list[dict]] = {bs: [] for bs in _BATCH_SIZES}
    sample_idx = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(_N_PER_SIZE * len(_BATCH_SIZES)):
            bs = _BATCH_SIZES[i % len(_BATCH_SIZES)]
            start = head - 200 - sample_idx * 100
            sample_idx += 1
            body = _build_receipts_only(start, bs)
            t0 = time.monotonic()
            try:
                ep, rtt, per_ep_errors = _hedged_batch(body)
                rtt_ms = rtt * 1000.0
                rec = {
                    "ts": time.time(), "batch_size": bs, "start_block": start,
                    "ok": True, "rtt_ms": rtt_ms, "winner": ep,
                    "per_ep_errors": per_ep_errors,
                    "blocks_per_sec": bs / rtt,
                }
                samples[bs].append(rtt_ms)
                print(f"[{i+1}/{_N_PER_SIZE*len(_BATCH_SIZES)}] bs={bs} "
                      f"rtt={rtt_ms:.0f}ms bps={bs/rtt:.0f} ep={ep}",
                      flush=True)
            except Exception as e:
                rec = {
                    "ts": time.time(), "batch_size": bs, "start_block": start,
                    "ok": False, "err": f"{type(e).__name__}: {e}",
                }
                failures[bs].append(rec)
                print(f"[{i+1}/{_N_PER_SIZE*len(_BATCH_SIZES)}] bs={bs} ERR {rec['err']}",
                      flush=True)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            spent = time.monotonic() - t0
            time.sleep(max(0.0, _INTERVAL_S - spent))

    def _pct(xs: list[float], p: float) -> float:
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        if n == 0:
            return 0.0
        return xs_sorted[min(n - 1, int(n * p))]

    summary: dict = {"per_size": {}}
    for bs in _BATCH_SIZES:
        xs = samples[bs]
        if not xs:
            summary["per_size"][bs] = {"n": 0, "n_failures": len(failures[bs])}
            continue
        summary["per_size"][bs] = {
            "n": len(xs),
            "n_failures": len(failures[bs]),
            "mean_ms": sum(xs) / len(xs),
            "p50_ms": _pct(xs, 0.50),
            "p95_ms": _pct(xs, 0.95),
            "p99_ms": _pct(xs, 0.99),
            "max_ms": max(xs),
            "mean_blocks_per_sec": bs / (sum(xs) / len(xs) / 1000.0),
        }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
