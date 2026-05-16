"""I5-Phase1 probe — per-endpoint isolated RTT measurements.

Hits each of the 6 hedged endpoints individually (not hedged) with two
call shapes:
  A) ``eth_getBlockReceipts`` 20-block batched array (= production
     backfill batch shape, receipts-only)
  B) ``eth_getBlockByNumber('latest', false)`` single (= anchor poll shape)

For each (endpoint, shape) pair, fires N=15 samples at 6s spacing.
Records per-call RTT + outcome. Computes p50/p95/p99/max/failure-rate.

Spacing rationale: bot polls hedged-to-all every 8s. Probe at 6s
spacing per-endpoint means each endpoint sees bot+probe ≈ 17 calls/min
combined. Well under any plausible per-IP rate limit.

Sequential per-endpoint (not parallel) to isolate per-endpoint behavior
from cross-endpoint contention.

Total runtime: 6 endpoints × 2 shapes × 15 samples × 6s = ~18 min.
Output: var/i5_per_endpoint_2026_05_15.json + jsonl.
"""
from __future__ import annotations

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
_N_PER_PAIR = 15
_SPACING_S = 6.0
_TIMEOUT_S = 5.0

_pool = urllib3.PoolManager(
    num_pools=len(_ENDPOINTS),
    maxsize=len(_ENDPOINTS),
    headers={
        "User-Agent": "probe-per-endpoint/1.0",
        "Content-Type": "application/json",
    },
)


def _post(url: str, body: bytes) -> tuple[bool, float, str]:
    """Returns (success, rtt_seconds, err_str_or_empty)."""
    t = time.monotonic()
    try:
        resp = _pool.request("POST", url, body=body, timeout=_TIMEOUT_S, retries=False)
        rtt = time.monotonic() - t
        if resp.status != 200:
            return False, rtt, f"HTTP {resp.status}"
        return True, rtt, ""
    except Exception as e:
        return False, time.monotonic() - t, f"{type(e).__name__}: {e}"


def _build_receipts_batch(start_block: int) -> bytes:
    return json.dumps([
        {"jsonrpc": "2.0", "id": j, "method": "eth_getBlockReceipts",
         "params": [hex(start_block + j)]}
        for j in range(_BATCH_SIZE)
    ]).encode()


def _build_anchor_call() -> bytes:
    return json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber",
        "params": ["latest", False],
    }).encode()


def _get_head() -> int:
    """Get current head via any endpoint that responds."""
    body = _build_anchor_call()
    for ep in _ENDPOINTS:
        ok, _rtt, _err = _post(ep, body)
        if ok:
            resp = _pool.request("POST", ep, body=body, timeout=_TIMEOUT_S, retries=False)
            payload = json.loads(resp.data)
            return int(payload["result"]["number"], 16)
    raise RuntimeError("could not fetch head from any endpoint")


def main() -> int:
    out_dir = Path("var")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "i5_per_endpoint_2026_05_15.jsonl"
    summary_path = out_dir / "i5_per_endpoint_2026_05_15.json"

    head = _get_head()
    print(f"head={head}; {_N_PER_PAIR} samples per (endpoint, shape); "
          f"{_SPACING_S}s spacing")

    samples: dict = {ep: {"batch": [], "anchor": [], "batch_err": [], "anchor_err": []} for ep in _ENDPOINTS}

    total = len(_ENDPOINTS) * 2 * _N_PER_PAIR
    fired = 0
    sample_idx = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for ep in _ENDPOINTS:
            for shape in ("batch", "anchor"):
                for i in range(_N_PER_PAIR):
                    if shape == "batch":
                        start = head - 500 - sample_idx * 30
                        body = _build_receipts_batch(start)
                    else:
                        body = _build_anchor_call()
                    sample_idx += 1
                    fired += 1
                    ok, rtt_s, err = _post(ep, body)
                    rec = {
                        "ts": time.time(), "endpoint": ep, "shape": shape,
                        "ok": ok, "rtt_ms": rtt_s * 1000.0, "err": err,
                    }
                    if ok:
                        samples[ep][shape].append(rtt_s * 1000.0)
                    else:
                        samples[ep][f"{shape}_err"].append(err)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if (fired % 10) == 0 or i == _N_PER_PAIR - 1:
                        print(f"[{fired}/{total}] {ep.split('/')[2]:32} shape={shape} "
                              f"rtt={rtt_s*1000:.0f}ms ok={ok}", flush=True)
                    time.sleep(_SPACING_S)

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        return xs_sorted[min(n - 1, int(n * p))]

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean_ms": sum(xs) / len(xs),
            "p50_ms": _pct(xs, 0.5),
            "p95_ms": _pct(xs, 0.95),
            "p99_ms": _pct(xs, 0.99),
            "max_ms": max(xs),
        }

    summary = {
        "n_per_pair": _N_PER_PAIR,
        "spacing_s": _SPACING_S,
        "batch_size": _BATCH_SIZE,
        "per_endpoint": {},
    }
    for ep in _ENDPOINTS:
        host = ep.split("/")[2]
        summary["per_endpoint"][host] = {
            "batch": _stats(samples[ep]["batch"]),
            "anchor": _stats(samples[ep]["anchor"]),
            "batch_failures": len(samples[ep]["batch_err"]),
            "anchor_failures": len(samples[ep]["anchor_err"]),
            "batch_err_sample": samples[ep]["batch_err"][:3],
            "anchor_err_sample": samples[ep]["anchor_err"][:3],
        }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
