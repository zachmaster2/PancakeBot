"""A/B test: requests.Session reuse vs fresh-conn vs cache-buster.

Tests three variants against OKX `/api/v5/market/candles?instId=BTC-USDT&bar=1s`:

  A. session_reuse: single requests.Session() reused across all calls
  B. fresh_conn:    new requests.Session() per call (forces fresh TCP/TLS)
  C. cache_buster:  session_reuse + extra `&_t=<unique>` query param

For each variant:
  - 30 fetches with 800ms delay between
  - records local time, OKX server time (one-time skew measure), newest_ts,
    response status, and key cache headers (cf-cache-status, age, server,
    x-amz-cf-pop, x-served-by)

Output: stdout summary + per-variant lag distribution + cache header counts.

Doesn't run against the live bot's session — independent probe so it can
run while the bot is also fetching.

Usage:
    python research/okx_connection_ab.py [--n 30] [--delay-ms 800]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from collections import Counter
from typing import Any


def _measure_skew_once() -> int:
    import requests
    r = requests.get("https://www.okx.com/api/v5/public/time", timeout=5)
    okx_ms = int(r.json()["data"][0]["ts"])
    local_ms = int(time.time() * 1000)
    return local_ms - okx_ms


_DIAG_HEADERS = (
    "cache-control", "age", "x-cache", "cf-cache-status",
    "x-served-by", "via", "server", "date", "x-amz-cf-id",
    "x-amz-cf-pop", "expires", "etag",
)


def _fetch(variant: str, sess, n: int) -> dict[str, Any]:
    import requests
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": "BTC-USDT", "bar": "1s", "limit": "1"}
    if variant == "cache_buster":
        params["_t"] = uuid.uuid4().hex
    elif variant == "fresh_conn":
        sess = requests.Session()  # new session per call

    t0 = int(time.time() * 1000)
    if variant == "fresh_conn":
        # Add Connection: close to discourage keep-alive
        resp = sess.get(url, params=params, timeout=5,
                        headers={"Connection": "close"})
    else:
        resp = sess.get(url, params=params, timeout=5)
    t1 = int(time.time() * 1000)
    body = resp.json()
    rows = body.get("data") or []
    newest_ts = int(rows[0][0]) if rows else 0
    headers = {h: resp.headers.get(h, "") for h in _DIAG_HEADERS
               if resp.headers.get(h)}
    return {
        "variant": variant,
        "local_ms": (t0 + t1) // 2,
        "newest_ts": newest_ts,
        "status": resp.status_code,
        "headers": headers,
        "elapsed_ms": t1 - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--delay-ms", type=int, default=800)
    args = ap.parse_args()

    skew = _measure_skew_once()
    print(f"clock skew (local - okx): {skew} ms")
    print()

    import requests
    shared_sess = requests.Session()
    variants = ("session_reuse", "fresh_conn", "cache_buster")
    results: dict[str, list[dict]] = {v: [] for v in variants}

    # Round-robin so all variants experience similar OKX load conditions.
    for i in range(args.n):
        for v in variants:
            r = _fetch(v, shared_sess, i)
            results[v].append(r)
            time.sleep(args.delay_ms / 1000.0 / 3)

    print(f"{'variant':<14} {'n':>4} {'mean_lag':>10} {'p50':>8} {'p95':>8} {'max':>8} {'>1s':>6} {'>3s':>6}")
    print("-" * 70)
    for v in variants:
        rows = results[v]
        # lag in OKX frame
        lags_okx = []
        for r in rows:
            okx_now = r["local_ms"] - skew
            close_time = r["newest_ts"] + 1000
            lags_okx.append(okx_now - close_time)
        n = len(lags_okx)
        sorted_l = sorted(lags_okx)
        mean_l = statistics.mean(lags_okx)
        p50 = sorted_l[n // 2]
        p95 = sorted_l[int(0.95 * n)] if n >= 20 else max(sorted_l)
        mx = max(lags_okx)
        gt1s = sum(1 for l in lags_okx if l > 1000)
        gt3s = sum(1 for l in lags_okx if l > 3000)
        print(f"{v:<14} {n:>4} {mean_l:>+10.0f} {p50:>+8} {p95:>+8} {mx:>+8} {gt1s:>6} {gt3s:>6}")

    # Header analysis per variant
    print("\n--- cache header presence per variant ---")
    for v in variants:
        rows = results[v]
        cf_status_counter: Counter = Counter()
        age_values = []
        server_counter: Counter = Counter()
        for r in rows:
            h = r.get("headers") or {}
            cf_status_counter[h.get("cf-cache-status", "(none)")] += 1
            try:
                age_values.append(int(h.get("age", "0")))
            except Exception:
                pass
            server_counter[h.get("server", "(none)")] += 1
        print(f"\n  {v}:")
        print(f"    cf-cache-status: {dict(cf_status_counter)}")
        if age_values:
            print(f"    age (seconds):  mean={sum(age_values)/len(age_values):.1f}  "
                  f"max={max(age_values)}  zero_count={sum(1 for a in age_values if a==0)}/{len(age_values)}")
        print(f"    server:         {dict(server_counter)}")

    # Sample headers from a high-lag and low-lag run
    print("\n--- sample headers from highest-lag fetch in each variant ---")
    for v in variants:
        rows = results[v]
        # Find highest lag in OKX frame
        worst = max(rows, key=lambda r: (r["local_ms"] - skew) - (r["newest_ts"] + 1000))
        worst_lag = (worst["local_ms"] - skew) - (worst["newest_ts"] + 1000)
        print(f"  {v} (worst lag={worst_lag:+d}ms):")
        for k, val in (worst.get("headers") or {}).items():
            print(f"    {k}: {val}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
