"""Probe OKX /candles vs /history-candles to characterize fetch lag.

H1: Is the lag intrinsic to OKX (always-N seconds behind real time)?
H2: Is the lag because we ask too early?
H5: Is the lag because our clock is skewed?

The probe loops N times asking for the most-recent 1s BTC candle and
records:
  - local time at request (ms)
  - OKX server time (from /api/v5/public/time)
  - newest candle ts in the response

Then computes:
  - lag_local = local_now - newest_ts        (what we observe)
  - lag_okx   = okx_now   - newest_ts        (what's true in OKX frame)
  - clock_skew = local_now - okx_now

If lag_okx is small (<1s), H5 alone explains the issue (clock skew).
If lag_okx is large (>1s) consistently, H1 (intrinsic OKX lag).
If lag_okx varies with timing, H2 (we fetch too early).

Usage:
    python research/okx_lag_probe.py [--n 50] [--delay-ms 500]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from urllib.request import urlopen, Request


def _get_okx_time_ms() -> int:
    import requests
    r = requests.get("https://www.okx.com/api/v5/public/time", timeout=5)
    body = r.json()
    return int(body["data"][0]["ts"])


def _fetch_btc_newest_ts(*, after_ms: int | None = None,
                         use_session: bool = True,
                         session_obj=None) -> tuple[int, int, int]:
    """Returns (request_local_ms, response_local_ms, newest_open_time_ms)."""
    import requests
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": "BTC-USDT", "bar": "1s", "limit": "1"}
    if after_ms is not None:
        params["after"] = str(after_ms)
    t0 = int(time.time() * 1000)
    if use_session:
        sess = session_obj or requests.Session()
        resp = sess.get(url, params=params, timeout=5)
    else:
        resp = requests.get(url, params=params, timeout=5,
                            headers={"Connection": "close"})
    t1 = int(time.time() * 1000)
    body = resp.json()
    rows = body.get("data") or []
    newest_ts = int(rows[0][0]) if rows else 0
    return t0, t1, newest_ts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--delay-ms", type=int, default=500)
    ap.add_argument("--mode", choices=("session", "no-session"), default="session")
    args = ap.parse_args()

    import requests
    sess = requests.Session() if args.mode == "session" else None

    okx_ts = _get_okx_time_ms()
    local_ts = int(time.time() * 1000)
    skew = local_ts - okx_ts
    print(f"clock skew (local - okx): {skew} ms")
    print(f"mode: {args.mode}, n={args.n}, delay={args.delay_ms}ms")
    print()
    print(f"{'#':>3} {'local_ms':>13} {'newest_ts':>13} "
          f"{'lag_local':>9} {'lag_okx_est':>11} {'inferred_publish':>16}")
    print("-" * 80)

    lag_local_dist = []
    lag_okx_dist = []
    publish_lag_dist = []

    for i in range(args.n):
        t0, t1, newest = _fetch_btc_newest_ts(
            use_session=(args.mode == "session"), session_obj=sess,
        )
        # We use the response time as "now" (closest to when OKX read its DB)
        local_now = (t0 + t1) // 2
        okx_now_est = local_now - skew
        # newest candle covers [newest .. newest+1000), close-time is newest+1000
        # lag in local frame = local_now - close_time
        # lag in OKX frame   = okx_now_est - close_time
        close_time = newest + 1000
        lag_local = local_now - close_time
        lag_okx = okx_now_est - close_time
        # OKX publish lag = how long after the candle CLOSED until OKX serves it
        # Equals lag_okx (assuming OKX would serve it instantly when it has it)
        publish_lag = max(0, lag_okx)
        lag_local_dist.append(lag_local)
        lag_okx_dist.append(lag_okx)
        publish_lag_dist.append(publish_lag)
        if i < 10 or i >= args.n - 5:
            print(f"{i:>3} {local_now:>13} {newest:>13} "
                  f"{lag_local:>9} {lag_okx:>11} {publish_lag:>16}")
        time.sleep(args.delay_ms / 1000.0)

    print()
    print(f"--- summary (n={args.n}) ---")
    for label, dist in (("lag_local", lag_local_dist),
                        ("lag_okx", lag_okx_dist),
                        ("publish_lag", publish_lag_dist)):
        if not dist:
            continue
        mean = statistics.mean(dist)
        median = statistics.median(dist)
        sorted_d = sorted(dist)
        p95 = sorted_d[int(0.95 * len(sorted_d))] if len(sorted_d) >= 20 else max(sorted_d)
        print(f"  {label:<14} mean={mean:>+8.1f}ms  median={median:>+6}ms  "
              f"p95={p95:>+6}ms  min={min(dist):>+6}ms  max={max(dist):>+6}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
