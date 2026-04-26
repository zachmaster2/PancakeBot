"""Test (b): does adding an artificial delay before fetch eliminate the lag?

Hypothesis: bot's lag is caused by firing OKX requests too early in TRUE UTC
because local clock is skewed +1.7s ahead. Adding a wait equal to the skew
should let OKX progress to where we expect, eliminating the lag.

Methodology:
- Use OKX `/public/time` to anchor TRUE UTC precisely (one-time skew measure)
- For each tested offset (0ms, 500ms, 1000ms, 1500ms, 2000ms, 2500ms past UTC second):
  - Wait until LOCAL time = next-second-boundary + offset_local
    where offset_local = offset_target + skew (so true-UTC fire happens at offset_target)
  - Fire 3 parallel /candles fetches with after=that_utc_second*1000 (mimics bot)
  - Measure lag = (after - 1000) - actual_newest_open_time

If lag is high at small offsets but drops to ~0 around offset = 1500-2000ms,
confirms the hypothesis: bot needs to wait skew+publish_lag before firing.

Usage:
    python research/okx_artificial_delay_probe.py [--rounds 3]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def _measure_skew_once() -> int:
    import requests
    samples = []
    for _ in range(3):
        t0 = time.time() * 1000
        r = requests.get("https://www.okx.com/api/v5/public/time", timeout=5)
        t1 = time.time() * 1000
        okx_ms = int(r.json()["data"][0]["ts"])
        local_at_req = (t0 + t1) / 2
        samples.append(int(local_at_req - okx_ms))
        time.sleep(0.3)
    return int(statistics.median(samples))


def _fetch_one(symbol: str, after_ms: int) -> tuple[int, int]:
    """Returns (request_send_local_ms, newest_open_time_ms)."""
    import requests
    sess = requests.Session()
    try:
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": symbol, "bar": "1s", "limit": "31",
                  "after": str(after_ms)}
        t0 = int(time.time() * 1000)
        resp = sess.get(url, params=params, timeout=5)
        body = resp.json()
        rows = body.get("data") or []
        newest = int(rows[0][0]) if rows else 0
        return (t0, newest)
    finally:
        sess.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3,
                    help="Number of trials per offset (more = better tail estimate)")
    args = ap.parse_args()

    skew = _measure_skew_once()
    print(f"clock skew (local - okx): {skew} ms")
    print()

    # Test offsets relative to a UTC second boundary -- we want to observe how
    # lag varies with how-late-after-true-UTC-second we fire.
    OFFSETS_MS = [0, 500, 1000, 1500, 2000, 2500, 3000]

    print(f"{'true-utc-offset':>18} {'btc_lag':>8} {'eth_lag':>8} {'sol_lag':>8}")
    print("-" * 50)

    results: dict[int, list[tuple[int, int, int]]] = {o: [] for o in OFFSETS_MS}

    for trial in range(args.rounds):
        for offset_target in OFFSETS_MS:
            # We want to fire at TRUE-UTC = (some integer second) + offset_target.
            # In LOCAL terms, that's LOCAL = TRUE-UTC + skew.
            # Wait until next integer-second-in-UTC boundary, then add offset_target.
            local_now = time.time() * 1000
            true_utc_now = local_now - skew
            next_utc_second = int(true_utc_now / 1000 + 1) * 1000  # next true-UTC sec
            target_local = next_utc_second + skew + offset_target  # local-clock target
            sleep_ms = target_local - local_now
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

            # The "cutoff" we're asking OKX for is the integer second we just
            # crossed. We want OKX to give us the candle covering [cutoff-1, cutoff).
            # In OKX terms, ask after=next_utc_second.
            cutoff_ms = next_utc_second
            with ThreadPoolExecutor(max_workers=3) as pool:
                futs = {
                    "BTC": pool.submit(_fetch_one, "BTC-USDT", cutoff_ms),
                    "ETH": pool.submit(_fetch_one, "ETH-USDT", cutoff_ms),
                    "SOL": pool.submit(_fetch_one, "SOL-USDT", cutoff_ms),
                }
                lags = {}
                for sym, fut in futs.items():
                    _, newest = fut.result()
                    expected_newest = cutoff_ms - 1000
                    lag_ms = expected_newest - newest
                    lags[sym] = lag_ms
            results[offset_target].append((lags["BTC"], lags["ETH"], lags["SOL"]))
            print(f"{offset_target:>18}ms {lags['BTC']:>+6}ms {lags['ETH']:>+6}ms {lags['SOL']:>+6}ms"
                  f"   (trial {trial+1}/{args.rounds})")

    print()
    print(f"=== summary across {args.rounds} trials ===")
    print(f"{'offset (ms)':>12}: {'btc_mean':>10} {'btc_p50':>10} {'eth_mean':>10} {'sol_mean':>10}")
    for o in OFFSETS_MS:
        rows = results[o]
        btc = [r[0] for r in rows]
        eth = [r[1] for r in rows]
        sol = [r[2] for r in rows]
        if not btc:
            continue
        print(f"{o:>12}: "
              f"{statistics.mean(btc):>+10.0f} "
              f"{statistics.median(btc):>+10.0f} "
              f"{statistics.mean(eth):>+10.0f} "
              f"{statistics.mean(sol):>+10.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
