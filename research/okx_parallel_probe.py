"""Parallel-fetch probe: replicates the bot's actual pattern.

The bot opens 3 simultaneous OKX /candles fetches via ThreadPoolExecutor
(BTC, ETH, SOL). The standalone okx_connection_ab.py does sequential
fetches and shows fresh_conn winning. Test if parallel-from-same-IP
behaves differently -- maybe OKX routes simultaneous source-IP
connections to the same backend.

Variants:
  A. session_reuse_serial: 1 fetch at a time, single shared Session
  B. fresh_serial:         1 fetch at a time, new Session each
  C. fresh_parallel_3:     3 fetches in parallel, each own Session
                           (matches bot's pattern exactly)

For each variant, 30 ROUNDS (so 30 BTC fetches per variant for
comparable n).

Usage:
    python research/okx_parallel_probe.py [--rounds 30] [--delay-ms 5000]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def _measure_skew_once() -> int:
    import requests
    r = requests.get("https://www.okx.com/api/v5/public/time", timeout=5)
    okx_ms = int(r.json()["data"][0]["ts"])
    local_ms = int(time.time() * 1000)
    return local_ms - okx_ms


def _fetch_one(symbol: str, use_fresh_session: bool, shared_sess,
               with_after_param: bool = False, count: int = 1) -> tuple[int, int, int]:
    """Returns (local_now_ms, newest_ts, elapsed_ms).

    When *with_after_param* is True, sends ``after=<now_ms - 1000>`` to mimic
    the bot's pattern of asking for "candles before cutoff" rather than
    "the latest candle as of right now."
    """
    import requests
    url = "https://www.okx.com/api/v5/market/candles"
    params: dict[str, str] = {"instId": symbol, "bar": "1s", "limit": str(count)}
    if with_after_param:
        # Mimic bot: cutoff = lock_at - 2s, fetch happens at cutoff + 0.25s,
        # request asks for "candles before cutoff_ms". So `after` is set to
        # roughly now - 250ms relative to the fetch time.
        params["after"] = str(int(time.time() * 1000) - 250)
    sess = requests.Session() if use_fresh_session else shared_sess
    try:
        t0 = int(time.time() * 1000)
        resp = sess.get(url, params=params, timeout=5)
        t1 = int(time.time() * 1000)
        body = resp.json()
        rows = body.get("data") or []
        newest = int(rows[0][0]) if rows else 0
        return ((t0 + t1) // 2, newest, t1 - t0)
    finally:
        if use_fresh_session:
            sess.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--delay-ms", type=int, default=5000,
                    help="Delay between rounds (mimics bot 5min interval, but compressed)")
    args = ap.parse_args()

    skew = _measure_skew_once()
    print(f"clock skew (local - okx): {skew} ms")
    print()

    import requests
    shared_sess = requests.Session()

    # Variant A: serial, shared session
    a_lags = []
    # Variant B: serial, fresh session per call
    b_lags = []
    # Variant C: parallel-3, fresh session per call (BOT PATTERN)
    c_lags = []
    c_eth_lags = []
    c_sol_lags = []

    # Variant D: parallel-3 fresh sessions WITH after= param (full bot pattern)
    d_lags = []

    for i in range(args.rounds):
        # A: serial shared
        local_ms, newest, _ = _fetch_one("BTC-USDT", False, shared_sess)
        okx_now = local_ms - skew
        a_lags.append(okx_now - (newest + 1000))

        # B: serial fresh
        local_ms, newest, _ = _fetch_one("BTC-USDT", True, None)
        okx_now = local_ms - skew
        b_lags.append(okx_now - (newest + 1000))

        # C: parallel 3 with fresh sessions (no after param)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {
                "BTC-USDT": pool.submit(_fetch_one, "BTC-USDT", True, None, False, 1),
                "ETH-USDT": pool.submit(_fetch_one, "ETH-USDT", True, None, False, 1),
                "SOL-USDT": pool.submit(_fetch_one, "SOL-USDT", True, None, False, 1),
            }
            for sym, fut in futs.items():
                local_ms, newest, _ = fut.result()
                okx_now = local_ms - skew
                lag = okx_now - (newest + 1000)
                if sym == "BTC-USDT":
                    c_lags.append(lag)
                elif sym == "ETH-USDT":
                    c_eth_lags.append(lag)
                else:
                    c_sol_lags.append(lag)

        # D: parallel-3 fresh + after= param + count=31 (FULL BOT PATTERN)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {
                "BTC-USDT": pool.submit(_fetch_one, "BTC-USDT", True, None, True, 31),
                "ETH-USDT": pool.submit(_fetch_one, "ETH-USDT", True, None, True, 31),
                "SOL-USDT": pool.submit(_fetch_one, "SOL-USDT", True, None, True, 31),
            }
            local_ms, newest, _ = futs["BTC-USDT"].result()
            okx_now = local_ms - skew
            d_lags.append(okx_now - (newest + 1000))
            for sym in ("ETH-USDT", "SOL-USDT"):
                futs[sym].result()  # drain

        if (i + 1) % 5 == 0:
            print(f"  round {i+1}/{args.rounds}: A={a_lags[-1]:>+5}ms B={b_lags[-1]:>+5}ms C_btc={c_lags[-1]:>+5}ms C_eth={c_eth_lags[-1]:>+5}ms C_sol={c_sol_lags[-1]:>+5}ms",
                  flush=True)
        time.sleep(args.delay_ms / 1000.0)

    print()
    print(f"{'variant':<28} {'n':>4} {'mean':>8} {'p50':>6} {'p95':>6} {'max':>6} {'>1s':>4}")
    print("-" * 70)
    for label, lags in (("A: serial shared session", a_lags),
                        ("B: serial fresh session", b_lags),
                        ("C: parallel-3 BTC fresh", c_lags),
                        ("C: parallel-3 ETH fresh", c_eth_lags),
                        ("C: parallel-3 SOL fresh", c_sol_lags),
                        ("D: parallel-3 fresh+after=", d_lags)):
        if not lags:
            continue
        n = len(lags)
        sorted_l = sorted(lags)
        mean_l = statistics.mean(lags)
        p50 = sorted_l[n // 2]
        p95 = sorted_l[int(0.95 * n)] if n >= 20 else max(sorted_l)
        gt1s = sum(1 for l in lags if l > 1000)
        print(f"{label:<28} {n:>4} {mean_l:>+8.0f} {p50:>+6} {p95:>+6} {max(lags):>+6} {gt1s:>4}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
