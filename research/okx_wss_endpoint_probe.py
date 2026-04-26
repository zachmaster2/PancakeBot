"""Probe OKX WSS to verify endpoint + candle1s push semantics.

Tests:
1. Does `candle1s` subscription succeed on /public vs /business?
2. Are pushes "one per closed candle" or "mid-bar updates"?

Captures 30s of pushes to one of the endpoints and reports.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

import websockets


ENDPOINTS = [
    ("public", "wss://ws.okx.com:8443/ws/v5/public"),
    ("business", "wss://ws.okx.com:8443/ws/v5/business"),
]

INST_ID = "BTC-USDT"
DURATION_S = 30


async def probe(label: str, url: str) -> dict:
    """Subscribe, capture pushes for DURATION_S, return summary."""
    pushes: list[dict] = []
    error = None
    sub_ack = None
    try:
        async with websockets.connect(url, open_timeout=5) as ws:
            await ws.send(json.dumps({
                "op": "subscribe",
                "args": [{"channel": "candle1s", "instId": INST_ID}],
            }))
            t_start = time.monotonic()
            while time.monotonic() - t_start < DURATION_S:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                obj = json.loads(msg)
                if "event" in obj:  # subscribe ack / error
                    sub_ack = obj
                    if obj.get("event") == "error":
                        error = f"sub_error: {obj}"
                        break
                else:
                    pushes.append({"ts": time.monotonic() - t_start, "obj": obj})
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    # Analyze pushes: how many unique open_time values? same-ts pushes?
    open_times = []
    for p in pushes:
        for row in p["obj"].get("data") or []:
            try:
                open_times.append(int(row[0]))
            except (ValueError, IndexError):
                pass
    unique_ots = sorted(set(open_times))
    same_ts_repeats = len(open_times) - len(unique_ots)

    return {
        "label": label,
        "url": url,
        "error": error,
        "subscribe_ack": sub_ack,
        "n_push_messages": len(pushes),
        "n_total_candle_rows": len(open_times),
        "n_unique_open_times": len(unique_ots),
        "same_ts_repeats": same_ts_repeats,
        "first_5_pushes": [
            {"recv_at": round(p["ts"], 2), "data": p["obj"].get("data")}
            for p in pushes[:5]
        ],
        "last_5_pushes": [
            {"recv_at": round(p["ts"], 2), "data": p["obj"].get("data")}
            for p in pushes[-5:]
        ],
    }


async def main() -> int:
    for label, url in ENDPOINTS:
        print(f"\n{'=' * 60}")
        print(f"PROBE: {label} -> {url}")
        print(f"{'=' * 60}")
        try:
            r = await asyncio.wait_for(probe(label, url), timeout=DURATION_S + 10)
        except asyncio.TimeoutError:
            print(f"  ERROR: probe timed out")
            continue
        print(f"  error: {r['error']}")
        print(f"  subscribe_ack: {r['subscribe_ack']}")
        print(f"  n_push_messages: {r['n_push_messages']}")
        print(f"  n_total_candle_rows: {r['n_total_candle_rows']}")
        print(f"  n_unique_open_times: {r['n_unique_open_times']}")
        print(f"  same_ts_repeats (mid-bar updates): {r['same_ts_repeats']}")
        print(f"  first 3 pushes:")
        for p in r["first_5_pushes"][:3]:
            print(f"    @{p['recv_at']}s  data={p['data']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
