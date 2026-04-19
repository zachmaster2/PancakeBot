"""Ping-interval A/B experiment for BSC WSS connections.

Measures whether client-side pings prevent server-side keepalive drops.
Records every session start, disconnect, and restart to a shared JSONL file.

Usage:
    python research/ping_experiment.py --ping-interval 60 --tag with_ping
    python research/ping_experiment.py --ping-interval 0  --tag no_ping
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_DEFAULT_OUTPUT = Path(__file__).parent / "data" / "ping_experiment.jsonl"
RECONNECT_WAIT = 30  # seconds between reconnects


def write_record(record: dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


async def run_session(tag: str, endpoint: str, ping_interval: int, output_file: Path) -> None:
    import websockets

    pi = ping_interval if ping_interval > 0 else None

    t0 = time.time()
    write_record({"tag": tag, "event": "session_start", "ts": t0,
                  "endpoint": endpoint, "ping_interval": ping_interval}, output_file)
    print(f"[{tag}] session_start endpoint={endpoint} ping_interval={pi}", flush=True)

    reason = "unknown"
    try:
        async with websockets.connect(endpoint, ping_interval=pi, open_timeout=15) as ws:
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "eth_subscribe", "params": ["newHeads"],
            }))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            if "result" not in resp:
                reason = f"subscribe_failed:{resp}"
                return

            print(f"[{tag}] subscribed ok", flush=True)

            block_count = 0
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg.get("method") == "eth_subscription":
                    block_count += 1
                    bn_hex = msg.get("params", {}).get("result", {}).get("number", "0x0")
                    bn = int(bn_hex, 16)
                    # Log every 100th block to keep file manageable; always log first
                    if block_count == 1 or block_count % 100 == 0:
                        write_record({"tag": tag, "event": "new_head", "ts": time.time(),
                                      "block_number": bn, "block_count": block_count}, output_file)
                    if block_count % 10 == 0:
                        print(f"[{tag}] block={bn} count={block_count}", flush=True)

        reason = "connection_closed_ok"

    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"[{tag}] disconnect reason={reason}", flush=True)

    finally:
        duration = time.time() - t0
        write_record({"tag": tag, "event": "disconnect", "ts": time.time(),
                      "session_duration_seconds": round(duration, 2), "reason": reason}, output_file)
        print(f"[{tag}] disconnect duration={duration:.1f}s reason={reason}", flush=True)


async def main(tag: str, endpoint: str, ping_interval: int, output_file: Path) -> None:
    attempt = 0
    while True:
        if attempt > 0:
            write_record({"tag": tag, "event": "session_restart", "ts": time.time(),
                          "attempt": attempt}, output_file)
            print(f"[{tag}] waiting {RECONNECT_WAIT}s before reconnect (attempt={attempt})",
                  flush=True)
            await asyncio.sleep(RECONNECT_WAIT)

        attempt += 1
        await run_session(tag, endpoint, ping_interval, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ping-interval", type=int, default=0,
                        help="Ping interval in seconds. 0 = disabled.")
    parser.add_argument("--tag", required=True, help="Label for records.")
    parser.add_argument("--endpoint", default="wss://bsc.drpc.org",
                        help="WSS endpoint URL.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to JSONL output file. Defaults to research/data/ping_experiment.jsonl.")
    args = parser.parse_args()

    output_file = Path(args.output_file) if args.output_file else _DEFAULT_OUTPUT

    print(f"Starting ping_experiment tag={args.tag} endpoint={args.endpoint} "
          f"ping_interval={args.ping_interval} output={output_file}", flush=True)

    asyncio.run(main(args.tag, args.endpoint, args.ping_interval, output_file))
