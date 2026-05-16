"""Spike script: validate cross-subscription ordering on a single WSS connection.

Tests the assumption underpinning Approach A' frontier advancement
in var/design/proactive_wss_reconciliation_2026_05_06.md (rev 3): on a
single WSS connection subscribed to BOTH newHeads and PredictionV2 logs,
the server emits messages in TCP order such that **all logs for block X
arrive BEFORE the newHead for block X+1**.

If this holds, then "newHead X+1 received" is the readiness signal for
auditing block X — no debounce needed.

Method:
  1. Open one WSS connection.
  2. Subscribe to newHeads + PredictionV2 BetBull/BetBear logs.
  3. For each received message, record (timestamp, type, block_number, payload).
  4. After T seconds (default 600s = 10 min), analyze:
     - For every log L for block X, did it arrive before newHead X+1?
     - Cross-block ordering violations are flagged.
     - Within-block ordering between newHead X and logs of X is fuzzy (OK).

Pass criteria:
  - Zero cross-block ordering violations (a log for block X arriving
    AFTER newHead X+1).

Usage: python research/probe_wss_ordering.py [seconds] [endpoint]
  seconds: subscription duration (default 600)
  endpoint: drpc | publicnode | both (default both, sequential)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

PREDICTION_V2 = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"
BET_BULL_TOPIC = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
BET_BEAR_TOPIC = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"

ENDPOINTS = {
    "drpc": "wss://bsc.drpc.org",
    "publicnode": "wss://bsc.publicnode.com",
}


@dataclass
class Event:
    arrived_at: float
    kind: str       # "newHead" | "log"
    block_number: int
    block_hash: str
    extra: dict = field(default_factory=dict)


@dataclass
class Result:
    endpoint: str
    duration_s: float
    n_newheads: int
    n_logs: int
    log_blocks: dict[int, int]      # block_number -> log_count
    violations: list[dict]          # log arrived after newHead block+1


async def run_session(label: str, url: str, duration_s: int) -> Result:
    import websockets

    print(f"\n[{label}] connecting to {url} ({duration_s}s)")

    events: list[Event] = []
    n_newheads = 0
    n_logs = 0

    async with websockets.connect(
        url, ping_interval=30, ping_timeout=10, open_timeout=15,
    ) as ws:
        # Subscribe to logs first.
        await ws.send(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "eth_subscribe",
            "params": ["logs", {
                "address": PREDICTION_V2,
                "topics": [[BET_BULL_TOPIC, BET_BEAR_TOPIC]],
            }],
        }))
        logs_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if "result" not in logs_resp:
            raise RuntimeError(f"logs subscribe failed: {logs_resp}")
        logs_sub_id = logs_resp["result"]
        print(f"[{label}] logs subscribed: {logs_sub_id}")

        await ws.send(json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "eth_subscribe",
            "params": ["newHeads"],
        }))
        heads_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if "result" not in heads_resp:
            raise RuntimeError(f"newHeads subscribe failed: {heads_resp}")
        heads_sub_id = heads_resp["result"]
        print(f"[{label}] newHeads subscribed: {heads_sub_id}")

        deadline = time.time() + duration_s
        while time.time() < deadline:
            timeout = max(0.5, deadline - time.time())
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break

            arrived_at = time.time()
            msg = json.loads(raw)
            params = msg.get("params") or {}
            sub_id = params.get("subscription")
            result = params.get("result")
            if result is None:
                continue

            if sub_id == heads_sub_id:
                bn = int(result.get("number", "0x0"), 16)
                bh = result.get("hash", "")
                events.append(Event(arrived_at, "newHead", bn, bh))
                n_newheads += 1
                if n_newheads % 50 == 0:
                    print(f"[{label}] {n_newheads} newHeads, {n_logs} logs")
            elif sub_id == logs_sub_id:
                bn = int(result.get("blockNumber", "0x0"), 16)
                bh = result.get("blockHash", "")
                tx_hash = result.get("transactionHash", "")
                log_idx = result.get("logIndex", "0x0")
                events.append(Event(
                    arrived_at, "log", bn, bh,
                    extra={"tx_hash": tx_hash, "log_idx": log_idx},
                ))
                n_logs += 1

    # Analyze ordering.
    log_blocks: dict[int, int] = {}
    last_seen_newhead_bn = -1
    violations: list[dict] = []
    for ev in events:
        if ev.kind == "newHead":
            last_seen_newhead_bn = max(last_seen_newhead_bn, ev.block_number)
        else:
            log_blocks[ev.block_number] = log_blocks.get(ev.block_number, 0) + 1
            # Violation: log for block X arrived after newHead X+1 (or later).
            if last_seen_newhead_bn >= ev.block_number + 1:
                violations.append({
                    "log_block": ev.block_number,
                    "current_newhead_block": last_seen_newhead_bn,
                    "lag_blocks": last_seen_newhead_bn - ev.block_number,
                    "tx_hash": ev.extra.get("tx_hash"),
                    "log_idx": ev.extra.get("log_idx"),
                })

    return Result(
        endpoint=label,
        duration_s=duration_s,
        n_newheads=n_newheads,
        n_logs=n_logs,
        log_blocks=log_blocks,
        violations=violations,
    )


def summarize(r: Result) -> bool:
    print(f"\n=== {r.endpoint} summary ===")
    print(f"  duration:       {r.duration_s}s")
    print(f"  newHeads:       {r.n_newheads}")
    print(f"  logs:           {r.n_logs}")
    print(f"  unique log-blocks: {len(r.log_blocks)}")
    print(f"  ordering violations: {len(r.violations)}")
    if r.violations:
        print(f"  violation samples (first 5):")
        for v in r.violations[:5]:
            print(
                f"    log_block={v['log_block']} "
                f"current_newhead={v['current_newhead_block']} "
                f"lag={v['lag_blocks']} blocks "
                f"tx={v['tx_hash']}"
            )
    expected_min_newheads = (r.duration_s // 1) - 5  # generous: ~2 blocks/s, so >> duration_s
    healthy = (
        r.n_newheads >= expected_min_newheads
        and len(r.violations) == 0
    )
    return healthy


async def main() -> None:
    duration_s = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    target = sys.argv[2] if len(sys.argv) > 2 else "both"

    if target == "both":
        labels = list(ENDPOINTS.keys())
    elif target in ENDPOINTS:
        labels = [target]
    else:
        print(f"unknown target {target!r}; choose: drpc | publicnode | both")
        sys.exit(1)

    print(f"=== Phase 0c spike: WSS cross-subscription ordering ({duration_s}s) ===")
    print(f"endpoints: {labels}")

    summaries = []
    for label in labels:
        try:
            r = await run_session(label, ENDPOINTS[label], duration_s)
        except Exception as e:
            print(f"[{label}] FATAL: {type(e).__name__}: {e}")
            continue
        summaries.append((label, r))

    print(f"\n=== VERDICT ===")
    overall_pass = True
    for label, r in summaries:
        ok = summarize(r)
        verdict = "PASS" if ok else "FAIL"
        print(f"  {label:12s}  newHeads={r.n_newheads}  logs={r.n_logs}  "
              f"violations={len(r.violations)}  {verdict}")
        overall_pass = overall_pass and ok

    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
