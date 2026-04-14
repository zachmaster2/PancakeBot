"""WebSocket event watcher for PancakeSwap Prediction V2 pool tracking.

Subscribes to:
  1. BetBull/BetBear log events — accumulates pool amounts per epoch
  2. newHeads — tracks block_number → timestamp mapping for time filtering

At decision time, the pipeline queries pools filtered to a specific
timestamp (e.g., lock_at - 6) for consistency with backtest.

Thread-safe: the background listener writes to lock-protected dicts,
and get_pool reads from them.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field

from pancakebot.core.constants import BNB_WEI, PREDICTION_V2_CONTRACT_ADDRESS
from pancakebot.core.logging import info, warn

# Public BSC WebSocket endpoint (no signup, no API key).
BSC_PUBLIC_WSS = "wss://bsc.publicnode.com"

# Event topic hashes (keccak256 of event signatures).
_BET_BULL_TOPIC = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
_BET_BEAR_TOPIC = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"


@dataclass
class _Bet:
    epoch: int
    side: str        # "Bull" or "Bear"
    amount_wei: int
    block_number: int
    block_ts: int    # block timestamp (from newHeads), 0 if not yet resolved


@dataclass
class _EpochPool:
    bets: list[_Bet] = field(default_factory=list)


class PoolEventWatcher:
    """Background thread that tracks PancakeSwap pools via confirmed events."""

    def __init__(
        self,
        *,
        wss_url: str = BSC_PUBLIC_WSS,
        contract_address: str = PREDICTION_V2_CONTRACT_ADDRESS,
    ) -> None:
        self._wss_url = wss_url
        self._contract_addr = contract_address
        self._lock = threading.Lock()
        self._pools: dict[int, _EpochPool] = {}  # epoch -> pool
        self._block_ts: dict[int, int] = {}       # block_number -> timestamp
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = False
        self._total_events = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "connected": self._connected,
                "epochs_tracked": len(self._pools),
                "total_events": self._total_events,
                "blocks_tracked": len(self._block_ts),
            }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="pool-event-watcher",
        )
        self._thread.start()
        info("POOL_WSS", "START", "OK", msg=f"Pool event watcher started ({self._wss_url})")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._connected = False
        info("POOL_WSS", "STOP", "OK", msg="Pool event watcher stopped")

    def get_pool(self, epoch: int, *, max_ts: int = 0) -> tuple[float, float]:
        """Return (bull_bnb, bear_bnb) from confirmed events for a given epoch.

        If max_ts > 0, only include bets with block_timestamp <= max_ts.
        This allows filtering to e.g. lock_at - 6 for consistent pool data.
        """
        bull_wei = 0
        bear_wei = 0

        with self._lock:
            pool = self._pools.get(epoch)
            if pool is None:
                return 0.0, 0.0

            for bet in pool.bets:
                # Resolve block timestamp if not yet known
                if bet.block_ts == 0:
                    ts = self._block_ts.get(bet.block_number, 0)
                    if ts > 0:
                        bet.block_ts = ts

                # Apply time filter
                if max_ts > 0:
                    if bet.block_ts == 0:
                        continue  # unknown timestamp, skip to be safe
                    if bet.block_ts > max_ts:
                        continue

                if bet.side == "Bull":
                    bull_wei += bet.amount_wei
                else:
                    bear_wei += bet.amount_wei

        return bull_wei / BNB_WEI, bear_wei / BNB_WEI

    def clear_old_epochs(self, keep_after: int) -> None:
        with self._lock:
            stale = [e for e in self._pools if e <= keep_after]
            for e in stale:
                del self._pools[e]
            # Also trim old block timestamps (keep last 1000)
            if len(self._block_ts) > 1000:
                sorted_blocks = sorted(self._block_ts.keys())
                for b in sorted_blocks[:-500]:
                    del self._block_ts[b]

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                asyncio.run(self._ws_listen())
            except Exception as e:
                self._connected = False
                if self._stop_event.is_set():
                    break
                warn("POOL_WSS", "ERR", "RECONN",
                     msg=f"WebSocket error, reconnecting in 5s: {e}")
                for _ in range(50):
                    if self._stop_event.is_set():
                        return
                    time.sleep(0.1)

    async def _ws_listen(self) -> None:
        import websockets

        async with websockets.connect(
            self._wss_url, ping_interval=None, open_timeout=10,
        ) as ws:
            # Subscribe to BetBull/BetBear events
            logs_sub = json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "eth_subscribe",
                "params": ["logs", {
                    "address": self._contract_addr,
                    "topics": [[_BET_BULL_TOPIC, _BET_BEAR_TOPIC]],
                }],
            })
            await ws.send(logs_sub)
            logs_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))

            if "result" not in logs_resp:
                warn("POOL_WSS", "SUB", "FAIL", msg=f"Logs subscription failed: {logs_resp}")
                await asyncio.sleep(5)
                return

            logs_sub_id = logs_resp["result"]

            # Subscribe to newHeads for block timestamps
            heads_sub = json.dumps({
                "jsonrpc": "2.0", "id": 2,
                "method": "eth_subscribe",
                "params": ["newHeads"],
            })
            await ws.send(heads_sub)
            heads_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))

            if "result" not in heads_resp:
                warn("POOL_WSS", "SUB", "FAIL", msg=f"newHeads subscription failed: {heads_resp}")
                await asyncio.sleep(5)
                return

            self._connected = True
            info("POOL_WSS", "SUB", "OK",
                 msg=f"Subscribed to bet events + newHeads")

            # Backfill: fetch recent bet events via eth_getLogs to catch
            # bets placed before we subscribed.  Covers the current open
            # round.  Uses _ws_rpc() to handle interleaved subscription
            # events while waiting for RPC responses.
            await self._backfill(ws, logs_sub_id, heads_resp["result"])

            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    # Send a ping to keep connection alive
                    try:
                        pong = await ws.ping()
                        await asyncio.wait_for(pong, timeout=5)
                    except Exception:
                        break
                    continue

                msg = json.loads(raw)
                params = msg.get("params", {})
                sub_id = params.get("subscription")
                result = params.get("result")
                if result is None:
                    continue

                if sub_id == logs_sub_id:
                    self._process_bet_event(result)
                elif sub_id == heads_resp["result"]:
                    self._process_new_head(result)

    def _process_bet_event(self, log: dict) -> None:
        topics = log.get("topics", [])
        if not topics or len(topics) < 3:
            return

        topic0 = topics[0]
        if topic0 == _BET_BULL_TOPIC:
            side = "Bull"
        elif topic0 == _BET_BEAR_TOPIC:
            side = "Bear"
        else:
            return

        try:
            epoch = int(topics[2], 16)
            amount_wei = int(log.get("data", "0x0"), 16)
            block_number = int(log.get("blockNumber", "0x0"), 16)
        except (ValueError, IndexError):
            return

        if amount_wei <= 0:
            return

        with self._lock:
            # Look up block timestamp if available
            block_ts = self._block_ts.get(block_number, 0)

            if epoch not in self._pools:
                self._pools[epoch] = _EpochPool()
            self._pools[epoch].bets.append(_Bet(
                epoch=epoch, side=side, amount_wei=amount_wei,
                block_number=block_number, block_ts=block_ts,
            ))
            self._total_events += 1

    def _process_new_head(self, head: dict) -> None:
        try:
            block_number = int(head.get("number", "0x0"), 16)
            timestamp = int(head.get("timestamp", "0x0"), 16)
        except (ValueError, TypeError):
            return

        with self._lock:
            self._block_ts[block_number] = timestamp

    async def _ws_rpc(self, ws, req_id: int, method: str, params: list,
                      logs_sub_id: str, heads_sub_id: str) -> dict | None:
        """Send an RPC request and return its response, processing any
        interleaved subscription events while waiting."""
        req = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        await ws.send(req)
        for _ in range(50):  # max 50 messages before giving up
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw)
            if "id" in msg and msg["id"] == req_id:
                return msg  # our RPC response
            # Subscription event — process it and keep waiting
            if "params" in msg:
                sub_id = msg["params"].get("subscription")
                result = msg["params"].get("result", {})
                if sub_id == logs_sub_id:
                    self._process_bet_event(result)
                elif sub_id == heads_sub_id:
                    self._process_new_head(result)
        return None

    async def _backfill(self, ws, logs_sub_id: str, heads_sub_id: str) -> None:
        """Backfill bets for the current round via eth_getLogs."""
        try:
            # Get current block number
            bn_resp = await self._ws_rpc(ws, 10, "eth_blockNumber", [],
                                          logs_sub_id, heads_sub_id)
            if not bn_resp or "result" not in bn_resp:
                warn("POOL_WSS", "BKFILL", "FAIL", msg="Could not get block number")
                return

            current_block = int(bn_resp["result"], 16)
            # ~10 minutes of BSC blocks (covers 2 full rounds)
            from_block = hex(max(0, current_block - 200))

            logs_resp = await self._ws_rpc(
                ws, 11, "eth_getLogs",
                [{"address": self._contract_addr,
                  "topics": [[_BET_BULL_TOPIC, _BET_BEAR_TOPIC]],
                  "fromBlock": from_block, "toBlock": "latest"}],
                logs_sub_id, heads_sub_id)

            if not logs_resp or "result" not in logs_resp:
                warn("POOL_WSS", "BKFILL", "FAIL", msg="eth_getLogs returned no result")
                return

            logs = logs_resp["result"]
            if not isinstance(logs, list):
                return

            # Collect unique block numbers for timestamp resolution
            blocks_needed = set()
            for log in logs:
                try:
                    bn = int(log.get("blockNumber", "0x0"), 16)
                    if bn > 0:
                        blocks_needed.add(bn)
                except ValueError:
                    pass

            # Fetch block timestamps (needed for pool cutoff filtering)
            for i, bn in enumerate(sorted(blocks_needed)):
                ts_resp = await self._ws_rpc(
                    ws, 100 + i, "eth_getBlockByNumber", [hex(bn), False],
                    logs_sub_id, heads_sub_id)
                if ts_resp and "result" in ts_resp and ts_resp["result"]:
                    block_ts = int(ts_resp["result"]["timestamp"], 16)
                    with self._lock:
                        self._block_ts[bn] = block_ts

            # Process all backfilled bet events
            count = 0
            for log in logs:
                self._process_bet_event(log)
                count += 1

            info("POOL_WSS", "BKFILL", "OK",
                 msg=f"Backfilled {count} bet events from {len(blocks_needed)} blocks "
                     f"(block range {from_block}..{hex(current_block)})")

        except Exception as e:
            warn("POOL_WSS", "BKFILL", "FAIL", msg=f"Backfill error: {e}")
