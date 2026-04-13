"""WebSocket event watcher for PancakeSwap Prediction V2 pool tracking.

Subscribes to confirmed BetBull/BetBear log events via a public WSS
endpoint, accumulating pool amounts per epoch in real time.  At cutoff,
the accumulated pools reflect every confirmed bet up to the latest
block the node has processed — more accurate than a single-point
round_data() RPC call which may read from a stale block.

Usage:
    watcher = PoolEventWatcher()
    watcher.start()
    ...
    bull, bear = watcher.get_pool(epoch=472344)
    watcher.stop()

Thread-safe: the background listener writes to a lock-protected dict,
and get_pool reads from it.
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
# BetBull(address indexed sender, uint256 indexed epoch, uint256 amount)
# BetBear(address indexed sender, uint256 indexed epoch, uint256 amount)
_BET_BULL_TOPIC = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
_BET_BEAR_TOPIC = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"


@dataclass
class _EpochPool:
    bull_wei: int = 0
    bear_wei: int = 0
    bet_count: int = 0


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
            }

    def start(self) -> None:
        """Start the background WebSocket listener thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="pool-event-watcher",
        )
        self._thread.start()
        info("POOL_WSS", "START", "OK", msg=f"Pool event watcher started ({self._wss_url})")

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._connected = False
        info("POOL_WSS", "STOP", "OK", msg="Pool event watcher stopped")

    def get_pool(self, epoch: int) -> tuple[float, float]:
        """Return (bull_bnb, bear_bnb) accumulated from confirmed events.

        Returns (0.0, 0.0) if no events have been seen for this epoch.
        Thread-safe.
        """
        with self._lock:
            pool = self._pools.get(epoch)
            if pool is None:
                return 0.0, 0.0
            return pool.bull_wei / BNB_WEI, pool.bear_wei / BNB_WEI

    def clear_old_epochs(self, keep_after: int) -> None:
        """Remove pool data for epochs <= keep_after to prevent memory growth."""
        with self._lock:
            stale = [e for e in self._pools if e <= keep_after]
            for e in stale:
                del self._pools[e]

    def _run_loop(self) -> None:
        """Main thread entry: reconnect loop for WebSocket."""
        while not self._stop_event.is_set():
            try:
                asyncio.run(self._ws_listen())
            except Exception as e:
                self._connected = False
                if self._stop_event.is_set():
                    break
                warn("POOL_WSS", "ERROR", "RECONNECT",
                     msg=f"WebSocket error, reconnecting in 5s: {e}")
                for _ in range(50):
                    if self._stop_event.is_set():
                        return
                    time.sleep(0.1)

    async def _ws_listen(self) -> None:
        """Connect to WSS and subscribe to BetBull/BetBear events."""
        import websockets

        async with websockets.connect(
            self._wss_url, ping_interval=None, open_timeout=10,
        ) as ws:
            sub_msg = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": ["logs", {
                    "address": self._contract_addr,
                    "topics": [[_BET_BULL_TOPIC, _BET_BEAR_TOPIC]],
                }],
            })
            await ws.send(sub_msg)
            response = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))

            if "result" not in response:
                warn("POOL_WSS", "SUB", "FAIL", msg=f"Subscription failed: {response}")
                await asyncio.sleep(5)
                return

            self._connected = True
            info("POOL_WSS", "SUB", "OK",
                 msg=f"Subscribed to BetBull/BetBear events, sub_id={response['result']}")

            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    # Send a ping to keep connection alive
                    try:
                        pong = await ws.ping()
                        await asyncio.wait_for(pong, timeout=5)
                    except Exception:
                        break  # connection dead, will reconnect
                    continue

                msg = json.loads(raw)
                result = msg.get("params", {}).get("result")
                if result is None:
                    continue

                self._process_event(result)

    def _process_event(self, log: dict) -> None:
        """Process a confirmed BetBull/BetBear event."""
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
        except (ValueError, IndexError):
            return

        if amount_wei <= 0:
            return

        with self._lock:
            if epoch not in self._pools:
                self._pools[epoch] = _EpochPool()
            pool = self._pools[epoch]
            if side == "Bull":
                pool.bull_wei += amount_wei
            else:
                pool.bear_wei += amount_wei
            pool.bet_count += 1
            self._total_events += 1
