"""WebSocket mempool watcher for PancakeSwap Prediction V2 bets.

Subscribes to pending transactions via a WSS endpoint, filters for
betBull/betBear calls to the Prediction contract, and maintains a
real-time pool estimate that includes not-yet-mined bets.

Usage:
    watcher = MempoolWatcher(wss_url=..., contract_address=...)
    watcher.start()           # spawns background thread
    ...
    bull, bear = watcher.get_pending_pool_delta(epoch=472100)
    watcher.stop()

Thread-safe: the background listener writes to a lock-protected dict,
and get_pending_pool_delta reads from it.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field

from pancakebot.core.constants import BNB_WEI, PREDICTION_V2_CONTRACT_ADDRESS
from pancakebot.core.logging import info, warn

# betBull(uint256 epoch) selector = 0x57fb096f
# betBear(uint256 epoch) selector = 0xaa6b873a
_BET_BULL_SELECTOR = "0x57fb096f"
_BET_BEAR_SELECTOR = "0xaa6b873a"

# Max age of pending bets before we discard them (seconds).
# A bet should be mined within ~3-6 seconds on BSC.
_PENDING_BET_TTL_SECONDS = 30


@dataclass
class _PendingBet:
    epoch: int
    side: str        # "Bull" or "Bear"
    amount_wei: int
    tx_hash: str
    seen_at: float   # time.time() when first observed


class MempoolWatcher:
    """Background thread that watches for pending PancakeSwap bets."""

    def __init__(
        self,
        *,
        wss_url: str,
        contract_address: str = PREDICTION_V2_CONTRACT_ADDRESS,
    ) -> None:
        self._wss_url = wss_url
        self._contract_addr = contract_address.lower()
        self._lock = threading.Lock()
        self._pending_bets: dict[str, _PendingBet] = {}  # tx_hash -> bet
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = False
        self._total_seen = 0
        self._total_matched = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "connected": self._connected,
                "pending_count": len(self._pending_bets),
                "total_seen": self._total_seen,
                "total_matched": self._total_matched,
            }

    def start(self) -> None:
        """Start the background WebSocket listener thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mempool-watcher",
        )
        self._thread.start()
        info("MEMPOOL", "WSS", "START", msg="Mempool watcher thread started")

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._connected = False
        info("MEMPOOL", "WSS", "STOP", msg="Mempool watcher stopped")

    def get_pending_pool_delta(self, epoch: int) -> tuple[float, float]:
        """Return (delta_bull_bnb, delta_bear_bnb) from pending bets for a given epoch.

        Thread-safe. Purges stale entries on each call.
        """
        now = time.time()
        delta_bull = 0
        delta_bear = 0

        with self._lock:
            # Purge stale entries
            stale = [
                h for h, b in self._pending_bets.items()
                if now - b.seen_at > _PENDING_BET_TTL_SECONDS
            ]
            for h in stale:
                del self._pending_bets[h]

            # Sum pending bets for the requested epoch
            for bet in self._pending_bets.values():
                if bet.epoch == epoch:
                    amount_bnb = bet.amount_wei / BNB_WEI
                    if bet.side == "Bull":
                        delta_bull += amount_bnb
                    else:
                        delta_bear += amount_bnb

        return delta_bull, delta_bear

    def clear_epoch(self, epoch: int) -> None:
        """Remove all pending bets for a given epoch (after it's been settled)."""
        with self._lock:
            to_remove = [h for h, b in self._pending_bets.items() if b.epoch == epoch]
            for h in to_remove:
                del self._pending_bets[h]

    def _run_loop(self) -> None:
        """Main thread entry: run asyncio event loop for WebSocket."""
        while not self._stop_event.is_set():
            try:
                asyncio.run(self._ws_listen())
            except Exception as e:
                self._connected = False
                if self._stop_event.is_set():
                    break
                warn("MEMPOOL", "WSS", "ERROR",
                     msg=f"WebSocket error, reconnecting in 5s: {e}")
                # Wait 5 seconds before reconnecting, but check stop event
                for _ in range(50):
                    if self._stop_event.is_set():
                        return
                    time.sleep(0.1)

    async def _ws_listen(self) -> None:
        """Connect to WSS and subscribe to pending transactions."""
        try:
            import websockets
        except ImportError:
            warn("MEMPOOL", "WSS", "MISSING_DEP",
                 msg="websockets package not installed. Run: pip install websockets")
            self._stop_event.set()
            return

        async with websockets.connect(self._wss_url, ping_interval=20) as ws:
            # Subscribe to pending transactions
            subscribe_msg = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": ["newPendingTransactions", True],  # True = full tx objects
            })
            await ws.send(subscribe_msg)

            # Read subscription confirmation
            response = json.loads(await ws.recv())
            if "result" in response:
                self._connected = True
                info("MEMPOOL", "WSS", "SUBSCRIBED",
                     msg=f"Subscribed to pending txs, sub_id={response['result']}")
            else:
                warn("MEMPOOL", "WSS", "SUB_FAIL",
                     msg=f"Subscription failed: {response}")
                return

            # Process incoming pending transactions
            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                msg = json.loads(raw)
                params = msg.get("params", {})
                result = params.get("result")
                if result is None:
                    continue

                self._total_seen += 1
                self._process_pending_tx(result)

    def _process_pending_tx(self, tx: dict) -> None:
        """Check if a pending tx is a PancakeSwap bet and record it."""
        to_addr = (tx.get("to") or "").lower()
        if to_addr != self._contract_addr:
            return

        input_data = tx.get("input", "")
        if len(input_data) < 10:  # "0x" + 8 hex chars minimum
            return

        selector = input_data[:10].lower()
        if selector == _BET_BULL_SELECTOR:
            side = "Bull"
        elif selector == _BET_BEAR_SELECTOR:
            side = "Bear"
        else:
            return

        # Decode epoch from input data (uint256, 32 bytes after selector)
        try:
            epoch = int(input_data[10:74], 16)
        except (ValueError, IndexError):
            return

        # Get bet amount from tx value
        value_hex = tx.get("value", "0x0")
        try:
            amount_wei = int(value_hex, 16)
        except ValueError:
            return

        if amount_wei <= 0:
            return

        tx_hash = tx.get("hash", "")

        with self._lock:
            if tx_hash not in self._pending_bets:
                self._pending_bets[tx_hash] = _PendingBet(
                    epoch=epoch,
                    side=side,
                    amount_wei=amount_wei,
                    tx_hash=tx_hash,
                    seen_at=time.time(),
                )
                self._total_matched += 1

                amount_bnb = amount_wei / BNB_WEI
                info("MEMPOOL", "BET", "PENDING",
                     side=side, epoch=epoch,
                     amount=f"{amount_bnb:.4f}",
                     tx=tx_hash[:16])
