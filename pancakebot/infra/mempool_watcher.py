"""WebSocket mempool watcher for PancakeSwap Prediction V2 bets.

Burst-mode: connects to WSS only when needed (around cutoff), collects
pending betBull/betBear transactions for a few seconds, then disconnects.
This keeps API credit usage minimal (~30 matched txs per burst vs 6k+
total BSC txs per minute from a persistent subscription).

Usage:
    watcher = MempoolWatcher(wss_url=...)
    watcher.burst_collect(epoch=472344, duration=5.0)  # blocking, ~5s
    bull, bear = watcher.get_pending_pool_delta(epoch=472344)

Thread-safe: burst_collect runs a temporary async loop in a background
thread, writing to a lock-protected dict.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass

from pancakebot.core.constants import BNB_WEI, PREDICTION_V2_CONTRACT_ADDRESS
from pancakebot.core.logging import info, warn

# betBull(uint256 epoch) selector = 0x57fb096f
# betBear(uint256 epoch) selector = 0xaa6b873a
_BET_BULL_SELECTOR = "0x57fb096f"
_BET_BEAR_SELECTOR = "0xaa6b873a"

# Max age of pending bets before we discard them (seconds).
_PENDING_BET_TTL_SECONDS = 30


@dataclass
class _PendingBet:
    epoch: int
    side: str        # "Bull" or "Bear"
    amount_wei: int
    tx_hash: str
    seen_at: float


class MempoolWatcher:
    """Burst-mode mempool watcher: connects briefly around cutoff."""

    def __init__(
        self,
        *,
        wss_url: str,
        contract_address: str = PREDICTION_V2_CONTRACT_ADDRESS,
    ) -> None:
        self._wss_url = wss_url
        self._contract_addr = contract_address.lower()
        self._lock = threading.Lock()
        self._pending_bets: dict[str, _PendingBet] = {}
        self._total_seen = 0
        self._total_matched = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "pending_count": len(self._pending_bets),
                "total_seen": self._total_seen,
                "total_matched": self._total_matched,
            }

    def burst_collect(self, *, epoch: int, duration: float = 5.0) -> None:
        """Connect to WSS, collect pending bets for `duration` seconds, disconnect.

        Runs in a temporary background thread so the caller isn't blocked
        by the async event loop. The caller should call this shortly before
        or at cutoff, then read results via get_pending_pool_delta().

        Blocking: returns after the burst completes (or on error).
        """
        done = threading.Event()
        error_holder: list[Exception] = []

        def _run():
            try:
                asyncio.run(self._burst_listen(epoch=epoch, duration=duration))
            except Exception as e:
                error_holder.append(e)
            finally:
                done.set()

        t = threading.Thread(target=_run, daemon=True, name="mempool-burst")
        t.start()
        # Wait for burst to complete (with generous timeout)
        done.wait(timeout=duration + 10)

        if error_holder:
            warn("MEMPOOL", "WSS", "BURST_ERROR",
                 msg=f"Burst collect failed: {error_holder[0]}")

    def burst_collect_async(self, *, epoch: int, duration: float = 5.0) -> threading.Event:
        """Non-blocking variant: starts burst in background, returns Event that signals completion."""
        done = threading.Event()

        def _run():
            try:
                asyncio.run(self._burst_listen(epoch=epoch, duration=duration))
            except Exception as e:
                warn("MEMPOOL", "WSS", "BURST_ERROR",
                     msg=f"Burst collect failed: {e}")
            finally:
                done.set()

        t = threading.Thread(target=_run, daemon=True, name="mempool-burst")
        t.start()
        return done

    def get_pending_pool_delta(self, epoch: int) -> tuple[float, float]:
        """Return (delta_bull_bnb, delta_bear_bnb) from pending bets for a given epoch."""
        now = time.time()
        delta_bull = 0.0
        delta_bear = 0.0

        with self._lock:
            stale = [
                h for h, b in self._pending_bets.items()
                if now - b.seen_at > _PENDING_BET_TTL_SECONDS
            ]
            for h in stale:
                del self._pending_bets[h]

            for bet in self._pending_bets.values():
                if bet.epoch == epoch:
                    amount_bnb = bet.amount_wei / BNB_WEI
                    if bet.side == "Bull":
                        delta_bull += amount_bnb
                    else:
                        delta_bear += amount_bnb

        return delta_bull, delta_bear

    def clear_epoch(self, epoch: int) -> None:
        """Remove all pending bets for a given epoch."""
        with self._lock:
            to_remove = [h for h, b in self._pending_bets.items() if b.epoch == epoch]
            for h in to_remove:
                del self._pending_bets[h]

    async def _burst_listen(self, *, epoch: int, duration: float) -> None:
        """Connect, subscribe, collect for `duration` seconds, disconnect."""
        import websockets

        deadline = time.time() + duration

        async with websockets.connect(self._wss_url, ping_interval=20, open_timeout=5) as ws:
            subscribe_msg = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": ["newPendingTransactions", True],
            })
            await ws.send(subscribe_msg)

            response = json.loads(await ws.recv())
            if "result" not in response:
                warn("MEMPOOL", "WSS", "SUB_FAIL", msg=f"Subscription failed: {response}")
                return

            matched_this_burst = 0

            while time.time() < deadline:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 1.0))
                except asyncio.TimeoutError:
                    continue

                msg = json.loads(raw)
                params = msg.get("params", {})
                result = params.get("result")
                if result is None:
                    continue

                self._total_seen += 1
                if self._process_pending_tx(result, epoch):
                    matched_this_burst += 1

            # Unsubscribe before disconnect
            try:
                unsub_msg = json.dumps({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "eth_unsubscribe",
                    "params": [response["result"]],
                })
                await ws.send(unsub_msg)
            except Exception:
                pass

        d_bull, d_bear = self.get_pending_pool_delta(epoch)
        info("MEMPOOL", "WSS", "BURST_DONE",
             epoch=epoch, matched=matched_this_burst,
             d_bull=f"{d_bull:.4f}", d_bear=f"{d_bear:.4f}",
             duration=f"{duration:.1f}s")

    def _process_pending_tx(self, tx: dict, target_epoch: int) -> bool:
        """Check if a pending tx is a PancakeSwap bet and record it. Returns True if matched."""
        to_addr = (tx.get("to") or "").lower()
        if to_addr != self._contract_addr:
            return False

        input_data = tx.get("input", "")
        if len(input_data) < 10:
            return False

        selector = input_data[:10].lower()
        if selector == _BET_BULL_SELECTOR:
            side = "Bull"
        elif selector == _BET_BEAR_SELECTOR:
            side = "Bear"
        else:
            return False

        try:
            epoch = int(input_data[10:74], 16)
        except (ValueError, IndexError):
            return False

        value_hex = tx.get("value", "0x0")
        try:
            amount_wei = int(value_hex, 16)
        except ValueError:
            return False

        if amount_wei <= 0:
            return False

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
                return True
        return False
