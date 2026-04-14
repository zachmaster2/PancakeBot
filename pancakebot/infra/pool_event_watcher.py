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
        self._seen_tx: set[str] = set()            # dedup by tx hash + log index
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
        # Backfill via HTTP first (independent of WebSocket)
        self._backfill_http()
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

            # Re-run backfill to cover gap between initial backfill and
            # WSS subscription start. Dedup prevents double-counting.
            self._backfill_http()

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

        # Dedup: use tx hash + log index to avoid double-counting
        tx_hash = log.get("transactionHash", "")
        log_idx = log.get("logIndex", "")
        dedup_key = f"{tx_hash}:{log_idx}"

        with self._lock:
            if dedup_key and dedup_key in self._seen_tx:
                return
            if dedup_key:
                self._seen_tx.add(dedup_key)

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

    def _backfill_http(self) -> None:
        """Backfill bets for the current round via HTTP eth_getLogs.

        Free BSC WebSocket nodes don't support eth_getLogs, so we use
        1rpc.io/bnb via HTTP instead.  Block count is computed from
        the round duration (INTERVAL_SECONDS / ~3s per BSC block).
        """
        import urllib.request as _urllib_req

        _BACKFILL_RPC = "https://1rpc.io/bnb"
        from pancakebot.core.constants import INTERVAL_SECONDS
        # BSC produces blocks every ~3 seconds
        blocks_per_round = INTERVAL_SECONDS // 3 + 10  # +10 margin

        try:
            # Get current block
            bn_req = json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "eth_blockNumber", "params": [],
            }).encode()
            resp = _urllib_req.urlopen(_urllib_req.Request(
                _BACKFILL_RPC, data=bn_req,
                headers={"Content-Type": "application/json"},
            ), timeout=10)
            current_block = int(json.loads(resp.read())["result"], 16)
            from_block = hex(max(0, current_block - blocks_per_round))

            # Fetch bet events
            logs_req = json.dumps({
                "jsonrpc": "2.0", "id": 2,
                "method": "eth_getLogs",
                "params": [{
                    "address": self._contract_addr.lower(),
                    "topics": [[_BET_BULL_TOPIC, _BET_BEAR_TOPIC]],
                    "fromBlock": from_block,
                    "toBlock": "latest",
                }],
            }).encode()
            resp = _urllib_req.urlopen(_urllib_req.Request(
                _BACKFILL_RPC, data=logs_req,
                headers={"Content-Type": "application/json"},
            ), timeout=10)
            result = json.loads(resp.read())

            if "error" in result:
                warn("POOL_WSS", "BKFILL", "FAIL",
                     msg=f"eth_getLogs error: {result['error'].get('message', '')}")
                return

            logs = result.get("result", [])
            if not isinstance(logs, list):
                return

            # Fetch block timestamps for each unique block
            blocks_needed = set()
            for log in logs:
                try:
                    bn = int(log.get("blockNumber", "0x0"), 16)
                    if bn > 0:
                        blocks_needed.add(bn)
                except ValueError:
                    pass

            for bn in sorted(blocks_needed):
                try:
                    ts_req = json.dumps({
                        "jsonrpc": "2.0", "id": 3,
                        "method": "eth_getBlockByNumber",
                        "params": [hex(bn), False],
                    }).encode()
                    resp = _urllib_req.urlopen(_urllib_req.Request(
                        _BACKFILL_RPC, data=ts_req,
                        headers={"Content-Type": "application/json"},
                    ), timeout=5)
                    ts_result = json.loads(resp.read())
                    if "result" in ts_result and ts_result["result"]:
                        block_ts = int(ts_result["result"]["timestamp"], 16)
                        with self._lock:
                            self._block_ts[bn] = block_ts
                except Exception:
                    pass

            # Process bet events
            count = 0
            for log in logs:
                self._process_bet_event(log)
                count += 1

            info("POOL_WSS", "BKFILL", "OK",
                 msg=f"Backfilled {count} bets from {len(blocks_needed)} blocks "
                     f"({from_block}..{hex(current_block)})")

        except Exception as e:
            warn("POOL_WSS", "BKFILL", "FAIL", msg=f"{e}")
