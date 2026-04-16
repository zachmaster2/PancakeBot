"""WebSocket watcher that accumulates PancakeSwap Prediction V2 bet pools.

Subscribes to BetBull/BetBear logs and newHeads over a public BSC WSS
endpoint, maintains per-epoch pool amounts with block-timestamp-aware
filtering, and supports RPC backfill for missed bets at startup.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request as _urllib_req
from dataclasses import dataclass, field

from pancakebot.constants import BNB_WEI, PREDICTION_V2_CONTRACT_ADDRESS, RPC_URLS
from pancakebot.log import info, warn

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

        If max_ts > 0, only include bets with block_timestamp < max_ts.
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
                    if bet.block_ts >= max_ts:
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
                 msg="Subscribed to bet events + newHeads")

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

    def _rpc_call(self, rpc: str, method: str, params: list) -> dict | None:
        """Single JSON-RPC call. Returns result or None on error."""
        req = json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": method, "params": params,
        }).encode()
        resp = _urllib_req.urlopen(_urllib_req.Request(
            rpc, data=req,
            headers={"Content-Type": "application/json"},
        ), timeout=10)
        body = json.loads(resp.read())
        return body.get("result")

    def _rpc_batch(self, rpc: str, calls: list[tuple[str, list]]) -> list[dict | None]:
        """Batch JSON-RPC call. Returns list of results (None for failures)."""
        batch = [
            {"jsonrpc": "2.0", "id": i, "method": method, "params": params}
            for i, (method, params) in enumerate(calls)
        ]
        req = json.dumps(batch).encode()
        resp = _urllib_req.urlopen(_urllib_req.Request(
            rpc, data=req,
            headers={"Content-Type": "application/json"},
        ), timeout=30)
        body = json.loads(resp.read())
        # Responses may arrive out of order; index by id.
        by_id = {r["id"]: r.get("result") for r in body} if isinstance(body, list) else {}
        return [by_id.get(i) for i in range(len(calls))]

    def backfill_round(self, round_start_ts: int) -> None:
        """Backfill bets by scanning blocks from round_start_ts to now.

        Uses batched eth_getBlockByNumber with full transactions (works on
        ALL free BSC RPCs) instead of eth_getLogs (unreliable on free nodes).
        Filters transactions to the prediction contract and parses
        bet events from transaction receipts.

        Called once by the runtime loop after the first epoch handshake.
        Dedup by tx_hash:log_index prevents double-counting with WSS.
        """
        _BSC_BLOCK_TIME = 0.5  # conservative (actual ~0.44s)
        _BATCH_SIZE = 100      # BSC free RPCs reject batches > 100
        rpc = RPC_URLS[0]

        try:
            # Get current block + timestamp
            current_block = int(self._rpc_call(rpc, "eth_blockNumber", []), 16)
            cur_block_data = self._rpc_call(
                rpc, "eth_getBlockByNumber", [hex(current_block), False],
            )
            current_ts = int(cur_block_data["timestamp"], 16)

            # Convert round_start_ts to block number
            seconds_back = max(0, current_ts - round_start_ts)
            blocks_back = int(seconds_back / _BSC_BLOCK_TIME) + 20
            from_block = max(0, current_block - blocks_back)

            # Scan blocks for transactions to the prediction contract
            contract_lower = self._contract_addr.lower()
            count = 0
            blocks_with_bets = 0
            blocks_failed = 0
            reverted_txs = 0
            receipt_fails = 0
            total_blocks = current_block - from_block + 1
            total_txs = 0
            contract_txs = 0
            contract_txs_with_value = 0

            info("POOL_WSS", "BKFILL", "SCAN",
                 msg=f"Scanning {total_blocks} blocks in batches of {_BATCH_SIZE} "
                     f"({hex(from_block)}..{hex(current_block)})")

            # Fetch blocks in batches
            all_blocks = range(from_block, current_block + 1)
            for batch_start in range(0, len(all_blocks), _BATCH_SIZE):
                batch_bns = list(all_blocks[batch_start:batch_start + _BATCH_SIZE])
                calls = [
                    ("eth_getBlockByNumber", [hex(bn), True])
                    for bn in batch_bns
                ]
                try:
                    results = self._rpc_batch(rpc, calls)
                except Exception:
                    blocks_failed += len(batch_bns)
                    continue

                # Collect tx hashes that need receipt fetches
                tx_candidates: list[tuple[int, dict]] = []  # (block_num, tx)

                for bn, block in zip(batch_bns, results):
                    if not block:
                        blocks_failed += 1
                        continue

                    block_ts = int(block["timestamp"], 16)
                    with self._lock:
                        self._block_ts[bn] = block_ts

                    txs = block.get("transactions", [])
                    total_txs += len(txs)
                    for tx in txs:
                        to_addr = (tx.get("to") or "").lower()
                        if to_addr != contract_lower:
                            continue
                        contract_txs += 1
                        value_wei = int(tx.get("value", "0x0"), 16)
                        if value_wei <= 0:
                            continue
                        contract_txs_with_value += 1
                        tx_candidates.append((bn, tx))

                # Batch-fetch receipts for candidate transactions
                if tx_candidates:
                    rcpt_calls = [
                        ("eth_getTransactionReceipt", [tx["hash"]])
                        for _, tx in tx_candidates
                    ]
                    try:
                        rcpt_results = self._rpc_batch(rpc, rcpt_calls)
                    except Exception:
                        receipt_fails += len(rcpt_calls)
                        continue

                    bet_blocks: set[int] = set()
                    for (bn, _tx), receipt in zip(tx_candidates, rcpt_results):
                        if not receipt:
                            receipt_fails += 1
                            continue
                        status = int(receipt.get("status", "0x0"), 16)
                        if status != 1:
                            reverted_txs += 1
                            continue
                        for log in receipt.get("logs", []):
                            topics = log.get("topics", [])
                            if not topics:
                                continue
                            if topics[0] in (_BET_BULL_TOPIC, _BET_BEAR_TOPIC):
                                self._process_bet_event(log)
                                count += 1
                                bet_blocks.add(bn)
                    blocks_with_bets += len(bet_blocks)

            info("POOL_WSS", "BKFILL", "OK",
                 msg=f"Backfilled {count} bets from {blocks_with_bets} blocks "
                     f"({total_blocks} blks, {total_txs} txs, "
                     f"{contract_txs} to_contract, {contract_txs_with_value} with_value, "
                     f"{blocks_failed} blk_fail, "
                     f"{reverted_txs} reverted, "
                     f"{receipt_fails} rcpt_fail)")

        except Exception as e:
            warn("POOL_WSS", "BKFILL", "FAIL", msg=f"{e}")
