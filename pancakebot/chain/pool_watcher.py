"""WebSocket watcher that accumulates PancakeSwap Prediction V2 bet pools.

Subscribes to BetBull/BetBear logs and newHeads over public BSC WSS
endpoints, maintains per-epoch pool amounts (bounded to the currently-open
round and the next) with block-timestamp-aware filtering, and backfills
bets via RPC on every successful WSS subscription.

Reliability features:
- Endpoint pool with round-robin failover across multiple public WSS URLs.
- Library-level keepalive via websockets ping_interval=30 / ping_timeout=10
  (the library raises ConnectionClosedError on silent TCP drops).
- Exponential backoff (5→10→20→40→80→120s cap) triggered only after
  cycling through every endpoint without a healthy session.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request as _urllib_req
from dataclasses import dataclass, field
from typing import Any

from pancakebot.constants import BNB_WEI, PREDICTION_V2_CONTRACT_ADDRESS, RPC_URLS
from pancakebot.log import info, warn
from pancakebot.util import InvariantError

# Public BSC WebSocket endpoints — tried in order, round-robin on failure.
# Tested 2026-04-17: drpc.org and publicnode.com verified working (block within 1s).
# bsc-rpc.publicnode.com (HTTP timeouts) and bsc.meowrpc.com (HTTP 405) removed.
# NOTE: only 2 public free endpoints found working from this environment; a third
# paid endpoint (e.g. NodeReal, QuickNode, Alchemy BSC) is recommended before live.
BSC_WSS_ENDPOINTS: list[str] = [
    "wss://bsc.drpc.org",
    "wss://bsc.publicnode.com",
]

# Keep old constant for backwards-compat imports.
BSC_PUBLIC_WSS = BSC_WSS_ENDPOINTS[0]

# Event topic hashes (keccak256 of event signatures).
_BET_BULL_TOPIC = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
_BET_BEAR_TOPIC = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"

# Backoff schedule (seconds), triggered after cycling through all endpoints without
# a healthy session. Reset once a session stays connected longer than _BACKOFF_RESET_SECONDS.
_BACKOFF_STEPS = [5, 10, 20, 40, 80, 120]
_BACKOFF_RESET_SECONDS = 60.0


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
        interval_seconds: int,
        wss_urls: list[str] | None = None,
        contract_address: str = PREDICTION_V2_CONTRACT_ADDRESS,
    ) -> None:
        if interval_seconds <= 0:
            raise InvariantError("interval_seconds_nonpositive")
        self._interval_seconds: int = interval_seconds
        self._wss_urls: list[str] = wss_urls if wss_urls is not None else list(BSC_WSS_ENDPOINTS)
        self._contract_addr = contract_address

        self._lock = threading.Lock()
        self._pools: dict[int, _EpochPool] = {}       # epoch -> pool (bounded to 2 entries)
        self._block_ts: dict[int, int] = {}            # block_number -> timestamp
        self._seen_tx: dict[int, set[str]] = {}        # epoch -> set of "tx_hash:log_idx"

        # Round-phase state (set by engine via set_round_phase).
        self._current_epoch: int = -1
        self._lock_at: int = 0

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._connected = False
        self._current_endpoint: str = ""
        self._last_connected_at: float = 0.0
        self._last_event_at: float = 0.0
        self._total_events = 0

        # Failure streak counter (incremented per unhealthy session, reset on healthy one).
        self._failure_streak: int = 0

        # Backfill tracking.
        self._backfill_count: int = 0
        self._last_backfill_at: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def current_endpoint(self) -> str:
        return self._current_endpoint

    @property
    def last_connected_at(self) -> float:
        return self._last_connected_at

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "connected": self._connected,
                "current_endpoint": self._current_endpoint,
                "last_connected_at": self._last_connected_at,
                "epochs_tracked": len(self._pools),
                "total_events": self._total_events,
                "blocks_tracked": len(self._block_ts),
                "backfill_count": self._backfill_count,
                "last_backfill_at": self._last_backfill_at,
            }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="pool-event-watcher",
        )
        self._thread.start()
        info("POOL_WSS", "START", "OK",
             msg=f"Pool event watcher started ({len(self._wss_urls)} endpoints)")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._connected = False
        info("POOL_WSS", "STOP", "OK", msg="Pool event watcher stopped")

    def get_pool(self, epoch: int, *, max_ts: int) -> tuple[float, float]:
        """Return (bull_bnb, bear_bnb) from confirmed events for a given epoch,
        including only bets with 0 < block_timestamp < max_ts.
        """
        if max_ts <= 0:
            raise InvariantError("get_pool_max_ts_nonpositive")
        bull_wei = 0
        bear_wei = 0

        with self._lock:
            pool = self._pools.get(epoch)
            if pool is None:
                return 0.0, 0.0

            for bet in pool.bets:
                if bet.block_ts == 0:
                    ts = self._block_ts.get(bet.block_number, 0)
                    if ts > 0:
                        bet.block_ts = ts

                if bet.block_ts == 0:
                    continue
                if bet.block_ts >= max_ts:
                    continue

                if bet.side == "Bull":
                    bull_wei += bet.amount_wei
                else:
                    bear_wei += bet.amount_wei

        return bull_wei / BNB_WEI, bear_wei / BNB_WEI

    def set_round_phase(self, *, current_epoch: int, lock_at: int) -> None:
        """Engine-driven state sync; called once per runtime iteration.

        Always strictly advances `current_epoch` after the first call
        (enforced as an invariant — the engine loop structure guarantees
        strictly increasing epochs between iterations that reach this
        method). On first call, triggers the initial backfill if a WSS
        session is already connected.
        """
        if current_epoch < 0:
            raise InvariantError("set_round_phase_negative_epoch")
        if lock_at <= 0:
            raise InvariantError("set_round_phase_lock_at_nonpositive")

        with self._lock:
            prev_epoch = self._current_epoch
            is_first_call = (prev_epoch == -1)

            if not is_first_call and current_epoch <= prev_epoch:
                raise InvariantError(
                    f"set_round_phase_non_advancing: prev={prev_epoch} new={current_epoch}"
                )

            if is_first_call:
                info("POOL_WSS", "EPOCH", "INIT",
                     msg=f"Initialized at epoch {current_epoch}",
                     epoch=current_epoch)
                self._current_epoch = current_epoch
            else:
                # Drop stale epochs (strictly less than new current_epoch) from
                # both _pools and _seen_tx. The "+1" next-epoch entries are kept.
                stale_pools = [e for e in self._pools if e < current_epoch]
                stale_seen = [e for e in self._seen_tx if e < current_epoch]
                for e in stale_pools:
                    del self._pools[e]
                for e in stale_seen:
                    del self._seen_tx[e]
                self._current_epoch = current_epoch

            self._lock_at = lock_at

            # Bounded _block_ts: keep most recent 500 once we exceed 1000.
            # (Migrated from clear_old_epochs; behavior unchanged.)
            if len(self._block_ts) > 1000:
                sorted_blocks = sorted(self._block_ts.keys())
                for bn in sorted_blocks[:-500]:
                    del self._block_ts[bn]

            trigger_backfill = is_first_call and self._connected

        # Outside the lock: backfill_round does HTTP and re-acquires self._lock
        # via _process_bet_event.
        if trigger_backfill:
            self.backfill_round(lock_at - self._interval_seconds)

    # ------------------------------------------------------------------
    # Internal: connection loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        n = len(self._wss_urls)
        idx = 0

        while not self._stop_event.is_set():
            url = self._wss_urls[idx]
            self._current_endpoint = url
            session_start = time.time()

            try:
                asyncio.run(self._ws_listen(url))
            except Exception as e:
                warn("POOL_WSS", "ERR", "RECONN",
                     msg=f"Endpoint {url}: {type(e).__name__}: {e}")
            self._connected = False

            session_duration = time.time() - session_start
            if session_duration >= _BACKOFF_RESET_SECONDS:
                self._failure_streak = 0  # healthy session
            else:
                self._failure_streak += 1

            idx = (idx + 1) % n

            # Back off only when we've cycled through every endpoint without a healthy session.
            if self._failure_streak >= n:
                step = min(self._failure_streak - n, len(_BACKOFF_STEPS) - 1)
                delay = _BACKOFF_STEPS[step]
                warn("POOL_WSS", "RETRY", "WAIT",
                     msg=f"All endpoints failed; backoff {delay}s (streak={self._failure_streak})")
                if self._stop_event.wait(timeout=delay):
                    break

    # ------------------------------------------------------------------
    # Internal: WebSocket session
    # ------------------------------------------------------------------

    async def _ws_listen(self, url: str) -> None:
        import websockets

        async with websockets.connect(
            url, ping_interval=30, ping_timeout=10, open_timeout=15,
        ) as ws:
            # Subscribe to BetBull/BetBear events
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "eth_subscribe",
                "params": ["logs", {
                    "address": self._contract_addr,
                    "topics": [[_BET_BULL_TOPIC, _BET_BEAR_TOPIC]],
                }],
            }))
            logs_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if "result" not in logs_resp:
                warn("POOL_WSS", "SUB", "FAIL",
                     msg=f"Logs subscription failed on {url}: {logs_resp}")
                await asyncio.sleep(2)
                return

            logs_sub_id = logs_resp["result"]

            # Subscribe to newHeads for block timestamps
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 2,
                "method": "eth_subscribe",
                "params": ["newHeads"],
            }))
            heads_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if "result" not in heads_resp:
                warn("POOL_WSS", "SUB", "FAIL",
                     msg=f"newHeads subscription failed on {url}: {heads_resp}")
                await asyncio.sleep(2)
                return

            heads_sub_id = heads_resp["result"]

            # Subscriptions confirmed — mark as connected.
            now = time.time()
            self._connected = True
            self._last_connected_at = now
            self._last_event_at = now
            session_start_at = now
            session_events = 0
            info("POOL_WSS", "SUB", "OK",
                 msg=f"Subscribed on {url}")

            # Reconnect-triggered backfill. Skip only when the engine hasn't
            # yet established round-phase state (first process start, pre-iter-1);
            # set_round_phase will trigger it instead.
            if self._lock_at > 0:
                self.backfill_round(self._lock_at - self._interval_seconds)

            disconnect_reason = "stop"
            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    # Short timeout exists only so the _stop_event check fires.
                    # Library-level ping_interval=30 / ping_timeout=10 handles
                    # real liveness: a silent TCP drop raises ConnectionClosedError
                    # out of ws.recv(), bubbling to _run_loop.
                    continue

                self._last_event_at = time.time()
                session_events += 1

                msg = json.loads(raw)
                params = msg.get("params", {})
                sub_id = params.get("subscription")
                result = params.get("result")
                if result is None:
                    continue

                if sub_id == logs_sub_id:
                    self._process_bet_event(result)
                elif sub_id == heads_sub_id:
                    self._process_new_head(result)

        # Log session summary before marking disconnected.
        duration = time.time() - session_start_at
        if duration > 0:
            warn("POOL_WSS", "WS", "CLOSED",
                 msg=f"Session ended on {url}: reason={disconnect_reason} "
                     f"duration={duration:.0f}s events={session_events}")
        self._connected = False

    # ------------------------------------------------------------------
    # Internal: event processing
    # ------------------------------------------------------------------

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

        # Epoch gate: accept only the tracked pair {current_epoch, current_epoch+1}.
        # Short-circuit before phase is initialized so backfill-before-first-phase
        # can populate _pools freely.
        if self._current_epoch >= 0 and epoch not in (
            self._current_epoch, self._current_epoch + 1
        ):
            return

        tx_hash = log.get("transactionHash", "")
        log_idx = log.get("logIndex", "")
        dedup_key = f"{tx_hash}:{log_idx}"

        with self._lock:
            seen = self._seen_tx.setdefault(epoch, set())
            if dedup_key and dedup_key in seen:
                return
            if dedup_key:
                seen.add(dedup_key)

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

    # ------------------------------------------------------------------
    # Internal: RPC helpers (for backfill)
    # ------------------------------------------------------------------

    @staticmethod
    def _rpc_call(rpc: str, method: str, params: list) -> Any:
        """Single JSON-RPC call. Returns result (str/dict/list) or None on error."""
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

    @staticmethod
    def _rpc_batch(rpc: str, calls: list[tuple[str, list]]) -> list[dict | None]:
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
            block_num_hex = self._rpc_call(rpc, "eth_blockNumber", [])
            if not isinstance(block_num_hex, str):
                raise InvariantError("backfill_block_number_failed")
            current_block = int(block_num_hex, 16)
            cur_block_data = self._rpc_call(
                rpc, "eth_getBlockByNumber", [hex(current_block), False],
            )
            if not isinstance(cur_block_data, dict):
                raise InvariantError("backfill_current_block_fetch_failed")
            current_ts = int(cur_block_data["timestamp"], 16)

            seconds_back = max(0, current_ts - round_start_ts)
            blocks_back = int(seconds_back / _BSC_BLOCK_TIME) + 20
            from_block = max(0, current_block - blocks_back)

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

            all_blocks = range(from_block, current_block + 1)
            for batch_start in range(0, len(all_blocks), _BATCH_SIZE):
                batch_bns = list(all_blocks[batch_start:batch_start + _BATCH_SIZE])
                calls = [
                    ("eth_getBlockByNumber", [hex(bn), True])
                    for bn in batch_bns
                ]
                # noinspection PyBroadException
                try:
                    results = self._rpc_batch(rpc, calls)
                except Exception:
                    blocks_failed += len(batch_bns)
                    continue

                tx_candidates: list[tuple[int, dict]] = []

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

                if tx_candidates:
                    rcpt_calls = [
                        ("eth_getTransactionReceipt", [tx["hash"]])
                        for _, tx in tx_candidates
                    ]
                    # noinspection PyBroadException
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

            with self._lock:
                self._backfill_count += 1
                self._last_backfill_at = time.time()

            info("POOL_WSS", "BKFILL", "OK",
                 msg=f"Backfilled {count} bets from {blocks_with_bets} blocks "
                     f"({total_blocks} blks, {total_txs} txs, "
                     f"{contract_txs} to_contract, {contract_txs_with_value} with_value, "
                     f"{blocks_failed} blk_fail, "
                     f"{reverted_txs} reverted, "
                     f"{receipt_fails} rcpt_fail)")

        except Exception as e:
            warn("POOL_WSS", "BKFILL", "FAIL", msg=f"{e}")
