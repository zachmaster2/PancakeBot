"""WebSocket watcher that accumulates PancakeSwap Prediction V2 bet pools.

Subscribes to BetBull/BetBear logs and newHeads over public BSC WSS
endpoints, maintains per-epoch pool amounts with block-timestamp-aware
filtering, and supports RPC backfill for missed bets at startup.

Reliability features:
- Endpoint pool with round-robin failover across multiple public WSS URLs.
- Per-endpoint exponential backoff with jitter (5→10→20→40→80→120s cap).
  Backoff resets after staying connected for >60s.
- Per-endpoint circuit breaker: 3 consecutive failures → skip endpoint for 5min.
- Watchdog thread: forces reconnect if connected but no event/newHead for 30s.
"""
from __future__ import annotations

import asyncio
import json
import random
import threading
import time
import urllib.request as _urllib_req
from dataclasses import dataclass, field
from typing import Any

from pancakebot.constants import BNB_WEI, PREDICTION_V2_CONTRACT_ADDRESS, RPC_URLS
from pancakebot.log import info, warn
from pancakebot.util import InvariantError

# Public BSC WebSocket endpoints — tried in order, round-robin on failure.
BSC_WSS_ENDPOINTS: list[str] = [
    "wss://bsc-rpc.publicnode.com",
    "wss://bsc.drpc.org",
    "wss://bsc.meowrpc.com",
]

# Keep old constant for backwards-compat imports.
BSC_PUBLIC_WSS = BSC_WSS_ENDPOINTS[0]

# Event topic hashes (keccak256 of event signatures).
_BET_BULL_TOPIC = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
_BET_BEAR_TOPIC = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"

# Backoff schedule (seconds), per endpoint.
_BACKOFF_STEPS = [5, 10, 20, 40, 80, 120]
_BACKOFF_JITTER = (0.75, 1.25)
_BACKOFF_RESET_SECONDS = 60.0   # reset step to 0 after staying connected this long

# Circuit breaker, per endpoint.
_CB_FAILURE_THRESHOLD = 3       # consecutive failures to open circuit
_CB_COOLDOWN_SECONDS = 300.0    # 5 minutes

# Watchdog.
_WATCHDOG_STALE_SECONDS = 30.0  # reconnect if no event/head for this long
_WATCHDOG_POLL_SECONDS = 5.0


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


@dataclass
class _EndpointState:
    url: str
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0    # epoch time; 0 = circuit closed (OK to use)
    backoff_step: int = 0
    session_connected_at: float = 0.0  # time.time() when _connected last became True


class PoolEventWatcher:
    """Background thread that tracks PancakeSwap pools via confirmed events."""

    def __init__(
        self,
        *,
        wss_urls: list[str] | None = None,
        contract_address: str = PREDICTION_V2_CONTRACT_ADDRESS,
    ) -> None:
        self._wss_urls: list[str] = wss_urls if wss_urls is not None else list(BSC_WSS_ENDPOINTS)
        self._contract_addr = contract_address

        self._lock = threading.Lock()
        self._pools: dict[int, _EpochPool] = {}   # epoch -> pool
        self._block_ts: dict[int, int] = {}        # block_number -> timestamp
        self._seen_tx: set[str] = set()            # dedup by tx hash + log index

        self._thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._force_reconnect = threading.Event()

        self._connected = False
        self._current_endpoint: str = ""
        self._last_connected_at: float = 0.0
        self._last_event_at: float = 0.0
        self._total_events = 0

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
            }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._force_reconnect.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="pool-event-watcher",
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="pool-watcher-watchdog",
        )
        self._thread.start()
        self._watchdog_thread.start()
        info("POOL_WSS", "START", "OK",
             msg=f"Pool event watcher started ({len(self._wss_urls)} endpoints)")

    def stop(self) -> None:
        self._stop_event.set()
        self._force_reconnect.set()  # unblock any sleeping ws_listen
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5)
            self._watchdog_thread = None
        self._connected = False
        info("POOL_WSS", "STOP", "OK", msg="Pool event watcher stopped")

    def get_pool(self, epoch: int, *, max_ts: int = 0) -> tuple[float, float]:
        """Return (bull_bnb, bear_bnb) from confirmed events for a given epoch.

        If max_ts > 0, only include bets with block_timestamp < max_ts.
        """
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

                if max_ts > 0:
                    if bet.block_ts == 0:
                        continue
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
            if len(self._block_ts) > 1000:
                sorted_blocks = sorted(self._block_ts.keys())
                for b in sorted_blocks[:-500]:
                    del self._block_ts[b]

    # ------------------------------------------------------------------
    # Internal: watchdog
    # ------------------------------------------------------------------

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_WATCHDOG_POLL_SECONDS)
            if self._stop_event.is_set():
                break
            if not self._connected:
                continue
            age = time.time() - self._last_event_at
            if age > _WATCHDOG_STALE_SECONDS:
                warn("POOL_WSS", "WDG", "STALE",
                     msg=f"No events for {age:.0f}s on {self._current_endpoint}, forcing reconnect")
                self._connected = False
                self._force_reconnect.set()

    # ------------------------------------------------------------------
    # Internal: connection loop
    # ------------------------------------------------------------------

    def _interruptible_sleep(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while not self._stop_event.is_set():
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(0.1, remaining))

    def _run_loop(self) -> None:
        ep_states = [_EndpointState(url=url) for url in self._wss_urls]
        ep_idx = 0

        while not self._stop_event.is_set():
            # --- Circuit breaker: find next available endpoint ---
            now = time.time()
            n = len(ep_states)
            available_indices = [
                i for i in range(n) if ep_states[i].circuit_open_until <= now
            ]

            if not available_indices:
                # All circuit-open; wait for the soonest to re-close.
                soonest = min(s.circuit_open_until for s in ep_states)
                wait = max(1.0, soonest - now)
                warn("POOL_WSS", "CB", "ALL_OPEN",
                     msg=f"All endpoints circuit-open, waiting {wait:.0f}s for cooldown")
                self._interruptible_sleep(min(wait, 30.0))
                continue

            # Round-robin within available endpoints: pick smallest index >= ep_idx,
            # or wrap to the first available.
            chosen = None
            for i in available_indices:
                if i >= ep_idx:
                    chosen = i
                    break
            if chosen is None:
                chosen = available_indices[0]

            ep_idx = chosen
            state = ep_states[ep_idx]

            # --- Backoff delay (skip on very first attempt, step==0 + no prior failures) ---
            if state.backoff_step > 0 or state.consecutive_failures > 0:
                step = min(state.backoff_step, len(_BACKOFF_STEPS) - 1)
                base = _BACKOFF_STEPS[step]
                delay = base * random.uniform(*_BACKOFF_JITTER)
                warn("POOL_WSS", "RETRY", "WAIT",
                     msg=f"Endpoint {state.url} backoff {delay:.1f}s "
                         f"(step={step}, failures={state.consecutive_failures})")
                self._interruptible_sleep(delay)

            if self._stop_event.is_set():
                break

            # --- Attempt connection ---
            self._current_endpoint = state.url
            state.session_connected_at = 0.0
            self._force_reconnect.clear()

            try:
                asyncio.run(self._ws_listen(state))
            except Exception as e:
                self._connected = False
                warn("POOL_WSS", "ERR", "RECONN",
                     msg=f"Endpoint {state.url}: {type(e).__name__}: {e}")

            # --- Post-session: update backoff / circuit breaker ---
            session_duration = 0.0
            if state.session_connected_at > 0:
                session_duration = time.time() - state.session_connected_at

            if session_duration >= _BACKOFF_RESET_SECONDS:
                # Healthy session — reset this endpoint's failure counters.
                state.backoff_step = 0
                state.consecutive_failures = 0
                info("POOL_WSS", "EP", "RESET",
                     msg=f"Endpoint {state.url} healthy ({session_duration:.0f}s), backoff reset")
            else:
                state.consecutive_failures += 1
                state.backoff_step = min(state.backoff_step + 1, len(_BACKOFF_STEPS) - 1)

                if state.consecutive_failures >= _CB_FAILURE_THRESHOLD:
                    state.circuit_open_until = time.time() + _CB_COOLDOWN_SECONDS
                    warn("POOL_WSS", "CB", "OPEN",
                         msg=f"Endpoint {state.url} circuit-open 5min "
                             f"(failures={state.consecutive_failures})")

            # Advance to next endpoint for the next attempt.
            ep_idx = (ep_idx + 1) % n

    # ------------------------------------------------------------------
    # Internal: WebSocket session
    # ------------------------------------------------------------------

    async def _ws_listen(self, state: _EndpointState) -> None:
        import websockets

        async with websockets.connect(
            state.url, ping_interval=None, open_timeout=10,
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
                     msg=f"Logs subscription failed on {state.url}: {logs_resp}")
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
                     msg=f"newHeads subscription failed on {state.url}: {heads_resp}")
                await asyncio.sleep(2)
                return

            heads_sub_id = heads_resp["result"]

            # Subscriptions confirmed — mark as connected.
            now = time.time()
            self._connected = True
            self._last_connected_at = now
            self._last_event_at = now
            state.session_connected_at = now
            info("POOL_WSS", "SUB", "OK",
                 msg=f"Subscribed on {state.url}")

            while not self._stop_event.is_set() and not self._force_reconnect.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    # No message for 10s — send ping to confirm liveness.
                    if self._force_reconnect.is_set():
                        break
                    try:
                        pong = await ws.ping()
                        await asyncio.wait_for(pong, timeout=5)
                    except Exception:
                        break
                    continue

                self._last_event_at = time.time()

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

        # Exiting cleanly — mark disconnected if we weren't already.
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

        tx_hash = log.get("transactionHash", "")
        log_idx = log.get("logIndex", "")
        dedup_key = f"{tx_hash}:{log_idx}"

        with self._lock:
            if dedup_key and dedup_key in self._seen_tx:
                return
            if dedup_key:
                self._seen_tx.add(dedup_key)

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

            info("POOL_WSS", "BKFILL", "OK",
                 msg=f"Backfilled {count} bets from {blocks_with_bets} blocks "
                     f"({total_blocks} blks, {total_txs} txs, "
                     f"{contract_txs} to_contract, {contract_txs_with_value} with_value, "
                     f"{blocks_failed} blk_fail, "
                     f"{reverted_txs} reverted, "
                     f"{receipt_fails} rcpt_fail)")

        except Exception as e:
            warn("POOL_WSS", "BKFILL", "FAIL", msg=f"{e}")
