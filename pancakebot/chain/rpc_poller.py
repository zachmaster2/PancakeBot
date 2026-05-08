"""HTTP RPC poller for PancakeSwap PredictionV2 bet pools.

Era 11 (2026-05-07): replaces the WSS-subscription pool watcher.
Architecture: deterministic poll schedule using batched
``eth_getBlockReceipts``. See:
- ``var/design/rpc_polling_architecture_2026_05_07.md`` (architecture)
- ``var/incident_reports/2026_05_07_rpc_polling_spike_results.md`` (provenance)

The poller has three trigger paths:

1. **Cold-start backfill** — synchronous; runs on first
   ``set_round_phase()`` call. Catches up bet events from round-start
   to head.

2. **Periodic polls** — daemon-thread timer; every
   ``RPC_PERIODIC_POLL_INTERVAL_SECONDS``. Catches new blocks since
   last poll. Off the critical path; failures are non-fatal (next
   periodic poll retries).

3. **Ramp + final polls** — engine-driven, called from the wake
   schedule. Synchronous; deadline-aware. If RTT exceeds budget the
   poll is marked stale and ``is_pool_ready()`` returns False so the
   engine skips the round with ``pool_not_ready_last_poll_too_slow``.

Public interface mirrors ``PoolEventWatcher`` where feasible
(``get_pool``, ``set_round_phase``, ``connected``, ``current_endpoint``,
``is_pool_ready``) so the engine call sites are minimally affected.

Single-endpoint dependency: per Phase 0c spike, drpc.org rejects
batched JSON-RPC arrays. publicnode is the SOLE batched endpoint;
RPC_URLS is filtered accordingly. drpc.org survives only as a WSS
newHeads source for the unrelated NTP-clock-skew probe (and even
that's deprecated by Era 9 NTP).
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request as _urllib_req
from dataclasses import dataclass, field
from typing import Any

from pancakebot import timing_constants as _tc
from pancakebot.constants import BNB_WEI, PREDICTION_V2_CONTRACT_ADDRESS
from pancakebot.log import info, warn
from pancakebot.util import InvariantError


# Event topic hashes (keccak256 of event signatures).
_BET_BULL_TOPIC = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
_BET_BEAR_TOPIC = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"


# HTTP RPC endpoints. publicnode is the SOLE batched endpoint per
# Phase 0c spike — drpc.org rejects batched JSON-RPC arrays with
# HTTP 500 at every tested batch size. The list is single-element
# by design (the architecture accepts the single-endpoint dependency).
RPC_BATCH_ENDPOINTS: list[str] = [
    "https://bsc-rpc.publicnode.com",
]

_USER_AGENT = "pancakebot-rpc-poller/1.0"
_HTTP_TIMEOUT_SECONDS = 10
_HTTP_BATCH_TIMEOUT_SECONDS = 30


@dataclass
class _Bet:
    epoch: int
    side: str        # "Bull" or "Bear"
    amount_wei: int
    block_number: int
    block_ts: int    # block timestamp


@dataclass
class _EpochPool:
    bets: list[_Bet] = field(default_factory=list)


class RpcPoller:
    """Polls PredictionV2 bet events from BSC via batched
    ``eth_getBlockReceipts`` over HTTP.

    Replaces ``PoolEventWatcher``. Public interface intentionally
    mirrors ``PoolEventWatcher`` so the engine integration is a
    rename rather than a rework.
    """

    def __init__(
        self,
        *,
        interval_seconds: int,
        rpc_urls: list[str] | None = None,
        contract_address: str = PREDICTION_V2_CONTRACT_ADDRESS,
        periodic_poll_interval_s: int = _tc.RPC_PERIODIC_POLL_INTERVAL_SECONDS,
        batch_size: int = _tc.RPC_BATCH_BLOCK_RECEIPTS_LIMIT,
    ) -> None:
        if interval_seconds <= 0:
            raise InvariantError("interval_seconds_nonpositive")
        if periodic_poll_interval_s <= 0:
            raise InvariantError("periodic_poll_interval_nonpositive")
        if batch_size <= 0 or batch_size > _tc.RPC_BATCH_BLOCK_RECEIPTS_LIMIT:
            raise InvariantError(
                f"batch_size_out_of_range: {batch_size} "
                f"(max {_tc.RPC_BATCH_BLOCK_RECEIPTS_LIMIT})"
            )

        self._interval_seconds = int(interval_seconds)
        self._rpc_urls = list(rpc_urls) if rpc_urls is not None else list(RPC_BATCH_ENDPOINTS)
        if not self._rpc_urls:
            raise InvariantError("rpc_urls_empty")
        self._contract_addr = contract_address.lower()
        self._periodic_poll_interval_s = int(periodic_poll_interval_s)
        self._batch_size = int(batch_size)

        self._lock = threading.Lock()

        # Pool state — same shapes as PoolEventWatcher for engine compat.
        self._pools: dict[int, _EpochPool] = {}
        self._block_ts: dict[int, int] = {}
        self._seen_tx: dict[int, set[str]] = {}

        # Round-phase state (set by engine).
        self._current_epoch: int = -1
        self._lock_at: int = 0

        # Cursor: the highest block number we've polled receipts for.
        # Cold-start sets this to round-start - margin; subsequent polls
        # advance it. Periodic and ramp/final polls all read+write under
        # self._lock to keep dedup honest.
        self._last_polled_block_number: int = 0

        # Connection / readiness state.
        self._connected: bool = False  # True after cold-start completes
        self._current_endpoint: str = self._rpc_urls[0]
        self._cold_start_done: threading.Event = threading.Event()
        self._cold_start_in_progress: bool = False
        self._last_poll_succeeded: bool = False
        self._last_poll_too_slow: bool = False
        self._last_poll_at: float = 0.0
        self._last_poll_rtt_ms: int = 0
        self._last_poll_error: str = ""

        # Periodic poll daemon thread.
        self._stop_event = threading.Event()
        self._periodic_thread: threading.Thread | None = None
        # Mutex preventing concurrent polls (periodic vs ramp vs final).
        self._poll_lock = threading.Lock()

        # Counters for stats / log lines.
        self._total_events: int = 0
        self._poll_count: int = 0

    # ------------------------------------------------------------------
    # Public properties (mirror PoolEventWatcher)
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True after cold-start completes successfully."""
        return self._connected

    @property
    def current_endpoint(self) -> str:
        return self._current_endpoint

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self._connected,
                "current_endpoint": self._current_endpoint,
                "poll_count": self._poll_count,
                "last_poll_at": self._last_poll_at,
                "last_poll_rtt_ms": self._last_poll_rtt_ms,
                "last_poll_succeeded": self._last_poll_succeeded,
                "last_poll_too_slow": self._last_poll_too_slow,
                "last_polled_block": self._last_polled_block_number,
                "epochs_tracked": len(self._pools),
                "total_events": self._total_events,
            }

    def is_pool_ready(self, epoch: int | None = None) -> tuple[bool, str]:
        """Engine gate. Returns ``(True, "")`` only when:
          - cold-start has completed
          - the most recent poll succeeded
          - the most recent poll's RTT was within deadline budget

        Otherwise ``(False, reason)`` where reason is one of:
          ``"cold_start_in_progress"``, ``"last_poll_failed"``,
          ``"last_poll_too_slow"``.

        ``epoch`` parameter is currently advisory; the poller polls
        whatever blocks are recent and the engine filters by epoch
        at decision time. Reserved for future use (e.g. checking
        the polled range covers ``pool_cutoff_seconds`` before lock).
        """
        with self._lock:
            if not self._connected:
                return False, "cold_start_in_progress"
            if not self._last_poll_succeeded:
                return False, "last_poll_failed"
            if self._last_poll_too_slow:
                return False, "last_poll_too_slow"
            return True, ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the periodic-poll daemon thread. Cold-start backfill
        runs lazily on the first ``set_round_phase`` call."""
        if self._periodic_thread is not None and self._periodic_thread.is_alive():
            return
        self._stop_event.clear()
        self._periodic_thread = threading.Thread(
            target=self._periodic_loop, daemon=True, name="rpc-poller-periodic",
        )
        self._periodic_thread.start()
        info("RPC_POLL", "START", "OK",
             msg=(f"RPC poller started endpoint={self._current_endpoint} "
                  f"periodic={self._periodic_poll_interval_s}s "
                  f"batch={self._batch_size}"))

    def stop(self) -> None:
        self._stop_event.set()
        if self._periodic_thread is not None:
            self._periodic_thread.join(timeout=10)
            self._periodic_thread = None
        info("RPC_POLL", "STOP", "OK", msg="RPC poller stopped")

    # ------------------------------------------------------------------
    # Engine integration: round-phase + decision-time pool read
    # ------------------------------------------------------------------

    def set_round_phase(self, *, current_epoch: int, lock_at: int) -> None:
        """Engine-driven state sync; called at the top of every
        runtime iteration after epoch handshake.

        Same idempotence semantics as the prior PoolEventWatcher:
        ``current_epoch`` is normally strictly-advancing, but the
        engine's catch-up ``_sleep_and_claim`` path may re-call with
        the SAME epoch. Same-epoch + same-lock_at is a no-op resync;
        same-epoch + DIFFERENT lock_at raises (chain corruption);
        strictly-decreasing epochs raise.

        Triggers cold-start on the first call. Subsequent calls drop
        stale-epoch state and update tracked epochs.
        """
        if current_epoch < 0:
            raise InvariantError("set_round_phase_negative_epoch")
        if lock_at <= 0:
            raise InvariantError("set_round_phase_lock_at_nonpositive")

        is_first_call = False
        with self._lock:
            prev_epoch = self._current_epoch
            is_first_call = (prev_epoch == -1)

            if not is_first_call and current_epoch < prev_epoch:
                raise InvariantError(
                    f"set_round_phase_decreasing: prev={prev_epoch} new={current_epoch}"
                )
            if not is_first_call and current_epoch == prev_epoch:
                if self._lock_at != lock_at:
                    raise InvariantError(
                        f"set_round_phase_same_epoch_lock_at_changed: "
                        f"epoch={current_epoch} prev_lock_at={self._lock_at} "
                        f"new_lock_at={lock_at}"
                    )
                return

            if is_first_call:
                info("RPC_POLL", "EPOCH", "INIT",
                     msg=f"Initialized at epoch {current_epoch}",
                     epoch=current_epoch)
                self._current_epoch = current_epoch
            else:
                # Drop stale epochs (strictly less than new current_epoch)
                # from both _pools and _seen_tx. The "+1" next-epoch entries
                # are kept.
                stale_pools = [e for e in self._pools if e < current_epoch]
                stale_seen = [e for e in self._seen_tx if e < current_epoch]
                for e in stale_pools:
                    del self._pools[e]
                for e in stale_seen:
                    del self._seen_tx[e]
                self._current_epoch = current_epoch

            self._lock_at = lock_at

            # Bounded _block_ts: keep most recent 500 once we exceed 1000.
            if len(self._block_ts) > 1000:
                sorted_blocks = sorted(self._block_ts.keys())
                for bn in sorted_blocks[:-500]:
                    del self._block_ts[bn]

        if is_first_call:
            # Cold-start outside the lock so the periodic poller (also
            # under self._lock for store updates) doesn't deadlock with
            # us. Cold-start is synchronous: set_round_phase blocks
            # until the first round's blocks are polled. This is the
            # same behaviour as the prior PoolEventWatcher backfill
            # but inline in set_round_phase rather than triggered by
            # the recv-loop's first newhead.
            self._cold_start()

    def get_pool(self, epoch: int, *, max_ts: int) -> tuple[float, float]:
        """Return ``(bull_bnb, bear_bnb)`` from confirmed events for an
        epoch, including only bets with ``0 < block_timestamp < max_ts``.

        Same shape as ``PoolEventWatcher.get_pool``.
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

    def is_backfill_done(self) -> bool:
        """Compatibility shim: always True after cold-start. The
        periodic-poll model has no in-flight backfill window the way
        the WSS model did."""
        return self._cold_start_done.is_set()

    # ------------------------------------------------------------------
    # Engine integration: deadline-driven polls (ramp + final)
    # ------------------------------------------------------------------

    def poll_ramp(self, deadline_ms: int = 0) -> None:
        """Engine-driven ramp poll. Synchronous; blocks until
        complete or until RTT exceeds deadline_ms (0 = no deadline).

        On success: updates _last_poll_succeeded=True,
        _last_poll_too_slow=False.
        On RTT-exceeds-deadline: sets _last_poll_too_slow=True; the
        next is_pool_ready() returns (False, 'last_poll_too_slow').
        On RPC error: sets _last_poll_succeeded=False; the next
        is_pool_ready() returns (False, 'last_poll_failed').
        """
        self._poll_now(deadline_ms=deadline_ms, label="ramp")

    def poll_final(self, deadline_ms: int = 0) -> None:
        """Engine-driven final poll. Same behaviour as poll_ramp;
        named distinctly for log readability."""
        self._poll_now(deadline_ms=deadline_ms, label="final")

    # ------------------------------------------------------------------
    # Internal: cold-start + periodic + poll mechanics
    # ------------------------------------------------------------------

    def _cold_start(self) -> None:
        """Synchronous backfill from round-start block to head.
        Called from the first set_round_phase() and blocks until done.
        """
        with self._lock:
            if self._cold_start_in_progress:
                return
            self._cold_start_in_progress = True

        try:
            head = self._rpc_eth_block_number()
            if head <= 0:
                warn("RPC_POLL", "COLD", "FAIL",
                     msg="cold_start: eth_blockNumber returned 0 or error")
                return

            # Round-start ≈ lock_at - interval_seconds. Compute the
            # block at round-start (allow some safety margin for
            # block-time variance).
            blocks_per_round = int(
                (self._interval_seconds * 1000) / _tc.BSC_BLOCK_TIME_MS
            ) + 20  # safety margin
            round_start_block = max(0, head - blocks_per_round)

            with self._lock:
                self._last_polled_block_number = round_start_block - 1

            info("RPC_POLL", "COLD", "START",
                 msg=f"cold_start: backfilling {head - round_start_block + 1} blocks "
                     f"({round_start_block}..{head})")
            self._poll_now(deadline_ms=0, label="cold")

            with self._lock:
                self._connected = True
                self._cold_start_done.set()
                self._cold_start_in_progress = False

            info("RPC_POLL", "COLD", "OK",
                 msg=f"cold_start complete; {self._total_events} events recorded")

        except Exception as e:  # noqa: BLE001
            warn("RPC_POLL", "COLD", "FAIL",
                 msg=f"{type(e).__name__}: {e}")
            with self._lock:
                self._cold_start_in_progress = False

    def _periodic_loop(self) -> None:
        """Daemon-thread loop. Wakes every periodic_poll_interval_s
        and runs a poll. Idempotent if cold-start hasn't yet completed
        (no-ops in that case)."""
        while not self._stop_event.is_set():
            # Sleep first so periodic and cold-start don't collide
            # at startup.
            if self._stop_event.wait(timeout=self._periodic_poll_interval_s):
                break
            if not self._cold_start_done.is_set():
                continue
            try:
                self._poll_now(deadline_ms=0, label="periodic")
            except Exception as e:  # noqa: BLE001
                warn("RPC_POLL", "PERIODIC", "FAIL",
                     msg=f"{type(e).__name__}: {e}")

    def _poll_now(self, *, deadline_ms: int, label: str) -> None:
        """Core poll: fetch new blocks since _last_polled_block_number,
        in chunks of self._batch_size, until caught up to head.

        Updates _last_poll_succeeded / _last_poll_too_slow / etc on
        completion. ``deadline_ms`` is the soft RTT budget; if any
        single batch's RTT exceeds it, _last_poll_too_slow is set to
        True (but the poll still completes normally — we want the
        data that did come back).
        """
        if not self._poll_lock.acquire(blocking=False):
            # Another poll is in flight; skip this one. Periodic polls
            # are advisory; if a ramp/final poll is concurrent, they
            # share the same data anyway.
            return
        try:
            t_start = time.time()
            try:
                head = self._rpc_eth_block_number()
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._last_poll_succeeded = False
                    self._last_poll_error = f"head_fetch:{type(e).__name__}:{e}"
                warn("RPC_POLL", label.upper(), "ERR",
                     msg=f"eth_blockNumber: {self._last_poll_error}")
                return

            with self._lock:
                from_block = self._last_polled_block_number + 1

            if head < from_block:
                # No new blocks; nothing to do. Still update the
                # success markers so is_pool_ready stays True.
                rtt_ms = int((time.time() - t_start) * 1000)
                with self._lock:
                    self._last_poll_succeeded = True
                    self._last_poll_too_slow = False
                    self._last_poll_at = time.time()
                    self._last_poll_rtt_ms = rtt_ms
                    self._poll_count += 1
                return

            n_blocks = head - from_block + 1
            blocks_polled = 0
            error_seen: str | None = None

            # Batch in chunks. Deadline check is total-RTT-based: if
            # the cumulative time since poll start exceeds deadline_ms
            # at any point, abort remaining batches (we'll process
            # what we got and mark too_slow). The engine passes
            # deadline_ms = (next_wake_time - now) - safety.
            for batch_start in range(from_block, head + 1, self._batch_size):
                if deadline_ms > 0:
                    elapsed_ms = int((time.time() - t_start) * 1000)
                    if elapsed_ms > deadline_ms:
                        warn("RPC_POLL", label.upper(), "SLOW",
                             msg=(f"deadline exceeded after batch_start={batch_start}: "
                                  f"elapsed={elapsed_ms}ms > deadline={deadline_ms}ms; "
                                  f"aborting remaining batches"))
                        break
                batch_end = min(batch_start + self._batch_size - 1, head)
                batch_nums = list(range(batch_start, batch_end + 1))
                try:
                    self._fetch_and_process_blocks(batch_nums)
                except Exception as e:  # noqa: BLE001
                    error_seen = f"batch[{batch_start}..{batch_end}]: {type(e).__name__}: {e}"
                    warn("RPC_POLL", label.upper(), "BATCH_ERR",
                         msg=error_seen)
                    break
                blocks_polled += len(batch_nums)
                with self._lock:
                    self._last_polled_block_number = batch_end

            rtt_ms = int((time.time() - t_start) * 1000)
            too_slow = (deadline_ms > 0 and rtt_ms > deadline_ms)
            with self._lock:
                self._last_poll_at = time.time()
                self._last_poll_rtt_ms = rtt_ms
                self._poll_count += 1
                if error_seen is not None:
                    self._last_poll_succeeded = False
                    self._last_poll_error = error_seen
                else:
                    self._last_poll_succeeded = True
                    self._last_poll_error = ""
                self._last_poll_too_slow = too_slow

            info("RPC_POLL", label.upper(), "OK",
                 msg=(f"polled {blocks_polled}/{n_blocks} blocks "
                      f"({from_block}..{head}) in {rtt_ms}ms"))

        finally:
            self._poll_lock.release()

    def _fetch_and_process_blocks(self, block_numbers: list[int]) -> None:
        """Fetch eth_getBlockReceipts for each block_number in a single
        batched HTTP request, then process bet events from the
        receipts."""
        if not block_numbers:
            return
        # batched JSON-RPC: each id corresponds to one block
        calls = [
            ("eth_getBlockReceipts", [hex(bn)])
            for bn in block_numbers
        ]
        results = self._rpc_batch(calls)
        if len(results) != len(calls):
            raise InvariantError(
                f"rpc_batch_length_mismatch: expected={len(calls)} got={len(results)}"
            )
        for bn, (receipts, err) in zip(block_numbers, results):
            if err is not None:
                # Single-block error: skip; the next periodic/ramp poll
                # will retry. Don't raise here because the rest of the
                # batch might be valid.
                continue
            if not isinstance(receipts, list):
                continue
            self._process_receipts_for_block(bn, receipts)

    def _process_receipts_for_block(self, block_number: int, receipts: list[dict]) -> None:
        """Extract BetBull/BetBear events from a block's receipts and
        update the local pool state. Same dedup + epoch-gate behaviour
        as the prior PoolEventWatcher._process_bet_event."""
        for r in receipts:
            if not isinstance(r, dict):
                continue
            for log in r.get("logs", []) or []:
                if (log.get("address") or "").lower() != self._contract_addr:
                    continue
                topics = log.get("topics") or []
                if len(topics) < 3:
                    continue
                topic0 = topics[0]
                if topic0 == _BET_BULL_TOPIC:
                    side = "Bull"
                elif topic0 == _BET_BEAR_TOPIC:
                    side = "Bear"
                else:
                    continue
                try:
                    epoch = int(topics[2], 16)
                    amount_wei = int(log.get("data", "0x0"), 16)
                    bn = int(log.get("blockNumber", "0x0"), 16)
                except (ValueError, IndexError):
                    continue
                if amount_wei <= 0:
                    continue
                # Epoch gate
                if self._current_epoch >= 0 and epoch not in (
                    self._current_epoch, self._current_epoch + 1
                ):
                    continue
                tx_hash = log.get("transactionHash", "")
                log_idx = log.get("logIndex", "")
                dedup_key = f"{tx_hash}:{log_idx}"
                with self._lock:
                    seen = self._seen_tx.setdefault(epoch, set())
                    if dedup_key and dedup_key in seen:
                        continue
                    if dedup_key:
                        seen.add(dedup_key)
                    block_ts = self._block_ts.get(bn, 0)
                    if epoch not in self._pools:
                        self._pools[epoch] = _EpochPool()
                    self._pools[epoch].bets.append(_Bet(
                        epoch=epoch, side=side, amount_wei=amount_wei,
                        block_number=bn, block_ts=block_ts,
                    ))
                    self._total_events += 1

        # Block timestamp: prefer block.timestamp from any receipt's
        # parent block; if not present, leave 0 and the next get_pool
        # call will lazy-resolve from a future block fetch. For
        # eth_getBlockReceipts the block timestamp is NOT included in
        # the response — we'd need a separate eth_getBlockByNumber.
        # That's an O(blocks) extra RPC; defer. For now, lazy-resolve
        # via cached timestamps from periodic block-header fetches in
        # _maybe_resolve_block_ts.
        if block_number not in self._block_ts:
            self._maybe_resolve_block_ts(block_number)

    def _maybe_resolve_block_ts(self, block_number: int) -> None:
        """Best-effort: fetch eth_getBlockByNumber(False) for a single
        block to extract its timestamp. Errors are silent (the
        timestamp will be lazy-resolved on next poll if missing)."""
        # Lightweight single call (not batched) to keep this off the
        # batched-poll critical path.
        try:
            blk = self._rpc_call_single("eth_getBlockByNumber", [hex(block_number), False])
            if isinstance(blk, dict) and "timestamp" in blk:
                ts = int(blk["timestamp"], 16)
                with self._lock:
                    self._block_ts[block_number] = ts
        except Exception:  # noqa: BLE001
            return

    # ------------------------------------------------------------------
    # Internal: HTTP RPC helpers (single + batched)
    # ------------------------------------------------------------------

    def _rpc_eth_block_number(self) -> int:
        """Return current head block number via single eth_blockNumber
        call. Raises on error (caller marks last_poll_succeeded=False)."""
        result = self._rpc_call_single("eth_blockNumber", [])
        if not isinstance(result, str):
            raise InvariantError(f"eth_blockNumber_unexpected_result: {result!r}")
        return int(result, 16)

    def _rpc_call_single(self, method: str, params: list) -> Any:
        """Single JSON-RPC call. Raises on transport error or RPC
        error; returns the ``result`` field on success."""
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
        }).encode()
        url = self._current_endpoint
        req = _urllib_req.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
        )
        with _urllib_req.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read())
        if "error" in payload:
            raise InvariantError(f"rpc_error:{payload['error']}")
        return payload.get("result")

    def _rpc_batch(self, calls: list[tuple[str, list]]) -> list[tuple[Any, str | None]]:
        """Batched JSON-RPC call. Returns list of (result, error_str)
        parallel to calls. On transport-level failures (HTTP error,
        non-list response, id mismatch) raises -- the entire batch is
        considered failed."""
        if not calls:
            return []
        batch = [
            {"jsonrpc": "2.0", "id": i, "method": method, "params": params}
            for i, (method, params) in enumerate(calls)
        ]
        body = json.dumps(batch).encode()
        url = self._current_endpoint
        req = _urllib_req.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
        )
        with _urllib_req.urlopen(req, timeout=_HTTP_BATCH_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read())
        if not isinstance(payload, list):
            raise InvariantError(
                f"rpc_batch_non_list_response: type={type(payload).__name__}"
            )
        # Build aligned result list (sort by id; verify all ids present)
        ids_returned = sorted(r.get("id", -1) for r in payload)
        ids_expected = list(range(len(calls)))
        if ids_returned != ids_expected:
            missing = set(ids_expected) - set(ids_returned)
            extras = set(ids_returned) - set(ids_expected)
            raise InvariantError(
                f"rpc_batch_id_mismatch: missing={sorted(missing)} extras={sorted(extras)}"
            )
        by_id = {r["id"]: r for r in payload}
        results: list[tuple[Any, str | None]] = []
        for i in range(len(calls)):
            r = by_id[i]
            if "error" in r:
                results.append((None, f"rpc_error:{r['error']}"))
            else:
                results.append((r.get("result"), None))
        return results
