"""HTTP RPC poller for PancakeSwap PredictionV2 bet pools.

Era 11 (2026-05-07): replaces the WSS-subscription pool watcher.
Architecture: deterministic poll schedule using batched
``eth_getBlockReceipts``. See:
- ``var/design/rpc_polling_architecture_2026_05_07.md`` (architecture)
- ``var/incident_reports/2026_05_07_rpc_polling_spike_results.md`` (provenance)
- ``var/incident_reports/2026_05_11_parallel_request_transport_bottleneck.md``
  (transport + hedging redesign)

The poller has three trigger paths:

1. **Cold-start backfill** — synchronous; runs on first
   ``set_round_phase()`` call. Catches up bet events from round-start
   to head.

2. **Periodic polls** — daemon-thread timer; every
   ``RPC_PERIODIC_POLL_INTERVAL_SECONDS``. Catches new blocks since
   last poll. Off the critical path; failures are non-fatal (next
   periodic poll retries).

3. **Ramp + final polls** — engine-driven, called from the wake
   schedule. Synchronous; deadline-aware. RTT-exceeds-deadline marks
   ``_last_poll_too_slow=True`` for diagnostics, but skips are driven
   by the round-aware feasibility check
   (``catchup_infeasible_for_round``), not by individual slow polls.
   Single transient failures are recoverable; the integrating
   feasibility signal is what matters at decision time.

Public interface mirrors ``PoolEventWatcher`` where feasible
(``get_pool``, ``set_round_phase``, ``connected``, ``current_endpoint``,
``is_pool_ready``) so the engine call sites are minimally affected.

**Endpoint hedging (fire-to-all-pool, 2026-05-11)**: every JSON-RPC
call fires in parallel to ALL endpoints in ``DEFAULT_HEDGED_ENDPOINTS``
via a shared ``ThreadPoolExecutor``. The first successful response
wins; the rest are abandoned. There is no endpoint selection logic,
no per-endpoint health tracking, no fan-out knob — if an endpoint
misbehaves chronically, the operator removes it from the pool by
editing the constant. Validated 2026-05-11: max wallclock dropped
from 4.745s to 2.502s vs the prior pick_n + urllib transport.

Persistent HTTP/1.1 connections via ``urllib3.PoolManager`` mean each
endpoint's TLS handshake amortizes across the bot's lifetime — after
warmup, every hedged batch reuses already-open sockets.
"""
from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import urllib3

from pancakebot import timing_constants as _tc
from pancakebot.constants import BNB_WEI, PREDICTION_V2_CONTRACT_ADDRESS
from pancakebot.log import info, warn
from pancakebot.util import InvariantError


# Event topic hashes (keccak256 of event signatures).
_BET_BULL_TOPIC = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
_BET_BEAR_TOPIC = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"


# Fire-to-all-pool endpoint set. Every JSON-RPC call fans out in
# parallel to every URL in this list; the first successful response
# wins. There is no selection logic — if an endpoint misbehaves
# chronically (sustained timeouts, wrong-chain responses, etc.),
# remove it from this list manually.
#
# Last measured 2026-05-08/10 (n=200 batch=20 from Track H respike +
# 2026-05-10 pool-extension audit):
#
#   bsc-dataseed1.defibit.io   p50=770ms  p99=2226ms  (BSC dataseed family)
#   bsc-dataseed1.ninicoin.io  p50=802ms  p99=2179ms  (BSC dataseed family)
#   bsc-dataseed1.binance.org  p50=828ms  p99=1797ms  (BSC dataseed family)
#   bsc-dataseed3.binance.org  p50=898ms  p99=1290ms  (BSC dataseed family)
#   bsc-rpc.publicnode.com     p50=938ms  p99=1842ms  (Allnodes, distinct)
#   bsc.rpc.blxrbdn.com        p50~250ms  batch~430ms (bloXroute, distinct)
#
# Two distinct-provider endpoints (publicnode, bloXroute) hedge
# against correlated outages on the bsc-dataseed-family infrastructure
# (observed 2026-05-09/10: hours-long windows where ALL bsc-dataseed*
# timed out simultaneously).
DEFAULT_HEDGED_ENDPOINTS: list[str] = [
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-dataseed1.ninicoin.io",
    "https://bsc-dataseed1.binance.org",
    "https://bsc-dataseed3.binance.org",
    "https://bsc-rpc.publicnode.com",
    "https://bsc.rpc.blxrbdn.com",
]

_USER_AGENT = "pancakebot-rpc-poller/1.0"


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


class HedgedAllFailed(Exception):
    """Composite exception raised when every endpoint in a hedged
    fan-out fails. Carries the per-endpoint (endpoint, exception)
    pairs so the operator log line surfaces all failures at once.
    """

    def __init__(self, errors: list[tuple[str, BaseException]]) -> None:
        self.errors: list[tuple[str, BaseException]] = list(errors)
        msg_parts = [
            f"{endpoint}: {type(e).__name__}: {e}"
            for endpoint, e in errors
        ]
        super().__init__(
            f"all_hedged_endpoints_failed ({len(errors)}): "
            + "; ".join(msg_parts)
        )


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
        endpoint_pool: list[str],
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

        pool = list(endpoint_pool)
        if not pool:
            raise InvariantError("endpoint_pool_empty")

        self._interval_seconds = int(interval_seconds)
        self._endpoint_pool: list[str] = pool

        # ThreadPoolExecutor for parallel fan-out across the full pool.
        # Sized to len(pool) so every endpoint can fire concurrently
        # without queueing. Always constructed (single-endpoint case is
        # supported via a pool of length 1).
        self._executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(self._endpoint_pool)),
                thread_name_prefix="rpc-hedge",
            )
        )

        # urllib3 PoolManager: persistent HTTP/1.1 connections per host.
        # Eliminates per-call DNS+TCP+TLS handshake cost — the bottleneck
        # that caused parallel calls to exceed the 5s deadline at ~10%
        # rate under bare urllib. Validated 2026-05-11 (max wallclock
        # 4.745s → 2.502s); see
        # var/incident_reports/2026_05_11_parallel_request_transport_bottleneck.md.
        # ``num_pools`` and ``maxsize`` both sized to len(pool) so every
        # endpoint can hold one persistent connection.
        self._pool: urllib3.PoolManager = urllib3.PoolManager(
            num_pools=max(1, len(self._endpoint_pool)),
            maxsize=max(1, len(self._endpoint_pool)),
            headers={
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/json",
            },
        )

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
        # Most-recently-used endpoint URL (informational; updated by
        # the hedging transport on each successful call). Display/log
        # only — every call still fires to every endpoint in the pool.
        self._current_endpoint: str = self._endpoint_pool[0]
        self._cold_start_done: threading.Event = threading.Event()
        self._cold_start_in_progress: bool = False
        self._last_poll_succeeded: bool = False
        self._last_poll_too_slow: bool = False
        self._last_poll_at: float = 0.0
        self._last_poll_rtt_ms: int = 0
        self._last_poll_error: str = ""

        # When True, math says we cannot catch up to head in time for
        # the current round's lock_at. Set by set_round_phase (after
        # the cursor clamp) and by _poll_now (if RTT degrades mid-round).
        # Reset on epoch advance.
        self._catchup_infeasible_for_round: bool = False

        # True while a poll is actively fetching/processing blocks.
        # is_pool_ready returns False when this is set so the engine
        # cannot read a half-built pool aggregate. Set/cleared under
        # self._lock; bracketed around _poll_now's batch-fetch loop.
        self._poll_in_progress: bool = False

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
                "endpoint_pool_size": len(self._endpoint_pool),
            }

    def is_pool_ready(self, epoch: int | None = None) -> tuple[bool, str]:
        """Engine gate. Returns ``(True, "")`` when the bot can place a
        bet for the current round; otherwise ``(False, reason)``.

        Skip reasons:
          - ``"cold_start_in_progress"`` — initial backfill not done.
          - ``"catchup_infeasible_for_round"`` — math says we cannot
            catch up to head before the current round's lock_at.
          - ``"poll_in_progress"`` — a poll is actively fetching;
            the pool aggregate is mid-build. Read after the poll
            completes.

        Notably we DO NOT skip on ``last_poll_succeeded == False`` or
        ``last_poll_too_slow``. A single poll failure or slow poll is
        informational (the next periodic poll might recover); the
        feasibility check in ``_on_epoch_advance`` and ``_poll_now`` is
        the integrating signal that decides whether we have time to
        catch up given current observed conditions. ``_last_poll_*``
        fields are still maintained for diagnostics/stats.

        ``epoch`` parameter is currently advisory; the poller polls
        whatever blocks are recent and the engine filters by epoch
        at decision time. Reserved for future use (e.g. checking
        the polled range covers ``pool_cutoff_seconds`` before lock).
        """
        with self._lock:
            if not self._connected:
                return False, "cold_start_in_progress"
            if self._catchup_infeasible_for_round:
                return False, "catchup_infeasible_for_round"
            if self._poll_in_progress:
                return False, "poll_in_progress"
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
        # wait=False — abandoned hedged requests should not block
        # shutdown. The PoolManager has no real cancellation; the
        # in-flight sockets will time out on their own.
        self._executor.shutdown(wait=False)
        # Drain the urllib3 connection pool — closes any persistent
        # sockets so the process exits cleanly.
        self._pool.clear()
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
        is_epoch_advance = False
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
                is_epoch_advance = True

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
        elif is_epoch_advance:
            # Round-aware cursor clamp + catch-up feasibility check.
            # Past rounds are archive-only — the bot only bets on the
            # CURRENT round, so polling cursor must not lag into prior
            # rounds. Failed RPC calls leave state untouched and the
            # next epoch advance retries.
            self._on_epoch_advance(lock_at=lock_at, current_epoch=current_epoch)

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
    # Round-aware cursor clamp + catch-up feasibility check
    # ------------------------------------------------------------------

    def _on_epoch_advance(self, *, lock_at: int, current_epoch: int) -> None:
        """Round-aware bookkeeping at epoch boundaries.

        Two responsibilities:

        1. **Cursor clamp**: advance ``_last_polled_block_number`` to the
           current round's start block (or just behind it). Forward-only —
           never rewinds, so normal in-round operation is a no-op. After
           a publicnode outage spanning N rounds, the clamp jumps the
           cursor forward, skipping ~N*660 stale-round blocks; past
           rounds are archive-only.

        2. **Feasibility check**: compute estimated catch-up wallclock
           from blocks-behind and the single-batch p99 RTT. If the
           estimate exceeds time-until-lock, set
           ``_catchup_infeasible_for_round`` so the engine skips with
           reason ``catchup_infeasible_for_round``.

        Both halves degrade gracefully on RPC failure: if the RPC calls
        needed for either step error out, we leave state untouched and
        rely on the next epoch advance to retry.
        """
        # Always reset the infeasibility flag at round start; otherwise
        # a past-round flag would carry forward into rounds where the
        # cursor has been clamped and catch-up is now feasible.
        with self._lock:
            self._catchup_infeasible_for_round = False

        round_start_ts = lock_at - self._interval_seconds
        rs_block = self._compute_round_start_block(round_start_ts)
        if rs_block is None:
            return  # RPC + cache both failed; leave state, retry next round

        # Forward-only cursor advance.
        with self._lock:
            prev_cursor = self._last_polled_block_number
            new_cursor = rs_block - 1  # re-poll round_start block itself
        if new_cursor > prev_cursor:
            with self._lock:
                self._last_polled_block_number = new_cursor
            info("RPC_POLL", "EPOCH", "RESET",
                 msg=(f"cursor {prev_cursor}->{new_cursor} "
                      f"(skipped {new_cursor - prev_cursor} stale-round blocks)"))

        # Feasibility check: how far behind are we vs how much time
        # remains, with a single fresh head fetch.
        try:
            head = self._rpc_eth_block_number()
        except Exception:  # noqa: BLE001
            return  # leave _catchup_infeasible_for_round at False; next
                    # poll/round will reassess.

        with self._lock:
            cursor = self._last_polled_block_number
        blocks_behind = max(0, head - cursor)
        if blocks_behind == 0:
            return

        if self._is_catchup_infeasible(blocks_behind=blocks_behind, lock_at=lock_at):
            with self._lock:
                self._catchup_infeasible_for_round = True
            time_until_lock_ms = max(0, int((lock_at - time.time()) * 1000))
            warn("RPC_POLL", "CATCH", "INFEAS",
                 msg=(f"behind {blocks_behind} blocks, "
                      f"est {self._estimated_catchup_ms(blocks_behind)}ms "
                      f"> avail {self._available_catchup_ms(time_until_lock_ms)}ms; "
                      f"skipping round {current_epoch}"))

    def _compute_round_start_block(self, round_start_ts: int) -> int | None:
        """Return the block-number whose timestamp ~= round_start_ts.

        Strategy:

        1. **Cache lookup** (free): pick the newest entry in
           ``_block_ts`` with ts <= round_start_ts and within 60s of it,
           then extrapolate forward by ``BSC_BLOCK_TIME_MS``.
        2. **RPC fallback**: ``eth_getBlockByNumber("latest", false)``
           returns ``(head_num, head_ts)``; extrapolate backward.

        Returns None if RPC fails AND cache is empty/stale.
        """
        # Method 1 — cache lookup.
        with self._lock:
            cached = [(b, t) for b, t in self._block_ts.items()
                      if t > 0 and t <= round_start_ts]
        if cached:
            b, t = max(cached, key=lambda x: x[1])
            # Reject anchors more than 60s before round_start_ts —
            # extrapolation accuracy degrades with distance.
            if round_start_ts - t <= 60:
                delta_blocks = round((round_start_ts - t) * 1000
                                     / _tc.BSC_BLOCK_TIME_MS)
                return b + delta_blocks

        # Method 2 — RPC fallback.
        try:
            head_num, head_ts = self._rpc_eth_get_latest_block_header()
        except Exception:  # noqa: BLE001
            return None
        if head_ts <= 0 or head_num <= 0:
            return None
        if head_ts <= round_start_ts:
            # Round hasn't started yet according to head-ts — treat the
            # cursor as already past.
            return head_num
        delta_blocks = round((head_ts - round_start_ts) * 1000
                             / _tc.BSC_BLOCK_TIME_MS)
        return max(0, head_num - delta_blocks)

    def _estimated_catchup_ms(self, blocks_behind: int) -> int:
        """Estimated wallclock to fetch ``blocks_behind`` blocks at the
        per-batch p99 RTT. Conservative — doesn't account for current
        degradation, and uses the static p99 table not a live observed
        p99.
        """
        if blocks_behind <= 0:
            return 0
        batches = (blocks_behind + self._batch_size - 1) // self._batch_size
        rtt_p99 = _tc.rpc_rtt_p99_for_batch(self._batch_size)
        return batches * rtt_p99

    def _available_catchup_ms(self, time_until_lock_ms: int) -> int:
        """Time available for catch-up, with the same safety buffer
        the deadline-driven polls use."""
        return max(0, time_until_lock_ms - _tc.RPC_POLL_FINAL_SAFETY_BUFFER_MS)

    def _is_catchup_infeasible(self, *, blocks_behind: int, lock_at: int) -> bool:
        """Return True if estimated catch-up wallclock exceeds the time
        budget remaining before lock_at."""
        if blocks_behind <= 0 or lock_at <= 0:
            return False
        estimated_ms = self._estimated_catchup_ms(blocks_behind)
        time_until_lock_ms = max(0, int((lock_at - time.time()) * 1000))
        available_ms = self._available_catchup_ms(time_until_lock_ms)
        return estimated_ms > available_ms

    # ------------------------------------------------------------------
    # Engine integration: deadline-driven polls (ramp + final)
    # ------------------------------------------------------------------

    def poll_ramp(self, deadline_ms: int = 0) -> None:
        """Engine-driven ramp poll. Synchronous; blocks until complete
        or until RTT exceeds deadline_ms (0 = no deadline).

        Side-effects (diagnostics only — none of these directly cause
        round skips; the round-aware feasibility check is the canonical
        skip signal):
          - On success: _last_poll_succeeded=True, _last_poll_too_slow=False.
          - On RTT-exceeds-deadline: _last_poll_too_slow=True.
          - On RPC error: _last_poll_succeeded=False.
        Skips are driven by ``_catchup_infeasible_for_round`` which the
        feasibility check (in _on_epoch_advance and _poll_now) sets when
        math says we cannot catch up before lock_at.
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
        """Synchronous backfill scoped to the CURRENT round only.
        Called from the first set_round_phase() and blocks until done.

        Round-aware: round_start_block is derived from
        ``lock_at - interval_seconds``, NOT from a head-relative
        full-round lookback. Past-round blocks are archive-only and
        never bet on, so backfilling them is wasted work.

        Feasibility-aware: if there isn't enough time to backfill the
        in-round blocks before lock_at, skip the backfill (cursor jumps
        to head), mark the round catch-up-infeasible, and let the next
        round start clean.
        """
        with self._lock:
            if self._cold_start_in_progress:
                return
            self._cold_start_in_progress = True

        try:
            # eth_getBlockByNumber('latest') returns head_number AND
            # head_timestamp in one call — both needed to derive
            # round_start_block from lock_at - interval_seconds.
            try:
                head, head_ts = self._rpc_eth_get_latest_block_header()
            except Exception as e:  # noqa: BLE001
                warn("RPC_POLL", "COLD", "FAIL",
                     msg=(f"cold_start: eth_getBlockByNumber(latest): "
                          f"{type(e).__name__}: {e}"))
                return
            if head <= 0 or head_ts <= 0:
                warn("RPC_POLL", "COLD", "FAIL",
                     msg=f"cold_start: invalid header head={head} ts={head_ts}")
                return

            with self._lock:
                lock_at_local = self._lock_at
            round_start_ts = lock_at_local - self._interval_seconds

            if head_ts <= round_start_ts:
                # Head is behind round_start (chain hasn't caught up
                # to round_start yet, or lock_at is in the future
                # beyond head_ts). Nothing to backfill — set cursor
                # at head and let periodic polls drive forward.
                with self._lock:
                    self._last_polled_block_number = head
                    self._connected = True
                    self._cold_start_done.set()
                    self._cold_start_in_progress = False
                info("RPC_POLL", "COLD", "OK",
                     msg=(f"cold_start: head_ts {head_ts} <= "
                          f"round_start_ts {round_start_ts}; no backfill "
                          f"needed (cursor at head={head})"))
                return

            # delta_blocks: how many blocks since round_start.
            # BSC_BLOCK_TIME_MS=500 is the CONSERVATIVE rounding of
            # empirical 452ms; using it as the divisor underestimates
            # blocks-elapsed by ~10%. The +20 safety margin handles
            # that bias plus general block-time variance, ensuring we
            # don't miss start-of-round blocks. Any over-fetched
            # prev-round blocks are filtered by the epoch gate in
            # _process_receipts.
            delta_blocks = round(
                (head_ts - round_start_ts) * 1000
                / _tc.BSC_BLOCK_TIME_MS
            ) + 20
            round_start_block = max(0, head - delta_blocks)
            blocks_to_backfill = max(0, head - round_start_block + 1)

            # Feasibility check: can we backfill the in-round range
            # before lock_at? Same math as Component 4 trigger A.
            if self._is_catchup_infeasible(
                blocks_behind=blocks_to_backfill, lock_at=lock_at_local,
            ):
                with self._lock:
                    self._catchup_infeasible_for_round = True
                    # Advance cursor past the un-backfilled range so
                    # the periodic poll doesn't try to refill it on
                    # the next tick.
                    self._last_polled_block_number = head
                    self._connected = True
                    self._cold_start_done.set()
                    self._cold_start_in_progress = False
                time_until_lock_ms = max(
                    0, int((lock_at_local - time.time()) * 1000),
                )
                warn("RPC_POLL", "COLD", "INFEAS",
                     msg=(f"cold_start: backfill {blocks_to_backfill} blocks "
                          f"would take ~{self._estimated_catchup_ms(blocks_to_backfill)}ms "
                          f"> {self._available_catchup_ms(time_until_lock_ms)}ms "
                          f"available; skipping backfill, "
                          f"will resume on next round"))
                return

            with self._lock:
                self._last_polled_block_number = round_start_block - 1

            info("RPC_POLL", "COLD", "START",
                 msg=f"cold_start: backfilling {blocks_to_backfill} blocks "
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
        (no-ops in that case).

        Label "period" (6 chars) fits log _SUB_W=6 — a prior version
        used "periodic" (8 chars) which raised InvariantError in log.py
        on the first periodic-poll log call, killing the daemon thread
        silently and leaving ramp/final polls to catch up many minutes
        of blocks at once.
        """
        while not self._stop_event.is_set():
            # Sleep first so periodic and cold-start don't collide
            # at startup.
            if self._stop_event.wait(timeout=self._periodic_poll_interval_s):
                break
            if not self._cold_start_done.is_set():
                continue
            try:
                self._poll_now(deadline_ms=0, label="period")
            except Exception as e:  # noqa: BLE001
                warn("RPC_POLL", "PERIOD", "FAIL",
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
        with self._lock:
            self._poll_in_progress = True
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
                lock_at_local = self._lock_at

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

            # Mid-round feasibility check: if RTT degrades during the
            # round, a periodic poll might find that math says it can't
            # catch up before lock_at. Abort early — no batches fetched,
            # set _catchup_infeasible_for_round so the engine skips.
            blocks_to_catchup = head - from_block + 1
            if self._is_catchup_infeasible(
                blocks_behind=blocks_to_catchup, lock_at=lock_at_local,
            ):
                with self._lock:
                    self._catchup_infeasible_for_round = True
                time_until_lock_ms = max(0, int((lock_at_local - time.time()) * 1000))
                warn("RPC_POLL", label.upper(), "INFEAS",
                     msg=(f"behind {blocks_to_catchup} blocks, "
                          f"est {self._estimated_catchup_ms(blocks_to_catchup)}ms "
                          f"> avail {self._available_catchup_ms(time_until_lock_ms)}ms; "
                          f"aborting poll"))
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
                    # Transient publicnode failures are expected; the
                    # cursor advance + feasibility check together prevent
                    # the catch-up backlog from compounding. INFO severity
                    # avoids alert noise on routine outages.
                    info("RPC_POLL", label.upper(), "BATCH_FAIL",
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

            # Status taxonomy:
            #   OK       - full poll succeeded
            #   PARTIAL  - some batches succeeded, then error_seen
            #   EMPTY    - zero batches succeeded (error_seen != None)
            #              OR endpoint returned empty for valid range
            # Severity: INFO for the timeout-driven cases (transient and
            # expected). WARN only when we got an empty-but-no-error
            # reply for a valid range — that IS unusual.
            if error_seen is None:
                status = "OK"
                emit = info
            elif blocks_polled == 0:
                status = "EMPTY"
                emit = info
            else:
                status = "PARTIAL"
                emit = info
            if blocks_polled == 0 and error_seen is None and n_blocks > 0:
                # Endpoint returned empty for a valid range — rare and
                # worth flagging.
                status = "EMPTY"
                emit = warn
            emit("RPC_POLL", label.upper(), status,
                 msg=(f"polled {blocks_polled}/{n_blocks} blocks "
                      f"({from_block}..{head}) in {rtt_ms}ms"))

        finally:
            with self._lock:
                self._poll_in_progress = False
            self._poll_lock.release()

    def _fetch_and_process_blocks(self, block_numbers: list[int]) -> None:
        """Fetch eth_getBlockReceipts AND eth_getBlockByNumber (header
        only, full_txs=False) for each block in a SINGLE batched HTTP
        request, then process bet events + cache block timestamps.

        Why bundle the header fetch into the same batch (Era 11 fix):
        eth_getBlockReceipts does NOT include block.timestamp in its
        response, but the engine needs block_ts to filter the pool
        aggregate by pool_cutoff_seconds. The first implementation
        called eth_getBlockByNumber lazily as a separate single HTTP
        call PER BLOCK, which added ~150ms per block (= ~3000ms for a
        20-block batch on top of the actual receipts batch). Bundling
        both calls into the same batched JSON-RPC array means a single
        HTTP roundtrip per batch covers both.
        """
        if not block_numbers:
            return
        # Each block contributes TWO sub-calls: receipts + header.
        # Sub-call layout: [recv(0), hdr(0), recv(1), hdr(1), ...].
        calls: list[tuple[str, list]] = []
        for bn in block_numbers:
            calls.append(("eth_getBlockReceipts", [hex(bn)]))
            calls.append(("eth_getBlockByNumber", [hex(bn), False]))
        results = self._rpc_batch(calls)
        if len(results) != len(calls):
            raise InvariantError(
                f"rpc_batch_length_mismatch: expected={len(calls)} got={len(results)}"
            )
        for i, bn in enumerate(block_numbers):
            receipts, recv_err = results[2 * i]
            header, hdr_err = results[2 * i + 1]
            # Cache block timestamp first; needed by _process_receipts
            # so newly-stored bets have non-zero block_ts at insert time
            # (avoids the lazy-resolve path inside get_pool).
            if hdr_err is None and isinstance(header, dict):
                ts_hex = header.get("timestamp")
                if isinstance(ts_hex, str):
                    try:
                        ts = int(ts_hex, 16)
                    except ValueError:
                        ts = 0
                    if ts > 0:
                        with self._lock:
                            self._block_ts[bn] = ts
            if recv_err is not None:
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

        # Block timestamp is cached by _fetch_and_process_blocks BEFORE
        # this call (the batched RPC includes eth_getBlockByNumber for
        # every block alongside eth_getBlockReceipts). No per-block
        # follow-up fetch needed; the get_pool() lazy-resolve path is
        # the safety net for any block that wasn't in a batch.

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

    def _rpc_eth_get_latest_block_header(self) -> tuple[int, int]:
        """Return ``(head_block_number, head_block_timestamp)`` via a
        single ``eth_getBlockByNumber("latest", false)`` call. Used by
        the round-start clamp's RPC fallback path. Raises on error."""
        result = self._rpc_call_single("eth_getBlockByNumber", ["latest", False])
        if not isinstance(result, dict):
            raise InvariantError(
                f"eth_getBlockByNumber_unexpected_result: {result!r}"
            )
        num_hex = result.get("number")
        ts_hex = result.get("timestamp")
        if not isinstance(num_hex, str) or not isinstance(ts_hex, str):
            raise InvariantError(
                f"eth_getBlockByNumber_missing_fields: {result!r}"
            )
        return int(num_hex, 16), int(ts_hex, 16)

    def _rpc_post(self, url: str, body: bytes, *, timeout_seconds: int) -> bytes:
        """Single-endpoint HTTP POST via the shared urllib3 PoolManager.
        Returns the raw response body. Raises on transport-level failure
        (``urllib3.exceptions.HTTPError`` subclasses: TimeoutError,
        MaxRetryError, ConnectTimeoutError, etc.) or non-200 status.
        The caller parses JSON and decodes the JSON-RPC envelope.

        Persistent connections via the shared PoolManager mean the
        first call to each host pays the TLS handshake; subsequent
        calls reuse the open connection. See
        var/incident_reports/2026_05_11_parallel_request_transport_bottleneck.md
        for measured impact.
        """
        resp = self._pool.request(
            "POST", url, body=body,
            timeout=urllib3.Timeout(
                connect=float(timeout_seconds),
                read=float(timeout_seconds),
            ),
            retries=False,
        )
        if resp.status != 200:
            raise urllib3.exceptions.HTTPError(
                f"http_{resp.status}: {resp.reason}"
            )
        return resp.data

    def _do_hedged_post(self, body: bytes, *, timeout_seconds: int) -> tuple[str, bytes]:
        """Hedged HTTP POST against every endpoint in the pool.

        Fires one request per endpoint in parallel; the first endpoint
        to return a 200 wins. The rest are abandoned (urllib3 has no
        real cancellation — abandoned sockets time out on their own
        and free their executor worker).

        Returns ``(winner_endpoint, response_bytes)``. Raises
        ``HedgedAllFailed`` (with the per-endpoint exceptions) when
        every endpoint fails before ``timeout_seconds``.
        """
        # Special-case length 1 to skip executor overhead. Same call
        # shape as the multi-endpoint path so callers see no difference.
        if len(self._endpoint_pool) == 1:
            url = self._endpoint_pool[0]
            try:
                resp = self._rpc_post(url, body, timeout_seconds=timeout_seconds)
            except BaseException as e:  # noqa: BLE001
                raise HedgedAllFailed([(url, e)]) from e
            self._current_endpoint = url
            return url, resp

        # Fire one request per endpoint.
        fut_to_endpoint: dict[concurrent.futures.Future, str] = {
            self._executor.submit(
                self._rpc_post, ep, body, timeout_seconds=timeout_seconds,
            ): ep
            for ep in self._endpoint_pool
        }

        pending = set(fut_to_endpoint.keys())
        errors: list[tuple[str, BaseException]] = []
        deadline = time.monotonic() + float(timeout_seconds)

        while pending:
            remaining = max(0.001, deadline - time.monotonic())
            done, pending = concurrent.futures.wait(
                pending,
                timeout=remaining,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                # Deadline fired; record the still-pending as timeouts.
                for fut in pending:
                    errors.append((
                        fut_to_endpoint[fut],
                        TimeoutError(f"hedged_timeout_after_{timeout_seconds}s"),
                    ))
                break
            for fut in done:
                ep = fut_to_endpoint[fut]
                try:
                    resp = fut.result()
                except BaseException as e:  # noqa: BLE001
                    errors.append((ep, e))
                    continue
                # First success wins. Pending futures are abandoned
                # (no cancellation; their sockets time out on their
                # own and the executor reclaims the workers).
                self._current_endpoint = ep
                return ep, resp

        raise HedgedAllFailed(errors)

    def _rpc_call_single(self, method: str, params: list) -> Any:
        """Single JSON-RPC call. Raises on transport error or RPC
        error; returns the ``result`` field on success. Hedged across
        every endpoint in the pool; first success wins.
        """
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
        }).encode()
        _ep, resp_bytes = self._do_hedged_post(
            body, timeout_seconds=_tc.RPC_HTTP_SINGLE_TIMEOUT_SECONDS,
        )
        payload = json.loads(resp_bytes)
        if "error" in payload:
            raise InvariantError(f"rpc_error:{payload['error']}")
        return payload.get("result")

    def _rpc_batch(self, calls: list[tuple[str, list]]) -> list[tuple[Any, str | None]]:
        """Batched JSON-RPC call. Returns list of (result, error_str)
        parallel to calls. On transport-level failures (HTTP error,
        non-list response, id mismatch) raises -- the entire batch is
        considered failed. Hedged across every endpoint in the pool;
        first endpoint to return a well-formed list response wins.
        """
        if not calls:
            return []
        batch = [
            {"jsonrpc": "2.0", "id": i, "method": method, "params": params}
            for i, (method, params) in enumerate(calls)
        ]
        body = json.dumps(batch).encode()
        _ep, resp_bytes = self._do_hedged_post(
            body, timeout_seconds=_tc.RPC_HTTP_BATCH_TIMEOUT_SECONDS,
        )
        payload = json.loads(resp_bytes)
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
