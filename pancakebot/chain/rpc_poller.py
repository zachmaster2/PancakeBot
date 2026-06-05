"""HTTP RPC poller for PancakeSwap PredictionV2 bet pools.

Era 11 (2026-05-07): replaces the WSS-subscription pool watcher.
Architecture: deterministic poll schedule using batched
``eth_getBlockReceipts``. See:
- ``var/design/rpc_polling_architecture_2026_05_07.md`` (architecture)
- ``var/incident_reports/2026_05_07_rpc_polling_spike_results.md`` (provenance)
- ``var/incident_reports/2026_05_11_parallel_request_transport_bottleneck.md``
  (transport + hedging redesign)

The poller has three trigger paths:

1. **Cursor initialization** — synchronous; runs on first
   ``set_round_phase()`` call (``_initialize_cursor_from_head``).
   One ``eth_getBlockByNumber(latest)`` RPC to derive the in-round
   ``round_start_block`` from ``lock_at - interval_seconds``, sets
   the cursor, returns. Bundle 2 (2026-05-13) replaced the prior
   synchronous backfill (30-90s engine block) with this fast path;
   the actual catch-up runs via the periodic daemon's first tick.

2. **Periodic polls** — daemon-thread timer; lock-anchored cadence
   (``round_open + k * RPC_PERIODIC_POLL_INTERVAL_SECONDS``). Catches
   new blocks since last poll AND drives the initial post-cursor-init
   catch-up. The first successful ``_poll_now`` flips
   ``_connected`` + ``_cold_start_done`` via
   ``_latch_first_successful_poll_locked``; until then
   ``is_pool_ready`` returns ``cold_start_in_progress``. Off the
   critical path; failures are non-fatal (next periodic poll retries).

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
call fires in parallel to ALL endpoints in ``READ_PATH_HEDGED_ENDPOINTS``
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
from pancakebot.runtime.regime_telemetry import (
    RollingMedianDriftMonitor,
    RollingRateMonitor,
)
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
# 3-endpoint pool (Bundle 6 trim, 2026-05-15): one endpoint per
# fault domain. Per-endpoint probe (n=15 isolated calls, 6s spacing)
# from research/probe_per_endpoint_isolated_2026_05_15.py:
#
#                                anchor p50 / p95     batch p50 / p95
#   bsc-dataseed1.binance.org    30ms / 837ms         1914ms / 2717ms   (AS14618 AWS EC2)
#   bsc-dataseed1.defibit.io     128ms / 1194ms       1496ms / 14760ms  (AS16509 AWS GA)
#   bsc-rpc.publicnode.com       103ms / 822ms        2357ms / 10816ms  (AS13335 Cloudflare)
#
# Dropped (Bundle 6 trim):
#   - bsc-dataseed3.binance.org  identical IP pool to dataseed1 (same
#                                AWS EC2 backend; pure redundancy)
#   - bsc-dataseed1.ninicoin.io  same AS16509 family as defibit; worst
#                                anchor p50 (525ms) in that family
#   - bsc.rpc.blxrbdn.com        same AS16509 family as defibit;
#                                middling on both metrics
#
# Rationale: 5 of the prior 6 endpoints were AWS-hosted (3 in AS16509,
# 2 in AS14618). Cold-start burst (~20 batches × 6 endpoints = 120
# concurrent in-flight connections) was triggering all-pool
# HedgedAllFailed timeouts at the ~400-block-backlog scale (verified
# at PID 2096 / PID 10656 spawns: 7 and 2 BATCH_FAILs respectively).
# Trimming to one endpoint per ASN-family reduces concurrent in-flight
# to ~60 and gives each AWS family 1× pool load instead of 2-3×.
#
# Fault diversity is preserved (or improved): AWS EC2 + AWS GA +
# Cloudflare. If AWS us-east goes down entirely, publicnode remains.
READ_PATH_HEDGED_ENDPOINTS: list[str] = [
    "https://bsc-dataseed1.binance.org",
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-rpc.publicnode.com",
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


@dataclass
class AnchorState:
    """BEP-520 ms-precise chain anchor for Bundle 4 dynamic deadline math.

    Captured from any observed block header (coarse-phase: existing batched
    eth_getBlockByNumber/eth_getBlockReceipts calls; fine-phase: dedicated
    latest-block polls during the last ~2s before critical_path wake).

    Monotonic on ``block_number``: updates only when the incoming block_number
    is strictly greater than the current anchor's. ``observed_at_local_ms`` is
    a local wallclock timestamp (microsecond-precision time.time() multiplied
    by 1000) at the moment the RPC response was parsed, used for staleness
    checks / debugging only — the dynamic deadline prediction uses
    ``milli_ts`` directly and extrapolates forward by exact 450ms increments.
    """
    block_number: int
    milli_ts: int           # BEP-520 MilliTimestamp = header.Time*1000 + mix_ms
    observed_at_local_ms: int


def decode_mixhash_ms(mix_hash_hex: str) -> int | None:
    """Extract the BEP-520 millisecond component from a block's mixHash.

    Returns ms in [0, 999], or ``None`` if the value is out of range
    (defensive guard against pre-Lorentz blocks where mixHash holds
    arbitrary content).

    Encoding (per BEP-520, Lorentz hardfork April 2025): the last 2 bytes
    of the 32-byte mixHash carry the millisecond component as big-endian
    uint16. Range observed in the chain (Bundle 4 reconnaissance
    2026-05-13): values quantized to 50ms multiples in [0, 950].
    """
    if not isinstance(mix_hash_hex, str):
        return None
    raw = mix_hash_hex[2:] if mix_hash_hex.startswith("0x") else mix_hash_hex
    if len(raw) != 64:
        return None
    try:
        ms = int(raw[-4:], 16)
    except ValueError:
        return None
    if not (0 <= ms <= 999):
        return None
    return ms


def compute_milli_ts(block_dict: dict) -> int | None:
    """Reconstruct full BEP-520 MilliTimestamp from a block-header dict.

    Returns ``header.Time * 1000 + mix_ms`` if the mixHash decode succeeds,
    or ``None`` if the encoding is missing/malformed (caller should fall
    back to static-deadline timing for that round).
    """
    ts_hex = block_dict.get("timestamp")
    mh_hex = block_dict.get("mixHash")
    if not isinstance(ts_hex, str) or not isinstance(mh_hex, str):
        return None
    ms = decode_mixhash_ms(mh_hex)
    if ms is None:
        return None
    try:
        ts_s = int(ts_hex, 16)
    except ValueError:
        return None
    return ts_s * 1000 + ms


def predict_predecessor_milli_ts(
    *, anchor_milli_ts: int, lock_ms: int,
    block_time_ms: int = None, jitter_ms: int = 10,
) -> int:
    """Predict the chain-time of the block immediately BEFORE the lock block.

    The lock block is the first block whose ``milli_ts >= lock_ms``. Its
    predecessor is one ``block_time_ms`` earlier — that's the latest block
    that can include our bet TX. The bet must reach the validator's
    mempool before THAT block's assembly window closes.

    Math: extrapolate forward from the anchor by exact 450ms increments.
    Use ``math.ceil(...)`` to find the smallest ``k`` such that
    ``anchor_milli_ts + k * block_time_ms >= lock_ms``; that's the lock
    block's milli_ts. Predecessor = lock_block_milli_ts - block_time_ms.

    The ``jitter_ms`` parameter (default 10) shaves a few ms off lock_ms
    in the ceil calculation to handle the edge case where the predicted
    lock-block milli_ts lands EXACTLY on lock_ms (in which case our bet's
    target window is the block *before* that one — still correct, but the
    formula needs the slight pull-back to round into the correct slot).

    No safety margin beyond ``jitter_ms``: the empirical "misses only
    delay, never advance" property (verified across 5000 consecutive
    blocks) makes the prediction conservative by construction — every
    slot miss gives us EXTRA budget, never less.
    """
    import math
    if block_time_ms is None:
        block_time_ms = _tc.BSC_BLOCK_TIME_MS
    delta = lock_ms - jitter_ms - anchor_milli_ts
    k = math.ceil(delta / block_time_ms)
    predicted_lock_block_milli_ts = anchor_milli_ts + k * block_time_ms
    return predicted_lock_block_milli_ts - block_time_ms


def compute_submit_deadline_ms(
    *, predicted_predecessor_milli_ts: int, lock_ms: int,
    block_time_ms: int = None, quantum_ms: int = None,
    assembly_window_ms: int = None, one_way_ms: int = None,
) -> int:
    """Per-round dynamic bet-submit deadline (Unix epoch ms, local wallclock).

    Returns the local-clock millisecond by which
    ``contract.send_raw_transaction(...)`` must START so the TX reaches the
    validator's mempool before the predecessor block's assembly window
    closes, ensuring inclusion in that block (one block before lock).

    Math:
      1. Start at predicted_predecessor_milli_ts (when that block "closes"
         per consensus — equals its successor's milli_ts in steady state).
      2. Quantum-shift guard: if the prediction is within one 50ms quantum
         of lock_ms, a quantum-shift event could push the predecessor's
         actual milli_ts past lock_ms, making our target block the lock
         block itself (definite revert). Back off one full block.
      3. Subtract the validator's TX-list freeze window (TX must arrive
         BEFORE the validator stops accepting TXs into the candidate).
      4. Subtract the one-way RPC send latency (TX must be sent from this
         host BEFORE the validator's mempool needs to receive it).

    The result is a local wallclock target. The engine compares it to
    ``time.time() * 1000`` at submit time and aborts if past.
    """
    if block_time_ms is None:
        block_time_ms = _tc.BSC_BLOCK_TIME_MS
    if quantum_ms is None:
        quantum_ms = _tc.BSC_QUANTUM_MS
    if assembly_window_ms is None:
        assembly_window_ms = _tc.VALIDATOR_ASSEMBLY_WINDOW_MS
    if one_way_ms is None:
        one_way_ms = _tc.BSC_BET_SUBMIT_ONE_WAY_MS

    bet_inclusion_deadline = predicted_predecessor_milli_ts
    # Quantum-shift protection: if predecessor's predicted milli_ts is
    # within one quantum of lock_ms, a 50ms shift could land it past
    # lock_ms (making the predicted "predecessor" actually the lock
    # block). Back off one full block to play safe.
    if (bet_inclusion_deadline + quantum_ms) >= lock_ms:
        bet_inclusion_deadline -= block_time_ms
    # Validator TX-list freeze: bet must arrive BEFORE this.
    bet_inclusion_deadline -= assembly_window_ms
    # One-way RPC send: deadline at this host is one_way_ms BEFORE the
    # validator must receive the TX.
    return bet_inclusion_deadline - one_way_ms


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
        batch_size: int = _tc.RPC_BATCH_MAX_BLOCKS,
        ramp_poll_1_wakeup_offset_before_lock_ms: int = 7500,
    ) -> None:
        if interval_seconds <= 0:
            raise InvariantError("interval_seconds_nonpositive")
        if periodic_poll_interval_s <= 0:
            raise InvariantError("periodic_poll_interval_nonpositive")
        if batch_size <= 0 or batch_size > _tc.RPC_BATCH_MAX_BLOCKS:
            raise InvariantError(
                f"batch_size_out_of_range: {batch_size} "
                f"(max {_tc.RPC_BATCH_MAX_BLOCKS})"
            )
        if ramp_poll_1_wakeup_offset_before_lock_ms <= 0:
            raise InvariantError("ramp_poll_1_wakeup_offset_ms_nonpositive")

        pool = list(endpoint_pool)
        if not pool:
            raise InvariantError("endpoint_pool_empty")

        self._interval_seconds = int(interval_seconds)
        self._endpoint_pool: list[str] = pool

        # ThreadPoolExecutor for parallel fan-out across the full pool.
        # Sized to 3 * len(pool) so abandoned-future stragglers from a
        # previous call (whose urllib3 sockets haven't timed out yet)
        # can't block the next call's submit. Workers are lazily
        # spawned, so unused slots are free.
        # Always constructed — the pool-size-1 fast path in
        # _do_hedged_post bypasses it, but a never-used executor is
        # cheap (no threads spawned until submit()).
        self._executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, 3 * len(self._endpoint_pool)),
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
        # Engine-side ramp_poll_1 lock-relative offset (config-derived in
        # production via pancakebot/config.py from pool_cutoff_seconds;
        # canonical value at pool_cutoff=6 is 7500ms). Used by the
        # lock-anchored _periodic_loop to suspend periodic ticks that
        # would otherwise land inside the (lock - ramp_1_offset, lock]
        # ramp/final window and race the engine-driven polls for the
        # non-blocking _poll_lock.
        self._ramp_poll_1_wakeup_offset_ms = int(ramp_poll_1_wakeup_offset_before_lock_ms)

        self._lock = threading.Lock()

        # Pool state — same shapes as PoolEventWatcher for engine compat.
        self._pools: dict[int, _EpochPool] = {}
        self._block_ts: dict[int, int] = {}
        self._processed_bet_log_ids: dict[int, set[str]] = {}

        # Round-phase state (set by engine).
        self._current_epoch: int = -1
        self._lock_at: int = 0

        # Cursor: the highest block number we've polled receipts for.
        # Cursor-init sets this to round_start_block - 1 on the first
        # set_round_phase; subsequent polls advance it. Periodic and
        # ramp/final polls all read+write under self._lock to keep
        # log-id dedup honest.
        self._last_polled_block_number: int = 0

        # Connection / readiness state.
        # Bundle 2 (2026-05-13): True after the first successful
        # periodic/ramp/final poll latches via
        # _latch_first_successful_poll_locked. Until then, is_pool_ready
        # returns (False, "cold_start_in_progress") and the engine skips.
        self._connected: bool = False
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

        # ``(needed_ms, available_ms)`` from the most recent catchup
        # feasibility check that returned True (= infeasible). Reset
        # to None on each round advance. Consumed by engine.py at the
        # SKIP narrative for ``catchup_infeasible_for_round`` rounds
        # so the operator-facing log line carries the actual numbers
        # ("need 39.5s, have 30.1s") rather than just the reason flag.
        self._last_catchup_detail: tuple[int, int] | None = None

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

        # -- Observability monitors (guard audit Tier 1+2) --
        # Anchor static-fallback rate (3.1): each round records True if the
        # anchor poll returned None (round fell back to the static wake) or
        # False if it produced a dynamic anchor. A sustained high rate means
        # the dynamic-wake optimization is silently inert.
        self._anchor_fallback_monitor = RollingRateMonitor(
            name="anchor_static_fallback",
            max_rate=0.10,
            window=50,
            min_samples=50,
        )
        # Observed block-time drift (5.2): per-round average block time from
        # consecutive anchor polls (delta_milli_ts / delta_block_number).
        # Confirms BSC_BLOCK_TIME_MS still holds (the load-bearing assumption
        # behind the no-margin predecessor extrapolation).
        self._block_time_monitor = RollingMedianDriftMonitor(
            name="bsc_block_time",
            expected=_tc.BSC_BLOCK_TIME_MS,
            tolerance=20.0,
            window=20,
            min_samples=10,
        )
        self._prev_anchor_block: int = 0
        self._prev_anchor_milli_ts: int = 0
        # Pool-understatement counters (2.1 / 3.2): blocks dropped from the
        # pool aggregate because a receipt fetch errored, or a bet's block
        # timestamp could not be resolved. Both silently understate a side's
        # pool — surface them.
        self._block_receipt_skips_total: int = 0
        self._pool_block_ts_zero_drops_total: int = 0
        # Failed-block retry queue (2.1 fix): a block whose receipts errored is
        # re-fetched on later periodic polls (off the critical path) instead of
        # being permanently dropped from the pool aggregate. ``{block: attempts}``.
        # Idempotent — ``_process_receipts_for_block`` dedups by log-id, so a
        # recovered block can never double-count, and a retry can only ADD the
        # block's real (previously-missed) bets (conservative-direction safe).
        # Bounded by ``_retry_max_attempts`` and a block-age window so the queue
        # cannot grow without bound; exhausted/aged blocks are dropped with a WARN.
        self._pending_retry_blocks: dict[int, int] = {}
        self._block_receipts_recovered_total: int = 0
        self._retry_max_attempts: int = 4
        self._retry_window_blocks: int = 900

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

    # ------------------------------------------------------------------
    # Bundle 5 v2 (2026-05-14): single anchor poll on the per-round
    # critical-path. Replaces the Bundle 4 continuous fine-phase poller
    # + persistent anchor state + watchdog. The engine fires ONE
    # ``fire_anchor_poll`` call per round; if it responds in the
    # 200ms timeout window, the engine uses the fresh anchor to compute
    # a dynamic critical-path wake and a dynamic bet-submit deadline.
    # Otherwise the engine falls back to its static wake + static
    # deadline. No persistent anchor state on RpcPoller — the anchor
    # lives in engine-local scope for one round.
    # ------------------------------------------------------------------

    def fire_anchor_poll(
        self, *, timeout_s: float,
    ) -> AnchorState | None:
        """Single ``eth_getBlockByNumber('latest')`` call with a hard
        timeout. Returns an ``AnchorState`` if the response arrived in
        time AND decoded a valid BEP-520 ms-encoded mixHash; otherwise
        ``None``.

        The bot's mainnet target is post-Lorentz, so ms-encoding is
        expected. A None return signals either:
          - RPC timeout / transport failure (network glitch, all hedged
            endpoints stalled), OR
          - The latest block's mixHash decoded to milli_ts that didn't
            parse (malformed encoding — unexpected on a Lorentz chain
            and would indicate an out-of-spec validator).

        Either way, the engine treats None as "use the static fallback
        for this round" — no per-session memory of the failure, no
        retry. The next round fires a fresh anchor poll.
        """
        # urllib3.PoolManager already imposes a per-request timeout via
        # ``_rpc_call_single`` (uses ``RPC_HTTP_SINGLE_TIMEOUT_SECONDS``).
        # We additionally enforce a tight ceiling via the hedged
        # transport's own ``timeout_seconds``: the engine wants any
        # response slower than ~200ms to count as "missed the window",
        # not just "transport-level timeout".
        try:
            block = self._rpc_call_single_with_timeout(
                "eth_getBlockByNumber", ["latest", False],
                timeout_s=timeout_s,
            )
        except Exception:  # noqa: BLE001
            # Anchor poll timeout / transport error: fallback to static wake.
            # Per round this is silent (wake_mode="static" in cycle_audit);
            # the rolling-rate monitor surfaces a sustained-fallback regime.
            self._record_anchor_outcome(fell_back=True, reason="timeout_or_transport")
            return None
        if not isinstance(block, dict):
            self._record_anchor_outcome(fell_back=True, reason="malformed_response")
            return None
        bn_hex = block.get("number")
        if not isinstance(bn_hex, str):
            self._record_anchor_outcome(fell_back=True, reason="malformed_block_number")
            return None
        try:
            bn = int(bn_hex, 16)
        except ValueError:
            self._record_anchor_outcome(fell_back=True, reason="malformed_block_number")
            return None
        milli_ts = compute_milli_ts(block)
        if milli_ts is None:
            self._record_anchor_outcome(fell_back=True, reason="malformed_milli_ts")
            return None
        self._record_anchor_outcome(
            fell_back=False, reason="ok", block_number=bn, milli_ts=milli_ts,
        )
        return AnchorState(
            block_number=bn,
            milli_ts=milli_ts,
            observed_at_local_ms=int(time.time() * 1000),
        )

    def _record_anchor_outcome(
        self, *, fell_back: bool, reason: str,
        block_number: int | None = None, milli_ts: int | None = None,
    ) -> None:
        """Update anchor observability monitors (guard audit 3.1 / 5.2).

        ``fell_back`` True means the anchor poll returned None and the round
        will use the static wake; the rolling-rate monitor alerts when that
        rate is sustained. On success, the average block time since the
        previous anchor (delta_milli_ts / delta_block_number) feeds the
        block-time drift monitor confirming BSC_BLOCK_TIME_MS still holds.
        Telemetry only; wrapped so it can never affect the poll result.
        """
        try:
            alert = self._anchor_fallback_monitor.observe(
                fell_back, detail=f"reason={reason}"
            )
            if alert is not None:
                warn("ALERT", alert)
            if not fell_back and block_number is not None and milli_ts is not None:
                if self._prev_anchor_block and block_number > self._prev_anchor_block:
                    span_blocks = block_number - self._prev_anchor_block
                    avg_block_ms = (milli_ts - self._prev_anchor_milli_ts) / span_blocks
                    bt_alert = self._block_time_monitor.observe(avg_block_ms)
                    if bt_alert is not None:
                        warn("ALERT", bt_alert)
                self._prev_anchor_block = block_number
                self._prev_anchor_milli_ts = milli_ts
        except Exception:  # noqa: BLE001 — telemetry must never affect polling
            pass

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
        """Start the periodic-poll daemon thread. Cursor initialization
        runs lazily on the first ``set_round_phase`` call (~1 RPC); the
        daemon's periodic ticks drive the in-round catch-up. The
        ``_connected`` latch flips on the first successful poll."""
        if self._periodic_thread is not None and self._periodic_thread.is_alive():
            return
        self._stop_event.clear()
        self._periodic_thread = threading.Thread(
            target=self._periodic_loop, daemon=True, name="rpc-poller-periodic",
        )
        self._periodic_thread.start()
        info("START",
             f"RPC poller started endpoint={self._current_endpoint} "
             f"periodic={self._periodic_poll_interval_s}s "
             f"batch={self._batch_size}")

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
        info("STOP", "RPC poller stopped")

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

        Triggers cursor initialization on the first call (~1 RPC,
        non-blocking; daemon's periodic ticks drive the catch-up).
        Subsequent calls drop past-round state and update tracked
        epochs via _on_epoch_advance.
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
                info("START", f"RPC poll initialized at epoch {current_epoch}")
                self._current_epoch = current_epoch
            else:
                # Drop past-round epochs (strictly less than new
                # current_epoch) from both _pools and
                # _processed_bet_log_ids. The "+1" next-epoch entries
                # are kept. The two dicts share the same epoch keyset
                # by construction (see population site in
                # _process_receipts_for_block), so a single key list
                # suffices for both deletes.
                epochs_to_drop = [e for e in self._pools if e < current_epoch]
                for e in epochs_to_drop:
                    del self._pools[e]
                    del self._processed_bet_log_ids[e]
                self._current_epoch = current_epoch
                is_epoch_advance = True

            self._lock_at = lock_at

            # Bounded _block_ts: keep most recent 500 once we exceed 1000.
            if len(self._block_ts) > 1000:
                sorted_blocks = sorted(self._block_ts.keys())
                for bn in sorted_blocks[:-500]:
                    del self._block_ts[bn]

        if is_first_call:
            # Bundle 2 (2026-05-13): synchronous cursor-init only — fast
            # (~1 RPC roundtrip). Actual catch-up happens via the daemon's
            # normal periodic ticks. _connected stays False until the
            # first successful _poll_now flips the latch; is_pool_ready
            # gates the engine away from acting on a half-built pool
            # aggregate during that window.
            self._initialize_cursor_from_head()
            # Bundle 5 v2 (2026-05-14): no session-level Lorentz check.
            # The per-round ``fire_anchor_poll`` parses mixHash inline
            # and returns None if decoding fails. A None return for a
            # round means "use static fallback for this round" — no
            # persistent state, no detection-failure mode.
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

        _ts_zero_drops = 0
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
                    # Block timestamp never resolved -> bet silently excluded
                    # from the aggregate (guard audit 3.2). Count + surface.
                    _ts_zero_drops += 1
                    continue
                if bet.block_ts >= max_ts:
                    continue

                if bet.side == "Bull":
                    bull_wei += bet.amount_wei
                else:
                    bear_wei += bet.amount_wei

        if _ts_zero_drops > 0:
            self._pool_block_ts_zero_drops_total += _ts_zero_drops
            warn(
                "ALERT",
                f"pool aggregate dropped {_ts_zero_drops} bet(s) with unresolved "
                f"block_ts for epoch {epoch}; understates a side's pool "
                f"(cumulative={self._pool_block_ts_zero_drops_total})",
            )

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
           cursor forward, skipping ~N*660 past-round blocks; past
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
        # cursor has been clamped and catch-up is now feasible. Reset
        # the detail tuple too so a SKIP in a later round can't surface
        # carried-over numbers from a past infeasibility event.
        with self._lock:
            self._catchup_infeasible_for_round = False
            self._last_catchup_detail = None

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
            # Gate the flag/warn on a live (pre-lock) round. After lock
            # passes, an INFEAS signal is moot — the round is already
            # closed; the cursor just needs to keep advancing for the
            # next round (no flag, no warn). _on_epoch_advance only fires
            # at round transition so this branch is defensively rare here
            # but mirrors the same gating as _poll_now for consistency.
            time_until_lock_ms = max(0, int((lock_at - time.time()) * 1000))
            if time_until_lock_ms > 0:
                with self._lock:
                    self._catchup_infeasible_for_round = True
                # INFEAS WARN folded into engine.py SKIP narrative
                # at Phase B v2 (the round is going to be skipped anyway;
                # one operator line per skip is enough).

    def _compute_round_start_block(self, round_start_ts: int) -> int | None:
        """Return the block-number whose timestamp ~= round_start_ts.

        Strategy:

        1. **Cache lookup** (free): pick the newest entry in
           ``_block_ts`` with ts <= round_start_ts and within 60s of it,
           then extrapolate forward by ``BSC_BLOCK_TIME_MS``.
        2. **RPC fallback**: ``eth_getBlockByNumber("latest", false)``
           returns ``(head_num, head_ts)``; extrapolate backward.

        Returns None if RPC fails AND cache has no usable anchor.
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

        Batch-size-aware (2026-05-12): full batches use the table's
        ``batch_size`` p99; the final partial batch uses the (smaller)
        p99 for ``remainder`` blocks. For a 7-block lag at batch_size=20,
        the prior unconditional ``rtt_p99(20)=1319ms`` estimate drops to
        ``rtt_p99(7) = 827ms`` (interpolated between table[5]=771 and
        table[10]=910), eliminating false-INFEAS at small backlogs.
        For a 47-block lag: 2 full batches * rtt_p99(20) + rtt_p99(7).
        """
        if blocks_behind <= 0:
            return 0
        full_batches, remainder = divmod(blocks_behind, self._batch_size)
        full_rtt = _tc.rpc_rtt_p99_for_batch(self._batch_size)
        total_ms = full_batches * full_rtt
        if remainder > 0:
            total_ms += _tc.rpc_rtt_p99_for_batch(remainder)
        return total_ms

    def _available_catchup_ms(self, time_until_lock_ms: int) -> int:
        """Time available for catch-up, with the same safety buffer
        the deadline-driven polls use."""
        return max(0, time_until_lock_ms - _tc.RPC_POLL_FINAL_TO_CRITICAL_PATH_SAFETY_MS)

    def _is_catchup_infeasible(self, *, blocks_behind: int, lock_at: int) -> bool:
        """Return True if estimated catch-up wallclock exceeds the time
        budget remaining before lock_at.

        Side effect: when the math returns True, stashes
        ``(estimated_ms, available_ms)`` on ``self._last_catchup_detail``
        so the engine SKIP narrative can render the actual numbers.
        Cleared in ``set_round_phase`` when a new round begins.
        """
        if blocks_behind <= 0 or lock_at <= 0:
            return False
        estimated_ms = self._estimated_catchup_ms(blocks_behind)
        time_until_lock_ms = max(0, int((lock_at - time.time()) * 1000))
        available_ms = self._available_catchup_ms(time_until_lock_ms)
        infeasible = estimated_ms > available_ms
        if infeasible:
            with self._lock:
                self._last_catchup_detail = (estimated_ms, available_ms)
        return infeasible

    @property
    def last_catchup_detail(self) -> tuple[int, int] | None:
        """Returns ``(needed_ms, available_ms)`` from the most recent
        infeasible catchup check, or None if no check has flagged the
        current round. Reset in ``set_round_phase`` when lock_at advances.
        Consumed by engine.py at the SKIP narrative for
        ``catchup_infeasible_for_round`` rounds.
        """
        with self._lock:
            return self._last_catchup_detail

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

    def _initialize_cursor_from_head(self) -> None:
        """Synchronous cursor-init from chain head. Called from the first
        ``set_round_phase()`` call; returns within ~1 RPC roundtrip.

        Bundle 2 refactor (2026-05-13): replaced the prior ``_cold_start``
        one-shot path that synchronously backfilled the in-round range
        before returning (30-90s engine block at startup). Now the cursor
        is set to ``round_start_block - 1`` and the actual catch-up
        happens via the daemon's normal periodic ticks. The engine is
        unblocked immediately; ``is_pool_ready`` returns False with
        ``cold_start_in_progress`` until the periodic loop's first
        successful ``_poll_now`` flips the ``_connected`` latch.

        Race fix Y1 (reviewer 2026-05-13): cursor-init acquires
        ``_poll_lock`` for its full duration. Without it, the periodic
        daemon's first tick (released from State A by ``_lock_at`` going
        non-zero in ``set_round_phase``) can race ``_poll_now`` against
        an un-initialized cursor (``_last_polled_block_number == 0``),
        compute a 50M-block backlog, INFEAS-flag the round, and wedge
        the engine into skipping round 1 with
        ``pool_not_ready_catchup_infeasible_for_round``. With the lock
        held, the racing ``_poll_now`` sees ``acquire(blocking=False)``
        fail and skips that tick cleanly.

        Belt-and-suspenders: cursor-init also clears
        ``_catchup_infeasible_for_round = False`` on every successful
        exit (feasible or head-behind-round-start branch). Covers the
        microsecond window where the daemon may have acquired
        ``_poll_lock`` before cursor-init did and already poisoned the
        flag — by the time cursor-init lands, the daemon's bad poll has
        completed and we authoritatively reset the verdict against the
        correct cursor.

        Round-aware: round_start_block is derived from
        ``lock_at - interval_seconds``, NOT from a head-relative
        full-round lookback. Past-round blocks are archive-only and
        never bet on, so seeking earlier is wasted.

        Feasibility-aware: if math says the first periodic tick can't
        plausibly catch up before lock_at, the round is marked
        catch-up-infeasible (cursor jumped to head) and the next round
        starts clean. Same semantics as the prior cold-start INFEAS
        branch — just routed via the new daemon-entry path.
        """
        # Y1: hold _poll_lock for the duration so a racing daemon tick
        # cannot enter _poll_now against an uninitialized cursor. Blocking
        # acquire is safe: cursor-init runs at most once per RpcPoller
        # lifetime (gated by is_first_call in set_round_phase), and the
        # only other holder of _poll_lock is _poll_now itself (non-blocking
        # acquire, so it yields immediately if we got there first).
        self._poll_lock.acquire()
        try:
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
                    warn("ALERT", f"init_cursor: eth_getBlockByNumber(latest) failed: {type(e).__name__}: {e}")
                    return
                if head <= 0 or head_ts <= 0:
                    warn("ALERT", f"init_cursor: invalid header head={head} ts={head_ts}")
                    return

                with self._lock:
                    lock_at_local = self._lock_at
                round_start_ts = lock_at_local - self._interval_seconds

                if head_ts <= round_start_ts:
                    # Head is behind round_start (chain hasn't caught up
                    # to round_start yet, or lock_at is in the future
                    # beyond head_ts). Nothing to seek — set cursor at
                    # head; daemon's periodic ticks will drive forward.
                    with self._lock:
                        self._last_polled_block_number = head
                        # Y1 safeguard: authoritatively clear any flag a
                        # racing daemon poll may have set.
                        self._catchup_infeasible_for_round = False
                    info("START",
                         f"init_cursor: head_ts {head_ts} <= "
                         f"round_start_ts {round_start_ts}; cursor at "
                         f"head={head} (no in-round blocks yet)")
                    return

                # delta_blocks: how many blocks since round_start.
                # Bundle 5 v2 (2026-05-14): no forward safety margin. The
                # post-Lorentz chain's empirical "misses only delay, never
                # advance" property means actual blocks-elapsed is at most
                # ``ceil((head_ts - round_start_ts) * 1000 / 450)`` — slot
                # misses produce FEWER blocks than the nominal divisor
                # predicts, never more. So ``round(...)`` is an upper bound
                # (modulo ≤ 0.5 block of rounding noise); cursor at
                # ``head - delta_blocks`` lands at or before
                # round_start_block by construction. Any over-fetched
                # pre-round blocks are filtered by the epoch gate in
                # _process_receipts.
                #
                # Prior versions added +20 (compensating for the old 500ms
                # divisor's ~10% under-count) and then +5 (over-defensive
                # carryover after the divisor was made exact in Bundle 4).
                # Q2 fix (2026-05-14, Bundle 5 v2): both are gone.
                delta_blocks = round(
                    (head_ts - round_start_ts) * 1000
                    / _tc.BSC_BLOCK_TIME_MS
                )
                round_start_block = max(0, head - delta_blocks)
                blocks_to_backfill = max(0, head - round_start_block + 1)

                # Feasibility check: can the upcoming periodic tick catch up
                # the in-round range before lock_at? Same math as the
                # _on_epoch_advance and _poll_now feasibility branches.
                if self._is_catchup_infeasible(
                    blocks_behind=blocks_to_backfill, lock_at=lock_at_local,
                ):
                    with self._lock:
                        self._catchup_infeasible_for_round = True
                        # Advance cursor past the un-backfilled range so
                        # the daemon's first periodic poll sees a small
                        # gap and doesn't try to refill round-start blocks.
                        self._last_polled_block_number = head
                    time_until_lock_ms = max(
                        0, int((lock_at_local - time.time()) * 1000),
                    )
                    warn("SKIP",
                         f"Skip cold-start round: catchup {blocks_to_backfill} blocks "
                         f"would take ~{self._estimated_catchup_ms(blocks_to_backfill)}ms "
                         f"> {self._available_catchup_ms(time_until_lock_ms)}ms "
                         f"available; cursor jumped to head, will resume on next round")
                    return

                with self._lock:
                    self._last_polled_block_number = round_start_block - 1
                    # Y1 safeguard: authoritatively clear any flag a
                    # racing daemon poll may have set against the
                    # uninitialized cursor=0 sentinel.
                    self._catchup_infeasible_for_round = False

                info("START",
                     f"init_cursor: cursor at {round_start_block - 1}; "
                     f"first periodic tick will catch up "
                     f"{blocks_to_backfill} blocks")

            except Exception as e:  # noqa: BLE001
                warn("ALERT", f"init_cursor failed: {type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self._cold_start_in_progress = False
        finally:
            self._poll_lock.release()

    def _periodic_loop(self) -> None:
        """Daemon-thread loop with lock-anchored 3-state cadence.

        Refactored 2026-05-12 from a pure wall-clock 8s tick to a
        lock-anchored cadence: ticks land at round_open + k*period for
        k=1..(interval_seconds//period − 1), aligned fresh to each
        round's lock_at. Solves the INFEAS spam pattern observed in
        the 1h soak (Event 2: late-round periodic firing at lock−1.13s
        with 7-block lag → math returned est=1319ms > avail=930ms →
        flag set → next round's critical_path READY check found the
        flag and SKIPped).

        Three-state design:

        State A — pre-init (``_lock_at <= 0``): sleep one periodic
            interval on ``_stop_event.wait(timeout=period)`` and re-check
            the loop. No polling. This state only triggers in the brief
            race window between ``start()`` and the first
            ``set_round_phase()`` call from the engine — once
            set_round_phase synchronously initializes the cursor via
            ``_initialize_cursor_from_head`` and sets ``_lock_at``, the
            next loop iteration falls into State B and drives the
            catch-up. Using ``_stop_event`` for the wait keeps
            ``stop()`` reactive — a shutdown signal during this race
            window exits the loop within ``period`` seconds.

        State B — steady (``now < _lock_at``): anchor cadence to
            ``round_open = _lock_at − interval_seconds``. Compute the
            next anchored tick ``next_at = round_open + k * period``
            for smallest k>=1 such that ``next_at > now``.

            Three sub-cases inside State B (see
            ``_compute_periodic_timeout``):
            - the anchored tick has comfortable margin before
              ramp_window_start — fire at next_at.
            - the anchored tick is outside the ramp window but its
              worst-case HTTP RTT could extend past ramp_window_start
              — reschedule to fire at the latest safe time
              (``ramp_window_start − max_rtt − safety``). If that
              time has already passed (previous poll overran), suspend.
            - the anchored tick lands INSIDE the ramp window — suspend.

            Suspended ticks sleep to ``_lock_at + 0.1s`` so the next
            loop iteration picks up State C (and shortly afterwards,
            the fresh anchor from the next set_round_phase).

        State C — post-lock fallback (``now >= _lock_at``): round just
            locked, engine is in _sleep_and_claim, set_round_phase for
            the next round hasn't fired yet (~35-40s latency). Fall back
            to wall-clock cadence — _poll_now still runs and the cursor
            advances for round N+2, but the post-lock INFEAS gating in
            ``_poll_now`` suppresses false flag/WARN noise for ticks
            in this window.

        Label "period" (6 chars) fits log _SUB_W=6 — a prior version
        used "periodic" (8 chars) which raised InvariantError in log.py
        on the first periodic-poll log call.
        """
        while not self._stop_event.is_set():
            with self._lock:
                lock_at = self._lock_at
            period = self._periodic_poll_interval_s
            now = time.time()

            # State A: cold-start (anchor not yet set by first
            # set_round_phase call). No polling — block on the event
            # with a periodic-interval timeout so stop() still works.
            if lock_at <= 0:
                if self._stop_event.wait(timeout=period):
                    break
                continue

            timeout = self._compute_periodic_timeout(
                now=now, lock_at=lock_at, period=period,
            )

            if self._stop_event.wait(timeout=timeout):
                break
            # Bundle 2 (2026-05-13): the periodic loop IS the cold-start
            # path now. The prior ``_cold_start_done.is_set()`` gate
            # would deadlock the latch (only _poll_now sets the event,
            # so gating _poll_now on it means nothing ever polls).
            # State A above already guards the pre-set_round_phase race
            # window; once _lock_at is non-zero, this loop must drive
            # the catch-up.
            try:
                self._poll_now(deadline_ms=0, label="period")
            except Exception as e:  # noqa: BLE001
                warn("ALERT", f"periodic poll failed: {type(e).__name__}: {e}")

    def _compute_periodic_timeout(
        self, *, now: float, lock_at: int, period: int,
    ) -> float:
        """Return the wait timeout (seconds) for the lock-anchored
        periodic loop. Extracted from ``_periodic_loop`` so the cadence
        math is unit-testable without spinning up the daemon thread.

        Caller is responsible for State A (``lock_at <= 0``) before
        invoking this helper.

        Post-lock fallback (``now >= lock_at``): wall-clock cadence
        while the engine is in ``_sleep_and_claim`` and
        ``set_round_phase`` hasn't yet advanced ``_lock_at``.

        Before lock, the periodic poll must not race ``ramp_poll_1``
        for ``_poll_lock``. The ramp wake fires at
        ``lock_at − ramp_poll_1_offset`` and uses non-blocking acquire;
        if a periodic poll is still in flight at that moment, ramp_1 is
        silently dropped. Two distinct dangers:

          * The anchored tick lands INSIDE the ramp window. Suspend
            (sleep past lock and let the post-lock branch resume).
            (Example: ramp_window_start=lock_at−7.5s, next_at=lock_at−4s.)

          * The anchored tick is just BEFORE the window, but its HTTP
            timeout (5s) plus jitter could extend it past
            ramp_window_start. Try to fire earlier, at
            ``ramp_window_start − max_rtt − safety``, so a full-timeout
            poll completes before the ramp wake. If the latest safe
            fire time has already PASSED (the prior poll overran),
            suspend rather than fire a doomed poll.

        Branches:

        - **steady**: fire at the anchored ``round_open + k*period``.
        - **reschedule**: fire earlier than anchored, at the latest
          time a worst-case-RTT poll can complete before the ramp
          window opens.
        - **suspend**: skip this round's tick; sleep to
          ``lock_at + 0.1s`` so the next iteration falls into the
          post-lock branch.
        - **post-lock**: wall-clock cadence (returns ``period``).
        """
        if now >= lock_at:
            return float(period)
        round_open = lock_at - self._interval_seconds
        ramp_window_start = (
            lock_at - self._ramp_poll_1_wakeup_offset_ms / 1000.0
        )
        # Latest moment a worst-case-RTT periodic poll can BEGIN and
        # still finish before the ramp window opens. A poll that hits
        # the full HTTP timeout (RPC_HTTP_BATCH_TIMEOUT_SECONDS) holds
        # ``_poll_lock`` for that long; firing later than this would
        # leave the lock held when ramp_1 wakes.
        safe_fire_latest = (
            ramp_window_start
            - _tc.RPC_HTTP_BATCH_TIMEOUT_SECONDS
            - _tc.RPC_PERIODIC_TO_RAMP_SAFETY_BUFFER_SECONDS
        )
        k = max(1, int((now - round_open) // period) + 1)
        next_at = round_open + k * period

        # The anchored tick lands inside the ramp window itself —
        # firing here would either race ramp_1 directly or block it.
        # Sleep past lock and let the post-lock branch resume.
        if next_at >= ramp_window_start:
            return max(0.1, lock_at + 0.1 - now)

        # The anchored tick is outside the ramp window but close
        # enough that its worst-case RTT could extend into ramp_1.
        # Try to fire earlier at the latest safe time; if that time
        # has already passed (the previous poll overran), suspend
        # instead of firing a doomed poll that would still overlap.
        # Example (reschedule): canonical config (period=8, ramp=7.5,
        #   max_rtt=5, safety=0.05). At now=lock_at−13, anchored
        #   next_at=lock_at−12 falls in (safe_fire_latest=lock_at−12.55,
        #   ramp_window_start=lock_at−7.5). Reschedule to lock_at−12.55.
        # Example (overrun → suspend): previous poll completed at
        #   lock_at−12.5 (overran past safe_fire_latest=lock_at−12.55
        #   by 50 ms). No safe time remains; suspend and let
        #   ramp_1+ramp_2 absorb the extra backlog.
        if next_at > safe_fire_latest:
            if safe_fire_latest > now:
                return safe_fire_latest - now
            return max(0.1, lock_at + 0.1 - now)

        # Anchored tick has comfortable margin before the ramp window.
        return max(0.0, next_at - now)

    def _latch_first_successful_poll_locked(self) -> None:
        """Flip ``_connected`` + ``_cold_start_done`` exactly once on the
        first successful poll. Caller MUST hold ``self._lock``.

        Bundle 2 refactor (2026-05-13): replaces the prior ``_cold_start``
        path's atomic "backfill complete -> set flags" handoff. Now the
        flags flip lazily once the daemon's first ``_poll_now`` succeeds
        (either fetched ≥1 batch with no error, or found head already
        caught up). Until then, ``is_pool_ready`` returns ``(False,
        "cold_start_in_progress")`` and the engine skips, identical to
        the pre-refactor behaviour during the synchronous backfill
        window.
        """
        if not self._connected:
            self._connected = True
            self._cold_start_done.set()
            info("READY",
                 f"first poll complete; pool aggregate ready "
                 f"(cursor={self._last_polled_block_number}, "
                 f"epochs_tracked={len(self._pools)})")

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
            # share the same data anyway. A dropped ramp/final poll IS a
            # critical-path event (cursor advance skipped on the wake the
            # engine scheduled), so surface it (guard audit 1.2); periodic
            # drops stay silent by design.
            if label in ("ramp", "final"):
                warn(
                    "ALERT",
                    f"{label} poll dropped: poll-lock contended (another poll "
                    f"in flight); critical-path cursor advance skipped this wake",
                )
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
                warn("ALERT", f"{label} poll: eth_blockNumber failed: {self._last_poll_error}")
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
                    self._latch_first_successful_poll_locked()
                return

            # Mid-round feasibility check: if RTT degrades during the
            # round, a periodic poll might find that math says it can't
            # catch up before lock_at. Abort early — no batches fetched,
            # set _catchup_infeasible_for_round so the engine skips.
            #
            # Post-lock gating (2026-05-12): when ``time_until_lock_ms``
            # has already gone non-positive (a trailing periodic firing
            # in the claim window before set_round_phase advances
            # _lock_at), an INFEAS verdict is moot — the round is closed
            # and the flag would not gate any live bet. Still abort the
            # poll (math says we cannot finish in the imagined window),
            # but DO NOT set the flag or emit WARN. The cursor advances
            # on the next non-INFEAS tick for the next round.
            blocks_to_catchup = head - from_block + 1
            if self._is_catchup_infeasible(
                blocks_behind=blocks_to_catchup, lock_at=lock_at_local,
            ):
                time_until_lock_ms = max(0, int((lock_at_local - time.time()) * 1000))
                if time_until_lock_ms > 0:
                    with self._lock:
                        self._catchup_infeasible_for_round = True
                    # Per-poll INFEAS WARN dropped at Phase B v2 — the
                    # engine's SKIP narrative for the round carries the
                    # detail and fires once per skipped round.
                    pass
                # else: trailing post-lock periodic; cursor will advance on
                # the next successful poll once set_round_phase updates
                # _lock_at for the new round. No flag, no warn.
                return

            n_blocks = head - from_block + 1
            blocks_polled = 0
            error_seen: str | None = None

            # Retry previously-skipped blocks (2.1 fix): periodic only, off the
            # critical path. The retry is one extra batched call; only run it
            # when an extra batch's worst-case lock-hold still finishes before
            # ramp_poll_1 (the periodic cadence budgets ONE batch before the
            # ramp window — this adds at most one more). Pre-init / post-lock
            # (_lock_at<=0) runs wall-clock with ample margin.
            if label == "period" and self._pending_retry_blocks:
                if lock_at_local <= 0:
                    self._retry_pending_receipt_blocks()
                else:
                    ramp_window_start = (
                        lock_at_local - self._ramp_poll_1_wakeup_offset_ms / 1000.0
                    )
                    extra_batch_safe_latest = (
                        ramp_window_start
                        - 2 * _tc.RPC_HTTP_BATCH_TIMEOUT_SECONDS
                        - _tc.RPC_PERIODIC_TO_RAMP_SAFETY_BUFFER_SECONDS
                    )
                    if time.time() < extra_batch_safe_latest:
                        self._retry_pending_receipt_blocks()

            # Batch in chunks. Deadline check is total-RTT-based: if
            # the cumulative time since poll start exceeds deadline_ms
            # at any point, abort remaining batches (we'll process
            # what we got and mark too_slow). The engine passes
            # deadline_ms = (next_wake_time - now) - safety.
            for batch_start in range(from_block, head + 1, self._batch_size):
                if deadline_ms > 0:
                    elapsed_ms = int((time.time() - t_start) * 1000)
                    if elapsed_ms > deadline_ms:
                        warn("ALERT",
                             f"{label} poll deadline exceeded after batch_start={batch_start}: "
                             f"elapsed={elapsed_ms}ms > deadline={deadline_ms}ms; "
                             f"aborting remaining batches")
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
                    warn("ALERT", f"{label} poll batch failed: {error_seen}")
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
                    self._latch_first_successful_poll_locked()
                self._last_poll_too_slow = too_slow

            # Phase B v2 (2026-05-18): per-poll OK lines DROPPED — they fired
            # every 8s (periodic) + ramp/final polls = ~30-50/round, mostly
            # operator-noise. PARTIAL / EMPTY remain at WARN level since
            # each indicates an anomaly worth reading.
            if blocks_polled == 0 and error_seen is None and n_blocks > 0:
                warn("ALERT",
                     f"{label} poll EMPTY: 0/{n_blocks} blocks "
                     f"({from_block}..{head}) in {rtt_ms}ms (endpoint returned empty for valid range)")
            elif error_seen is not None and blocks_polled > 0:
                warn("ALERT",
                     f"{label} poll PARTIAL: {blocks_polled}/{n_blocks} blocks "
                     f"({from_block}..{head}) in {rtt_ms}ms")

        finally:
            with self._lock:
                self._poll_in_progress = False
            self._poll_lock.release()

    def _fetch_and_process_blocks(self, block_numbers: list[int]) -> None:
        """Fetch ``eth_getBlockReceipts`` for each block in a SINGLE
        batched HTTP request, then process bet events.

        Bundle 5 v2 (2026-05-14): receipts-only backfill. Previously
        (commit f02d736, 2026-05-08) we bundled ``eth_getBlockByNumber``
        alongside each receipts call to populate ``_block_ts`` for the
        pool_cutoff filter — but the I3 probe (2026-05-14, 20 samples
        per shape) showed that doubling sub-calls per batch added ~1%
        to per-batch RTT, vs the 2× speedup hoped for. The savings from
        receipts-only batching come from halving JSON-RPC payload size,
        not from halving sub-call count. Sub-call count is dominated by
        per-batch fixed costs (TLS, urllib3 PoolManager contention,
        rate-limit decisions).

        Block timestamps for bet-containing blocks are resolved lazily
        in ``_process_receipts_for_block`` via a single
        ``eth_getBlockByNumber`` per bet-containing block. Bet rate is
        sparse (~5-20 bets per 5-min round, ~1-3% of backfilled blocks),
        so the lazy path fires ~1-3 single-RPCs per round at most,
        off the critical path.
        """
        if not block_numbers:
            return
        # Each block contributes ONE sub-call (receipts only).
        calls: list[tuple[str, list]] = [
            ("eth_getBlockReceipts", [hex(bn)]) for bn in block_numbers
        ]
        results = self._rpc_batch(calls)
        if len(results) != len(calls):
            raise InvariantError(
                f"rpc_batch_length_mismatch: expected={len(calls)} got={len(results)}"
            )
        _block_skips = 0
        _skipped_bns: list[int] = []
        for i, bn in enumerate(block_numbers):
            receipts, recv_err = results[i]
            if recv_err is not None:
                # Single-block error: queue for retry on a later periodic
                # poll (2.1 fix). Don't raise here because the rest of the
                # batch might be valid.
                _block_skips += 1
                _skipped_bns.append(bn)
                continue
            if not isinstance(receipts, list):
                _block_skips += 1
                _skipped_bns.append(bn)
                continue
            self._process_receipts_for_block(bn, receipts)
            # Recovered (or never-failed) block: clear any pending retry. If
            # it WAS pending, this read recovered its previously-missed bets.
            with self._lock:
                if self._pending_retry_blocks.pop(bn, None) is not None:
                    self._block_receipts_recovered_total += 1
        if _skipped_bns:
            # Queue skipped blocks for retry (forward-only cursor is untouched;
            # these blocks are re-fetched explicitly, not via the cursor).
            with self._lock:
                for bn in _skipped_bns:
                    self._pending_retry_blocks[bn] = self._pending_retry_blocks.get(bn, 0) + 1
        if _block_skips > 0:
            # Pool-understatement signal (guard audit 2.1): bets in the
            # skipped blocks are absent from the pool aggregate UNTIL the
            # retry queue recovers them on a later periodic poll — surface so
            # a corrupted bull/bear ratio doesn't go unnoticed in the interim.
            self._block_receipt_skips_total += _block_skips
            with self._lock:
                _pending_n = len(self._pending_retry_blocks)
            warn(
                "ALERT",
                f"block receipt fetch skipped {_block_skips}/{len(block_numbers)} "
                f"blocks (receipt error); queued for retry "
                f"(pending={_pending_n}, cumulative_skips={self._block_receipt_skips_total}, "
                f"recovered={self._block_receipts_recovered_total})",
            )

    def _retry_pending_receipt_blocks(self) -> None:
        """Re-fetch blocks whose receipts errored on an earlier poll (2.1 fix).

        Called only from periodic polls, off the critical path. Recovers bets
        that would otherwise be permanently missing from the pool aggregate
        (the cursor already advanced past these blocks, so nothing else
        re-fetches them). ``_fetch_and_process_blocks`` itself maintains the
        queue: a recovered block is popped (and counted), a still-failing one
        is re-incremented. Idempotent — log-id dedup means a partially-counted
        block can't double-count, and the retry only ever ADDS real missed bets.

        Bounded two ways so the queue can't grow without limit:
          * attempts >= ``_retry_max_attempts`` -> give up, drop with a WARN.
          * block older than ``_retry_window_blocks`` behind the cursor -> drop
            (its bets are epoch-gated out of the current round anyway).
        Both drops are surfaced (no silent truncation of the understatement)."""
        with self._lock:
            floor = self._last_polled_block_number - self._retry_window_blocks
            retry: list[int] = []
            dropped: list[int] = []
            for bn, attempts in list(self._pending_retry_blocks.items()):
                if attempts >= self._retry_max_attempts or bn < floor:
                    dropped.append(bn)
                    del self._pending_retry_blocks[bn]
                else:
                    retry.append(bn)
        if dropped:
            warn(
                "ALERT",
                f"pool understatement: dropped {len(dropped)} block(s) after "
                f"exhausting receipt retries (blocks={sorted(dropped)[:10]}); "
                f"their bets remain missing from the pool aggregate",
            )
        if retry:
            # Single extra batched call; success/failure bookkeeping happens
            # inside _fetch_and_process_blocks (pop-on-recover, increment-on-fail).
            self._fetch_and_process_blocks(sorted(retry))

    def _process_receipts_for_block(self, block_number: int, receipts: list[dict]) -> None:
        """Extract BetBull/BetBear events from a block's receipts and
        update the local pool state. Same log-id dedup + epoch-gate
        behaviour as the prior PoolEventWatcher._process_bet_event.

        Bundle 5 v2 (2026-05-14): when a bet log is detected, resolve
        the block timestamp lazily via a single ``eth_getBlockByNumber``
        call (cached in ``_block_ts``). The receipts-only backfill no
        longer pre-populates ``_block_ts`` from a bundled header call,
        so this lazy path is the canonical source of block_ts for the
        pool_cutoff filter. Bet rate is sparse (~5-20 per round across
        ~600 backfilled blocks); the single-RPC cost per bet-containing
        block is off the critical path.
        """
        bet_logs: list[dict] = []
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
                if topic0 == _BET_BULL_TOPIC or topic0 == _BET_BEAR_TOPIC:
                    bet_logs.append(log)

        if not bet_logs:
            return  # no bets in this block — nothing to do

        # Resolve block_ts ONCE for this block (cheaper than per-bet).
        # Cache hit if a prior pass already resolved this block (e.g.,
        # epoch_advance cache lookup); otherwise fire one RPC.
        with self._lock:
            cached_ts = self._block_ts.get(block_number, 0)
        block_ts = cached_ts if cached_ts > 0 else self._resolve_block_ts(block_number)

        for log in bet_logs:
            topic0 = log.get("topics", [None])[0]
            side = "Bull" if topic0 == _BET_BULL_TOPIC else "Bear"
            try:
                epoch = int(log["topics"][2], 16)
                amount_wei = int(log.get("data", "0x0"), 16)
                bn = int(log.get("blockNumber", "0x0"), 16)
            except (ValueError, IndexError, KeyError):
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
            bet_log_id = f"{tx_hash}:{log_idx}"
            with self._lock:
                processed_log_ids = self._processed_bet_log_ids.setdefault(epoch, set())
                if bet_log_id in processed_log_ids:
                    continue
                processed_log_ids.add(bet_log_id)
                if epoch not in self._pools:
                    self._pools[epoch] = _EpochPool()
                self._pools[epoch].bets.append(_Bet(
                    epoch=epoch, side=side, amount_wei=amount_wei,
                    block_number=bn, block_ts=block_ts,
                ))
                self._total_events += 1

    def _resolve_block_ts(self, block_number: int) -> int:
        """Best-effort lazy fetch of a block's timestamp via a single
        ``eth_getBlockByNumber(hex(bn), false)`` call. Returns 0 on any
        failure (transport error, malformed response). On success,
        caches the result in ``_block_ts`` and returns the timestamp
        in chain seconds.

        Called from ``_process_receipts_for_block`` when a bet log is
        detected. Sparse bet rate (~5-20 per round) means this fires
        at most ~5-20 times per round, off the critical path.
        """
        try:
            result = self._rpc_call_single(
                "eth_getBlockByNumber", [hex(block_number), False],
            )
        except Exception:  # noqa: BLE001
            return 0
        if not isinstance(result, dict):
            return 0
        ts_hex = result.get("timestamp")
        if not isinstance(ts_hex, str):
            return 0
        try:
            ts = int(ts_hex, 16)
        except ValueError:
            return 0
        if ts <= 0:
            return 0
        with self._lock:
            self._block_ts[block_number] = ts
        return ts

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

    def _rpc_post(self, url: str, body: bytes, *, timeout_seconds: float) -> bytes:
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

    def _do_hedged_post(self, body: bytes, *, timeout_seconds: float) -> tuple[str, bytes]:
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
        return self._rpc_call_single_with_timeout(
            method, params, timeout_s=float(_tc.RPC_HTTP_SINGLE_TIMEOUT_SECONDS),
        )

    def _rpc_call_single_with_timeout(
        self, method: str, params: list, *, timeout_s: float,
    ) -> Any:
        """Single JSON-RPC call with an explicit timeout (seconds, may be
        fractional). Hedged across every endpoint in the pool; first
        success wins. Used by the Bundle 5 v2 anchor poll, which wants
        a 200ms ceiling rather than the default 5s.
        """
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
        }).encode()
        _ep, resp_bytes = self._do_hedged_post(
            body, timeout_seconds=timeout_s,
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
