"""HTTP RPC poller for PancakeSwap PredictionV2 bet pools.

Era 11 (2026-05-07): replaces the WSS-subscription pool watcher.
Era 12 (2026-06-09): bet events fetched via ``eth_getLogs`` range queries
(replaced the per-block ``eth_getBlockReceipts`` firehose — byte-identical
extraction, day-rate ~60 GB -> ~7 MB).
Era 12b (2026-06-10): single-source read path. EVERY read RPC (getLogs +
head + anchor + round-start header + bet-block timestamps) goes to
``RPC_BLOXROUTE_ENDPOINT`` through ``_bloxroute_call`` — tight per-attempt
timeouts sized from measured p99, bounded per-call-site retries, and a
wall-clock cap on each poll operation. The prior 3-endpoint hedged dataseed
pool is gone: it could not serve getLogs (so it never provided event-fetch
failover), its head was redundant as a cross-check (the F0 coverage gate
compares the cursor's block-time against the CONTRACT-anchored cutoff, a
node-independent reference), and min(ref_head, bloxroute_head) could
false-skip rounds when the dataseeds lagged bloXroute. Failure layering:
momentary blip -> in-call retry; sustained degradation -> wall-cap abort /
feasibility -> F0 skip. Architecture: deterministic poll schedule. See:
- ``var/design/rpc_polling_architecture_2026_05_07.md`` (architecture)
- ``var/incident_reports/2026_05_07_rpc_polling_spike_results.md`` (provenance)

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
   Wall-clock cap ``_POLL_WALL_CAP_PERIODIC_MS`` bounds each tick.

3. **Single poll** — engine-driven catch-up before the critical path
   (Candidate C, 2026-06-06), called from the wake schedule.
   Synchronous; wall-clock capped at ``_POLL_WALL_CAP_SINGLE_MS`` (no
   single RPC attempt is even STARTED unless it can finish inside the
   cap, so the engine's downstream anchor/submit budget is protected
   by construction). RTT-exceeds-cap marks ``_last_poll_too_slow=True``
   for diagnostics, but skips are driven by the round-aware feasibility
   check (``catchup_infeasible_for_round``) and the F0 coverage gate,
   not by individual slow polls.

Public interface mirrors ``PoolEventWatcher`` where feasible
(``get_pool``, ``set_round_phase``, ``connected``, ``current_endpoint``,
``is_pool_ready``) so the engine call sites are minimally affected.

Persistent HTTP/1.1 connections via ``urllib3.PoolManager`` mean the
endpoint's TLS handshake amortizes across the bot's lifetime — after
warmup, every call (including the parallel block-ts fan-out) reuses
already-open sockets.
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


# THE read endpoint (Era 12b, 2026-06-10). Single source for every read RPC:
# eth_getLogs (bet events), eth_blockNumber (head), eth_getBlockByNumber
# (anchor poll + round-start header + lazy bet-block timestamps).
#
# Why bloXroute, alone:
#   - getLogs: the only validated no-key endpoint that serves it (the free
#     dataseed set rejects it with -32005 "limit exceeded" / 403 — verified
#     2026-06-09, research/survey_getlogs_endpoints_2026_06_09.py); byte-
#     identical extraction vs the retired getBlockReceipts path
#     (research/parity_getlogs_v2_blxr_2026_06_09.py: 114==114, IDENTICAL);
#     12h soak 2026-06-09: 5400 polls, 0 errors, 675/675 parity clean.
#   - block-ts: bloXroute served the bet log, so it PROVABLY has that block —
#     resolving the timestamp on the same endpoint makes the null-result
#     ("node hasn't synced block N yet") failure mode structurally impossible.
#   - head: the getLogs toBlock is bloXroute's own head, so a range beyond its
#     synced tip (-32000 "invalid block range params") cannot be requested by
#     construction, and a dataseed pool lagging bloXroute can no longer cap
#     the cursor below the pool cutoff (false F0 skip).
#   - latency (VM, n=100 each, 2026-06-10): blockNumber p50 28 / p99 69;
#     getBlockByNumber(latest) p50 29 / p99 71; getBlockByNumber(bn) p50 30 /
#     p99 42; getLogs soak p50 19 / p99 145 / max 492.
# Failure layering: momentary blip -> in-call retry (_bloxroute_call);
# sustained degradation -> wall-cap abort / feasibility -> F0 skip. A getLogs-
# endpoint failure means no bet data regardless of any other endpoint, so a
# second read source would add cross-node skew, not availability.
RPC_BLOXROUTE_ENDPOINT: str = "https://bsc.rpc.blxrbdn.com"

# Max blocks per eth_getLogs range. Sized to cover every fetch path in ONE
# call: steady ~8s poll (~18 blocks), engine single poll (~18), and cold-start
# catch-up (<= 667 = one 5-min round). Backlogs above this loop in chunks.
_GETLOGS_CHUNK_BLOCKS: int = 750

# Conservative p99 wallclock for one getLogs chunk fetch — the per-chunk cost
# in the catch-up feasibility estimate. Soak p99 ~137ms (18-block ranges) /
# survey ~90ms (1000-block); 250ms leaves headroom for tail latency.
_GETLOGS_FETCH_RTT_P99_MS: int = 250

# -- Per-attempt RPC timeouts (ms), sized comfortably above measured p99 ----
# (VM measurements above). A timeout is a FAIL-FAST bound, not a budget: a
# healthy call returns in p50 ~20-30ms; a call that hits one of these is
# treated as failed and retried (or surfaced) instead of hanging the poll.
_GETLOGS_TIMEOUT_MS: int = 600       # > soak max 492ms
_BLX_HEAD_TIMEOUT_MS: int = 250      # eth_blockNumber, > ~3.6x p99 69ms
_BLX_HEADER_TIMEOUT_MS: int = 250    # getBlockByNumber(latest), > ~3.5x p99 71ms
_BLX_BLOCK_TS_TIMEOUT_MS: int = 250  # getBlockByNumber(bn), > ~5.9x p99 42ms
# (anchor poll keeps its own 200ms ceiling via timing_constants.ANCHOR_POLL_TIMEOUT_MS,
# attempts=1 — its recovery path is the engine's static-deadline fallback.)

# Backoff between retry attempts. The endpoint just served (or is about to
# serve) other calls, so a blip is momentary — a short pause beats a long one.
_BLX_RETRY_BACKOFF_MS: int = 25

# Per-call-site attempt counts, derived from each site's time budget:
# the single poll runs inside the lock-2500ms -> critical-path window (wall
# cap 1000ms below), so one retry per call is all the budget affords; the
# periodic poll has the 8s interval (wall cap 4000ms), affording two.
# Sustained failure is NOT the retries' job — that is the wall cap +
# feasibility check + F0 gate, which skip the round cleanly.
_RPC_ATTEMPTS_SINGLE: int = 2
_RPC_ATTEMPTS_PERIODIC: int = 3

# -- Wall-clock caps: hard bound on one whole poll operation (head fetch +
# getLogs chunks + block-ts resolution, including all retries). Enforced
# BEFORE each RPC attempt ("would this attempt's timeout overrun the cap?"),
# so a poll can never run past its cap — by construction, not estimation.
#   single:   fires at lock - single_poll_offset (2500ms canonical); must
#             leave the anchor poll (lock-1500ms) + critical path intact.
#             1000ms < the engine's deadline_ms budget (~1105ms canonical);
#             _poll_now takes min(deadline_ms, cap) so a tighter engine
#             budget always wins.
#   periodic: must release _poll_lock well inside the 8s cadence so ticks
#             never pile up and the single poll's wake finds the lock free
#             (see _compute_periodic_timeout's safe_fire_latest, which
#             assumes this cap as the worst-case hold).
# On cap-abort mid-chunk the cursor is NOT advanced past that chunk; the next
# poll re-fetches the same range, and if coverage is short at decision time
# F0 skips the round.
_POLL_WALL_CAP_SINGLE_MS: int = 1000
_POLL_WALL_CAP_PERIODIC_MS: int = 4000

# Parallel block-ts resolution fan-out width. The single poll's chunk can
# carry up to ~20 unique bet blocks; resolving serially would put K
# sequential RPCs on the critical path (unbounded in K), so all K fire
# concurrently on the executor and total time is ~attempts x timeout
# INDEPENDENT of K. Width 8: bloXroute serves ~1.7x throughput at 20-wide
# (VM probe 2026-06-10) — wider fan-out inflates per-call latency without
# cutting wall time, and 8 keeps the connection pool small.
_BLOCK_TS_PARALLEL_WORKERS: int = 8

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

    Captured from any observed block header (coarse-phase: the poller's
    eth_getBlockByNumber head/anchor calls; fine-phase: dedicated
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


class RpcPoller:
    """Polls PredictionV2 bet events from BSC via ``eth_getLogs`` range
    queries over HTTP.

    Replaces ``PoolEventWatcher``. Public interface intentionally
    mirrors ``PoolEventWatcher`` so the engine integration is a
    rename rather than a rework.
    """

    def __init__(
        self,
        *,
        interval_seconds: int,
        contract_address: str = PREDICTION_V2_CONTRACT_ADDRESS,
        periodic_poll_interval_s: int = _tc.RPC_PERIODIC_POLL_INTERVAL_SECONDS,
        single_poll_wakeup_offset_before_lock_ms: int = 4750,
        pool_cutoff_seconds: int = 6,
    ) -> None:
        if interval_seconds <= 0:
            raise InvariantError("interval_seconds_nonpositive")
        if periodic_poll_interval_s <= 0:
            raise InvariantError("periodic_poll_interval_nonpositive")
        if single_poll_wakeup_offset_before_lock_ms <= 0:
            raise InvariantError("single_poll_wakeup_offset_ms_nonpositive")

        self._interval_seconds = int(interval_seconds)
        # F0 pool-coverage gate (Era 12): get_pool sizes the bet from bets with
        # block_ts < lock - pool_cutoff_seconds; is_pool_ready refuses to bet
        # unless the cursor has polled THROUGH that cutoff block — the hard
        # guarantee that sizing uses COMPLETE data despite the getLogs endpoint
        # (bloXroute) possibly lagging real time. See
        # _pool_coverage_shortfall_locked.
        self._pool_cutoff_seconds = int(pool_cutoff_seconds)

        # ThreadPoolExecutor for the parallel block-ts fan-out in
        # _process_bet_logs. Workers are lazily spawned, so polls with no
        # unresolved bet blocks (cache hits / no bets) cost nothing here.
        self._executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=_BLOCK_TS_PARALLEL_WORKERS,
                thread_name_prefix="rpc-blockts",
            )
        )

        # urllib3 PoolManager: persistent HTTP/1.1 connections to bloXroute.
        # Eliminates per-call DNS+TCP+TLS handshake cost. ``maxsize`` covers
        # the block-ts fan-out width plus the poll thread's own call, so the
        # parallel resolves each hold a persistent connection instead of
        # churning sockets.
        self._pool: urllib3.PoolManager = urllib3.PoolManager(
            num_pools=1,
            maxsize=_BLOCK_TS_PARALLEL_WORKERS + 2,
            headers={
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/json",
            },
        )

        self._contract_addr = contract_address.lower()
        self._periodic_poll_interval_s = int(periodic_poll_interval_s)
        # Engine-side single-poll lock-relative offset (config-derived in
        # production via pancakebot/config.py from pool_cutoff_seconds;
        # canonical value at pool_cutoff=6 is 4750ms). Used by the
        # lock-anchored _periodic_loop to suspend periodic ticks that
        # would otherwise land inside the (lock - single_poll_offset, lock]
        # window and race the engine-driven single poll for the
        # non-blocking _poll_lock (Candidate C, 2026-06-06).
        self._single_poll_wakeup_offset_ms = int(single_poll_wakeup_offset_before_lock_ms)

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
        # single polls all read+write under self._lock to keep
        # log-id dedup honest.
        self._last_polled_block_number: int = 0

        # Connection / readiness state.
        # Bundle 2 (2026-05-13): True after the first successful
        # periodic/single poll latches via
        # _latch_first_successful_poll_locked. Until then, is_pool_ready
        # returns (False, "cold_start_in_progress") and the engine skips.
        self._connected: bool = False
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
        # Pool-understatement counter (3.2): bets dropped from the pool
        # aggregate because their block timestamp could not be resolved
        # (eth_getBlockByNumber failed) — silently understates a side's pool,
        # so surface it. (Era 12 removed the per-block receipt-skip retry
        # queue: a getLogs range either returns ALL the range's Bet logs or
        # raises; on raise the cursor doesn't advance and the next poll
        # re-fetches the same range — no per-block skip/retry bookkeeping.)
        self._pool_block_ts_zero_drops_total: int = 0

    # ------------------------------------------------------------------
    # Public properties (mirror PoolEventWatcher)
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True after cold-start completes successfully."""
        return self._connected

    @property
    def current_endpoint(self) -> str:
        return RPC_BLOXROUTE_ENDPOINT

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self._connected,
                "current_endpoint": RPC_BLOXROUTE_ENDPOINT,
                "poll_count": self._poll_count,
                "last_poll_at": self._last_poll_at,
                "last_poll_rtt_ms": self._last_poll_rtt_ms,
                "last_poll_succeeded": self._last_poll_succeeded,
                "last_poll_too_slow": self._last_poll_too_slow,
                "last_polled_block": self._last_polled_block_number,
                "epochs_tracked": len(self._pools),
                "total_events": self._total_events,
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
          - RPC timeout / transport failure (network glitch, bloXroute
            stalled), OR
          - The latest block's mixHash decoded to milli_ts that didn't
            parse (malformed encoding — unexpected on a Lorentz chain
            and would indicate an out-of-spec validator).

        Either way, the engine treats None as "use the static fallback
        for this round" — no per-session memory of the failure, no
        retry. The next round fires a fresh anchor poll.
        """
        # attempts=1: the anchor's hard ~200ms ceiling has no room for a
        # retry, and unlike the data-completeness calls it doesn't need
        # one — the engine's static-deadline fallback IS the recovery
        # path. Any response slower than the ceiling counts as "missed
        # the window", not just transport-level timeout.
        try:
            block = self._bloxroute_call(
                "eth_getBlockByNumber", ["latest", False],
                timeout_ms=int(timeout_s * 1000), attempts=1,
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
            shortfall = self._pool_coverage_shortfall_locked()
        # F0 coverage gate (Era 12) — the HARD GUARANTEE that bet sizing uses
        # COMPLETE pool data. If the cursor (clamped to bloXroute in _poll_now)
        # has NOT polled through the pool-cutoff block, the get_pool sizing
        # window has a hole -> skip + ALERT rather than size on partial data.
        # ALERT outside the lock; expected ~once/round (the engine calls
        # is_pool_ready once at the decision). See _pool_coverage_shortfall_locked.
        if shortfall is None:
            return True, ""
        cursor_ms, cutoff_ms, blocks_short = shortfall
        warn(
            "ALERT",
            f"POOL UNCOVERED epoch={epoch}: getLogs endpoint lagged the "
            f"pool-cutoff block — cursor_block_ts={cursor_ms}ms < "
            f"cutoff_ts={cutoff_ms}ms (~{blocks_short} blocks short). Skipping "
            f"to avoid sizing the bet on an incomplete pool (F0 guarantee).",
        )
        return False, "pool_uncovered"

    def _pool_coverage_shortfall_locked(self) -> tuple[int, int, int] | None:
        """F0 pool-coverage check (call holding ``self._lock``).

        THE HARD GUARANTEE that bet sizing uses COMPLETE pool data. ``get_pool``
        sizes the bet from bets with ``block_ts < lock - pool_cutoff_seconds``
        (the "cutoff block" = the newest block that counts). Bet events come
        from the getLogs endpoint (bloXroute), whose head can lag the reference
        pool. If bloXroute lagged far enough that the cursor (clamped to
        bloXroute in ``_poll_now``) is SHORT of the cutoff block, the sizing
        window has a HOLE — the caller SKIPs rather than size on partial data.
        Normal jitter (<= ~1 block) leaves the cursor ~3.5s past the cutoff (the
        single poll fires lock-2500ms vs the lock-6000ms cutoff), so this fires
        only under a multi-second getLogs lag (degradation).

        Returns ``None`` if covered (or not evaluable), else
        ``(cursor_block_milli_ts, cutoff_milli_ts, blocks_short)``.

        The cursor block's BEP-520 ms-timestamp is extrapolated from the latest
        anchor by exact 450ms increments — NO extra RPC. The extrapolation
        error over the ~2-5 blocks between cursor and anchor is far below the
        ~3.5s cutoff buffer, so it cannot mask a real multi-second lag. Returns
        ``None`` (no gate) when no anchor or lock is set: cold-start is gated by
        ``_connected``, and an anchor-absent round already runs on the
        static-deadline fallback path.
        """
        if self._prev_anchor_block <= 0 or self._lock_at <= 0:
            return None
        cutoff_milli = (self._lock_at - self._pool_cutoff_seconds) * 1000
        cursor = self._last_polled_block_number
        cursor_milli = (
            self._prev_anchor_milli_ts
            + (cursor - self._prev_anchor_block) * _tc.BSC_BLOCK_TIME_MS
        )
        if cursor_milli >= cutoff_milli:
            return None
        blocks_short = (
            (cutoff_milli - cursor_milli + _tc.BSC_BLOCK_TIME_MS - 1)
            // _tc.BSC_BLOCK_TIME_MS
        )
        return (cursor_milli, cutoff_milli, blocks_short)

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
             f"RPC poller started endpoint={RPC_BLOXROUTE_ENDPOINT} "
             f"periodic={self._periodic_poll_interval_s}s "
             f"getlogs_chunk={_GETLOGS_CHUNK_BLOCKS} "
             f"wall_caps={_POLL_WALL_CAP_SINGLE_MS}/{_POLL_WALL_CAP_PERIODIC_MS}ms")

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
                # _process_bet_logs), so a single key list
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
            head = self._bloxroute_block_number(attempts=_RPC_ATTEMPTS_PERIODIC)
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
            head_num, head_ts, _head_milli = self._bloxroute_latest_header(
                attempts=_RPC_ATTEMPTS_PERIODIC,
            )
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
        """Estimated wallclock to catch up ``blocks_behind`` blocks via
        eth_getLogs range queries chunked at ``_GETLOGS_CHUNK_BLOCKS``.

        A getLogs range returns only the (sparse, server-filtered) Bet logs
        regardless of how many blocks it spans, so cost is per-CHUNK, not
        per-block: ``ceil(blocks_behind / _GETLOGS_CHUNK_BLOCKS)`` chunks,
        each at ``_GETLOGS_FETCH_RTT_P99_MS``. Conservative (static p99, no
        live-degradation term). With a 750-block chunk every realistic backlog
        — steady ~18, one round <= 667 — is a single ~250ms chunk, so the
        INFEAS gate now only trips for pathological multi-thousand-block
        backlogs very near lock.
        """
        if blocks_behind <= 0:
            return 0
        chunks = -(-blocks_behind // _GETLOGS_CHUNK_BLOCKS)  # ceil division
        return chunks * _GETLOGS_FETCH_RTT_P99_MS

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
    # Engine integration: deadline-driven single poll
    # ------------------------------------------------------------------

    def poll(self, deadline_ms: int = 0) -> None:
        """Engine-driven single batched catch-up poll before the critical
        path (Candidate C, 2026-06-06 — replaced the 3-leg ramp ladder).
        Synchronous; blocks until complete or until RTT exceeds deadline_ms
        (0 = no deadline).

        Side-effects (diagnostics only — none of these directly cause round
        skips; the round-aware feasibility check is the canonical skip
        signal):
          - On success: _last_poll_succeeded=True, _last_poll_too_slow=False.
          - On RTT-exceeds-deadline: _last_poll_too_slow=True.
          - On RPC error: _last_poll_succeeded=False.
        Skips are driven by ``_catchup_infeasible_for_round`` which the
        feasibility check (in _on_epoch_advance and _poll_now) sets when
        math says we cannot catch up before lock_at.
        """
        self._poll_now(deadline_ms=deadline_ms, label="single")

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
                    head, head_ts, _head_milli = self._bloxroute_latest_header(
                        attempts=_RPC_ATTEMPTS_PERIODIC,
                    )
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
              single_poll_window_start — fire at next_at.
            - the anchored tick is outside the single-poll window but its
              worst-case HTTP RTT could extend past single_poll_window_start
              — reschedule to fire at the latest safe time
              (``single_poll_window_start − max_rtt − safety``). If that
              time has already passed (previous poll overran), suspend.
            - the anchored tick lands INSIDE the single-poll window — suspend.

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

        Before lock, the periodic poll must not race the engine-driven
        ``single`` poll for ``_poll_lock``. The single poll's wake fires at
        ``lock_at − single_poll_offset`` and uses non-blocking acquire; if a
        periodic poll is still in flight at that moment, the single poll is
        silently dropped. Two distinct dangers:

          * The anchored tick lands INSIDE the single-poll window. Suspend
            (sleep past lock and let the post-lock branch resume).
            (Example: single_poll_window_start=lock_at−4.75s, next_at=lock_at−2s.)

          * The anchored tick is just BEFORE the window, but its worst-case
            duration (the periodic wall cap) plus jitter could extend it past
            single_poll_window_start. Try to fire earlier, at
            ``single_poll_window_start − wall_cap − safety``, so a cap-bound
            poll completes before the single poll's wake. If the latest safe
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
        single_poll_window_start = (
            lock_at - self._single_poll_wakeup_offset_ms / 1000.0
        )
        # Latest moment a worst-case periodic poll can BEGIN and still
        # finish before the single-poll window opens. The wall-clock cap
        # bounds a periodic poll's _poll_lock hold to
        # _POLL_WALL_CAP_PERIODIC_MS by construction (no attempt starts
        # unless its timeout fits the cap); firing later than this would
        # leave the lock held when the single poll wakes.
        safe_fire_latest = (
            single_poll_window_start
            - _POLL_WALL_CAP_PERIODIC_MS / 1000.0
            - _tc.RPC_PERIODIC_TO_SINGLE_POLL_SAFETY_BUFFER_SECONDS
        )
        k = max(1, int((now - round_open) // period) + 1)
        next_at = round_open + k * period

        # The anchored tick lands inside the single-poll window itself —
        # firing here would either race the single poll directly or block it.
        # Sleep past lock and let the post-lock branch resume.
        if next_at >= single_poll_window_start:
            return max(0.1, lock_at + 0.1 - now)

        # The anchored tick is outside the single-poll window but close
        # enough that its worst-case RTT could extend into it. Try to fire
        # earlier at the latest safe time; if that time has already passed
        # (the previous poll overran), suspend instead of firing a doomed
        # poll that would still overlap.
        # Example (reschedule): canonical config (period=8, single_poll=4.75,
        #   wall_cap=4, safety=0.05) → safe_fire_latest=lock_at−8.8. At
        #   now=lock_at−9, anchored next_at=lock_at−8.5 falls in
        #   (safe_fire_latest=lock_at−8.8, single_poll_window_start=lock_at−4.75).
        #   Reschedule to lock_at−8.8.
        # Example (overrun → suspend): previous poll completed at lock_at−8.7
        #   (overran past safe_fire_latest=lock_at−8.8 by 100 ms). No safe
        #   time remains; suspend and let the single poll absorb the backlog.
        if next_at > safe_fire_latest:
            if safe_fire_latest > now:
                return safe_fire_latest - now
            return max(0.1, lock_at + 0.1 - now)

        # Anchored tick has comfortable margin before the single-poll window.
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
        in eth_getLogs ranges of _GETLOGS_CHUNK_BLOCKS, until caught up to
        bloXroute's head.

        Wall-clock capped (``_POLL_WALL_CAP_SINGLE_MS`` /
        ``_POLL_WALL_CAP_PERIODIC_MS``, further tightened by ``deadline_ms``
        when the engine passes one): no RPC attempt is started unless its
        timeout fits inside the cap, so the poll's total duration is bounded
        by construction. A cap-abort mid-chunk raises out of the chunk's
        processing, the cursor is NOT advanced past that chunk, and the next
        poll re-fetches the same range (F0 skips the round if coverage is
        short at decision time).

        Updates _last_poll_succeeded / _last_poll_too_slow / etc on
        completion.
        """
        if not self._poll_lock.acquire(blocking=False):
            # Another poll is in flight; skip this one. Periodic polls
            # are advisory; if the single poll is concurrent, they
            # share the same data anyway. A dropped single poll IS a
            # critical-path event (cursor advance skipped on the wake the
            # engine scheduled), so surface it (guard audit 1.2); periodic
            # drops stay silent by design.
            if label == "single":
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
            is_single = label == "single"
            attempts = _RPC_ATTEMPTS_SINGLE if is_single else _RPC_ATTEMPTS_PERIODIC
            cap_ms = _POLL_WALL_CAP_SINGLE_MS if is_single else _POLL_WALL_CAP_PERIODIC_MS
            if deadline_ms > 0:
                cap_ms = min(cap_ms, deadline_ms)
            abort_at = time.monotonic() + cap_ms / 1000.0

            # Head = bloXroute's own tip. The getLogs toBlock can never
            # exceed what bloXroute has synced (-32000 "invalid block range
            # params") because it IS bloXroute's reported head. The cursor
            # tracks bloXroute's polled position contiguously — NO block is
            # skipped — and the F0 gate in is_pool_ready is the hard
            # guarantee that we never BET while the cursor is short of the
            # pool-cutoff block (the cutoff is contract+clock-anchored, so
            # bloXroute lagging real time cannot hide from it).
            try:
                head = self._bloxroute_block_number(
                    attempts=attempts, abort_at=abort_at,
                )
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._last_poll_succeeded = False
                    self._last_poll_error = f"head_fetch:{type(e).__name__}:{e}"
                warn("ALERT",
                     f"{label} poll: bloXroute eth_blockNumber failed: {self._last_poll_error}")
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

            # Fetch in eth_getLogs ranges, wall-clock capped. Between chunks:
            # abort cleanly (completed chunks' data is kept, cursor advanced
            # through them). Mid-chunk: _fetch_and_process_logs raises on its
            # own cap check, which lands in the except below — the cursor is
            # NOT advanced past that chunk.
            for chunk_start in range(from_block, head + 1, _GETLOGS_CHUNK_BLOCKS):
                if time.monotonic() >= abort_at:
                    elapsed_ms = int((time.time() - t_start) * 1000)
                    warn("ALERT",
                         f"{label} poll wall-cap exceeded before chunk_start={chunk_start}: "
                         f"elapsed={elapsed_ms}ms > cap={cap_ms}ms; "
                         f"aborting remaining chunks")
                    break
                chunk_end = min(chunk_start + _GETLOGS_CHUNK_BLOCKS - 1, head)
                try:
                    self._fetch_and_process_logs(
                        chunk_start, chunk_end,
                        attempts=attempts, abort_at=abort_at,
                    )
                except Exception as e:  # noqa: BLE001
                    error_seen = f"getlogs[{chunk_start}..{chunk_end}]: {type(e).__name__}: {e}"
                    # Transient getLogs-endpoint failures: the cursor is NOT
                    # advanced past a failed chunk, so the next poll re-fetches
                    # the same range; the feasibility check prevents the
                    # catch-up backlog from compounding.
                    warn("ALERT", f"{label} poll getlogs failed: {error_seen}")
                    break
                blocks_polled += (chunk_end - chunk_start + 1)
                with self._lock:
                    # Forward-only: a poll whose range was snapshotted before
                    # an epoch-advance cursor jump must never stomp the jump
                    # backward with its (lower) chunk_end values.
                    self._last_polled_block_number = max(
                        self._last_polled_block_number, chunk_end,
                    )

            rtt_ms = int((time.time() - t_start) * 1000)
            too_slow = rtt_ms > cap_ms
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
            # every 8s (periodic) + the single poll = ~30-50/round, mostly
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

    def _fetch_and_process_logs(
        self, from_block: int, to_block: int,
        *, attempts: int = 1, abort_at: float | None = None,
    ) -> None:
        """Fetch Bet logs in [from_block, to_block] via a SINGLE eth_getLogs
        range query (server-side filtered by contract address + Bet topic0),
        then process them.

        Era 12 (2026-06-09) replacement for the per-block eth_getBlockReceipts
        firehose: receipts returned every full receipt in the range (~315
        KB/block, ~60 GB/day) just to find the sparse Bet logs, and the
        dataseed pool errored on ~5% of those heavy responses. getLogs returns
        only the Bet logs (~KB), byte-identical extraction
        (research/parity_getlogs_v2_blxr_2026_06_09.py: 114==114), at ~7 MB/day.

        Raises on transport/RPC failure (after ``attempts`` tries) or on the
        wall-clock cap so the caller aborts the chunk loop with the cursor
        un-advanced — the next poll re-fetches the same range. A range fetch
        is all-or-nothing, so there are no partial-block skips to track.
        """
        flt = {
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
            "address": self._contract_addr,
            "topics": [[_BET_BULL_TOPIC, _BET_BEAR_TOPIC]],
        }
        logs = self._bloxroute_call(
            "eth_getLogs", [flt],
            timeout_ms=_GETLOGS_TIMEOUT_MS, attempts=attempts,
            abort_at=abort_at,
        )
        if not isinstance(logs, list):
            raise InvariantError(
                f"getlogs_result_not_list: {type(logs).__name__}"
            )
        self._process_bet_logs(logs, attempts=attempts, abort_at=abort_at)

    def _process_bet_logs(
        self, logs: list[dict],
        *, attempts: int = 1, abort_at: float | None = None,
    ) -> None:
        """Extract BetBull/BetBear events from a batch of eth_getLogs log
        objects (which may span multiple blocks) and update the local pool
        state. Same log-id dedup + epoch-gate + lazy block_ts resolution as
        the prior receipts path — a getLogs log object has the identical shape
        to a receipt's log entry (address / topics / data / blockNumber /
        transactionHash / logIndex), so the per-log extraction is verbatim
        (parity-verified 2026-06-09).

        Logs arrive server-side-filtered (address + Bet topic0); the address /
        topic guards below are a defensive re-check.

        block_ts resolution: parse + epoch-gate ALL logs first (gate-dropped
        archive bets never cost an RPC), then resolve every still-missing
        block timestamp IN PARALLEL on the executor — total resolution time
        is ~attempts x timeout independent of how many unique bet blocks the
        chunk carries. The wall-clock cap is enforced when gathering: if the
        cap expires with resolutions pending, this raises and the caller
        leaves the cursor un-advanced (the next poll re-fetches; abandoned
        workers still populate ``_block_ts`` when they finish, so the
        re-fetch hits cache). Bets are appended only after the full chunk
        resolves — a cap-abort loses the whole chunk cleanly, never a
        partial append (dedup makes the re-fetch idempotent regardless).
        """
        bet_logs: list[dict] = []
        for log in logs:
            if not isinstance(log, dict):
                continue
            if (log.get("address") or "").lower() != self._contract_addr:
                continue
            topics = log.get("topics") or []
            if len(topics) < 3:
                continue
            topic0 = topics[0]
            if topic0 == _BET_BULL_TOPIC or topic0 == _BET_BEAR_TOPIC:
                bet_logs.append(log)

        if not bet_logs:
            return  # no bets in this range — nothing to do

        # Phase 1 — parse + epoch-gate (no RPC).
        parsed: list[tuple[int, str, int, int, str, Any]] = []
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
            # Epoch gate — BEFORE any block_ts cost.
            if self._current_epoch >= 0 and epoch not in (
                self._current_epoch, self._current_epoch + 1
            ):
                continue
            parsed.append((
                epoch, side, amount_wei, bn,
                log.get("transactionHash", ""), log.get("logIndex", ""),
            ))
        if not parsed:
            return

        # Phase 2 — resolve block_ts for every unique bet block: cache first,
        # then one parallel wave for the misses.
        unique_bns = {p[3] for p in parsed}
        ts_map: dict[int, int] = {}
        with self._lock:
            for bn in unique_bns:
                cached = self._block_ts.get(bn, 0)
                if cached > 0:
                    ts_map[bn] = cached
        missing = sorted(unique_bns - ts_map.keys())
        if missing:
            if abort_at is not None and time.monotonic() >= abort_at:
                raise InvariantError(
                    f"poll_wall_cap_exceeded:block_ts_not_started:"
                    f"{len(missing)}_blocks"
                )
            fut_to_bn = {
                self._executor.submit(
                    self._resolve_block_ts, bn, attempts=attempts,
                ): bn
                for bn in missing
            }
            timeout_s = (
                None if abort_at is None
                else max(0.001, abort_at - time.monotonic())
            )
            done, pending = concurrent.futures.wait(
                fut_to_bn.keys(), timeout=timeout_s,
            )
            if pending:
                # Abandoned workers run to completion on their own (each is
                # bounded by attempts x timeout) and cache their results.
                raise InvariantError(
                    f"poll_wall_cap_exceeded:block_ts_pending:"
                    f"{len(pending)}_of_{len(missing)}"
                )
            for fut in done:
                # _resolve_block_ts never raises; 0 = unresolved (the
                # sizing-time drop + ALERT in get_pool is the backstop).
                ts_map[fut_to_bn[fut]] = fut.result()

        # Phase 3 — dedup + append.
        for epoch, side, amount_wei, bn, tx_hash, log_idx in parsed:
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
                    block_number=bn, block_ts=ts_map.get(bn, 0),
                ))
                self._total_events += 1

    def _resolve_block_ts(self, block_number: int, *, attempts: int = 1) -> int:
        """Best-effort lazy fetch of a block's timestamp via
        ``eth_getBlockByNumber(hex(bn), false)`` on bloXroute. Returns 0
        after ``attempts`` failed tries (25ms backoff between them); never
        raises. On success, caches the result in ``_block_ts`` and returns
        the timestamp in chain seconds.

        bloXroute served the bet log this block number came from, so it
        provably has the block — a null result ("node hasn't synced block N
        yet", the old hedged-pool drop cause) is structurally impossible
        here; the retries cover momentary transport blips only. Residual
        (all attempts fail) is the sizing-time drop + ALERT in ``get_pool``.

        Runs on the block-ts executor (parallel across a chunk's unique
        blocks); per-call bound = attempts x (timeout + backoff).
        """
        result: Any = None
        for i in range(attempts):
            if i:
                time.sleep(_BLX_RETRY_BACKOFF_MS / 1000.0)
            try:
                result = self._bloxroute_call(
                    "eth_getBlockByNumber", [hex(block_number), False],
                    timeout_ms=_BLX_BLOCK_TS_TIMEOUT_MS, attempts=1,
                )
            except Exception:  # noqa: BLE001
                continue
            if isinstance(result, dict):
                break
            result = None
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
    # Internal: HTTP RPC helpers (all bloXroute)
    # ------------------------------------------------------------------

    def _bloxroute_call(
        self, method: str, params: list, *,
        timeout_ms: int, attempts: int, abort_at: float | None = None,
    ) -> Any:
        """THE transport for every read RPC: a JSON-RPC POST to
        ``RPC_BLOXROUTE_ENDPOINT`` with a tight per-attempt timeout and up to
        ``attempts`` tries (``_BLX_RETRY_BACKOFF_MS`` between them). Returns
        the decoded ``result`` field; raises the last error on exhaustion.

        ``abort_at`` (``time.monotonic()`` seconds) is the enclosing poll's
        wall-clock cap: an attempt whose timeout could not complete before
        the cap is NOT started (raises ``poll_wall_cap_exceeded``), which is
        what makes the cap a hard bound on the whole poll rather than an
        advisory check between calls.

        Retries cover momentary blips only (the backoff is deliberately
        short); sustained endpoint degradation is handled one level up by
        the wall cap + feasibility check + F0 gate, which skip cleanly.
        """
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
        }).encode()
        last_error: Exception | None = None
        for i in range(attempts):
            if i:
                time.sleep(_BLX_RETRY_BACKOFF_MS / 1000.0)
            if (
                abort_at is not None
                and time.monotonic() + timeout_ms / 1000.0 > abort_at
            ):
                raise InvariantError(
                    f"poll_wall_cap_exceeded:{method}:attempt_{i + 1}_not_started"
                )
            try:
                resp_bytes = self._rpc_post(
                    RPC_BLOXROUTE_ENDPOINT, body,
                    timeout_seconds=timeout_ms / 1000.0,
                )
                payload = json.loads(resp_bytes)
                if "error" in payload:
                    raise InvariantError(f"rpc_error:{method}:{payload['error']}")
                return payload.get("result")
            except Exception as e:  # noqa: BLE001
                last_error = e
        assert last_error is not None  # attempts >= 1 guaranteed by callers
        raise last_error

    def _bloxroute_block_number(
        self, *, attempts: int, abort_at: float | None = None,
    ) -> int:
        """Current head block number as seen by bloXroute. This IS the poll
        head: a getLogs range never requests a toBlock beyond it, so the
        -32000 "invalid block range params" reply (range past the endpoint's
        synced tip) is unreachable by construction. Raises on transport/RPC
        error; the caller fails the poll (the next poll retries)."""
        result = self._bloxroute_call(
            "eth_blockNumber", [],
            timeout_ms=_BLX_HEAD_TIMEOUT_MS, attempts=attempts,
            abort_at=abort_at,
        )
        if not isinstance(result, str):
            raise InvariantError(f"eth_blockNumber_unexpected_result: {result!r}")
        return int(result, 16)

    def _bloxroute_latest_header(
        self, *, attempts: int,
    ) -> tuple[int, int, int | None]:
        """Return ``(head_block_number, head_ts_seconds, head_milli_ts)``
        via ``eth_getBlockByNumber("latest", false)``. ``head_milli_ts`` is
        the BEP-520 ms-precise timestamp (``compute_milli_ts``), or None if
        the mixHash decode fails (callers fall back to seconds-granularity
        math with extra margin). Used by cursor-init and the round-start
        computation. Raises on error."""
        result = self._bloxroute_call(
            "eth_getBlockByNumber", ["latest", False],
            timeout_ms=_BLX_HEADER_TIMEOUT_MS, attempts=attempts,
        )
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
        return int(num_hex, 16), int(ts_hex, 16), compute_milli_ts(result)

    def _rpc_post(self, url: str, body: bytes, *, timeout_seconds: float) -> bytes:
        """HTTP POST via the shared urllib3 PoolManager. Returns the raw
        response body. Raises on transport-level failure
        (``urllib3.exceptions.HTTPError`` subclasses: TimeoutError,
        MaxRetryError, ConnectTimeoutError, etc.) or non-200 status.
        The caller parses JSON and decodes the JSON-RPC envelope.

        Persistent connections via the shared PoolManager mean the first
        call (per pooled connection) pays the TLS handshake; subsequent
        calls reuse the open socket.
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
