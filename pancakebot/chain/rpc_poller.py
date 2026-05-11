"""HTTP RPC poller for PancakeSwap PredictionV2 bet pools.

Era 11 (2026-05-07): replaces the WSS-subscription pool watcher.
Architecture: deterministic poll schedule using batched
``eth_getBlockReceipts``. See:
- ``var/design/rpc_polling_architecture_2026_05_07.md`` (architecture)
- ``var/incident_reports/2026_05_07_rpc_polling_spike_results.md`` (provenance)
- ``var/design/rpc_endpoint_hedging_2026_05_08.md`` (hedging transport)

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

Endpoint hedging: the ``hedge_fan_out`` ctor knob defaults to 1
(single-endpoint, bit-identical with the pre-hedging codepath) and
is also exposed as ``[runtime].hedge_fan_out`` in config.toml.
Production currently runs N=4 across the top-4 batched BSC
endpoints; tests and legacy fixtures keep the ctor default of 1.

When ``hedge_fan_out > 1`` each JSON-RPC call fans out to N
endpoints in parallel; the first successful response wins. The
other futures' outcomes are still recorded in the health tracker
(Bug #7 fix, 2026-05-10) — already-done siblings with their actual
RTT, still-pending siblings via add_done_callback using a sentinel
RTT (``_RTT_SENTINEL_UNKNOWN``). The sentinel is filtered out of
the p50/p99 latency gates but still counts toward success_rate and
consecutive_failures, so a slow-but-broken endpoint accumulates
the right kind of evidence to eventually trip the health gates.

Per-endpoint health is tracked in ``EndpointHealthTracker``
(rolling 100-outcome window; success_rate/p99/consecutive_failures
gates).
"""
from __future__ import annotations

import collections
import concurrent.futures
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


# HTTP RPC endpoints (legacy single-endpoint default). Pre-hedging
# default; new code paths use ``DEFAULT_HEDGED_ENDPOINTS`` when
# hedging is enabled via config. Kept for backwards compatibility
# with existing callers passing ``rpc_urls`` only.
RPC_BATCH_ENDPOINTS: list[str] = [
    "https://bsc-rpc.publicnode.com",
]

# Hedging-mode default endpoint pool. Top 3 by P50 batch RTT from the
# 2026-05-08 Track H respike (n=200, batch_size=20), extended 2026-05-10
# with two non-bsc-dataseed-family providers to defeat correlated
# multi-endpoint outages observed in production (2026-05-09/10: hours-
# long windows where ALL bsc-dataseed* endpoints timed out at 5s
# simultaneously, confirming shared upstream infrastructure):
#
#   bsc-dataseed1.defibit.io   p50=770ms  p99=2226ms  (BSC dataseed family)
#   bsc-dataseed1.ninicoin.io  p50=802ms  p99=2179ms  (BSC dataseed family)
#   bsc-dataseed1.binance.org  p50=828ms  p99=1797ms  (BSC dataseed family)
#   bsc-dataseed3.binance.org  p50=898ms  p99=1290ms  (BSC dataseed family)
#   bsc-rpc.publicnode.com     p50=938ms  p99=1842ms  (Allnodes, distinct)
#   bsc.rpc.blxrbdn.com        p50~250ms  batch~430ms (bloXroute, distinct)
#
# Pool of 6; hedge_fan_out picks top-N by p50 via EndpointHealthTracker.
# publicnode and bloXroute are kept in the pool even when not selected
# by pick_n's p50 ordering so they're available as failover when the
# bsc-dataseed family experiences correlated outages.
#
# Use this list when constructing an RpcPoller with
# ``hedge_fan_out >= 2``. NOT the default for single-endpoint
# operation (``hedge_fan_out=1``); see ``RPC_BATCH_ENDPOINTS``.
DEFAULT_HEDGED_ENDPOINTS: list[str] = [
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-dataseed1.ninicoin.io",
    "https://bsc-dataseed1.binance.org",
    "https://bsc-dataseed3.binance.org",
    "https://bsc-rpc.publicnode.com",
    "https://bsc.rpc.blxrbdn.com",
]

_USER_AGENT = "pancakebot-rpc-poller/1.0"

# EndpointHealthTracker rolling-window size (per-endpoint).
_HEALTH_WINDOW_SIZE: int = 100
# Health classification gates (memo §4.2).
_HEALTH_MIN_SUCCESS_RATE: float = 0.90
_HEALTH_MAX_P99_RTT_MS: int = 5000
_HEALTH_MAX_CONSECUTIVE_FAILURES: int = 5
# Periodic re-test cadence: 1-in-N requests, AND any unhealthy
# endpoint that hasn't been retested in this many seconds gets a
# probe regardless of the counter.
_HEALTH_RETEST_EVERY_N_REQUESTS: int = 10
_HEALTH_RETEST_FORCED_AFTER_SECONDS: float = 60.0

# Sentinel RTT for outcomes whose wallclock isn't known to the caller
# at record time. Specifically: hedged-fan-out siblings that are still
# pending when the winning future returns are recorded via a done-
# callback in their executor thread, but the actual elapsed wallclock
# from request-start to that completion is not captured (the
# fut_start map is local to the original call; capturing it through
# the callback is what the recording does). To keep that case from
# polluting p50/p99 with values that are correlated with "won the
# hedge race" rather than baseline endpoint latency, the success/
# failure flag IS recorded (so consecutive_failures and
# success_rate stay honest) but the RTT is set to this sentinel,
# which is then filtered OUT of the rolling-window p50/p99 stats.
# See Bug #7 (2026-05-10): abandoned-future observation gap.
_RTT_SENTINEL_UNKNOWN: int = -1


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


@dataclass
class _EndpointHealth:
    url: str
    # Rolling window of (success, rtt_ms) tuples (max _HEALTH_WINDOW_SIZE).
    recent_outcomes: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=_HEALTH_WINDOW_SIZE),
    )
    consecutive_failures: int = 0
    last_failure_at: float = 0.0
    last_request_at: float = 0.0
    total_requests: int = 0


class EndpointHealthTracker:
    """Per-endpoint rolling-window health bookkeeping for the hedging
    transport.

    State per endpoint: rolling window of last 100 (success, rtt_ms)
    outcomes, consecutive-failure counter, last-failure timestamp,
    total-request counter. Outcomes recorded for abandoned hedged-
    fan-out siblings carry ``rtt_ms == _RTT_SENTINEL_UNKNOWN`` and
    are filtered out of the p50/p99 latency calculations but still
    count toward success_rate / consecutive_failures (Bug #7 fix,
    2026-05-10).

    An endpoint is **healthy** iff it has fewer than 100 outcomes
    (warmup; unconditionally healthy unless consecutive_failures
    fast-trips) OR all three:
      - success_rate over the window > 0.90
      - p99 RTT (sentinel-filtered) over the window < 5000 ms
      - consecutive_failures < 5

    ``pick_n(n)`` returns up to n endpoints, healthy first sorted by
    p50 RTT ascending. If too few healthy endpoints are available it
    falls back to unhealthy ones (degraded mode beats no-RPC mode).
    Periodic re-test: one unhealthy endpoint is included in pick_n
    every ``_HEALTH_RETEST_EVERY_N_REQUESTS`` requests, OR any
    unhealthy endpoint not probed in the last 60s, so recovered
    endpoints can rejoin the pool.

    Thread-safe via internal ``threading.Lock``.
    """

    def __init__(self, endpoints: list[str]) -> None:
        if not endpoints:
            raise InvariantError("endpoint_health_tracker_empty_pool")
        self._lock = threading.Lock()
        self._endpoints: list[str] = list(endpoints)
        self._health: dict[str, _EndpointHealth] = {
            url: _EndpointHealth(url=url) for url in self._endpoints
        }
        # Global pick_n counter — drives the 1-in-N retest pressure.
        self._pick_counter: int = 0
        # Last-known healthy bool for transition logging.
        self._last_healthy_state: dict[str, bool] = {url: True for url in self._endpoints}

    # ------------------------------------------------------------------
    # Recording outcomes
    # ------------------------------------------------------------------

    def record(self, endpoint: str, *, success: bool, rtt_ms: int) -> None:
        """Record a request outcome. Idempotent for unknown endpoints
        (no-op + invariant violation in tests is more harmful than
        silent skip during fanned-out edge cases).

        ``rtt_ms`` may be ``_RTT_SENTINEL_UNKNOWN`` (-1) when the actual
        elapsed wallclock isn't captured at record time. In that case
        the outcome (success/failure) still counts toward success_rate
        and consecutive_failures, but the RTT is filtered OUT of the
        rolling-window p50/p99 (otherwise sentinel values would skew
        the latency stats). See Bug #7 (2026-05-10).
        """
        with self._lock:
            h = self._health.get(endpoint)
            if h is None:
                # Unknown endpoint: silently ignore. Defensive — the
                # hedging transport only ever calls record() with
                # endpoints it got from pick_n(), but this protects
                # against future refactors.
                return
            h.recent_outcomes.append((bool(success), int(rtt_ms)))
            h.total_requests += 1
            h.last_request_at = time.time()
            if success:
                h.consecutive_failures = 0
            else:
                h.consecutive_failures += 1
                h.last_failure_at = time.time()
            self._maybe_log_transition_locked(endpoint, h)

    # ------------------------------------------------------------------
    # Health classification
    # ------------------------------------------------------------------

    def _is_healthy_locked(self, h: _EndpointHealth) -> bool:
        n = len(h.recent_outcomes)
        if n < _HEALTH_WINDOW_SIZE:
            # Warmup: insufficient evidence to mark unhealthy. The
            # consecutive-failures fast-trip still applies, so a brand-
            # new endpoint that fails 5 in a row is correctly excluded.
            if h.consecutive_failures >= _HEALTH_MAX_CONSECUTIVE_FAILURES:
                return False
            return True
        if h.consecutive_failures >= _HEALTH_MAX_CONSECUTIVE_FAILURES:
            return False
        successes = sum(1 for ok, _ in h.recent_outcomes if ok)
        if (successes / n) <= _HEALTH_MIN_SUCCESS_RATE:
            return False
        # p99 RTT over the window. Filter out sentinel RTTs (recorded
        # by abandoned-future callbacks where the actual wallclock
        # wasn't captured) — see Bug #7. Empty-after-filter is treated
        # as "warmup-like for the RTT gate": skip the gate.
        rtts = sorted(
            rtt for _, rtt in h.recent_outcomes
            if rtt != _RTT_SENTINEL_UNKNOWN
        )
        if rtts:
            # 99th percentile: index = ceil(0.99 * len) - 1 (clamp).
            m = len(rtts)
            p99_idx = max(0, min(m - 1, int(0.99 * m)))
            if rtts[p99_idx] >= _HEALTH_MAX_P99_RTT_MS:
                return False
        return True

    def is_healthy(self, endpoint: str) -> bool:
        with self._lock:
            h = self._health.get(endpoint)
            if h is None:
                return False
            return self._is_healthy_locked(h)

    def _p50_locked(self, h: _EndpointHealth) -> int:
        # Filter out sentinel RTTs (abandoned-future recordings) — they
        # carry success/failure info but no measured wallclock; see Bug
        # #7. If everything in the window is sentinel-valued, fall
        # through to the warmup-rank-0 behaviour.
        rtts = sorted(
            rtt for _, rtt in h.recent_outcomes
            if rtt != _RTT_SENTINEL_UNKNOWN
        )
        if not rtts:
            # Warmup or sentinel-only: no measured latency yet. Sort
            # key needs SOMETHING — return 0 so warmup endpoints rank
            # ahead of post-warmup with measured latency. That's the
            # desired warmup behaviour: send traffic to under-measured
            # endpoints first to gather data.
            return 0
        return rtts[len(rtts) // 2]

    def _p99_locked(self, h: _EndpointHealth) -> int:
        rtts = sorted(
            rtt for _, rtt in h.recent_outcomes
            if rtt != _RTT_SENTINEL_UNKNOWN
        )
        if not rtts:
            return 0
        n = len(rtts)
        idx = max(0, min(n - 1, int(0.99 * n)))
        return rtts[idx]

    def _success_rate_locked(self, h: _EndpointHealth) -> float:
        n = len(h.recent_outcomes)
        if n == 0:
            return 1.0
        successes = sum(1 for ok, _ in h.recent_outcomes if ok)
        return successes / n

    # ------------------------------------------------------------------
    # Endpoint selection
    # ------------------------------------------------------------------

    def pick_n(self, n: int) -> list[str]:
        """Return up to n endpoints, healthy first sorted by p50
        ascending. Falls back to unhealthy endpoints when fewer than n
        healthy ones are available, plus a periodic 1-in-N retest of
        an unhealthy endpoint to allow recovery."""
        if n <= 0:
            return []
        with self._lock:
            self._pick_counter += 1
            counter = self._pick_counter

            healthy: list[tuple[int, str]] = []
            unhealthy: list[tuple[float, str]] = []
            for url, h in self._health.items():
                if self._is_healthy_locked(h):
                    healthy.append((self._p50_locked(h), url))
                else:
                    # Sort key: oldest last_failure_at first (longest-
                    # down endpoints get retested first).
                    unhealthy.append((h.last_failure_at, url))

            healthy.sort(key=lambda x: x[0])
            unhealthy.sort(key=lambda x: x[0])

            healthy_urls = [u for _, u in healthy]
            unhealthy_urls = [u for _, u in unhealthy]

            now = time.time()

            # Periodic re-test: include one unhealthy endpoint in the
            # pick whenever the global pick counter is divisible by N
            # OR an unhealthy endpoint hasn't been tested in 60+ s.
            should_retest = (
                bool(unhealthy_urls)
                and (
                    counter % _HEALTH_RETEST_EVERY_N_REQUESTS == 0
                    or any(
                        now - self._health[u].last_request_at >= _HEALTH_RETEST_FORCED_AFTER_SECONDS
                        for u in unhealthy_urls
                    )
                )
            )

            selected: list[str] = []
            if should_retest:
                # Pick the longest-down or longest-untested unhealthy
                # endpoint as the retest candidate.
                retest_candidates = [
                    u for u in unhealthy_urls
                    if now - self._health[u].last_request_at >= _HEALTH_RETEST_FORCED_AFTER_SECONDS
                ]
                if not retest_candidates:
                    retest_candidates = unhealthy_urls
                selected.append(retest_candidates[0])

            # Fill with healthy endpoints (preferred).
            for url in healthy_urls:
                if url in selected:
                    continue
                if len(selected) >= n:
                    break
                selected.append(url)

            # Still short — fall back to remaining unhealthy.
            for url in unhealthy_urls:
                if url in selected:
                    continue
                if len(selected) >= n:
                    break
                selected.append(url)

            return selected[:n]

    # ------------------------------------------------------------------
    # Diagnostics + transition logging
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, dict[str, Any]]:
        """Return per-endpoint health snapshot for stats display."""
        with self._lock:
            out: dict[str, dict[str, Any]] = {}
            for url, h in self._health.items():
                out[url] = {
                    "healthy": self._is_healthy_locked(h),
                    "success_rate": self._success_rate_locked(h),
                    "p50_rtt_ms": self._p50_locked(h),
                    "p99_rtt_ms": self._p99_locked(h),
                    "consecutive_failures": h.consecutive_failures,
                    "total_requests": h.total_requests,
                    "window_size": len(h.recent_outcomes),
                }
            return out

    def _maybe_log_transition_locked(self, endpoint: str, h: _EndpointHealth) -> None:
        """Emit a one-shot log line on healthy<->unhealthy transitions.
        Called under self._lock. Uses sub="HEDGE" (5 chars, fits sub
        width)."""
        prev = self._last_healthy_state.get(endpoint, True)
        curr = self._is_healthy_locked(h)
        if prev == curr:
            return
        self._last_healthy_state[endpoint] = curr
        sr = self._success_rate_locked(h)
        p99 = self._p99_locked(h)
        cf = h.consecutive_failures
        if curr:
            info("RPC_POLL", "HEDGE", "UP",
                 msg=(f"{endpoint} recovered: success_rate={sr:.2%} "
                      f"p99={p99}ms"))
        else:
            info("RPC_POLL", "HEDGE", "DOWN",
                 msg=(f"{endpoint} unhealthy: success_rate={sr:.2%} "
                      f"p99={p99}ms consec_fail={cf}"))


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
        endpoint_pool: list[str] | None = None,
        hedge_fan_out: int = 1,
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

        # endpoint_pool is the canonical pool used by the hedging
        # transport. ``rpc_urls`` is retained as a backwards-compat
        # alias (existing callers); when both are supplied,
        # ``endpoint_pool`` wins. Default: single-endpoint
        # publicnode (current behaviour).
        if endpoint_pool is not None:
            pool = list(endpoint_pool)
        elif rpc_urls is not None:
            pool = list(rpc_urls)
        else:
            pool = list(RPC_BATCH_ENDPOINTS)
        if not pool:
            # Existing test contract: empty rpc_urls -> "rpc_urls_empty".
            raise InvariantError("rpc_urls_empty")

        self._interval_seconds = int(interval_seconds)
        self._endpoint_pool: list[str] = pool
        # Keep _rpc_urls populated for any internal code/tests still
        # touching it; treat as alias of the pool.
        self._rpc_urls = list(self._endpoint_pool)

        if hedge_fan_out < 1:
            raise InvariantError(
                f"hedge_fan_out_below_one: {hedge_fan_out}"
            )
        if hedge_fan_out > len(self._endpoint_pool):
            raise InvariantError(
                f"hedge_fan_out_exceeds_pool_size: "
                f"fan_out={hedge_fan_out} pool_size={len(self._endpoint_pool)}"
            )
        self._hedge_fan_out: int = int(hedge_fan_out)

        # Per-endpoint health tracker. Always constructed so the stats
        # surface is uniform regardless of fan_out. At fan_out=1 the
        # tracker is exercised but pick_n always returns the single
        # endpoint (since there's only one in the pool).
        self._health = EndpointHealthTracker(self._endpoint_pool)

        # ThreadPoolExecutor for fan-out. Sized at fan_out * 4 per memo
        # §2 to give headroom for overlapping calls (cold-start + ramp
        # + final + periodic). Min size 1 to handle fan_out=1 cleanly.
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        if self._hedge_fan_out > 1:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, self._hedge_fan_out * 4),
                thread_name_prefix="rpc-hedge",
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
        # the hedging transport on each successful call). For
        # display/log purposes only — picking is driven by the health
        # tracker, not this field.
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
            base = {
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
                "hedge_fan_out": self._hedge_fan_out,
            }
        # endpoint_health acquires its own lock; do NOT nest under self._lock.
        base["endpoint_health"] = self._health.stats()
        return base

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
        if self._executor is not None:
            # wait=False — abandoned hedged requests should not block
            # shutdown. urllib has no real cancellation; the in-flight
            # sockets will time out on their own.
            self._executor.shutdown(wait=False)
            self._executor = None
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
           from blocks-behind and the operating-mode single-batch p99
           RTT (hedged-table column when ``hedge_fan_out>1``, single-
           endpoint baseline otherwise). If the estimate exceeds
           time-until-lock, set ``_catchup_infeasible_for_round`` so
           the engine skips with reason
           ``catchup_infeasible_for_round``.

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
        """Estimated wallclock to fetch ``blocks_behind`` blocks at
        the current operating-mode p99 RTT. Conservative — doesn't
        account for current degradation, and uses the static p99
        table not the live observed p99 from the health tracker.

        Hedging-aware: when ``hedge_fan_out > 1`` the per-fan_out
        hedged-P99 column is used (Track H respike, 2026-05-08).
        At ``hedge_fan_out=1`` this returns the single-endpoint
        baseline P99 unchanged.
        """
        if blocks_behind <= 0:
            return 0
        batches = (blocks_behind + self._batch_size - 1) // self._batch_size
        rtt_p99 = _tc.rpc_rtt_p99_for_batch(
            self._batch_size, hedge_n=self._hedge_fan_out,
        )
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
        """Single-endpoint HTTP POST. Returns the raw response body.
        Raises on transport-level failure (urllib.error.URLError,
        timeout, etc.). The caller parses JSON and decodes the
        JSON-RPC envelope.
        """
        req = _urllib_req.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
        )
        with _urllib_req.urlopen(req, timeout=timeout_seconds) as resp:
            return resp.read()

    def _do_hedged_post(self, body: bytes, *, timeout_seconds: int) -> tuple[str, bytes]:
        """Hedged HTTP POST. Fans out to ``hedge_fan_out`` endpoints
        in parallel, returns (endpoint_used, response_bytes) from the
        first successful endpoint, abandons the rest.

        At fan_out=1 this delegates to a single ``_rpc_post`` call —
        matching the pre-hedging codepath exactly (no executor, no
        fan-out overhead). This is the bit-identical fast path.

        Raises ``HedgedAllFailed`` if every endpoint fails. Records
        per-endpoint outcomes in the health tracker.
        """
        endpoints = self._health.pick_n(self._hedge_fan_out)
        if not endpoints:
            # pick_n only returns [] when n<=0; with hedge_fan_out>=1
            # validated at construction, this is a logic bug.
            raise InvariantError("rpc_no_endpoints_available")

        # Fast path for fan_out=1: keep the call shape identical to
        # the pre-hedging single-endpoint behaviour. No threadpool,
        # no future, just a direct urlopen + outcome record.
        if len(endpoints) == 1:
            url = endpoints[0]
            t0 = time.monotonic()
            try:
                resp = self._rpc_post(url, body, timeout_seconds=timeout_seconds)
            except BaseException as e:  # noqa: BLE001
                rtt_ms = int((time.monotonic() - t0) * 1000)
                self._health.record(url, success=False, rtt_ms=rtt_ms)
                raise
            rtt_ms = int((time.monotonic() - t0) * 1000)
            self._health.record(url, success=True, rtt_ms=rtt_ms)
            self._current_endpoint = url
            return url, resp

        # Hedged path: fan out to N endpoints, FIRST_COMPLETED wins.
        if self._executor is None:
            # Defensive: ctor builds executor when fan_out>1, but
            # protect against later mutation.
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, self._hedge_fan_out * 4),
                thread_name_prefix="rpc-hedge",
            )

        # Submit per-endpoint, recording start time per future.
        fut_to_endpoint: dict[concurrent.futures.Future, str] = {}
        fut_start: dict[concurrent.futures.Future, float] = {}
        for ep in endpoints:
            t0 = time.monotonic()
            fut = self._executor.submit(
                self._rpc_post, ep, body, timeout_seconds=timeout_seconds,
            )
            fut_to_endpoint[fut] = ep
            fut_start[fut] = t0

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
                # Timeout fired; mark all still-pending as failed.
                for fut in list(pending):
                    ep = fut_to_endpoint[fut]
                    rtt_ms = int((time.monotonic() - fut_start[fut]) * 1000)
                    self._health.record(ep, success=False, rtt_ms=rtt_ms)
                    errors.append((ep, TimeoutError(
                        f"hedged_timeout_after_{rtt_ms}ms"
                    )))
                pending.clear()
                break
            # Find the first successful future among `done`. Record
            # outcomes for ALL `done` futures (Bug #7: previously
            # already-done siblings beyond the winner were dropped on
            # the floor, hiding their failure rate from the health
            # tracker). Failed siblings get their actual RTT; the
            # winner is recorded with actual RTT and returned.
            winner: tuple[str, bytes] | None = None
            for fut in done:
                ep = fut_to_endpoint[fut]
                rtt_ms = int((time.monotonic() - fut_start[fut]) * 1000)
                try:
                    resp = fut.result()
                except BaseException as e:  # noqa: BLE001
                    self._health.record(ep, success=False, rtt_ms=rtt_ms)
                    errors.append((ep, e))
                    continue
                if winner is None:
                    # First successful response wins. Record it and
                    # remember to return after we've handled the rest
                    # of the `done` set + registered callbacks for any
                    # still-pending siblings.
                    self._health.record(ep, success=True, rtt_ms=rtt_ms)
                    self._current_endpoint = ep
                    winner = (ep, resp)
                else:
                    # Already have a winner; record the sibling's
                    # success but don't return its body. Use ACTUAL
                    # RTT (not sentinel) — the wallclock is known.
                    self._health.record(ep, success=True, rtt_ms=rtt_ms)
            if winner is not None:
                # Pending futures are abandoned (urllib doesn't
                # support real cancellation; their bodies are
                # discarded). At fan_out=3 this is ~2 wasted RPC/sec
                # — acceptable per memo §2. Register a callback so
                # their eventual outcome IS recorded in the health
                # tracker (Bug #7: prior code dropped them, which kept
                # slow-but-broken endpoints in unconditional warmup
                # since their fail rate never reached 100 outcomes).
                # The callback runs in the executor's worker thread
                # when the urllib socket returns/errors; the executor
                # is owned by RpcPoller for the process lifetime so
                # there's no shutdown-race.
                self._register_abandoned_callbacks(pending, fut_to_endpoint)
                return winner

        # All endpoints failed.
        raise HedgedAllFailed(errors)

    def _register_abandoned_callbacks(
        self,
        pending: set[concurrent.futures.Future],
        fut_to_endpoint: dict[concurrent.futures.Future, str],
    ) -> None:
        """Register done-callbacks on still-pending hedged futures
        so their eventual outcome reaches the health tracker (Bug #7).

        RTT is recorded with the ``_RTT_SENTINEL_UNKNOWN`` sentinel —
        the actual wallclock from request-start to completion is
        captured here too, but recording it as p50/p99 would skew the
        rolling-window stats with values correlated with "lost the
        hedge race". Sentinel preserves the success/failure signal
        (consecutive_failures + success_rate stay honest) and is
        filtered out of the latency percentile gates.
        """
        for fut in pending:
            ep = fut_to_endpoint[fut]

            def _on_done(f: concurrent.futures.Future, _ep: str = ep) -> None:
                # Runs in the executor's worker thread. Guard against
                # all exception types — a raised exception inside a
                # done-callback is silently swallowed by Future, but
                # we shouldn't even start that fire.
                try:
                    f.result()
                    self._health.record(
                        _ep, success=True, rtt_ms=_RTT_SENTINEL_UNKNOWN,
                    )
                except BaseException:  # noqa: BLE001
                    self._health.record(
                        _ep, success=False, rtt_ms=_RTT_SENTINEL_UNKNOWN,
                    )

            fut.add_done_callback(_on_done)

    def _rpc_call_single(self, method: str, params: list) -> Any:
        """Single JSON-RPC call. Raises on transport error or RPC
        error; returns the ``result`` field on success.

        Hedging is transparent: at ``hedge_fan_out=1`` this matches
        the pre-hedging single-endpoint codepath; at fan_out>1 the
        request is fanned out and the first success wins.
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
        considered failed.

        Hedging is transparent: at ``hedge_fan_out=1`` this matches
        the pre-hedging single-endpoint codepath; at fan_out>1 the
        batch is fanned out and the first endpoint to return a
        well-formed list response wins.
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
