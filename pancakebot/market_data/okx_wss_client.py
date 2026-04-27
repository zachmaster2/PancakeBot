"""OKX WebSocket subscription client for `candle1s` channels.

Live runtime reads 1s klines for all four traded instruments
(BTC-USDT, ETH-USDT, SOL-USDT, BNB-USDT) from per-symbol in-memory ring
buffers populated by a daemon thread WSS subscription. Decision-time read
is sub-ms (in-memory dict + list slice).

BNB-USDT is a FIRST-CLASS instrument: subscribed, bootstrapped, gap-filled,
and gated by ``is_ready()`` identically to BTC/ETH/SOL. The bot bets on
BNB/USD on PancakeSwap Prediction V2; even when the current strategy gate
doesn't read BNB klines for signal computation, the foundation must support
it for upcoming research without re-plumbing.

Architecture
------------

  ┌── Main thread ────────────┐         ┌── Daemon thread ─────────────┐
  │  gate.evaluate(...)       │         │  asyncio loop                │
  │   └─ wss.get_window(BTC)  │ ◄─────► │   ws.recv() → ring buffer    │
  │      (in-memory read)     │  lock   │   gap-detect → REST repair   │
  └───────────────────────────┘         └──────────────────────────────┘

Bootstrap (subscribe-FIRST flow, per Phase 2 spec 2026-04-27):
  1. Subscribe to ``candle1s`` for all instruments.
  2. Wait for the first ``confirm=="1"`` push per symbol (10s watchdog,
     5s warning; 3 reconnect cycles before fatal). Record ``T = open_time_ms``.
  3. Sleep ``_REST_OVERLAP_DELAY_S`` (2s) so ``/history-candles`` catches
     up to T.
  4. REST fetch via ``OkxClient.fetch_kline_window(symbol,
     oldest=next_lock_at_ms - 301_000, newest_inclusive=T)``.
  5. Boundary verify: REST entry at T must equal the WSS push at T
     (open / high / low / close / volume all match). Mismatch →
     ``InvariantError`` (real REST/WSS divergence; fail loud).
     If REST cannot return T (boundary unavailable), retry once with
     ``_REST_RETRY_EXTRA_DELAY_S`` (3s) of extra delay. Second failure →
     ``InvariantError``.
  6. Atomically replace the ring with REST ``[oldest..T-1]``, then drain
     buffered WSS pushes through the steady-state path. A drain that
     reveals another gap recursively triggers gap-fill.

Steady-state (post-bootstrap) push handling (per spec items 12 + 14):
  - ``confirm=="0"`` (mid-bar) pushes are FULLY DISCARDED -- no buffer,
    no ``last_received_ms`` update, no state mutation. Tracking mid-bars
    in ``last_received_ms`` would mask the silent-WSS-death failure mode
    that ``newest_lagging`` is designed to catch.
  - For every ``confirm=="1"`` push at ts T_new (with last == ring.last_candle_ts_ms,
    expected == last + 1000):
      T_new == expected   → append
      T_new >  expected   → GAP. Buffer the push, set
                            ``gap_fill_in_progress`` (which forces
                            ``is_ready()`` False), trigger the same
                            REST-repair flow as bootstrap (steps 3-6).
      T_new == last       → DUPLICATE. Silent if OHLCV matches ring tail;
                            warn-and-discard if differs.
      T_new <  last       → OUT-OF-ORDER late arrival. Silent if matches
                            an existing ring entry or older than oldest;
                            warn-and-discard otherwise.

Decision-time skip vocabulary (returned by ``get_window``):
  ``wss_unknown_symbol``       -- symbol not in this client's instrument set
  ``wss_bootstrap_pending``    -- post-reconnect or initial REST fill not done
  ``wss_gap_fill_in_progress`` -- REST repair in flight; ring being rebuilt
  ``wss_insufficient``         -- ring has fewer than expected_count candles before cutoff
  ``wss_newest_lagging``       -- ring's newest is behind ``cutoff_ms - 1000``;
                                  also flags the ring for daemon reconnect
                                  (silent-WSS-death recovery, item 13)

The legacy ``wss_stale`` reason (last_received_ms wall-clock threshold) was
retired in spec item 10 -- ``wss_newest_lagging`` subsumes it (catches both
"no recent push" AND the post-push-but-still-behind race condition).

Multi-instrument failure handling lives in the gate, not here. The gate
collapses any single-symbol skip to a generic ``risk_kline_wss_failure``
+ a per-symbol ``WSS_GATE`` warn log (spec item 16).
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from pancakebot.log import error, info, warn
from pancakebot.market_data.okx_client import RETRY_WSS
from pancakebot.util import InvariantError

if TYPE_CHECKING:
    from pancakebot.market_data.okx_client import OkxClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OKX_WSS_BUSINESS = "wss://ws.okx.com:8443/ws/v5/business"
_OKX_WSS_BUSINESS_AWS = "wsaws.okx.com:8443/ws/v5/business"  # backup endpoint
_OKX_WSS_ENDPOINTS: tuple[str, ...] = (
    _OKX_WSS_BUSINESS,
    f"wss://{_OKX_WSS_BUSINESS_AWS}",
)

_BACKOFF_STEPS: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0, 120.0)
_BACKOFF_RESET_SECONDS = 60.0  # session lasting >= 60s = healthy, reset streak

_DEFAULT_RING_MAX = 300              # 300s = 5min sliding window
_RECV_LOOP_TIMEOUT_S = 2.0           # short for stop-event + reconnect-signal responsiveness

# Bootstrap / gap-fill timing (Phase 2 spec 2026-04-27).
_FIRST_PUSH_TIMEOUT_S = 10.0         # per-session first-push watchdog
_FIRST_PUSH_WARN_S = 5.0             # log warning at this threshold
_FIRST_PUSH_MAX_RECONNECTS = 3       # cycles before InvariantError
_REST_OVERLAP_DELAY_S = 2.0          # let /history-candles catch up to T
_REST_RETRY_EXTRA_DELAY_S = 3.0      # boundary-unavailable single retry

# Newest-lagging escalation. If a ring's gate-time newest_ts is behind
# the expected ``cutoff_ms - 1000`` for ``_NEWEST_LAGGING_MAX_RECONNECTS``
# *consecutive* reconnect cycles (without any successful read in between),
# the bot fail-louds. Catches silent WSS death where the connection is
# alive but pushes have stopped flowing for the affected symbol.
_NEWEST_LAGGING_MAX_RECONNECTS = 3

# Per-instrument REST window. Anchored to ``next_lock_at`` so the WSS ring
# matches the on-disk rebuild's shape (newest=lock-2000, oldest=lock-301000).
# See research/okx_wss_migration_design.md (Phase 2 spec 2026-04-27).
_HISTORY_OLDEST_OFFSET_MS = 301_000


# ---------------------------------------------------------------------------
# _InstrumentRing -- per-symbol state
# ---------------------------------------------------------------------------

@dataclass
class _InstrumentRing:
    """Per-instrument state. Mutated by daemon thread; read by main thread.

    All field reads/writes occur under the parent client's ``_lock`` -- this
    dataclass is pure data, no methods that take the lock themselves.
    """
    symbol: str
    klines: deque = field(default_factory=lambda: deque(maxlen=_DEFAULT_RING_MAX))
    # ``last_received_ms`` updates ONLY on confirm="1" pushes (per Phase 2
    # spec item 14, 2026-04-27). Mid-bar (confirm="0") updates do NOT touch
    # it; otherwise WSS that emits only mid-bars (no closes) would mask the
    # silent-death failure mode that ``newest_lagging`` is designed to catch.
    last_received_ms: int = 0
    last_candle_ts_ms: int = 0      # newest open_time_ms in ring (or 0 = empty)

    # Bootstrap state. All three reset on each WSS reconnect.
    bootstrap_sub_ack: bool = False
    bootstrap_first_push_done: bool = False  # first confirm="1" push received
    bootstrap_rest_done: bool = False        # REST+verify+drain completed

    # Gap-fill machinery.
    gap_fill_in_progress: bool = False       # True while a REST-repair task is in flight
    gap_buffer: list = field(default_factory=list)  # list of (confirm:str, arr:list)
    first_push_open_ts_ms: int = 0           # T from first push this session

    # Silent-death recovery (Phase 2 spec item 13, 2026-04-27).
    # ``needs_reconnect`` is set by ``get_window`` when ``newest_lagging`` fires
    # and cleared by the daemon at WSS reconnect. ``newest_lagging_streak``
    # counts consecutive reconnect cycles without a successful read; when
    # it reaches ``_NEWEST_LAGGING_MAX_RECONNECTS`` the client fail-louds.
    needs_reconnect: bool = False
    newest_lagging_streak: int = 0


# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------

class _FirstPushTimeout(Exception):
    """Raised by the daemon-thread watchdog when no first push lands within
    ``_FIRST_PUSH_TIMEOUT_S``. Caught by ``_run_loop`` to trigger reconnect
    cycle counting (escalates to ``InvariantError`` after 3 cycles)."""

    def __init__(self, pending: tuple[str, ...]) -> None:
        super().__init__(f"first_push_timeout: pending={list(pending)}")
        self.pending = pending


# ---------------------------------------------------------------------------
# OkxWssClient -- public API
# ---------------------------------------------------------------------------

class OkxWssClient:
    """Subscribe to OKX ``candle1s`` channels and serve windows in-memory.

    Threading model
    ---------------
      - Daemon thread runs an asyncio loop (``_run_loop`` -> ``_ws_listen``).
      - Public methods (``get_window``, ``is_ready``, ``stats``) are
        synchronous and acquire ``_lock`` for ring access.
      - REST fetches inside the asyncio loop run via
        ``loop.run_in_executor`` (network I/O off the asyncio thread).

    The lock guards all ring mutations and reads. The locked sections are
    pure in-memory operations -- no I/O, no logging, no JSON parse/dumps.
    Lock granularity is low enough that decision-time reads are sub-ms.

    Lifetime
    --------
      - ``start()``: spawn daemon, block until is_ready() or fatal.
      - ``stop()``: signal stop_event, close WSS, join thread (<= 10s).
      - Register ``stop`` via ``atexit`` from the engine for graceful exit.
    """

    def __init__(
        self,
        *,
        okx_client: "OkxClient",
        instruments: tuple[str, ...],
        next_lock_at_ms_provider: Callable[[], int],
        ring_max: int = _DEFAULT_RING_MAX,
        rate_acquire_fn: Callable[[], None] | None = None,
    ) -> None:
        if not instruments:
            raise ValueError("instruments must be a non-empty tuple")
        self._client = okx_client
        self._instruments: tuple[str, ...] = tuple(instruments)
        self._ring_max = int(ring_max)
        self._next_lock_at_ms_provider = next_lock_at_ms_provider
        self._rate_acquire_fn = rate_acquire_fn

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._rings: dict[str, _InstrumentRing] = {
            sym: _InstrumentRing(symbol=sym, klines=deque(maxlen=self._ring_max))
            for sym in self._instruments
        }

        self._connected = False
        self._current_endpoint: str | None = None
        self._failure_streak = 0
        self._first_push_failure_streak = 0
        self._last_connected_at: float = 0.0
        # Set by daemon thread on unrecoverable failure (boundary mismatch,
        # boundary unavailable, repeated first-push timeout). Polled by
        # ``start()`` to surface a precise InvariantError.
        self._fatal_error: str | None = None
        # In-flight asyncio tasks spawned by ``_ws_listen`` (initial REST-fill
        # per first-push) AND by ``_rest_fill_to_T`` (nested REST-fill per
        # gap discovered during drain). Lives on the instance so nested-task
        # spawns can register themselves for shutdown cancellation. Reset
        # at the top of each ``_ws_listen`` session. Mutated only on the
        # asyncio loop thread (no cross-thread access -> no lock needed).
        self._async_bootstrap_tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True iff every ring has completed bootstrap AND no gap-fill is
        in flight. Per Phase 2 spec, all four conditions:
          - bootstrap_sub_ack
          - bootstrap_first_push_done    (reset on reconnect)
          - bootstrap_rest_done          (set ONLY after boundary verify)
          - NOT gap_fill_in_progress     (suppresses readiness mid-repair)
        """
        with self._lock:
            return all(
                r.bootstrap_sub_ack and r.bootstrap_first_push_done
                and r.bootstrap_rest_done and not r.gap_fill_in_progress
                for r in self._rings.values()
            )

    def get_window(
        self,
        symbol: str,
        cutoff_ms: int,
        expected_count: int,
    ) -> tuple[list[list] | None, str | None]:
        """Return ``(klines_list, skip_reason)`` for *symbol*.

        Skip reasons (returned with klines=None):
          ``"wss_unknown_symbol"``       -- symbol not in this client's set
          ``"wss_bootstrap_pending"``    -- REST bootstrap not yet completed (post-reconnect or initial)
          ``"wss_gap_fill_in_progress"`` -- REST repair in flight; ring being rebuilt
          ``"wss_insufficient"``         -- ring has fewer than expected_count candles before cutoff
          ``"wss_newest_lagging"``       -- newest candle is behind expected ``cutoff - 1000``;
                                            also signals daemon to reconnect (silent-WSS-death recovery)

        Otherwise returns ``(klines, None)`` where klines is a list of
        ``[ts_ms, o, h, l, c, v]`` arrays oldest-first.

        On any successful read the per-ring ``newest_lagging_streak`` and
        ``needs_reconnect`` flag are reset (recovery confirmed). On
        ``wss_newest_lagging`` the streak increments only when the flag
        was previously False (so multiple gate calls between reconnects
        don't double-count); when it reaches ``_NEWEST_LAGGING_MAX_RECONNECTS``
        the client fail-louds via ``_fatal_error``.

        The locked section contains only in-memory ops (timestamp compare,
        list comprehension over <=300 entries). No I/O, no logging.
        """
        with self._lock:
            ring = self._rings.get(symbol)
            if ring is None:
                return None, "wss_unknown_symbol"
            # Bootstrap_rest_done is reset to False on reconnect (and clears
            # only after boundary verification + drain completes). Gate the
            # window read on it so callers can't accidentally read a ring
            # whose contents are stale-from-a-prior-session.
            if not ring.bootstrap_rest_done:
                return None, "wss_bootstrap_pending"
            if ring.gap_fill_in_progress:
                return None, "wss_gap_fill_in_progress"
            valid = [k for k in ring.klines if int(k[0]) < cutoff_ms]
            if len(valid) < expected_count:
                return None, "wss_insufficient"
            # Newest-lagging check (Phase 2 spec item 9, 2026-04-27). The
            # filtered window's newest must equal ``cutoff_ms - 1000`` --
            # the last fully-closed second before cutoff. If it lags, either
            # the relevant push hasn't arrived yet or the WSS feed has gone
            # silent. Signal the daemon to reconnect (item 13).
            expected_newest_ms = cutoff_ms - 1000
            if valid[-1][0] != expected_newest_ms:
                if not ring.needs_reconnect:
                    ring.needs_reconnect = True
                    ring.newest_lagging_streak += 1
                    if ring.newest_lagging_streak >= _NEWEST_LAGGING_MAX_RECONNECTS:
                        self._fatal_error = (
                            f"okx_wss_newest_lagging_unrecoverable: "
                            f"symbol={symbol} {_NEWEST_LAGGING_MAX_RECONNECTS} "
                            f"consecutive newest_lagging without recovery"
                        )
                        self._stop_event.set()
                return None, "wss_newest_lagging"
            # Successful read -- recovery confirmed; clear streak + flag.
            ring.newest_lagging_streak = 0
            ring.needs_reconnect = False
            return valid[-expected_count:], None

    def stats(self) -> dict[str, dict]:
        """Diagnostic snapshot. Safe to call any time."""
        with self._lock:
            return {
                sym: {
                    "ring_size": len(r.klines),
                    "newest_ts_ms": r.last_candle_ts_ms,
                    "last_received_ms": r.last_received_ms,
                    "sub_ack": r.bootstrap_sub_ack,
                    "first_push_done": r.bootstrap_first_push_done,
                    "rest_done": r.bootstrap_rest_done,
                    "gap_fill_in_progress": r.gap_fill_in_progress,
                    "gap_buffer_size": len(r.gap_buffer),
                    "first_push_open_ts_ms": r.first_push_open_ts_ms,
                    "needs_reconnect": r.needs_reconnect,
                    "newest_lagging_streak": r.newest_lagging_streak,
                }
                for sym, r in self._rings.items()
            }

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def current_endpoint(self) -> str | None:
        return self._current_endpoint

    def fatal_error(self) -> str | None:
        """Return the daemon-thread fatal-error message, or None if healthy.

        Set by the daemon when an unrecoverable failure occurs (boundary
        mismatch, REST boundary unavailable after retry, repeated first-push
        timeout escalation, ``newest_lagging`` 3-cycle escalation). The
        daemon ALSO signals ``_stop_event`` and exits cleanly when this
        fires -- but ``get_window`` will continue returning skip reasons
        from the now-stale ring, and ``is_ready()`` doesn't observe this
        field by design (it describes readiness, not failure mode).

        Engine housekeeping MUST poll this each round and raise
        ``InvariantError`` when set, so the bot fails loud (supervisor
        restart, alert, root-cause investigation) instead of silently
        skipping every round forever (Phase 2 spec item 17 part A,
        2026-04-27).
        """
        return self._fatal_error

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, *, ready_timeout_s: float = 60.0) -> None:
        """Spawn the daemon thread and block until is_ready() or fatal.

        The new flow does subscribe-first (no synchronous REST fill on the
        caller thread). Default timeout raised to 60s to accommodate:
          - up to 10s first-push watchdog × 3 reconnects worst case (30s)
          - 2s REST overlap delay
          - 1-2× retry budget on boundary unavailable

        Raises ``InvariantError`` on bootstrap failure (boundary mismatch,
        boundary unavailable, first-push timeout escalation, plain timeout).
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._fatal_error = None
        self._first_push_failure_streak = 0

        self._thread = threading.Thread(
            target=self._run_loop, name="okx-wss-listener", daemon=True,
        )
        self._thread.start()
        info("OKX_WSS", "INIT", "DAEMON",
             msg=f"WSS daemon thread started (instruments={list(self._instruments)})")

        deadline = time.time() + max(0.0, ready_timeout_s)
        while time.time() < deadline:
            if self._fatal_error is not None:
                raise InvariantError(self._fatal_error)
            if self.is_ready():
                info("OKX_WSS", "INIT", "READY",
                     msg=f"WSS bootstrap complete for {len(self._instruments)} instruments")
                return
            if self._stop_event.is_set():
                raise InvariantError(
                    self._fatal_error or "okx_wss_start_aborted_before_ready"
                )
            time.sleep(0.2)
        raise InvariantError(
            f"okx_wss_bootstrap_timeout: not ready within {ready_timeout_s}s. "
            f"stats={self.stats()}"
        )

    def stop(self, *, join_timeout_s: float = 10.0) -> None:
        """Signal stop, close WSS, join daemon thread. Best-effort."""
        if self._thread is None:
            return
        self._stop_event.set()
        try:
            self._thread.join(timeout=max(0.0, join_timeout_s))
        except Exception:  # noqa: BLE001 -- never block bot shutdown on cleanup
            pass
        s = self.stats()
        info("OKX_WSS", "INIT", "STOP",
             msg=f"WSS daemon stopped, stats={s}")

    # ------------------------------------------------------------------
    # Daemon thread: outer reconnect loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        n = len(_OKX_WSS_ENDPOINTS)
        idx = 0
        while not self._stop_event.is_set():
            if self._fatal_error is not None:
                return
            url = _OKX_WSS_ENDPOINTS[idx]
            self._current_endpoint = url
            session_start = time.time()
            first_push_complete_at_session_start = self._all_first_push_done()
            try:
                asyncio.run(self._ws_listen(url))
            except _FirstPushTimeout as e:
                warn("OKX_WSS", "INIT", "FPUSH_TO",
                     msg=f"First-push timeout on {url}: pending={list(e.pending)}")
            except Exception as e:  # noqa: BLE001 -- log + reconnect
                warn("OKX_WSS", "ERR", "RECONN",
                     msg=f"Endpoint {url}: {type(e).__name__}: {e}")
            self._connected = False

            # First-push streak: count cycles where bootstrap was incomplete
            # at session start AND still incomplete after session.
            if (not first_push_complete_at_session_start
                    and not self._all_first_push_done()):
                self._first_push_failure_streak += 1
                if self._first_push_failure_streak >= _FIRST_PUSH_MAX_RECONNECTS:
                    pending = self._pending_first_push_symbols()
                    self._fatal_error = (
                        f"okx_wss_first_push_repeated_timeout: "
                        f"{_FIRST_PUSH_MAX_RECONNECTS} consecutive bootstrap "
                        f"cycles failed; pending={pending}"
                    )
                    error("OKX_WSS", "INIT", "FATAL", msg=self._fatal_error)
                    return
            else:
                self._first_push_failure_streak = 0

            session_duration = time.time() - session_start
            if session_duration >= _BACKOFF_RESET_SECONDS:
                self._failure_streak = 0
            else:
                self._failure_streak += 1
            idx = (idx + 1) % n
            if self._failure_streak >= n:
                step = min(self._failure_streak - n, len(_BACKOFF_STEPS) - 1)
                delay = _BACKOFF_STEPS[step]
                warn("OKX_WSS", "RETRY", "WAIT",
                     msg=f"All endpoints failed; backoff {delay}s "
                         f"(streak={self._failure_streak})")
                if self._stop_event.wait(timeout=delay):
                    break

    def _all_first_push_done(self) -> bool:
        with self._lock:
            return all(r.bootstrap_first_push_done for r in self._rings.values())

    def _pending_first_push_symbols(self) -> list[str]:
        """Return symbols whose ``bootstrap_first_push_done`` is still False
        right now. Used for diagnostic fatal-error messages so the operator
        sees exactly which feed(s) failed to push."""
        with self._lock:
            return [
                sym for sym, r in self._rings.items()
                if not r.bootstrap_first_push_done
            ]

    def _any_needs_reconnect(self) -> bool:
        """True iff any ring has been flagged for daemon reconnect by
        ``get_window`` (silent-WSS-death recovery, Phase 2 spec item 13).
        Polled in the recv-loop timeout cycle; True -> exit ``_ws_listen``
        so ``_run_loop`` cycles to a fresh connection."""
        with self._lock:
            return any(r.needs_reconnect for r in self._rings.values())

    # ------------------------------------------------------------------
    # Daemon thread: per-session asyncio listener
    # ------------------------------------------------------------------

    async def _ws_listen(self, url: str) -> None:
        import websockets

        async with websockets.connect(
            url, ping_interval=25, ping_timeout=10, open_timeout=15,
        ) as ws:
            # Reset session-scoped per-ring state BEFORE sending subscribe.
            # The recv loop hasn't started yet, so no race.
            with self._lock:
                for r in self._rings.values():
                    r.bootstrap_sub_ack = False
                    r.bootstrap_first_push_done = False
                    r.bootstrap_rest_done = False
                    r.gap_fill_in_progress = False
                    r.first_push_open_ts_ms = 0
                    r.gap_buffer.clear()
                    # Clear newest-lagging reconnect signal so the next
                    # ``newest_lagging`` after this session can re-trigger
                    # (and increment the streak counter for escalation).
                    # Streak itself is NOT reset here -- only a successful
                    # ``get_window`` clears it (= recovery confirmed).
                    r.needs_reconnect = False
                    # last_candle_ts_ms / klines preserved across reconnect:
                    # drain logic uses last_candle_ts_ms to compute gap, and
                    # the ring will be wholesale replaced by the bootstrap
                    # REST fill. Conceptually equivalent to a clean bootstrap
                    # because rest_done is False.

            sub_args = [{"channel": "candle1s", "instId": sym} for sym in self._instruments]
            await ws.send(json.dumps({"op": "subscribe", "args": sub_args}))

            now = time.time()
            self._connected = True
            self._last_connected_at = now
            session_first_push_warn_logged: set[str] = set()
            session_first_push_warn_at = now + _FIRST_PUSH_WARN_S
            session_first_push_deadline = now + _FIRST_PUSH_TIMEOUT_S
            info("OKX_WSS", "SUB", "OK",
                 msg=f"Connected + subscribed on {url}")

            # Fresh task list per session. Both the listener loop (here) and
            # nested gap-fill spawns inside ``_rest_fill_to_T`` register tasks
            # to this list so shutdown cancellation tracks them all.
            self._async_bootstrap_tasks.clear()
            try:
                while not self._stop_event.is_set():
                    if self._fatal_error is not None:
                        return
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=_RECV_LOOP_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError:
                        # Silent-WSS-death recovery (Phase 2 spec item 13).
                        # If any ring's ``get_window`` flagged the daemon for
                        # reconnect, exit the session here so ``_run_loop``
                        # cycles to a fresh connection + bootstrap.
                        if self._any_needs_reconnect():
                            info("OKX_WSS", "RECONN", "NEWEST_LAG",
                                 msg="newest_lagging signal received; "
                                     "reconnecting per item-13 silent-death recovery")
                            return
                        self._first_push_watchdog_check(
                            warn_at=session_first_push_warn_at,
                            deadline=session_first_push_deadline,
                            warned_set=session_first_push_warn_logged,
                        )
                        await self._reap_done_tasks()
                        continue

                    msg = json.loads(raw)
                    if msg.get("event") == "subscribe":
                        self._handle_sub_ack(msg)
                        continue
                    if msg.get("event") == "error":
                        warn("OKX_WSS", "ERR", "SUB_FAIL",
                             msg=f"OKX error on {url}: {msg}")
                        return
                    arg = msg.get("arg") or {}
                    if arg.get("channel") != "candle1s":
                        continue
                    sym = arg.get("instId")
                    data = msg.get("data") or []
                    actions = self._handle_candle_push(sym, data)
                    for action_sym, t_action in actions:
                        self._async_bootstrap_tasks.append(
                            asyncio.create_task(self._rest_fill_to_T(action_sym, t_action))
                        )
                    await self._reap_done_tasks()
            finally:
                # Cancel any in-flight REST-fill tasks (initial AND nested)
                # before tearing down the loop. Snapshot the list to a local
                # so concurrent appends from in-flight cancellations don't
                # mutate the iterator we're walking.
                in_flight = list(self._async_bootstrap_tasks)
                for t in in_flight:
                    if not t.done():
                        t.cancel()
                for t in in_flight:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass  # expected on shutdown cancellation
                    except Exception as e:  # noqa: BLE001
                        # Real bug surfaced on shutdown -- log so it isn't
                        # silently lost (was the prior implementation's gap).
                        warn("OKX_WSS", "BOOT", "SHUTDOWN_ERR",
                             msg=f"task during shutdown: "
                                 f"{type(e).__name__}: {e}")
                self._async_bootstrap_tasks.clear()

    async def _reap_done_tasks(self) -> None:
        """Await completed tasks in ``self._async_bootstrap_tasks`` and
        remove them in place. Mutates the live list (does NOT replace) so
        nested gap-fill tasks appended during the awaits stay tracked.
        """
        # Iterate over a snapshot; remove completed tasks individually after
        # awaiting them, leaving any tasks appended during the await window
        # (e.g. by a nested gap-fill spawn) in the live list.
        snapshot = list(self._async_bootstrap_tasks)
        for t in snapshot:
            if not t.done():
                continue
            try:
                await t  # surface exceptions to the daemon's log
            except Exception as e:  # noqa: BLE001
                warn("OKX_WSS", "BOOT", "ERR",
                     msg=f"REST-fill task error: {type(e).__name__}: {e}")
            try:
                self._async_bootstrap_tasks.remove(t)
            except ValueError:
                pass  # already removed somehow; harmless

    def _first_push_watchdog_check(
        self,
        *,
        warn_at: float,
        deadline: float,
        warned_set: set[str],
    ) -> None:
        """Called from the recv-timeout cycle. Logs a warning at the
        warn-at threshold; raises ``_FirstPushTimeout`` past the deadline.
        The raise propagates out of ``_ws_listen`` and is caught by
        ``_run_loop`` to advance the first-push reconnect counter.
        """
        now = time.time()
        if now < warn_at:
            return
        pending: list[str] = []
        with self._lock:
            for sym, r in self._rings.items():
                if not r.bootstrap_first_push_done:
                    pending.append(sym)
        if not pending:
            return
        if now >= deadline:
            raise _FirstPushTimeout(tuple(pending))
        for sym in pending:
            if sym not in warned_set:
                warned_set.add(sym)
                warn("OKX_WSS", "INIT", "FPUSH_LATE",
                     msg=f"{sym}: no first push within {_FIRST_PUSH_WARN_S}s")

    # ------------------------------------------------------------------
    # Daemon thread: message handlers
    # ------------------------------------------------------------------

    def _handle_sub_ack(self, msg: dict) -> None:
        arg = msg.get("arg") or {}
        sym = arg.get("instId")
        if sym is None:
            return
        with self._lock:
            ring = self._rings.get(sym)
            if ring is not None:
                ring.bootstrap_sub_ack = True
        # Logging OUTSIDE the lock per design (no I/O under lock).
        info("OKX_WSS", "SUB", "ACK", msg=f"subscribed {sym}")

    def _handle_candle_push(
        self, symbol: str | None, data: list,
    ) -> list[tuple[str, int]]:
        """Process one candle1s push. Returns a list of ``(symbol, T)`` tuples
        the listener should spawn REST-fill tasks for (either bootstrap
        first-push or steady-state gap detection).

        Three per-ring states drive the row-handling decision:

          1. ``not bootstrap_first_push_done``: pre-bootstrap. Discard
             confirm="0" entirely; the first ``confirm="1"`` row records T,
             buffers itself, and signals an action.
          2. ``bootstrap_first_push_done`` but ``not bootstrap_rest_done``
             (or ``gap_fill_in_progress``): a REST-repair task is pending.
             Buffer confirm="1" rows; discard confirm="0".
          3. Otherwise: steady-state. Differentiated anomaly handling
             (continuation / real gap / duplicate / out-of-order); see
             ``_apply_steady_state_row``.

        ``confirm="0"`` (mid-bar) pushes are NOT committed to the ring,
        do NOT update ``last_received_ms``, and do NOT trigger gap detection.
        Per Phase 2 spec item 14 (2026-04-27), tracking confirm="0" in
        ``last_received_ms`` would mask the silent-WSS-death failure mode
        where mid-bar updates flow but no closes ever land --
        ``newest_lagging`` (item 9) catches that, and item 13's reconnect
        signal then recovers.

        Locking: a SINGLE ``with self._lock`` wraps the entire row loop so
        a multi-row push is applied atomically -- main-thread ``get_window``
        readers can never observe a partial-state ring (e.g. some rows
        appended but not others). The locked section is pure in-memory work
        (no I/O, no logging, no JSON parse), so contention is sub-ms.
        Anomaly logs queue into a list and emit AFTER the lock releases.
        """
        actions: list[tuple[str, int]] = []
        anomalies: list[tuple[str, str]] = []  # (code, msg) -- logged outside lock
        if symbol is None or not data:
            return actions
        recv_local_ms = int(time.time() * 1000)
        with self._lock:
            ring = self._rings.get(symbol)
            if ring is None:
                return actions
            for row in data:
                if not isinstance(row, list) or len(row) < 9:
                    continue
                try:
                    ts_ms = int(row[0])
                    o = float(row[1])
                    h = float(row[2])
                    l_ = float(row[3])
                    c = float(row[4])
                    v = float(row[5])
                    confirm = str(row[8])
                except (ValueError, IndexError, TypeError):
                    continue

                # Item 14: confirm="0" (mid-bar) is fully ignored everywhere.
                # No ring mutation, no last_received_ms update, no buffering.
                if confirm != "1":
                    continue

                arr = [ts_ms, o, h, l_, c, v]
                # Confirm-1 push: refresh last_received_ms BEFORE the state
                # machine applies the row (so any subsequent code sees a
                # fresh timestamp regardless of which state branch fires).
                ring.last_received_ms = recv_local_ms

                if not ring.bootstrap_first_push_done:
                    # State 1 -- pre-bootstrap: this row IS the first push T.
                    ring.gap_buffer.append((confirm, arr))
                    ring.first_push_open_ts_ms = ts_ms
                    ring.bootstrap_first_push_done = True
                    actions.append((symbol, ts_ms))
                    continue

                if not ring.bootstrap_rest_done or ring.gap_fill_in_progress:
                    # State 2 -- REST-repair pending; buffer for drain.
                    ring.gap_buffer.append((confirm, arr))
                    continue

                # State 3 -- steady state with differentiated anomaly handling.
                self._apply_steady_state_row(ring, arr, actions, anomalies)
        # Logs OUTSIDE the lock per design. Anomalies are rare; emitting
        # them here avoids blocking other threads on log I/O.
        for code, msg in anomalies:
            warn("OKX_WSS", "PUSH", code, msg=msg)
        return actions

    def _apply_steady_state_row(
        self,
        ring: _InstrumentRing,
        arr: list,
        actions: list[tuple[str, int]],
        anomalies: list[tuple[str, str]],
    ) -> None:
        """Apply one confirm=1 row in steady state. Caller MUST hold
        ``self._lock``. Implements Phase 2 spec item 12's differentiated
        anomaly handling (replaces the old binary "match-or-gap-fill").

        Cases (with ``last == ring.last_candle_ts_ms``,
        ``expected == last + 1000``, ``T == arr[0]``):

          T == expected   -> normal continuation; append.
          T > expected    -> REAL GAP; buffer + trigger gap-fill.
          T == last       -> DUPLICATE; silent if values match,
                             warn-and-discard if values differ.
          T < last        -> OUT-OF-ORDER late arrival; if matches a
                             ring entry: silent. If older than oldest in
                             ring: silent. If "should be" in ring but
                             isn't: warn-and-discard.

        Gap-fill suppression: if ``gap_fill_in_progress`` is already True
        (set by an earlier row in the same drain), subsequent rows buffer
        without firing additional actions -- the in-flight REST repair
        will pick them up on the next drain.
        """
        if ring.gap_fill_in_progress:
            ring.gap_buffer.append(("1", arr))
            return
        ts_ms = arr[0]
        # ``last_candle_ts_ms`` is guaranteed nonzero here: caller invariant
        # is that ``bootstrap_rest_done`` -> last_candle_ts_ms == T (set in
        # ``_rest_fill_to_T``), and steady-state logic only runs once
        # rest_done is True.
        last = ring.last_candle_ts_ms
        expected = last + 1000

        if ts_ms == expected:
            # Normal continuation.
            ring.klines.append(arr)
            ring.last_candle_ts_ms = ts_ms
            return

        if ts_ms > expected:
            # Real gap -- missed candles. Trigger gap-fill.
            ring.gap_fill_in_progress = True
            ring.gap_buffer.append(("1", arr))
            actions.append((ring.symbol, ts_ms))
            return

        if ts_ms == last:
            # Duplicate of last appended candle. Compare against ring tail.
            ring_tail = ring.klines[-1] if ring.klines else None
            if ring_tail is not None and _rows_equal(arr, ring_tail):
                # Silent discard -- benign retransmission.
                return
            anomalies.append((
                "DUP_DIFF",
                f"{ring.symbol} duplicate ts={ts_ms} OHLCV differs from ring "
                f"tail: ring={ring_tail} new={arr}",
            ))
            return

        # ts_ms < last -- out-of-order late arrival.
        # Search ring for a matching ts. (deque iteration is O(N=300).)
        existing = None
        for k in ring.klines:
            if k[0] == ts_ms:
                existing = k
                break
        if existing is not None:
            if _rows_equal(arr, existing):
                # Silent discard -- duplicate of an entry already in ring.
                return
            anomalies.append((
                "OOO_DIFF",
                f"{ring.symbol} late ts={ts_ms} OHLCV differs from ring entry: "
                f"ring={existing} new={arr}",
            ))
            return
        # Not in ring. Either older than oldest (silent) or "should be"
        # there but isn't (warn).
        if not ring.klines or ts_ms < ring.klines[0][0]:
            # Older than oldest -- irrelevant; silent discard.
            return
        anomalies.append((
            "OOO_MISSING",
            f"{ring.symbol} late ts={ts_ms} not in ring "
            f"(oldest={ring.klines[0][0]} newest={ring.klines[-1][0]})",
        ))

    # ------------------------------------------------------------------
    # Daemon thread: REST-repair (bootstrap + gap-fill, unified)
    # ------------------------------------------------------------------

    async def _rest_fill_to_T(self, symbol: str, T_ms: int) -> None:
        """Sleep, REST-fetch ``[oldest..T]``, boundary-verify against the
        WSS push at T, atomically replace ring with REST ``[oldest..T-1]``,
        and drain the buffered pushes through the steady-state path.

        Used for BOTH bootstrap (first push after subscribe) and steady-state
        gap-fill. Identical mechanics: REST fills the historical context up
        to T, the WSS push at T is the bridge, WSS pushes from T onward fill
        forward.

        ``oldest_needed`` is captured ONCE at the entry of this flow (Phase 2
        spec item 11, 2026-04-27): if the round transitions during the 2s
        sleep + REST + verify + drain (~3-5s end-to-end), the data we fetch
        is for the OLD round's window. The bot at next decision time sees
        ``newest_lagging``, the silent-death recovery (item 13) reconnects,
        and a fresh bootstrap re-anchors to the new round. Self-correcting
        at the cost of one missed first-round-after-transition.

        On boundary mismatch / unavailable, sets ``self._fatal_error`` and
        ``self._stop_event`` so ``start()`` surfaces a precise error.
        """
        # Snapshot oldest_needed ONCE -- see docstring rationale above.
        next_lock_at_ms = self._next_lock_at_ms_provider()
        oldest_needed_ms = next_lock_at_ms - _HISTORY_OLDEST_OFFSET_MS
        if T_ms < oldest_needed_ms:
            # T older than the rolling window we'd request. Should never
            # happen in normal operation -- T comes from a live WSS push
            # (current-time-ish), next_lock_at_ms is in the future. Fail loud.
            self._fatal_error = (
                f"okx_wss_t_before_oldest: symbol={symbol} T={T_ms} "
                f"oldest_needed={oldest_needed_ms}"
            )
            error("OKX_WSS", "BOOT", "FATAL", msg=self._fatal_error)
            self._stop_event.set()
            return
        await asyncio.sleep(_REST_OVERLAP_DELAY_S)
        arrays = await self._fetch_rest_with_retry(symbol, T_ms, oldest_needed_ms)
        if arrays is None:
            return  # _fetch_rest_with_retry already set fatal state

        rest_T_row = arrays[-1]
        with self._lock:
            ring = self._rings.get(symbol)
            if ring is None:
                return
            # Locate the WSS T row in the buffer.
            wss_T_row: list | None = None
            for confirm_b, arr_b in ring.gap_buffer:
                if confirm_b == "1" and arr_b[0] == T_ms:
                    wss_T_row = arr_b
                    break
            if wss_T_row is None:
                self._fatal_error = (
                    f"okx_rest_wss_t_missing: symbol={symbol} T={T_ms} "
                    f"buffer_size={len(ring.gap_buffer)}"
                )
                error("OKX_WSS", "BOOT", "FATAL", msg=self._fatal_error)
                self._stop_event.set()
                return
            if not _rows_equal(rest_T_row, wss_T_row):
                self._fatal_error = (
                    f"okx_rest_wss_boundary_mismatch: symbol={symbol} T={T_ms} "
                    f"rest={rest_T_row} wss={wss_T_row}"
                )
                error("OKX_WSS", "BOOT", "FATAL", msg=self._fatal_error)
                self._stop_event.set()
                return
            # Atomic replace: ring = REST[oldest..T-1] + WSS[T].
            # We boundary-verified REST[T] == WSS[T]; we keep the WSS row for
            # the T slot so the "ring head was authored by a WSS push" invariant
            # holds uniformly for every entry past oldest. This also eliminates
            # the would-be empty-ring edge case when REST returns only [T]
            # (oldest_needed == T at start-of-round).
            ring.klines.clear()
            for arr in arrays[:-1]:
                ring.klines.append(arr)
            ring.klines.append(list(wss_T_row))
            ring.last_candle_ts_ms = T_ms  # always T after bootstrap; never 0
            # Mark bootstrap done BEFORE drain so steady-state logic applies
            # to drained rows (correctly detects further gaps).
            ring.bootstrap_rest_done = True
            ring.gap_fill_in_progress = False
            buffered = list(ring.gap_buffer)
            ring.gap_buffer.clear()
            # Drain via steady-state path. Skip rows at ts <= T -- the
            # authoritative T row is already in the ring (appended above), and
            # any earlier rows are superseded by it. Rows past T flow through
            # normal gap detection (and may discover further gaps).
            new_actions: list[tuple[str, int]] = []
            drain_anomalies: list[tuple[str, str]] = []
            for _confirm_b, arr_b in buffered:
                if arr_b[0] <= T_ms:
                    continue
                # Note: gap_buffer only contains confirm="1" entries
                # (see ``_handle_candle_push`` -- confirm="0" is discarded
                # at the row-loop top per Phase 2 spec item 14).
                self._apply_steady_state_row(ring, arr_b, new_actions, drain_anomalies)
        # Logs OUTSIDE the lock per design.
        for code, msg in drain_anomalies:
            warn("OKX_WSS", "PUSH", code, msg=msg)
        info("OKX_WSS", "BOOT", "OK",
             msg=f"REST-fill {symbol} done: T={T_ms} ring_size={len(arrays)} "
                 f"drained={len(buffered)} nested_actions={len(new_actions)}")
        # Spawn nested gap-fill if the drain revealed further gaps. Rare;
        # would require WSS to gap during the gap-fill window itself.
        # Tasks register on the shared instance list so the listener's
        # shutdown finally cancels them too.
        for next_sym, next_T in new_actions:
            self._async_bootstrap_tasks.append(
                asyncio.create_task(self._rest_fill_to_T(next_sym, next_T))
            )

    async def _fetch_rest_with_retry(
        self, symbol: str, T_ms: int, oldest_ms: int,
    ) -> list[list] | None:
        """Two-attempt REST fetch wrapping ``OkxClient.fetch_kline_window``.
        On first ``InvariantError`` (boundary unavailable / network exhaust),
        sleep ``_REST_RETRY_EXTRA_DELAY_S`` and retry once. Returns None and
        sets ``_fatal_error`` on second failure.

        ``oldest_ms`` is provided by the caller (snapshotted at flow start
        per Phase 2 spec item 11) so retries inside this function reuse the
        SAME window even if ``next_lock_at_ms_provider`` would now return a
        different value.
        """
        # ``get_running_loop`` instead of the deprecated ``get_event_loop``
        # (the latter is removed in Python 3.14; we're already inside an
        # async context so the running-loop variant is the correct call).
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, self._fetch_kline_window_sync, symbol, oldest_ms, T_ms,
            )
        except InvariantError as e1:
            info("OKX_WSS", "BOOT", "RETRY",
                 msg=f"{symbol} REST fetch failed: {e1}; "
                     f"retry +{_REST_RETRY_EXTRA_DELAY_S}s")
            await asyncio.sleep(_REST_RETRY_EXTRA_DELAY_S)
            try:
                return await loop.run_in_executor(
                    None, self._fetch_kline_window_sync, symbol, oldest_ms, T_ms,
                )
            except InvariantError as e2:
                self._fatal_error = (
                    f"okx_rest_boundary_unavailable: symbol={symbol} T={T_ms} "
                    f"err={e2}"
                )
                error("OKX_WSS", "BOOT", "FATAL", msg=self._fatal_error)
                self._stop_event.set()
                return None

    def _fetch_kline_window_sync(
        self, symbol: str, oldest_ms: int, T_ms: int,
    ) -> list[list]:
        """Sync wrapper for ``run_in_executor``. Uses ``RETRY_WSS`` (bounded
        retry, fail-loud) and the optional shared rate budget."""
        return self._client.fetch_kline_window(
            symbol=symbol,
            oldest_open_ms=oldest_ms,
            newest_open_ms_inclusive=T_ms,
            retry_policy=RETRY_WSS,
            rate_acquire_fn=self._rate_acquire_fn,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OHLCV_TOL_REL = 1e-12
_OHLCV_TOL_ABS = 1e-12


def _rows_equal(a: list, b: list) -> bool:
    """Compare ``[ts, o, h, l, c, v]`` rows with strict equality on the
    integer timestamp and ``math.isclose`` on the 5 OHLCV floats.

    REST and WSS report finalised candles for the same timestamp from the
    same OKX source, so the OHLCV floats SHOULD be byte-identical -- but
    OKX could in principle serialise the same price slightly differently
    across endpoints (trailing zero, scientific notation, sub-ULP wobble)
    and we don't want to fail-loud on representation noise. The tolerance
    (``rel_tol=abs_tol=1e-12``) is tight enough that anything sub-cent at
    OKX price scales is caught and anything bit-level is ignored. Real
    divergence (different prices) is way outside this band.
    """
    if len(a) < 6 or len(b) < 6:
        return False
    if a[0] != b[0]:  # timestamp: ints, exact match required
        return False
    for i in range(1, 6):
        if not math.isclose(a[i], b[i], rel_tol=_OHLCV_TOL_REL, abs_tol=_OHLCV_TOL_ABS):
            return False
    return True
