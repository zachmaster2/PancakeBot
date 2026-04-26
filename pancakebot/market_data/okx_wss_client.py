"""OKX WebSocket subscription client for `candle1s` channels.

Replaces per-round REST kline fetches in live runtime mode. Subscribes
once at startup, holds a sliding window per instrument in memory,
exposes synchronous accessors for the strategy gate to read at decision
time. Sub-ms decision-time read; no fetch latency.

Architecture mirrors ``pancakebot.chain.pool_watcher.WssPoolWatcher``:

  ┌── Main thread ────────────┐         ┌── Daemon thread ─────────────┐
  │  gate.evaluate(...)       │         │  asyncio loop                │
  │   └─ wss.get_window(BTC)  │ ◄─────► │   ws.recv() → ring_buffer    │
  │      (in-memory read)     │  lock   │   reconnect on drop          │
  │      sub-ms latency       │         │   refill from REST on gap    │
  └───────────────────────────┘         └──────────────────────────────┘

Bootstrap (per-instrument, all 4 must satisfy before is_ready() returns True):
  1. REST `/history-candles` initial fill (>= 31 candles)
  2. WSS subscribe-ack received for this instrument
  3. >= 1 push observed with open_time > newest REST ts AND confirm == "1"

Stale-data refusal:
  - Each ring tracks ``last_received_ms`` (LOCAL clock at recv() return).
  - get_window() returns None+"wss_stale" when LOCAL_now - last_received_ms
    exceeds the threshold (default 5000ms; configurable).

Multi-instrument independence (BTC/ETH/SOL/BNB rings):
  - BTC stale -> caller skips with risk_kline_wss_stale.
  - ETH+SOL both stale within ~5s -> caller skips with
    risk_kline_wss_correlated_stale.
  - One of ETH/SOL alone stale -> degraded BTC-primary mode.
  - BNB stale -> log only (capture-only, no decision impact).

Per design doc: research/okx_wss_migration_design.md (commit 4c71784/3d87d40).
Empirical endpoint+push verification: research/okx_wss_endpoint_probe.py.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pancakebot.log import info, warn
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
_DEFAULT_BOOTSTRAP_REST_LIMIT = 100  # candles to fetch via REST at startup
_MIN_CANDLES_FOR_READY = 31          # gate requires 31 candles for signal
_DEFAULT_STALE_THRESHOLD_MS = 5000   # 5s -- 5 missed pushes at 1s cadence
_GAP_FILL_THRESHOLD = 5              # gap > 5 candles -> REST gap-fill
_GAP_REBOOT_THRESHOLD = 100          # gap > 100 candles -> full re-bootstrap

_RECV_LOOP_TIMEOUT_S = 2.0           # short for stop-event responsiveness


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
    last_received_ms: int = 0       # LOCAL clock at most-recent recv() return
    last_candle_ts_ms: int = 0      # newest open_time_ms in ring (or 0 = empty)
    pending_open_time_ms: int = 0   # candle currently being received (mid-bar)
    pending_row: list | None = None  # the in-progress candle's [ts, o, h, l, c, v] form
    bootstrap_rest_done: bool = False
    bootstrap_sub_ack: bool = False
    bootstrap_first_push_done: bool = False  # >=1 confirmed push past newest REST ts


# ---------------------------------------------------------------------------
# OkxWssClient -- public API
# ---------------------------------------------------------------------------

class OkxWssClient:
    """Subscribe to OKX `candle1s` channels and serve windows in-memory.

    Threading model:
      - Daemon thread runs an asyncio loop (``_run_loop`` -> ``_ws_listen``).
      - Public methods (``get_window``, ``is_ready``, ``stats``) are
        synchronous and acquire ``_lock`` for ring access.
      - Bootstrap REST fill blocks the daemon thread (off the critical
        path; happens once at startup).

    The lock guards all ring mutations and reads. The locked sections are
    pure in-memory operations -- no I/O, no logging, no JSON parse/dumps.
    Lock granularity is low enough that decision-time reads are sub-ms.

    Lifetime:
      - ``start()``: spawn daemon, run REST bootstrap, wait until is_ready()
        returns True (or raise on bootstrap failure).
      - ``stop()``: signal stop_event, close WSS, join thread (<= 10s).
      - Register ``stop`` via ``atexit`` from the engine for graceful exit.
    """

    def __init__(
        self,
        *,
        okx_client: "OkxClient",
        instruments: tuple[str, ...],
        ring_max: int = _DEFAULT_RING_MAX,
        bootstrap_rest_limit: int = _DEFAULT_BOOTSTRAP_REST_LIMIT,
        stale_threshold_ms: int = _DEFAULT_STALE_THRESHOLD_MS,
    ) -> None:
        if not instruments:
            raise ValueError("instruments must be a non-empty tuple")
        self._client = okx_client
        self._instruments: tuple[str, ...] = tuple(instruments)
        self._ring_max = int(ring_max)
        self._bootstrap_rest_limit = int(bootstrap_rest_limit)
        self._stale_threshold_ms = int(stale_threshold_ms)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._rings: dict[str, _InstrumentRing] = {
            sym: _InstrumentRing(symbol=sym, klines=deque(maxlen=self._ring_max))
            for sym in self._instruments
        }

        # Initial-bootstrap completion signal so start() can block on
        # full readiness before returning.
        self._initial_ready_event = threading.Event()
        self._connected = False
        self._current_endpoint: str | None = None
        self._failure_streak = 0
        self._last_connected_at: float = 0.0

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True iff all rings have completed the 3-step bootstrap."""
        with self._lock:
            return all(
                r.bootstrap_rest_done and r.bootstrap_sub_ack and r.bootstrap_first_push_done
                for r in self._rings.values()
            )

    def get_window(
        self,
        symbol: str,
        cutoff_ms: int,
        expected_count: int = _MIN_CANDLES_FOR_READY,
        stale_threshold_ms: int | None = None,
    ) -> tuple[list[list] | None, str | None]:
        """Return ``(klines_list, skip_reason)`` for *symbol*.

        Returns ``(None, "wss_stale")`` if the ring hasn't received a push
        within the threshold; ``(None, "wss_insufficient")`` if the ring
        doesn't have enough candles before *cutoff_ms*; otherwise
        ``(klines, None)`` where klines is a list of
        ``[ts_ms, o, h, l, c, v]`` arrays, oldest-first.

        IMPORTANT: ``last_received_ms`` is LOCAL-clock; threshold compares
        LOCAL-vs-LOCAL (skew cancels naturally). Do NOT skew-correct here.

        The locked section contains only in-memory ops (timestamp compare,
        list comprehension over <=300 entries). No I/O, no logging.
        """
        threshold = int(stale_threshold_ms if stale_threshold_ms is not None
                        else self._stale_threshold_ms)
        with self._lock:
            ring = self._rings.get(symbol)
            if ring is None:
                return None, "wss_unknown_symbol"
            now_ms_local = int(time.time() * 1000)
            if now_ms_local - ring.last_received_ms > threshold:
                return None, "wss_stale"
            # ring.klines is a deque; iteration is O(N=ring_max=300).
            valid = [k for k in ring.klines if int(k[0]) < cutoff_ms]
            if len(valid) < expected_count:
                return None, "wss_insufficient"
            return valid[-expected_count:], None

    def stats(self) -> dict[str, dict]:
        """Diagnostic snapshot. Safe to call any time."""
        with self._lock:
            return {
                sym: {
                    "ring_size": len(r.klines),
                    "newest_ts_ms": r.last_candle_ts_ms,
                    "last_received_ms": r.last_received_ms,
                    "rest_done": r.bootstrap_rest_done,
                    "sub_ack": r.bootstrap_sub_ack,
                    "first_push_done": r.bootstrap_first_push_done,
                }
                for sym, r in self._rings.items()
            }

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def current_endpoint(self) -> str | None:
        return self._current_endpoint

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, *, ready_timeout_s: float = 30.0) -> None:
        """Spawn daemon thread + run REST bootstrap + wait until is_ready().

        Blocks until the bootstrap completes for all instruments, or until
        the timeout fires (in which case raises InvariantError -- the bot
        cannot run without a fully-bootstrapped WSS client).
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._initial_ready_event.clear()

        # Phase 1: REST bootstrap fill, synchronous, blocks startup.
        # If this fails for any instrument, raise -- fail fast.
        self._bootstrap_rest_fill()

        # Phase 2: spawn daemon thread for asyncio WSS loop.
        self._thread = threading.Thread(
            target=self._run_loop, name="okx-wss-listener", daemon=True,
        )
        self._thread.start()
        info("OKX_WSS", "INIT", "DAEMON",
             msg=f"WSS daemon thread started (instruments={list(self._instruments)})")

        # Phase 3: wait for is_ready() (sub-ack + first-confirmed-push for all).
        deadline = time.time() + max(0.0, ready_timeout_s)
        while time.time() < deadline:
            if self.is_ready():
                info("OKX_WSS", "INIT", "READY",
                     msg=f"WSS bootstrap complete for {len(self._instruments)} instruments")
                self._initial_ready_event.set()
                return
            if self._stop_event.is_set():
                raise InvariantError("okx_wss_start_aborted_before_ready")
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
        # The daemon's `_ws_listen` polls _stop_event every recv-timeout
        # cycle; should exit within ~3s typically.
        try:
            self._thread.join(timeout=max(0.0, join_timeout_s))
        except Exception:  # noqa: BLE001 -- never block bot shutdown on cleanup
            pass
        s = self.stats()
        info("OKX_WSS", "INIT", "STOP",
             msg=f"WSS daemon stopped, stats={s}")

    # ------------------------------------------------------------------
    # Bootstrap REST fill
    # ------------------------------------------------------------------

    def _bootstrap_rest_fill(self) -> None:
        """For each instrument, fetch initial candles via REST and seed ring.

        Synchronous; blocks startup. Each instrument's
        ``bootstrap_rest_done`` flag set after success. On failure (any
        instrument), raises InvariantError -- bot cannot run with a partially
        bootstrapped WSS client.
        """
        for sym in self._instruments:
            try:
                # OkxClient.fetch_1s_klines() fetches via /candles (live); for
                # bootstrap we want /history-candles. Use the underlying
                # rate-limited fetch path.
                klines_dicts = self._fetch_history_via_okx_client(sym, self._bootstrap_rest_limit)
            except Exception as e:  # noqa: BLE001 -- surface clearly
                raise InvariantError(
                    f"okx_wss_bootstrap_rest_failed: symbol={sym} "
                    f"err={type(e).__name__}: {e}"
                ) from e
            # Convert dicts to [ts, o, h, l, c, v] arrays.
            arrays = [
                [int(k["open_time_ms"]),
                 float(k.get("open", k["close_price"])),
                 float(k.get("high", k["close_price"])),
                 float(k.get("low", k["close_price"])),
                 float(k["close_price"]),
                 float(k.get("volume", 0.0))]
                for k in klines_dicts
            ]
            # Sort oldest-first (defensive; OkxClient already does this).
            arrays.sort(key=lambda a: a[0])
            with self._lock:
                ring = self._rings[sym]
                ring.klines.clear()
                for arr in arrays[-self._ring_max:]:
                    ring.klines.append(arr)
                ring.last_candle_ts_ms = arrays[-1][0] if arrays else 0
                ring.bootstrap_rest_done = True
            info("OKX_WSS", "INIT", "REST",
                 msg=f"REST bootstrap {sym}: {len(arrays)} candles, newest_ts={arrays[-1][0] if arrays else 0}")

    def _fetch_history_via_okx_client(self, symbol: str, limit: int) -> list[dict]:
        """Wrapper around OkxClient -- uses the live fetch path which OKX
        accepts for recent (<=200 candle) windows. Sufficient for our 100-
        candle bootstrap.

        For longer windows we'd switch to fetch_raw('history-candles'),
        but that's not needed at boot.
        """
        return self._client.fetch_1s_klines(symbol=symbol, count=limit)

    # ------------------------------------------------------------------
    # Daemon thread: WSS listener loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        n = len(_OKX_WSS_ENDPOINTS)
        idx = 0
        while not self._stop_event.is_set():
            url = _OKX_WSS_ENDPOINTS[idx]
            self._current_endpoint = url
            session_start = time.time()
            try:
                asyncio.run(self._ws_listen(url))
            except Exception as e:  # noqa: BLE001 -- log + reconnect
                warn("OKX_WSS", "ERR", "RECONN",
                     msg=f"Endpoint {url}: {type(e).__name__}: {e}")
            self._connected = False
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
                     msg=f"All endpoints failed; backoff {delay}s (streak={self._failure_streak})")
                if self._stop_event.wait(timeout=delay):
                    break

    async def _ws_listen(self, url: str) -> None:
        import websockets

        async with websockets.connect(
            url, ping_interval=25, ping_timeout=10, open_timeout=15,
        ) as ws:
            # Subscribe to candle1s for each instrument
            sub_args = [{"channel": "candle1s", "instId": sym} for sym in self._instruments]
            await ws.send(json.dumps({"op": "subscribe", "args": sub_args}))

            # Reset session-scoped sub-ack flags
            with self._lock:
                for r in self._rings.values():
                    r.bootstrap_sub_ack = False

            now = time.time()
            self._connected = True
            self._last_connected_at = now
            session_start_at = now
            session_pushes = 0
            info("OKX_WSS", "SUB", "OK",
                 msg=f"Connected + subscribed on {url}")

            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=_RECV_LOOP_TIMEOUT_S)
                except asyncio.TimeoutError:
                    continue
                msg = json.loads(raw)
                # Subscribe ack
                if msg.get("event") == "subscribe":
                    self._handle_sub_ack(msg)
                    continue
                if msg.get("event") == "error":
                    warn("OKX_WSS", "ERR", "SUB_FAIL",
                         msg=f"OKX error on {url}: {msg}")
                    return
                # Push data
                arg = msg.get("arg") or {}
                if arg.get("channel") != "candle1s":
                    continue
                sym = arg.get("instId")
                data = msg.get("data") or []
                self._handle_candle_push(sym, data)
                session_pushes += 1

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

    def _handle_candle_push(self, symbol: str | None, data: list) -> None:
        """Process one candle1s push. Each ``data`` is a list of rows; each
        row is ``[ts_ms, o, h, l, c, vol, volCcy, volCcyQuote, confirm]``.

        Commit policy: append to ring only when confirm == "1" (closed
        candle). Mid-bar updates (confirm == "0") update the pending slot.
        """
        if symbol is None or not data:
            return
        recv_local_ms = int(time.time() * 1000)
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
            except (ValueError, IndexError):
                continue
            arr = [ts_ms, o, h, l_, c, v]
            with self._lock:
                ring = self._rings.get(symbol)
                if ring is None:
                    continue
                ring.last_received_ms = recv_local_ms
                if confirm == "1":
                    # Closed candle. Append to ring iff strictly newer than
                    # current last_candle_ts_ms (handles REST/WSS overlap).
                    if ts_ms > ring.last_candle_ts_ms:
                        ring.klines.append(arr)
                        ring.last_candle_ts_ms = ts_ms
                        # If this is the first WSS-confirmed push past the
                        # REST tail, mark bootstrap-first-push-done.
                        if ring.bootstrap_rest_done and not ring.bootstrap_first_push_done:
                            ring.bootstrap_first_push_done = True
                    # Clear pending slot regardless.
                    ring.pending_open_time_ms = 0
                    ring.pending_row = None
                else:
                    # Mid-bar update; track pending but don't commit.
                    ring.pending_open_time_ms = ts_ms
                    ring.pending_row = arr


# ---------------------------------------------------------------------------
# Module-level singleton convenience (engine wires this once)
# ---------------------------------------------------------------------------

_singleton: OkxWssClient | None = None


def init_singleton(client: OkxWssClient) -> None:
    """Engine startup: stash the singleton so other modules can find it."""
    global _singleton
    _singleton = client


def get_singleton() -> OkxWssClient | None:
    return _singleton
