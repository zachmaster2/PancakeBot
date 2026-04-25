"""Per-round kline + signal capture for live-vs-history divergence analysis.

Observability infrastructure -- fully decoupled from the bot's decision
path via a bounded queue + daemon worker thread.

Architecture
------------

  Decision path (engine.py)              Capture subsystem
  -----------------------------          --------------------------------
  After signal computed:                 ``CaptureWorker`` daemon thread
    snapshot = {epoch, klines, ...}       loops on ``queue.get(timeout=...)``
    worker.enqueue(snapshot)               -> serialise to JSON
       (sub-ms put_nowait)                 -> append to JSONL file
                                           -> swallow all exceptions

The producer never blocks. The queue is bounded (100 entries) so a stuck
worker can't memory-leak; on full, the new snapshot is dropped with a
``warn`` log. The worker thread is daemon=True so it dies with the bot
without joining.

On graceful shutdown (atexit) the worker is asked to drain remaining
items with a short timeout. If draining doesn't finish in time, the
remaining items are dropped -- bot exit isn't blocked by capture.
The shutdown info-log reports ``queue_remaining`` so the drop count
is observable, not silent.

Schema: append-only JSONL, one line per round-decision.
``schema_version`` field gates forward compatibility -- readers must
check it before parsing anything else. Bump ``CAPTURE_SCHEMA_VERSION``
only when changing the SEMANTICS of an existing field; adding new
optional fields is forward-compatible and does NOT require a bump
(old readers ignore unknown keys).

Storage layout (gitignored):
    var/dry/captured_klines.jsonl   (paths.DRY_CAPTURE_PATH)
    var/live/captured_klines.jsonl  (paths.LIVE_CAPTURE_PATH)

Backtest replay reads these files; see the ``--kline-source captured``
flag on the backtest harness.
"""
from __future__ import annotations

import atexit
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pancakebot.log import warn, info

CAPTURE_SCHEMA_VERSION = 1

# Queue capacity. 100 rounds = ~8 hours of bot time at 1 round per 5min.
# If the worker ever falls behind by 100 rounds, something is very wrong
# and dropping new captures (with a warn) is correct -- we'd rather
# observe a gap than memory-leak waiting for a stuck disk.
_QUEUE_CAPACITY = 100

# Worker poll timeout. Controls latency between enqueue and disk landing
# under low load. 0.25s is a fine balance: fast enough that captures land
# in tens of ms typically, slow enough that the daemon thread doesn't
# burn CPU when idle.
_WORKER_POLL_TIMEOUT_S = 0.25

# Drain timeout on graceful shutdown. Captures still in the queue at
# bot exit get this much time to land before we give up. Kept short
# so atexit doesn't visibly delay the bot exit.
_SHUTDOWN_DRAIN_TIMEOUT_S = 5.0

# Hard ceiling on per-write wall time (worker side, not producer). If
# any individual append takes longer than this, log a warn -- it
# indicates disk contention worth investigating. Capture remains
# best-effort; this is purely diagnostic.
_SLOW_WRITE_WARN_MS = 100


# ---------------------------------------------------------------------------
# Snapshot construction (called inline from the producer)
# ---------------------------------------------------------------------------

def _kline_dict_to_array(k: dict) -> list:
    """Convert OKX-fetch dict {open_time_ms, open, high, low, close_price, volume}
    to ``[ts_ms, o, h, l, c, v]`` array form (matches kline_store schema).
    Missing OHLCV fields fall back to close so older closes-only fetches
    still serialise cleanly.
    """
    ts = int(k.get("open_time_ms", 0))
    c = float(k.get("close_price", 0.0))
    o = float(k.get("open", c))
    h = float(k.get("high", c))
    lo = float(k.get("low", c))
    v = float(k.get("volume", 0.0))
    return [ts, o, h, lo, c, v]


def build_snapshot(
    *,
    epoch: int,
    lock_at_unix: int,
    cutoff_ms: int,
    mode: str,
    btc_klines_raw: list[dict] | None,
    eth_klines_raw: list[dict] | None,
    sol_klines_raw: list[dict] | None,
    returns: dict[str, float | None] | None,
    decision: str,
    skip_reason: str | None,
    selected_strategy: str | None,
    bet_side: str | None,
    bet_size_bnb: float | None,
    pool_bull_bnb: float,
    pool_bear_bnb: float,
) -> dict[str, Any]:
    """Pure data shaping. No I/O. No exceptions on missing fields.

    Called from the producer side (engine.py decision path), so this
    must be cheap. The only non-trivial work is array-flattening the
    klines, which is ~30-90 small list comprehensions and adds <1ms.
    """
    btc_arr = [_kline_dict_to_array(k) for k in (btc_klines_raw or [])]
    eth_arr = (
        [_kline_dict_to_array(k) for k in eth_klines_raw]
        if eth_klines_raw is not None else None
    )
    sol_arr = (
        [_kline_dict_to_array(k) for k in sol_klines_raw]
        if sol_klines_raw is not None else None
    )
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"
    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "epoch": int(epoch),
        "decision_time_utc": now_iso,
        "lock_at_unix": int(lock_at_unix),
        "cutoff_ms": int(cutoff_ms),
        "mode": str(mode),
        "klines_btc": btc_arr,
        "klines_eth": eth_arr,
        "klines_sol": sol_arr,
        "returns": dict(returns or {}),
        "decision": str(decision),
        "skip_reason": skip_reason,
        "selected_strategy": selected_strategy,
        "bet_side": bet_side,
        "bet_size_bnb": (float(bet_size_bnb) if bet_size_bnb is not None else None),
        "pool_total_bnb": float(pool_bull_bnb + pool_bear_bnb),
        "pool_bull_bnb": float(pool_bull_bnb),
        "pool_bear_bnb": float(pool_bear_bnb),
    }


# ---------------------------------------------------------------------------
# CaptureWorker -- bounded queue + daemon thread
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _CaptureStats:
    enqueued: int = 0
    dropped_full: int = 0
    written: int = 0
    write_failures: int = 0
    build_failures: int = 0


class CaptureWorker:
    """Background JSONL writer for round-decision snapshots.

    Producer-safe: ``enqueue()`` is non-blocking; failures are swallowed
    (logged, never raised). Worker thread is daemon, so bot shutdown
    doesn't depend on it. ``shutdown()`` is best-effort with a timeout.
    """

    def __init__(self, path: Path, *, capacity: int = _QUEUE_CAPACITY) -> None:
        self._path = Path(path)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=capacity)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats = _CaptureStats()
        # Lock guards stats only; queue itself is thread-safe.
        self._stats_lock = threading.Lock()

    # -- Producer API ---------------------------------------------------

    def enqueue(self, snapshot: dict[str, Any]) -> bool:
        """Non-blocking put. Returns True iff snapshot was queued.

        On Full: log a warn, drop the snapshot, return False. The bot
        path must never block on capture, so we never wait.
        """
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            with self._stats_lock:
                self._stats.dropped_full += 1
            warn(
                "CAPTURE",
                "QUEUE",
                "FULL",
                msg=f"capture queue full -- dropping snapshot epoch={snapshot.get('epoch', '?')}",
                queue_size=self._queue.qsize(),
                dropped_total=self._stats.dropped_full,
            )
            return False
        with self._stats_lock:
            self._stats.enqueued += 1
        return True

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "enqueued": self._stats.enqueued,
                "dropped_full": self._stats.dropped_full,
                "written": self._stats.written,
                "write_failures": self._stats.write_failures,
                "build_failures": self._stats.build_failures,
                "queue_size": self._queue.qsize(),
            }

    # -- Lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Spawn the daemon worker thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        # Ensure target dir exists once at startup so the worker doesn't
        # need to do it per-write.
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            warn("CAPTURE", "INIT", "MKDIR_FAIL",
                 msg=f"capture dir mkdir failed: {type(e).__name__}: {e}",
                 path=str(self._path.parent))
        self._thread = threading.Thread(
            target=self._run,
            name="kline-capture-writer",
            daemon=True,
        )
        self._thread.start()
        info("CAPTURE", "INIT", "START",
             msg=f"capture worker started -> {self._path.name}",
             capacity=_QUEUE_CAPACITY)

    def shutdown(self, timeout_s: float = _SHUTDOWN_DRAIN_TIMEOUT_S) -> None:
        """Signal stop, drain remaining queue items up to *timeout_s*.

        Best-effort. If draining doesn't finish, remaining items are
        dropped silently -- bot exit is never blocked by capture.
        """
        if self._thread is None:
            return
        self._stop_event.set()
        # Inject a sentinel so the worker wakes from queue.get() promptly.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass  # fine -- worker is busy and will see stop_event soon enough
        deadline = time.monotonic() + max(0.0, timeout_s)
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        s = self.stats()
        info("CAPTURE", "INIT", "STOP",
             msg=f"capture worker stopped",
             enqueued=s["enqueued"],
             written=s["written"],
             dropped_full=s["dropped_full"],
             write_failures=s["write_failures"],
             build_failures=s["build_failures"],
             queue_remaining=s["queue_size"])

    # -- Worker loop ----------------------------------------------------

    def _run(self) -> None:
        """Daemon thread body. Drains the queue, swallows all exceptions."""
        path = self._path
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                snapshot = self._queue.get(timeout=_WORKER_POLL_TIMEOUT_S)
            except queue.Empty:
                continue
            if snapshot is None:
                # Sentinel from shutdown(). Continue to drain remaining items.
                continue
            self._write_one(path, snapshot)

    def _write_one(self, path: Path, snapshot: dict[str, Any]) -> None:
        try:
            line = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            with self._stats_lock:
                self._stats.build_failures += 1
            warn("CAPTURE", "WRITE", "SERFAIL",
                 msg=f"capture serialise failed: {type(e).__name__}: {e}",
                 epoch=snapshot.get("epoch", -1))
            return
        t0 = time.monotonic()
        try:
            with path.open("a", encoding="utf-8", newline="") as f:
                f.write(line + "\n")
        except Exception as e:  # noqa: BLE001 -- never propagate from worker
            with self._stats_lock:
                self._stats.write_failures += 1
            warn("CAPTURE", "WRITE", "FAIL",
                 msg=f"capture write failed: {type(e).__name__}: {e}",
                 path=str(path),
                 epoch=snapshot.get("epoch", -1))
            return
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        with self._stats_lock:
            self._stats.written += 1
        if elapsed_ms > _SLOW_WRITE_WARN_MS:
            warn("CAPTURE", "WRITE", "SLOW",
                 msg=f"capture write took {elapsed_ms:.0f}ms",
                 path=str(path),
                 epoch=snapshot.get("epoch", -1))


# ---------------------------------------------------------------------------
# Module-level singleton API (engine wires this once at startup)
# ---------------------------------------------------------------------------

_WORKER: CaptureWorker | None = None
_WORKER_LOCK = threading.Lock()


def init_capture_worker(path: Path) -> CaptureWorker:
    """Initialise the module-level capture worker. Idempotent.

    Returns the worker (which is also stored in module-level state so
    ``record_round_decision`` can find it). Registers an atexit hook
    to drain the queue on graceful shutdown.
    """
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is not None and _WORKER._thread is not None and _WORKER._thread.is_alive():
            return _WORKER
        _WORKER = CaptureWorker(path)
        _WORKER.start()
        # atexit fires LIFO; capture should land before noisier shutdown
        # routines run. _atexit_drain checks the worker still exists in
        # case the user reset state explicitly.
        atexit.register(_atexit_drain)
    return _WORKER


def _atexit_drain() -> None:
    global _WORKER
    w = _WORKER
    if w is None:
        return
    w.shutdown(timeout_s=_SHUTDOWN_DRAIN_TIMEOUT_S)


def get_capture_worker() -> CaptureWorker | None:
    return _WORKER


def reset_capture_worker_for_tests() -> None:
    """Test hook: tear down the module-level worker so each test starts clean."""
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is not None:
            _WORKER.shutdown(timeout_s=1.0)
        _WORKER = None


def record_round_decision(
    *,
    epoch: int,
    lock_at_unix: int,
    cutoff_ms: int,
    mode: str,
    gate: Any,
    decision: str,
    skip_reason: str | None,
    selected_strategy: str | None,
    bet_side: str | None,
    bet_size_bnb: float | None,
    pool_bull_bnb: float,
    pool_bear_bnb: float,
) -> None:
    """High-level capture entry point -- safe to call from the engine.

    Builds a snapshot from the gate's last_*_klines_raw / last_returns
    and enqueues it to the module-level worker. No-op when:

      * the worker has not been initialised (``init_capture_worker`` not
        called -- e.g. in a tool that imports the engine but doesn't
        run a real bot), or
      * gate is None, or
      * the queue is full (drops with a warn).

    Never raises. Producer-side build is sub-millisecond.
    """
    worker = _WORKER
    if worker is None or gate is None:
        return
    try:
        snapshot = build_snapshot(
            epoch=epoch,
            lock_at_unix=lock_at_unix,
            cutoff_ms=cutoff_ms,
            mode=mode,
            btc_klines_raw=getattr(gate, "last_btc_klines_raw", None),
            eth_klines_raw=getattr(gate, "last_eth_klines_raw", None),
            sol_klines_raw=getattr(gate, "last_sol_klines_raw", None),
            returns=getattr(gate, "last_returns", None),
            decision=decision,
            skip_reason=skip_reason,
            selected_strategy=selected_strategy,
            bet_side=bet_side,
            bet_size_bnb=bet_size_bnb,
            pool_bull_bnb=pool_bull_bnb,
            pool_bear_bnb=pool_bear_bnb,
        )
    except Exception as e:  # noqa: BLE001 -- never raise from capture
        warn("CAPTURE", "BUILD", "FAIL",
             msg=f"capture build failed: {type(e).__name__}: {e}",
             epoch=epoch)
        return
    worker.enqueue(snapshot)


# ---------------------------------------------------------------------------
# Reader (used by backtest replay; trivially used by tests)
# ---------------------------------------------------------------------------

def iter_captures(path: Path):
    """Stream captures from a JSONL file. Skips malformed lines.

    Yields dicts with version-checked schema. Forward-compat: callers
    that don't recognise a higher schema_version should skip the
    record.
    """
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(obj, dict):
                    continue
                if "schema_version" not in obj or "epoch" not in obj:
                    continue
                yield obj
    except (OSError, PermissionError):
        return
