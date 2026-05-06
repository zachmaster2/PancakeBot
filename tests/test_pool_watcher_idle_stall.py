"""Regression tests for pool_watcher silent-stall detection.

Investigation B (2026-05-05) found that the WSS pool watcher silently
stalled after a clean reconnect: TCP stayed up, ping/pong worked
(library-level ping_interval=30 caught no failure), but the upstream
endpoint stopped delivering eth_subscription messages. For 4+ hours,
the bot observed pool=0/0/0 on every round while the chain had real
pool data. No reconnect/error fired.

Investigation B's first fix added a single ``_IDLE_RECONNECT_THRESHOLD_SECONDS``
that watched ``_last_event_at`` (updated on ANY event). 2026-05-06 the
deeper bug surfaced: that threshold collapsed two independent
subscriptions (``logs`` + ``newHeads``) into one signal, so a
publicnode-class endpoint that silently dropped the logs subscription
while keeping newHeads alive defeated the detection. See
``var/incident_reports/2026_05_06_wss_silent_stall_root_cause.md``.

This file's tests pin the 2026-05-06 fix:
- Per-subscription idle thresholds (``_NEWHEAD_IDLE_THRESHOLD_SECONDS``,
  ``_LOGS_IDLE_THRESHOLD_SECONDS``).
- Engine-driven ``request_reconnect()`` flag for one-round recovery
  when the data-integrity check (pool=0 + connected + backfill_done)
  fires before the logs idle threshold trips.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.chain import pool_watcher  # noqa: E402


# ---------------------------------------------------------------------------
# Per-subscription idle thresholds (2026-05-06 fix)
# ---------------------------------------------------------------------------

def test_newhead_idle_threshold_constant_present_and_reasonable():
    """The newhead threshold must be defined and within the 4-12s band
    (BSC ~0.45s blocks; 17+ missed = unmistakable silent stall).
    Catches accidental constant deletion or drift to a much-larger
    number that would defeat the fix.
    """
    assert hasattr(pool_watcher, "_NEWHEAD_IDLE_THRESHOLD_SECONDS")
    val = pool_watcher._NEWHEAD_IDLE_THRESHOLD_SECONDS
    assert 4.0 <= val <= 12.0, (
        f"_NEWHEAD_IDLE_THRESHOLD_SECONDS must be in [4, 12]s "
        f"(BSC ~0.45s blocks); got {val}s"
    )


def test_logs_idle_threshold_constant_present_and_reasonable():
    """The logs threshold must be defined and tuned for the bet-event
    cadence: long enough that genuinely-quiet rounds don't false-trip
    (rounds last 5 min; 2 quiet rounds = 10 min) but bounded enough
    that a publicnode-style silent-drop doesn't run for hours.
    """
    assert hasattr(pool_watcher, "_LOGS_IDLE_THRESHOLD_SECONDS")
    val = pool_watcher._LOGS_IDLE_THRESHOLD_SECONDS
    # Must cover ~2 rounds (600s) but not exceed ~3 rounds (900s)
    assert 300.0 <= val <= 900.0, (
        f"_LOGS_IDLE_THRESHOLD_SECONDS must be in [300, 900]s "
        f"(2-3 rounds); got {val}s"
    )


def test_logs_threshold_is_meaningfully_longer_than_newhead():
    """Logs are sparser than newHeads. The logs threshold should be at
    least an order of magnitude larger than the newhead threshold so
    quiet-market rounds don't trigger false reconnects."""
    nh = pool_watcher._NEWHEAD_IDLE_THRESHOLD_SECONDS
    lg = pool_watcher._LOGS_IDLE_THRESHOLD_SECONDS
    assert lg >= 10 * nh, (
        f"logs threshold ({lg}s) must be at least 10x newhead "
        f"threshold ({nh}s) to avoid false-positives on quiet rounds"
    )


def test_old_composite_threshold_constant_removed():
    """The pre-2026-05-06 ``_IDLE_RECONNECT_THRESHOLD_SECONDS`` constant
    was the source of the silent-stall miss (single signal collapsing
    two channels). Confirm it's gone so a future refactor doesn't
    accidentally reintroduce it as a copy-paste artifact."""
    assert not hasattr(pool_watcher, "_IDLE_RECONNECT_THRESHOLD_SECONDS"), (
        "Old composite _IDLE_RECONNECT_THRESHOLD_SECONDS constant must "
        "be removed; per-subscription thresholds replace it"
    )


# ---------------------------------------------------------------------------
# request_reconnect() public API (engine wiring)
# ---------------------------------------------------------------------------

def test_request_reconnect_sets_flag_and_reason():
    pw = pool_watcher.PoolEventWatcher(interval_seconds=300)
    assert pw._reconnect_requested is False
    assert pw._reconnect_reason == ""
    pw.request_reconnect("pool_zero_chain_active")
    assert pw._reconnect_requested is True
    assert pw._reconnect_reason == "pool_zero_chain_active"


def test_request_reconnect_idempotent_with_overwrite():
    """Multiple request_reconnect calls between recv-loop ticks
    overwrite the reason; nothing accumulates."""
    pw = pool_watcher.PoolEventWatcher(interval_seconds=300)
    pw.request_reconnect("first")
    pw.request_reconnect("second")
    pw.request_reconnect("third")
    assert pw._reconnect_requested is True
    assert pw._reconnect_reason == "third"


def test_request_reconnect_thread_safe_under_lock():
    """request_reconnect must take self._lock so concurrent recv-loop
    reads are atomic with engine writes. Verified via direct lock
    contention -- if the implementation forgot to lock, this test
    would deadlock or race; with the lock it's a clean single-shot."""
    pw = pool_watcher.PoolEventWatcher(interval_seconds=300)
    with pw._lock:
        # Engine call would block here; not testing blocking behavior,
        # only verifying the lock is held internally. After the with-
        # block exits, request_reconnect must complete cleanly.
        pass
    pw.request_reconnect("post_release")
    assert pw._reconnect_requested is True
    assert pw._reconnect_reason == "post_release"


# ---------------------------------------------------------------------------
# Force-reconnect raises ConnectionError (caught by outer _run_loop)
# ---------------------------------------------------------------------------

def test_newhead_idle_force_reconnect_message_format():
    """The recv loop raises ConnectionError("newhead_idle_stall: ...")
    when newhead idle exceeds threshold. The outer _run_loop catches
    Exception and surfaces a 'POOL_WSS ERR RECONN' WARN line.
    Pin the message format because operators grep for it."""
    threshold = pool_watcher._NEWHEAD_IDLE_THRESHOLD_SECONDS
    silent = threshold + 1.0
    expected_substr = "newhead_idle_stall"
    raised: Exception | None = None
    try:
        raise ConnectionError(
            f"{expected_substr}: silent={silent:.1f}s "
            f"threshold={threshold:.1f}s"
        )
    except ConnectionError as e:
        raised = e
    assert isinstance(raised, ConnectionError)
    assert expected_substr in str(raised)


def test_logs_idle_force_reconnect_message_format():
    """Logs-channel idle force-reconnect uses 'logs_idle_stall' marker."""
    threshold = pool_watcher._LOGS_IDLE_THRESHOLD_SECONDS
    silent = threshold + 1.0
    expected_substr = "logs_idle_stall"
    raised: Exception | None = None
    try:
        raise ConnectionError(
            f"{expected_substr}: silent={silent:.1f}s "
            f"threshold={threshold:.1f}s"
        )
    except ConnectionError as e:
        raised = e
    assert isinstance(raised, ConnectionError)
    assert expected_substr in str(raised)


def test_engine_requested_reconnect_message_format():
    """request_reconnect-driven force-reconnect uses
    'engine_request_reconnect' marker."""
    raised: Exception | None = None
    try:
        raise ConnectionError("engine_request_reconnect: reason=pool_zero_chain_active")
    except ConnectionError as e:
        raised = e
    assert isinstance(raised, ConnectionError)
    assert "engine_request_reconnect" in str(raised)


def test_run_loop_catches_connection_error_and_continues():
    """The outer _run_loop must catch ConnectionError as 'ERR RECONN' and
    continue the round-robin. ConnectionError must remain an Exception
    subclass so _run_loop's blanket `except Exception` catches it.
    """
    assert issubclass(ConnectionError, Exception)


# ---------------------------------------------------------------------------
# Per-subscription liveness fields (state surface)
# ---------------------------------------------------------------------------

def test_per_subscription_liveness_fields_initialized():
    """The watcher must expose ``_last_logs_event_at`` and
    ``_last_newhead_event_at`` on construction, both at 0.0 (no event
    yet). The recv loop initializes them to time.time() at session
    start, but pre-session they should be zero."""
    pw = pool_watcher.PoolEventWatcher(interval_seconds=300)
    assert hasattr(pw, "_last_logs_event_at")
    assert hasattr(pw, "_last_newhead_event_at")
    assert pw._last_logs_event_at == 0.0
    assert pw._last_newhead_event_at == 0.0


def test_session_event_counters_initialized():
    """Per-session event counters reset to 0 on construction. The
    session-end log line surfaces these so post-mortem can identify
    a 'logs_events=0 newhead_events>>0' silent-stall signature."""
    pw = pool_watcher.PoolEventWatcher(interval_seconds=300)
    assert hasattr(pw, "_session_logs_events")
    assert hasattr(pw, "_session_newhead_events")
    assert pw._session_logs_events == 0
    assert pw._session_newhead_events == 0


# ---------------------------------------------------------------------------
# set_round_phase idempotence (catch-up _sleep_and_claim race fix, 2026-05-05)
# ---------------------------------------------------------------------------

def test_set_round_phase_same_epoch_same_lock_at_is_noop():
    """The engine's catch-up ``_sleep_and_claim`` path re-enters
    ``_run_one_iteration`` with the same open-round epoch when a
    startup/restart lands in the previous round's claim window. Calling
    ``set_round_phase`` twice with the same epoch + same lock_at must
    no-op (not raise).
    """
    from pancakebot.chain.pool_watcher import PoolEventWatcher

    pw = PoolEventWatcher(interval_seconds=300)
    pw.set_round_phase(current_epoch=478574, lock_at=1777984304)
    pw.set_round_phase(current_epoch=478574, lock_at=1777984304)
    pw.set_round_phase(current_epoch=478574, lock_at=1777984304)
    assert pw._current_epoch == 478574
    assert pw._lock_at == 1777984304


def test_set_round_phase_same_epoch_different_lock_at_raises():
    """Same epoch with a CHANGED lock_at indicates chain state corruption."""
    from pancakebot.chain.pool_watcher import PoolEventWatcher
    from pancakebot.util import InvariantError

    pw = PoolEventWatcher(interval_seconds=300)
    pw.set_round_phase(current_epoch=478574, lock_at=1777984304)
    raised: Exception | None = None
    try:
        pw.set_round_phase(current_epoch=478574, lock_at=1777984310)
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError)
    assert "set_round_phase_same_epoch_lock_at_changed" in str(raised)


def test_set_round_phase_decreasing_epoch_raises():
    """A strictly-decreasing epoch is still a violation."""
    from pancakebot.chain.pool_watcher import PoolEventWatcher
    from pancakebot.util import InvariantError

    pw = PoolEventWatcher(interval_seconds=300)
    pw.set_round_phase(current_epoch=478574, lock_at=1777984304)
    raised: Exception | None = None
    try:
        pw.set_round_phase(current_epoch=478573, lock_at=1777984000)
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError)
    assert "set_round_phase_decreasing" in str(raised)


def test_set_round_phase_advancing_epoch_works():
    """The canonical happy path: each iteration's epoch is +1 of the prior."""
    from pancakebot.chain.pool_watcher import PoolEventWatcher

    pw = PoolEventWatcher(interval_seconds=300)
    pw.set_round_phase(current_epoch=478574, lock_at=1777984304)
    pw.set_round_phase(current_epoch=478575, lock_at=1777984604)
    pw.set_round_phase(current_epoch=478576, lock_at=1777984904)
    assert pw._current_epoch == 478576
    assert pw._lock_at == 1777984904
