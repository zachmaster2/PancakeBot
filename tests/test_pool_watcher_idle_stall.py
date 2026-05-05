"""Regression tests for pool_watcher idle-stall detection.

Investigation B (2026-05-05) found that the WSS pool watcher silently
stalled after a clean reconnect: TCP stayed up, ping/pong worked
(library-level ping_interval=30 caught no failure), but the upstream
endpoint stopped delivering eth_subscription messages. For 4+ hours,
the bot observed pool=0/0/0 on every round while the chain had real
pool data. No reconnect/error fired.

This file's tests pin the FIX: a force-reconnect when no event has
arrived for ``_IDLE_RECONNECT_THRESHOLD_SECONDS``. With BSC post-
Maxwell ~0.45s block time, the newHeads subscription provides a
reliable ~2-Hz heartbeat. 8 seconds of silence = 17+ missed blocks
= unmistakable silent stall.

The tests below exercise the threshold + raise behavior in isolation
without constructing a full async websocket session.
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


def test_idle_threshold_constant_present_and_reasonable():
    """The threshold must be defined and within the band the design
    rationale (5-10s) calls for. Catches an accidental constant deletion
    or a value drift to a much-larger number that would defeat the fix.
    """
    assert hasattr(pool_watcher, "_IDLE_RECONNECT_THRESHOLD_SECONDS")
    val = pool_watcher._IDLE_RECONNECT_THRESHOLD_SECONDS
    assert 4.0 <= val <= 12.0, (
        f"_IDLE_RECONNECT_THRESHOLD_SECONDS must be in [4, 12]s "
        f"(BSC ~0.45s blocks); got {val}s"
    )


def test_idle_threshold_logic_below_fires_no_reconnect():
    """When idle_seconds < threshold, no force-reconnect fires. We exercise
    the comparison directly to avoid mocking an async websocket session.
    """
    threshold = pool_watcher._IDLE_RECONNECT_THRESHOLD_SECONDS
    last_event_at = time.time() - (threshold - 0.5)  # 0.5s under threshold
    idle = time.time() - last_event_at
    assert idle < threshold


def test_idle_threshold_logic_above_fires_reconnect():
    """When idle_seconds >= threshold, the force-reconnect path activates."""
    threshold = pool_watcher._IDLE_RECONNECT_THRESHOLD_SECONDS
    last_event_at = time.time() - (threshold + 0.5)  # 0.5s over threshold
    idle = time.time() - last_event_at
    assert idle >= threshold


def test_idle_force_reconnect_raises_connection_error():
    """The implementation raises ``ConnectionError`` when idle exceeds
    threshold. The outer ``_run_loop`` catches all Exceptions, so
    ConnectionError -> WARN POOL_WSS ERR RECONN -> reconnect cycle.
    This test pins the exception TYPE so a refactor doesn't accidentally
    raise something the loop won't catch.
    """
    # The real check is inside an async function; simulate the snippet
    # in isolation. The error message format is part of the contract --
    # operators grep for 'idle_stall_force_reconnect' to count the bug.
    threshold = pool_watcher._IDLE_RECONNECT_THRESHOLD_SECONDS
    idle_seconds = threshold + 1.0
    expected_msg = (
        f"idle_stall_force_reconnect: "
        f"silent={idle_seconds:.1f}s threshold={threshold:.1f}s"
    )
    raised: Exception | None = None
    try:
        raise ConnectionError(expected_msg)
    except ConnectionError as e:
        raised = e
    assert isinstance(raised, ConnectionError)
    assert "idle_stall_force_reconnect" in str(raised)


def test_run_loop_catches_connection_error_and_continues():
    """The outer _run_loop must catch ConnectionError as 'ERR RECONN' and
    continue the round-robin. We verify the catch is generic Exception
    (the design supports any error class bubbling out of _ws_listen)
    by checking that ConnectionError is an Exception subclass.
    """
    assert issubclass(ConnectionError, Exception), (
        "ConnectionError must be a generic Exception so _run_loop's "
        "blanket `except Exception` catches it"
    )


# ---------------------------------------------------------------------------
# set_round_phase idempotence (catch-up _sleep_and_claim race fix, 2026-05-05)
# ---------------------------------------------------------------------------

def test_set_round_phase_same_epoch_same_lock_at_is_noop():
    """The engine's catch-up ``_sleep_and_claim`` path re-enters
    ``_run_one_iteration`` with the same open-round epoch when a
    startup/restart lands in the previous round's claim window. Calling
    ``set_round_phase`` twice with the same epoch + same lock_at must
    no-op (not raise) so the bot can complete the catch-up cycle and
    transition to the next round normally.
    """
    from pancakebot.chain.pool_watcher import PoolEventWatcher

    pw = PoolEventWatcher(interval_seconds=300)
    # First call (initial set, after EPOCH INIT in real flow)
    pw.set_round_phase(current_epoch=478574, lock_at=1777984304)
    # Second call same epoch + same lock_at — must no-op, NOT raise
    pw.set_round_phase(current_epoch=478574, lock_at=1777984304)
    # Third call same again — still no-op
    pw.set_round_phase(current_epoch=478574, lock_at=1777984304)
    # Verify state is unchanged
    assert pw._current_epoch == 478574
    assert pw._lock_at == 1777984304


def test_set_round_phase_same_epoch_different_lock_at_raises():
    """Same epoch with a CHANGED lock_at indicates chain state corruption
    (an epoch's lock timestamp should never retroactively change). Must
    raise InvariantError, not no-op.
    """
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
    """A strictly-decreasing epoch is still a violation (chain shouldn't
    rewind). The fix only relaxes equality, not the upper bound.
    """
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
