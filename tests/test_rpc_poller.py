"""Unit tests for RpcPoller (the Era 11 replacement for PoolEventWatcher).

Covers:
- Construction validates inputs.
- is_pool_ready() reflects internal state correctly.
- set_round_phase() idempotence + invariants (mirror PoolEventWatcher
  semantics so the engine integration remains compatible).
- get_pool() filtering + dedup.
- Public-property contracts.

Network-touching paths (cold-start, periodic poll, ramp poll) are NOT
exercised here — those would require mocked HTTP and are integration-
level. The integration validates them when the bot starts on the new
architecture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.chain.rpc_poller import RpcPoller, _Bet, _EpochPool  # noqa: E402
from pancakebot.util import InvariantError  # noqa: E402


def _make_poller() -> RpcPoller:
    """Construct a poller with default args + lightweight rpc_urls."""
    return RpcPoller(
        interval_seconds=300,
        rpc_urls=["https://test.example.com"],
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construct_default_params():
    p = _make_poller()
    assert p.connected is False
    assert p.current_endpoint == "https://test.example.com"


def test_construct_invalid_interval_raises():
    with pytest.raises(InvariantError, match="interval_seconds"):
        RpcPoller(interval_seconds=0, rpc_urls=["https://x"])
    with pytest.raises(InvariantError, match="interval_seconds"):
        RpcPoller(interval_seconds=-1, rpc_urls=["https://x"])


def test_construct_invalid_periodic_interval_raises():
    with pytest.raises(InvariantError, match="periodic_poll_interval"):
        RpcPoller(
            interval_seconds=300,
            rpc_urls=["https://x"],
            periodic_poll_interval_s=0,
        )


def test_construct_batch_size_too_large_raises():
    from pancakebot import timing_constants as _tc
    with pytest.raises(InvariantError, match="batch_size_out_of_range"):
        RpcPoller(
            interval_seconds=300,
            rpc_urls=["https://x"],
            batch_size=_tc.RPC_BATCH_BLOCK_RECEIPTS_LIMIT + 1,
        )


def test_construct_empty_rpc_urls_raises():
    with pytest.raises(InvariantError, match="rpc_urls_empty"):
        RpcPoller(interval_seconds=300, rpc_urls=[])


# ---------------------------------------------------------------------------
# is_pool_ready() state machine
# ---------------------------------------------------------------------------

def test_is_pool_ready_false_before_cold_start():
    """Brand-new poller hasn't completed cold-start; predicate returns
    False with cold_start_in_progress."""
    p = _make_poller()
    ready, reason = p.is_pool_ready()
    assert ready is False
    assert reason == "cold_start_in_progress"


def test_is_pool_ready_true_after_cold_start_and_successful_poll():
    """Simulate cold-start completion + successful poll; predicate
    returns (True, '')."""
    p = _make_poller()
    p._connected = True
    p._cold_start_done.set()
    p._last_poll_succeeded = True
    p._last_poll_too_slow = False
    ready, reason = p.is_pool_ready()
    assert ready is True
    assert reason == ""


def test_is_pool_ready_false_when_last_poll_failed():
    p = _make_poller()
    p._connected = True
    p._cold_start_done.set()
    p._last_poll_succeeded = False
    p._last_poll_too_slow = False
    ready, reason = p.is_pool_ready()
    assert ready is False
    assert reason == "last_poll_failed"


def test_is_pool_ready_false_when_last_poll_too_slow():
    p = _make_poller()
    p._connected = True
    p._cold_start_done.set()
    p._last_poll_succeeded = True
    p._last_poll_too_slow = True
    ready, reason = p.is_pool_ready()
    assert ready is False
    assert reason == "last_poll_too_slow"


def test_is_pool_ready_priority_cold_start_beats_other_failures():
    """Cold-start incomplete dominates other failure reasons."""
    p = _make_poller()
    p._connected = False
    p._last_poll_succeeded = False
    p._last_poll_too_slow = True
    ready, reason = p.is_pool_ready()
    assert ready is False
    assert reason == "cold_start_in_progress"


def test_is_pool_ready_priority_failed_beats_too_slow():
    """If last poll outright failed AND was too slow, 'failed' wins —
    a failed poll is more severe than a slow one."""
    p = _make_poller()
    p._connected = True
    p._cold_start_done.set()
    p._last_poll_succeeded = False
    p._last_poll_too_slow = True
    ready, reason = p.is_pool_ready()
    assert ready is False
    assert reason == "last_poll_failed"


# ---------------------------------------------------------------------------
# set_round_phase() (mirror PoolEventWatcher semantics)
# ---------------------------------------------------------------------------

def test_set_round_phase_negative_epoch_raises():
    p = _make_poller()
    with pytest.raises(InvariantError, match="set_round_phase_negative_epoch"):
        p.set_round_phase(current_epoch=-1, lock_at=1000)


def test_set_round_phase_nonpositive_lock_at_raises():
    p = _make_poller()
    with pytest.raises(InvariantError, match="lock_at_nonpositive"):
        p.set_round_phase(current_epoch=100, lock_at=0)


def test_set_round_phase_decreasing_epoch_raises():
    """A second call with a strictly-smaller epoch is a chain-state
    invariant violation."""
    p = _make_poller()
    # Bypass cold_start: directly set the state without triggering
    # the synchronous network call.
    p._current_epoch = 100
    p._lock_at = 1000
    p._cold_start_done.set()
    p._connected = True
    with pytest.raises(InvariantError, match="set_round_phase_decreasing"):
        p.set_round_phase(current_epoch=99, lock_at=900)


def test_set_round_phase_same_epoch_lock_at_changed_raises():
    """Same epoch with a different lock_at indicates chain corruption."""
    p = _make_poller()
    p._current_epoch = 100
    p._lock_at = 1000
    p._cold_start_done.set()
    p._connected = True
    with pytest.raises(InvariantError, match="lock_at_changed"):
        p.set_round_phase(current_epoch=100, lock_at=1500)


def test_set_round_phase_same_epoch_same_lock_at_is_noop():
    """Same epoch + same lock_at is the catch-up _sleep_and_claim
    path; must not raise (mirror PoolEventWatcher idempotence)."""
    p = _make_poller()
    p._current_epoch = 100
    p._lock_at = 1000
    p._cold_start_done.set()
    p._connected = True
    # Should not raise.
    p.set_round_phase(current_epoch=100, lock_at=1000)
    p.set_round_phase(current_epoch=100, lock_at=1000)
    assert p._current_epoch == 100
    assert p._lock_at == 1000


def test_set_round_phase_advancing_epoch_drops_stale_pools():
    """When epoch advances, stale-epoch pool entries are dropped."""
    p = _make_poller()
    p._current_epoch = 100
    p._lock_at = 1000
    p._cold_start_done.set()
    p._connected = True
    # Stuff in some stale + current + future epochs
    p._pools[99] = _EpochPool(bets=[
        _Bet(epoch=99, side="Bull", amount_wei=int(0.1 * 10**18),
             block_number=1, block_ts=900),
    ])
    p._pools[100] = _EpochPool(bets=[
        _Bet(epoch=100, side="Bear", amount_wei=int(0.2 * 10**18),
             block_number=2, block_ts=1000),
    ])
    p._pools[101] = _EpochPool(bets=[])
    p._seen_tx[99] = {"a:0"}
    p._seen_tx[100] = {"b:0"}
    p._seen_tx[101] = set()

    p.set_round_phase(current_epoch=101, lock_at=2000)

    assert 99 not in p._pools, "stale epoch should be dropped"
    assert 100 not in p._pools, "stale epoch should be dropped"
    assert 101 in p._pools, "current epoch retained"
    assert 99 not in p._seen_tx
    assert 100 not in p._seen_tx
    assert 101 in p._seen_tx


# ---------------------------------------------------------------------------
# get_pool() filtering
# ---------------------------------------------------------------------------

def test_get_pool_invalid_max_ts_raises():
    p = _make_poller()
    with pytest.raises(InvariantError, match="max_ts_nonpositive"):
        p.get_pool(epoch=100, max_ts=0)


def test_get_pool_unknown_epoch_returns_zero():
    p = _make_poller()
    bull, bear = p.get_pool(epoch=999, max_ts=1000)
    assert bull == 0.0
    assert bear == 0.0


def test_get_pool_filters_by_max_ts():
    """Only bets with 0 < block_ts < max_ts are counted."""
    p = _make_poller()
    p._pools[100] = _EpochPool(bets=[
        _Bet(epoch=100, side="Bull", amount_wei=int(0.1 * 10**18),
             block_number=1, block_ts=900),
        _Bet(epoch=100, side="Bear", amount_wei=int(0.3 * 10**18),
             block_number=2, block_ts=1100),  # excluded (> max_ts)
        _Bet(epoch=100, side="Bull", amount_wei=int(0.2 * 10**18),
             block_number=3, block_ts=950),
    ])
    bull, bear = p.get_pool(epoch=100, max_ts=1000)
    assert abs(bull - 0.3) < 1e-9, f"expected 0.3 BNB bull, got {bull}"
    assert bear == 0.0


def test_get_pool_zero_block_ts_excluded():
    """block_ts == 0 means timestamp not yet resolved; should be
    excluded from the aggregate."""
    p = _make_poller()
    p._pools[100] = _EpochPool(bets=[
        _Bet(epoch=100, side="Bull", amount_wei=int(0.5 * 10**18),
             block_number=1, block_ts=0),
    ])
    # No matching block_ts in _block_ts either, so still 0.
    bull, bear = p.get_pool(epoch=100, max_ts=1000)
    assert bull == 0.0
    assert bear == 0.0


def test_get_pool_resolves_lazy_block_ts_from_cache():
    """If a bet was added with block_ts=0 but a later block fetch
    populated _block_ts, get_pool resolves it on read."""
    p = _make_poller()
    p._pools[100] = _EpochPool(bets=[
        _Bet(epoch=100, side="Bull", amount_wei=int(0.4 * 10**18),
             block_number=42, block_ts=0),
    ])
    p._block_ts[42] = 950  # cached resolution
    bull, bear = p.get_pool(epoch=100, max_ts=1000)
    assert abs(bull - 0.4) < 1e-9
    assert bear == 0.0


# ---------------------------------------------------------------------------
# Public properties (engine surface)
# ---------------------------------------------------------------------------

def test_stats_shape():
    p = _make_poller()
    s = p.stats
    assert "connected" in s
    assert "current_endpoint" in s
    assert "poll_count" in s
    assert "last_poll_at" in s
    assert "last_poll_rtt_ms" in s
    assert "last_poll_succeeded" in s
    assert "last_poll_too_slow" in s
    assert "last_polled_block" in s


def test_is_backfill_done_compat_shim():
    """Compat shim for engine code that polls is_backfill_done.
    False before cold-start, True after."""
    p = _make_poller()
    assert p.is_backfill_done() is False
    p._cold_start_done.set()
    assert p.is_backfill_done() is True
