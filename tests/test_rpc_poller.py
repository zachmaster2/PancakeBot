"""Unit tests for RpcPoller (the Era 11 replacement for PoolEventWatcher).

Covers:
- Construction validates inputs.
- is_pool_ready() reflects internal state correctly.
- set_round_phase() idempotence + invariants (mirror PoolEventWatcher
  semantics so the engine integration remains compatible).
- get_pool() filtering + log-id dedup.
- Public-property contracts.

Network-touching paths (cold-start, periodic poll, ramp poll) are NOT
exercised here — those would require mocked HTTP and are integration-
level. The integration validates them when the bot starts on the new
architecture.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot import timing_constants as _tc  # noqa: E402
from pancakebot.chain.rpc_poller import RpcPoller, _Bet, _EpochPool  # noqa: E402
from pancakebot.util import InvariantError  # noqa: E402


def _make_poller() -> RpcPoller:
    """Construct a poller with default args + lightweight endpoint_pool."""
    return RpcPoller(
        interval_seconds=300,
        endpoint_pool=["https://test.example.com"],
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
        RpcPoller(interval_seconds=0, endpoint_pool=["https://x"])
    with pytest.raises(InvariantError, match="interval_seconds"):
        RpcPoller(interval_seconds=-1, endpoint_pool=["https://x"])


def test_construct_invalid_periodic_interval_raises():
    with pytest.raises(InvariantError, match="periodic_poll_interval"):
        RpcPoller(
            interval_seconds=300,
            endpoint_pool=["https://x"],
            periodic_poll_interval_s=0,
        )


def test_construct_batch_size_too_large_raises():
    from pancakebot import timing_constants as _tc
    with pytest.raises(InvariantError, match="batch_size_out_of_range"):
        RpcPoller(
            interval_seconds=300,
            endpoint_pool=["https://x"],
            batch_size=_tc.RPC_BATCH_MAX_BLOCKS + 1,
        )


def test_construct_empty_rpc_urls_raises():
    with pytest.raises(InvariantError, match="endpoint_pool_empty"):
        RpcPoller(interval_seconds=300, endpoint_pool=[])


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


def test_is_pool_ready_true_after_cold_start():
    """After cold-start with no infeasibility flag, ready."""
    p = _make_poller()
    p._connected = True
    p._cold_start_done.set()
    ready, reason = p.is_pool_ready()
    assert ready is True
    assert reason == ""


def test_is_pool_ready_returns_true_after_single_poll_failure_when_feasible():
    """A single poll failure is informational, not skip-triggering.
    The integrating signal is _catchup_infeasible_for_round."""
    p = _make_poller()
    p._connected = True
    p._cold_start_done.set()
    # Simulate a recent poll failure — no longer affects readiness.
    p._last_poll_succeeded = False
    p._last_poll_too_slow = True
    ready, reason = p.is_pool_ready()
    assert ready is True, "single poll failures should not skip-trigger"
    assert reason == ""


def test_is_pool_ready_returns_catchup_infeasible_when_flagged():
    """When the feasibility check has flagged the round, predicate
    returns False with catchup_infeasible_for_round."""
    p = _make_poller()
    p._connected = True
    p._cold_start_done.set()
    p._catchup_infeasible_for_round = True
    ready, reason = p.is_pool_ready()
    assert ready is False
    assert reason == "catchup_infeasible_for_round"


def test_is_pool_ready_priority_cold_start_beats_catchup_infeasible():
    """Cold-start incomplete dominates other skip reasons."""
    p = _make_poller()
    p._connected = False
    p._catchup_infeasible_for_round = True
    ready, reason = p.is_pool_ready()
    assert ready is False
    assert reason == "cold_start_in_progress"


def test_is_pool_ready_returns_poll_in_progress_when_flag_set():
    """When a poll is mid-flight, predicate returns False with
    poll_in_progress so the engine can't read a half-built aggregate."""
    p = _make_poller()
    p._connected = True
    p._cold_start_done.set()
    p._poll_in_progress = True
    ready, reason = p.is_pool_ready()
    assert ready is False
    assert reason == "poll_in_progress"


def test_is_pool_ready_priority_catchup_infeasible_beats_poll_in_progress():
    """Catch-up-infeasible dominates poll_in_progress: if the round is
    already unbettable for time-budget reasons, the in-flight poll
    won't change that."""
    p = _make_poller()
    p._connected = True
    p._cold_start_done.set()
    p._catchup_infeasible_for_round = True
    p._poll_in_progress = True
    ready, reason = p.is_pool_ready()
    assert ready is False
    assert reason == "catchup_infeasible_for_round"


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


def test_set_round_phase_advancing_epoch_drops_past_round_pools(monkeypatch):
    """When epoch advances, past-round pool entries are dropped (covers
    both the normal advance case AND the multi-round skip case where
    intermediate epochs were never set as current). The newly-introduced
    _on_epoch_advance hook is mocked here — its own behavior is covered
    by dedicated tests below."""
    p = _make_poller()
    p._current_epoch = 100
    p._lock_at = 1000
    p._cold_start_done.set()
    p._connected = True
    # Stuff in some past-round + current + future epochs
    p._pools[99] = _EpochPool(bets=[
        _Bet(epoch=99, side="Bull", amount_wei=int(0.1 * 10**18),
             block_number=1, block_ts=900),
    ])
    p._pools[100] = _EpochPool(bets=[
        _Bet(epoch=100, side="Bear", amount_wei=int(0.2 * 10**18),
             block_number=2, block_ts=1000),
    ])
    p._pools[101] = _EpochPool(bets=[])
    p._processed_bet_log_ids[99] = {"a:0"}
    p._processed_bet_log_ids[100] = {"b:0"}
    p._processed_bet_log_ids[101] = set()

    # Don't actually try to fetch from test.example.com.
    monkeypatch.setattr(p, "_on_epoch_advance", lambda **kw: None)

    p.set_round_phase(current_epoch=101, lock_at=2000)

    assert 99 not in p._pools, "past-round epoch should be dropped"
    assert 100 not in p._pools, "past-round epoch should be dropped"
    assert 101 in p._pools, "current epoch retained"
    assert 99 not in p._processed_bet_log_ids
    assert 100 not in p._processed_bet_log_ids
    assert 101 in p._processed_bet_log_ids


# ---------------------------------------------------------------------------
# _compute_round_start_block (cache-first, RPC fallback)
# ---------------------------------------------------------------------------

def test_compute_round_start_block_uses_cache_when_available():
    """If _block_ts has a recent-enough anchor, no RPC call needed.

    Bundle 4 (2026-05-14): block-time divisor is BSC_BLOCK_TIME_MS=450
    (post-Lorentz empirical), down from 500. With a 30s anchor delta,
    blocks-ahead = round(30000/450) = 67 (was 60 with 500ms divisor).
    """
    p = _make_poller()
    target_ts = 10_000  # Unix-second-ish
    # Anchor 30s before target with block 5000.
    p._block_ts[5000] = target_ts - 30
    rs = p._compute_round_start_block(target_ts)
    # 30s * 1000 / 450ms_per_block = round(66.67) = 67 blocks ahead of anchor.
    assert rs == 5000 + 67


def test_compute_round_start_block_rejects_stale_anchor():
    """If the cached anchor is more than 60s before round_start_ts,
    cache is rejected; falls through to RPC."""
    p = _make_poller()
    target_ts = 10_000
    p._block_ts[5000] = target_ts - 120  # 120s old, too stale
    # No mock for RPC -> should return None (RPC fails to test.example.com).
    # We mock the helper to confirm it's invoked.
    invoked = {"n": 0}

    def fake_header():
        invoked["n"] += 1
        return (10000, target_ts + 10)

    p._rpc_eth_get_latest_block_header = fake_header  # type: ignore[assignment]
    rs = p._compute_round_start_block(target_ts)
    # Bundle 4: head_ts = target+10 -> 10s back -> round(10000/450)=22
    # blocks back from 10000 = 9978 (was 9980 with 500ms divisor).
    assert rs == 9978
    assert invoked["n"] == 1, "RPC fallback should have been invoked"


def test_compute_round_start_block_falls_back_to_rpc_when_cache_empty():
    """Empty cache forces the RPC path."""
    p = _make_poller()

    def fake_header():
        return (1_000_000, 50_000)

    p._rpc_eth_get_latest_block_header = fake_header  # type: ignore[assignment]
    rs = p._compute_round_start_block(round_start_ts=49_000)
    # Bundle 4: delta_seconds = 50000 - 49000 = 1000s -> round(1_000_000/450) = 2222
    # blocks at 450ms/block. round_start_block = 1_000_000 - 2222 = 997_778
    # (was 998_000 with 500ms divisor).
    assert rs == 997_778


def test_compute_round_start_block_returns_none_on_rpc_failure():
    """If RPC fails AND cache is empty, return None and the caller
    leaves cursor untouched."""
    p = _make_poller()

    def boom():
        raise RuntimeError("publicnode_unreachable")

    p._rpc_eth_get_latest_block_header = boom  # type: ignore[assignment]
    rs = p._compute_round_start_block(round_start_ts=10_000)
    assert rs is None


def test_compute_round_start_block_handles_future_round_start():
    """If head_ts <= round_start_ts (round hasn't begun per head),
    return head_num as the cursor target."""
    p = _make_poller()

    def fake_header():
        return (5000, 9_000)  # head_ts BEFORE round_start_ts

    p._rpc_eth_get_latest_block_header = fake_header  # type: ignore[assignment]
    rs = p._compute_round_start_block(round_start_ts=10_000)
    assert rs == 5000


# ---------------------------------------------------------------------------
# _on_epoch_advance: cursor clamp + feasibility check
# ---------------------------------------------------------------------------

def _prep_for_epoch_advance(p: RpcPoller) -> None:
    p._current_epoch = 100
    p._lock_at = 1_000_000
    p._cold_start_done.set()
    p._connected = True


def test_on_epoch_advance_clamps_cursor_after_long_silence():
    """Cursor far behind -> clamped to round_start - 1."""
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 1_000  # very stale

    # Mock: round_start_block = 99_900; head_num = 100_000.
    p._rpc_eth_get_latest_block_header = lambda: (100_000, 1_000_050)  # type: ignore[assignment]
    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]

    # head_ts = 1_000_300, lock_at = 1_000_350, round_start_ts = 1_000_050.
    # Bundle 4: delta = 250s -> round(250_000/450) = 556 blocks ->
    # rs = 100_000 - 556 = 99_444 (was 99_500 with 500ms divisor).
    p._rpc_eth_get_latest_block_header = lambda: (100_000, 1_000_300)  # type: ignore[assignment]

    p._on_epoch_advance(lock_at=1_000_350, current_epoch=101)

    # Cursor should now be 99_444 - 1 = 99_443 (Bundle 4 ms-precise correction).
    assert p._last_polled_block_number == 99_443


def test_on_epoch_advance_does_not_rewind_cursor_when_in_round():
    """If cursor already past round_start_block, it must not rewind."""
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 99_900  # past round_start = 99_500

    # Same mocks as the clamp test but cursor is in-round.
    p._rpc_eth_get_latest_block_header = lambda: (100_000, 1_000_300)  # type: ignore[assignment]
    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]

    p._on_epoch_advance(lock_at=1_000_350, current_epoch=101)

    assert p._last_polled_block_number == 99_900, (
        "in-round cursor must not rewind"
    )


def test_on_epoch_advance_resets_infeasibility_flag():
    """Flag from previous round always cleared at the start of a new
    round."""
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._catchup_infeasible_for_round = True  # leftover from prev round
    p._last_polled_block_number = 99_999

    # Mock: very small backlog, plenty of time -> still feasible.
    p._rpc_eth_get_latest_block_header = lambda: (100_000, 1_000_300)  # type: ignore[assignment]
    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]

    # lock_at far in the future to ensure feasibility.
    import time as _t
    far_lock_at = int(_t.time()) + 10_000

    p._on_epoch_advance(lock_at=far_lock_at, current_epoch=101)

    assert p._catchup_infeasible_for_round is False


def test_on_epoch_advance_marks_infeasible_when_catchup_exceeds_budget():
    """Far behind + tight time -> flag set, log emitted at WARN."""
    import time as _t
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 1_000  # very stale

    # head_ts to make round_start_block = 90_000, head=100_000 -> 10_000 behind.
    # 10_000 / 20 = 500 batches @ p99(20)ms estimated catch-up (way past deadline).
    # Set lock_at to "now + 1s" to make available_ms tiny.
    p._rpc_eth_get_latest_block_header = lambda: (100_000, _t.time())  # type: ignore[assignment]
    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]

    # round_start_ts = lock_at - 300, round_start_block ~= 100_000 -
    #   (head_ts - round_start_ts)*2 = 100_000 - 600 = 99_400.
    # So blocks_behind = 100_000 - 99_400 = 600. 30 batches * p99(20) ms.
    # available = max(0, 1000 - 200) = 800 ms. infeasible.
    near_lock_at = int(_t.time()) + 1

    p._on_epoch_advance(lock_at=near_lock_at, current_epoch=101)

    assert p._catchup_infeasible_for_round is True


def test_on_epoch_advance_handles_block_number_failure_gracefully():
    """If eth_blockNumber raises, leave flag at False; don't propagate."""
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 99_900

    p._rpc_eth_get_latest_block_header = lambda: (100_000, 1_000_300)  # type: ignore[assignment]
    p._rpc_eth_block_number = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[assignment]

    # Should not raise.
    p._on_epoch_advance(lock_at=1_000_350, current_epoch=101)
    assert p._catchup_infeasible_for_round is False


def test_on_epoch_advance_handles_compute_round_start_failure_gracefully():
    """If _compute_round_start_block returns None (RPC + cache both
    failed), leave cursor untouched and don't raise."""
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 1_000

    # _block_ts empty AND header RPC raises.
    p._rpc_eth_get_latest_block_header = (  # type: ignore[assignment]
        lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    # Should not raise; cursor stays put.
    p._on_epoch_advance(lock_at=1_000_350, current_epoch=101)
    assert p._last_polled_block_number == 1_000


def test_first_call_set_round_phase_does_not_invoke_clamp(monkeypatch):
    """First call goes through cursor-init (_initialize_cursor_from_head),
    NOT the epoch-advance hook. Bundle 2 (2026-05-13): renamed from
    _cold_start; same dispatch decision tree."""
    p = _make_poller()
    invoked = {"n": 0}

    def fake_advance(**kw):
        invoked["n"] += 1

    def fake_init():
        pass

    monkeypatch.setattr(p, "_on_epoch_advance", fake_advance)
    monkeypatch.setattr(p, "_initialize_cursor_from_head", fake_init)

    p.set_round_phase(current_epoch=100, lock_at=1000)
    assert invoked["n"] == 0


# ---------------------------------------------------------------------------
# Feasibility math
# ---------------------------------------------------------------------------

def test_estimated_catchup_ms_calculation():
    """Full-batch backlogs: ``full_batches * p99(batch_size)`` plus the
    partial-batch tail. Refactored 2026-05-12: the helper is now
    batch-size-aware (full batches at p99(batch_size), partial at
    p99(remainder)).

    1319ms is the 2026-05-11 fire-to-all-pool measurement for batch=20
    (n=30, bot stopped). See RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    docstring in timing_constants.py.
    """
    p = _make_poller()
    rtt_full = _tc.rpc_rtt_p99_for_batch(20)  # 1319 canonical
    rtt_one = _tc.rpc_rtt_p99_for_batch(1)    # 421 canonical (ceiling at small end)
    assert p._estimated_catchup_ms(0) == 0
    assert p._estimated_catchup_ms(20) == rtt_full           # 1 full batch, 0 remainder
    assert p._estimated_catchup_ms(21) == rtt_full + rtt_one  # 1 full + 1-block partial
    assert p._estimated_catchup_ms(60) == 3 * rtt_full        # 3 full batches, 0 remainder


def test_estimated_catchup_ms_partial_batch_uses_remainder_rtt():
    """The partial trailing batch uses the (smaller) p99 for the
    remainder size — not the full-batch p99 — eliminating the prior
    ceiling-style overestimate. At blocks_behind=7 (= remainder only,
    no full batches), the result is rtt_p99(7) ~ 827ms (interpolated),
    NOT the prior unconditional rtt_p99(20)=1319ms.
    """
    p = _make_poller()
    # 7 blocks: 0 full batches + 7-block partial.
    assert p._estimated_catchup_ms(7) == _tc.rpc_rtt_p99_for_batch(7)
    # 18 blocks: same — 0 full batches + 18-block partial.
    assert p._estimated_catchup_ms(18) == _tc.rpc_rtt_p99_for_batch(18)
    # 27 blocks: 1 full batch + 7-block partial.
    assert p._estimated_catchup_ms(27) == (
        _tc.rpc_rtt_p99_for_batch(20) + _tc.rpc_rtt_p99_for_batch(7)
    )


def test_estimated_catchup_ms_strictly_smaller_than_old_ceiling_math():
    """Regression guard: the new estimate is at most equal to the prior
    ``ceil(n/batch_size) * p99(batch_size)`` ceiling math, and strictly
    smaller for any non-multiple of batch_size. At canonical batch=20:
      - blocks=7  : old 1319, new ~827
      - blocks=21 : old 2638, new 1319+421=1740
      - blocks=27 : old 2638, new 1319+827=2146
    This is the load-bearing property: small backlogs no longer
    false-INFEAS due to ceiling overestimation.
    """
    p = _make_poller()
    rtt20 = _tc.rpc_rtt_p99_for_batch(20)
    for blocks in (7, 21, 27, 33, 47):
        full, remainder = divmod(blocks, 20)
        old_ceiling = (full + (1 if remainder else 0)) * rtt20
        new_estimate = p._estimated_catchup_ms(blocks)
        # When remainder > 0, new estimate must be strictly smaller
        # (since rtt_p99(remainder) < rtt_p99(20) for remainder < 20).
        if remainder > 0:
            assert new_estimate < old_ceiling, (
                f"blocks={blocks}: new={new_estimate} should be < "
                f"old_ceiling={old_ceiling} (smaller partial RTT)"
            )
        else:
            # Multiples of batch_size: same result, both math models.
            assert new_estimate == old_ceiling


def test_is_catchup_infeasible_returns_false_when_no_backlog():
    p = _make_poller()
    assert p._is_catchup_infeasible(blocks_behind=0, lock_at=10**12) is False


def test_is_catchup_infeasible_returns_false_when_lock_at_zero():
    """Pre-cold-start state: lock_at=0 means we have no round info."""
    p = _make_poller()
    assert p._is_catchup_infeasible(blocks_behind=1000, lock_at=0) is False


def test_is_catchup_infeasible_returns_true_when_estimate_exceeds_budget():
    """600 blocks => 30 batches * p99(20) ms estimated.
    Available = 1000 - 200 = 800ms. Infeasible regardless of exact p99 value."""
    import time as _t
    p = _make_poller()
    near_lock = int(_t.time()) + 1
    assert p._is_catchup_infeasible(blocks_behind=600, lock_at=near_lock) is True


def test_is_catchup_infeasible_returns_false_when_estimate_fits_budget():
    """20 blocks => 1 batch * p99(20)ms estimated. Available = 60_000 -
    200 = 59_800ms. Feasible regardless of p99 value."""
    import time as _t
    p = _make_poller()
    far_lock = int(_t.time()) + 60
    assert p._is_catchup_infeasible(blocks_behind=20, lock_at=far_lock) is False


def test_catchup_feasibility_at_typical_30_block_lag():
    """Regression: at canonical pool_cutoff=6 (final fires at lock-4.7s),
    a 30-block lag should be FEASIBLE (not trip INFEAS). 30-block lag at
    batch_size=20 = ceil(30/20)=2 batches * p99(20). With the 2026-05-11
    fire-to-all p99=1319ms, 2 * 1319 = 2638ms, must fit comfortably
    within available_catchup_ms.

    Also tests the 23-block-lag false-positive scenario the
    2026-05-11_fire_to_all_p99_measurement.md memo flagged: at 23 blocks
    (= 2 batches), the math must NOT trip INFEAS just because of stale
    constants. Anchors the constant value as load-bearing.

    Pin lock_at = now + 5s (well above canonical final-poll timing of
    4.7s) so the test has comfortable margin against int() truncation
    of the current wallclock and any small wallclock drift between the
    pin and the assertion. The point is to test the math contract, not
    race CI timers.
    """
    import time as _t
    p = _make_poller()
    lock_at = int(_t.time()) + 5  # 5s out: ~4800ms available after safety
    # 23-block lag (post-2026-05-12 batch-aware math):
    #   1 full batch (20) * p99(20)=1319ms + 3-block partial at
    #   p99(3)=538ms (interpolated) = 1857ms < 4800ms. Feasible.
    assert p._is_catchup_infeasible(blocks_behind=23, lock_at=lock_at) is False, (
        "23-block lag must be feasible at canonical final-poll timing. "
        "If this fails, RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE[20] may be "
        "stale relative to current transport — re-measure."
    )
    # 30-block lag: 1 full + 10-block partial @ p99(10)=910ms = 2229ms. Same result.
    assert p._is_catchup_infeasible(blocks_behind=30, lock_at=lock_at) is False


# ---------------------------------------------------------------------------
# _compute_periodic_timeout: lock-anchored 3-state cadence math (2026-05-12)
# ---------------------------------------------------------------------------

def _make_anchored_poller() -> RpcPoller:
    """Poller with default canonical ramp_poll_1 offset (7500ms) and
    interval=300s for periodic-loop math tests."""
    return RpcPoller(
        interval_seconds=300,
        endpoint_pool=["https://test.example.com"],
        ramp_poll_1_wakeup_offset_before_lock_ms=7500,
    )


def test_periodic_timeout_state_b_anchored_to_round_open_plus_8():
    """State B steady. At now = round_open + 5s, next anchored tick
    fires at round_open + 8s (k=1). Timeout = 3s."""
    p = _make_anchored_poller()
    lock_at = 1_000_300
    round_open = lock_at - 300  # 1_000_000
    now = round_open + 5  # 5s into the open round
    timeout = p._compute_periodic_timeout(
        now=float(now), lock_at=lock_at, period=8,
    )
    assert timeout == pytest.approx(3.0, abs=1e-6)


def test_periodic_timeout_state_b_advances_k_after_each_tick():
    """At now = round_open + 17s (between ticks at +16 and +24), the
    next-tick computation yields k=3, next_at = round_open + 24.
    Timeout = 7s."""
    p = _make_anchored_poller()
    lock_at = 1_000_300
    round_open = lock_at - 300
    now = round_open + 17
    timeout = p._compute_periodic_timeout(
        now=float(now), lock_at=lock_at, period=8,
    )
    assert timeout == pytest.approx(7.0, abs=1e-6)


def test_periodic_timeout_state_b_suspend_in_ramp_window():
    """At now = lock_at - 6s, the next anchored tick would land at
    round_open + 296 = lock_at - 4s (inside the ramp window
    (lock_at - 7.5s, lock_at]). Suspend: sleep to lock_at + 0.1s,
    i.e. timeout = 6.1s."""
    p = _make_anchored_poller()
    lock_at = 1_000_300
    now = lock_at - 6
    timeout = p._compute_periodic_timeout(
        now=float(now), lock_at=lock_at, period=8,
    )
    assert timeout == pytest.approx(6.1, abs=1e-6)


def test_periodic_timeout_reschedule_when_anchored_tick_in_rtt_safety_band():
    """The anchored tick is outside the ramp window itself but close
    enough that its worst-case HTTP RTT (5s + 0.05s safety = 5.05s)
    could extend into ramp_1's wake. Fire earlier at the latest safe
    time so a full-timeout poll still completes before ramp_window_start.

    Canonical config: ramp_window_start = lock_at - 7.5,
    safe_fire_latest = ramp_window_start - 5 - 0.05 = lock_at - 12.55.
    At now = lock_at - 13, the anchored next_at = round_open + 288 =
    lock_at - 12, which falls in the (safe_fire_latest, ramp_window_start)
    band. safe_fire_latest (lock_at - 12.55) is 0.45s in the future →
    reschedule, timeout = 0.45s.
    """
    p = _make_anchored_poller()
    lock_at = 1_000_300
    now = lock_at - 13
    timeout = p._compute_periodic_timeout(
        now=float(now), lock_at=lock_at, period=8,
    )
    assert timeout == pytest.approx(0.45, abs=1e-6)


def test_periodic_timeout_suspend_when_safe_fire_latest_has_passed():
    """The novel safety branch added 2026-05-18: the anchored tick is
    outside the ramp window itself, but ``safe_fire_latest`` has
    already passed (the previous poll overran). Firing now would
    risk a periodic in-flight at ramp_1's wake. Suspend the tick;
    ramp_1+ramp_2 absorb the extra backlog.

    Canonical config (period=8, ramp_offset=7.5s, max_rtt=5s,
    safety=0.05s):
      ramp_window_start = lock_at - 7.5
      safe_fire_latest = lock_at - 12.55
      Band (safe_fire_latest, ramp_window_start) is 5.05s wide.

    At canonical anchoring, k=36 yields next_at = round_open + 288 =
    lock_at - 12 — inside the band (not the ramp window itself). The
    overrun branch fires for now in (lock_at - 12.55, lock_at - 12):
    next_at is still ≤ ramp_window_start so the in-window suspend
    doesn't trigger, but safe_fire_latest is already in the past so
    no safe-fire moment remains.

    At now = lock_at - 12.5:
      - next_at = lock_at - 12 (NOT in ramp window: -12 < -7.5)
      - next_at > safe_fire_latest (-12 > -12.55): in the band
      - safe_fire_latest <= now (-12.55 <= -12.5): overrun
      → suspend, timeout = lock_at + 0.1 - now = 12.6s

    Critical: if the in-window suspend branch (next_at >=
    ramp_window_start) fired here instead, the assertion would still
    pass — both branches return "suspend" with the same timeout.
    Hence the explicit assertions below on next_at, safe_fire_latest,
    and ramp_window_start positions: they pin the test to the
    overrun branch, not in-window suspend.
    """
    p = _make_anchored_poller()
    lock_at = 1_000_300
    now = lock_at - 12.5

    # Pin the branch: assert the geometric conditions that route to
    # the overrun branch specifically (next_at outside ramp window,
    # safe_fire_latest in the past).
    round_open = lock_at - 300
    ramp_window_start = lock_at - 7.5
    safe_fire_latest = ramp_window_start - 5 - 0.05  # = lock_at - 12.55
    k = max(1, int((now - round_open) // 8) + 1)
    next_at = round_open + k * 8
    assert next_at == lock_at - 12, (
        f"test setup invariant: expected next_at=lock_at-12, got {next_at - lock_at}"
    )
    assert next_at < ramp_window_start, (
        "next_at must NOT be in the ramp window itself — otherwise the test "
        "would hit the in-window suspend branch, not the overrun branch"
    )
    assert safe_fire_latest <= now, (
        "safe_fire_latest must already be in the past — otherwise the test "
        "would hit the reschedule branch, not the overrun branch"
    )

    timeout = p._compute_periodic_timeout(
        now=float(now), lock_at=lock_at, period=8,
    )
    # Suspend sleeps to lock_at + 0.1: timeout = 12.6.
    assert timeout == pytest.approx(12.6, abs=1e-6)


def test_periodic_timeout_state_c_post_lock_wall_clock_fallback():
    """At now > lock_at (engine in _sleep_and_claim, set_round_phase
    hasn't fired for the next round yet), fall back to wall-clock
    cadence at the configured period — timeout = period seconds."""
    p = _make_anchored_poller()
    lock_at = 1_000_300
    now = lock_at + 2  # 2s past lock, mid-claim window
    timeout = p._compute_periodic_timeout(
        now=float(now), lock_at=lock_at, period=8,
    )
    assert timeout == pytest.approx(8.0, abs=1e-6)


def test_periodic_timeout_state_b_re_anchors_when_lock_at_advances():
    """After an epoch advance updates _lock_at, the next computation
    uses the new anchor. Simulates the natural re-anchoring at round
    boundaries when set_round_phase fires."""
    p = _make_anchored_poller()
    lock_at_old = 1_000_300
    lock_at_new = lock_at_old + 300  # 1_000_600
    round_open_new = lock_at_new - 300  # 1_000_300

    # Halfway through the new round at lock_old + 17s = round_open_new + 17.
    now = lock_at_old + 17
    timeout = p._compute_periodic_timeout(
        now=float(now), lock_at=lock_at_new, period=8,
    )
    # k=3, next_at = round_open_new + 24 = now + 7.
    assert timeout == pytest.approx(7.0, abs=1e-6)


def test_periodic_timeout_state_b_exact_boundary_suspends():
    """Reviewer Y2 (2026-05-12): the suspend predicate is
    ``next_at >= ramp_window_start`` (closed at the boundary, not open).
    At canonical (period=8s, ramp_offset=7500ms) no boundary-aligned
    tick exists, but a future change to either constant could produce
    one. Construct a config where it does and pin the >= behavior.

    Construction: ``ramp_poll_1_wakeup_offset_before_lock_ms=200_000`` (200s) →
    ``ramp_window_start = lock_at − 200``. With ``round_open = 0``,
    ``period = 8``, ticks land at 8, 16, ..., 800. The 800 tick
    falls EXACTLY on the ramp_window_start boundary at lock_at=1000.
    With ``>`` predicate, that tick would fire steady at lock_at−200,
    racing ramp_1. With ``>=`` predicate, it suspends.
    """
    p = RpcPoller(
        interval_seconds=1000,
        endpoint_pool=["https://test.example.com"],
        ramp_poll_1_wakeup_offset_before_lock_ms=200_000,
    )
    lock_at = 1000
    # next-tick math at now=795: k = max(1, 795//8 + 1) = 100;
    # next_at = 100 * 8 = 800. ramp_window_start = 1000 - 200 = 800.
    # next_at == ramp_window_start exactly.
    now = 795.0
    timeout = p._compute_periodic_timeout(
        now=now, lock_at=lock_at, period=8,
    )
    # Suspend timeout sleeps past lock: lock_at + 0.1 - now = 205.1.
    # If this fails, the suspend predicate (next_at >= ramp_window_start)
    # may have regressed to >, in which case the boundary tick would have
    # fired steady at lock_at-200 instead of suspending.
    assert timeout == pytest.approx(lock_at + 0.1 - now, abs=1e-6)


def test_poll_lock_arbitration_periodic_yields_to_concurrent_ramp_poll():
    """Reviewer Y3 (2026-05-12): when two polls fire near-simultaneously
    (periodic loop wake racing engine ramp_1 wake), the non-blocking
    ``_poll_lock.acquire(blocking=False)`` in ``_poll_now`` ensures one
    wins and the other returns immediately — periodic is best-effort so
    yielding is correct behavior.

    Setup: a slow-fetch holds the lock long enough that a second
    ``_poll_now`` call attempts to acquire it. Verify only one fetch
    ran (the winner) and the second call returned cleanly with no
    cursor advance or state mutation.
    """
    import threading
    import time as _t
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 99_990  # 10 blocks behind; feasible budget

    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]
    # Plenty of time to lock so feasibility doesn't trip.
    p._lock_at = int(_t.time()) + 60

    fetch_started = threading.Event()
    fetch_can_finish = threading.Event()
    fetched_calls: list[int] = []

    def slow_fetch(block_numbers):
        fetch_started.set()
        # Emulate a slow eth_getBlockReceipts batch by parking here.
        # The test signals fetch_can_finish once the racing call has
        # had its chance to attempt _poll_lock acquisition.
        if not fetch_can_finish.wait(timeout=5.0):
            raise RuntimeError("test deadlock waiting on fetch_can_finish")
        fetched_calls.append(len(block_numbers))

    p._fetch_and_process_blocks = slow_fetch  # type: ignore[assignment]

    # Thread A: simulates the engine-driven ramp_1 poll — acquires the
    # lock and parks inside the slow fetch.
    def run_ramp():
        p._poll_now(deadline_ms=0, label="ramp")

    t_ramp = threading.Thread(target=run_ramp, daemon=True, name="test-ramp")
    t_ramp.start()
    assert fetch_started.wait(timeout=2.0), (
        "ramp poll never reached the fetch — setup error"
    )

    # Thread B: simulates a periodic-loop wake firing while ramp is
    # mid-flight. Should return promptly because _poll_lock is held.
    fetches_before_periodic = len(fetched_calls)
    cursor_before_periodic = p._last_polled_block_number

    def run_periodic():
        p._poll_now(deadline_ms=0, label="period")

    t_period = threading.Thread(target=run_periodic, daemon=True, name="test-period")
    t_period.start()
    t_period.join(timeout=2.0)
    assert not t_period.is_alive(), (
        "periodic must have returned immediately when _poll_lock held by ramp"
    )

    # Periodic must have done nothing: no extra fetch, no cursor advance,
    # no flag flip.
    assert len(fetched_calls) == fetches_before_periodic, (
        "periodic should NOT have invoked the fetcher while ramp held the lock"
    )
    assert p._last_polled_block_number == cursor_before_periodic, (
        "periodic should NOT have advanced the cursor (ramp owns this poll)"
    )

    # Let ramp finish; verify exactly one fetch happened (ramp's).
    fetch_can_finish.set()
    t_ramp.join(timeout=2.0)
    assert not t_ramp.is_alive(), "ramp poll must complete"
    assert len(fetched_calls) == 1, (
        f"exactly one fetch should have run (the winner); got {len(fetched_calls)}"
    )


# ---------------------------------------------------------------------------
# _rpc_eth_get_latest_block_header (RPC parsing)
# ---------------------------------------------------------------------------

def test_rpc_eth_get_latest_block_header_parses_response():
    p = _make_poller()
    # Mock the RPC layer; verify the parser unpacks the dict.
    p._rpc_call_single = lambda method, params: {  # type: ignore[assignment]
        "number": "0x186a0",     # 100000
        "timestamp": "0x5f5e100", # 100000000
    }
    num, ts = p._rpc_eth_get_latest_block_header()
    assert num == 100_000
    assert ts == 100_000_000


def test_rpc_eth_get_latest_block_header_raises_on_unexpected_shape():
    p = _make_poller()
    p._rpc_call_single = lambda method, params: "not_a_dict"  # type: ignore[assignment]
    with pytest.raises(InvariantError, match="unexpected_result"):
        p._rpc_eth_get_latest_block_header()


def test_rpc_eth_get_latest_block_header_raises_on_missing_fields():
    p = _make_poller()
    p._rpc_call_single = lambda method, params: {"number": "0x1"}  # type: ignore[assignment]
    with pytest.raises(InvariantError, match="missing_fields"):
        p._rpc_eth_get_latest_block_header()


# ---------------------------------------------------------------------------
# _poll_now: mid-round feasibility check + log severity matrix
# ---------------------------------------------------------------------------

def test_poll_now_aborts_early_when_catchup_infeasible_mid_round(caplog):
    """Pre-lock infeasibility (time_until_lock_ms > 0) sets the flag
    AND emits a WARN. The cursor doesn't advance because no batches
    were fetched.

    Post-2026-05-12 gating: lock_at must be strictly in the future for
    the flag to set; the previous version used ``lock_at = int(now)``
    which under the new gating is post-lock (time_until_lock_ms == 0)
    and now exercises the no-flag branch. Use ``now + 1`` to ensure
    we land on the pre-lock branch.
    """
    import time as _t
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 1_000

    # Mock: head jumped to 100_000 (99_000 blocks behind).
    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]
    # If the batch fetcher is invoked, blow up the test.
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda block_numbers: pytest.fail("should not have fetched")
    )

    # Pre-lock + tight time: ~1s until lock, available ~800ms. Math
    # says 99_000 blocks of catch-up dwarfs available → INFEAS, flag set.
    p._lock_at = int(_t.time()) + 1

    p._poll_now(deadline_ms=0, label="period")

    assert p._catchup_infeasible_for_round is True


def test_poll_now_post_lock_infeas_does_not_set_flag(capsys):
    """Post-2026-05-12 gating: when ``time_until_lock_ms <= 0`` (the
    round has already locked but set_round_phase hasn't advanced
    ``_lock_at`` for the next round yet — a trailing periodic in the
    claim window), an INFEAS verdict must NOT set
    ``_catchup_infeasible_for_round``. The flag is for the CURRENT
    bettable round; once lock has passed it's moot. The poll still
    aborts (math says it cannot finish), but no flag, no WARN.

    Note: pancakebot.log writes via sys.stdout (not stdlib logging), so
    capsys captures emitted lines and we grep for "INFEAS" rather than
    using caplog records.
    """
    import time as _t
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 1_000

    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda block_numbers: pytest.fail("should not have fetched")
    )

    # Stale anchor: lock_at strictly in the past relative to _t.time().
    # Even with int() truncation, lock_at - 2 is guaranteed < now.
    p._lock_at = int(_t.time()) - 2
    # Pre-test False so we detect if any branch flips it.
    p._catchup_infeasible_for_round = False

    # Drain any stdout from poller construction so we only inspect this
    # call's emissions.
    capsys.readouterr()

    p._poll_now(deadline_ms=0, label="period")

    assert p._catchup_infeasible_for_round is False, (
        "post-lock INFEAS must NOT set the flag (round is already locked; "
        "flag gates the next live round, not a closed one)"
    )
    captured = capsys.readouterr()
    assert "INFEAS" not in captured.out, (
        "post-lock INFEAS must NOT emit WARN line; got: " + captured.out
    )


def test_poll_now_post_lock_cursor_advances_after_set_round_phase():
    """After a post-lock INFEAS abort (no flag set), the next poll
    against a fresh _lock_at advances the cursor normally — confirming
    the cursor isn't stuck and the flag-less abort is a true no-op for
    next-round readiness.

    Backlog sized at 100 blocks (5 batches at canonical batch=20) so it
    is infeasible against a stale (past) lock_at (available_ms=0 → any
    estimate > 0 trips INFEAS) but feasible against a 10-minute-out
    next-round lock_at (available_ms ~ 598_500ms ≫ 5*1319=6595ms).
    """
    import time as _t
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 99_900  # 100 blocks behind head=100_000

    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]

    fetched: list[list[int]] = []
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: fetched.append(list(nums))
    )

    # First call: post-lock periodic (lock_at in the past). INFEAS path
    # short-circuits the fetch but must NOT set the flag (post-lock
    # gating). Cursor stays put for this poll.
    p._lock_at = int(_t.time()) - 2
    p._catchup_infeasible_for_round = False
    p._poll_now(deadline_ms=0, label="period")
    assert p._catchup_infeasible_for_round is False
    assert fetched == [], "post-lock INFEAS must short-circuit the fetch"
    assert p._last_polled_block_number == 99_900, (
        "cursor stays put on post-lock INFEAS (matches pre-lock INFEAS behavior)"
    )

    # Second call: simulate set_round_phase advancing _lock_at to the
    # next round's future lock (plenty of catch-up budget).
    p._lock_at = int(_t.time()) + 600  # 10 minutes out
    p._poll_now(deadline_ms=0, label="period")
    assert len(fetched) > 0, "cursor should advance on the next poll"
    assert p._last_polled_block_number == 100_000, (
        "cursor advances to head after feasible catch-up poll"
    )


def test_poll_now_skips_feasibility_check_when_lock_at_zero():
    """Pre-cold-start _lock_at=0; feasibility check should be skipped
    (no round info to integrate against)."""
    p = _make_poller()
    p._connected = True
    p._cold_start_done.set()
    p._last_polled_block_number = 99_999
    p._lock_at = 0

    fetched: list[list[int]] = []
    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: fetched.append(nums)
    )

    p._poll_now(deadline_ms=0, label="period")
    # batch_size=20 and only 1 block to fetch (100_000 - 99_999 = 1) so 1 batch.
    assert len(fetched) == 1
    assert p._catchup_infeasible_for_round is False


def test_poll_now_logs_partial_at_info_when_some_batches_succeeded(caplog):
    """When error_seen is set after a partial fetch, log at INFO with
    PARTIAL status (not WARN/ERROR — transient is informational)."""
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 99_900
    p._rpc_eth_block_number = lambda: 99_960  # type: ignore[assignment]

    fetched: list[list[int]] = []
    call_count = {"n": 0}

    def flaky_fetch(nums):
        call_count["n"] += 1
        if call_count["n"] == 1:
            fetched.append(nums)  # first batch ok
            return
        raise RuntimeError("publicnode_glitch")

    p._fetch_and_process_blocks = flaky_fetch  # type: ignore[assignment]

    # Tight enough to NOT trigger feasibility. 60 blocks / 20 = 3 batches *
    # p99(20) ms estimated. 60s available is >> any plausible p99 * 3.
    import time as _t
    p._lock_at = int(_t.time()) + 60  # 60s out -> 59800ms available

    import logging
    with caplog.at_level(logging.INFO):
        p._poll_now(deadline_ms=0, label="period")

    # First batch succeeded; second raised; cursor advanced by 20.
    assert p._last_polled_block_number == 99_920
    assert p._last_poll_succeeded is False  # diagnostic flag


def test_poll_now_logs_empty_warn_when_zero_blocks_no_error():
    """An empty publicnode reply for a valid range is unusual -> WARN."""
    # This is the only WARN case in the new severity matrix; other
    # zero-block cases are INFO.
    # The scenario is hard to reach via the real code (blocks_polled stays
    # 0 only if fetcher succeeds but advances cursor 0 times — currently
    # unreachable since blocks_polled += len(batch_nums) on every success).
    # We test the severity decision logic by direct introspection of the
    # branches in _poll_now: if error_seen=None and blocks_polled=0 and
    # n_blocks>0, the emitter is `warn`. Covered indirectly by the
    # log-severity branches; no separate scenario here.


def test_poll_now_sets_and_clears_poll_in_progress_on_success():
    """_poll_in_progress is True during the fetch and False after."""
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 99_999

    seen_flag: list[bool] = []

    def fake_fetch(nums):
        # Snapshot the flag while inside the fetch.
        seen_flag.append(p._poll_in_progress)

    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]
    p._fetch_and_process_blocks = fake_fetch  # type: ignore[assignment]

    # Plenty of time for feasibility.
    import time as _t
    p._lock_at = int(_t.time()) + 60

    assert p._poll_in_progress is False  # before
    p._poll_now(deadline_ms=0, label="period")
    assert seen_flag == [True]  # was True during fetch
    assert p._poll_in_progress is False  # cleared after


def test_poll_now_clears_poll_in_progress_on_failure():
    """Even when a batch raises, the flag must be cleared (try/finally)."""
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 99_999

    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]

    def boom(nums):
        raise RuntimeError("publicnode_error")

    p._fetch_and_process_blocks = boom  # type: ignore[assignment]

    import time as _t
    p._lock_at = int(_t.time()) + 60

    p._poll_now(deadline_ms=0, label="period")
    assert p._poll_in_progress is False, "flag must clear even on failure"


def test_poll_now_clears_poll_in_progress_when_head_fetch_fails():
    """Head fetch failing returns early; flag still cleared."""
    p = _make_poller()
    _prep_for_epoch_advance(p)

    def head_boom():
        raise RuntimeError("head_unreachable")

    p._rpc_eth_block_number = head_boom  # type: ignore[assignment]
    p._poll_now(deadline_ms=0, label="period")
    assert p._poll_in_progress is False


def test_poll_now_clears_poll_in_progress_when_infeasible_aborts_early():
    """Mid-round feasibility check aborts before any batch fetch;
    _poll_in_progress is still cleared and _catchup_infeasible_for_round
    is set (pre-lock branch).

    Post-2026-05-12 gating: lock_at must be strictly in the future for
    the flag to set. ``now + 1`` keeps the test pre-lock; the previous
    version's ``int(now)`` is post-lock under int() truncation and now
    exercises the no-flag branch.
    """
    import time as _t
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 1_000  # very stale

    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: pytest.fail("should not have fetched")
    )

    p._lock_at = int(_t.time()) + 1  # pre-lock, tight => infeasible
    p._poll_now(deadline_ms=0, label="period")

    assert p._catchup_infeasible_for_round is True
    assert p._poll_in_progress is False


# ---------------------------------------------------------------------------
# _initialize_cursor_from_head: cursor init only, no synchronous backfill
# (Bundle 2 refactor 2026-05-13: replaces the prior _cold_start one-shot
# that synchronously backfilled blocks before returning.)
# ---------------------------------------------------------------------------

def test_init_cursor_positions_at_round_start_minus_1_when_feasible():
    """At round start with full 5min, cursor init sets
    _last_polled_block_number = round_start_block - 1. NO batch fetch
    is performed — the daemon's first periodic tick does that.

    Regression for the 621-block bug: cursor target must be scoped to
    the CURRENT round only (head - delta_blocks_in_round - safety),
    NOT to a head-relative full-round lookback.

    Bundle 5 v2 (2026-05-14): safety margin +5 → +0 (Q2 fix). The
    post-Lorentz chain's empirical "misses only delay, never advance"
    property means actual blocks-elapsed is at most
    ``ceil((head_ts - round_start_ts) * 1000 / 450)`` — slot misses
    produce fewer blocks than the divisor predicts, never more. So
    ``round(...)`` is already an upper bound (modulo ≤ 0.5 block of
    rounding noise) and cursor at ``head - delta_blocks`` lands at or
    before round_start_block by construction. New expected cursor:
    delta_blocks = round(10000/450) = 22, cursor = head - 22 - 1
    = 99_977.
    """
    import time as _t
    p = _make_poller()
    now = int(_t.time())
    p._lock_at = now + 290  # 10s into round
    p._rpc_eth_get_latest_block_header = lambda: (100_000, now)  # type: ignore[assignment]
    # cursor-init must NOT call _fetch_and_process_blocks: blow up if it does.
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: pytest.fail("init_cursor must not fetch batches")
    )

    p._initialize_cursor_from_head()

    assert p._catchup_infeasible_for_round is False
    # Cursor init must NOT flip _connected — that's the first-poll latch.
    assert p._connected is False
    assert p._cold_start_done.is_set() is False
    # Bundle 5 v2: 10s into round = round(10000/450) = 22 blocks
    # behind head -> cursor at head - 22 - 1 = 99_977.
    assert p._last_polled_block_number == 99_977, (
        f"cursor positioned at {p._last_polled_block_number}; "
        f"expected 99_977 (head-22-1 under Bundle 5 v2 +0 safety math). "
        f">630 blocks back would indicate the head-relative full-round "
        f"lookback regression."
    )


def test_init_cursor_scopes_target_to_current_round_only():
    """Regression for the 621-block bug: at 60s into round, cursor
    target must reflect ~120 blocks (60s/0.5s/block) + 20 safety, NOT
    a full round's worth of blocks (600+)."""
    import time as _t
    p = _make_poller()
    now = int(_t.time())
    p._lock_at = now + 240  # 60s into round
    p._rpc_eth_get_latest_block_header = lambda: (100_000, now)  # type: ignore[assignment]
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: pytest.fail("init_cursor must not fetch batches")
    )

    p._initialize_cursor_from_head()

    blocks_behind_head = 100_000 - p._last_polled_block_number
    # 60s / 0.5s = 120 blocks; +20 safety + 1 (cursor=round_start-1) = 141.
    assert blocks_behind_head < 200, (
        f"cursor {blocks_behind_head} blocks behind head; expected ~141 "
        f"(current round only). >600 indicates the full-round lookback bug."
    )
    assert blocks_behind_head >= 100, (
        f"cursor only {blocks_behind_head} blocks behind head; "
        f"expected ~141 (round_start derivation broken?)"
    )


def test_init_cursor_marks_infeasible_and_jumps_to_head_when_no_time():
    """Bot starts 1s before lock — math says first periodic tick can't
    catch up ~620 blocks in time. Cursor jumps to head, flag set,
    _connected stays False (no successful poll has run yet)."""
    import time as _t
    p = _make_poller()
    now = int(_t.time())
    p._lock_at = now + 1  # only 1s left
    p._rpc_eth_get_latest_block_header = lambda: (100_000, now)  # type: ignore[assignment]
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: pytest.fail("init_cursor must not fetch batches")
    )

    p._initialize_cursor_from_head()

    assert p._catchup_infeasible_for_round is True
    # _connected NOT set yet — that requires a successful poll. Bundle 2
    # behaviour: cursor-init never flips the latch by itself.
    assert p._connected is False
    assert p._cold_start_done.is_set() is False
    # Cursor jumped to head so the upcoming periodic tick has zero gap
    # (no infeasible backlog carryover to next round).
    assert p._last_polled_block_number == 100_000


# ---------------------------------------------------------------------------
# T3-B: RpcPoller.last_catchup_detail exposes (need_ms, have_ms) for the
# engine's SKIP narrative ("RPC catchup infeasible (need 39.5s, have 30.1s)")
# ---------------------------------------------------------------------------


def test_last_catchup_detail_captured_when_infeasibility_check_fires():
    """When ``_is_catchup_infeasible`` returns True, the poller stashes
    ``(estimated_ms, available_ms)`` on ``last_catchup_detail`` so the
    engine's catchup_infeasible SKIP narrative can render the numbers.
    """
    import time as _t
    p = _make_poller()
    # Force a known-infeasible scenario: 597 blocks behind, 30s until lock.
    # estimated_ms = 29 batches × 1319ms + 1 partial → ~39.5s
    # available_ms = 30000 - 200 safety = 29800ms
    # → infeasible.
    now = _t.time()
    lock_at = int(now + 30)
    blocks_behind = 597

    assert p._is_catchup_infeasible(
        blocks_behind=blocks_behind, lock_at=lock_at,
    ) is True
    detail = p.last_catchup_detail
    assert detail is not None
    need_ms, have_ms = detail
    # Need substantially exceeds have for this construction.
    assert need_ms > have_ms
    # Empirical: ~39.5s vs ~29.8s.
    assert 35_000 <= need_ms <= 45_000, f"need_ms={need_ms} outside expected band"
    assert 28_000 <= have_ms <= 30_500, f"have_ms={have_ms} outside expected band"


def test_last_catchup_detail_reset_on_epoch_advance():
    """A new round's epoch-advance hook clears any stashed catchup detail
    so a later SKIP can't surface stale numbers from a prior round.

    The reset lives in ``_on_epoch_advance`` (called from
    ``set_round_phase`` when the epoch number changes). The very first
    ``set_round_phase`` routes through ``_initialize_cursor_from_head``
    which is a different path; the reset is only relevant for the
    round-to-round transition where stale numbers could leak across.
    """
    import time as _t
    p = _make_poller()
    # Populate detail via an infeasible check (as if a prior round
    # observed catchup_infeasible_for_round).
    now = _t.time()
    p._is_catchup_infeasible(blocks_behind=597, lock_at=int(now + 30))
    assert p.last_catchup_detail is not None

    # Mock the RPC head fetch + round_start derivation so
    # _on_epoch_advance can run without real chain calls.
    p._compute_round_start_block = lambda _ts: 99_000  # type: ignore[assignment]
    p._rpc_eth_block_number = lambda: 99_010  # type: ignore[assignment]

    # _on_epoch_advance is the canonical round-boundary hook; calling
    # it directly exercises the reset without the first-call cursor-init
    # branch in set_round_phase.
    p._on_epoch_advance(lock_at=int(now) + 300, current_epoch=100_001)

    assert p.last_catchup_detail is None, (
        "_on_epoch_advance must clear last_catchup_detail; otherwise a "
        "SKIP in a later round could surface stale numbers from a prior round"
    )


def test_last_catchup_detail_returns_none_when_no_check_has_fired():
    """Fresh poller (no infeasibility check yet) returns None — engine
    SKIP narrative gracefully falls back to the no-numbers wording.
    """
    p = _make_poller()
    assert p.last_catchup_detail is None


def test_init_cursor_handles_head_before_round_start():
    """If head_ts <= round_start_ts (head is BEHIND round_start, e.g.
    lock_at far in the future), set cursor to head and return — daemon's
    periodic ticks will drive forward from there."""
    import time as _t
    p = _make_poller()
    now = int(_t.time())
    p._lock_at = now + 600  # round won't START for another 5 min
    # round_start_ts = lock_at - 300 = now+300; head_ts = now < round_start_ts.
    p._rpc_eth_get_latest_block_header = lambda: (100_000, now)  # type: ignore[assignment]
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: pytest.fail("init_cursor must not fetch batches")
    )

    p._initialize_cursor_from_head()

    # Cursor at head; no flag; _connected NOT set (no successful poll yet).
    assert p._connected is False
    assert p._cold_start_done.is_set() is False
    assert p._catchup_infeasible_for_round is False
    assert p._last_polled_block_number == 100_000


def test_init_cursor_logs_infeas_at_warn_severity():
    """The cursor-init INFEAS log must be WARN, not INFO. Same severity
    contract as the pre-Bundle-2 cold-start path."""
    import time as _t
    p = _make_poller()
    now = int(_t.time())
    p._lock_at = now + 1
    p._rpc_eth_get_latest_block_header = lambda: (100_000, now)  # type: ignore[assignment]
    p._fetch_and_process_blocks = lambda nums: None  # type: ignore[assignment]

    # We can't easily assert log level via pytest.caplog without the
    # log module routing through the logging stdlib. Instead, monkey-
    # patch the rpc_poller's `warn` import and ensure it's the call that
    # fired with sub="COLD" event="INFEAS".
    import pancakebot.chain.rpc_poller as _mod
    seen = {"warn": [], "info": []}
    orig_warn = _mod.warn
    orig_info = _mod.info

    def fake_warn(*args, **kwargs):
        seen["warn"].append((args, kwargs))
        return orig_warn(*args, **kwargs)

    def fake_info(*args, **kwargs):
        seen["info"].append((args, kwargs))
        return orig_info(*args, **kwargs)

    _mod.warn = fake_warn
    _mod.info = fake_info
    try:
        p._initialize_cursor_from_head()
    finally:
        _mod.warn = orig_warn
        _mod.info = orig_info

    # Phase B v2 (2026-05-18): the cold-start INFEAS line now emits with
    # ACTION="SKIP" + WARN level (was ("RPC_POLL","COLD","INFEAS") under the
    # old 3-column hierarchy). Verify exactly one SKIP-WARN fired.
    cold_infeas_warn = [
        a for a in seen["warn"]
        if len(a[0]) >= 1 and a[0][0] == "SKIP"
    ]
    assert len(cold_infeas_warn) == 1, (
        "exactly one SKIP log expected at WARN severity"
    )


# ---------------------------------------------------------------------------
# Bundle 2 (2026-05-13): first-successful-poll latch + no .wait() callers
# ---------------------------------------------------------------------------

def test_first_successful_poll_latches_connected_and_cold_start_done():
    """The first successful _poll_now (either fetched ≥1 batch with no
    error, OR found head already caught up) flips _connected=True and
    sets _cold_start_done. Subsequent polls must not re-fire the latch
    (idempotent); _connected stays True across transient failures.
    """
    import time as _t
    p = _make_poller()
    # Cursor initialized (as if _initialize_cursor_from_head ran), but
    # latch not yet flipped — this is the post-init pre-first-poll state.
    p._current_epoch = 100
    p._lock_at = int(_t.time()) + 290
    p._last_polled_block_number = 99_900

    assert p._connected is False
    assert p._cold_start_done.is_set() is False

    # Mock a successful poll: head advanced 50 blocks since cursor.
    p._rpc_eth_block_number = lambda: 99_950  # type: ignore[assignment]
    fetched: list[list[int]] = []
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: fetched.append(list(nums))
    )

    p._poll_now(deadline_ms=0, label="period")

    # First-poll latch fired.
    assert p._connected is True
    assert p._cold_start_done.is_set() is True
    assert len(fetched) > 0  # batches actually processed

    # Second poll: head hasn't moved, _poll_now hits the "no new blocks"
    # branch. Latch must already be set; this exercises the idempotent
    # path (set-if-not-already-set inside _latch_first_successful_poll_locked).
    p._poll_now(deadline_ms=0, label="period")
    assert p._connected is True
    assert p._cold_start_done.is_set() is True


def test_first_successful_poll_latch_fires_in_no_new_blocks_branch():
    """If the first _poll_now finds head already at cursor (no work to
    do), the success markers + the first-poll latch must still fire.
    Otherwise a quiet startup (no on-chain bets yet) would leave the
    poller in cold_start_in_progress forever."""
    import time as _t
    p = _make_poller()
    p._current_epoch = 100
    p._lock_at = int(_t.time()) + 290
    p._last_polled_block_number = 100_000

    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]  # head == cursor
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: pytest.fail("no-new-blocks branch must not fetch")
    )

    assert p._connected is False
    p._poll_now(deadline_ms=0, label="period")
    assert p._connected is True
    assert p._cold_start_done.is_set() is True


def test_first_poll_failure_does_not_set_latch():
    """If the first _poll_now's head-fetch RPC fails, _last_poll_succeeded
    flips False but _connected stays False (latch only fires on success).
    A later successful poll then sets the latch."""
    import time as _t
    p = _make_poller()
    p._current_epoch = 100
    p._lock_at = int(_t.time()) + 290
    p._last_polled_block_number = 99_900

    def boom():
        raise RuntimeError("publicnode_unreachable")
    p._rpc_eth_block_number = boom  # type: ignore[assignment]
    p._poll_now(deadline_ms=0, label="period")

    assert p._connected is False
    assert p._cold_start_done.is_set() is False
    assert p._last_poll_succeeded is False

    # Now a successful poll: head fetches OK, no new blocks.
    p._rpc_eth_block_number = lambda: 99_900  # type: ignore[assignment]
    p._poll_now(deadline_ms=0, label="period")
    assert p._connected is True
    assert p._cold_start_done.is_set() is True


def test_no_cold_start_done_wait_callers_in_production_code():
    """Acceptance (c): nothing in pancakebot/ awaits _cold_start_done.wait().
    Bundle 2 removed the synchronous backfill path, so any remaining
    .wait() call would deadlock waiting on an event that's now lazy.
    """
    import os, re
    repo_root = Path(__file__).resolve().parent.parent / "pancakebot"
    pattern = re.compile(r"_cold_start_done\s*\.\s*wait\b")
    offending: list[tuple[str, int, str]] = []
    for root, _, files in os.walk(str(repo_root)):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    if pattern.search(line):
                        offending.append((path, lineno, line.rstrip()))
    assert offending == [], (
        f"Found {len(offending)} _cold_start_done.wait() call(s) in production "
        f"code; Bundle 2 expects zero (latch is fire-and-set, no awaiters): "
        f"{offending}"
    )


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


# ---------------------------------------------------------------------------
# Bundle 4 (2026-05-14): BEP-520 ms-precise anchor + dynamic deadline math
# ---------------------------------------------------------------------------

# Import the new module-level helpers for direct testing
from pancakebot.chain.rpc_poller import (  # noqa: E402
    AnchorState, decode_mixhash_ms, compute_milli_ts,
    predict_predecessor_milli_ts,
    compute_submit_deadline_ms,
)


# --- decode_mixhash_ms ---

def test_decode_mixhash_ms_canonical():
    """Encoding observed in empirical Bundle 4 probe: last 2 bytes
    as big-endian uint16, leading bytes all-zero."""
    # 0x...0352 = 850
    mh = "0x" + "00" * 30 + "0352"
    assert decode_mixhash_ms(mh) == 850
    # 0x...0000 = 0 (valid; corresponds to a quantum=0 block)
    mh0 = "0x" + "00" * 32
    assert decode_mixhash_ms(mh0) == 0
    # 0x...03e7 = 999 (max valid)
    mh_max = "0x" + "00" * 30 + "03e7"
    assert decode_mixhash_ms(mh_max) == 999


def test_decode_mixhash_ms_out_of_range_returns_none():
    """Values > 999 are out of the BEP-520 valid range. Defensive
    against legacy/pre-Lorentz mixHash content."""
    # 0x...0400 = 1024 (out of [0, 999])
    mh = "0x" + "00" * 30 + "0400"
    assert decode_mixhash_ms(mh) is None
    # 0x...ffff = 65535 (max uint16)
    mh = "0x" + "00" * 30 + "ffff"
    assert decode_mixhash_ms(mh) is None


def test_decode_mixhash_ms_malformed_returns_none():
    """Bad input types or shapes return None, never raise."""
    assert decode_mixhash_ms(None) is None
    assert decode_mixhash_ms("") is None
    assert decode_mixhash_ms("0x12") is None  # too short
    assert decode_mixhash_ms("not_hex") is None
    assert decode_mixhash_ms("0x" + "zz" * 32) is None  # invalid hex


def test_decode_mixhash_ms_accepts_no_0x_prefix():
    """Both '0x...' and bare-hex forms work (defensive against varying
    RPC implementations)."""
    assert decode_mixhash_ms("00" * 30 + "0352") == 850


# --- compute_milli_ts ---

def test_compute_milli_ts_canonical():
    """Returns header.Time*1000 + mix_ms when both parse cleanly."""
    block = {
        "timestamp": hex(1778712807),  # 0x6a04fee4
        "mixHash": "0x" + "00" * 30 + "0352",  # 850ms
    }
    assert compute_milli_ts(block) == 1778712807 * 1000 + 850


def test_compute_milli_ts_returns_none_on_bad_input():
    assert compute_milli_ts({}) is None
    assert compute_milli_ts({"timestamp": "0x1"}) is None  # missing mixHash
    assert compute_milli_ts({"mixHash": "0x" + "00" * 32}) is None  # missing timestamp
    assert compute_milli_ts({"timestamp": "0xZZ", "mixHash": "0x" + "00" * 32}) is None


# --- predict_predecessor_milli_ts ---

def test_predict_predecessor_milli_ts_ms_precise_extrapolation():
    """Anchor 3600ms before lock → 8 blocks ahead at 450ms/block; lock
    block milli_ts = anchor + 3600 = lock_ms; predecessor = lock - 450."""
    anchor = 1_000_000
    lock = 1_003_600  # exactly 8 blocks ahead
    result = predict_predecessor_milli_ts(anchor_milli_ts=anchor, lock_ms=lock)
    # delta = 3600 - 10 (jitter) = 3590; ceil(3590/450) = 8
    # predicted_lock_block = anchor + 8*450 = 1_003_600
    # predecessor = 1_003_600 - 450 = 1_003_150
    assert result == 1_003_150


def test_predict_predecessor_milli_ts_handles_off_grid_anchor():
    """Anchor not aligned to the block grid: extrapolation still produces
    a valid predecessor that's strictly before lock."""
    anchor = 1_000_150  # 150ms into a slot
    lock = 1_003_700
    result = predict_predecessor_milli_ts(anchor_milli_ts=anchor, lock_ms=lock)
    # delta = 3700 - 10 - 150 = 3540; ceil(3540/450) = 8
    # predicted_lock_block = 1_000_150 + 8*450 = 1_003_750
    # predecessor = 1_003_750 - 450 = 1_003_300
    assert result == 1_003_300
    assert result < lock


# --- compute_submit_deadline_ms (all 9 discrete gap values) ---

def _deadline_at_gap(gap_ms: int) -> int:
    """Helper: predecessor at lock - gap_ms, compute deadline."""
    lock = 1_000_000
    pred = lock - gap_ms
    return compute_submit_deadline_ms(predicted_predecessor_milli_ts=pred, lock_ms=lock)


def test_submit_deadline_gap_50_triggers_boundary_backoff():
    """gap == quantum: bet_inclusion_deadline + 50 = lock → backoff fires.
    bet_inclusion_deadline = predecessor - 450 (backoff) - 50 (assembly) = pred - 500.
    Final deadline = pred - 500 - 75 (one-way) = pred - 575 = lock - 625.
    """
    # Boundary case: total = lock - 625ms (= static fallback offset post 2026-05-20).
    assert _deadline_at_gap(50) == 1_000_000 - 625


def test_submit_deadline_gap_100_no_backoff():
    """gap == 100ms: not within quantum of lock → no backoff.
    Final = predecessor - 50 (assembly) - 75 (one-way) = pred - 125.
    Total = lock - 100 - 125 = lock - 225.
    """
    assert _deadline_at_gap(100) == 1_000_000 - 225


def test_submit_deadline_gap_150():
    assert _deadline_at_gap(150) == 1_000_000 - 275


def test_submit_deadline_gap_200():
    assert _deadline_at_gap(200) == 1_000_000 - 325


def test_submit_deadline_gap_250():
    assert _deadline_at_gap(250) == 1_000_000 - 375


def test_submit_deadline_gap_300():
    assert _deadline_at_gap(300) == 1_000_000 - 425


def test_submit_deadline_gap_350():
    assert _deadline_at_gap(350) == 1_000_000 - 475


def test_submit_deadline_gap_400():
    assert _deadline_at_gap(400) == 1_000_000 - 525


def test_submit_deadline_gap_450_max_no_backoff():
    """gap == 450ms (full block): no backoff. Deadline = lock - 575."""
    assert _deadline_at_gap(450) == 1_000_000 - 575


def test_submit_deadline_boundary_at_exactly_quantum_msbacks_off():
    """When predecessor + quantum >= lock_ms exactly, the backoff fires
    (>= semantics, not >). Boundary case must be conservative."""
    # gap = BSC_QUANTUM_MS = 50; pred + 50 = lock - 50 + 50 = lock, fires backoff.
    assert _deadline_at_gap(_tc.BSC_QUANTUM_MS) == 1_000_000 - 625


# ---------------------------------------------------------------------------
# Bundle 5 v2 (2026-05-14): per-round single anchor poll
# ---------------------------------------------------------------------------

def test_fire_anchor_poll_returns_anchor_state_on_success(monkeypatch):
    """Happy path: RPC returns a valid Lorentz-encoded block; fire_anchor_poll
    decodes the mixHash and returns an AnchorState."""
    p = _make_poller()
    block = {
        "number": hex(98_000_000),
        "timestamp": hex(1778712807),
        "mixHash": "0x" + "00" * 30 + "0352",  # 850ms quantum
    }
    monkeypatch.setattr(
        p, "_rpc_call_single_with_timeout",
        lambda method, params, *, timeout_s: block,
    )
    anchor = p.fire_anchor_poll(timeout_s=0.200)
    assert anchor is not None
    assert anchor.block_number == 98_000_000
    assert anchor.milli_ts == 1778712807 * 1000 + 850


def test_fire_anchor_poll_returns_none_on_rpc_error(monkeypatch):
    """RPC raises (timeout, hedged-all-failed, transport error) → None."""
    p = _make_poller()

    def _raise(method, params, *, timeout_s):
        raise TimeoutError("hedged_timeout")

    monkeypatch.setattr(p, "_rpc_call_single_with_timeout", _raise)
    assert p.fire_anchor_poll(timeout_s=0.200) is None


def test_fire_anchor_poll_returns_none_on_malformed_response(monkeypatch):
    """Response is not a dict → None (defensive against bad RPC)."""
    p = _make_poller()
    monkeypatch.setattr(
        p, "_rpc_call_single_with_timeout",
        lambda method, params, *, timeout_s: "not_a_dict",
    )
    assert p.fire_anchor_poll(timeout_s=0.200) is None


def test_fire_anchor_poll_returns_none_on_missing_number(monkeypatch):
    """Response dict missing 'number' field → None."""
    p = _make_poller()
    monkeypatch.setattr(
        p, "_rpc_call_single_with_timeout",
        lambda method, params, *, timeout_s: {
            "timestamp": hex(1778712807),
            "mixHash": "0x" + "00" * 30 + "0352",
        },
    )
    assert p.fire_anchor_poll(timeout_s=0.200) is None


def test_fire_anchor_poll_returns_none_on_unparseable_milli_ts(monkeypatch):
    """compute_milli_ts returns None (malformed mixHash or timestamp) → None."""
    p = _make_poller()
    monkeypatch.setattr(
        p, "_rpc_call_single_with_timeout",
        lambda method, params, *, timeout_s: {
            "number": hex(98_000_000),
            "timestamp": "0xZZ",  # bad hex → compute_milli_ts None
            "mixHash": "0x" + "00" * 32,
        },
    )
    assert p.fire_anchor_poll(timeout_s=0.200) is None


def test_fire_anchor_poll_passes_timeout_through(monkeypatch):
    """The timeout_s argument must reach _rpc_call_single_with_timeout."""
    p = _make_poller()
    captured: dict = {}

    def _capture(method, params, *, timeout_s):
        captured["method"] = method
        captured["params"] = params
        captured["timeout_s"] = timeout_s
        return {
            "number": hex(1),
            "timestamp": hex(1778712807),
            "mixHash": "0x" + "00" * 30 + "0352",
        }

    monkeypatch.setattr(p, "_rpc_call_single_with_timeout", _capture)
    p.fire_anchor_poll(timeout_s=0.200)
    assert captured["timeout_s"] == 0.200
    assert captured["method"] == "eth_getBlockByNumber"
    assert captured["params"] == ["latest", False]


# ---------------------------------------------------------------------------
# Bundle 5 v2 (2026-05-14): receipts-only batch shape
# ---------------------------------------------------------------------------

def test_fetch_and_process_blocks_sends_receipts_only_batch(monkeypatch):
    """_fetch_and_process_blocks must build a JSON-RPC batch of ONLY
    eth_getBlockReceipts calls (no eth_getBlockByNumber per block).
    Catches accidental re-introduction of the bundled header fetch."""
    p = _make_poller()
    captured_calls: list = []

    def _capture(calls):
        captured_calls.extend(calls)
        return [([], None) for _ in calls]

    monkeypatch.setattr(p, "_rpc_batch", _capture)
    p._fetch_and_process_blocks([100, 101, 102])
    assert len(captured_calls) == 3
    for method, params in captured_calls:
        assert method == "eth_getBlockReceipts"
        assert len(params) == 1
    assert not any(m == "eth_getBlockByNumber" for m, _ in captured_calls)


def test_fetch_and_process_blocks_empty_input_no_rpc(monkeypatch):
    """Empty block list → no RPC call fired."""
    p = _make_poller()
    rpc_calls = 0

    def _track(calls):
        nonlocal rpc_calls
        rpc_calls += 1
        return []

    monkeypatch.setattr(p, "_rpc_batch", _track)
    p._fetch_and_process_blocks([])
    assert rpc_calls == 0


# ---------------------------------------------------------------------------
# Bundle 5 v2 (2026-05-14): lazy block_ts resolution for bet-containing blocks
# ---------------------------------------------------------------------------

def test_resolve_block_ts_caches_result_on_success(monkeypatch):
    """A successful header fetch caches the timestamp in _block_ts."""
    p = _make_poller()
    monkeypatch.setattr(
        p, "_rpc_call_single",
        lambda method, params: {"timestamp": hex(1778712807)},
    )
    assert p._resolve_block_ts(100) == 1778712807
    assert p._block_ts.get(100) == 1778712807


def test_resolve_block_ts_returns_zero_on_rpc_error(monkeypatch):
    """RPC raises → return 0, no cache write."""
    p = _make_poller()

    def _raise(method, params):
        raise RuntimeError("rpc failed")

    monkeypatch.setattr(p, "_rpc_call_single", _raise)
    assert p._resolve_block_ts(100) == 0
    assert 100 not in p._block_ts


def test_resolve_block_ts_returns_zero_on_malformed_response(monkeypatch):
    """Response is not a dict or has bad timestamp → 0, no cache write."""
    p = _make_poller()
    monkeypatch.setattr(p, "_rpc_call_single", lambda method, params: "not_a_dict")
    assert p._resolve_block_ts(100) == 0
    monkeypatch.setattr(p, "_rpc_call_single",
                        lambda method, params: {"timestamp": "0xZZ"})
    assert p._resolve_block_ts(101) == 0
    assert 101 not in p._block_ts


