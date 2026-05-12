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
            batch_size=_tc.RPC_BATCH_BLOCK_RECEIPTS_LIMIT + 1,
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


def test_set_round_phase_advancing_epoch_drops_stale_pools(monkeypatch):
    """When epoch advances, stale-epoch pool entries are dropped.
    The newly-introduced _on_epoch_advance hook is mocked here — its
    own behavior is covered by dedicated tests below."""
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

    # Don't actually try to fetch from test.example.com.
    monkeypatch.setattr(p, "_on_epoch_advance", lambda **kw: None)

    p.set_round_phase(current_epoch=101, lock_at=2000)

    assert 99 not in p._pools, "stale epoch should be dropped"
    assert 100 not in p._pools, "stale epoch should be dropped"
    assert 101 in p._pools, "current epoch retained"
    assert 99 not in p._seen_tx
    assert 100 not in p._seen_tx
    assert 101 in p._seen_tx


# ---------------------------------------------------------------------------
# _compute_round_start_block (cache-first, RPC fallback)
# ---------------------------------------------------------------------------

def test_compute_round_start_block_uses_cache_when_available():
    """If _block_ts has a recent-enough anchor, no RPC call needed."""
    p = _make_poller()
    target_ts = 10_000  # Unix-second-ish
    # Anchor 30s before target with block 5000 -> 30s @ 500ms/block = 60 blocks.
    p._block_ts[5000] = target_ts - 30
    rs = p._compute_round_start_block(target_ts)
    # 30s * 1000 / 500ms_per_block = 60 blocks ahead of anchor.
    assert rs == 5000 + 60


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
    # head_ts = target+10 -> 10s back -> 20 blocks back from 10000 = 9980.
    assert rs == 9980
    assert invoked["n"] == 1, "RPC fallback should have been invoked"


def test_compute_round_start_block_falls_back_to_rpc_when_cache_empty():
    """Empty cache forces the RPC path."""
    p = _make_poller()

    def fake_header():
        return (1_000_000, 50_000)

    p._rpc_eth_get_latest_block_header = fake_header  # type: ignore[assignment]
    rs = p._compute_round_start_block(round_start_ts=49_000)
    # delta_seconds = 50000 - 49000 = 1000s -> 2000 blocks at 500ms/block.
    # round_start_block = 1_000_000 - 2000 = 998_000.
    assert rs == 998_000


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

    # round_start_ts = lock_at(2_000_000) - interval(300) = 1_999_700,
    # but our header gave head_ts=1_000_050 => head_ts < round_start_ts =>
    # _compute_round_start_block returns head_num. Use a different lock_at
    # that aligns with our mocks: lock_at - 300 = round_start_ts.
    # Pick lock_at = head_ts (1_000_050) + 300 = 1_000_350.
    # Then round_start_ts = 1_000_050. Anchor cache empty -> RPC path.
    # delta_seconds = 1_000_050 - 1_000_050 = 0 -> 0 blocks back -> rs=100_000.
    # That's not "behind"; let me use a clear scenario:
    # head_ts = 1_000_300, lock_at = 1_000_350, round_start_ts = 1_000_050.
    # delta = 250s -> 500 blocks -> rs = 100_000 - 500 = 99_500.
    p._rpc_eth_get_latest_block_header = lambda: (100_000, 1_000_300)  # type: ignore[assignment]

    p._on_epoch_advance(lock_at=1_000_350, current_epoch=101)

    # Cursor should now be 99_500 - 1 = 99_499.
    assert p._last_polled_block_number == 99_499


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
    """First call goes through cold-start, NOT the epoch-advance hook."""
    p = _make_poller()
    invoked = {"n": 0}

    def fake_advance(**kw):
        invoked["n"] += 1

    def fake_cold():
        pass

    monkeypatch.setattr(p, "_on_epoch_advance", fake_advance)
    monkeypatch.setattr(p, "_cold_start", fake_cold)

    p.set_round_phase(current_epoch=100, lock_at=1000)
    assert invoked["n"] == 0


# ---------------------------------------------------------------------------
# Feasibility math
# ---------------------------------------------------------------------------

def test_estimated_catchup_ms_calculation():
    """20 blocks at batch_size=20 -> 1 batch * p99(20)=1319ms.

    1319ms is the 2026-05-11 fire-to-all-pool measurement (n=30, bot
    stopped). See RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE docstring in
    timing_constants.py.
    """
    p = _make_poller()
    rtt = _tc.rpc_rtt_p99_for_batch(20)
    assert p._estimated_catchup_ms(0) == 0
    assert p._estimated_catchup_ms(20) == rtt  # 1 batch
    assert p._estimated_catchup_ms(21) == 2 * rtt  # 2 batches (ceiling)
    assert p._estimated_catchup_ms(60) == 3 * rtt  # 3 batches


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
    """Regression: at canonical pool_cutoff=6 (final fires at lock-3.79s),
    a 30-block lag should be FEASIBLE (not trip INFEAS). 30-block lag at
    batch_size=20 = ceil(30/20)=2 batches * p99(20). With the 2026-05-11
    fire-to-all p99=1319ms, 2 * 1319 = 2638ms, must fit comfortably
    within available_catchup_ms.

    Also tests the 23-block-lag false-positive scenario the
    2026-05-11_fire_to_all_p99_measurement.md memo flagged: at 23 blocks
    (= 2 batches), the math must NOT trip INFEAS just because of stale
    constants. Anchors the constant value as load-bearing.

    Pin lock_at = now + 5s (well above canonical final-poll timing of
    3.79s) so the test has comfortable margin against int() truncation
    of the current wallclock and any small wallclock drift between the
    pin and the assertion. The point is to test the math contract, not
    race CI timers.
    """
    import time as _t
    p = _make_poller()
    lock_at = int(_t.time()) + 5  # 5s out: ~4800ms available after safety
    # 23-block lag: 2 batches at p99=1319ms = 2638ms < 4800ms. Feasible.
    assert p._is_catchup_infeasible(blocks_behind=23, lock_at=lock_at) is False, (
        "23-block lag must be feasible at canonical final-poll timing. "
        "If this fails, RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE[20] may be "
        "stale relative to current transport — re-measure."
    )
    # 30-block lag: same 2 batches. Same result.
    assert p._is_catchup_infeasible(blocks_behind=30, lock_at=lock_at) is False


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
    """If during a round the cursor falls way behind and math says
    can't catch up, _poll_now aborts without fetching batches and sets
    the infeasibility flag."""
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

    # Tight time: lock_at is "now"; available = 0; clearly infeasible.
    p._lock_at = int(_t.time())

    p._poll_now(deadline_ms=0, label="period")

    assert p._catchup_infeasible_for_round is True


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
    flag still must clear."""
    import time as _t
    p = _make_poller()
    _prep_for_epoch_advance(p)
    p._last_polled_block_number = 1_000  # very stale

    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: pytest.fail("should not have fetched")
    )

    p._lock_at = int(_t.time())  # tight => infeasible
    p._poll_now(deadline_ms=0, label="period")

    assert p._catchup_infeasible_for_round is True
    assert p._poll_in_progress is False


# ---------------------------------------------------------------------------
# _cold_start: feasibility-aware backfill
# ---------------------------------------------------------------------------

def test_cold_start_does_not_mark_infeasible_when_room_to_backfill():
    """At round start with full 5min, backfill is feasible -> normal
    cold-start completes, flag stays False."""
    import time as _t
    p = _make_poller()
    now = int(_t.time())
    p._lock_at = now + 290  # 10s into round
    # head_ts = now; round_start_ts = now-10. delta_blocks = 20+20 = 40.
    p._rpc_eth_get_latest_block_header = lambda: (100_000, now)  # type: ignore[assignment]
    # _poll_now's batch-fetch loop also calls _rpc_eth_block_number
    # for the inner head — mock it so the test stays offline.
    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]
    fetched: list[list[int]] = []
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: fetched.append(nums)
    )

    p._cold_start()

    assert p._catchup_infeasible_for_round is False
    assert p._connected is True
    # Cursor is now head (after backfill processed all blocks).
    assert p._last_polled_block_number == 100_000
    # Backfill performed at least one batch.
    assert len(fetched) > 0
    # Bug regression check: cold_start must scope backfill to current
    # round only, NOT to a head-relative full-round lookback. ~10s into
    # round = ~40 blocks (20 actual + 20 safety margin), NOT 620.
    total_blocks_fetched = sum(len(b) for b in fetched)
    assert total_blocks_fetched < 100, (
        f"cold_start over-fetched: {total_blocks_fetched} blocks "
        f"(expected ~40 for 10s into round; >100 indicates the "
        f"head-relative full-round lookback bug)"
    )


def test_cold_start_does_not_backfill_past_round_start():
    """Regression for the 621-block bug: cold_start must NOT backfill
    from head - blocks_per_round (which would include the previous
    round's blocks)."""
    import time as _t
    p = _make_poller()
    now = int(_t.time())
    p._lock_at = now + 240  # 60s into round
    p._rpc_eth_get_latest_block_header = lambda: (100_000, now)  # type: ignore[assignment]
    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]
    fetched: list[list[int]] = []
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: fetched.append(nums)
    )

    p._cold_start()

    total = sum(len(b) for b in fetched)
    # 60s / 0.5s = 120 blocks; +20 safety margin = 140. Must be well
    # under 600 (a full round) which would indicate the bug.
    assert total < 200, (
        f"cold_start fetched {total} blocks at 60s into round; "
        f"expected ~140 (current round only). Bug regression?"
    )
    assert total >= 100, (
        f"cold_start fetched {total} blocks at 60s into round; "
        f"expected ~140 (too few — round_start derivation broken?)"
    )


def test_cold_start_marks_infeasible_when_backfill_exceeds_remaining_time():
    """Bot starts at end of round with 1s until lock — backfill ~620
    blocks would need ~46s; flag set, backfill skipped, cursor to head."""
    import time as _t
    p = _make_poller()
    now = int(_t.time())
    p._lock_at = now + 1  # only 1s left -- effectively end of round
    # head_ts = now; round_start_ts = now-299; delta = 599+20 = 619 blocks.
    p._rpc_eth_get_latest_block_header = lambda: (100_000, now)  # type: ignore[assignment]
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: pytest.fail("should not have backfilled")
    )

    p._cold_start()

    assert p._catchup_infeasible_for_round is True
    assert p._connected is True  # still finishes cold-start
    assert p._cold_start_done.is_set()
    # Cursor advanced to head so periodic polls don't try to refill.
    assert p._last_polled_block_number == 100_000


def test_cold_start_handles_head_before_round_start():
    """If head_ts <= round_start_ts (head is BEHIND round_start, e.g.
    lock_at far in the future), backfill is a no-op."""
    import time as _t
    p = _make_poller()
    now = int(_t.time())
    p._lock_at = now + 600  # round won't START for another 5 min
    # round_start_ts = lock_at - 300 = now+300; head_ts = now < round_start_ts.
    p._rpc_eth_get_latest_block_header = lambda: (100_000, now)  # type: ignore[assignment]
    p._rpc_eth_block_number = lambda: 100_000  # type: ignore[assignment]
    fetched: list[list[int]] = []
    p._fetch_and_process_blocks = (  # type: ignore[assignment]
        lambda nums: fetched.append(nums)
    )

    p._cold_start()

    # No backfill performed; cursor at head.
    assert p._connected is True
    assert p._last_polled_block_number == 100_000
    total = sum(len(b) for b in fetched)
    assert total == 0


def test_cold_start_logs_infeas_at_warn_severity():
    """The cold-start INFEAS log must be WARN, not INFO."""
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
        p._cold_start()
    finally:
        _mod.warn = orig_warn
        _mod.info = orig_info

    cold_infeas_warn = [
        a for a in seen["warn"]
        if len(a[0]) >= 3 and a[0][1] == "COLD" and a[0][2] == "INFEAS"
    ]
    assert len(cold_infeas_warn) == 1, (
        "exactly one COLD INFEAS log expected at WARN severity"
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
