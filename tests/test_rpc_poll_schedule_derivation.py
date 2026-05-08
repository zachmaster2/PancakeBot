"""Code-invariant tests for the RPC-poll wake schedule derivation.

The derivation chain in pancakebot/config.py computes
final_rpc_poll_wakeup_offset_ms, ramp_poll_2_wakeup_offset_ms, and
ramp_poll_1_wakeup_offset_ms from pool_cutoff_seconds + the
empirical timing constants. These tests pin invariants the formula
must satisfy across the canonical pool_cutoff range.

What lives HERE vs in config.py runtime checks: runtime checks
validate USER CONFIG (pool_cutoff_seconds the user picked). These
tests validate CODE CORRECTNESS (the formula always produces
sensible orderings for any reasonable pool_cutoff).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot import timing_constants as _tc  # noqa: E402


def _derive_schedule(pool_cutoff_seconds: int) -> dict[str, int]:
    """Derive the four lock-relative offsets (ms) from pool_cutoff
    using the exact same formula as pancakebot/config.py."""
    bet_submit_deadline_offset_ms = (
        _tc.BSC_BET_SUBMIT_RTT_P95_MS
        + _tc.BSC_BLOCK_TIME_MS
        + _tc.BET_SUBMIT_SAFETY_BUFFER_MS
    )
    critical_path_wakeup_offset_ms = (
        bet_submit_deadline_offset_ms
        + _tc.OKX_KLINE_FETCH_RTT_P95_MS
        + _tc.SIGNAL_COMPUTE_TIME_MS
        + _tc.POOL_READ_TIME_MS
    )
    final_rpc_poll_wakeup_offset_ms = (
        pool_cutoff_seconds * 1000
        - _tc.BSC_BLOCK_TIME_MS
        - _tc.RPC_BLOCK_AVAILABILITY_DELAY_P99_MS
        - _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_FINAL_POLL_BATCH_SIZE)
        - _tc.RPC_POLL_FINAL_SAFETY_BUFFER_MS
    )
    ramp_poll_2_wakeup_offset_ms = (
        final_rpc_poll_wakeup_offset_ms
        + _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_RAMP_POLL_2_BATCH_SIZE)
        + _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    )
    ramp_poll_1_wakeup_offset_ms = (
        ramp_poll_2_wakeup_offset_ms
        + _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_RAMP_POLL_1_BATCH_SIZE)
        + _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    )
    return {
        "bet_submit": bet_submit_deadline_offset_ms,
        "critical_path": critical_path_wakeup_offset_ms,
        "final": final_rpc_poll_wakeup_offset_ms,
        "ramp_2": ramp_poll_2_wakeup_offset_ms,
        "ramp_1": ramp_poll_1_wakeup_offset_ms,
    }


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

def test_rtt_curve_constant_present():
    assert hasattr(_tc, "RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE")
    table = _tc.RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    # Must include the EXPECTED_*_BATCH_SIZE values used by the
    # derivation. If any of those keys are missing, rpc_rtt_p99_for_batch
    # falls back to next ceiling, which is a measurement gap and
    # should be flagged.
    expected_keys = {
        _tc.EXPECTED_FINAL_POLL_BATCH_SIZE,
        _tc.EXPECTED_RAMP_POLL_2_BATCH_SIZE,
        _tc.EXPECTED_RAMP_POLL_1_BATCH_SIZE,
    }
    missing = expected_keys - set(table.keys())
    assert not missing, (
        f"RTT curve missing keys for EXPECTED_*_BATCH_SIZE: {missing}"
    )


def test_rtt_curve_monotonic():
    """Bigger batches should have >= RTT (with tolerance for raw probe
    noise). Sorts by size and asserts non-decreasing."""
    table = _tc.RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    pairs = sorted(table.items())
    for (s_lo, rtt_lo), (s_hi, rtt_hi) in zip(pairs, pairs[1:]):
        # Allow up to 100ms drop (probe noise) but flag anything
        # larger.
        assert rtt_hi >= rtt_lo - 100, (
            f"non-monotonic RTT: size={s_lo} rtt={rtt_lo} > "
            f"size={s_hi} rtt={rtt_hi} (drop > 100ms = probe noise OOB)"
        )


def test_rpc_rtt_p99_for_batch_helper():
    """Helper returns ceiling-key value, with sensible boundaries."""
    table = _tc.RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    keys = sorted(table.keys())
    # Below the smallest key returns the smallest key's RTT.
    assert _tc.rpc_rtt_p99_for_batch(1) == table[keys[0]]
    # Each key returns its own RTT.
    for k in keys:
        assert _tc.rpc_rtt_p99_for_batch(k) == table[k]
    # Above the largest key returns the largest key's RTT (fallback).
    assert _tc.rpc_rtt_p99_for_batch(keys[-1] + 100) == table[keys[-1]]
    # Zero or negative returns 0.
    assert _tc.rpc_rtt_p99_for_batch(0) == 0
    assert _tc.rpc_rtt_p99_for_batch(-1) == 0


# ---------------------------------------------------------------------------
# Derivation invariants (parameterized over pool_cutoff)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pool_cutoff", [6, 7, 8, 10, 12])
def test_final_offset_within_pool_cutoff_window(pool_cutoff):
    """final_rpc_poll_offset must be > 0 and < pool_cutoff * 1000."""
    s = _derive_schedule(pool_cutoff)
    assert s["final"] > 0, f"final_offset <= 0 at pool_cutoff={pool_cutoff}"
    assert s["final"] < pool_cutoff * 1000, (
        f"final_offset {s['final']}ms >= pool_cutoff_window "
        f"{pool_cutoff * 1000}ms at pool_cutoff={pool_cutoff}"
    )


@pytest.mark.parametrize("pool_cutoff", [6, 7, 8, 10, 12])
def test_final_offset_leaves_room_for_critical_path_completion(pool_cutoff):
    """final_rpc_poll_offset must be > critical_path_offset + safety so
    the engine has time between the final poll and the critical-path
    snapshot. This is the cross-validation gate enforced at runtime.
    """
    s = _derive_schedule(pool_cutoff)
    safety = _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    assert s["final"] > s["critical_path"] + safety, (
        f"final_offset {s['final']}ms <= critical_path_offset "
        f"{s['critical_path']}ms + safety {safety}ms at "
        f"pool_cutoff={pool_cutoff}"
    )


@pytest.mark.parametrize("pool_cutoff", [6, 7, 8, 10, 12])
def test_ramp_2_after_final_in_chronology(pool_cutoff):
    """ramp_2 fires BEFORE final (i.e., its lock-relative offset is
    bigger). The ramp_2 -> final gap must absorb the ramp_2 RTT."""
    s = _derive_schedule(pool_cutoff)
    assert s["ramp_2"] > s["final"], (
        f"ramp_2 offset {s['ramp_2']}ms not > final offset "
        f"{s['final']}ms at pool_cutoff={pool_cutoff}"
    )
    gap = s["ramp_2"] - s["final"]
    ramp_2_rtt = _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_RAMP_POLL_2_BATCH_SIZE)
    safety = _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    assert gap >= ramp_2_rtt + safety, (
        f"ramp_2->final gap {gap}ms < ramp_2_rtt {ramp_2_rtt}ms + "
        f"safety {safety}ms"
    )


@pytest.mark.parametrize("pool_cutoff", [6, 7, 8, 10, 12])
def test_ramp_1_after_ramp_2_in_chronology(pool_cutoff):
    """ramp_1 fires BEFORE ramp_2 (bigger lock-relative offset).
    ramp_1 -> ramp_2 gap must absorb the ramp_1 RTT."""
    s = _derive_schedule(pool_cutoff)
    assert s["ramp_1"] > s["ramp_2"], (
        f"ramp_1 offset {s['ramp_1']}ms not > ramp_2 offset "
        f"{s['ramp_2']}ms at pool_cutoff={pool_cutoff}"
    )
    gap = s["ramp_1"] - s["ramp_2"]
    ramp_1_rtt = _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_RAMP_POLL_1_BATCH_SIZE)
    safety = _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    assert gap >= ramp_1_rtt + safety


def test_canonical_pool_cutoff_6_produces_expected_offsets():
    """Pin the canonical-baseline schedule values. If a future change
    to RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE shifts these by more than
    a few ms, this test fails so the operator notices the shift."""
    s = _derive_schedule(6)
    # critical_path is unchanged from pre-Era-11
    assert s["critical_path"] == 1095
    assert s["bet_submit"] == 750
    # New offsets at canonical pool_cutoff=6
    assert s["final"] == 3790
    assert s["ramp_2"] == 5203
    assert s["ramp_1"] == 6616


def test_pool_cutoff_too_small_would_violate_final_offset_floor():
    """pool_cutoff=2 (or any value where final_offset <=
    critical_path + safety) would trip the runtime cross-validation
    gate. Sanity-check the trip point."""
    # At pool_cutoff=2: final_offset = 2000 - 500 - 600 - 910 - 200 = -210
    # which is below critical_path (1095) + safety (200) = 1295.
    # The runtime gate raises InvariantError; here we just assert
    # the math works as expected.
    s = _derive_schedule(2)
    safety = _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    assert s["final"] <= s["critical_path"] + safety, (
        "pool_cutoff=2 should be too small for RPC completion"
    )
