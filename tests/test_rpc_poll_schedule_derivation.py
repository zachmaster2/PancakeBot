"""Code-invariant tests for the RPC-poll wake schedule derivation.

The derivation chain in pancakebot/config.py computes
final_rpc_poll_wakeup_offset_before_lock_ms, ramp_poll_2_wakeup_offset_before_lock_ms, and
ramp_poll_1_wakeup_offset_before_lock_ms from pool_cutoff_seconds + the
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
    using the exact same formula as pancakebot/config.py.

    Refactor 2026-05-12: runtime offsets no longer depend on
    rpc_rtt_p99_for_batch lookups. The RTT lives in the startup
    invariant (validates the chosen offset can absorb the actual
    p99 + safety), not in the wake-time derivation itself.
    """
    bet_submit_deadline_offset_before_lock_ms = (
        _tc.BSC_QUANTUM_MS
        + _tc.BSC_BLOCK_TIME_MS
        + _tc.VALIDATOR_ASSEMBLY_WINDOW_MS
        + _tc.BSC_BET_SUBMIT_ONE_WAY_MS
    )
    critical_path_wakeup_offset_before_lock_ms = (
        bet_submit_deadline_offset_before_lock_ms
        + _tc.OKX_KLINE_FETCH_RTT_P95_MS
        + _tc.MOMENTUM_GATE_COMPUTE_TIME_MS
        + _tc.POOL_READ_TIME_MS
    )
    final_rpc_poll_wakeup_offset_before_lock_ms = (
        pool_cutoff_seconds * 1000
        - _tc.BSC_BLOCK_TIME_MS
        - _tc.RPC_BLOCK_AVAILABILITY_DELAY_P99_MS
        - _tc.RPC_POLL_FINAL_TO_CRITICAL_PATH_SAFETY_MS
    )
    ramp_poll_2_wakeup_offset_before_lock_ms = (
        final_rpc_poll_wakeup_offset_before_lock_ms
        + _tc.RPC_RAMP_2_TO_FINAL_INTERVAL_MS
    )
    ramp_poll_1_wakeup_offset_before_lock_ms = (
        ramp_poll_2_wakeup_offset_before_lock_ms
        + _tc.RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS
    )
    return {
        "bet_submit": bet_submit_deadline_offset_before_lock_ms,
        "critical_path": critical_path_wakeup_offset_before_lock_ms,
        "final": final_rpc_poll_wakeup_offset_before_lock_ms,
        "ramp_2": ramp_poll_2_wakeup_offset_before_lock_ms,
        "ramp_1": ramp_poll_1_wakeup_offset_before_lock_ms,
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
    """Helper passes exact keys through, ceilings at small/large ends,
    and returns 0 for non-positive sizes. Same contract as before the
    2026-05-12 interpolation refactor — all current callers (config.py
    invariants + rpc_poller with _batch_size=20) use exact keys, so this
    test guards backward-compat at the canonical pin points."""
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


def test_rpc_rtt_p99_for_batch_interp_passthrough_at_keys():
    """Refactor 2026-05-12: rpc_rtt_p99_for_batch interpolates linearly
    between measured keys. At every measured key, the result must equal
    the table value exactly (pure passthrough, no rounding drift)."""
    table = _tc.RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    for k, v in table.items():
        assert _tc.rpc_rtt_p99_for_batch(k) == v, (
            f"passthrough failed at key={k}: table={v} got={_tc.rpc_rtt_p99_for_batch(k)}"
        )


def test_rpc_rtt_p99_for_batch_interp_interior_points():
    """Linear-interpolation between bracketing adjacent keys, with
    rounding to the nearest integer. Spec values computed against the
    canonical table {2:421, 5:771, 10:910, 15:1213, 20:1319}.
    """
    # batch=3: between (2,421) and (5,771); 421 + (350)*1/3 = 537.67 -> 538
    assert _tc.rpc_rtt_p99_for_batch(3) == 538
    # batch=4: between (2,421) and (5,771); 421 + (350)*2/3 = 654.33 -> 654
    assert _tc.rpc_rtt_p99_for_batch(4) == 654
    # batch=7: between (5,771) and (10,910); 771 + (139)*2/5 = 826.6 -> 827
    assert _tc.rpc_rtt_p99_for_batch(7) == 827
    # batch=12: between (10,910) and (15,1213); 910 + (303)*2/5 = 1031.2 -> 1031
    assert _tc.rpc_rtt_p99_for_batch(12) == 1031
    # batch=18: between (15,1213) and (20,1319); 1213 + (106)*3/5 = 1276.6 -> 1277
    assert _tc.rpc_rtt_p99_for_batch(18) == 1277


def test_rpc_rtt_p99_for_batch_interp_edges():
    """Edge cases preserved by the refactor:
      - 0, -1 -> 0
      - 1 -> table[2] (ceiling at small end)
      - 25, 100 -> table[20] (ceiling at large end)
    """
    table = _tc.RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    keys = sorted(table.keys())
    smallest, largest = keys[0], keys[-1]
    assert _tc.rpc_rtt_p99_for_batch(0) == 0
    assert _tc.rpc_rtt_p99_for_batch(-1) == 0
    assert _tc.rpc_rtt_p99_for_batch(1) == table[smallest]
    assert _tc.rpc_rtt_p99_for_batch(25) == table[largest]
    assert _tc.rpc_rtt_p99_for_batch(100) == table[largest]


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
    """final_rpc_poll completion (at empirical rtt_p99) must arrive
    before critical_path + safety. Mirrors the runtime invariant
    ``final_rpc_poll_rtt_budget_insufficient`` enforced in
    pancakebot/config.py at load time (refactored 2026-05-12 to be
    strictly stronger than the prior ``final > critical_path + safety``
    check by additionally subtracting rtt_p99).
    """
    s = _derive_schedule(pool_cutoff)
    rtt = _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_FINAL_POLL_BATCH_SIZE)
    safety = _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    final_completion_offset = s["final"] - rtt - safety
    assert final_completion_offset >= s["critical_path"], (
        f"final_completion_offset {final_completion_offset}ms < "
        f"critical_path {s['critical_path']}ms at pool_cutoff={pool_cutoff} "
        f"(final={s['final']}ms - rtt_p99({_tc.EXPECTED_FINAL_POLL_BATCH_SIZE})="
        f"{rtt}ms - safety={safety}ms)"
    )


@pytest.mark.parametrize("pool_cutoff", [6, 7, 8, 10, 12])
def test_ramp_2_after_final_in_chronology(pool_cutoff):
    """ramp_2 fires BEFORE final (i.e., its lock-relative offset is
    bigger). The ramp_2 -> final gap = RPC_RAMP_2_TO_FINAL_INTERVAL_MS,
    which the startup invariant validates is >= rtt_p99(ramp_2) +
    safety."""
    s = _derive_schedule(pool_cutoff)
    assert s["ramp_2"] > s["final"], (
        f"ramp_2 offset {s['ramp_2']}ms not > final offset "
        f"{s['final']}ms at pool_cutoff={pool_cutoff}"
    )
    gap = s["ramp_2"] - s["final"]
    assert gap == _tc.RPC_RAMP_2_TO_FINAL_INTERVAL_MS, (
        f"ramp_2->final gap {gap}ms != RPC_RAMP_2_TO_FINAL_INTERVAL_MS "
        f"({_tc.RPC_RAMP_2_TO_FINAL_INTERVAL_MS}ms) at pool_cutoff={pool_cutoff}"
    )
    # Sanity: the interval must cover rtt_p99 + safety. This is the
    # invariant enforced at config-load time.
    ramp_2_rtt = _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_RAMP_POLL_2_BATCH_SIZE)
    safety = _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    assert _tc.RPC_RAMP_2_TO_FINAL_INTERVAL_MS >= ramp_2_rtt + safety, (
        f"RPC_RAMP_2_TO_FINAL_INTERVAL_MS {_tc.RPC_RAMP_2_TO_FINAL_INTERVAL_MS}ms "
        f"< ramp_2_rtt {ramp_2_rtt}ms + safety {safety}ms"
    )


@pytest.mark.parametrize("pool_cutoff", [6, 7, 8, 10, 12])
def test_ramp_1_after_ramp_2_in_chronology(pool_cutoff):
    """ramp_1 fires BEFORE ramp_2 (bigger lock-relative offset).
    ramp_1 -> ramp_2 gap = RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS."""
    s = _derive_schedule(pool_cutoff)
    assert s["ramp_1"] > s["ramp_2"], (
        f"ramp_1 offset {s['ramp_1']}ms not > ramp_2 offset "
        f"{s['ramp_2']}ms at pool_cutoff={pool_cutoff}"
    )
    gap = s["ramp_1"] - s["ramp_2"]
    assert gap == _tc.RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS, (
        f"ramp_1->ramp_2 gap {gap}ms != RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS "
        f"({_tc.RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS}ms) at pool_cutoff={pool_cutoff}"
    )
    ramp_1_rtt = _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_RAMP_POLL_1_BATCH_SIZE)
    safety = _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    assert _tc.RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS >= ramp_1_rtt + safety


def test_canonical_pool_cutoff_6_produces_expected_offsets():
    """Pin the canonical-baseline schedule values.

    Refactor 2026-05-12 (round 2): per-leg ramp intervals replace the
    uniform RPC_RAMP_POLL_INTERVAL_MS=1500. ramp_1's interval to ramp_2
    covers its worst-case 8s-periodic-catchup workload (~18 blocks,
    batch=20 → 1319ms p99 + 200ms safety + ~181ms margin = 1700ms).
    ramp_2's interval to final covers the small incremental top-up
    (~4 blocks, batch=5 → 771ms p99 + 200ms safety + ~129ms margin
    = 1100ms).

    Bundle 4 (2026-05-14): all offsets shifted by ±50ms due to
    BSC_BLOCK_TIME_MS: 500 → 450 (post-Lorentz empirical correction).
    Derivation now uses the new constants
    (BSC_QUANTUM + BSC_BLOCK_TIME + VALIDATOR_ASSEMBLY + ONE_WAY = 625)
    for bet_submit_deadline, and the block-time
    update ripples through final = pool_cutoff*1000 - BSC_BLOCK_TIME - ...

    Schedule at canonical pool_cutoff=6 (Bundle 4):
        final  = 6000 - 450 - 600 - 200             = 4750ms
        ramp_2 = 4750 + 1100 (RPC_RAMP_2_TO_FINAL)  = 5850ms
        ramp_1 = 5850 + 1700 (RPC_RAMP_1_TO_RAMP_2) = 7550ms
        critical_path = bet_submit + 290 + 50 + 5   = 970ms (post 2026-05-20 re-measurement)
        bet_submit    = 50 + 450 + 50 + 75          = 625ms (post 2026-05-20 re-measurement)
    """
    s = _derive_schedule(6)
    # Bundle 4 shifts (50ms tighter on critical_path/bet_submit; 50ms later on RPC polls).
    # 2026-05-20: BSC_BET_SUBMIT_ONE_WAY_MS 150 → 75 shrinks critical_path and bet_submit by 75ms each.
    assert s["critical_path"] == 970
    assert s["bet_submit"] == 625
    assert s["final"] == 4750
    assert s["ramp_2"] == 5850
    assert s["ramp_1"] == 7550


def test_pool_cutoff_too_small_would_violate_final_offset_floor():
    """pool_cutoff=2 (or any value where final_offset - rtt_p99 -
    safety <= critical_path) would trip the startup invariant.
    Sanity-check the trip point.

    At pool_cutoff=2 the refactored formula:
        final = 2000 - 500 - 600 - 200 = 700ms
    rtt_p99(10) = 910ms (current table); safety = 200ms.
    final - 910 - 200 = -410ms, way below critical_path = 1095ms.
    The startup invariant raises InvariantError; here we just assert
    the math works as expected.
    """
    s = _derive_schedule(2)
    rtt = _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_FINAL_POLL_BATCH_SIZE)
    safety = _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    final_completion_offset = s["final"] - rtt - safety
    assert final_completion_offset < s["critical_path"], (
        f"pool_cutoff=2 should violate startup invariant "
        f"(final_completion_offset={final_completion_offset}ms "
        f">= critical_path={s['critical_path']}ms)"
    )
