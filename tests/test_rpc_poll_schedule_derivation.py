"""Code-invariant tests for the RPC-poll wake schedule derivation.

Candidate C (2026-06-06): the 3-leg ramp ladder (ramp_1/ramp_2/final) was
replaced by ONE batched poll before the critical path. pancakebot/config.py
derives single_poll_wakeup_offset_before_lock_ms from pool_cutoff_seconds + the
empirical timing constants. These tests pin the invariants the formula must
satisfy across the canonical pool_cutoff range.

What lives HERE vs in config.py runtime checks: runtime checks validate USER
CONFIG (the pool_cutoff_seconds the user picked). These tests validate CODE
CORRECTNESS (the formula always produces a sensible offset for any reasonable
pool_cutoff).
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
    """Derive the lock-relative offsets (ms) using the exact same formula as
    pancakebot/config.py. The single-poll offset is a FIXED rail
    (SINGLE_POLL_WAKEUP_OFFSET_BEFORE_LOCK_MS), pool_cutoff-independent;
    pool_cutoff only sets the CAPTURE upper bound (single_poll_max_capture)
    the rail must stay under. The completion invariant (rtt_p99 + safety) is
    validated separately at config load.
    """
    bet_submit_deadline_offset_before_lock_ms = (
        _tc.BSC_QUANTUM_MS
        + _tc.BSC_BLOCK_TIME_MS
        + _tc.VALIDATOR_ASSEMBLY_WINDOW_MS
        + _tc.BSC_BET_SUBMIT_ONE_WAY_MS
    )
    critical_path_wakeup_offset_before_lock_ms = (
        bet_submit_deadline_offset_before_lock_ms
        + _tc.OKX_KLINE_FETCH_RTT_P99_MS
        + _tc.SIGNAL_COMPUTE_TIME_MS
        + _tc.POOL_READ_TIME_MS
    )
    single_poll_wakeup_offset_before_lock_ms = (
        _tc.SINGLE_POLL_WAKEUP_OFFSET_BEFORE_LOCK_MS
    )
    single_poll_max_capture_offset = (
        pool_cutoff_seconds * 1000
        - _tc.BSC_BLOCK_TIME_MS
        - _tc.RPC_BLOCK_AVAILABILITY_DELAY_P99_MS
        - _tc.RPC_POLL_FINAL_TO_CRITICAL_PATH_SAFETY_MS
    )
    return {
        "bet_submit": bet_submit_deadline_offset_before_lock_ms,
        "critical_path": critical_path_wakeup_offset_before_lock_ms,
        "single_poll": single_poll_wakeup_offset_before_lock_ms,
        "single_poll_max_capture": single_poll_max_capture_offset,
    }


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

def test_rtt_curve_constant_present():
    assert hasattr(_tc, "RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE")
    table = _tc.RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    # The single-poll startup invariant looks up EXPECTED_SINGLE_POLL_BATCH_SIZE.
    # If that key is missing, rpc_rtt_p99_for_batch interpolates/ceilings, which
    # is a measurement gap and should be flagged.
    assert _tc.EXPECTED_SINGLE_POLL_BATCH_SIZE in table, (
        f"RTT curve missing key for EXPECTED_SINGLE_POLL_BATCH_SIZE="
        f"{_tc.EXPECTED_SINGLE_POLL_BATCH_SIZE}"
    )


def test_rtt_curve_monotonic():
    """Bigger batches should have >= RTT (with tolerance for raw probe
    noise). Sorts by size and asserts non-decreasing."""
    table = _tc.RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    pairs = sorted(table.items())
    for (s_lo, rtt_lo), (s_hi, rtt_hi) in zip(pairs, pairs[1:]):
        assert rtt_hi >= rtt_lo - 100, (
            f"non-monotonic RTT: size={s_lo} rtt={rtt_lo} > "
            f"size={s_hi} rtt={rtt_hi} (drop > 100ms = probe noise OOB)"
        )


def test_rpc_rtt_p99_for_batch_helper():
    """Helper passes exact keys through, ceilings at small/large ends, and
    returns 0 for non-positive sizes. All current callers (the config.py
    single-poll invariant + rpc_poller with _batch_size=20) use exact keys, so
    this guards backward-compat at the canonical pin points."""
    table = _tc.RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    keys = sorted(table.keys())
    assert _tc.rpc_rtt_p99_for_batch(1) == table[keys[0]]
    for k in keys:
        assert _tc.rpc_rtt_p99_for_batch(k) == table[k]
    assert _tc.rpc_rtt_p99_for_batch(keys[-1] + 100) == table[keys[-1]]
    assert _tc.rpc_rtt_p99_for_batch(0) == 0
    assert _tc.rpc_rtt_p99_for_batch(-1) == 0


def test_rpc_rtt_p99_for_batch_interp_passthrough_at_keys():
    """rpc_rtt_p99_for_batch interpolates linearly between measured keys. At
    every measured key, the result must equal the table value exactly."""
    table = _tc.RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    for k, v in table.items():
        assert _tc.rpc_rtt_p99_for_batch(k) == v, (
            f"passthrough failed at key={k}: table={v} got={_tc.rpc_rtt_p99_for_batch(k)}"
        )


def test_rpc_rtt_p99_for_batch_interp_interior_points():
    """Linear-interpolation between bracketing adjacent keys, rounded to the
    nearest integer. Spec values against {2:79, 5:122, 10:222, 15:229, 20:240}.
    """
    assert _tc.rpc_rtt_p99_for_batch(3) == 93
    assert _tc.rpc_rtt_p99_for_batch(4) == 108
    assert _tc.rpc_rtt_p99_for_batch(7) == 162
    assert _tc.rpc_rtt_p99_for_batch(12) == 225
    assert _tc.rpc_rtt_p99_for_batch(18) == 236


def test_rpc_rtt_p99_for_batch_interp_edges():
    """Edge cases: 0/-1 -> 0; 1 -> table[2] (small-end ceiling); 25/100 ->
    table[20] (large-end ceiling)."""
    table = _tc.RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    keys = sorted(table.keys())
    smallest, largest = keys[0], keys[-1]
    assert _tc.rpc_rtt_p99_for_batch(0) == 0
    assert _tc.rpc_rtt_p99_for_batch(-1) == 0
    assert _tc.rpc_rtt_p99_for_batch(1) == table[smallest]
    assert _tc.rpc_rtt_p99_for_batch(25) == table[largest]
    assert _tc.rpc_rtt_p99_for_batch(100) == table[largest]


# ---------------------------------------------------------------------------
# Single-poll derivation invariants (parameterized over pool_cutoff)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pool_cutoff", [6, 7, 8, 10, 12])
def test_single_poll_offset_within_pool_cutoff_window(pool_cutoff):
    """The fixed single_poll rail must fire inside the round (0 < offset <
    pool_cutoff window) and no earlier than the capture bound (so the cutoff
    block's receipts are available when it polls)."""
    s = _derive_schedule(pool_cutoff)
    assert s["single_poll"] > 0, f"single_poll <= 0 at pool_cutoff={pool_cutoff}"
    assert s["single_poll"] < pool_cutoff * 1000, (
        f"single_poll {s['single_poll']}ms >= pool_cutoff_window "
        f"{pool_cutoff * 1000}ms at pool_cutoff={pool_cutoff}"
    )
    assert s["single_poll"] <= s["single_poll_max_capture"], (
        f"single_poll {s['single_poll']}ms > capture bound "
        f"{s['single_poll_max_capture']}ms at pool_cutoff={pool_cutoff}"
    )


@pytest.mark.parametrize("pool_cutoff", [6, 7, 8, 10, 12])
def test_single_poll_leaves_room_for_critical_path_completion(pool_cutoff):
    """The single poll's completion (at empirical rtt_p99 for the worst-case
    batch) must arrive before critical_path + safety. Mirrors the runtime
    invariant ``single_poll_rtt_budget_insufficient`` in pancakebot/config.py.
    """
    s = _derive_schedule(pool_cutoff)
    rtt = _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_SINGLE_POLL_BATCH_SIZE)
    safety = _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    completion_offset = s["single_poll"] - rtt - safety
    assert completion_offset >= s["critical_path"], (
        f"single_poll completion_offset {completion_offset}ms < critical_path "
        f"{s['critical_path']}ms at pool_cutoff={pool_cutoff} "
        f"(single_poll={s['single_poll']}ms - rtt_p99("
        f"{_tc.EXPECTED_SINGLE_POLL_BATCH_SIZE})={rtt}ms - safety={safety}ms)"
    )


def test_canonical_pool_cutoff_6_produces_expected_offsets():
    """Pin the canonical-baseline schedule.

    2026-06-06 VM re-baseline: the single poll is a fixed rail (no longer
    pool_cutoff-derived); critical_path uses OKX P99 (the P95 tier is retired):
        single_poll   = SINGLE_POLL_WAKEUP_OFFSET_BEFORE_LOCK_MS = 2500ms
        critical_path = bet_submit + 351 + 50 + 5 = 1195ms
        bet_submit    = 50 + 450 + 214 + 75       = 789ms
        capture bound = 6000 - 450 - 625 - 200    = 4725ms (>= 2500)
    """
    s = _derive_schedule(6)
    assert s["critical_path"] == 1195
    assert s["bet_submit"] == 789
    assert s["single_poll"] == 2500
    assert s["single_poll_max_capture"] == 4725


def test_pool_cutoff_too_small_would_violate_single_poll_capture():
    """With a FIXED single_poll rail, too-small pool_cutoff trips the CAPTURE
    bound (not the completion floor). At pool_cutoff=2 the cutoff block sits at
    lock-2000, but the fixed 2500ms rail would fire at lock-2500 — before that
    block even exists. capture bound = 2000 - 450 - 625 - 200 = 725ms;
    single_poll=2500 > 725 → the startup CAPTURE invariant raises
    (single_poll_fires_before_cutoff_available). Here we just assert the math.
    """
    s = _derive_schedule(2)
    assert s["single_poll"] > s["single_poll_max_capture"], (
        f"pool_cutoff=2 should violate the capture bound "
        f"(single_poll={s['single_poll']}ms <= max_capture="
        f"{s['single_poll_max_capture']}ms)"
    )
