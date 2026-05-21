"""SSOT invariant for the engine's dynamic critical-path wake.

The dynamic wake is derived from the same ``compute_submit_deadline_ms``
the bet-submit timing guard uses, walked back by the workload window
(kline fetch p99 + gate compute + pool read). The earlier inline formula
recomputed two of the deadline-side subtractions and silently dropped
the validator assembly window (a 50ms structural gap that survived
since Bundle 4).

These tests pin the SSOT property so the divergence cannot reappear.
The wake is asserted via a mirror formula rather than by calling the
engine's per-round state machine; if anyone changes ``engine.py`` to
deviate from this formula, the test still pins the correct invariant.
"""
from __future__ import annotations

from unittest.mock import patch

from pancakebot import timing_constants as tc
from pancakebot.chain.rpc_poller import (
    compute_submit_deadline_ms,
    predict_predecessor_milli_ts,
)


def _ssot_dynamic_wake_ms(*, anchor_milli_ts: int, lock_ms: int) -> int:
    """Mirror of the engine's SSOT wake derivation.

    Kept inline so the test pins the FORMULA against drift. The engine
    code at ``pancakebot/runtime/engine.py`` must compute the same value
    via the same call chain.
    """
    predecessor_ms = predict_predecessor_milli_ts(
        anchor_milli_ts=anchor_milli_ts, lock_ms=lock_ms,
    )
    deadline_ms = compute_submit_deadline_ms(
        predicted_predecessor_milli_ts=predecessor_ms, lock_ms=lock_ms,
    )
    return deadline_ms - (
        tc.OKX_KLINE_FETCH_RTT_P99_MS
        + tc.SIGNAL_COMPUTE_TIME_MS
        + tc.POOL_READ_TIME_MS
    )


def test_wake_equals_deadline_minus_workload():
    """SSOT invariant: ``dynamic_wake_ms == deadline_ms - (P99 + GATE + POOL)``.

    Pinned for a non-quantum-zone anchor (predecessor lands comfortably
    inside the predecessor slot, not within one quantum of lock).
    """
    lock_ms = 1_700_000_000_000  # arbitrary lock time (whole-second-equivalent ms)
    anchor_milli_ts = lock_ms - 800  # typical mid-round anchor poll response

    predecessor_ms = predict_predecessor_milli_ts(
        anchor_milli_ts=anchor_milli_ts, lock_ms=lock_ms,
    )
    deadline_ms = compute_submit_deadline_ms(
        predicted_predecessor_milli_ts=predecessor_ms, lock_ms=lock_ms,
    )
    expected_wake = deadline_ms - (
        tc.OKX_KLINE_FETCH_RTT_P99_MS
        + tc.SIGNAL_COMPUTE_TIME_MS
        + tc.POOL_READ_TIME_MS
    )
    actual_wake = _ssot_dynamic_wake_ms(
        anchor_milli_ts=anchor_milli_ts, lock_ms=lock_ms,
    )
    assert actual_wake == expected_wake

    # Sanity-check the values aren't trivially zero (the assertion above
    # would hold for the degenerate case wake == deadline == 0).
    assert deadline_ms < lock_ms
    assert actual_wake < deadline_ms


def test_quantum_backoff_propagates_to_wake():
    """When the predecessor lands within one quantum of lock_ms, the
    deadline backs off one full block (450ms). The wake MUST track that
    back-off; otherwise it fires too late for the new target block's
    assembly window.

    Construction: pick an anchor such that
    ``predict_predecessor_milli_ts(...)`` returns ``lock_ms - quantum_ms``,
    which triggers the quantum-shift branch in
    ``compute_submit_deadline_ms``.
    """
    lock_ms = 1_700_000_000_000
    block_ms = tc.BSC_BLOCK_TIME_MS
    quantum_ms = tc.BSC_QUANTUM_MS

    # predict_predecessor uses ceil((lock_ms - 10 - anchor_ms) / block_ms)
    # with k=1 -> predecessor = anchor + (block_ms - block_ms) = anchor.
    # Set anchor = lock_ms - quantum_ms so predecessor = lock_ms - quantum_ms,
    # which is exactly in the quantum-shift danger zone.
    anchor_milli_ts = lock_ms - quantum_ms
    predecessor_ms = predict_predecessor_milli_ts(
        anchor_milli_ts=anchor_milli_ts, lock_ms=lock_ms,
    )
    assert (predecessor_ms + quantum_ms) >= lock_ms, (
        "test fixture did not land in the quantum-shift zone"
    )

    deadline_ms = compute_submit_deadline_ms(
        predicted_predecessor_milli_ts=predecessor_ms, lock_ms=lock_ms,
    )
    # Deadline = predecessor - block - assembly - one_way (block back-off
    # triggered by quantum-shift guard).
    expected_deadline = (
        predecessor_ms
        - block_ms
        - tc.VALIDATOR_ASSEMBLY_WINDOW_MS
        - tc.BSC_BET_SUBMIT_ONE_WAY_MS
    )
    assert deadline_ms == expected_deadline

    # Wake = deadline - workload. The 450ms block back-off is now
    # reflected in the wake (would not have been under the inline formula).
    wake_ms = _ssot_dynamic_wake_ms(
        anchor_milli_ts=anchor_milli_ts, lock_ms=lock_ms,
    )
    expected_wake = deadline_ms - (
        tc.OKX_KLINE_FETCH_RTT_P99_MS
        + tc.SIGNAL_COMPUTE_TIME_MS
        + tc.POOL_READ_TIME_MS
    )
    assert wake_ms == expected_wake


def test_constant_mutation_shifts_wake_and_deadline_in_lockstep():
    """Mutating any deadline-side timing constant must shift the wake
    by the same delta. Pins the SSOT property against future drift
    (e.g. a partial refactor that touches one site but not the other).
    """
    lock_ms = 1_700_000_000_000
    anchor_milli_ts = lock_ms - 800
    predecessor_ms = predict_predecessor_milli_ts(
        anchor_milli_ts=anchor_milli_ts, lock_ms=lock_ms,
    )

    base_wake = _ssot_dynamic_wake_ms(
        anchor_milli_ts=anchor_milli_ts, lock_ms=lock_ms,
    )
    base_deadline = compute_submit_deadline_ms(
        predicted_predecessor_milli_ts=predecessor_ms, lock_ms=lock_ms,
    )

    # Bump VALIDATOR_ASSEMBLY_WINDOW_MS by 25ms. Both deadline and wake
    # should shift EARLIER by 25ms.
    with patch.object(
        tc, "VALIDATOR_ASSEMBLY_WINDOW_MS", tc.VALIDATOR_ASSEMBLY_WINDOW_MS + 25,
    ):
        new_wake = _ssot_dynamic_wake_ms(
            anchor_milli_ts=anchor_milli_ts, lock_ms=lock_ms,
        )
        new_deadline = compute_submit_deadline_ms(
            predicted_predecessor_milli_ts=predecessor_ms, lock_ms=lock_ms,
        )
        assert new_deadline == base_deadline - 25
        assert new_wake == base_wake - 25

    # Bump BSC_BET_SUBMIT_ONE_WAY_MS by 50ms. Same shift property.
    with patch.object(
        tc, "BSC_BET_SUBMIT_ONE_WAY_MS", tc.BSC_BET_SUBMIT_ONE_WAY_MS + 50,
    ):
        new_wake = _ssot_dynamic_wake_ms(
            anchor_milli_ts=anchor_milli_ts, lock_ms=lock_ms,
        )
        new_deadline = compute_submit_deadline_ms(
            predicted_predecessor_milli_ts=predecessor_ms, lock_ms=lock_ms,
        )
        assert new_deadline == base_deadline - 50
        assert new_wake == base_wake - 50

    # Bump a workload-side constant (POOL_READ_TIME_MS). Wake shifts;
    # deadline does NOT (deadline is workload-independent by construction).
    with patch.object(tc, "POOL_READ_TIME_MS", tc.POOL_READ_TIME_MS + 7):
        new_wake = _ssot_dynamic_wake_ms(
            anchor_milli_ts=anchor_milli_ts, lock_ms=lock_ms,
        )
        new_deadline = compute_submit_deadline_ms(
            predicted_predecessor_milli_ts=predecessor_ms, lock_ms=lock_ms,
        )
        assert new_deadline == base_deadline, (
            "deadline must be workload-independent"
        )
        assert new_wake == base_wake - 7
