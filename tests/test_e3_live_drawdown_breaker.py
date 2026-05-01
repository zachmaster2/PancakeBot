"""E3 live drawdown breaker: end-to-end verification of the wired live path.

Per p4a (review protocol topic ``p4a_e3_live_drawdown_fix``) — verifies that the
``record_settlement`` → ``peak_bankroll`` → ``dd_frac`` → ``set_paused`` chain
fires when a drawdown breaches the 15% threshold and recovers after cooldown.

The wiring lives in the live runtime at ``pancakebot/runtime/engine.py`` (the
post-RPC-fetch ``record_settlement`` call in ``_run_one_iteration``'s
housekeeping phase). The risk-check block lives in
``pancakebot/strategy/momentum_pipeline.py:228-246``. This test simulates that
block directly against an ``InMemoryBankrollTracker`` to exercise every branch
without needing a full pipeline + gate construction.

Two scenarios:
  - **FAST drawdown (R4)**: peak captured early, bankroll drops 15% within a
    short span (within the rolling-7d window). Verifies breaker fires AT
    threshold, cooldown registers, ticks down, then betting resumes.
  - **Idempotency (R2)**: ``record_settlement`` called repeatedly with same
    ``start_at`` and either same or different bankroll values. Pre-registers
    the intended dedup semantic: same-value retries are skipped (no double
    count); different-value writes for the same ``start_at`` are BOTH
    appended (history-honest, last-write wins for ``current_bankroll``).

NOTE: this test does NOT cover the rolling-7d slow-drain blindness from p1e
(extension cohort, peak follows decline downward). That is a structural
property of rolling-7d semantics, NOT a wiring bug, and is out of p4a scope —
addressed (if at all) by p2a's V1 absolute-ratchet promotion.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.bankroll_tracker import InMemoryBankrollTracker  # noqa: E402


# Synthetic round timestamps spaced 5 minutes apart (PancakeBot V2 round size).
_ROUND_SECONDS = 300
_BASE_START = 2_000_000_000  # arbitrary far-future epoch; avoids any prune edge


def _evaluate_risk_gate(
    tracker: InMemoryBankrollTracker,
    *,
    start_at: int,
    max_drawdown_frac_from_peak: float,
    cooldown_rounds: int,
) -> str:
    """Mirror momentum_pipeline.py:228-246 risk-check block.

    Returns one of: "ok", "risk_cooldown_active", "risk_drawdown_breaker_fired".
    """
    if tracker.is_paused(start_at):
        tracker.tick_cooldown()
        return "risk_cooldown_active"
    current = tracker.current_bankroll()
    peak = tracker.peak_bankroll(start_at)
    if peak > 0:
        dd_frac = (peak - current) / peak
        if dd_frac >= max_drawdown_frac_from_peak:
            tracker.set_paused(cooldown_rounds, start_at)
            return "risk_drawdown_breaker_fired"
    return "ok"


# ---------------------------------------------------------------------------
# R4 — FAST drawdown scenario
# ---------------------------------------------------------------------------

def test_fast_drawdown_fires_at_threshold():
    """Peak 100 BNB, drop to 85 BNB within a short span -> breaker fires AT 15%."""
    initial = 100.0
    threshold = 0.15
    cooldown_rounds = 72
    tracker = InMemoryBankrollTracker(
        initial_bankroll=initial,
        window_days=7,
        peak_mode="rolling_7d",
    )

    # Round 0: seed at peak (100). Risk gate: peak=100, current=100, dd=0 -> ok.
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    verdict = _evaluate_risk_gate(
        tracker,
        start_at=_BASE_START,
        max_drawdown_frac_from_peak=threshold,
        cooldown_rounds=cooldown_rounds,
    )
    assert verdict == "ok", f"round 0 (at peak): expected ok, got {verdict}"

    # Rounds 1-9: gradual decline 99.0, 98.0, ..., 91.0 (dd 1%..9%). All below
    # 15% threshold. Breaker MUST NOT fire. Spans 9 * 300s = 45 min, well
    # within the 7-day rolling window so peak stays at 100.
    for i, b in enumerate([99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0], start=1):
        sa = _BASE_START + i * _ROUND_SECONDS
        tracker.record_settlement(bankroll=b, start_at=sa)
        verdict = _evaluate_risk_gate(
            tracker,
            start_at=sa,
            max_drawdown_frac_from_peak=threshold,
            cooldown_rounds=cooldown_rounds,
        )
        assert verdict == "ok", (
            f"round {i} (bankroll={b}, dd={(100.0 - b) / 100.0:.4f}): "
            f"expected ok (below 15% threshold), got {verdict}"
        )

    # Round 10: drop to 85 BNB -- exactly 15% drawdown. Breaker MUST fire.
    sa = _BASE_START + 10 * _ROUND_SECONDS
    tracker.record_settlement(bankroll=85.0, start_at=sa)
    # Peak should still be 100 (well within the 7-day window).
    assert tracker.peak_bankroll(sa) == 100.0, (
        f"peak should still be 100.0 at threshold; got {tracker.peak_bankroll(sa)}"
    )
    assert tracker.current_bankroll() == 85.0
    verdict = _evaluate_risk_gate(
        tracker,
        start_at=sa,
        max_drawdown_frac_from_peak=threshold,
        cooldown_rounds=cooldown_rounds,
    )
    assert verdict == "risk_drawdown_breaker_fired", (
        f"round 10 (bankroll=85, dd=0.15): expected breaker fired, got {verdict}"
    )
    # Cooldown should now be set.
    assert tracker.cooldown_remaining() == cooldown_rounds, (
        f"after fire: cooldown should be {cooldown_rounds}, got {tracker.cooldown_remaining()}"
    )
    assert tracker.is_paused(sa) is True


def test_fast_drawdown_does_not_fire_at_14_999_pct():
    """Just-below threshold: dd = 14.999% must NOT fire (strict >= comparison)."""
    initial = 100.0
    threshold = 0.15
    tracker = InMemoryBankrollTracker(
        initial_bankroll=initial, window_days=7, peak_mode="rolling_7d",
    )
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    # 85.001 BNB -> dd_frac = 0.14999 -- one tick below threshold.
    sa = _BASE_START + _ROUND_SECONDS
    tracker.record_settlement(bankroll=85.001, start_at=sa)
    verdict = _evaluate_risk_gate(
        tracker,
        start_at=sa,
        max_drawdown_frac_from_peak=threshold,
        cooldown_rounds=72,
    )
    assert verdict == "ok", (
        f"dd=14.999% must NOT fire (strict >= 15% required); got {verdict}"
    )
    assert tracker.cooldown_remaining() == 0


def test_fast_drawdown_cooldown_ticks_then_resumes():
    """After breaker fires, cooldown ticks down; betting resumes when cooldown elapses."""
    initial = 100.0
    threshold = 0.15
    cooldown_rounds = 5  # short cooldown so the test is fast
    tracker = InMemoryBankrollTracker(
        initial_bankroll=initial, window_days=7, peak_mode="rolling_7d",
    )

    # Climb to peak then crash to fire the breaker.
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    crash_at = _BASE_START + _ROUND_SECONDS
    tracker.record_settlement(bankroll=85.0, start_at=crash_at)
    verdict = _evaluate_risk_gate(
        tracker,
        start_at=crash_at,
        max_drawdown_frac_from_peak=threshold,
        cooldown_rounds=cooldown_rounds,
    )
    assert verdict == "risk_drawdown_breaker_fired"
    assert tracker.cooldown_remaining() == cooldown_rounds

    # Each subsequent round during cooldown should observe is_paused -> ticks down.
    for i in range(1, cooldown_rounds + 1):
        sa = crash_at + i * _ROUND_SECONDS
        verdict = _evaluate_risk_gate(
            tracker,
            start_at=sa,
            max_drawdown_frac_from_peak=threshold,
            cooldown_rounds=cooldown_rounds,
        )
        assert verdict == "risk_cooldown_active", (
            f"round +{i}: expected cooldown active, got {verdict}"
        )
        assert tracker.cooldown_remaining() == cooldown_rounds - i

    # Next round: cooldown should be 0, betting resumes (verdict=ok).
    sa = crash_at + (cooldown_rounds + 1) * _ROUND_SECONDS
    # Move bankroll back up so dd is below threshold (otherwise the breaker
    # immediately re-fires the moment cooldown elapses).
    tracker.record_settlement(bankroll=95.0, start_at=sa)
    verdict = _evaluate_risk_gate(
        tracker,
        start_at=sa,
        max_drawdown_frac_from_peak=threshold,
        cooldown_rounds=cooldown_rounds,
    )
    assert verdict == "ok", (
        f"after cooldown elapsed + recovery: expected ok, got {verdict}"
    )
    assert tracker.cooldown_remaining() == 0
    assert tracker.is_paused(sa) is False


def test_fast_drawdown_breaker_disabled_when_threshold_one():
    """Sanity: max_drawdown_frac_from_peak=1.0 disables the breaker entirely."""
    tracker = InMemoryBankrollTracker(
        initial_bankroll=100.0, window_days=7, peak_mode="rolling_7d",
    )
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    # Drop to near-zero. Even this should NOT fire when threshold = 1.0.
    sa = _BASE_START + _ROUND_SECONDS
    tracker.record_settlement(bankroll=0.5, start_at=sa)
    verdict = _evaluate_risk_gate(
        tracker,
        start_at=sa,
        max_drawdown_frac_from_peak=1.0,
        cooldown_rounds=72,
    )
    # dd_frac = 0.995, threshold = 1.0, so dd_frac < threshold -> ok.
    assert verdict == "ok", f"threshold=1.0 should disable breaker; got {verdict}"


# ---------------------------------------------------------------------------
# R2 — Idempotency on claim retries
# ---------------------------------------------------------------------------

def test_idempotency_same_value_same_start_at_dedup():
    """Same (start_at, bankroll) called twice -> dedup, no double-count."""
    tracker = InMemoryBankrollTracker(
        initial_bankroll=100.0, window_days=7, peak_mode="rolling_7d",
    )
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    n_after_first = len(tracker._entries)
    # Retry: identical call.
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    n_after_retry = len(tracker._entries)
    assert n_after_retry == n_after_first, (
        f"identical retry must dedup; entries went from {n_after_first} -> {n_after_retry}"
    )
    assert tracker.current_bankroll() == 100.0


def test_idempotency_within_tolerance_same_start_at_dedup():
    """Float-tolerance dedup: 1e-13 difference must still dedup (line 138 < 1e-12)."""
    tracker = InMemoryBankrollTracker(
        initial_bankroll=100.0, window_days=7, peak_mode="rolling_7d",
    )
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    n_before = len(tracker._entries)
    # Within 1e-12 tolerance -> dedupes.
    tracker.record_settlement(bankroll=100.0 + 1e-13, start_at=_BASE_START)
    assert len(tracker._entries) == n_before, "1e-13 difference must dedup"


def test_idempotency_different_value_same_start_at_both_stored():
    """Different bankroll values, same start_at -> BOTH appended (history-honest).

    Pre-registered semantic: the dedup at bankroll_tracker.py:138 only matches
    on the LAST recorded value. Two retries with intermediate balance changes
    produce two distinct entries. ``current_bankroll`` returns the last write.
    """
    tracker = InMemoryBankrollTracker(
        initial_bankroll=100.0, window_days=7, peak_mode="rolling_7d",
    )
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    n_after_seed = len(tracker._entries)
    # Different value, same start_at (e.g., a retry where balance changed
    # between the two RPC reads).
    tracker.record_settlement(bankroll=99.5, start_at=_BASE_START)
    assert len(tracker._entries) == n_after_seed + 1, (
        "different value at same start_at must append a new entry"
    )
    # Last-write wins for current_bankroll.
    assert tracker.current_bankroll() == 99.5
    # Peak still reflects the maximum across all entries.
    assert tracker.peak_bankroll(_BASE_START) == 100.0


def test_idempotency_does_not_falsely_fire_breaker_on_retry():
    """Same-value retry on a settlement above threshold must NOT re-fire breaker.

    Scenario: breaker fires once on a 15% drawdown; a duplicate
    record_settlement (e.g., claim TX retry on chain re-org) must not
    re-trigger set_paused or extend cooldown.
    """
    threshold = 0.15
    cooldown_rounds = 72
    tracker = InMemoryBankrollTracker(
        initial_bankroll=100.0, window_days=7, peak_mode="rolling_7d",
    )
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    crash_at = _BASE_START + _ROUND_SECONDS
    tracker.record_settlement(bankroll=85.0, start_at=crash_at)
    verdict = _evaluate_risk_gate(
        tracker,
        start_at=crash_at,
        max_drawdown_frac_from_peak=threshold,
        cooldown_rounds=cooldown_rounds,
    )
    assert verdict == "risk_drawdown_breaker_fired"
    cooldown_after_first = tracker.cooldown_remaining()
    assert cooldown_after_first == cooldown_rounds

    # Retry: identical record_settlement call. The tracker dedupes (entry not
    # appended). The pipeline-level risk gate is invoked separately by the
    # live loop, but a same-iteration retry won't re-evaluate the gate (the
    # iteration already returned SKIP). What we DO verify here is that the
    # tracker state is unchanged after the retry.
    n_before_retry = len(tracker._entries)
    cooldown_before_retry = tracker.cooldown_remaining()
    tracker.record_settlement(bankroll=85.0, start_at=crash_at)
    assert len(tracker._entries) == n_before_retry, (
        "retry must not append a duplicate entry"
    )
    assert tracker.cooldown_remaining() == cooldown_before_retry, (
        "retry must not modify cooldown counter"
    )


def test_idempotency_seeded_init_entry_is_dedup_safe():
    """First record_settlement seeds an 'init' entry at the initial bankroll.

    A second record_settlement at the same (start_at, bankroll) as the seed
    must not append a third entry (dedup against the seed).
    """
    tracker = InMemoryBankrollTracker(
        initial_bankroll=100.0, window_days=7, peak_mode="rolling_7d",
    )
    # First call seeds an init entry AND appends a settlement entry IF the
    # value differs from initial. Same value -> only the init entry exists.
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    n_after_first = len(tracker._entries)
    assert n_after_first == 1, (
        f"first call with bankroll==initial should leave only the init entry; "
        f"got {n_after_first} entries"
    )
    # Retry: identical call.
    tracker.record_settlement(bankroll=100.0, start_at=_BASE_START)
    assert len(tracker._entries) == n_after_first, "retry must dedup"
