"""The disable leg must reason about a BOUND, not a point estimate.

WHY. The leg was `wr < NEG_WR_FLOOR` — the raw weekly win rate against a
fixed number. A weekly window carries n ~ 10-50 fires, so SE(WR) is
~0.07-0.15 and the test fires deep inside the noise. Concretely, at n=15
and WR=0.40 the raw floor disables while the 95% upper bound on that same
observation is 0.6404: the evidence cannot tell 0.40 from comfortably
above breakeven.

Raising the floor to 0.52, as first proposed, would have made it worse.
With a true win rate sitting exactly at BREAKEVEN_WR, a raw floor at 0.52
disables a healthy strategy 23.7% of weeks at n=50 and 37.4% at n=20 —
roughly one week in four, on noise.

The enable side already reasons about a bound (`p_upper < POS_RAW_P`).
This makes the disable side consistent: fire only when even the optimistic
end of the interval cannot reach breakeven.

BACKTESTED ON THE REAL SERIES, 8 evaluable weeks. The bound rule fires
once — 2026-07-12, n=15 WR=0.20 upper95=0.4398, the week the bot genuinely
was disabled. The raw floor at 0.52 fires twice, adding 2026-08-02
(n=15 WR=0.40 upper95=0.6404). Those cases are the fixtures below.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import research.weekly_monitor_state_machine as wm  # noqa: E402


def _leg(n: int, wr: float, window: str = "1w") -> bool:
    return wm.negative_wr_leg(window, dict(n=n, wr=wr))


# ---- the bound itself -----------------------------------------------------

def test_upper_bound_is_exact_not_a_normal_approximation():
    """At weekly n the approximation moves the decision: n=15, wins=6 reads
    0.6081 normal against 0.6404 exact, and the bar sits between them."""
    assert wm.wr_upper_bound(6, 15) == __import__("pytest").approx(0.6404, abs=5e-4)
    assert wm.wr_upper_bound(3, 15) == __import__("pytest").approx(0.4398, abs=5e-4)
    assert wm.wr_upper_bound(0, 10) == __import__("pytest").approx(0.2589, abs=5e-4)


def test_upper_bound_degenerates_safely():
    assert wm.wr_upper_bound(10, 10) == 1.0     # all wins -> no upper info
    assert wm.wr_upper_bound(0, 0) == 1.0       # no data -> no upper info


def test_more_evidence_tightens_the_bound():
    """The whole point: the same point estimate at larger n is more
    disable-able, because the evidence is stronger."""
    loose = wm.wr_upper_bound(round(0.40 * 15), 15)
    tight = wm.wr_upper_bound(round(0.40 * 50), 50)
    assert tight < loose
    assert loose > wm.BREAKEVEN_WR > tight


# ---- the real weekly series ----------------------------------------------

def test_the_one_real_disable_still_fires():
    """2026-07-12, n=15 WR=0.20. The week the bot genuinely was disabled."""
    assert _leg(15, 0.20) is True


def test_the_noise_driven_disable_no_longer_fires():
    """2026-08-02, n=15 WR=0.40, upper95=0.6404. THE regression: the raw
    floor disabled a strategy that 15 observations could not distinguish
    from healthy."""
    assert _leg(15, 0.40) is False


def test_every_healthy_week_in_the_real_series_is_left_alone():
    for n, wr in ((18, 0.5556), (10, 0.60), (14, 0.6429),
                  (19, 0.6842), (10, 0.70), (50, 0.52)):
        assert _leg(n, wr) is False, f"n={n} wr={wr} disabled a live strategy"


def test_sundays_window_does_not_disable():
    """2026-08-30: n=50 WR=0.52, upper95=0.6427. Under the originally
    proposed raw floor of 0.52 this survived only by strict `<` and would
    have disabled on any drop at all."""
    assert _leg(50, 0.52) is False


# ---- catastrophe is still caught -----------------------------------------

def test_a_catastrophic_week_still_disables_at_the_smallest_admissible_n():
    """POS_MIN_FIRES gates the spent window at n>=10, and the bound rule is
    comfortably decisive there — so no raw floor is needed beneath it."""
    assert wm.POS_MIN_FIRES >= 10
    assert _leg(10, 0.0) is True
    assert wm.wr_upper_bound(0, 10) < wm.BREAKEVEN_WR


def test_the_raw_floor_is_gone():
    """One disable criterion, not two. A raw floor beneath the bound rule
    would re-introduce exactly the noise-firing this change removes."""
    assert not hasattr(wm, "NEG_WR_FLOOR"), (
        "NEG_WR_FLOOR is back — two overlapping disable criteria is how "
        "surprising interactions get in")


# ---- both sides move together --------------------------------------------

def test_the_leg_is_keyed_off_breakeven_not_a_separate_number():
    """Moving what 'breakeven' means must tighten the enable bar and the
    disable bar coherently, instead of leaving a silent gap."""
    n, wins = 20, 11                       # wr 0.55, upper95 ~0.7455
    base = wm.wr_upper_bound(wins, n)
    orig = wm.BREAKEVEN_WR
    try:
        wm.BREAKEVEN_WR = base + 0.01      # bar above the bound -> fires
        assert wm.negative_wr_leg("1w", dict(n=n, wr=wins / n)) is True
        wm.BREAKEVEN_WR = base - 0.01      # bar below the bound -> does not
        assert wm.negative_wr_leg("1w", dict(n=n, wr=wins / n)) is False
    finally:
        wm.BREAKEVEN_WR = orig


# ---- interaction with the weak-week path ---------------------------------

def test_the_two_disable_paths_do_not_double_count():
    """`neg_trigger = bound_leg OR consec >= 3`. They use DIFFERENT
    statistics (a CP bound on WR vs a permutation p on PnL) over different
    horizons (one week vs three), so they are complementary. Both true is
    still one disable."""
    losing = dict(n=15, wr=0.20, p_upper=0.9947)
    assert wm.negative_wr_leg("1w", losing) is True
    assert wm.weak_week("1w", losing) is True
    assert bool(wm.negative_wr_leg("1w", losing)
                or 1 >= wm.NEG_CONSECUTIVE_WEAK) is True


def test_a_weak_week_alone_does_not_disable():
    """Sunday 2026-08-30 booked weak (p=0.625) but is not statistically
    below breakeven. It must take three of them, not one."""
    sunday = dict(n=50, wr=0.52, p_upper=0.625)
    assert wm.weak_week("1w", sunday) is True
    assert wm.negative_wr_leg("1w", sunday) is False


def test_a_good_week_resets_the_weak_streak():
    """The streak self-resets, so the two paths cannot silently accumulate
    against each other."""
    st = {"consecutive_weak": 2}
    _, consec = wm.book_weak_week(st, same_week_rerun=False, weak=False)
    assert consec == 0


# ---- unevaluable windows --------------------------------------------------

def test_no_spendable_window_is_unevaluable():
    assert wm.negative_wr_leg("none", dict(n=50, wr=0.20)) is False


def test_missing_stats_do_not_fire():
    assert wm.negative_wr_leg("1w", dict(n=None, wr=None)) is False
    assert wm.negative_wr_leg("1w", dict(n=0, wr=0.0)) is False


# ---- the decision itself, pinned -----------------------------------------

def test_breakeven_wr_is_the_value_that_was_decided():
    """A DECISION-RECORDING assertion, not a behavioural one.

    Every other test here is written relative to wm.BREAKEVEN_WR so it
    survives a future move — which is right, but it means a silent revert
    of the constant changes what the bot does and breaks nothing. Mutation
    testing caught exactly that: setting it back to 0.55 left all 66 tests
    green.

    0.56 was chosen by the operator on 2026-08-30 as a deliberate
    tightening of the bar, and it now sets BOTH sides (the enable leg's
    `wr > BREAKEVEN_WR` and the disable leg's `upper95 < BREAKEVEN_WR`).
    Moving it is a live-money risk decision; this test makes that move
    explicit rather than silent.
    """
    assert wm.BREAKEVEN_WR == 0.56


def test_pos_raw_p_and_min_fires_are_unchanged():
    """The other two enable legs were NOT part of this change; pin them so
    a future edit to the leg cannot quietly drag them along."""
    assert wm.POS_RAW_P == 0.10
    assert wm.POS_MIN_FIRES == 10
    assert wm.NEG_WEAK_P == 0.5
    assert wm.NEG_CONSECUTIVE_WEAK == 3
