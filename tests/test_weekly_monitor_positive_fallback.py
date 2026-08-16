"""Positive-trigger window selection + fallback legs for the weekly monitor.

Covers the pure helpers of research/weekly_monitor_state_machine.py added
for the 2w fallback (2026-08-16 user decision): `positive_window` (which
window the positive trigger spends — the POS_MIN_FIRES floor is an
information floor, not a time floor), `evaluate_positive` (the four legs
applied to the spent window's stats + that window's REAL risk-off
backtest), `_window_desc` (a starved window must render as insufficient,
never as 'WR=None p=None'), `weak_week` (weak is judged on the spent
window; both-starved counts weak), and `book_weak_week` (same-ISO-week
re-runs recompute the booking from the prior-week baseline — overwrite,
never freeze or double-advance).
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "weekly_monitor_state_machine",
    REPO / "research" / "weekly_monitor_state_machine.py")
wm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wm)

# Stats shapes exactly as perm() produces them.
INSUF_1W = dict(n=6, insufficient=True)
GOOD_1W = dict(n=15, wr=0.6667, obs_mean_pnl=0.21, null_mean=-0.01,
               p_upper=0.041)
WEAK_1W = dict(n=12, wr=0.5, obs_mean_pnl=-0.02, null_mean=-0.01,
               p_upper=0.61)
# The 2026-08-16 week that motivated the fallback.
GOOD_2W = dict(n=19, wr=0.6842, obs_mean_pnl=0.2442, null_mean=-0.005,
               p_upper=0.0285)
INSUF_2W = dict(n=9, insufficient=True)
BT_POS = dict(net_pnl_bnb=0.2693, num_bets=19, win_rate=0.6842,
              gas_per_bet=0.0006)
BT_NEG = dict(net_pnl_bnb=-0.11, num_bets=19, win_rate=0.5263,
              gas_per_bet=0.0006)


# ---- window selection -----------------------------------------------------

def test_sufficient_1w_spends_1w_even_when_2w_is_strong():
    assert wm.positive_window(GOOD_1W, GOOD_2W) == "1w"
    assert wm.positive_window(WEAK_1W, GOOD_2W) == "1w"


def test_insufficient_1w_falls_back_to_2w():
    assert wm.positive_window(INSUF_1W, GOOD_2W) == "2w_fallback"


def test_both_insufficient_spends_nothing():
    assert wm.positive_window(INSUF_1W, INSUF_2W) == "none"


# ---- the four legs on the spent window ------------------------------------

def test_fallback_with_qualifying_2w_fires():
    assert wm.positive_window(INSUF_1W, GOOD_2W) == "2w_fallback"
    assert wm.evaluate_positive(GOOD_2W, BT_POS) is True


def test_fallback_non_qualifying_2w_does_not_fire():
    # each leg fails independently
    assert wm.evaluate_positive(dict(GOOD_2W, p_upper=0.12), BT_POS) is False
    assert wm.evaluate_positive(dict(GOOD_2W, wr=0.52), BT_POS) is False
    assert wm.evaluate_positive(dict(GOOD_2W, n=9), BT_POS) is False
    assert wm.evaluate_positive(GOOD_2W, BT_NEG) is False


def test_backtest_error_or_missing_fails_pnl_leg():
    assert wm.evaluate_positive(GOOD_2W, {}) is False
    assert wm.evaluate_positive(GOOD_2W, dict(error="backtest timed out")) is False


def test_sufficient_1w_evaluates_1w_stats_not_2w():
    # A weak-but-evaluable 1w week must not borrow the strong 2w stats.
    spent = wm.positive_window(WEAK_1W, GOOD_2W)
    assert spent == "1w"
    assert wm.evaluate_positive(WEAK_1W, BT_POS) is False


def test_both_insufficient_cannot_fire():
    # _main hands the (insufficient) 1w stats to the evaluator on "none";
    # the not-insufficient leg keeps the trigger off.
    assert wm.evaluate_positive(INSUF_1W, BT_POS) is False


def test_boundary_values_do_not_fire():
    # legs are strict inequalities / the floor is >=
    at_bars = dict(n=10, wr=wm.BREAKEVEN_WR, obs_mean_pnl=0.0,
                   null_mean=0.0, p_upper=wm.POS_RAW_P)
    assert wm.evaluate_positive(at_bars, BT_POS) is False
    assert wm.evaluate_positive(dict(at_bars, wr=0.56), BT_POS) is False  # p at bar
    assert wm.evaluate_positive(
        dict(at_bars, wr=0.56, p_upper=0.09), dict(net_pnl_bnb=0.0)) is False
    assert wm.evaluate_positive(
        dict(at_bars, wr=0.56, p_upper=0.09), BT_POS) is True


# ---- message formatting ---------------------------------------------------

def test_insufficient_window_renders_as_starved_never_none():
    s = wm._window_desc("1w", INSUF_1W)
    assert s == "1w: n=6<10 insufficient"
    assert "None" not in s


def test_insufficient_window_with_backtest_appends_pnl():
    s = wm._window_desc("1w", INSUF_1W, BT_POS)
    assert s == "1w: n=6<10 insufficient btPnL=0.2693"


def test_evaluable_window_renders_legs():
    assert wm._window_desc("2w(info)", GOOD_2W) == \
        "2w(info): n=19 WR=0.6842 p=0.0285"


def test_fallback_spent_label_carries_2w_legs():
    s = wm._window_desc("2w(fallback SPENT)", GOOD_2W, BT_POS)
    assert s == "2w(fallback SPENT): n=19 WR=0.6842 p=0.0285 btPnL=0.2693"


def test_empty_backtest_dict_appends_nothing():
    assert wm._window_desc("1w", INSUF_1W, {}) == "1w: n=6<10 insufficient"


# ---- spent-window weak semantics ------------------------------------------

def test_spent_2w_strong_is_not_weak():
    # The 2026-08-16 shape: 1w starved, 2w spent with p=0.0285 -> NOT weak.
    assert wm.weak_week("2w_fallback", GOOD_2W) is False


def test_spent_2w_weak_p_books_weak():
    assert wm.weak_week("2w_fallback", dict(GOOD_2W, p_upper=0.61)) is True


def test_both_starved_books_weak():
    assert wm.weak_week("none", INSUF_1W) is True


def test_sufficient_1w_weak_semantics_unchanged():
    assert wm.weak_week("1w", WEAK_1W) is True       # p=0.61 > 0.5
    assert wm.weak_week("1w", GOOD_1W) is False      # p=0.041
    # bar is a strict inequality
    assert wm.weak_week("1w", dict(GOOD_1W, p_upper=wm.NEG_WEAK_P)) is False


# ---- weekly booking (baseline overwrite semantics) ------------------------

def test_booking_first_run_advances_and_resets():
    assert wm.book_weak_week({"consecutive_weak": 1},
                             same_week_rerun=False, weak=True) == (1, 2)
    assert wm.book_weak_week({"consecutive_weak": 2},
                             same_week_rerun=False, weak=False) == (2, 0)


def test_booking_same_week_rerun_recomputes_from_baseline():
    st = {"consecutive_weak": 1, "consecutive_weak_baseline": 0}
    assert wm.book_weak_week(st, same_week_rerun=True, weak=False) == (0, 0)
    # re-running a weak week must not double-advance
    assert wm.book_weak_week(st, same_week_rerun=True, weak=True) == (0, 1)
    st2 = {"consecutive_weak": 3, "consecutive_weak_baseline": 2}
    assert wm.book_weak_week(st2, same_week_rerun=True, weak=False) == (2, 0)


def test_booking_migration_default_covers_2026_08_16_state():
    # The one real pre-baseline state file: Sunday 2026-08-16 booked weak
    # under 1w-only semantics (consec 0->1, no baseline key persisted).
    # The directed same-week re-run (2w spent, p=0.0285 -> not weak) must
    # land consecutive_weak back at 0.
    st = {"consecutive_weak": 1}
    assert wm.book_weak_week(st, same_week_rerun=True, weak=False) == (0, 0)
