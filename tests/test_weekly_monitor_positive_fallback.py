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
import json
import sys
import time
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
    tw = wm.positive_window(WEAK_1W, GOOD_2W)
    assert tw == "1w"
    assert wm.spent_stats(tw, WEAK_1W, GOOD_2W) is WEAK_1W
    assert wm.evaluate_positive(WEAK_1W, BT_POS) is False


def test_both_insufficient_cannot_fire():
    tw = wm.positive_window(INSUF_1W, INSUF_2W)
    assert tw == "none"
    # contract: with no spendable window the evaluator is handed the 1w
    # stats (insufficient) — the 2w stats are never bound to any leg
    handed = wm.spent_stats(tw, INSUF_1W, INSUF_2W)
    assert handed is INSUF_1W
    assert wm.evaluate_positive(handed, BT_POS) is False


def test_negative_wr_leg_follows_spent_window():
    # The reviewer's starved-1w regime: 2w evaluable at WR=0.40 with
    # 0.10 < p_upper <= 0.5 — not weak, but the WR floor must still bite.
    losing_2w = dict(n=19, wr=0.40, obs_mean_pnl=-0.12, null_mean=0.0,
                     p_upper=0.30)
    assert wm.weak_week("2w_fallback", losing_2w) is False
    assert wm.negative_wr_leg("2w_fallback", losing_2w) is True
    # healthy spent windows do not trip it
    assert wm.negative_wr_leg("2w_fallback", GOOD_2W) is False
    assert wm.negative_wr_leg("1w", GOOD_1W) is False
    assert wm.negative_wr_leg("1w", dict(GOOD_1W, wr=0.40)) is True
    # strict inequality at the floor
    assert wm.negative_wr_leg("1w", dict(GOOD_1W, wr=wm.NEG_WR_FLOOR)) is False
    # no spendable window: the leg is unevaluable (weak-booking covers it)
    assert wm.negative_wr_leg("none", INSUF_1W) is False


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


# ---- backtest window-identity guard ----------------------------------------

def test_backtest_summary_window_mismatch_is_error(tmp_path, monkeypatch):
    import types
    repo = tmp_path / "repo"
    (repo / "var" / "backtest").mkdir(parents=True)
    (repo / "config.toml").write_text(
        "[backtest]\ninitial_bankroll_bnb = 5.0\n# epoch_start = 0\n"
        "# epoch_end = 0\n\n[strategy.risk]\n"
        "max_drawdown_fraction_from_peak = 0.15\n"
        "min_bankroll_bnb_to_bet = 0.5\ncooldown_rounds = 3\n",
        encoding="utf-8")
    monkeypatch.setattr(wm, "REPO", repo)
    monkeypatch.setattr(wm, "subprocess", types.SimpleNamespace(
        run=lambda *a, **k: types.SimpleNamespace(returncode=0, stderr=""),
        TimeoutExpired=Exception))
    (repo / "var" / "backtest" / "summary.json").write_text(json.dumps(dict(
        first_epoch=100, last_epoch=118, net_pnl_bnb=0.41, num_bets=19,
        win_rate=0.6842)), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    assert wm.risk_off_backtest(100, 118, out) == dict(
        net_pnl_bnb=0.41, num_bets=19, win_rate=0.6842,
        gas_per_bet=wm.MAX_GAS_COST_BET_BNB)
    # a summary left behind by another window's run must never stand in —
    # the mismatch fails the PnL leg
    stale = wm.risk_off_backtest(200, 250, out)
    assert "window mismatch" in stale["error"]
    assert wm.evaluate_positive(GOOD_2W, stale) is False


# ---- end-to-end: fallback enable path through _main ------------------------

def test_fallback_enable_path_end_to_end(tmp_path, monkeypatch):
    """1w starved + strong 2w drives _main end-to-end: window selection ->
    spent stats/backtest binding -> enable action + override flag ->
    booking overwrite. The seeded state mimics the real 2026-08-16 Sunday
    run (weak-booked consec=1, pre-baseline schema), so this also pins the
    directed same-week re-run recomputing the booking 1 -> 0."""
    root = tmp_path / "weekly_monitors"
    repo = tmp_path / "repo"
    (repo / "var" / "live").mkdir(parents=True)
    (repo / "var" / "live" / "pause_state.json").write_text(
        json.dumps({"paused": True}), encoding="utf-8")
    root.mkdir(parents=True)
    (root / "state.json").write_text(json.dumps(dict(
        consecutive_weak=1, last_week="2026-08-16", last_action="none",
        history=[dict(week="2026-08-16", action="none", wr_1w=None,
                      p_1w=None, sidak=0.0563)])), encoding="utf-8")
    monkeypatch.setattr(wm, "ROOT", root)
    monkeypatch.setattr(wm, "STATE_PATH", root / "state.json")
    monkeypatch.setattr(wm, "RETRY_MARKER_PATH", root / "retry_pending.json")
    monkeypatch.setattr(wm, "REPO", repo)

    now = time.time()
    newest_fire = int(now - 3600)
    bets = [dict(epoch=100 + i, lock=int(newest_fire - (8 + i * 0.4) * 86400),
                 win=True) for i in range(13)]
    bets += [dict(epoch=113 + i, lock=int(newest_fire - (5 - i) * 86400),
                  win=True) for i in range(6)]
    monkeypatch.setattr(wm, "build_canonical_bets",
                        lambda: (bets, int(now - 1800)))

    def fake_perm(w, n_iter=None, seed=None):
        if len(w) < wm.POS_MIN_FIRES:
            return dict(n=len(w), insufficient=True)
        return dict(n=len(w), wr=0.6842, obs_mean_pnl=0.2442,
                    null_mean=0.0, p_upper=0.0285)
    monkeypatch.setattr(wm, "perm", fake_perm)

    bt_calls = []

    def fake_bt(epoch_start, epoch_end, out_dir, bankroll=5.0,
                cfg_name="risk_off_config.toml"):
        bt_calls.append((epoch_start, epoch_end, cfg_name))
        if cfg_name == "risk_off_config_2w.toml":
            return dict(net_pnl_bnb=0.41, num_bets=19, win_rate=0.6842,
                        gas_per_bet=0.0006)
        return dict(net_pnl_bnb=0.2693, num_bets=6, win_rate=0.8333,
                    gas_per_bet=0.0006)
    monkeypatch.setattr(wm, "risk_off_backtest", fake_bt)

    monkeypatch.setattr(wm, "read_bot_state", lambda: dict(
        available=True, active="inactive", enabled="disabled",
        is_running=False, is_enabled=False))
    monkeypatch.setattr(wm, "do_enable",
                        lambda: (True, "enable --now rc=0: ok"))
    messages = []
    monkeypatch.setattr(wm, "discord",
                        lambda msg: (messages.append(msg), True)[1])
    monkeypatch.setattr(sys, "argv", [
        "wm", "--apply", "--no-sync", "--iso-week", "2026-08-16"])

    assert wm._main() == 0

    decision = json.loads(
        (root / "2026-08-16" / "decision.json").read_text(encoding="utf-8"))
    assert decision["action"] == "enable"
    assert decision["triggers"]["positive"] is True
    assert decision["triggers"]["trigger_window"] == "2w_fallback"
    assert decision["triggers"]["negative"] is False
    assert decision["triggers"]["weak_this_week"] is False
    assert decision["triggers"]["consecutive_weak"] == 0
    assert decision["window_2w"]["backtest"]["net_pnl_bnb"] == 0.41

    # binding: 1w backtest over the 1w fires, then the REAL 2w run
    assert bt_calls == [(113, 118, "risk_off_config.toml"),
                        (100, 118, "risk_off_config_2w.toml")]

    st = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert st["consecutive_weak"] == 0            # retroactive 1 -> 0
    assert st["consecutive_weak_baseline"] == 0
    assert st["last_week"] == "2026-08-16"
    assert len(st["history"]) == 1                # replaced, not appended
    assert st["history"][0]["action"] == "enable"
    assert st["history"][0]["trigger_window"] == "2w_fallback"

    flag = json.loads((repo / "var" / "live" / "cooldown_override.json")
                      .read_text(encoding="utf-8"))
    assert flag["trigger_window"] == "2w_fallback"
    assert flag["window"]["wr"] == 0.6842

    assert len(messages) == 1
    assert "STATE CHANGED" in messages[0]
    assert "1w: n=6<10 insufficient" in messages[0]
    assert "2w(fallback SPENT)" in messages[0]
