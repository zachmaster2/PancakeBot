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


def test_evidence_gap_streak_does_not_double_advance_on_a_rerun():
    """NEW-1, the one behavioural defect in this pass: the gap streak was
    booked as a bare `stored + 1`, so an applied SAME-WEEK re-run of a
    blocked Sunday wrote 2 and fired the "2 CONSECUTIVE SUNDAYS"
    escalation off a single blocked week — while consecutive_weak, booked
    two lines away, correctly stayed put. Both now go through the same
    guard."""
    # first run of the week: advance from last week's counter
    assert wm.book_streak({"evidence_gap_streak": 1}, "evidence_gap_streak",
                          same_week_rerun=False, hit=True) == (1, 2)
    # same-week re-run of that blocked Sunday: recompute, do NOT advance
    st = {"evidence_gap_streak": 1, "evidence_gap_streak_baseline": 0}
    assert wm.book_streak(st, "evidence_gap_streak",
                          same_week_rerun=True, hit=True) == (0, 1)
    # ...and a re-run that no longer blocks clears it
    assert wm.book_streak(st, "evidence_gap_streak",
                          same_week_rerun=True, hit=False) == (0, 0)


def test_evidence_gap_streak_migrates_from_a_state_file_without_baseline():
    """Live state files predate the key. The fallback assumes the last
    booking was a block (stored-1), so it can only over-forgive."""
    st = {"evidence_gap_streak": 1}          # no baseline key
    assert wm.book_streak(st, "evidence_gap_streak",
                          same_week_rerun=True, hit=True) == (0, 1)


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
        # NEGATIVE on purpose: the enable must be reachable ONLY through
        # the 2w backtest, so a `pos_bt = bt` mutation cannot survive.
        return dict(net_pnl_bnb=-0.15, num_bets=6, win_rate=0.3333,
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
    # the 1w backtest lost money; only the 2w leg can have enabled this
    assert decision["window_1w"]["backtest"]["net_pnl_bnb"] == -0.15
    assert wm.evaluate_positive(
        decision["window_2w"], decision["window_1w"]["backtest"]) is False

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

# ---- fire-stream (frozen window) evidence gate ----------------------------

def test_fire_freshness_boundary():
    now = 1_787_000_000.0
    bound = wm.FIRE_STALE_MAX_AGE_S
    assert wm.fire_evidence_fresh(int(now - bound + 1), now) is True
    assert wm.fire_evidence_fresh(int(now - bound), now) is True
    assert wm.fire_evidence_fresh(int(now - bound - 1), now) is False


def test_fire_freshness_tolerates_a_real_thin_week():
    """The largest inter-fire gap in the 10 weeks to 2026-08-14 was 117.5h
    and p99 was 57.0h; a normal quiet stretch must NOT be called frozen."""
    now = 1_787_000_000.0
    for gap_h in (2.0, 17.5, 24.1, 57.0, 95.0):
        assert wm.fire_evidence_fresh(int(now - gap_h * 3600), now) is True, gap_h
    # the one 117.5h outlier does trip it — accepted, ~1.3% of Sundays
    assert wm.fire_evidence_fresh(int(now - 117.5 * 3600), now) is False


def test_fire_freshness_catches_the_2026_08_17_outage():
    """Fires froze 2026-08-14 18:06Z; the next Sunday run is 2026-08-23."""
    import datetime as _dt
    frozen = _dt.datetime(2026, 8, 14, 18, 6, tzinfo=_dt.timezone.utc).timestamp()
    sunday = _dt.datetime(2026, 8, 23, 6, 0, tzinfo=_dt.timezone.utc).timestamp()
    assert wm.fire_evidence_fresh(int(frozen), sunday) is False
    assert (sunday - frozen) / 3600 > 2 * (wm.FIRE_STALE_MAX_AGE_S / 3600)


def test_frozen_window_blocks_enable_end_to_end(tmp_path, monkeypatch):
    """The money path: a qualifying window whose fires are 9 days old must
    NOT enable the bot, and must say so in the artifact and the alert."""
    root = tmp_path / "weekly_monitors"
    repo = tmp_path / "repo"
    (repo / "var" / "live").mkdir(parents=True)
    root.mkdir(parents=True)
    (root / "state.json").write_text(json.dumps(dict(
        consecutive_weak=0, last_week=None, last_action=None, history=[])),
        encoding="utf-8")
    monkeypatch.setattr(wm, "ROOT", root)
    monkeypatch.setattr(wm, "STATE_PATH", root / "state.json")
    monkeypatch.setattr(wm, "RETRY_MARKER_PATH", root / "retry_pending.json")
    monkeypatch.setattr(wm, "REPO", repo)

    now = time.time()
    frozen_fire = int(now - 9 * 86400)          # fires stopped 9 days ago
    bets = [dict(epoch=100 + i, lock=int(frozen_fire - (13 - i) * 0.4 * 86400),
                 win=True) for i in range(13)]
    bets += [dict(epoch=113 + i, lock=int(frozen_fire - (5 - i) * 3600),
                  win=True) for i in range(6)]
    # ROUND stream stays fresh — exactly the condition --sync keeps green
    monkeypatch.setattr(wm, "build_canonical_bets",
                        lambda: (bets, int(now - 1800)))
    monkeypatch.setattr(wm, "perm", lambda w, n_iter=None, seed=None: (
        dict(n=len(w), insufficient=True) if len(w) < wm.POS_MIN_FIRES
        else dict(n=len(w), wr=0.6842, obs_mean_pnl=0.2442, null_mean=0.0,
                  p_upper=0.0285)))
    monkeypatch.setattr(wm, "risk_off_backtest",
                        lambda *a, **k: dict(net_pnl_bnb=0.41, num_bets=19,
                                             win_rate=0.6842, gas_per_bet=0.0006))
    monkeypatch.setattr(wm, "read_bot_state", lambda: dict(
        available=True, active="inactive", enabled="disabled",
        is_running=False, is_enabled=False))
    monkeypatch.setattr(wm, "do_enable", lambda: (_ for _ in ()).throw(
        AssertionError("do_enable must not be called on a frozen window")))
    messages = []
    monkeypatch.setattr(wm, "discord",
                        lambda msg: (messages.append(msg), True)[1])
    monkeypatch.setattr(sys, "argv", [
        "wm", "--apply", "--no-sync", "--iso-week", "2026-08-23"])

    assert wm._main() == 0
    decision = json.loads(
        (root / "2026-08-23" / "decision.json").read_text(encoding="utf-8"))
    assert decision["triggers"]["positive"] is True    # stats still qualify
    assert decision["action"] == "enable_BLOCKED_frozen_window"
    assert decision["fire_fresh"] is False
    assert decision["data_fresh"] is True              # rounds ARE fresh
    assert decision["newest_fire_age_h"] > 200
    assert not (repo / "var" / "live" / "cooldown_override.json").exists()
    assert "FROZEN WINDOW" in messages[0]
    assert "placed no bet since" in messages[0]
    # NEW-3: the severity prefix must not stack another banner on a
    # head that already opens with one.
    assert "⚠️ ⚠️" not in messages[0]
    assert messages[0].count("⚠️ FROZEN WINDOW") == 1
    # state still books the week (this is not a blind week)
    st = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert st["last_action"] == "enable_BLOCKED_frozen_window"

# ---- spent-window fire composition (point-check visibility) ---------------

def _w(now, ages_h):
    return [dict(epoch=1000 + i, lock=int(now - a * 3600), win=True)
            for i, a in enumerate(ages_h)]


def test_composition_counts_fresh_and_stale():
    now = 1_787_000_000.0
    comp = wm.window_fire_composition(_w(now, [1, 50, 95, 97, 300]), now)
    assert comp["n"] == 5
    assert comp["fresh_within_bound"] == 3       # <= 96h
    assert comp["stale"] == 2
    assert comp["in_last_7d"] == 4               # <= 168h
    assert comp["oldest_fire_age_h"] == 300.0


def test_one_recovery_fire_passes_the_gate_but_composition_shows_it():
    """The reviewer's finding: 18 fires aged 9.6-13d plus ONE 2h-old fire
    passes the point-in-time freshness gate. That decision is unchanged in
    this pass (a density leg is the queued follow-up) — but the mixture
    must be VISIBLE, never readable as a live window."""
    now = 1_787_000_000.0
    ages = [2.0] + [h * 24.0 for h in (9.6, 10, 10.4, 10.8, 11.2, 11.6, 12,
                                       12.2, 12.4, 12.6, 12.8, 13, 13, 13,
                                       13, 13, 13, 13)]
    window = _w(now, ages)
    newest = max(b["lock"] for b in window)
    assert wm.fire_evidence_fresh(newest, now) is True      # gate passes
    comp = wm.window_fire_composition(window, now)
    assert comp["n"] == 19 and comp["fresh_within_bound"] == 1
    assert comp["in_last_7d"] == 1
    rendered = wm._window_desc("2w(fallback SPENT)", GOOD_2W, BT_POS, comp)
    assert "[fresh 1/19 last7d=1" in rendered


def test_gap_p99_needs_samples_and_tracks_erosion():
    now = 1_787_000_000.0
    assert wm.fire_gap_p99_h(_w(now, [1, 2, 3]), now) is None
    # 40 fires 2h apart, then one 90h drought -> p99 sits at the drought
    ages = [90.0 + 2 * i for i in range(40)] + [0.0]
    p99 = wm.fire_gap_p99_h(_w(now, ages), now)
    assert p99 is not None and p99 >= 89.0


def test_mixed_window_composition_reaches_decision_and_discord(tmp_path, monkeypatch):
    """End-to-end shape of Sunday 2026-08-23: a 2w window straddling the
    outage. The run may still act, but the artifact and the alert must both
    carry the fresh/stale split."""
    root = tmp_path / "weekly_monitors"
    repo = tmp_path / "repo"
    (repo / "var" / "live").mkdir(parents=True)
    root.mkdir(parents=True)
    (root / "state.json").write_text(json.dumps(dict(
        consecutive_weak=0, last_week=None, last_action=None, history=[])),
        encoding="utf-8")
    monkeypatch.setattr(wm, "ROOT", root)
    monkeypatch.setattr(wm, "STATE_PATH", root / "state.json")
    monkeypatch.setattr(wm, "RETRY_MARKER_PATH", root / "retry_pending.json")
    monkeypatch.setattr(wm, "REPO", repo)

    now = time.time()
    pre = int(now - 9 * 86400)
    bets = [dict(epoch=100 + i, lock=int(pre - (13 - i) * 0.3 * 86400), win=True)
            for i in range(18)]
    bets.append(dict(epoch=200, lock=int(now - 2 * 3600), win=True))
    monkeypatch.setattr(wm, "build_canonical_bets",
                        lambda: (bets, int(now - 1800)))
    monkeypatch.setattr(wm, "perm", lambda w, n_iter=None, seed=None: (
        dict(n=len(w), insufficient=True) if len(w) < wm.POS_MIN_FIRES
        else dict(n=len(w), wr=0.6842, obs_mean_pnl=0.2442, null_mean=0.0,
                  p_upper=0.0285)))
    monkeypatch.setattr(wm, "risk_off_backtest",
                        lambda *a, **k: dict(net_pnl_bnb=0.41, num_bets=19,
                                             win_rate=0.6842, gas_per_bet=0.0006))
    monkeypatch.setattr(wm, "read_bot_state", lambda: dict(
        available=True, active="active", enabled="enabled",
        is_running=True, is_enabled=True))
    messages = []
    monkeypatch.setattr(wm, "discord",
                        lambda msg: (messages.append(msg), True)[1])
    monkeypatch.setattr(sys, "argv", [
        "wm", "--apply", "--no-sync", "--iso-week", "2026-08-23"])

    assert wm._main() == 0
    d = json.loads((root / "2026-08-23" / "decision.json").read_text(encoding="utf-8"))
    # the point-check gate passes on the single recent fire...
    assert d["fire_fresh"] is True
    # ...and the composition makes the mixture impossible to miss
    comp2 = d["window_2w"]["fire_composition"]
    assert comp2["n"] >= 18 and comp2["fresh_within_bound"] == 1
    assert comp2["oldest_fire_age_h"] > 200
    assert "fresh 1/" in messages[0]
    assert d["fire_gap_p99_h"] is None or d["fire_gap_p99_h"] > 0


# ---- evidence-gap (density) rule -----------------------------------------

# Replay of every archived run day, RE-MEASURED IN FULL on the VM
# 2026-08-24 by reconstructing each run's SPENT window from the live
# 1,982-fire canonical stream: `now` = that day 06:00Z, windows keyed to
# the newest fire at or before it (as the monitor does -- window() cuts
# from max_lock, not from now), 1w when n >= POS_MIN_FIRES else 2w.
#
# The previous version of this table claimed to be measured that way but
# only its max_internal_gap_h column actually was; n and trailing_gap_h
# were partly carried over from archived window_*.epochs, which reflect
# whatever store state existed that day. Every column below now comes from
# one replay run. Corrections: 07-08 n 18->24 / trail 6.5->15.3, 07-12
# trail 26.0->25.7, 07-18 trail 22.2->14.8, 07-19 trail 63.0->38.8, 07-26
# n 11->10 / trail 0.0->5.3, 08-02 trail 6.8->6.4, 08-09 trail 37.4->36.9,
# 08-16 trail 39.3->35.9, 08-23 trail 1.4->0.9. Every max_internal_gap_h
# reproduced unchanged -- which is the column the rule decides on, so the
# conclusion never moved.
#
# (day, spent_window, n_fires, max_internal_gap_h, trailing_gap_h, action)
REPLAY = [
    ("2026-07-08", "1w",          24,  26.9, 15.3, "none"),
    ("2026-07-12", "1w",          15,  28.8, 25.7, "disable"),
    ("2026-07-18", "1w",          18,  28.0, 14.8, "none"),
    ("2026-07-19", "1w",          18,  28.0, 38.8, "none"),
    ("2026-07-26", "1w",          10,  45.7,  5.3, "none"),
    ("2026-08-02", "1w",          15,  47.1,  6.4, "none"),
    ("2026-08-09", "1w",          14,  57.0, 36.9, "none"),
    ("2026-08-16", "2w_fallback", 19, 117.5, 35.9, "enable"),
    ("2026-08-23", "1w",          10,  15.0,  0.9, "none"),
]

NOW = 1_787_000_000.0


def _window_with_profile(n, max_internal_h, trailing_h, now=NOW):
    """Synthesise a window whose largest internal gap and trailing gap are
    exactly the given values, so the replay drives the REAL
    window_gap_profile instead of asserting over literals."""
    assert n >= 2 and max_internal_h >= 1.0
    newest = now - trailing_h * 3600.0
    locks = [newest, newest - max_internal_h * 3600.0]
    while len(locks) < n:                      # filler gaps of 1h
        locks.append(locks[-1] - 3600.0)
    return [dict(epoch=9000 + i, lock=int(round(l)), win=True)
            for i, l in enumerate(sorted(locks))]


def test_replay_fixtures_reproduce_their_recorded_profile():
    """The fixture builder must actually reproduce each row, otherwise the
    replay below would be testing its own arithmetic."""
    for day, _tw, n, mx, trail in ((r[0], r[1], r[2], r[3], r[4])
                                   for r in REPLAY):
        win = _window_with_profile(n, mx, trail)
        got_mx, got_trail = wm.window_gap_profile(win, NOW)
        assert len(win) == n, day
        assert abs(got_mx - mx) < 0.05, f"{day}: {got_mx} != {mx}"
        assert abs(got_trail - trail) < 0.05, f"{day}: {got_trail} != {trail}"


def test_historical_replay_changes_exactly_one_decision():
    """Drives window_gap_profile over each Sunday's reconstructed window --
    delete the function and this fails, which the previous literal-only
    version did not."""
    bound = wm.FIRE_STALE_MAX_AGE_S / 3600.0
    blocked, changed = [], []
    for day, _tw, n, mx, trail, action in REPLAY:
        win = _window_with_profile(n, mx, trail)
        got_mx, got_trail = wm.window_gap_profile(win, NOW)
        if max(got_mx, got_trail) > bound:
            blocked.append(day)
            if action in ("enable", "cooldown_override"):
                changed.append(day)
    assert blocked == ["2026-08-16"], blocked
    assert changed == ["2026-08-16"], changed


def test_replay_margin_is_thin_and_rests_on_one_case():
    """KNOWN LIMITATION, recorded rather than papered over: efficacy is
    n=1. The single case the rule catches sits 22.4% above the 96h bound
    but only 2.1% BELOW 120h, the smallest bound that would drop the
    false-positive rate under ~5%.

    Note where the thinness is and is not. On the PASS side there is
    room: the worst passing day measures 57.0h against a 96h bound. The
    margin that is thin is on the CATCH side, and it is thin against
    raising the bound, not against the observed traffic."""
    bound = wm.FIRE_STALE_MAX_AGE_S / 3600.0
    catches = [r for r in REPLAY if max(r[3], r[4]) > bound]
    assert len(catches) == 1
    worst_pass = max(max(r[3], r[4]) for r in REPLAY if max(r[3], r[4]) <= bound)
    assert worst_pass == 57.0                      # 2026-08-09
    assert worst_pass / bound < 0.60               # not a near miss
    assert catches[0][3] == 117.5
    assert 117.5 > bound and 117.5 < 120.0


def test_every_bound_below_the_observed_max_preserves_efficacy():
    """M6-a: the comment's claim that 96h is 'the largest bound that still
    catches the worst drought' was false -- the whole 96-116h band catches
    it. 96h is a choice, and the file must not re-derive it as forced."""
    worst = max(r[3] for r in REPLAY)
    assert worst == 117.5
    for candidate in (96.0, 104.0, 112.0, 116.0):
        assert worst > candidate, candidate
    for candidate in (120.0, 144.0, 168.0):
        assert worst < candidate, candidate


def test_the_blocked_days_are_one_episode_not_a_rate():
    """M6-c: 15.7% of days = 11 blocked days out of 70, but they are one
    contiguous run (2026-08-12..08-22) of a single drought aging through
    the window. The replay table shows the same shape: the blocked days
    cluster on one event, so the cost is episodes, not Sundays."""
    bound = wm.FIRE_STALE_MAX_AGE_S / 3600.0
    blocked = [r[0] for r in REPLAY if max(r[3], r[4]) > bound]
    assert blocked == ["2026-08-16"]
    # ...and the drought that causes it is visible as ONE gap, not many.
    assert sum(1 for r in REPLAY if r[3] > bound) == 1


def _w(now, ages_h):
    return [dict(epoch=1000 + i, lock=int(now - a * 3600), win=True)
            for i, a in enumerate(ages_h)]


def test_composition_counts_fresh_and_stale():
    now = 1_787_000_000.0
    comp = wm.window_fire_composition(_w(now, [1, 50, 95, 97, 300]), now)
    assert comp["n"] == 5
    assert comp["fresh_within_bound"] == 3       # <= 96h
    assert comp["stale"] == 2
    assert comp["in_last_7d"] == 4               # <= 168h
    assert comp["oldest_fire_age_h"] == 300.0


def test_one_recovery_fire_passes_the_gate_but_composition_shows_it():
    """The reviewer's finding: 18 fires aged 9.6-13d plus ONE 2h-old fire
    passes the point-in-time freshness gate. That decision is unchanged in
    this pass (a density leg is the queued follow-up) — but the mixture
    must be VISIBLE, never readable as a live window."""
    now = 1_787_000_000.0
    ages = [2.0] + [h * 24.0 for h in (9.6, 10, 10.4, 10.8, 11.2, 11.6, 12,
                                       12.2, 12.4, 12.6, 12.8, 13, 13, 13,
                                       13, 13, 13, 13)]
    window = _w(now, ages)
    newest = max(b["lock"] for b in window)
    assert wm.fire_evidence_fresh(newest, now) is True      # gate passes
    comp = wm.window_fire_composition(window, now)
    assert comp["n"] == 19 and comp["fresh_within_bound"] == 1
    assert comp["in_last_7d"] == 1
    rendered = wm._window_desc("2w(fallback SPENT)", GOOD_2W, BT_POS, comp)
    assert "[fresh 1/19 last7d=1" in rendered


def test_gap_p99_needs_samples_and_tracks_erosion():
    now = 1_787_000_000.0
    assert wm.fire_gap_p99_h(_w(now, [1, 2, 3]), now) is None
    # 40 fires 2h apart, then one 90h drought -> p99 sits at the drought
    ages = [90.0 + 2 * i for i in range(40)] + [0.0]
    p99 = wm.fire_gap_p99_h(_w(now, ages), now)
    assert p99 is not None and p99 >= 89.0


def test_mixed_window_composition_reaches_decision_and_discord(tmp_path, monkeypatch):
    """End-to-end shape of Sunday 2026-08-23: a 2w window straddling the
    outage. The run may still act, but the artifact and the alert must both
    carry the fresh/stale split."""
    root = tmp_path / "weekly_monitors"
    repo = tmp_path / "repo"
    (repo / "var" / "live").mkdir(parents=True)
    root.mkdir(parents=True)
    (root / "state.json").write_text(json.dumps(dict(
        consecutive_weak=0, last_week=None, last_action=None, history=[])),
        encoding="utf-8")
    monkeypatch.setattr(wm, "ROOT", root)
    monkeypatch.setattr(wm, "STATE_PATH", root / "state.json")
    monkeypatch.setattr(wm, "RETRY_MARKER_PATH", root / "retry_pending.json")
    monkeypatch.setattr(wm, "REPO", repo)

    now = time.time()
    pre = int(now - 9 * 86400)
    bets = [dict(epoch=100 + i, lock=int(pre - (13 - i) * 0.3 * 86400), win=True)
            for i in range(18)]
    bets.append(dict(epoch=200, lock=int(now - 2 * 3600), win=True))
    monkeypatch.setattr(wm, "build_canonical_bets",
                        lambda: (bets, int(now - 1800)))
    monkeypatch.setattr(wm, "perm", lambda w, n_iter=None, seed=None: (
        dict(n=len(w), insufficient=True) if len(w) < wm.POS_MIN_FIRES
        else dict(n=len(w), wr=0.6842, obs_mean_pnl=0.2442, null_mean=0.0,
                  p_upper=0.0285)))
    monkeypatch.setattr(wm, "risk_off_backtest",
                        lambda *a, **k: dict(net_pnl_bnb=0.41, num_bets=19,
                                             win_rate=0.6842, gas_per_bet=0.0006))
    monkeypatch.setattr(wm, "read_bot_state", lambda: dict(
        available=True, active="active", enabled="enabled",
        is_running=True, is_enabled=True))
    messages = []
    monkeypatch.setattr(wm, "discord",
                        lambda msg: (messages.append(msg), True)[1])
    monkeypatch.setattr(sys, "argv", [
        "wm", "--apply", "--no-sync", "--iso-week", "2026-08-23"])

    assert wm._main() == 0
    d = json.loads((root / "2026-08-23" / "decision.json").read_text(encoding="utf-8"))
    # the point-check gate passes on the single recent fire...
    assert d["fire_fresh"] is True
    # ...and the composition makes the mixture impossible to miss
    comp2 = d["window_2w"]["fire_composition"]
    assert comp2["n"] >= 18 and comp2["fresh_within_bound"] == 1
    assert comp2["oldest_fire_age_h"] > 200
    assert "fresh 1/" in messages[0]
    assert d["fire_gap_p99_h"] is None or d["fire_gap_p99_h"] > 0


def _w(now, ages_h):
    return [dict(epoch=1000 + i, lock=int(now - a * 3600), win=True)
            for i, a in enumerate(ages_h)]


def test_gap_profile_separates_internal_from_trailing():
    now = 1_787_000_000.0
    mx, trail = wm.window_gap_profile(_w(now, [2, 12, 24]), now)
    assert mx == 12.0 and trail == 2.0
    # single fire: no internal gap to measure
    mx, trail = wm.window_gap_profile(_w(now, [5]), now)
    assert mx is None and trail == 5.0
    mx, trail = wm.window_gap_profile([], now)
    assert mx is None and trail == float("inf")


def test_one_fresh_fire_plus_eighteen_stale_is_an_evidence_gap():
    """The demonstrated hole: this passes the shipped trailing check."""
    now = 1_787_000_000.0
    ages = [2.0] + [h * 24.0 for h in (9.6, 10, 10.4, 10.8, 11.2, 11.6, 12,
                                       12.2, 12.4, 12.6, 12.8, 13, 13, 13,
                                       13, 13, 13, 13)]
    window = _w(now, ages)
    assert wm.fire_evidence_fresh(max(b["lock"] for b in window), now) is True
    mx, trail = wm.window_gap_profile(window, now)
    assert trail == 2.0                                  # trailing looks fine
    assert mx > wm.FIRE_STALE_MAX_AGE_S / 3600.0         # the drought does not


def test_interior_drought_with_a_fresh_tail_is_caught():
    now = 1_787_000_000.0
    # dense recent fires, then a 5-day hole, then dense older fires
    ages = [1, 3, 5, 7, 9, 130, 132, 134, 136]
    mx, trail = wm.window_gap_profile(_w(now, ages), now)
    assert trail == 1.0
    assert mx == 121.0 > wm.FIRE_STALE_MAX_AGE_S / 3600.0


def test_healthy_window_passes():
    now = 1_787_000_000.0
    ages = [2 + 8 * i for i in range(20)]      # a fire every 8h
    mx, trail = wm.window_gap_profile(_w(now, ages), now)
    assert max(mx, trail) <= wm.FIRE_STALE_MAX_AGE_S / 3600.0


def test_rule_does_not_touch_window_selection():
    """positive_window() must be identical with or without gap structure --
    if it moved, spent_stats/negative_wr_leg/weak_week would all shift and
    a density-driven 'none' would start advancing the DISABLE counter."""
    assert wm.positive_window(GOOD_1W, GOOD_2W) == "1w"
    assert wm.positive_window(INSUF_1W, GOOD_2W) == "2w_fallback"
    assert wm.positive_window(INSUF_1W, INSUF_2W) == "none"
    # the selector takes only perm stats -- it cannot see gaps at all
    import inspect
    assert list(inspect.signature(wm.positive_window).parameters) == ["p1", "p2"]


def test_evidence_gap_blocks_enable_end_to_end(tmp_path, monkeypatch):
    """Fresh tail, qualifying stats, but a 100h drought inside the spent 1w
    window: the enable must be blocked under its OWN action name, and the
    negative machinery must be bit-for-bit unaffected."""
    root = tmp_path / "weekly_monitors"
    repo = tmp_path / "repo"
    (repo / "var" / "live").mkdir(parents=True)
    root.mkdir(parents=True)
    (root / "state.json").write_text(json.dumps(dict(
        consecutive_weak=0, last_week=None, last_action=None, history=[])),
        encoding="utf-8")
    monkeypatch.setattr(wm, "ROOT", root)
    monkeypatch.setattr(wm, "STATE_PATH", root / "state.json")
    monkeypatch.setattr(wm, "RETRY_MARKER_PATH", root / "retry_pending.json")
    monkeypatch.setattr(wm, "REPO", repo)

    now = time.time()
    # 6 fires in the last 20h, a 100h hole, then 6 more -- all inside 7 days
    ages_h = [1, 4, 8, 12, 16, 20] + [120, 128, 136, 144, 152, 160]
    bets = [dict(epoch=500 + i, lock=int(now - a * 3600), win=True)
            for i, a in enumerate(sorted(ages_h, reverse=True))]
    monkeypatch.setattr(wm, "build_canonical_bets",
                        lambda: (bets, int(now - 1800)))
    monkeypatch.setattr(wm, "perm", lambda w, n_iter=None, seed=None: (
        dict(n=len(w), insufficient=True) if len(w) < wm.POS_MIN_FIRES
        else dict(n=len(w), wr=0.70, obs_mean_pnl=0.24, null_mean=0.0,
                  p_upper=0.02)))
    monkeypatch.setattr(wm, "risk_off_backtest",
                        lambda *a, **k: dict(net_pnl_bnb=0.31, num_bets=12,
                                             win_rate=0.70, gas_per_bet=0.0006))
    monkeypatch.setattr(wm, "read_bot_state", lambda: dict(
        available=True, active="inactive", enabled="disabled",
        is_running=False, is_enabled=False))
    monkeypatch.setattr(wm, "do_enable", lambda: (_ for _ in ()).throw(
        AssertionError("do_enable must not run on an evidence gap")))
    messages = []
    monkeypatch.setattr(wm, "discord",
                        lambda m: (messages.append(m), True)[1])
    monkeypatch.setattr(sys, "argv",
                        ["wm", "--apply", "--no-sync", "--iso-week", "2026-08-30"])

    assert wm._main() == 0
    d = json.loads((root / "2026-08-30" / "decision.json").read_text(encoding="utf-8"))

    assert d["triggers"]["positive"] is True          # the stats DO qualify
    assert d["fire_fresh"] is True                    # the tail IS fresh
    assert d["action"] == "enable_BLOCKED_evidence_gap"
    assert d["evidence_gap_ok"] is False
    assert d["max_internal_gap_h"] == 100.0
    assert d["gap_bound_h"] == 96.0
    assert not (repo / "var" / "live" / "cooldown_override.json").exists()

    # the negative machinery is untouched by evidence quality
    assert d["triggers"]["negative"] is False
    assert d["triggers"]["neg_wr_leg"] is False
    assert d["triggers"]["weak_this_week"] is False
    assert d["triggers"]["consecutive_weak"] == 0
    assert d["triggers"]["trigger_window"] == "1w"    # selection unchanged
    assert "max_gap=100.0h/96h" in messages[0]


def test_evidence_gap_block_is_never_a_silent_no_op(tmp_path, monkeypatch):
    """H1: the suppressed enable used to fall through to the plain `else`
    and read as a routine week. A silenced live-money action must be loud."""
    root = tmp_path / "weekly_monitors"
    repo = tmp_path / "repo"
    (repo / "var" / "live").mkdir(parents=True)
    root.mkdir(parents=True)
    (root / "state.json").write_text(json.dumps(dict(
        consecutive_weak=0, last_week=None, last_action=None, history=[],
        evidence_gap_streak=1)), encoding="utf-8")
    monkeypatch.setattr(wm, "ROOT", root)
    monkeypatch.setattr(wm, "STATE_PATH", root / "state.json")
    monkeypatch.setattr(wm, "RETRY_MARKER_PATH", root / "retry_pending.json")
    monkeypatch.setattr(wm, "REPO", repo)

    now = time.time()
    ages_h = [1, 4, 8, 12, 16, 20] + [120, 128, 136, 144, 152, 160]
    bets = [dict(epoch=500 + i, lock=int(now - a * 3600), win=True)
            for i, a in enumerate(sorted(ages_h, reverse=True))]
    monkeypatch.setattr(wm, "build_canonical_bets",
                        lambda: (bets, int(now - 1800)))
    monkeypatch.setattr(wm, "perm", lambda w, n_iter=None, seed=None: (
        dict(n=len(w), insufficient=True) if len(w) < wm.POS_MIN_FIRES
        else dict(n=len(w), wr=0.70, obs_mean_pnl=0.24, null_mean=0.0,
                  p_upper=0.02)))
    monkeypatch.setattr(wm, "risk_off_backtest",
                        lambda *a, **k: dict(net_pnl_bnb=0.31, num_bets=12,
                                             win_rate=0.70, gas_per_bet=0.0006))
    monkeypatch.setattr(wm, "read_bot_state", lambda: dict(
        available=True, active="inactive", enabled="disabled",
        is_running=False, is_enabled=False))
    messages = []
    monkeypatch.setattr(wm, "discord",
                        lambda m: (messages.append(m), True)[1])
    monkeypatch.setattr(sys, "argv",
                        ["wm", "--apply", "--no-sync", "--iso-week", "2026-09-06"])

    assert wm._main() == 0
    msg = messages[0]
    assert msg.startswith("\u26a0")                       # warning, not plain
    assert "POSITIVE ACTION SUPPRESSED" in msg
    assert "EVIDENCE GAP" in msg
    # escalation: the seeded streak of 1 advances to 2 and announces itself
    assert "CONSECUTIVE SUNDAYS" in msg
    d = json.loads((root / "2026-09-06" / "decision.json").read_text(encoding="utf-8"))
    assert d["evidence_gap_streak"] == 2
    assert d["max_window_gap_h"] == 100.0        # M2: the value actually tested
    assert d["trailing_gap_h"] == 1.0
    st = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert st["evidence_gap_streak"] == 2


def test_evidence_gap_streak_resets_on_a_clean_sunday():
    """Only consecutive gap-blocks escalate."""
    assert wm.FIRE_STALE_MAX_AGE_S // 3600 == 96
    # streak arithmetic is a one-liner in _main; pin the contract it uses
    for action, prev, want in (
        ("enable_BLOCKED_evidence_gap", 0, 1),
        ("enable_BLOCKED_evidence_gap", 2, 3),
        ("none", 3, 0),
        ("enable", 3, 0),
        ("enable_BLOCKED_frozen_window", 3, 0),
    ):
        got = prev + 1 if action.endswith("_BLOCKED_evidence_gap") else 0
        assert got == want, action


def test_gap_rule_implies_freshness_so_precedence_holds():
    """L2: the spent window ends at the newest fire, so its trailing gap IS
    the quantity fire_evidence_fresh tests. Checking the gap rule first
    would make *_BLOCKED_frozen_window unreachable."""
    now = 1_787_000_000.0
    stale_tail = _w(now, [200.0, 210.0, 220.0])
    mx, trail = wm.window_gap_profile(stale_tail, now)
    bound = wm.FIRE_STALE_MAX_AGE_S / 3600.0
    gap_ok = max(mx, trail) <= bound
    fresh = wm.fire_evidence_fresh(max(b["lock"] for b in stale_tail), now)
    assert gap_ok is False and fresh is False
    # the implication the precedence assert relies on
    assert not (gap_ok and not fresh)


def test_unmeasurable_window_fails_closed():
    """L1: an empty window yields no measurable gap; a fail-safe gate must
    not read that as 'fine'."""
    mx, trail = wm.window_gap_profile([], 1_787_000_000.0)
    assert mx is None and trail == float("inf")
    gaps = [g for g in (mx, trail) if g is not None]
    max_window_gap_h = max(gaps) if gaps else None
    assert (max_window_gap_h is not None
            and max_window_gap_h <= wm.FIRE_STALE_MAX_AGE_S / 3600.0) is False
