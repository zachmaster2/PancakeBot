"""The weekly monitor must describe the bot that is actually trading —
and must degrade safely when it cannot tell which bot that is.

Until 2026-08-24 `build_canonical_bets()` built its strategy config from
`load_strategy_config_from_dict({})` — CODE DEFAULTS — while the risk-off
backtest copied `config.toml`. When the pool filter moved 1.5 -> 1.25 the
two halves of the positive trigger started describing different bots, and
the monitor's fire stream ran 29.3% short of the live one all-history
(51.4% over the last 70 days).

The disable path is the exposed one: replaying window selection, this
stream sat at EXACTLY `POS_MIN_FIRES` (n=10) on 2026-07-26 and
2026-08-23 while the deployed filter gave 18 and 11. One fire lower and
`trigger_window` becomes "none", which books `weak=True`; three
consecutive weak weeks auto-disable the live unit.

The first fix for that introduced a worse bug — see
`test_unreadable_config_does_not_stop_a_protective_disable`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "weekly_monitor_config_tracking",
    REPO / "research" / "weekly_monitor_state_machine.py")
wm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wm)

from pancakebot.config import load_app_config  # noqa: E402


# ---- the fingerprint -----------------------------------------------------

def test_fingerprint_reports_the_deployed_values():
    """Not a restatement of config.toml: it loads through the same loader
    the bot uses, so a change in either moves this together."""
    fp = wm.strategy_fingerprint()
    s = load_app_config(str(REPO / "config.toml")).strategy
    assert fp["min_pool_bnb_at_cutoff"] == s.pool_filter.min_pool_bnb_at_cutoff
    assert fp["min_payout_multiple_at_cutoff"] == \
        s.pool_filter.min_payout_multiple_at_cutoff
    assert fp["max_bet_bnb_btc_primary"] == s.risk.max_bet_bnb_btc_primary
    assert fp["max_bet_bnb_eth_sol_fallback"] == \
        s.risk.max_bet_bnb_eth_sol_fallback
    assert fp["max_bet_fraction_of_bankroll"] == \
        s.risk.max_bet_fraction_of_bankroll
    assert "error" not in fp


def test_fingerprint_captures_the_knobs_that_actually_drift():
    """The pool filter changes WHICH ROUNDS FIRE (so n/wr/p_upper); the
    stake caps scale every btPnL; the module constants below are NOT read
    from config at all, which is why their drift is invisible without
    this."""
    fp = wm.strategy_fingerprint()
    for key in ("min_pool_bnb_at_cutoff", "max_bet_bnb_btc_primary",
                "max_bet_bnb_eth_sol_fallback", "kline_cutoff_seconds",
                "mtf_lookbacks_used_for_slicing", "mtf_lookbacks_deployed",
                "pool_cutoff_seconds", "treasury_fee_fraction",
                "min_bet_threshold_bnb"):
        assert key in fp, key


def test_the_lookbacks_drift_is_visible_because_it_is_half_wired():
    """MED-A2's sharpest case: the gate is handed the DEPLOYED lookbacks
    while the kline slice is cut with the module's own LOOKBACKS, so
    raising the deployed value past the constant silently starves the gate
    of history. Recording both makes that divergence readable."""
    fp = wm.strategy_fingerprint()
    assert fp["mtf_lookbacks_used_for_slicing"] == list(wm.LOOKBACKS)
    assert fp["mtf_lookbacks_deployed"] == fp["mtf_lookbacks_used_for_slicing"], (
        "deployed lookbacks have drifted from the slicing constant — the "
        "gate is being starved of history")


def test_stake_cap_inertness_is_contingent_not_structural():
    """The caps do not change which rounds fire only because a capped
    stake still clears the contract minimum. Recorded so the contingency
    is checkable rather than assumed."""
    fp = wm.strategy_fingerprint()
    assert fp["min_bet_threshold_bnb"] >= wm.MIN_BET_AMOUNT_BNB
    assert min(fp["max_bet_bnb_btc_primary"],
               fp["max_bet_bnb_eth_sol_fallback"]) >= wm.MIN_BET_AMOUNT_BNB


@pytest.mark.parametrize("field", [
    "min_pool_bnb_at_cutoff", "max_bet_bnb_btc_primary",
    "max_bet_bnb_eth_sol_fallback", "max_bet_fraction_of_bankroll",
    "min_payout_multiple_at_cutoff", "kline_cutoff_seconds",
    "pool_cutoff_seconds", "treasury_fee_fraction", "min_bet_threshold_bnb",
])
def test_fingerprint_values_are_numeric_and_positive(field):
    """A None or 0 here would make the artifact lie about which bot ran."""
    v = wm.strategy_fingerprint()[field]
    assert isinstance(v, (int, float)) and v > 0, (field, v)


# ---- MED-A1: the safety regression --------------------------------------

def test_unreadable_config_falls_back_to_defaults_instead_of_raising(
        tmp_path, monkeypatch):
    """MED-A1. Reading the deployed config UNGUARDED turned an unreadable
    config.toml into rc=1 with no decision.json, no systemd action and no
    protective disable — while the bot stayed enabled and trading. Before
    that, an unreadable config was invisible here and protection was
    intact. So the fix must degrade, not raise."""
    monkeypatch.setattr(wm, "REPO", tmp_path)
    sc, err = wm.deployed_strategy()
    assert err is not None
    assert sc is not None
    # fell back to the code defaults, which is the pre-change behaviour
    assert sc.pool_filter.min_pool_bnb_at_cutoff == 1.5


def test_unreadable_config_is_recorded_in_the_fingerprint(
        tmp_path, monkeypatch):
    monkeypatch.setattr(wm, "REPO", tmp_path)
    fp = wm.strategy_fingerprint()
    assert "error" in fp
    assert fp["fell_back_to_defaults"] is True
    assert "min_pool_bnb_at_cutoff" not in fp


def test_build_canonical_bets_uses_the_guarded_loader():
    """The stream must go through deployed_strategy(), not call the
    loader directly — that directness WAS the regression."""
    src = (REPO / "research" / "weekly_monitor_state_machine.py").read_text(
        encoding="utf-8")
    body = src.split("def build_canonical_bets", 1)[1]
    assert "deployed_strategy()" in body
    assert "load_app_config" not in body.split("def ", 1)[0]


def test_only_deployed_strategy_may_reach_code_defaults():
    """LOW-A4, renamed from 'can no longer reach code defaults' — which
    was wrong twice over. Config loading resolves PER KEY, so an absent
    key still yields a default; and defaults are now a DELIBERATE
    fallback. What must hold is that the fallback is confined to the one
    guarded function."""
    src = (REPO / "research" / "weekly_monitor_state_machine.py").read_text(
        encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#"))
    # the import plus exactly one call site, inside deployed_strategy()
    assert code.count("load_strategy_config_from_dict") == 2
    after = code.split("def deployed_strategy", 1)[1]
    assert "load_strategy_config_from_dict" in after.split("def ", 1)[0]


# ---- LOW-A5: the other degrade paths ------------------------------------

@pytest.mark.parametrize("content", [
    "",                                   # empty file
    "this is not toml at all {{{",        # unparseable
    "[strategy.pool_filter]\n",           # section present, keys absent
    "[strategy.pool_filter]\nmin_pool_bnb_at_cutoff = -1\n",   # invalid value
    "[strategy.pool_filter]\nmin_pool_bnb_at_cutoff = 'x'\n",  # wrong type
])
def test_every_bad_config_degrades_rather_than_raising(
        tmp_path, monkeypatch, content):
    """LOW-A5. Whatever is wrong with config.toml, the weekly run must
    still complete: it carries a dead-man contract, so a raise costs a
    Sunday. Some of these load fine (per-key defaults) — the requirement
    is only that none of them raise."""
    (tmp_path / "config.toml").write_text(content, encoding="utf-8")
    monkeypatch.setattr(wm, "REPO", tmp_path)
    sc, err = wm.deployed_strategy()          # must not raise
    assert sc is not None
    fp = wm.strategy_fingerprint()            # must not raise
    assert isinstance(fp, dict)


# ---- MED-A3(b): the discontinuity must be recorded ----------------------

def test_window_blocks_explain_n_versus_backtest_num_bets():
    """MED-1: the two counts sit side by side in every window block and
    are not the same quantity. 2026-07-26 recorded n=10 beside
    num_bets~18 over an identical epoch range with nothing saying why."""
    note = wm._N_VS_BACKTEST_NOTE
    assert "canonical" in note and "backtest.num_bets" in note
    assert "strategy_fingerprint" in note
