"""The weekly monitor must describe the bot that is actually trading.

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
    stake caps scale every btPnL. Both are silent-drift class, which is
    why they are recorded together."""
    fp = wm.strategy_fingerprint()
    for key in ("min_pool_bnb_at_cutoff", "max_bet_bnb_btc_primary",
                "max_bet_bnb_eth_sol_fallback"):
        assert key in fp, key


def test_fingerprint_degrades_instead_of_killing_the_weekly_run(
        tmp_path, monkeypatch):
    """A reporting field must never cost a Sunday. The run carries a
    dead-man contract, so an unreadable config is recorded, not raised."""
    monkeypatch.setattr(wm, "REPO", tmp_path)
    fp = wm.strategy_fingerprint()
    assert "error" in fp
    assert "min_pool_bnb_at_cutoff" not in fp


def test_the_monitor_can_no_longer_reach_code_defaults():
    """Regression guard on the actual defect: the module must not retain a
    path that silently substitutes defaults for the deployed config."""
    src = (REPO / "research" / "weekly_monitor_state_machine.py").read_text(
        encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#"))
    assert "load_strategy_config_from_dict" not in code
    assert "load_app_config" in code


def test_window_blocks_explain_n_versus_backtest_num_bets():
    """MED-1: the two counts sit side by side in every window block and
    are not the same quantity. 2026-07-26 recorded n=10 next to
    num_bets~18 over an identical epoch range with nothing saying why."""
    note = wm._N_VS_BACKTEST_NOTE
    assert "canonical" in note and "backtest.num_bets" in note
    assert "strategy_fingerprint" in note


@pytest.mark.parametrize("field", [
    "min_pool_bnb_at_cutoff", "max_bet_bnb_btc_primary",
    "max_bet_bnb_eth_sol_fallback", "max_bet_fraction_of_bankroll",
    "min_payout_multiple_at_cutoff",
])
def test_fingerprint_values_are_numeric_and_positive(field):
    """A None or 0 here would make the artifact lie about which bot ran."""
    v = wm.strategy_fingerprint()[field]
    assert isinstance(v, (int, float)) and v > 0, (field, v)
