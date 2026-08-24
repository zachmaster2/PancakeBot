"""Absolute per-bet cap at 0.05 BNB: what actually sets the stake.

`_compute_bet_size` clamps in order bankroll-fraction -> absolute cap ->
floor, and on live pool depths the adaptive terms never survive: the raw
payout-proportional bet lands far above both caps, so the bot flat-stakes
at whichever cap binds. The code reads as adaptive sizing and is not.

These pin the post-change behaviour (0.05 flat above a 1.0 BNB bankroll,
bankroll-fraction below it) and the fact that admission is untouched.
"""
import pytest

from pancakebot.config import load_app_config
from pancakebot.strategy.momentum_pipeline import _compute_bet_size

# Real production sizing parameters (config.toml, btc_primary).
BASE_FRAC = 0.04
SLOPE = 100.0
MAX_POOL_FRACTION = 0.3
FEE = 0.03
FLOOR = 0.01
BANKROLL_FRAC = 0.05

# A live-shaped round: pool ~2.72 BNB, our side 0.535.
POOL = 2.72
OUR_SIDE = 0.535
# Weakest signal the BTC-primary gate admits.
WEAKEST_SIGNAL = 0.0002


def _size(signal, cap, bankroll, pool=POOL, our_side=OUR_SIDE):
    return _compute_bet_size(
        signal_strength=signal, pool_bnb=pool, our_side_bnb=our_side,
        base_frac=BASE_FRAC, cap_bnb=cap, pool_fraction_slope=SLOPE,
        max_pool_fraction=MAX_POOL_FRACTION, treasury_fee_fraction=FEE,
        min_bet_threshold_bnb=FLOOR, current_bankroll=bankroll,
        max_bet_fraction_of_bankroll=BANKROLL_FRAC,
    )


def _raw(signal, pool=POOL, our_side=OUR_SIDE):
    """Unclamped bet, to show the adaptive terms never bind."""
    return _compute_bet_size(
        signal_strength=signal, pool_bnb=pool, our_side_bnb=our_side,
        base_frac=BASE_FRAC, cap_bnb=1e9, pool_fraction_slope=SLOPE,
        max_pool_fraction=MAX_POOL_FRACTION, treasury_fee_fraction=FEE,
        min_bet_threshold_bnb=FLOOR, current_bankroll=None,
    )


# ---- the finding: this is flat-staking, not adaptive sizing --------------

def test_even_the_weakest_admissible_signal_exceeds_both_caps():
    """Raw ~0.11 BNB at the admission threshold, rising to ~0.66 saturated.
    Every one is above 0.1 and far above 0.05, so the cap is the stake."""
    weakest = _raw(WEAKEST_SIGNAL)
    assert weakest > 0.10, weakest
    mid = _raw(0.001)
    saturating = _raw(0.01)
    assert weakest < mid <= saturating          # adaptive in principle...
    assert min(weakest, mid, saturating) > 0.05  # ...never in practice


def test_stake_is_the_cap_regardless_of_signal_strength():
    for signal in (WEAKEST_SIGNAL, 0.0005, 0.001, 0.01, 0.1):
        assert _size(signal, cap=0.05, bankroll=2.0) == pytest.approx(0.05)


# ---- post-change sizing --------------------------------------------------

def test_bankroll_at_or_above_one_bnb_yields_the_flat_cap():
    """0.05 fraction x 1.0 BNB == the 0.05 absolute cap, so at and above a
    1.0 bankroll the absolute cap binds and the stake is flat."""
    for bankroll in (1.0, 1.5, 1.889, 2.081, 5.0):
        assert _size(WEAKEST_SIGNAL, cap=0.05, bankroll=bankroll) == \
            pytest.approx(0.05)


def test_below_one_bnb_the_bankroll_fraction_binds_again():
    """The risk floor resumes control -- that is why the bankroll fraction
    is kept rather than being the lever that was changed."""
    assert _size(WEAKEST_SIGNAL, cap=0.05, bankroll=0.8) == pytest.approx(0.04)
    assert _size(WEAKEST_SIGNAL, cap=0.05, bankroll=0.5) == pytest.approx(0.025)
    assert _size(WEAKEST_SIGNAL, cap=0.05, bankroll=0.4) == pytest.approx(0.02)


def test_the_change_halves_the_stake_where_it_used_to_bind():
    """Above a 2.0 BNB bankroll the old absolute cap of 0.1 was binding;
    the observed live stakes of 0.0945-0.1000 sat exactly there."""
    for bankroll in (2.0, 2.081, 3.0):
        assert _size(WEAKEST_SIGNAL, cap=0.10, bankroll=bankroll) == \
            pytest.approx(0.10)
        assert _size(WEAKEST_SIGNAL, cap=0.05, bankroll=bankroll) == \
            pytest.approx(0.05)


def test_floor_still_wins_on_a_tiny_bankroll():
    """min_bet_threshold_bnb is applied last and has never bound in
    production; it still guards the bottom."""
    assert _size(WEAKEST_SIGNAL, cap=0.05, bankroll=0.05) == pytest.approx(FLOOR)


# ---- admission is unchanged ----------------------------------------------

def test_every_stake_clears_the_contract_minimum():
    """The only size-dependent gate compares against the on-chain minimum
    (0.001 BNB). The 0.01 floor clears it at every bankroll, so the same
    rounds are bet -- just smaller."""
    contract_min = 0.001
    for bankroll in (0.05, 0.4, 0.8, 1.0, 2.081, 5.0):
        assert _size(WEAKEST_SIGNAL, cap=0.05, bankroll=bankroll) >= contract_min


# ---- the config itself ---------------------------------------------------

def test_config_carries_the_new_caps_and_keeps_the_risk_floor():
    cfg = load_app_config("config.toml")
    risk = cfg.strategy.risk
    assert risk.max_bet_bnb_btc_primary == 0.05
    assert risk.max_bet_bnb_eth_sol_fallback == 0.05
    # unchanged on purpose: the dilution optimum is absolute, the fraction
    # is the risk control
    assert risk.max_bet_fraction_of_bankroll == 0.05


def test_min_bet_only_key_is_still_present_and_false():
    """The code default is True; if this key were ever dropped from
    config.toml every bet would silently clamp to the contract minimum."""
    import re
    with open("config.toml", encoding="utf-8") as f:
        body = f.read()
    assert re.search(r"^min_bet_only\s*=\s*false\s*$", body, re.M)
    assert load_app_config("config.toml").live_min_bet_only is False
