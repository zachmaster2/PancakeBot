"""Absolute per-bet cap at 0.05 BNB: which clamp actually sets the stake.

`_compute_bet_size` clamps in order bankroll-fraction -> absolute cap ->
floor. An earlier version of this file asserted that the adaptive terms
"never survive" -- that the raw payout-proportional bet always lands above
both caps. That was FALSE, and it was false in the way tautological tests
usually are: it was pinned to one hand-picked BTC-primary pool/signal pair,
so it could not fail no matter what the sizing path did.

Measured across all 1,982 canonical fires with their real pools and real
signal strengths: the minimum raw bet is 0.0249 BNB and 391 of 1,982
(19.7%) size below 0.10. The tests below are therefore driven by REAL
recorded inputs sampled across the raw-size distribution of BOTH regimes,
so they fail if the sizing path changes.

The corrected picture:
  * ETH/SOL FALLBACK is where sub-cap bets live -- 280 of the 391 sub-0.10
    cases and all 74 sub-0.05 cases. Two effects compound: it sizes off
    base_pool_fraction 0.02 against the primary's 0.04, and its admitted
    signals are far weaker (median 1.68e-4 vs 4.31e-4).
  * BTC PRIMARY never sizes below 0.05 -- but its minimum raw bet is
    0.0505, barely 1% above the new cap. The cap binds on all 1,433 BTC
    rounds, with almost no margin at the bottom.
  * At the live bankroll of 1.889 BNB the OLD 0.1 absolute cap bound 0
    times out of 1,982; max_bet_fraction_of_bankroll was the operative
    clamp. The change moves control TO the absolute cap (96.3% of rounds
    at any bankroll >= 1.0).
"""
import pytest

from pancakebot.config import load_app_config
from pancakebot.strategy.momentum_pipeline import _compute_bet_size

# Production sizing parameters shared by both regimes (config.toml).
SLOPE = 100.0
MAX_POOL_FRACTION = 0.3
FEE = 0.03
FLOOR = 0.01
BANKROLL_FRAC = 0.05

BTC_BASE_FRAC = 0.04           # [strategy.backtest.sizing]
ES_BASE_FRAC = 0.02            # [strategy.eth_sol_fallback.sizing]

LIVE_BANKROLL = 1.889          # ledger at the time of the change

# Real (signal_strength, pool_bnb, our_side_bnb) triples captured from the
# canonical fire stream, sampled at the min / p05 / p25 / p50 / p75 / p95 /
# max of each regime's raw-bet distribution, with the raw bet each one
# actually produces.
BTC_FIRES = [
    (0.00021273535,   1.5012708, 0.93981737, 0.0505),
    (0.0003898896548, 1.7860715, 1.0688375,  0.0876),
    (0.0007104373585, 1.8089402, 0.99846131, 0.1521),
    (0.0006613631055, 1.9304332, 0.83550397, 0.2543),
    (0.0003285661848, 2.3614603, 0.64102448, 0.4427),
    (0.000160053341,  3.0206999, 0.48060344, 0.8622),
    (0.0002832761513, 8.3776932, 1.4805352,  2.5133),
]
ES_FIRES = [
    (0.0001175672265, 1.5044364, 0.95982187, 0.0249),
    (0.0001090997275, 1.9077946, 1.1121843,  0.0391),
    (0.0001327277688, 1.6731807, 0.7673591,  0.0621),
    (0.0001992477872, 1.5180322, 0.56645278, 0.0969),
    (0.0002149617267, 2.8310842, 1.152206,   0.1625),
    (9.814269016e-05, 4.0669179, 0.82011831, 0.4620),
    (0.000180189119,  4.5835065, 0.5272959,  1.2950),
]
REGIMES = (("btc_primary", BTC_BASE_FRAC, BTC_FIRES),
           ("eth_sol_fallback", ES_BASE_FRAC, ES_FIRES))

# Population facts from the same measurement (n = 1,982).
POP_MIN_RAW = 0.0249
POP_BELOW_010 = 391
POP_BELOW_005 = 74


def _size(base_frac, fire, cap, bankroll, floor=FLOOR):
    signal, pool, our_side = fire[:3]
    return _compute_bet_size(
        signal_strength=signal, pool_bnb=pool, our_side_bnb=our_side,
        base_frac=base_frac, cap_bnb=cap, pool_fraction_slope=SLOPE,
        max_pool_fraction=MAX_POOL_FRACTION, treasury_fee_fraction=FEE,
        min_bet_threshold_bnb=floor, current_bankroll=bankroll,
        max_bet_fraction_of_bankroll=BANKROLL_FRAC,
    )


def _raw(base_frac, fire):
    """The unclamped adaptive bet, with every clamp disarmed."""
    return _size(base_frac, fire, cap=1e9, bankroll=None, floor=0.0)


# ---- the sizing path itself ----------------------------------------------

@pytest.mark.parametrize("regime,base_frac,fires", REGIMES)
def test_real_fires_reproduce_their_measured_raw_sizes(regime, base_frac, fires):
    """Regression pin on real recorded inputs. If the adaptive terms, the
    payout multiplier or the fee handling change, these move."""
    for signal, pool, our_side, expected in fires:
        got = _raw(base_frac, (signal, pool, our_side))
        assert got == pytest.approx(expected, abs=5e-5), (
            f"{regime} signal={signal} pool={pool}: {got} != {expected}")


def test_adaptive_sizing_does_reach_below_the_caps():
    """The claim this file used to make -- that every raw bet clears both
    caps -- is false. Sub-cap bets are a real and regular occurrence."""
    es_raws = [_raw(ES_BASE_FRAC, f) for f in ES_FIRES]
    btc_raws = [_raw(BTC_BASE_FRAC, f) for f in BTC_FIRES]
    assert min(es_raws) < 0.05, "ETH/SOL fallback sizes below the new cap"
    assert min(es_raws) == pytest.approx(POP_MIN_RAW, abs=5e-5)
    assert any(r < 0.10 for r in btc_raws), "BTC primary sizes below the old cap"
    assert POP_BELOW_005 > 0 and POP_BELOW_010 > POP_BELOW_005


def test_sub_cap_bets_are_the_eth_sol_fallback_regime():
    """All 74 sub-0.05 fires are fallback rounds. BTC primary's minimum
    clears the cap -- but by 1%, so that margin is worth watching."""
    btc_min = min(_raw(BTC_BASE_FRAC, f) for f in BTC_FIRES)
    es_min = min(_raw(ES_BASE_FRAC, f) for f in ES_FIRES)
    assert es_min < 0.05 < btc_min
    assert btc_min == pytest.approx(0.0505, abs=5e-5)
    assert btc_min / 0.05 < 1.02, "BTC primary barely clears the cap"


def test_the_two_regimes_differ_by_more_than_base_fraction():
    """Sizing the SAME round under both base fractions does not reproduce
    the gap: weaker admitted signals in the fallback regime compound it."""
    fire = ES_FIRES[0][:3]
    as_fallback = _raw(ES_BASE_FRAC, fire)
    as_primary = _raw(BTC_BASE_FRAC, fire)
    assert as_fallback < as_primary
    # ...and the fallback regime's own weakest signal is weaker than any
    # signal the primary gate admits in this sample.
    assert min(f[0] for f in ES_FIRES) < min(f[0] for f in BTC_FIRES)


# ---- which clamp binds, before and after ---------------------------------

@pytest.mark.parametrize("regime,base_frac,fires", REGIMES)
def test_at_the_live_bankroll_the_old_absolute_cap_never_bound(
        regime, base_frac, fires):
    """0.05 x 1.889 = 0.0945 < 0.10, so under the old cap the bankroll
    fraction (or the raw size) always won. This is why observed live stakes
    sat at 0.0945 and never at 0.1000."""
    for fire in fires:
        stake = _size(base_frac, fire, cap=0.10, bankroll=LIVE_BANKROLL)
        assert stake != pytest.approx(0.10)
        assert stake == pytest.approx(
            min(_raw(base_frac, fire), BANKROLL_FRAC * LIVE_BANKROLL))


@pytest.mark.parametrize("regime,base_frac,fires", REGIMES)
@pytest.mark.parametrize("bankroll", (1.0, 1.5, LIVE_BANKROLL, 2.081, 5.0))
def test_after_the_change_the_cap_binds_except_where_raw_is_smaller(
        regime, base_frac, fires, bankroll):
    """At any bankroll >= 1.0 the absolute cap is the operative clamp --
    but NOT universally: where the raw bet is already under 0.05 it passes
    through unchanged. That is the 3.7% the bankroll fraction still owns."""
    for fire in fires:
        raw = _raw(base_frac, fire)
        stake = _size(base_frac, fire, cap=0.05, bankroll=bankroll)
        assert stake == pytest.approx(min(0.05, max(raw, FLOOR)))
        if raw >= 0.05:
            assert stake == pytest.approx(0.05)


def test_the_crossover_is_exactly_one_bnb():
    """max_bet_fraction_of_bankroll x 1.0 == the new absolute cap, so the
    two clamps cross precisely at a 1.0 BNB bankroll."""
    assert BANKROLL_FRAC * 1.0 == pytest.approx(0.05)
    saturating = BTC_FIRES[-1][:3]
    assert _size(BTC_BASE_FRAC, saturating, cap=0.05, bankroll=1.0) == \
        pytest.approx(0.05)
    assert _size(BTC_BASE_FRAC, saturating, cap=0.05, bankroll=0.999) < 0.05


def test_below_one_bnb_the_bankroll_fraction_binds_again():
    """The risk floor resumes control -- that is why the bankroll fraction
    is kept rather than being the lever that was changed."""
    saturating = BTC_FIRES[-1][:3]
    for bankroll, expected in ((0.8, 0.04), (0.5, 0.025), (0.4, 0.02)):
        assert _size(BTC_BASE_FRAC, saturating, cap=0.05, bankroll=bankroll) \
            == pytest.approx(expected)


def test_floor_still_wins_on_a_tiny_bankroll():
    """min_bet_threshold_bnb is applied last and has never bound in
    production; it still guards the bottom."""
    assert _size(BTC_BASE_FRAC, BTC_FIRES[-1][:3], cap=0.05, bankroll=0.05) \
        == pytest.approx(FLOOR)


# ---- admission is unchanged ----------------------------------------------

@pytest.mark.parametrize("regime,base_frac,fires", REGIMES)
def test_every_stake_clears_the_contract_minimum(regime, base_frac, fires):
    """The only size-dependent gate compares against the on-chain minimum
    (0.001 BNB). The 0.01 floor clears it at every bankroll, so the same
    rounds are bet -- just smaller."""
    for bankroll in (0.05, 0.4, 0.8, 1.0, LIVE_BANKROLL, 5.0):
        for fire in fires:
            assert _size(base_frac, fire, cap=0.05, bankroll=bankroll) >= 0.001


# ---- the config itself ---------------------------------------------------

def test_config_carries_the_new_caps_and_keeps_the_risk_floor():
    cfg = load_app_config("config.toml")
    risk = cfg.strategy.risk
    assert risk.max_bet_bnb_btc_primary == 0.05
    assert risk.max_bet_bnb_eth_sol_fallback == 0.05
    # Unchanged on purpose: the dilution optimum is absolute, the fraction
    # is the risk control -- and it is NOT inert, it still binds below a
    # 1.0 bankroll and on the 3.7% of rounds that size under 0.05.
    assert risk.max_bet_fraction_of_bankroll == 0.05


def test_min_bet_only_key_is_still_present_and_false():
    """The code default is True; if this key were ever dropped from
    config.toml every bet would silently clamp to the contract minimum."""
    import re
    with open("config.toml", encoding="utf-8") as f:
        body = f.read()
    assert re.search(r"^min_bet_only\s*=\s*false\s*$", body, re.M)
    assert load_app_config("config.toml").live_min_bet_only is False
