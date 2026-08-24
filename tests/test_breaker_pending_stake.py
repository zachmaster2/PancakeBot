"""The drawdown breaker must value an open position at COST, not at zero.

A PancakeSwap bet is payable: the stake leaves the wallet with the bet TX
and returns only when the claim lands. Reading the raw wallet balance as
"how we are doing" therefore counts every pending bet as a 100% loss of
its stake.

That fired the live breaker on 2026-08-24 18:41:20 UTC and suspended the
bot for ~23h on a measurement artifact. The numbers below are the real
ones from that incident, not a synthetic analogue.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime.bet_ledger import (  # noqa: E402
    _OPEN_STATUSES,
    _TERMINAL_STATUSES,
    open_stake_bnb,
)

# --- the 2026-08-24 18:41:20 UTC incident, exactly ------------------------
INCIDENT_PEAK = 2.131114364
INCIDENT_WALLET = 1.805402854      # raw balance the breaker compared
INCIDENT_STAKE = 0.05              # epoch 509852, CONFIRMED, later WON
THRESHOLD = 0.15


def _dd(peak: float, current: float) -> float:
    return (peak - current) / peak


def _ledger(tmp_path, records):
    p = tmp_path / "bets.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records),
                 encoding="utf-8")
    return str(p)


# ---- the incident ---------------------------------------------------------

def test_the_incident_does_not_trip_once_the_open_stake_is_valued_at_cost():
    """THE regression. Peak 2.131114364, wallet 1.805402854, one CONFIRMED
    0.05 position -> settled-equivalent 1.855402854 -> 12.94%, no trip."""
    settled = INCIDENT_WALLET + INCIDENT_STAKE
    dd = _dd(INCIDENT_PEAK, settled)
    assert dd == pytest.approx(0.129374, abs=1e-5)
    assert dd < THRESHOLD, "the 2026-08-24 suspension must not reproduce"


def test_the_same_numbers_with_no_open_position_still_trip():
    """The fix must not blunt the breaker: if that 0.05 had been a realised
    loss rather than an open position, 15.28% must still fire."""
    dd = _dd(INCIDENT_PEAK, INCIDENT_WALLET)
    assert dd == pytest.approx(0.152836, abs=1e-5)
    assert dd >= THRESHOLD, "a genuinely realised drawdown must still trip"


def test_the_incident_margin_was_smaller_than_one_stake():
    """Why this was invisible: the trip cleared the bar by 0.006 BNB, an
    eighth of a single 0.05 stake."""
    margin = (_dd(INCIDENT_PEAK, INCIDENT_WALLET) - THRESHOLD) * INCIDENT_PEAK
    assert 0.0 < margin < INCIDENT_STAKE
    assert margin == pytest.approx(0.006044, abs=1e-5)


# ---- open_stake_bnb uses the ledger's own semantics ----------------------

def test_confirmed_position_counts_as_open(tmp_path):
    """CONFIRMED, not just SUBMITTED. Epoch 509852 was CONFIRMED at the
    moment of the trip; a SUBMITTED-only test misses it."""
    path = _ledger(tmp_path, [
        {"epoch": 509852, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 509852, "status": "CONFIRMED", "gas_paid_bnb": 0.000119},
    ])
    assert open_stake_bnb(path) == pytest.approx(0.05)


@pytest.mark.parametrize("terminal", sorted(_TERMINAL_STATUSES))
def test_no_terminal_status_is_ever_added_back(tmp_path, terminal):
    """GUARD AGAINST THE REVERSE FAILURE: a stake that is added back but
    never settles would inflate the bankroll indefinitely and blind the
    breaker. Every terminal status must close that off -- including LATE
    and REVERTED, where the TX failed and the stake was never actually
    spent."""
    path = _ledger(tmp_path, [
        {"epoch": 1, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 1, "status": terminal},
    ])
    assert open_stake_bnb(path) == 0.0


def test_a_late_bet_is_not_added_back(tmp_path):
    """The concrete case: 509821 went LATE on 2026-08-24 (TX reverted, no
    position taken). Adding its stake back would overstate the bankroll."""
    path = _ledger(tmp_path, [
        {"epoch": 509821, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 509821, "status": "LATE", "gas_paid_bnb": 3.6e-05},
    ])
    assert open_stake_bnb(path) == 0.0


def test_multiple_open_positions_accumulate(tmp_path):
    """Resolution spans (p50 342s) exceed the round interval (~306s), so
    two positions can be open at once."""
    path = _ledger(tmp_path, [
        {"epoch": 1, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 1, "status": "CONFIRMED"},
        {"epoch": 2, "status": "SUBMITTED", "amount_bnb": 0.05},
    ])
    assert open_stake_bnb(path) == pytest.approx(0.10)


def test_settled_history_contributes_nothing(tmp_path):
    path = _ledger(tmp_path, [
        {"epoch": 1, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 1, "status": "SETTLED_LOST", "delta_bnb": -0.05},
        {"epoch": 2, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 2, "status": "SETTLED_WON", "delta_bnb": 0.0444},
    ])
    assert open_stake_bnb(path) == 0.0


def test_missing_or_unreadable_ledger_degrades_to_no_correction(tmp_path):
    """Applying no correction is the pre-fix behaviour, which errs toward
    tripping rather than toward trading. A read failure must not raise on
    the pre-lock path."""
    assert open_stake_bnb(str(tmp_path / "nope.jsonl")) == 0.0
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    assert open_stake_bnb(str(bad)) == 0.0


def test_open_statuses_are_imported_not_restated():
    """Three hand-rolled versions of this test have been wrong. The helper
    must derive from the ledger's own definition."""
    assert _OPEN_STATUSES == frozenset({"SUBMITTED", "CONFIRMED"})
    assert not (_OPEN_STATUSES & _TERMINAL_STATUSES)


# ---- the basis argument (item 2) -----------------------------------------

def test_using_the_raw_peak_can_only_understate_drawdown():
    """The two sides of the ratio are on different bases by design, and
    that is safe by inequality rather than by luck.

    Every raw sample is <= its settled-equivalent value (they differ by the
    non-negative stake open at that moment), so peak_raw <= peak_settled,
    and therefore the drawdown computed against the raw peak is <= the one
    a fully settled-basis comparison would give. The raw peak can only ever
    DELAY a trip, never manufacture one.
    """
    current = 1.855402854
    for open_at_peak in (0.0, 0.05, 0.10, 0.25):
        peak_raw = INCIDENT_PEAK
        peak_settled = peak_raw + open_at_peak
        assert _dd(peak_raw, current) <= _dd(peak_settled, current) + 1e-12


def test_the_correction_is_bounded_by_the_stake_in_flight():
    """It cannot silently absorb a large real drawdown: the adjustment is
    exactly the open stake, so at a 0.05 cap it is worth ~2.6pp of drawdown
    on a ~1.9 bankroll, not an open-ended allowance."""
    peak, wallet = INCIDENT_PEAK, INCIDENT_WALLET
    shift = _dd(peak, wallet) - _dd(peak, wallet + 0.05)
    assert shift == pytest.approx(0.05 / peak, abs=1e-9)
    assert shift < 0.03


# ---- end-to-end through the real pipeline --------------------------------

def _pipeline_with(tmp_path, *, wallet, peak, open_stake):
    """Real MomentumOnlyPipeline whose tracker reports `wallet`/`peak` and
    whose provider reports `open_stake`."""
    from unittest.mock import MagicMock

    from pancakebot.config import load_app_config
    from pancakebot.strategy.momentum_gate import (
        MomentumGateConfig, MomentumGateResult,
    )
    from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline

    strategy = load_app_config("config.toml").strategy
    tracker = MagicMock()
    tracker.current_bankroll.return_value = wallet
    tracker.peak_bankroll.return_value = peak
    tracker.is_paused.return_value = False
    tracker.cooldown_remaining.return_value = 0
    gate = MagicMock()
    gate.evaluate.return_value = MomentumGateResult(
        signal="Bull", tier="multi_tf", skip_reason=None,
        signal_strength=0.0003, eth_signal=None, eth_signal_strength=0.0,
        sol_signal=None, sol_signal_strength=0.0,
    )
    cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT", kline_cutoff_seconds=2,
        mtf_lookbacks=strategy.gate.mtf_lookbacks,
        mtf_min_return_threshold=strategy.gate.mtf_min_return_threshold,
    )
    pipe = MomentumOnlyPipeline(
        config=cfg, strategy_config=strategy, gate=gate,
        kline_cutoff_seconds=2, pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001, treasury_fee_fraction=0.03,
        bankroll_tracker=tracker,
        open_stake_provider=lambda: open_stake,
    )
    return pipe, tracker


def _decide(pipe):
    from pancakebot.types import Round
    return pipe.decide_open_round(
        round_t=Round(epoch=509852, start_at=1787596880 - 300,
                      lock_at=1787596880, lock_price=None, close_price=None,
                      position=None, failed=False, bets=()),
        pool_bull_bnb=2.72 * 0.4, pool_bear_bnb=2.72 * 0.6,
    )


def test_pipeline_does_not_fire_the_breaker_on_the_incident(tmp_path):
    """Drives the REAL breaker with the real incident numbers."""
    pipe, tracker = _pipeline_with(
        tmp_path, wallet=INCIDENT_WALLET, peak=INCIDENT_PEAK,
        open_stake=INCIDENT_STAKE)
    d = _decide(pipe)
    assert d.skip_reason != "risk_drawdown_breaker_fired", d.skip_reason
    tracker.set_paused.assert_not_called()


def test_pipeline_still_fires_when_the_loss_is_realised(tmp_path):
    """Same numbers, nothing in flight -> the breaker must still fire, and
    must still suspend."""
    pipe, tracker = _pipeline_with(
        tmp_path, wallet=INCIDENT_WALLET, peak=INCIDENT_PEAK, open_stake=0.0)
    d = _decide(pipe)
    assert d.skip_reason == "risk_drawdown_breaker_fired"
    assert d.skip_context["drawdown_pct"] == pytest.approx(15.2836, abs=1e-3)
    tracker.set_paused.assert_called_once()


def test_the_liquidity_gates_still_read_the_raw_balance(tmp_path):
    """Check 2 (min bankroll) and the sizing clamp are about LIQUIDITY:
    funds locked in an open position cannot be staked again, so they must
    NOT be added back. A wallet under the minimum must still refuse to bet
    even when an open stake would lift the settled-equivalent above it."""
    from pancakebot.config import load_app_config
    floor = load_app_config("config.toml").strategy.risk.min_bankroll_bnb_to_bet
    pipe, _ = _pipeline_with(
        tmp_path, wallet=floor - 0.01, peak=floor * 2, open_stake=1.0)
    d = _decide(pipe)
    assert d.skip_reason == "risk_bankroll_below_min", d.skip_reason


def test_the_suspension_baseline_is_seeded_on_the_settled_basis(tmp_path):
    """The shadow ledger seeds from the same value the breaker compared
    (`shadow.start(bankroll=...)`). If that seed were the raw balance, the
    whole suspension would be measured against a stake-depressed baseline
    -- which is what happened on 2026-08-24, where suspension_bankroll was
    recorded as 1.805403 instead of 1.855403.

    Uses a genuinely tripping case WITH a position open, so the raw and
    settled values differ and the assertion can discriminate.
    """
    wallet, open_stake = 1.70, 0.05
    settled = wallet + open_stake
    assert _dd(INCIDENT_PEAK, settled) >= THRESHOLD, "must actually trip"

    pipe, tracker = _pipeline_with(
        tmp_path, wallet=wallet, peak=INCIDENT_PEAK, open_stake=open_stake)
    d = _decide(pipe)
    assert d.skip_reason == "risk_drawdown_breaker_fired"
    tracker.set_paused.assert_called_once()

    shadow = pipe._shadow
    if shadow is None or not getattr(shadow, "active", False):
        pytest.skip("shadow ledger disabled in this config")
    assert shadow.suspension_bankroll == pytest.approx(settled), (
        "suspension baseline must be the settled-equivalent value, not the "
        "stake-depressed wallet balance")
    assert shadow.suspension_bankroll != pytest.approx(wallet)


# ---- Check 4: worst-case exposure admission gate -------------------------
#
# TWO DECISIONS, TWO VALUATIONS:
#   suspend?      -> open positions valued at COST (settled-equivalent)
#   open a bet?   -> open positions valued at ZERO (worst case = raw wallet)
# The worst case IS the raw wallet balance, because a losing stake never
# comes back -- the same number the breaker was wrongly tripping on, wired
# to the question it actually answers.

def test_the_incident_declines_a_new_bet_without_suspending(tmp_path):
    """THE incident, exercising both halves at once. At 18:41:20 the raw
    wallet gave 15.28% >= 15%, so a new position was not survivable; the
    settled-equivalent gave 12.94%, so the account was not in trouble.
    Correct behaviour: DECLINE the round, do NOT suspend."""
    pipe, tracker = _pipeline_with(
        tmp_path, wallet=INCIDENT_WALLET, peak=INCIDENT_PEAK,
        open_stake=INCIDENT_STAKE)
    d = _decide(pipe)
    assert d.action == "SKIP"
    assert d.skip_reason == "risk_worst_case_exposure", d.skip_reason
    assert d.skip_context["worst_case_pct"] == pytest.approx(15.2836, abs=1e-3)
    assert d.skip_context["open_stake_bnb"] == pytest.approx(0.05)
    tracker.set_paused.assert_not_called()      # declining != suspending


def test_the_gate_is_a_no_op_when_nothing_is_in_flight(tmp_path):
    """With no open position, worst case == settled-equivalent, so any
    round that passes the breaker passes this by construction. It costs
    nothing in normal conditions."""
    pipe, tracker = _pipeline_with(
        tmp_path, wallet=2.00, peak=INCIDENT_PEAK, open_stake=0.0)
    d = _decide(pipe)
    assert d.skip_reason != "risk_worst_case_exposure"
    tracker.set_paused.assert_not_called()


def test_a_survivable_pending_loss_still_permits_consecutive_bets(tmp_path):
    """The rule is 'bet consecutive rounds IF the breaker would not fire
    should the pending bet lose'. Well clear of the bar -> bet allowed
    even with a position open."""
    pipe, tracker = _pipeline_with(
        tmp_path, wallet=2.00, peak=INCIDENT_PEAK, open_stake=0.05)
    d = _decide(pipe)
    assert d.skip_reason != "risk_worst_case_exposure"
    tracker.set_paused.assert_not_called()


def test_the_gate_generalises_to_several_open_positions(tmp_path):
    """No special casing for N: `current` already excludes every open
    stake, so 'assume they all lose' is just 'use current'. Two stakes
    open, each individually survivable, together not."""
    peak = 2.0
    wallet = peak * 0.86                      # 14% -- under the bar alone
    pipe, _ = _pipeline_with(tmp_path, wallet=wallet, peak=peak,
                             open_stake=0.0)
    assert _decide(pipe).skip_reason != "risk_worst_case_exposure"
    # same wallet, but 0.10 is in flight: worst case is unchanged at 14%,
    # so it still admits -- the gate keys on the WALLET, not the stake count
    pipe2, _ = _pipeline_with(tmp_path, wallet=wallet, peak=peak,
                              open_stake=0.10)
    assert _decide(pipe2).skip_reason != "risk_worst_case_exposure"
    # push the wallet under the bar and it declines regardless of N
    pipe3, _ = _pipeline_with(tmp_path, wallet=peak * 0.84, peak=peak,
                              open_stake=0.10)
    assert _decide(pipe3).skip_reason == "risk_worst_case_exposure"


def test_declining_never_fires_the_cooldown(tmp_path):
    """Conflating 'decline this round' with 'suspend for 24h' would let a
    single in-flight bet trigger a stand-down."""
    pipe, tracker = _pipeline_with(
        tmp_path, wallet=INCIDENT_WALLET, peak=INCIDENT_PEAK, open_stake=0.05)
    _decide(pipe)
    tracker.set_paused.assert_not_called()


def test_the_breaker_still_wins_when_the_account_is_genuinely_down(tmp_path):
    """Check 3 runs first: if even the settled-equivalent value is under
    the bar, this is a suspension, not a declined round."""
    pipe, tracker = _pipeline_with(
        tmp_path, wallet=1.70, peak=INCIDENT_PEAK, open_stake=0.05)
    d = _decide(pipe)
    assert d.skip_reason == "risk_drawdown_breaker_fired"
    tracker.set_paused.assert_called_once()


def test_the_lockout_releases_when_the_position_reaches_a_terminal_status(
        tmp_path):
    """DEGENERATE CASE. A stuck position must not produce a silent,
    permanent no-bet state: blocked from betting, never suspended, no
    alert. Terminal statuses drop open_stake to 0, after which worst case
    == settled-equivalent and the state resolves one way or the other --
    the position won (wallet recovered, betting resumes) or it lost (the
    breaker fires on the next round). It cannot hang."""
    from pancakebot.runtime.bet_ledger import _TERMINAL_STATUSES

    stakes = {"open": 0.05}
    pipe, tracker = _pipeline_with(
        tmp_path, wallet=INCIDENT_WALLET, peak=INCIDENT_PEAK, open_stake=0.05)
    assert _decide(pipe).skip_reason == "risk_worst_case_exposure"

    # every terminal status clears the stake from open_stake_bnb ...
    for terminal in sorted(_TERMINAL_STATUSES):
        path = _ledger(tmp_path, [
            {"epoch": 1, "status": "SUBMITTED", "amount_bnb": 0.05},
            {"epoch": 1, "status": terminal},
        ])
        assert open_stake_bnb(path) == 0.0, terminal

    # ... and with nothing open the round is no longer merely declined:
    # the same wallet is now a genuine 15.28% drawdown, so it SUSPENDS.
    pipe2, tracker2 = _pipeline_with(
        tmp_path, wallet=INCIDENT_WALLET, peak=INCIDENT_PEAK, open_stake=0.0)
    d2 = _decide(pipe2)
    assert d2.skip_reason == "risk_drawdown_breaker_fired"
    tracker2.set_paused.assert_called_once()


def test_a_recovering_position_releases_the_gate(tmp_path):
    """The other resolution path: the pending bet wins, the wallet rises,
    and betting resumes with no operator action."""
    pipe, _ = _pipeline_with(
        tmp_path, wallet=INCIDENT_WALLET + 0.0444, peak=INCIDENT_PEAK,
        open_stake=0.0)
    d = _decide(pipe)
    assert d.skip_reason not in (
        "risk_worst_case_exposure", "risk_drawdown_breaker_fired")
