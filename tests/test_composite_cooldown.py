"""Composite cooldown state machine (2026-07-09) — shadow ledger, extend-
while-bleeding, monitor override, and peak reseed.

Covers:
  - ShadowLedger settle math (win/loss vs realized pools, gas included),
    release_ok decision table (small-n / bleeding / below-recovery /
    recovered), and JSON persistence round-trip.
  - BankrollTracker.reset_peak_baseline (in-memory collapse + the
    persisted 'peak_reseed' load barrier that survives restarts).
  - Pipeline integration: breaker fire -> shadow evaluation during the
    suspension -> extension at expiry while bleeding -> release at expiry
    on recovery (peak reseeded so the breaker does NOT instantly re-fire)
    -> monitor override flag releases immediately.
  - Legacy flag-off behavior: extend_while_bleeding=False reproduces the
    fixed-length cooldown exactly (no shadow evals, plain expiry).
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.bankroll_tracker import (  # noqa: E402
    InMemoryBankrollTracker,
    PersistedBankrollTracker,
)
from pancakebot.constants import (  # noqa: E402
    BNB_WEI,
    MAX_GAS_COST_BET_BNB,
    MAX_GAS_COST_CLAIM_BNB,
)
from pancakebot.strategy.momentum_gate import (  # noqa: E402
    MomentumGateConfig,
    MomentumGateResult,
)
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402
from pancakebot.strategy.shadow_ledger import ShadowLedger  # noqa: E402
from pancakebot.types import Bet, Round  # noqa: E402

from tests.test_pipeline_skip_reason_propagation import (  # noqa: E402
    _make_strategy_config,
)

_ROUND_SECONDS = 300
_FEE = 0.03


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _closed_round(
    epoch: int, *, winner: str, bull_bnb: float = 5.0, bear_bnb: float = 5.0,
    start_at: int = 1_790_000_000,
) -> Round:
    bets = (
        Bet(wallet_address="0xaa", amount_wei=int(bull_bnb * BNB_WEI),
            position="Bull", created_at=start_at + 10),
        Bet(wallet_address="0xbb", amount_wei=int(bear_bnb * BNB_WEI),
            position="Bear", created_at=start_at + 10),
    )
    return Round(
        epoch=epoch, start_at=start_at, lock_at=start_at + _ROUND_SECONDS,
        lock_price=600.0, close_price=601.0 if winner == "Bull" else 599.0,
        position=winner, failed=False, bets=bets,
    )


def _open_round(epoch: int, start_at: int) -> Round:
    return Round(
        epoch=epoch, start_at=start_at, lock_at=start_at + _ROUND_SECONDS,
        lock_price=None, close_price=None, position=None, failed=False,
        bets=(),
    )


def _firing_gate_result() -> MomentumGateResult:
    return MomentumGateResult(
        signal="Bull", tier=None, skip_reason=None, signal_strength=0.001,
        eth_signal=None, sol_signal=None,
    )


def _silent_gate_result() -> MomentumGateResult:
    return MomentumGateResult(signal=None, tier=None, skip_reason="gate_no_signal")


def _strategy(risk_overrides: dict | None = None):
    sc = _make_strategy_config()
    if risk_overrides:
        sc = dataclasses.replace(sc, risk=dataclasses.replace(sc.risk, **risk_overrides))
    return sc


def _pipeline(gate, strategy, tracker) -> MomentumOnlyPipeline:
    cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT", kline_cutoff_seconds=2,
        mtf_lookbacks=(3, 7, 15), mtf_min_return_threshold=0.0001,
        max_consecutive_kline_fetch_failures=5,
    )
    return MomentumOnlyPipeline(
        config=cfg, strategy_config=strategy, gate=gate,
        kline_cutoff_seconds=2, pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001, treasury_fee_fraction=_FEE,
        bankroll_tracker=tracker,
    )


# ---------------------------------------------------------------------------
# ShadowLedger units
# ---------------------------------------------------------------------------

def test_shadow_settle_win_and_loss_math():
    sl = ShadowLedger(path=None)
    sl.start(bankroll=2.0, start_at=1_790_000_000)

    # WIN: 0.1 Bull into 5/5 pools. Impact-aware settlement:
    # total_after = 10.1, bull_after = 5.1
    # credit = 0.1 * (10.1 * 0.97 / 5.1) - claim_gas
    sl.record_fire(epoch=100, side="Bull", size_bnb=0.1)
    r_win = _closed_round(100, winner="Bull")
    pnl_win = sl.settle_round(
        round_t=r_win, treasury_fee_fraction=_FEE,
        bet_gas_bnb=MAX_GAS_COST_BET_BNB,
    )
    expected_credit = 0.1 * (10.1 * (1 - _FEE) / 5.1) - MAX_GAS_COST_CLAIM_BNB
    expected_win = expected_credit - 0.1 - MAX_GAS_COST_BET_BNB
    assert pnl_win == pytest.approx(expected_win, abs=1e-12)
    assert sl.n_settled == 1 and sl.n_wins == 1

    # LOSS: full stake + bet gas gone.
    sl.record_fire(epoch=101, side="Bear", size_bnb=0.05)
    pnl_loss = sl.settle_round(
        round_t=_closed_round(101, winner="Bull"),
        treasury_fee_fraction=_FEE, bet_gas_bnb=MAX_GAS_COST_BET_BNB,
    )
    assert pnl_loss == pytest.approx(-0.05 - MAX_GAS_COST_BET_BNB, abs=1e-12)
    assert sl.n_settled == 2 and sl.n_wins == 1
    assert sl.cum_pnl == pytest.approx(pnl_win + pnl_loss, abs=1e-12)
    assert sl.hypo_bankroll() == pytest.approx(2.0 + sl.cum_pnl, abs=1e-12)

    # Settling a round with no pending shadow bet is a no-op.
    assert sl.settle_round(
        round_t=_closed_round(999, winner="Bull"),
        treasury_fee_fraction=_FEE, bet_gas_bnb=MAX_GAS_COST_BET_BNB,
    ) is None


def test_shadow_settle_ignores_engine_stub_rounds():
    """The live engine passes epoch-tracking STUBS (position=None, bets=())
    to settle_closed_rounds. A stub must neither crash nor consume the
    pending bet — regression for the 2026-07-09 live crash (5 restarts,
    InvariantError settle_round_not_closed)."""
    sl = ShadowLedger(path=None)
    sl.start(bankroll=2.0, start_at=1_790_000_000)
    sl.record_fire(epoch=600, side="Bear", size_bnb=0.09)

    stub = Round(epoch=600, start_at=0, lock_at=None, lock_price=None,
                 close_price=None, position=None, failed=False, bets=())
    assert sl.settle_round(
        round_t=stub, treasury_fee_fraction=_FEE,
        bet_gas_bnb=MAX_GAS_COST_BET_BNB,
    ) is None
    assert 600 in sl.pending, "stub must not consume the pending bet"

    # The real closed round settles it normally afterwards.
    pnl = sl.settle_round(
        round_t=_closed_round(600, winner="Bear"),
        treasury_fee_fraction=_FEE, bet_gas_bnb=MAX_GAS_COST_BET_BNB,
    )
    assert pnl is not None and pnl > 0
    assert sl.n_settled == 1


def test_shadow_release_decision_table():
    t0 = 1_790_000_000
    sl = ShadowLedger(path=None)
    sl.start(bankroll=2.0, start_at=t0)

    # small-n: fewer settled fires than the floor -> extend.
    ok, why = sl.release_ok(min_fires=3, recovery_frac=0.85, as_of_start_at=t0)
    assert not ok and why.startswith("insufficient_fires")

    # 3 settled fires, net bleeding -> extend.
    for i, winner in enumerate(("Bear", "Bear", "Bull")):
        sl.record_fire(epoch=200 + i, side="Bull", size_bnb=0.1)
        sl.settle_round(
            round_t=_closed_round(200 + i, winner=winner,
                                  start_at=t0 + i * _ROUND_SECONDS),
            treasury_fee_fraction=_FEE, bet_gas_bnb=MAX_GAS_COST_BET_BNB,
        )
    assert sl.cum_pnl < 0
    ok, why = sl.release_ok(min_fires=3, recovery_frac=0.85, as_of_start_at=t0)
    assert not ok and why.startswith("bleeding")

    # Recovered: wins push cum_pnl positive AND above the recovery line.
    for i, ep in enumerate((300, 301, 302, 303)):
        sl.record_fire(epoch=ep, side="Bull", size_bnb=0.1)
        sl.settle_round(
            round_t=_closed_round(ep, winner="Bull",
                                  start_at=t0 + (10 + i) * _ROUND_SECONDS),
            treasury_fee_fraction=_FEE, bet_gas_bnb=MAX_GAS_COST_BET_BNB,
        )
    assert sl.cum_pnl > 0
    ok, why = sl.release_ok(
        min_fires=3, recovery_frac=0.85,
        as_of_start_at=t0 + 20 * _ROUND_SECONDS,
    )
    assert ok and why.startswith("recovered")


def test_shadow_below_recovery_peak_extends():
    """cum_pnl >= 0 alone is not enough: a big hypothetical peak followed by
    a fade back toward zero must NOT release (both conditions per spec)."""
    t0 = 1_790_000_000
    sl = ShadowLedger(path=None)
    sl.start(bankroll=1.0, start_at=t0)
    # Three big wins -> hypo peak well above start.
    for i, ep in enumerate((400, 401, 402)):
        sl.record_fire(epoch=ep, side="Bull", size_bnb=0.3)
        sl.settle_round(
            round_t=_closed_round(ep, winner="Bull",
                                  start_at=t0 + i * _ROUND_SECONDS),
            treasury_fee_fraction=_FEE, bet_gas_bnb=MAX_GAS_COST_BET_BNB,
        )
    peak = sl.hypo_peak(t0 + 10 * _ROUND_SECONDS)
    # Then losses that keep cum_pnl barely positive but drop hypo bankroll
    # to <= 85% of the hypothetical peak.
    i = 0
    while sl.cum_pnl > 0 and sl.hypo_bankroll() > peak * 0.85:
        ep = 410 + i
        sl.record_fire(epoch=ep, side="Bull", size_bnb=0.3)
        sl.settle_round(
            round_t=_closed_round(ep, winner="Bear",
                                  start_at=t0 + (5 + i) * _ROUND_SECONDS),
            treasury_fee_fraction=_FEE, bet_gas_bnb=MAX_GAS_COST_BET_BNB,
        )
        i += 1
        assert i < 10, "test setup runaway"
    if sl.cum_pnl >= 0:
        ok, why = sl.release_ok(
            min_fires=3, recovery_frac=0.85,
            as_of_start_at=t0 + 20 * _ROUND_SECONDS,
        )
        assert not ok and why.startswith("below_recovery")


def test_shadow_persistence_roundtrip(tmp_path):
    p = tmp_path / "shadow_state.json"
    sl = ShadowLedger(path=p)
    sl.start(bankroll=2.5, start_at=1_790_000_000)
    sl.record_fire(epoch=500, side="Bear", size_bnb=0.07)
    sl.settle_round(
        round_t=_closed_round(500, winner="Bear"),
        treasury_fee_fraction=_FEE, bet_gas_bnb=MAX_GAS_COST_BET_BNB,
    )
    sl.record_fire(epoch=501, side="Bull", size_bnb=0.08)  # stays pending
    sl.extend()

    reloaded = ShadowLedger(path=p)
    assert reloaded.active
    assert reloaded.suspension_bankroll == pytest.approx(2.5)
    assert reloaded.cum_pnl == pytest.approx(sl.cum_pnl)
    assert reloaded.n_settled == 1 and reloaded.n_wins == 1
    assert reloaded.extensions == 1
    assert reloaded.pending == {501: {"side": "Bull", "size_bnb": 0.08}}

    reloaded.clear()
    assert not ShadowLedger(path=p).active


# ---------------------------------------------------------------------------
# Shadow settlement feed (engine round_data -> settleable Round)
# ---------------------------------------------------------------------------

def _round_data(**kw):
    from pancakebot.chain.prediction_contract import RoundData
    base = dict(epoch=700, start_ts=1_790_000_000, lock_ts=1_790_000_300,
                close_ts=1_790_000_600, lock_price_usd=600.0,
                close_price_usd=601.0, bull_amount_wei=int(4.0 * BNB_WEI),
                bear_amount_wei=int(6.0 * BNB_WEI), oracle_called=True)
    base.update(kw)
    return RoundData(**base)


def test_build_settlement_round_winner_derivation():
    from pancakebot.runtime.engine import build_shadow_settlement_round
    now = 1_790_001_000
    assert build_shadow_settlement_round(
        _round_data(close_price_usd=601.0), now_ts=now, buffer_seconds=30,
    ).position == "Bull"
    assert build_shadow_settlement_round(
        _round_data(close_price_usd=599.0), now_ts=now, buffer_seconds=30,
    ).position == "Bear"
    house = build_shadow_settlement_round(
        _round_data(close_price_usd=600.0), now_ts=now, buffer_seconds=30,
    )
    assert house.position == "House" and house.failed is False


def test_build_settlement_round_not_final_and_cancelled():
    from pancakebot.runtime.engine import build_shadow_settlement_round
    # Not oracle-called, shortly after close -> not final yet (retry later).
    assert build_shadow_settlement_round(
        _round_data(oracle_called=False),
        now_ts=1_790_000_700, buffer_seconds=30,
    ) is None
    # Not oracle-called, WAY past close (>10x buffer) -> cancelled/refund.
    cancelled = build_shadow_settlement_round(
        _round_data(oracle_called=False),
        now_ts=1_790_000_600 + 301, buffer_seconds=30,
    )
    assert cancelled is not None and cancelled.failed is True
    assert cancelled.position is None


def test_shadow_settles_against_built_round_exact_math():
    """End-to-end: pending shadow bet settles against the engine-built Round
    with pools carried from on-chain amounts (impact-aware math)."""
    from pancakebot.runtime.engine import build_shadow_settlement_round
    sl = ShadowLedger(path=None)
    sl.start(bankroll=1.889, start_at=1_790_000_000)
    sl.record_fire(epoch=700, side="Bear", size_bnb=0.0945)

    built = build_shadow_settlement_round(
        _round_data(close_price_usd=599.0),   # Bear wins
        now_ts=1_790_001_000, buffer_seconds=30,
    )
    pnl = sl.settle_round(
        round_t=built, treasury_fee_fraction=_FEE,
        bet_gas_bnb=MAX_GAS_COST_BET_BNB,
    )
    # pools 4 bull / 6 bear + our 0.0945 bear;
    # credit = 0.0945 * (10.0945*0.97/6.0945) - claim_gas
    expected_credit = 0.0945 * (10.0945 * (1 - _FEE) / 6.0945) - MAX_GAS_COST_CLAIM_BNB
    assert pnl == pytest.approx(expected_credit - 0.0945 - MAX_GAS_COST_BET_BNB, abs=1e-9)
    assert sl.n_settled == 1 and sl.n_wins == 1

    # Cancelled round -> refund semantics (stake back minus gas).
    sl.record_fire(epoch=701, side="Bull", size_bnb=0.05)
    cancelled = build_shadow_settlement_round(
        _round_data(epoch=701, oracle_called=False),
        now_ts=1_790_000_600 + 400, buffer_seconds=30,
    )
    pnl2 = sl.settle_round(
        round_t=cancelled, treasury_fee_fraction=_FEE,
        bet_gas_bnb=MAX_GAS_COST_BET_BNB,
    )
    assert pnl2 == pytest.approx(
        (0.05 - MAX_GAS_COST_CLAIM_BNB) - 0.05 - MAX_GAS_COST_BET_BNB, abs=1e-9)


def test_build_settlement_round_one_sided_pool_no_zero_bets():
    """Review BLOCKER (2026-07-09): a zero on-chain side must not produce a
    zero-amount synthetic Bet (settlement raises bet_amount_wei_nonpositive
    on those). One-sided rounds settle via the impact-aware math."""
    from pancakebot.runtime.engine import build_shadow_settlement_round
    built = build_shadow_settlement_round(
        _round_data(bear_amount_wei=0, close_price_usd=599.0),  # Bear wins
        now_ts=1_790_001_000, buffer_seconds=30,
    )
    assert len(built.bets) == 1 and built.bets[0].position == "Bull"

    # Shadow bet on the EMPTY winning side: we'd have been the entire
    # winning pool -> payout = (bull + our stake) * (1-fee) / our stake.
    sl = ShadowLedger(path=None)
    sl.start(bankroll=2.0, start_at=1_790_000_000)
    sl.record_fire(epoch=700, side="Bear", size_bnb=0.0945)
    pnl = sl.settle_round(
        round_t=built, treasury_fee_fraction=_FEE,
        bet_gas_bnb=MAX_GAS_COST_BET_BNB,
    )
    expected_credit = 0.0945 * ((4.0 + 0.0945) * (1 - _FEE) / 0.0945) - MAX_GAS_COST_CLAIM_BNB
    assert pnl == pytest.approx(
        expected_credit - 0.0945 - MAX_GAS_COST_BET_BNB, abs=1e-9)
    # House win with a zero side settles as a plain loss (no crash).
    sl.record_fire(epoch=701, side="Bull", size_bnb=0.05)
    house = build_shadow_settlement_round(
        _round_data(epoch=701, bull_amount_wei=0, close_price_usd=600.0),
        now_ts=1_790_001_000, buffer_seconds=30,
    )
    pnl2 = sl.settle_round(
        round_t=house, treasury_fee_fraction=_FEE,
        bet_gas_bnb=MAX_GAS_COST_BET_BNB,
    )
    assert pnl2 == pytest.approx(-0.05 - MAX_GAS_COST_BET_BNB, abs=1e-12)


def test_settle_round_atomic_on_settle_failure():
    """Review MAJOR: a raise inside the settle math must leave the pending
    bet intact in memory AND on disk (settle-then-pop ordering)."""
    from pancakebot.util import InvariantError as IE
    sl = ShadowLedger(path=None)
    sl.start(bankroll=2.0, start_at=1_790_000_000)
    sl.record_fire(epoch=750, side="Bear", size_bnb=0.05)
    # A Round that passes the settleability guard but fails settle math
    # (position set, but a zero-amount bet in the pool list).
    bad = Round(
        epoch=750, start_at=1_790_000_000, lock_at=1_790_000_300,
        lock_price=600.0, close_price=599.0, position="Bear", failed=False,
        bets=(Bet(wallet_address="0x0", amount_wei=0, position="Bull",
                  created_at=1_790_000_010),),
    )
    with pytest.raises(IE):
        sl.settle_round(round_t=bad, treasury_fee_fraction=_FEE,
                        bet_gas_bnb=MAX_GAS_COST_BET_BNB)
    assert 750 in sl.pending, "failed settle must not consume the bet"
    assert sl.n_settled == 0 and sl.cum_pnl == 0.0


class _FakeContract:
    """round_data stand-in for the engine wiring tests."""

    def __init__(self, data: dict, fail_epochs: set | None = None):
        self.data = data
        self.fail_epochs = fail_epochs or set()
        self.calls: list[int] = []

    def round_data(self, epoch: int):
        self.calls.append(int(epoch))
        if epoch in self.fail_epochs:
            from pancakebot.util import TransientRpcError
            raise TransientRpcError(f"fake rpc fail epoch={epoch}")
        return self.data[int(epoch)]


def test_engine_wiring_fetch_pending_shadow_rounds():
    """Review MAJOR: drive the REAL engine wiring (filter, bound, RPC-fail
    continue) with a fake contract + a real pipeline holding pending bets."""
    from pancakebot.runtime.engine import fetch_pending_shadow_rounds
    gate = MagicMock()
    gate.evaluate.return_value = _firing_gate_result()
    sc = _strategy({"cooldown_rounds": 10, "extend_while_bleeding": True})
    tracker = InMemoryBankrollTracker(initial_bankroll=2.3, drawdown_peak_window_days=7)
    pipe = _pipeline(gate, sc, tracker)
    t0 = 1_790_000_000
    _drive_to_breaker(pipe, tracker, t0)
    # Three shadow fires -> three pending epochs.
    for i, ep in enumerate((900, 901, 902)):
        pipe.decide_open_round(
            round_t=_open_round(ep, t0 + (3 + i) * _ROUND_SECONDS),
            pool_bull_bnb=5.0, pool_bear_bnb=5.0,
        )
    assert pipe.pending_shadow_epochs == (900, 901, 902)

    contract = _FakeContract(
        data={ep: _round_data(epoch=ep, close_price_usd=599.0)
              for ep in (900, 901, 902)},
        fail_epochs={900},
    )
    # locked_epoch=902 -> only epochs <= 901 fetchable; 902 excluded.
    rounds = fetch_pending_shadow_rounds(
        contract=contract, pipeline=pipe, locked_epoch=902,
        now_ts=t0 + 10 * _ROUND_SECONDS, buffer_seconds=30,
    )
    assert contract.calls == [900, 901], "filter + [:2] bound + order"
    assert [r.epoch for r in rounds] == [901], "RPC-fail epoch skipped"

    # Settle through the pipeline: 901 drains; 900 (failed fetch) and 902
    # (not yet closed) stay pending.
    stub = Round(epoch=901, start_at=0, lock_at=None, lock_price=None,
                 close_price=None, position=None, failed=False, bets=())
    pipe.settle_closed_rounds(rounds=[stub] + rounds)
    assert pipe.pending_shadow_epochs == (900, 902)


def test_pending_shadow_epochs_property():
    gate = MagicMock()
    gate.evaluate.return_value = _firing_gate_result()
    sc = _strategy({"cooldown_rounds": 5, "extend_while_bleeding": True})
    tracker = InMemoryBankrollTracker(initial_bankroll=2.3, drawdown_peak_window_days=7)
    pipe = _pipeline(gate, sc, tracker)
    assert pipe.pending_shadow_epochs == ()
    t0 = 1_790_000_000
    _drive_to_breaker(pipe, tracker, t0)
    pipe.decide_open_round(
        round_t=_open_round(801, t0 + 3 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert pipe.pending_shadow_epochs == (801,)
    # Settle via a real round -> pending drains.
    pipe.settle_closed_rounds(rounds=[
        _closed_round(801, winner="Bull", start_at=t0 + 3 * _ROUND_SECONDS)])
    assert pipe.pending_shadow_epochs == ()


# ---------------------------------------------------------------------------
# Tracker reseed units
# ---------------------------------------------------------------------------

def test_inmemory_reset_peak_baseline():
    t = InMemoryBankrollTracker(initial_bankroll=2.3, drawdown_peak_window_days=7)
    t0 = 1_790_000_000
    t.record_settlement(2.3, t0)
    t.record_settlement(1.9, t0 + 600)
    assert t.peak_bankroll(t0 + 600) == pytest.approx(2.3)
    t.reset_peak_baseline(t0 + 1200)
    assert t.current_bankroll() == pytest.approx(1.9)
    assert t.peak_bankroll(t0 + 1200) == pytest.approx(1.9)


def test_persisted_reseed_barrier_survives_restart(tmp_path):
    hist = tmp_path / "bankroll_history.jsonl"
    t = PersistedBankrollTracker(
        path=hist, initial_bankroll=2.3, drawdown_peak_window_days=7,
    )
    t0 = 1_790_000_000
    t.record_settlement(2.3, t0)
    t.record_settlement(1.9, t0 + 600)
    t.reset_peak_baseline(t0 + 1200)
    assert t.peak_bankroll(t0 + 1200) == pytest.approx(1.9)

    # Restart: the pre-reseed 2.3 entry is on disk but must NOT re-enter
    # the peak window (the 'peak_reseed' line is a load barrier).
    t2 = PersistedBankrollTracker(
        path=hist, initial_bankroll=2.3, drawdown_peak_window_days=7,
    )
    assert t2.current_bankroll() == pytest.approx(1.9)
    assert t2.peak_bankroll(t0 + 1300) == pytest.approx(1.9)

    # persist_dir surfaces the state directory for sibling files.
    assert t2.persist_dir == tmp_path


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def _drive_to_breaker(pipe, tracker, t0):
    """Record a drawdown > 15% then decide once: breaker must fire."""
    pipe.record_settlement(bankroll=2.3, start_at=t0)
    pipe.record_settlement(bankroll=1.9, start_at=t0 + _ROUND_SECONDS)
    d = pipe.decide_open_round(
        round_t=_open_round(1000, t0 + 2 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.skip_reason == "risk_drawdown_breaker_fired"
    assert tracker.is_paused(t0)
    return d


def test_full_cycle_fire_shadow_extend_release():
    """Breaker fire -> shadow fires recorded while paused -> bleeding at
    expiry extends -> recovery at a later expiry releases + reseeds peak
    so the bot can BET the same round (no instant re-fire)."""
    gate = MagicMock()
    gate.evaluate.return_value = _firing_gate_result()
    sc = _strategy({"cooldown_rounds": 3, "extend_while_bleeding": True,
                    "shadow_min_fires_to_release": 2})
    tracker = InMemoryBankrollTracker(initial_bankroll=2.3, drawdown_peak_window_days=7)
    pipe = _pipeline(gate, sc, tracker)
    t0 = 1_790_000_000
    _drive_to_breaker(pipe, tracker, t0)

    # Paused rounds 1..2: gate fires -> shadow records (no real bet).
    for i, ep in enumerate((1001, 1002)):
        d = pipe.decide_open_round(
            round_t=_open_round(ep, t0 + (3 + i) * _ROUND_SECONDS),
            pool_bull_bnb=5.0, pool_bear_bnb=5.0,
        )
        assert d.skip_reason == "risk_cooldown_active"
        assert d.skip_context.get("shadow_pending") == i + 1
    # Settle both as LOSSES (bleeding).
    pipe.settle_closed_rounds(rounds=[
        _closed_round(1001, winner="Bear", start_at=t0 + 3 * _ROUND_SECONDS),
        _closed_round(1002, winner="Bear", start_at=t0 + 4 * _ROUND_SECONDS),
    ])

    # Expiry round (tick 3 of 3): bleeding -> EXTENDED.
    d = pipe.decide_open_round(
        round_t=_open_round(1003, t0 + 5 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.skip_reason == "risk_cooldown_active"
    assert d.skip_context.get("extended") is True
    assert "bleeding" in d.skip_context.get("extend_reason", "")
    assert tracker.cooldown_remaining() == 3

    # Two shadow WINS during the extension -> recovery.
    for i, ep in enumerate((1004, 1005)):
        d = pipe.decide_open_round(
            round_t=_open_round(ep, t0 + (6 + i) * _ROUND_SECONDS),
            pool_bull_bnb=5.0, pool_bear_bnb=5.0,
        )
        assert d.skip_reason == "risk_cooldown_active"
    pipe.settle_closed_rounds(rounds=[
        _closed_round(1004, winner="Bull", start_at=t0 + 6 * _ROUND_SECONDS,
                      bull_bnb=2.0, bear_bnb=8.0),
        _closed_round(1005, winner="Bull", start_at=t0 + 7 * _ROUND_SECONDS,
                      bull_bnb=2.0, bear_bnb=8.0),
    ])

    # Expiry round again: recovered -> RELEASED, and the SAME round can BET
    # (peak reseeded, so the 17% real drawdown does not re-fire the breaker).
    d = pipe.decide_open_round(
        round_t=_open_round(1006, t0 + 8 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.action == "BET", f"expected BET after release, got {d.skip_reason}"
    assert not tracker.is_paused(t0 + 8 * _ROUND_SECONDS)
    assert tracker.peak_bankroll(t0 + 8 * _ROUND_SECONDS) == pytest.approx(1.9)


def test_monitor_override_releases_immediately(tmp_path):
    """A fresh override flag releases the suspension mid-cooldown and is
    consumed; the same round proceeds to a normal decision."""
    gate = MagicMock()
    gate.evaluate.return_value = _firing_gate_result()
    sc = _strategy({"cooldown_rounds": 288, "extend_while_bleeding": True,
                    "monitor_override_enabled": True})
    hist = tmp_path / "bankroll_history.jsonl"
    tracker = PersistedBankrollTracker(
        path=hist, initial_bankroll=2.3, drawdown_peak_window_days=7,
    )
    pipe = _pipeline(gate, sc, tracker)
    t0 = 1_790_000_000
    _drive_to_breaker(pipe, tracker, t0)

    flag = tmp_path / "cooldown_override.json"
    flag.write_text(json.dumps({"ts": time.time(), "week": "test"}),
                    encoding="utf-8")
    d = pipe.decide_open_round(
        round_t=_open_round(2000, t0 + 3 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.action == "BET"
    assert not flag.exists(), "override flag must be consumed"
    assert not tracker.is_paused(t0 + 3 * _ROUND_SECONDS)
    # shadow_state.json reflects the cleared ledger.
    assert not ShadowLedger(path=tmp_path / "shadow_state.json").active


def test_stale_override_flag_is_discarded(tmp_path):
    gate = MagicMock()
    gate.evaluate.return_value = _silent_gate_result()
    sc = _strategy({"cooldown_rounds": 288, "extend_while_bleeding": True,
                    "monitor_override_enabled": True})
    tracker = PersistedBankrollTracker(
        path=tmp_path / "bankroll_history.jsonl",
        initial_bankroll=2.3, drawdown_peak_window_days=7,
    )
    pipe = _pipeline(gate, sc, tracker)
    t0 = 1_790_000_000
    _drive_to_breaker(pipe, tracker, t0)

    flag = tmp_path / "cooldown_override.json"
    flag.write_text(json.dumps({"ts": time.time() - 30 * 86400}), encoding="utf-8")
    d = pipe.decide_open_round(
        round_t=_open_round(2100, t0 + 3 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.skip_reason == "risk_cooldown_active"
    assert not flag.exists(), "stale flag still consumed (discarded)"
    assert tracker.is_paused(t0 + 3 * _ROUND_SECONDS)


def test_legacy_flag_off_reproduces_fixed_cooldown():
    """extend_while_bleeding=False: no shadow activity, no gate calls while
    paused, plain expiry -> (real bankroll unchanged) breaker re-fires."""
    gate = MagicMock()
    gate.evaluate.return_value = _firing_gate_result()
    sc = _strategy({"cooldown_rounds": 2, "extend_while_bleeding": False})
    tracker = InMemoryBankrollTracker(initial_bankroll=2.3, drawdown_peak_window_days=7)
    pipe = _pipeline(gate, sc, tracker)
    t0 = 1_790_000_000
    _drive_to_breaker(pipe, tracker, t0)
    gate.evaluate.reset_mock()

    # 2 paused rounds wind the counter down; the gate must NOT be evaluated.
    for i, ep in enumerate((3001, 3002)):
        d = pipe.decide_open_round(
            round_t=_open_round(ep, t0 + (3 + i) * _ROUND_SECONDS),
            pool_bull_bnb=5.0, pool_bear_bnb=5.0,
        )
        assert d.skip_reason == "risk_cooldown_active"
        assert "extended" not in (d.skip_context or {})
    gate.evaluate.assert_not_called()

    # Next round: unpaused, still 17% below peak -> breaker re-fires
    # (the legacy re-arm cycle).
    d = pipe.decide_open_round(
        round_t=_open_round(3003, t0 + 5 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.skip_reason == "risk_drawdown_breaker_fired"


def test_inherited_suspension_expires_under_legacy_rules():
    """A suspension entered BEFORE the shadow machinery existed (pause state
    persisted, shadow never started) must expire legacy-style — NOT extend
    on an empty ledger — and the next round's breaker re-fire starts the
    shadow (the 2026-07-09 deploy-transition path)."""
    gate = MagicMock()
    gate.evaluate.return_value = _silent_gate_result()
    sc = _strategy({"cooldown_rounds": 2, "extend_while_bleeding": True})
    tracker = InMemoryBankrollTracker(initial_bankroll=2.3, drawdown_peak_window_days=7)
    pipe = _pipeline(gate, sc, tracker)
    t0 = 1_790_000_000
    pipe.record_settlement(bankroll=2.3, start_at=t0)
    pipe.record_settlement(bankroll=1.9, start_at=t0 + _ROUND_SECONDS)
    # Old-code state: paused WITHOUT shadow.start().
    tracker.set_paused(2, t0 + _ROUND_SECONDS)
    assert pipe._shadow is not None and not pipe._shadow.active

    d = pipe.decide_open_round(
        round_t=_open_round(5001, t0 + 2 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.skip_reason == "risk_cooldown_active"
    # Expiry round: inactive shadow -> legacy plain skip, NO extension.
    d = pipe.decide_open_round(
        round_t=_open_round(5002, t0 + 3 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.skip_reason == "risk_cooldown_active"
    assert "extended" not in (d.skip_context or {})
    # Next round: breaker re-fires under NEW rules -> shadow starts.
    d = pipe.decide_open_round(
        round_t=_open_round(5003, t0 + 4 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.skip_reason == "risk_drawdown_breaker_fired"
    assert pipe._shadow.active


def test_shadow_small_n_extends_at_expiry():
    """Silent gate during the whole cooldown -> insufficient fires -> extend."""
    gate = MagicMock()
    gate.evaluate.return_value = _silent_gate_result()
    sc = _strategy({"cooldown_rounds": 2, "extend_while_bleeding": True,
                    "shadow_min_fires_to_release": 3})
    tracker = InMemoryBankrollTracker(initial_bankroll=2.3, drawdown_peak_window_days=7)
    pipe = _pipeline(gate, sc, tracker)
    t0 = 1_790_000_000
    _drive_to_breaker(pipe, tracker, t0)

    d = pipe.decide_open_round(
        round_t=_open_round(4001, t0 + 3 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.skip_reason == "risk_cooldown_active"
    d = pipe.decide_open_round(
        round_t=_open_round(4002, t0 + 4 * _ROUND_SECONDS),
        pool_bull_bnb=5.0, pool_bear_bnb=5.0,
    )
    assert d.skip_context.get("extended") is True
    assert "insufficient_fires" in d.skip_context.get("extend_reason", "")
