"""Pool-filter admission threshold at 1.25 BNB.

`min_pool_bnb_at_cutoff` gates admission in `MomentumOnlyPipeline`:
`pool_total < threshold` -> SKIP `pool_below_minimum`. The comparison is
STRICT, so a pool of exactly the threshold is admitted -- that boundary is
pinned below because an accidental `<=` would silently drop a round class.

These drive the REAL pipeline with the REAL config.toml strategy config,
so they fail if the deployed value moves or the gate changes shape. They
deliberately do not re-implement the comparison.

Two things this file also guards:
  * `min_payout_multiple_at_cutoff` is a SEPARATE gate on a different
    quantity (payout multiple on our side, not pool size in BNB) that
    happens to share the value 1.5. It must not be dragged along.
  * The key must stay present in config.toml. The code default is 1.5,
    so dropping it reverts to the tighter filter -- the safe direction,
    but a silent behaviour change either way.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.config import load_app_config  # noqa: E402
from pancakebot.types import Round  # noqa: E402
from pancakebot.strategy.momentum_gate import (  # noqa: E402
    MomentumGateConfig,
    MomentumGateResult,
)
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402

THRESHOLD = 1.25
OLD_THRESHOLD = 1.5


def _deployed_strategy():
    return load_app_config("config.toml").strategy


def _pipeline():
    """Pipeline wired to the DEPLOYED strategy config and a stub gate that
    always returns a clean Bull signal, so the pool filter is the only
    thing that can reject."""
    strategy = _deployed_strategy()
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
    return MomentumOnlyPipeline(
        config=cfg, strategy_config=strategy, gate=gate,
        kline_cutoff_seconds=2, pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001, treasury_fee_fraction=0.03,
    )


def _round(epoch: int = 478456) -> Round:
    return Round(epoch=epoch, start_at=1777966856, lock_at=1777967156,
                 lock_price=None, close_price=None, position=None,
                 failed=False, bets=())


def _decide(pool_total: float):
    """Decide a round at the given TOTAL pool, split 40/60 so the separate
    payout gate (>= 1.5x on our side) is comfortably clear and cannot be
    confused with a pool-size rejection."""
    return _pipeline().decide_open_round(
        round_t=_round(),
        pool_bull_bnb=pool_total * 0.4,
        pool_bear_bnb=pool_total * 0.6,
    )


# ---- the deployed value --------------------------------------------------

def test_config_carries_the_new_pool_threshold():
    pf = _deployed_strategy().pool_filter
    assert pf.min_pool_bnb_at_cutoff == THRESHOLD


def test_the_payout_gate_is_a_separate_knob_and_did_not_move():
    """It shares the value 1.5 by coincidence and gates a different
    quantity. Changing the pool filter must not touch it."""
    pf = _deployed_strategy().pool_filter
    assert pf.min_payout_multiple_at_cutoff == 1.5
    assert pf.min_pool_bnb_at_cutoff != pf.min_payout_multiple_at_cutoff


def test_pool_threshold_key_is_present_in_config():
    """The code default is 1.5; a dropped key silently re-tightens the
    filter. Guard the literal as well as the loaded value."""
    import re
    body = Path("config.toml").read_text(encoding="utf-8")
    assert re.search(r"^min_pool_bnb_at_cutoff\s*=\s*1\.25\s*$", body, re.M)


# ---- admission behaviour, driven through the real pipeline ---------------

@pytest.mark.parametrize("pool", [0.5, 0.8, 1.0, 1.2, 1.2499])
def test_pools_below_the_threshold_are_still_rejected(pool):
    d = _decide(pool)
    assert d.action == "SKIP"
    assert d.skip_reason == "pool_below_minimum"
    assert d.skip_context["min_pool_bnb_at_cutoff"] == THRESHOLD


def test_a_pool_exactly_at_the_threshold_is_admitted():
    """The comparison is strict (`pool_total < threshold`). An accidental
    `<=` would drop this class silently, so pin it."""
    d = _decide(THRESHOLD)
    assert d.action == "BET", d.skip_reason
    assert d.bet_side == "Bull"


@pytest.mark.parametrize("pool", [1.25, 1.3, 1.4, 1.4999])
def test_the_newly_admitted_band_is_exactly_what_the_change_buys(pool):
    """These pools were rejected at 1.5 and are admitted at 1.25 -- the
    whole behavioural content of the change lives in this band."""
    assert pool < OLD_THRESHOLD
    assert pool >= THRESHOLD
    d = _decide(pool)
    assert d.action == "BET", d.skip_reason


@pytest.mark.parametrize("pool", [1.5, 2.0, 2.72, 5.0])
def test_pools_that_already_passed_still_pass_unchanged(pool):
    """The change is one-sided: it admits more, and rejects nothing that
    the old threshold admitted."""
    d = _decide(pool)
    assert d.action == "BET", d.skip_reason


def test_every_admitted_stake_respects_the_deployed_cap():
    """The looser filter must not interact with sizing: the marginal
    rounds are small pools, and they still clamp to the 0.05 cap or below,
    never above it."""
    cap = _deployed_strategy().risk.max_bet_bnb_btc_primary
    assert cap == 0.05
    for pool in (1.25, 1.4, 1.5, 2.72, 5.0):
        d = _decide(pool)
        assert d.action == "BET", d.skip_reason
        assert 0.0 < d.bet_size_bnb <= cap + 1e-12, (pool, d.bet_size_bnb)


def test_the_threshold_is_the_only_thing_separating_the_two_regimes():
    """A/B on the SAME round, signal and split, with only the threshold
    swapped: rejected under 1.5, admitted under the deployed 1.25. Proves
    the gate is what changed rather than the signal or the sizing path,
    and would fail if the two arms ever stopped disagreeing."""
    import dataclasses
    pool = 1.3

    strategy = _deployed_strategy()
    old_strategy = dataclasses.replace(
        strategy,
        pool_filter=dataclasses.replace(
            strategy.pool_filter, min_pool_bnb_at_cutoff=OLD_THRESHOLD),
    )
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
    old_pipe = MomentumOnlyPipeline(
        config=cfg, strategy_config=old_strategy, gate=gate,
        kline_cutoff_seconds=2, pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001, treasury_fee_fraction=0.03,
    )
    rejected = old_pipe.decide_open_round(
        round_t=_round(), pool_bull_bnb=pool * 0.4, pool_bear_bnb=pool * 0.6)

    assert rejected.action == "SKIP"
    assert rejected.skip_reason == "pool_below_minimum"
    assert _decide(pool).action == "BET"
