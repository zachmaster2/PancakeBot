"""Conformance: MomentumOnlyPipeline satisfies strategy.base.StrategyPipeline.

Doubles as the executable template for any second pipeline: construct it
the way runtime/dry.py and backtest/runner.py do, assert the Protocol holds,
and assert the two optional attributes the engine probes (see base.py's
docstring) exist under the expected names.

Run:
    python -m pytest tests/test_strategy_pipeline_protocol.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.config import load_strategy_config_from_dict  # noqa: E402
from pancakebot.strategy.base import StrategyPipeline  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402


def _build_pipeline() -> MomentumOnlyPipeline:
    strategy_cfg = load_strategy_config_from_dict({})
    gate_cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=2,
        mtf_lookbacks=strategy_cfg.gate.mtf_lookbacks,
        mtf_min_return_threshold=strategy_cfg.gate.mtf_min_return_threshold,
    )
    return MomentumOnlyPipeline(
        config=gate_cfg,
        strategy_config=strategy_cfg,
        gate=None,  # backtest-style construction
        kline_cutoff_seconds=2,
        pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001,
        treasury_fee_fraction=0.03,
    )


def test_momentum_pipeline_satisfies_protocol():
    pipeline = _build_pipeline()
    assert isinstance(pipeline, StrategyPipeline), (
        "MomentumOnlyPipeline drifted from strategy/base.py's Protocol — "
        "update both sides together"
    )


def test_engine_probe_attributes_exist():
    """The engine's optional-attribute probes (documented in base.py) rely
    on these exact private names; a pipeline renaming them silently loses
    OKX warmup + telemetry (and the audit fallback path reads
    _bankroll_tracker via getattr)."""
    pipeline = _build_pipeline()
    assert hasattr(pipeline, "_gate")
    assert hasattr(pipeline, "_bankroll_tracker")


def test_mismatched_gate_and_strategy_configs_fail_fast():
    """The config-duality guard: live computes from MomentumGateConfig,
    backtest from strategy_config.gate — construction must refuse
    divergent copies."""
    from pancakebot.util import InvariantError

    strategy_cfg = load_strategy_config_from_dict({})
    bad_gate_cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=2,
        mtf_lookbacks=(2, 6, 14),  # diverges from strategy_cfg.gate
        mtf_min_return_threshold=strategy_cfg.gate.mtf_min_return_threshold,
    )
    with pytest.raises(InvariantError, match="gate_config_strategy_config_mismatch"):
        MomentumOnlyPipeline(
            config=bad_gate_cfg,
            strategy_config=strategy_cfg,
            gate=None,
            kline_cutoff_seconds=2,
            pool_cutoff_seconds=6,
            min_bet_amount_bnb=0.001,
            treasury_fee_fraction=0.03,
        )
