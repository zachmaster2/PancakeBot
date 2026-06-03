"""Backtest path skips on any ETH/SOL symbol shortfall (parity with live).

``compute_signal_from_klines`` (the backtest signal entry) used to DEGRADE
to a BTC-only signal when ETH/SOL klines were missing/short. That degrade
is -EV and was never shippable (see memory project-btc-only-degrade-holdout),
and it diverged from the live path, which SKIPS the round when any symbol
fetch fails. These tests pin the new skip-on-shortfall behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.strategy.momentum_gate import compute_signal_from_klines  # noqa: E402

_LOOKBACKS = (3, 7, 15)
_CANDLE_COUNT = max(_LOOKBACKS) + 1  # 16
_THRESHOLD = 0.001
_CUTOFF_MS = 16_000  # newest candle ts must equal cutoff_ms - 1000 = 15_000


def _klines(*, count: int = _CANDLE_COUNT, trend: float = 5.0) -> list[list]:
    """Synthetic klines: ``count`` candles, 1s apart, newest at cutoff-1000,
    monotone uptrend so the BTC multi-TF gate fires Bull."""
    # Newest candle (index count-1) must land at _CUTOFF_MS - 1000.
    oldest_ts = (_CUTOFF_MS - 1000) - (count - 1) * 1000
    out: list[list] = []
    for i in range(count):
        ts = oldest_ts + i * 1000
        close = 100.0 + i * trend
        out.append([ts, close, close, close, close, 0.0])
    return out


def _call(btc, eth, sol):
    return compute_signal_from_klines(
        btc, _CUTOFF_MS,
        mtf_lookbacks=_LOOKBACKS,
        mtf_min_return_threshold=_THRESHOLD,
        candle_count=_CANDLE_COUNT,
        eth_klines=eth, sol_klines=sol,
    )


def test_all_three_clean_bets_as_before():
    res = _call(_klines(), _klines(), _klines())
    assert res.skip_reason is None
    assert res.signal == "Bull"
    assert res.tier == "multi_tf"


def test_eth_missing_skips():
    res = _call(_klines(), None, _klines())
    assert res.signal is None
    assert res.skip_reason == "gate_eth_klines_unavailable"


def test_eth_short_skips():
    res = _call(_klines(), _klines(count=_CANDLE_COUNT - 1), _klines())
    assert res.signal is None
    assert res.skip_reason == "gate_eth_klines_unavailable"


def test_sol_missing_skips():
    res = _call(_klines(), _klines(), None)
    assert res.signal is None
    assert res.skip_reason == "gate_sol_klines_unavailable"


def test_sol_short_skips():
    res = _call(_klines(), _klines(), _klines(count=_CANDLE_COUNT - 1))
    assert res.signal is None
    assert res.skip_reason == "gate_sol_klines_unavailable"


def test_no_longer_degrades_to_btc_only():
    """The pre-change behavior would BET (BTC-only) with ETH+SOL both
    absent. The aligned behavior skips — ETH is checked first."""
    res = _call(_klines(), None, None)
    assert res.signal is None
    assert res.skip_reason == "gate_eth_klines_unavailable"


def test_btc_shortfall_still_skips_before_symbol_check():
    """A BTC shortfall is still the first gate (unchanged)."""
    res = _call(_klines(count=_CANDLE_COUNT - 1), _klines(), _klines())
    assert res.signal is None
    assert res.skip_reason is not None
    assert res.skip_reason.startswith("gate_btc")
