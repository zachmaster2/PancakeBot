"""Tests for Web3PredictionContract.assert_gas_cap_not_breached().

The bot posts live bet/claim TXs at MAX_GAS_PRICE_WEI (the worst-case
ceiling). If eth.gas_price exceeds the ceiling, the cap is below
current network reality — the operator must lift it before resuming.
The check is invoked before each bet/claim TX; breach is fatal for
THAT TX (raises GasPriceCapBreachedError) but the bot keeps running.

These tests pin:
  - strict > comparison (cap == suggested passes)
  - graceful degradation on transient RPC errors and zero readings
  - the FLOOR < CEIL invariant (constants are sane)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from pancakebot.constants import MAX_GAS_PRICE_WEI
from pancakebot.util import GasPriceCapBreachedError, TransientRpcError


class _FakeContract:
    """Minimal stand-in for Web3PredictionContract that exposes the two
    methods under test. We bind ``assert_gas_cap_not_breached`` from the
    real class so we exercise the real implementation (only
    ``suggest_gas_price_wei`` is mocked per test case)."""

    def __init__(self, suggested_wei_or_exc):
        self._suggested = suggested_wei_or_exc
        # Mirror the real object's gas-cap-bypass streak state (guard audit 4.3).
        self._gas_cap_bypass_streak = 0

    def suggest_gas_price_wei(self) -> int:
        if isinstance(self._suggested, BaseException):
            raise self._suggested
        return int(self._suggested)

    # Bind the real methods so tests exercise production code, not a clone.
    from pancakebot.chain.prediction_contract import Web3PredictionContract
    assert_gas_cap_not_breached = Web3PredictionContract.assert_gas_cap_not_breached
    _note_gas_cap_bypass = Web3PredictionContract._note_gas_cap_bypass


def test_suggested_below_max_passes():
    """When eth.gas_price < MAX_GAS_PRICE_WEI, the assertion returns
    silently. This is the steady-state expected path on BSC (today's
    mainnet floor is ~0.05 Gwei = 5e7 wei; cap is 1 Gwei = 1e9 wei)."""
    c = _FakeContract(MAX_GAS_PRICE_WEI - 1)
    # No exception expected.
    c.assert_gas_cap_not_breached()


def test_suggested_equals_max_passes():
    """Strict greater-than: eth.gas_price == cap is on the boundary, not
    over. The assertion passes — submitting at MAX_GAS_PRICE_WEI matches
    the network exactly, which is still competitive."""
    c = _FakeContract(MAX_GAS_PRICE_WEI)
    c.assert_gas_cap_not_breached()  # No exception.


def test_suggested_above_max_raises():
    """eth.gas_price > cap is the breach condition. Raises with a clear
    operator-action message in the exception text."""
    c = _FakeContract(MAX_GAS_PRICE_WEI + 1)
    with pytest.raises(GasPriceCapBreachedError) as excinfo:
        c.assert_gas_cap_not_breached()
    msg = str(excinfo.value)
    assert "eth.gas_price" in msg
    assert str(MAX_GAS_PRICE_WEI) in msg
    assert "raise the cap" in msg.lower() or "raise the cap" in msg


def test_suggested_zero_warns_and_returns():
    """eth.gas_price == 0 is node misbehavior. Don't raise — proceed
    with MAX (the bet still goes out at the ceiling, no worse than
    steady state). Single-round transient; the next round repeats the
    check."""
    c = _FakeContract(0)
    # No exception expected.
    c.assert_gas_cap_not_breached()


def test_transient_rpc_error_returns_silently():
    """A TransientRpcError fetching eth.gas_price (network timeout,
    parse failure, etc.) does NOT raise. The bet/claim proceeds at MAX
    rather than skip-this-round on a single-round hiccup; sustained
    outages surface via other paths (preflight wake fetch, etc.)."""
    c = _FakeContract(TransientRpcError("simulated_rpc_failure"))
    # No exception expected.
    c.assert_gas_cap_not_breached()


def test_unrelated_exception_propagates():
    """Only TransientRpcError is swallowed; any other exception
    propagates so we don't accidentally mask a real bug. This is a
    correctness pin against future broadening of the except clause."""
    class _UnexpectedError(RuntimeError):
        pass

    c = _FakeContract(_UnexpectedError("not a transient rpc error"))
    with pytest.raises(_UnexpectedError):
        c.assert_gas_cap_not_breached()


def test_constants_satisfy_floor_lt_max_invariant():
    """MAX_GAS_PRICE_WEI must be a positive integer. Pin so a future
    edit doesn't accidentally invert or zero it."""
    assert isinstance(MAX_GAS_PRICE_WEI, int)
    assert MAX_GAS_PRICE_WEI > 0
    # Sanity: cap is well above what BSC charges in 2026 (0.05 Gwei = 5e7).
    # If this fails, either the constant has decayed catastrophically or
    # network economics have changed enough to warrant a doc update.
    assert MAX_GAS_PRICE_WEI >= 100_000_000  # >= 0.1 Gwei
