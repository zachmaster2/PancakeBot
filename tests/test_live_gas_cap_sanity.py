"""Tests for Web3PredictionContract.assert_gas_cap_not_breached() (cache-based).

The bot posts live bet/claim TXs at MAX_GAS_PRICE_WEI. The gas price is fetched
OFF the critical path (refresh_gas_price at the preflight wake) and CACHED; the
cap check reads the cache so the bet path makes no gas RPC. The check is
fail-LOUD: it raises GasPriceCapBreachedError (caller skips + alerts) on a real
breach (cached > MAX) AND on a broken cache (unpopulated / stale / zero) — never
a silent proceed-at-MAX, which would re-mask the latency the cache removes and
paper over a broken refresh (pre-cache rework, 2026-06-06).

These tests pin:
  - strict > comparison (cap == cached passes)
  - fail-loud on unpopulated, stale, and zero caches (was: graceful proceed)
  - the freshness bound is honored just under / just over
  - the MAX_GAS_PRICE_WEI sanity invariant
"""
from __future__ import annotations

import time

import pytest

from pancakebot.chain.prediction_contract import _GAS_CACHE_MAX_AGE_MS
from pancakebot.constants import MAX_GAS_PRICE_WEI
from pancakebot.util import GasPriceCapBreachedError


class _FakeContract:
    """Minimal stand-in exposing the cache fields the real
    assert_gas_cap_not_breached + gas_cache_age_ms read. Binds the real methods
    so tests exercise production code, not a clone."""

    def __init__(self, cached_wei, *, age_ms: float = 0.0):
        self._cached_gas_price_wei = cached_wei
        if cached_wei is None:
            # Never populated: leave the timestamp None too.
            self._cached_gas_price_mono_ms = None
        else:
            self._cached_gas_price_mono_ms = time.perf_counter() * 1000.0 - age_ms

    from pancakebot.chain.prediction_contract import Web3PredictionContract
    assert_gas_cap_not_breached = Web3PredictionContract.assert_gas_cap_not_breached
    gas_cache_age_ms = Web3PredictionContract.gas_cache_age_ms


def test_cached_below_max_passes():
    _FakeContract(MAX_GAS_PRICE_WEI - 1).assert_gas_cap_not_breached()  # no raise


def test_cached_equals_max_passes():
    """Strict greater-than: cached == cap is on the boundary, not over."""
    _FakeContract(MAX_GAS_PRICE_WEI).assert_gas_cap_not_breached()  # no raise


def test_cached_above_max_raises():
    c = _FakeContract(MAX_GAS_PRICE_WEI + 1)
    with pytest.raises(GasPriceCapBreachedError) as excinfo:
        c.assert_gas_cap_not_breached()
    msg = str(excinfo.value)
    assert str(MAX_GAS_PRICE_WEI) in msg
    assert "raise the cap" in msg.lower()


def test_unpopulated_cache_raises_fail_loud():
    """None cache (refresh never ran) => fail-loud skip, NOT a silent proceed."""
    with pytest.raises(GasPriceCapBreachedError) as excinfo:
        _FakeContract(None).assert_gas_cap_not_breached()
    assert "unpopulated" in str(excinfo.value)


def test_stale_cache_raises_fail_loud():
    """A cache older than the freshness bound (sustained refresh outage) =>
    fail-loud skip rather than bet on an ancient gas price."""
    c = _FakeContract(MAX_GAS_PRICE_WEI - 1, age_ms=_GAS_CACHE_MAX_AGE_MS + 1000.0)
    with pytest.raises(GasPriceCapBreachedError) as excinfo:
        c.assert_gas_cap_not_breached()
    assert "stale" in str(excinfo.value)


def test_fresh_cache_just_under_bound_passes():
    """A cache within the freshness bound is honored (no raise)."""
    c = _FakeContract(MAX_GAS_PRICE_WEI - 1, age_ms=_GAS_CACHE_MAX_AGE_MS - 1000.0)
    c.assert_gas_cap_not_breached()


def test_zero_cache_raises_fail_loud():
    """A cached 0 (misbehaving node) can't validate the cap => fail-loud."""
    with pytest.raises(GasPriceCapBreachedError) as excinfo:
        _FakeContract(0).assert_gas_cap_not_breached()
    assert "zero" in str(excinfo.value)


def test_constants_satisfy_positive_invariant():
    """MAX_GAS_PRICE_WEI must be a sane positive integer. Pin so a future edit
    doesn't accidentally invert or zero it."""
    assert isinstance(MAX_GAS_PRICE_WEI, int)
    assert MAX_GAS_PRICE_WEI > 0
    assert MAX_GAS_PRICE_WEI >= 100_000_000  # >= 0.1 Gwei
