"""Tests for the off-critical-path send caches on Web3PredictionContract
(nonce + gas price) — the 2026-06-06 pre-cache rework that drops the bet
critical path from two cold rotated RPCs (~220ms) to build+sign+send_raw.

Pins:
  - the bet build path reads the CACHED nonce + an explicit chainId and makes
    NO get_transaction_count / gas_price RPC (the whole point: a hot path)
  - nonce advances monotonically across successive sends (bet -> claim)
  - a send error invalidates the cache so the next preflight wake re-prefetches
  - prefetch reconciles + surfaces drift, adopting chain truth
  - send_caches_ready gates fail-loud on unpopulated / stale gas
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest import mock

import pytest

from pancakebot.chain.prediction_contract import (
    Web3PredictionContract,
    _GAS_CACHE_MAX_AGE_MS,
)
from pancakebot.constants import EXPECTED_CHAIN_ID, MAX_GAS_PRICE_WEI
from pancakebot.util import InvariantError


def _bare() -> Web3PredictionContract:
    """A Web3PredictionContract with only the cache state set — no real RPC
    wiring. The real cache methods resolve via the class."""
    c = object.__new__(Web3PredictionContract)
    c._cached_nonce = None
    c._cached_gas_price_wei = None
    c._cached_gas_price_mono_ms = None
    c._account = None
    return c


# -- nonce cache mechanics --------------------------------------------------

def test_next_nonce_returns_cached_without_rpc():
    c = _bare()
    c._cached_nonce = 77
    c._w3 = SimpleNamespace(eth=SimpleNamespace(
        get_transaction_count=mock.Mock(side_effect=AssertionError("RPC on hot path"))))
    assert c._next_nonce() == 77
    assert not c._w3.eth.get_transaction_count.called


def test_next_nonce_unpopulated_raises():
    with pytest.raises(InvariantError, match="nonce_cache_unpopulated"):
        _bare()._next_nonce()


def test_on_send_success_advances_monotonically():
    """bet -> claim within a round: two sends advance the cached nonce by 2."""
    c = _bare()
    c._cached_nonce = 100
    c._on_send_success()   # bet accepted
    assert c._cached_nonce == 101
    c._on_send_success()   # claim accepted
    assert c._cached_nonce == 102


def test_on_send_success_unpopulated_raises():
    with pytest.raises(InvariantError, match="nonce_cache_unpopulated_on_increment"):
        _bare()._on_send_success()


def test_invalidate_nonce_forces_reprefetch():
    c = _bare()
    c._cached_nonce = 100
    c._invalidate_nonce()
    assert c._cached_nonce is None


# -- bet build path makes ZERO nonce/gas RPC --------------------------------

def test_build_bet_tx_uses_cached_nonce_no_rpc():
    c = _bare()
    c._cached_nonce = 42
    c._account = SimpleNamespace(address="0xWALLET")
    # echo build_transaction so we can inspect the assembled dict
    c._contract = SimpleNamespace(functions=SimpleNamespace(
        betBull=lambda epoch: SimpleNamespace(build_transaction=lambda d: dict(d))))
    # ANY nonce/gas RPC on this path is a bug -> blow up
    gtc = mock.Mock(side_effect=AssertionError("get_transaction_count on hot path"))
    gp = mock.Mock(side_effect=AssertionError("gas_price on hot path"))
    c._w3 = SimpleNamespace(eth=SimpleNamespace(get_transaction_count=gtc, gas_price=gp))
    # bypass endpoint rotation (no real providers in this bare instance)
    c._rpc_call = lambda *, op, fn: fn()

    tx = c._build_bet_tx(side="Bull", epoch=123, amount_wei=1000,
                         gas_limit=200_000, gas_price_wei=MAX_GAS_PRICE_WEI)
    assert tx["nonce"] == 42
    assert tx["chainId"] == int(EXPECTED_CHAIN_ID)
    assert tx["from"] == "0xWALLET"
    assert not gtc.called and not gp.called


# -- prefetch reconcile / drift ---------------------------------------------

def test_prefetch_nonce_populates_from_chain():
    c = _bare()
    c._account = SimpleNamespace(address="0xWALLET")
    c._rpc_call = lambda *, op, fn: 555    # stand in for the chain "pending" count
    c.prefetch_nonce()
    assert c._cached_nonce == 555


def test_prefetch_nonce_noop_without_account():
    c = _bare()           # _account is None (dry / unsigned)
    c.prefetch_nonce()
    assert c._cached_nonce is None


def test_prefetch_nonce_reconciles_and_surfaces_drift():
    c = _bare()
    c._account = SimpleNamespace(address="0xWALLET")
    c._cached_nonce = 200                  # local belief
    c._rpc_call = lambda *, op, fn: 205    # chain truth disagrees
    with mock.patch("pancakebot.chain.prediction_contract.warn") as m_warn:
        c.prefetch_nonce()
    assert c._cached_nonce == 205          # adopt chain truth
    assert m_warn.called
    assert "NONCE_RECONCILE" in m_warn.call_args[0][1]


def test_prefetch_nonce_no_warn_when_consistent():
    c = _bare()
    c._account = SimpleNamespace(address="0xWALLET")
    c._cached_nonce = 200
    c._rpc_call = lambda *, op, fn: 200    # agrees -> no reconcile
    with mock.patch("pancakebot.chain.prediction_contract.warn") as m_warn:
        c.prefetch_nonce()
    assert not m_warn.called


# -- gas cache + readiness gate ---------------------------------------------

def test_refresh_gas_price_populates_and_timestamps():
    c = _bare()
    c._rpc_call = lambda *, op, fn: 50_000_000
    c.refresh_gas_price()
    assert c._cached_gas_price_wei == 50_000_000
    assert c._cached_gas_price_mono_ms is not None
    assert (c.gas_cache_age_ms() or -1) >= 0


def test_send_caches_ready_requires_both_populated_and_fresh():
    c = _bare()
    assert c.send_caches_ready() is False           # nothing populated
    c._cached_nonce = 1
    assert c.send_caches_ready() is False            # gas missing
    c._cached_gas_price_wei = 50_000_000
    c._cached_gas_price_mono_ms = time.perf_counter() * 1000.0
    assert c.send_caches_ready() is True             # both + fresh
    c._cached_gas_price_mono_ms = (
        time.perf_counter() * 1000.0 - (_GAS_CACHE_MAX_AGE_MS + 1000.0)
    )
    assert c.send_caches_ready() is False            # gas stale
