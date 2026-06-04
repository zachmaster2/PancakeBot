"""Mechanism test for the non-rotating post-TX balance read.

`wallet_balance_bnb` rotates to the NEXT RPC provider before every call. A
balance read immediately after one of our transactions therefore lands on a
sibling node that may lag the just-mined block and return PRE-tx state — the
BET WON stale-bankroll bug (2026-06-03). `wallet_balance_bnb_no_rotate` reads
on the CURRENT provider (the node that confirmed the TX, which provably holds
the block) without rotating. These tests pin both behaviors.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.chain.prediction_contract import Web3PredictionContract  # noqa: E402
from pancakebot.util import TransientRpcError  # noqa: E402

# A real checksummed address so Web3.to_checksum_address passes unmocked.
_ADDR = "0x4898144B11F9DCf6167c65866AE9936b8227b6dF"

# index 0 = the node that confirmed our claim (post-claim, fresh).
# index 1 = a sibling node lagging the claim block (pre-claim, stale).
_FRESH_WEI = 100_012_300_000_000_000_000   # 100.0123 BNB
_STALE_WEI = 99_992_300_000_000_000_000    # 99.9923 BNB
_OTHER_WEI = 100_000_000_000_000_000_000   # 100.0 BNB


def _bare_contract(balances_wei: list[int]) -> Web3PredictionContract:
    """A Web3PredictionContract with mock providers, _rpc_index at 0, bypassing
    __init__ (which needs a real chain)."""
    c = object.__new__(Web3PredictionContract)
    providers = []
    for bal in balances_wei:
        w3 = mock.Mock()
        w3.eth.get_balance = mock.Mock(return_value=bal)
        providers.append((w3, mock.Mock()))
    c._providers = providers
    c._rpc_urls = [f"http://ep{i}" for i in range(len(balances_wei))]
    c._rpc_index = 0
    c._w3, c._contract = providers[0]
    return c


def test_rotating_read_returns_stale_sibling_value():
    """Confirms the bug: the rotating read rotates 0->1 first, so it reads the
    STALE sibling node, not the fresh confirming node."""
    c = _bare_contract([_FRESH_WEI, _STALE_WEI, _OTHER_WEI])
    got = c.wallet_balance_bnb(_ADDR)
    assert got == pytest.approx(99.9923, abs=1e-4)   # sibling (index 1), stale
    assert c._rpc_index == 1                          # rotated away from the confirming node


def test_no_rotate_read_returns_current_node_fresh_value():
    """Confirms the fix: the non-rotating read stays on index 0 (the confirming
    node) and returns the FRESH post-claim balance; the sibling is untouched."""
    c = _bare_contract([_FRESH_WEI, _STALE_WEI, _OTHER_WEI])
    got = c.wallet_balance_bnb_no_rotate(_ADDR)
    assert got == pytest.approx(100.0123, abs=1e-4)   # current node (index 0), fresh
    assert c._rpc_index == 0                            # NOT rotated
    c._providers[0][0].eth.get_balance.assert_called_once()
    c._providers[1][0].eth.get_balance.assert_not_called()  # sibling never queried


def test_no_rotate_wraps_errors_as_transient():
    """A node-side error is wrapped as TransientRpcError so callers fall back."""
    c = _bare_contract([_OTHER_WEI])
    c._providers[0][0].eth.get_balance.side_effect = RuntimeError("node down")
    with pytest.raises(TransientRpcError):
        c.wallet_balance_bnb_no_rotate(_ADDR)
    assert c._rpc_index == 0  # still no rotation even on error
