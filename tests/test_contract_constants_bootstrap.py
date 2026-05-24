"""Tests for the contract_constants bootstrap helper.

Background: before this helper existed, ``--sync`` would crash with
``contract_constants_cache_missing`` on a clean install because the
Graph round parser called ``load_contract_constants()`` for every
round but ``--sync`` itself never wrote the cache. The cache was only
populated by ``--dry``/``--live`` startup. Caught during the 2026-05-23
var/ cleanup when the user deleted the cache expecting --sync to
regenerate it.

The fix: ``fetch_and_save_contract_constants(contract)`` is the SSOT
for the "read 4 contract constants from chain + persist to disk"
operation. Called by both ``--sync`` (before the Graph parser runs)
and ``--dry``/``--live`` startup.

These tests verify:
  - The helper writes a valid cache file from a mock contract
  - A subsequent load_contract_constants() reads back the same values
  - The "missing then bootstrap then load" flow works end-to-end
    (i.e. --sync from clean state)
  - BNB_WEI conversion is applied to min_bet_amount
  - The error message in load_contract_constants no longer blames the
    user with "(run --sync first)"

Run:
    python -m pytest tests/test_contract_constants_bootstrap.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from pancakebot.constants import BNB_WEI  # noqa: E402
from pancakebot.market_data.contract_constants import (  # noqa: E402
    ContractConstants,
    fetch_and_save_contract_constants,
    load_contract_constants,
    save_contract_constants,
)
from pancakebot.util import InvariantError  # noqa: E402


class _MockContract:
    """Minimal Web3PredictionContract stand-in (only the 4 read methods
    fetch_and_save_contract_constants calls)."""
    def __init__(
        self,
        *,
        min_bet_wei: int = 1_000_000_000_000_000,  # 0.001 BNB
        treasury_fee: float = 0.03,
        interval: int = 300,
        buffer: int = 30,
    ):
        self._min_bet_wei = min_bet_wei
        self._treasury_fee = treasury_fee
        self._interval = interval
        self._buffer = buffer
        self.calls: list[str] = []  # for ordering / count assertions

    def min_bet_amount(self):
        self.calls.append("min_bet_amount")
        return self._min_bet_wei

    def treasury_fee_rate(self):
        self.calls.append("treasury_fee_rate")
        return self._treasury_fee

    def interval_seconds(self):
        self.calls.append("interval_seconds")
        return self._interval

    def buffer_seconds(self):
        self.calls.append("buffer_seconds")
        return self._buffer


# ---------------------------------------------------------------------------
# fetch_and_save_contract_constants — writes a valid cache
# ---------------------------------------------------------------------------

def test_helper_writes_file_with_correct_fields(tmp_path):
    cache_path = tmp_path / "contract_constants.json"
    contract = _MockContract()

    result = fetch_and_save_contract_constants(contract, path=cache_path)

    assert cache_path.exists(), "helper must create the cache file"
    obj = json.loads(cache_path.read_text())
    assert set(obj.keys()) == {
        "min_bet_amount_bnb", "treasury_fee_fraction",
        "interval_seconds", "buffer_seconds",
    }
    assert obj["min_bet_amount_bnb"] == 0.001
    assert obj["treasury_fee_fraction"] == 0.03
    assert obj["interval_seconds"] == 300
    assert obj["buffer_seconds"] == 30

    # Return value matches the persisted state
    assert isinstance(result, ContractConstants)
    assert result.min_bet_amount_bnb == 0.001
    assert result.treasury_fee_fraction == 0.03


def test_helper_applies_bnb_wei_conversion(tmp_path):
    """min_bet_amount() returns wei; helper must divide by BNB_WEI to get BNB."""
    contract = _MockContract(min_bet_wei=5_000_000_000_000_000)  # 0.005 BNB in wei
    cache_path = tmp_path / "cc.json"
    fetch_and_save_contract_constants(contract, path=cache_path)
    obj = json.loads(cache_path.read_text())
    expected_bnb = 5_000_000_000_000_000 / BNB_WEI
    assert obj["min_bet_amount_bnb"] == pytest.approx(expected_bnb)


def test_helper_calls_all_four_contract_methods_exactly_once(tmp_path):
    contract = _MockContract()
    fetch_and_save_contract_constants(contract, path=tmp_path / "cc.json")
    # Exactly one call per chain field, no duplicates / extras
    assert sorted(contract.calls) == sorted([
        "min_bet_amount", "treasury_fee_rate",
        "interval_seconds", "buffer_seconds",
    ])


# ---------------------------------------------------------------------------
# End-to-end: missing cache -> bootstrap -> load (the --sync from clean
# state scenario the structural fix addresses)
# ---------------------------------------------------------------------------

def test_sync_bootstrap_from_clean_state(tmp_path):
    """The exact scenario --sync hits on a clean install:
    cache missing -> bootstrap from chain -> load by Graph parser succeeds."""
    cache_path = tmp_path / "contract_constants.json"
    assert not cache_path.exists(), "precondition: cache must be absent"

    # The load that --sync does indirectly via graph_client._parse_round
    # MUST raise when the cache is missing.
    with pytest.raises(InvariantError, match="contract_constants_cache_missing"):
        load_contract_constants(path=cache_path)

    # Bootstrap step (what --sync now does before sync_runtime_market_data):
    contract = _MockContract()
    written = fetch_and_save_contract_constants(contract, path=cache_path)

    # After bootstrap, the same load succeeds.
    loaded = load_contract_constants(path=cache_path)
    assert loaded == written
    assert loaded.interval_seconds == 300


def test_helper_overwrites_existing_cache(tmp_path):
    """Repeated calls (e.g., every --sync run) refresh the cache from
    current chain state — must not stack or skip the write."""
    cache_path = tmp_path / "cc.json"

    # First run with one set of values.
    fetch_and_save_contract_constants(
        _MockContract(min_bet_wei=1_000_000_000_000_000),
        path=cache_path,
    )
    first = load_contract_constants(path=cache_path)
    assert first.min_bet_amount_bnb == 0.001

    # Chain values change (e.g., governance updated min bet).
    fetch_and_save_contract_constants(
        _MockContract(min_bet_wei=2_000_000_000_000_000),
        path=cache_path,
    )
    second = load_contract_constants(path=cache_path)
    assert second.min_bet_amount_bnb == 0.002


# ---------------------------------------------------------------------------
# Error message — no longer blames the user
# ---------------------------------------------------------------------------

def test_missing_cache_error_message_does_not_blame_user(tmp_path):
    """The old message was '(run --sync first)' which was actively
    misleading after the cache was decoupled from --sync. The new
    message should mention --sync / --dry / --live as the populators
    and direct the operator at RPC/disk-write troubleshooting."""
    cache_path = tmp_path / "missing.json"
    with pytest.raises(InvariantError) as exc_info:
        load_contract_constants(path=cache_path)
    msg = str(exc_info.value)
    # Must mention all three modes that now populate it
    assert "--sync" in msg
    assert "--dry" in msg
    assert "--live" in msg
    # Must direct toward the actual failure modes (network or disk)
    assert "RPC" in msg or "disk" in msg
    # Must NOT use the old "(run --sync first)" phrasing that blames user
    assert "(run --sync first)" not in msg


# ---------------------------------------------------------------------------
# Round-trip preservation: helper output matches load output exactly
# ---------------------------------------------------------------------------

def test_helper_output_matches_loader_output_exactly(tmp_path):
    cache_path = tmp_path / "cc.json"
    contract = _MockContract(
        min_bet_wei=1_500_000_000_000_000,
        treasury_fee=0.0299,
        interval=300,
        buffer=30,
    )
    written = fetch_and_save_contract_constants(contract, path=cache_path)
    loaded = load_contract_constants(path=cache_path)
    assert written == loaded


# ---------------------------------------------------------------------------
# Sanity: save_contract_constants is still functional (back-compat)
# ---------------------------------------------------------------------------

def test_save_contract_constants_independent_of_helper(tmp_path):
    """The lower-level save_contract_constants still works on its own
    (used by tests + as a building block of the helper)."""
    cache_path = tmp_path / "direct.json"
    cc = ContractConstants(
        min_bet_amount_bnb=0.001,
        treasury_fee_fraction=0.03,
        interval_seconds=300,
        buffer_seconds=30,
    )
    out = save_contract_constants(constants=cc, path=cache_path)
    assert out == cache_path
    assert cache_path.exists()
    loaded = load_contract_constants(path=cache_path)
    assert loaded == cc
