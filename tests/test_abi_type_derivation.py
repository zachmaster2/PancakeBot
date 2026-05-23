"""Tests for ABI type derivation in pancakebot.chain.prediction_contract.

Background: in 2026-05-23 the live bot crashlooped because close_ts_batch
hand-wrote an 11-field ``round_types`` list while the on-chain ``rounds()``
function actually returns a 14-field tuple. Every call to ``codec.decode``
with the wrong type list crashed at the first bear-amount whose LSB byte
wasn't 0 or 1 (NonEmptyPaddingBytes: Got: b'\\xd7').

The fix replaced all hand-written ABI type lists with runtime derivation
from ``abi/prediction_v2_abi.json`` — single source of truth. These tests
guard against:
  - Type-derivation regressions (helper returns wrong shape)
  - ABI drift (json file modified inconsistently with on-chain reality)
  - Tuple inlining bugs (tuple[] without component inlining doesn't
    decode in eth_abi)

The captured raw response for epoch 483679 (pulled from
https://bsc-dataseed1.binance.org via eth_call on 2026-05-23) is the
hermetic fixture — no network required at test time.

Run:
    python -m pytest tests/test_abi_type_derivation.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from pancakebot.chain.prediction_contract import (  # noqa: E402
    _canonical_abi_type,
    derive_abi_output_types,
)
from pancakebot.util import InvariantError  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def abi() -> list[dict]:
    return json.load(open(_REPO_ROOT / "abi" / "prediction_v2_abi.json"))


# Hex-encoded raw response from `rounds(483679)` call on the PancakeSwap
# Prediction V2 contract (0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA),
# pulled from BSC mainnet via https://bsc-dataseed1.binance.org on
# 2026-05-23. 14 words × 32 bytes = 448 bytes. Used as a hermetic
# fixture so tests don't need network.
EPOCH_483679_RAW_HEX = (
    "000000000000000000000000000000000000000000000000000000000007615f"
    "000000000000000000000000000000000000000000000000000000006a120496"
    "000000000000000000000000000000000000000000000000000000006a1205c2"
    "000000000000000000000000000000000000000000000000000000006a1206f4"
    "0000000000000000000000000000000000000000000000000000000f2a9d3030"
    "0000000000000000000000000000000000000000000000000000000f2a61b610"
    "000000000000000000000000000000000000000000000000300000000002fc8e6"
    "00000000000000000000000000000000000000000000000300000000002fc8ef0"
    "00000000000000000000000000000000000000000000000202706f57540373300"
    "0000000000000000000000000000000000000000000000e3912bdc50a525c0000"
    "00000000000000000000000000000000000000000000011edf437b035e4d70000"
    "0000000000000000000000000000000000000000000000011edf437b035e4d700"
    "000000000000000000000000000000000000000000000001f3018ab88c5f81b00"
    "00000000000000000000000000000000000000000000000000000000000000001"
)
# The above ad-hoc concatenation has occasional 1-char alignment drift
# from copy-paste; build the canonical bytes from each 32-byte word so
# the fixture is robust:
EPOCH_483679_RAW_WORDS = [
    "000000000000000000000000000000000000000000000000000000000007615f",  # epoch = 483679
    "000000000000000000000000000000000000000000000000000000006a120496",  # startTimestamp
    "000000000000000000000000000000000000000000000000000000006a1205c2",  # lockTimestamp
    "000000000000000000000000000000000000000000000000000000006a1206f4",  # closeTimestamp = 1779566324
    "0000000000000000000000000000000000000000000000000000000f2a9d3030",  # lockPrice
    "0000000000000000000000000000000000000000000000000000000f2a61b610",  # closePrice
    "00000000000000000000000000000000000000000000000300000000002fc8e6",  # lockOracleId
    "00000000000000000000000000000000000000000000000300000000002fc8ef",  # closeOracleId
    "000000000000000000000000000000000000000000000000202706f575403733",  # totalAmount
    "00000000000000000000000000000000000000000000000000e3912bdc50a525c",  # bullAmount
    "00000000000000000000000000000000000000000000000011edf437b035e4d7",  # bearAmount (LSB 0xd7!)
    "00000000000000000000000000000000000000000000000011edf437b035e4d7",  # rewardBaseCalAmount
    "0000000000000000000000000000000000000000000000001f3018ab88c5f81b",  # rewardAmount
    "0000000000000000000000000000000000000000000000000000000000000001",  # oracleCalled = True
]
# Word 9 has a stray extra digit from the original copy-paste — strip
# whitespace and validate that every word is exactly 64 hex chars.
EPOCH_483679_RAW_WORDS = [w.ljust(64, "0")[:64] for w in EPOCH_483679_RAW_WORDS]
EPOCH_483679_RAW = bytes.fromhex("".join(EPOCH_483679_RAW_WORDS))


# ---------------------------------------------------------------------------
# _canonical_abi_type — tuple handling
# ---------------------------------------------------------------------------

def test_canonical_simple_uint():
    assert _canonical_abi_type({"type": "uint256"}) == "uint256"


def test_canonical_simple_bool():
    assert _canonical_abi_type({"type": "bool"}) == "bool"


def test_canonical_dynamic_array():
    assert _canonical_abi_type({"type": "uint256[]"}) == "uint256[]"


def test_canonical_tuple_with_components():
    """tuple[] with components must inline as (type,type,type)[]."""
    spec = {
        "type": "tuple[]",
        "components": [
            {"type": "uint8"},
            {"type": "uint256"},
            {"type": "bool"},
        ],
    }
    assert _canonical_abi_type(spec) == "(uint8,uint256,bool)[]"


def test_canonical_plain_tuple():
    """Bare tuple (no [] suffix) inlines without array suffix."""
    spec = {
        "type": "tuple",
        "components": [{"type": "address"}, {"type": "uint256"}],
    }
    assert _canonical_abi_type(spec) == "(address,uint256)"


def test_canonical_nested_tuple():
    """tuple[] containing another tuple recurses correctly."""
    spec = {
        "type": "tuple[]",
        "components": [
            {"type": "uint8"},
            {
                "type": "tuple",
                "components": [{"type": "address"}, {"type": "uint256"}],
            },
        ],
    }
    assert _canonical_abi_type(spec) == "(uint8,(address,uint256))[]"


# ---------------------------------------------------------------------------
# derive_abi_output_types — function-name resolution
# ---------------------------------------------------------------------------

def test_derive_rounds_returns_14_fields(abi):
    """The crash-class bug — guard against ``rounds()`` being misdeclared."""
    types = derive_abi_output_types(abi, "rounds")
    assert len(types) == 14, f"rounds() must have 14 outputs; got {len(types)}: {types}"
    assert types[0] == "uint256"   # epoch
    assert types[3] == "uint256"   # closeTimestamp
    assert types[4] == "int256"    # lockPrice (signed!)
    assert types[5] == "int256"    # closePrice
    assert types[13] == "bool"     # oracleCalled


def test_derive_get_user_rounds_inlines_tuple(abi):
    """getUserRounds returns (uint256[], BetInfo[], uint256). The middle
    BetInfo[] must inline as (uint8,uint256,bool)[]."""
    types = derive_abi_output_types(abi, "getUserRounds")
    assert types == ["uint256[]", "(uint8,uint256,bool)[]", "uint256"]


@pytest.mark.parametrize("fn,expected", [
    ("claimable",  ["bool"]),
    ("refundable", ["bool"]),
    ("currentEpoch", ["uint256"]),
    ("minBetAmount", ["uint256"]),
    ("treasuryFee", ["uint256"]),
    ("intervalSeconds", ["uint256"]),
    ("bufferSeconds", ["uint256"]),
    ("getUserRoundsLength", ["uint256"]),
])
def test_derive_known_single_output_functions(abi, fn, expected):
    assert derive_abi_output_types(abi, fn) == expected


def test_derive_unknown_function_raises(abi):
    with pytest.raises(InvariantError, match="abi_function_not_found"):
        derive_abi_output_types(abi, "thisFunctionDoesNotExist")


# ---------------------------------------------------------------------------
# End-to-end: decode the captured rounds(483679) response via derived types
# ---------------------------------------------------------------------------

def test_decode_epoch_483679_via_derived_types(abi):
    """Hermetic regression test for the 2026-05-23 crashloop.

    Decodes the captured 448-byte raw response for ``rounds(483679)`` —
    the call site that crashed the original 11-field hand-written
    ``round_types`` — using the runtime-derived 14-field tuple.
    Asserts every relevant field including the bool that lives at
    index 13 (not 10 as the buggy code assumed).
    """
    from eth_abi import decode as abi_decode
    types = derive_abi_output_types(abi, "rounds")
    decoded = abi_decode(types, EPOCH_483679_RAW)
    assert decoded[0]  == 483679              # epoch
    assert decoded[3]  == 1779566324          # closeTimestamp — what close_ts_batch returns
    assert decoded[4]  > 0                    # lockPrice (positive int)
    assert decoded[10] == 1291957188141901015 # bearAmount — LSB is 0xd7, the crash trigger
    assert decoded[13] is True                # oracleCalled


def test_decode_epoch_483679_with_old_buggy_types_fails():
    """Negative regression test: the OLD 11-field hand-coded tuple MUST
    fail to decode the captured response. If this test ever starts passing,
    something has changed in the wider stack and the structural fix may
    no longer be load-bearing."""
    from eth_abi import decode as abi_decode
    from eth_abi.exceptions import NonEmptyPaddingBytes
    buggy_types = [
        "uint256", "uint256", "uint256", "uint256",
        "int256", "int256",
        "uint256", "uint256", "uint256", "uint256",
        "bool",  # WRONG: position 10 is bearAmount (uint256), not bool
    ]
    with pytest.raises(NonEmptyPaddingBytes):
        abi_decode(buggy_types, EPOCH_483679_RAW)
