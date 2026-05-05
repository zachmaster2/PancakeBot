"""Strict-int coercion tests for the config-loading helpers.

Covers ``_coerce_strict_int`` (the shared int helper used by ``_req_int``
and ``_opt_int``) plus the ``_opt_int_or_none``, ``_opt_int_tuple``, and
``_opt_float`` helpers. The R6 fix (2026-05-04) tightened these to reject
floats outright (no silent ``int(2.5) -> 2`` truncation) and to reject
bools (which were silently accepted because ``bool`` is a subclass of
``int`` in Python).

Run:
    python -m pytest tests/test_config_int_helpers.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.config import (  # noqa: E402
    _coerce_strict_int,
    _opt_float,
    _opt_int,
    _opt_int_or_none,
    _opt_int_tuple,
    _req_int,
)
from pancakebot.util import InvariantError  # noqa: E402


# ---------------------------------------------------------------------------
# _coerce_strict_int / _req_int / _opt_int — accept paths
# ---------------------------------------------------------------------------

def test_coerce_strict_int_accepts_int():
    assert _coerce_strict_int(2, "k") == 2
    assert _coerce_strict_int(0, "k") == 0
    assert _coerce_strict_int(-1, "k") == -1
    assert _coerce_strict_int(2_000_000, "k") == 2_000_000


def test_coerce_strict_int_accepts_int_string():
    """Strings are a deliberate user-typed format (e.g. env-var override).
    Preserve the existing string-parsing behavior locked in by R6.
    """
    assert _coerce_strict_int("2", "k") == 2
    assert _coerce_strict_int("0", "k") == 0
    assert _coerce_strict_int("-15", "k") == -15


# ---------------------------------------------------------------------------
# _coerce_strict_int — reject paths (the R6 fix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v", [2.0, 2.5, 0.5, -3.7])
def test_coerce_strict_int_rejects_float(v):
    """Floats must NOT silently truncate to int; this is the R6 fix.
    Even ``2.0`` (which would truncate cleanly to 2) is rejected: a float
    in an int field is a config error, not a truncation candidate.
    """
    with pytest.raises(InvariantError, match=r"config_key_not_int.*float not allowed"):
        _coerce_strict_int(v, "kline_cutoff_seconds")


@pytest.mark.parametrize("v", [math.nan, math.inf, -math.inf])
def test_coerce_strict_int_rejects_nan_and_inf(v):
    """NaN and +/-inf are floats; same float-reject path catches them."""
    with pytest.raises(InvariantError, match=r"config_key_not_int.*float not allowed"):
        _coerce_strict_int(v, "k")


@pytest.mark.parametrize("v", [True, False])
def test_coerce_strict_int_rejects_bool(v):
    """Bools must be rejected explicitly. ``bool`` is a subclass of ``int``
    in Python, so ``isinstance(True, int)`` is True; without the bool
    check, ``true``/``false`` in TOML would silently coerce to 1/0.
    """
    with pytest.raises(InvariantError, match=r"config_key_not_int.*bool not allowed"):
        _coerce_strict_int(v, "k")


@pytest.mark.parametrize("v", [None, [1, 2], (1, 2), {"a": 1}, b"bytes"])
def test_coerce_strict_int_rejects_other_types(v):
    """Lists, tuples, dicts, bytes, None — all rejected with type info."""
    with pytest.raises(InvariantError, match=r"config_key_not_int"):
        _coerce_strict_int(v, "k")


def test_coerce_strict_int_rejects_unparseable_string():
    with pytest.raises(InvariantError, match=r"config_key_not_int.*str parse failed"):
        _coerce_strict_int("not_an_int", "k")
    with pytest.raises(InvariantError, match=r"config_key_not_int.*str parse failed"):
        _coerce_strict_int("2.5", "k")


def test_coerce_strict_int_error_includes_field_name():
    """Error messages must include the field name so the operator knows
    WHICH config key is wrong."""
    with pytest.raises(InvariantError, match=r"my_special_field"):
        _coerce_strict_int(2.5, "my_special_field")


# ---------------------------------------------------------------------------
# _req_int — wrapper behavior
# ---------------------------------------------------------------------------

def test_req_int_pass():
    assert _req_int({"k": 2}, "k") == 2


def test_req_int_missing_key():
    with pytest.raises(InvariantError, match=r"missing_config_key.*k"):
        _req_int({}, "k")


def test_req_int_rejects_float():
    with pytest.raises(InvariantError, match=r"config_key_not_int.*float"):
        _req_int({"k": 2.0}, "k")


def test_req_int_rejects_bool():
    with pytest.raises(InvariantError, match=r"config_key_not_int.*bool"):
        _req_int({"k": True}, "k")


# ---------------------------------------------------------------------------
# _opt_int — wrapper behavior
# ---------------------------------------------------------------------------

def test_opt_int_uses_default_when_missing():
    assert _opt_int({}, "k", 7) == 7


def test_opt_int_passes_present_int():
    assert _opt_int({"k": 5}, "k", 7) == 5


def test_opt_int_rejects_present_float():
    with pytest.raises(InvariantError, match=r"config_key_not_int.*float"):
        _opt_int({"k": 5.0}, "k", 7)


def test_opt_int_rejects_present_bool():
    with pytest.raises(InvariantError, match=r"config_key_not_int.*bool"):
        _opt_int({"k": True}, "k", 7)


# ---------------------------------------------------------------------------
# _opt_int_or_none — bool subclass of int issue
# ---------------------------------------------------------------------------

def test_opt_int_or_none_returns_int():
    assert _opt_int_or_none({"k": 5}, "k") == 5


def test_opt_int_or_none_returns_none_for_missing():
    assert _opt_int_or_none({}, "k") is None


def test_opt_int_or_none_returns_none_for_non_int():
    assert _opt_int_or_none({"k": "5"}, "k") is None
    assert _opt_int_or_none({"k": 5.0}, "k") is None
    assert _opt_int_or_none({"k": None}, "k") is None


def test_opt_int_or_none_returns_none_for_bool():
    """``bool`` is a subclass of ``int``; without the explicit bool
    check, ``True`` would round-trip as the int 1 -- wrong."""
    assert _opt_int_or_none({"k": True}, "k") is None
    assert _opt_int_or_none({"k": False}, "k") is None


# ---------------------------------------------------------------------------
# _opt_int_tuple — bool elements
# ---------------------------------------------------------------------------

def test_opt_int_tuple_passes_ints():
    assert _opt_int_tuple({"k": [3, 7, 15]}, "k", (1,)) == (3, 7, 15)


def test_opt_int_tuple_uses_default_when_missing():
    assert _opt_int_tuple({}, "k", (1, 2, 3)) == (1, 2, 3)


def test_opt_int_tuple_rejects_non_list():
    with pytest.raises(InvariantError, match=r"config_key_not_int_list.*not a list"):
        _opt_int_tuple({"k": 5}, "k", ())


def test_opt_int_tuple_rejects_bool_element():
    """Elements must be strict ints; bool elements are rejected even
    though ``isinstance(True, int)`` is True."""
    with pytest.raises(InvariantError, match=r"config_key_not_int_list.*bool"):
        _opt_int_tuple({"k": [3, True, 15]}, "k", ())


def test_opt_int_tuple_rejects_float_element():
    with pytest.raises(InvariantError, match=r"config_key_not_int_list.*float"):
        _opt_int_tuple({"k": [3, 7.5, 15]}, "k", ())


# ---------------------------------------------------------------------------
# _opt_float — bool reject; int accepted (no truncation risk)
# ---------------------------------------------------------------------------

def test_opt_float_uses_default_when_missing():
    assert _opt_float({}, "k", 1.5) == 1.5


def test_opt_float_accepts_float():
    assert _opt_float({"k": 1.25}, "k", 0.0) == 1.25


def test_opt_float_accepts_int():
    """Int -> float is safe (no precision loss for typical config
    magnitudes); preserve this behavior so users can write ``5`` instead
    of ``5.0`` in TOML."""
    assert _opt_float({"k": 5}, "k", 0.0) == 5.0


def test_opt_float_rejects_bool():
    with pytest.raises(InvariantError, match=r"config_key_not_number.*bool"):
        _opt_float({"k": True}, "k", 0.0)
    with pytest.raises(InvariantError, match=r"config_key_not_number.*bool"):
        _opt_float({"k": False}, "k", 0.0)


def test_opt_float_rejects_string():
    with pytest.raises(InvariantError, match=r"config_key_not_number"):
        _opt_float({"k": "1.5"}, "k", 0.0)
