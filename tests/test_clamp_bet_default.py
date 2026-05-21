"""Tests pinning ``min_bet_only`` default + override behavior.

The implicit default (when the operator's ``config.toml`` omits the
key in ``[live]``) has been ``True`` since the May-16 strict-mode pass
(commit ``7c233a7``). Explicit operator-set values override the default.

These tests guard against a future regression that flips the default
back to ``False`` — promoting full-size live bets to anyone who forgets
to set the key.

Added 2026-05-20 as part of the min-bet-only live-execution-validation
soak prep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.config import load_app_config  # noqa: E402


_BASE_TOML_NO_CLAMP = """
[runtime]
kline_cutoff_seconds = 2

[dry]
initial_bankroll_bnb = 50.0

[live]

[backtest]
backtest_round_count = 1000
initial_bankroll_bnb = 50.0
"""

_BASE_TOML_WITH_CLAMP = """
[runtime]
kline_cutoff_seconds = 2

[dry]
initial_bankroll_bnb = 50.0

[live]
min_bet_only = {clamp}

[backtest]
backtest_round_count = 1000
initial_bankroll_bnb = 50.0
"""


def _write_cfg(tmp_path: Path, *, clamp: str | None = None) -> Path:
    """Write a minimal config.toml. clamp=None omits the key entirely
    (tests the implicit default); otherwise sets it to the given value."""
    p = tmp_path / "config.toml"
    if clamp is None:
        p.write_text(_BASE_TOML_NO_CLAMP, encoding="utf-8")
    else:
        p.write_text(_BASE_TOML_WITH_CLAMP.format(clamp=clamp), encoding="utf-8")
    return p


def test_default_is_true_when_omitted(tmp_path):
    """When ``[live] min_bet_only`` is omitted from
    config.toml, the implicit default is True. This is the safety-first
    posture: anyone who forgets the key still gets contract-minimum bets,
    not full-strategy sizing."""
    cfg = load_app_config(str(_write_cfg(tmp_path, clamp=None)))
    assert cfg.live_min_bet_only is True


def test_explicit_false_overrides_default(tmp_path):
    """Explicit ``min_bet_only = false`` opts out of
    the safety-clamp. This is the "I know what I'm doing, run full bets"
    path. The default must NOT clobber an explicit operator decision."""
    cfg = load_app_config(str(_write_cfg(tmp_path, clamp="false")))
    assert cfg.live_min_bet_only is False


def test_explicit_true_matches_default(tmp_path):
    """Explicit ``min_bet_only = true`` is a no-op
    relative to the implicit default — both produce True. Pinning this
    so the parser doesn't accidentally reject explicit-True as a duplicate
    of the default."""
    cfg = load_app_config(str(_write_cfg(tmp_path, clamp="true")))
    assert cfg.live_min_bet_only is True
