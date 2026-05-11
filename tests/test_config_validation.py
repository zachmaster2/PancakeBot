"""Tests for ``load_app_config`` runtime-section validation.

Covers ``kline_cutoff_seconds`` range validation: ``cutoff < 1`` would
request a candle past lock_at (which OKX hasn't published), and
``cutoff > 30`` erodes the gate's predictive horizon for no benefit.
The wake-offset cross-validation is enforced separately via
``kline_fetch_wakeup_offset_ms <= kline_cutoff_seconds * 1000 -
OKX_KLINE_PUBLISH_DELAY_P95_MS``; see test_p4c_lock_safety_margin.py
for that path.

Run:
    python -m pytest tests/test_config_validation.py -v
    python tests/test_config_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.config import load_app_config  # noqa: E402
from pancakebot.util import InvariantError  # noqa: E402


_BASE_TOML = """
[runtime]
kline_cutoff_seconds = {cutoff}

[dry]
initial_bankroll_bnb = 50.0

[live]
min_bet_only = true

[backtest]
simulation_size = 1000
initial_bankroll_bnb = 50.0
"""


def _write_cfg(tmp_path: Path, cutoff: int) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(_BASE_TOML.format(cutoff=cutoff), encoding="utf-8")
    return p


@pytest.mark.parametrize("cutoff", [2, 3, 15, 30])
def test_kline_cutoff_seconds_accepts_valid_range(tmp_path, cutoff):
    """Range [1..30] PLUS wake-offset cross-validation:
    kline_fetch_wakeup_offset_ms <= kline_cutoff*1000 -
    OKX_KLINE_PUBLISH_DELAY_P95_MS. With kline_fetch_wakeup=1090 and
    P95=700, the minimum valid cutoff is 2 (=2000ms; 1090 <= 1300).
    """
    cfg = load_app_config(str(_write_cfg(tmp_path, cutoff)))
    assert cfg.kline_cutoff_seconds == cutoff


@pytest.mark.parametrize("cutoff", [-1, 0, 31, 60])
def test_kline_cutoff_seconds_rejects_out_of_range(tmp_path, cutoff):
    """0, negative, and >30 must raise ``InvariantError``."""
    raised: Exception | None = None
    try:
        load_app_config(str(_write_cfg(tmp_path, cutoff)))
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError), (
        f"cutoff={cutoff} must raise InvariantError; got "
        f"{type(raised).__name__}: {raised}"
    )
    assert "kline_cutoff_seconds_out_of_range" in str(raised)


# ---------------------------------------------------------------------------
# hedge_fan_out config knob (LOW-2, 2026-05-10)
# ---------------------------------------------------------------------------

_BASE_TOML_WITH_HEDGE = """
[runtime]
kline_cutoff_seconds = 2
hedge_fan_out = {fan_out}

[dry]
initial_bankroll_bnb = 50.0

[live]
min_bet_only = true

[backtest]
simulation_size = 1000
initial_bankroll_bnb = 50.0
"""


def _write_cfg_with_hedge(tmp_path: Path, fan_out: int) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(
        _BASE_TOML_WITH_HEDGE.format(fan_out=fan_out),
        encoding="utf-8",
    )
    return p


def test_hedge_fan_out_default_is_one_when_absent(tmp_path):
    """When [runtime].hedge_fan_out is absent, default to 1 (legacy
    single-endpoint behaviour, bit-identical with pre-hedging path)."""
    cfg = load_app_config(str(_write_cfg(tmp_path, 2)))
    assert cfg.hedge_fan_out == 1


@pytest.mark.parametrize("fan_out", [1, 2, 3, 4, 5, 16])
def test_hedge_fan_out_accepts_valid_range(tmp_path, fan_out):
    """Range [1..16] (16 is the soft sanity upper bound; the real
    binding constraint of fan_out <= len(endpoint_pool) is enforced
    by RpcPoller's constructor)."""
    cfg = load_app_config(str(_write_cfg_with_hedge(tmp_path, fan_out)))
    assert cfg.hedge_fan_out == fan_out


@pytest.mark.parametrize("fan_out", [-1, 0, 17, 100])
def test_hedge_fan_out_rejects_out_of_range(tmp_path, fan_out):
    """Negative, zero, and >16 must raise InvariantError."""
    raised: Exception | None = None
    try:
        load_app_config(str(_write_cfg_with_hedge(tmp_path, fan_out)))
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError), (
        f"hedge_fan_out={fan_out} must raise InvariantError; got "
        f"{type(raised).__name__}: {raised}"
    )
    assert "hedge_fan_out_out_of_range" in str(raised)


def test_hedge_fan_out_rejects_float(tmp_path):
    """Float values are rejected at the strict-int helper."""
    p = tmp_path / "config.toml"
    p.write_text(
        _BASE_TOML_WITH_HEDGE.replace(
            "hedge_fan_out = {fan_out}", "hedge_fan_out = 2.5"
        ).format(),
        encoding="utf-8",
    )
    with pytest.raises(InvariantError, match="config_key_not_int"):
        load_app_config(str(p))


def _run_all() -> int:
    """Standalone runner (parametrize unrolled manually)."""
    failed = 0
    import tempfile
    for valid in [2, 3, 15, 30]:
        with tempfile.TemporaryDirectory() as td:
            try:
                test_kline_cutoff_seconds_accepts_valid_range(Path(td), valid)
                print(f"PASS  test_kline_cutoff_seconds_accepts_valid_range[{valid}]")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  test_kline_cutoff_seconds_accepts_valid_range[{valid}]: {e}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"ERROR test_kline_cutoff_seconds_accepts_valid_range[{valid}]: "
                      f"{type(e).__name__}: {e}")
    for invalid in [-1, 0, 31, 60]:
        with tempfile.TemporaryDirectory() as td:
            try:
                test_kline_cutoff_seconds_rejects_out_of_range(Path(td), invalid)
                print(f"PASS  test_kline_cutoff_seconds_rejects_out_of_range[{invalid}]")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  test_kline_cutoff_seconds_rejects_out_of_range[{invalid}]: {e}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"ERROR test_kline_cutoff_seconds_rejects_out_of_range[{invalid}]: "
                      f"{type(e).__name__}: {e}")
    print(f"\n{8 - failed}/8 tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
