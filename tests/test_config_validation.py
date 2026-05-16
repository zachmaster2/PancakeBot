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
clamp_bet_to_contract_minimum = true

[backtest]
backtest_round_count = 1000
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
# Strict-mode schema validation
# ---------------------------------------------------------------------------

_FULL_TOML = """
[runtime]
kline_cutoff_seconds = 2

[dry]
initial_bankroll_bnb = 50.0

[live]
clamp_bet_to_contract_minimum = true

[backtest]
backtest_round_count = 1000
initial_bankroll_bnb = 50.0
"""


def _write_full_cfg(tmp_path: Path, extra: str = "") -> Path:
    p = tmp_path / "config.toml"
    p.write_text(_FULL_TOML + extra, encoding="utf-8")
    return p


def test_strict_mode_rejects_unknown_key_in_known_section(tmp_path):
    cfg_path = _write_full_cfg(tmp_path, "\n[strategy.risk]\nbogus_key_xyz = 999\n")
    with pytest.raises(InvariantError, match=r"config_unknown_key:.*bogus_key_xyz.*\[strategy\.risk\]"):
        load_app_config(str(cfg_path))


def test_strict_mode_rejects_unknown_section(tmp_path):
    cfg_path = _write_full_cfg(tmp_path, "\n[strategy.unknown_section]\nfoo = 1\n")
    with pytest.raises(InvariantError, match=r"config_unknown_section:.*strategy\.unknown_section"):
        load_app_config(str(cfg_path))


def test_strict_mode_suggests_close_typo_for_key(tmp_path):
    # ``windows_days`` is one char off from ``drawdown_peak_window_days`` --
    # well, far off actually. Try a closer typo.
    cfg_path = _write_full_cfg(tmp_path, "\n[strategy.risk]\ndrawdown_peak_window_dayz = 7\n")
    raised: Exception | None = None
    try:
        load_app_config(str(cfg_path))
    except InvariantError as e:
        raised = e
    assert raised is not None
    assert "did you mean" in str(raised)
    assert "drawdown_peak_window_days" in str(raised)


def test_strict_mode_suggests_close_typo_for_section(tmp_path):
    cfg_path = _write_full_cfg(tmp_path, "\n[strategy.rsk]\ncooldown_rounds = 1\n")
    raised: Exception | None = None
    try:
        load_app_config(str(cfg_path))
    except InvariantError as e:
        raised = e
    assert raised is not None
    assert "did you mean" in str(raised)
    assert "strategy.risk" in str(raised)


def test_strict_mode_accepts_canonical_operator_config():
    """The in-repo config.toml at repo root must always pass strict mode.

    This is the canonical example operators copy from; if it ever drifts
    out of the schema, the loader is broken before any operator ever
    sees their own config.
    """
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_app_config(str(repo_root / "config.toml"))
    assert cfg.kline_cutoff_seconds == 2


def test_strict_mode_rejects_top_level_scalar(tmp_path):
    """Top-level scalars (keys before any [section] header) are rejected.

    TOML scopes scalars to the most-recent section, so the bogus scalar
    must appear BEFORE the first section header to actually be a
    top-level key.
    """
    p = tmp_path / "config.toml"
    p.write_text("bogus_top_scalar = 42\n" + _FULL_TOML, encoding="utf-8")
    with pytest.raises(InvariantError, match=r"config_unknown_top_level_key"):
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
