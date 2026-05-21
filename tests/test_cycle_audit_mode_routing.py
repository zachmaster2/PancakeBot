"""Cycle-audit mode routing: dry → var/dry/cycle_audit.csv,
live → var/live/cycle_audit.csv.

Added 2026-05-20 alongside the live-mode cycle_audit refactor. Verifies:

1. ``_record_cycle_audit`` with ``cfg.dry=True`` writes to
   ``DRY_CYCLE_AUDIT_PATH``.
2. ``_record_cycle_audit`` with ``cfg.dry=False`` writes to
   ``LIVE_CYCLE_AUDIT_PATH``.
3. Both write the same column schema (single source of truth in
   ``audit.py``).
4. The function no longer short-circuits when ``cfg.dry=False`` (the
   prior ``if not cfg.dry: return`` guard is gone).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime.audit import (  # noqa: E402
    ensure_cycle_audit_csv,
    _CYCLE_AUDIT_HEADER_OK_PATHS,
)
from pancakebot.runtime.dry import _record_cycle_audit, _ClosedState  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_header_cache():
    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()
    yield
    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()


def _make_cfg(dry: bool):
    """Stub for RuntimeConfig — only the `dry` field is read in the
    function under test, so a minimal namespace suffices."""
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.dry = dry
    return cfg


def _call(cfg, tmp_path, monkeypatch, dry_path, live_path):
    """Patch the path constants to use tmp_path and call _record_cycle_audit
    with a minimal but complete row. open_round=None is fine because the
    function uses the RPC-fetched pool values when pool_bull_bnb>0 (we set
    those) and only falls back to open_round.bets when both are zero."""
    from pancakebot import paths as _paths_mod
    monkeypatch.setattr(_paths_mod, "DRY_CYCLE_AUDIT_PATH", str(dry_path), raising=True)
    monkeypatch.setattr(_paths_mod, "LIVE_CYCLE_AUDIT_PATH", str(live_path), raising=True)
    # _record_cycle_audit reads _paths.DRY_CYCLE_AUDIT_PATH / .LIVE_... via
    # the module alias `_paths`. The monkeypatches on the original module
    # propagate through that alias.

    closed = _ClosedState()
    closed.simulated_bankroll_bnb = 5.0

    _record_cycle_audit(
        cfg, closed,
        current_epoch=124,
        locked_epoch=123,
        lock_ts=1300,
        cutoff_ts=1298,
        locked_price_bnbusd=600.0,
        action="SKIP",
        decision_stage="pipeline",
        open_round=None,
        bankroll_before_action_bnb=5.0,
        bankroll_after_action_bnb=5.0,
        skip_reason="gate_no_signal",
        decision_latency_ms=275.0,
        pool_bull_bnb=1.0,
        pool_bear_bnb=0.5,
        btc_fetch_ms=270,
        eth_fetch_ms=265,
        sol_fetch_ms=275,
        wake_mode="dynamic",
        kline_fire_offset_before_lock_ms=727,
        btc_fetch_result="ok",
        eth_fetch_result="ok",
        sol_fetch_result="ok",
    )


def test_dry_mode_writes_to_dry_path(tmp_path, monkeypatch):
    """cfg.dry=True writes to var/dry/cycle_audit.csv (DRY_CYCLE_AUDIT_PATH)."""
    dry_path = tmp_path / "dry" / "cycle_audit.csv"
    live_path = tmp_path / "live" / "cycle_audit.csv"
    cfg = _make_cfg(dry=True)
    _call(cfg, tmp_path, monkeypatch, dry_path, live_path)

    assert dry_path.exists(), "dry-mode write should create DRY_CYCLE_AUDIT_PATH"
    assert not live_path.exists(), "dry-mode write must NOT touch LIVE_CYCLE_AUDIT_PATH"

    with open(dry_path, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2, f"expected header + 1 row, got {len(rows)} rows"
    assert rows[0][0] == "cycle_ts"


def test_live_mode_writes_to_live_path(tmp_path, monkeypatch):
    """cfg.dry=False writes to var/live/cycle_audit.csv (LIVE_CYCLE_AUDIT_PATH).
    Before this refactor, the function had `if not cfg.dry: return` and
    live mode wrote nothing."""
    dry_path = tmp_path / "dry" / "cycle_audit.csv"
    live_path = tmp_path / "live" / "cycle_audit.csv"
    cfg = _make_cfg(dry=False)
    _call(cfg, tmp_path, monkeypatch, dry_path, live_path)

    assert live_path.exists(), "live-mode write should create LIVE_CYCLE_AUDIT_PATH"
    assert not dry_path.exists(), "live-mode write must NOT touch DRY_CYCLE_AUDIT_PATH"

    with open(live_path, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2, f"expected header + 1 row, got {len(rows)} rows"
    assert rows[0][0] == "cycle_ts"


def test_dry_and_live_share_schema(tmp_path, monkeypatch):
    """Both modes write the SAME column schema (single source of truth in
    audit.py). Future schema additions touch exactly one place."""
    dry_path = tmp_path / "dry" / "cycle_audit.csv"
    live_path = tmp_path / "live" / "cycle_audit.csv"

    # Two separate temp paths so the file lives independently.
    cfg_dry = _make_cfg(dry=True)
    _call(cfg_dry, tmp_path, monkeypatch, dry_path, live_path)
    # Clear cache so the live-mode write also runs through ensure_*.
    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()
    cfg_live = _make_cfg(dry=False)
    _call(cfg_live, tmp_path, monkeypatch, dry_path, live_path)

    with open(dry_path, newline="") as f:
        dry_header = next(csv.reader(f))
    with open(live_path, newline="") as f:
        live_header = next(csv.reader(f))

    assert dry_header == live_header, (
        f"dry/live schemas diverged: dry has {len(dry_header)} cols, "
        f"live has {len(live_header)} cols. The function is supposed to "
        f"route to mode-specific path while sharing the schema."
    )


def test_live_mode_includes_observability_columns(tmp_path, monkeypatch):
    """The wake_mode + fire_offset + per-symbol fetch_result columns
    (added 2026-05-17 for the wave analysis) MUST exist in live-mode
    output. Regression guard against accidentally bypassing the schema."""
    dry_path = tmp_path / "dry" / "cycle_audit.csv"
    live_path = tmp_path / "live" / "cycle_audit.csv"
    cfg = _make_cfg(dry=False)
    _call(cfg, tmp_path, monkeypatch, dry_path, live_path)

    with open(live_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    # Schema columns added 2026-05-17 must be present and populated
    assert row["wake_mode"] == "dynamic"
    assert row["kline_fire_offset_before_lock_ms"] == "727"
    assert row["btc_fetch_result"] == "ok"
    assert row["eth_fetch_result"] == "ok"
    assert row["sol_fetch_result"] == "ok"
