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


# ---------------------------------------------------------------------------
# Actual fetch-fire offset (Regime A/B telemetry, added 2026-06-02): the
# t_features_start_offset_ms column surfaces when the dynamic wake is bypassed.
# ---------------------------------------------------------------------------

def test_cycle_audit_schema_has_actual_fetch_fire_column(tmp_path):
    """The t_features_start_offset_ms column exists and sits adjacent to the
    computed kline_fire_offset_before_lock_ms for easy A/B reading."""
    from pancakebot.runtime.audit import ensure_cycle_audit_csv
    p = tmp_path / "c.csv"
    ensure_cycle_audit_csv(str(p))
    with open(p, newline="") as f:
        cols = next(csv.reader(f))
    assert "t_features_start_offset_ms" in cols
    assert (cols.index("t_features_start_offset_ms")
            == cols.index("kline_fire_offset_before_lock_ms") + 1)


def test_cycle_audit_logs_actual_fetch_fire_time(tmp_path, monkeypatch):
    """A BET row (both dry and live) records t_features_start_offset_ms — the
    ACTUAL fetch-fire offset — distinct from the COMPUTED kline_fire_offset. The
    Regime-B signature (actual > computed = dynamic wake bypassed) round-trips."""
    from pancakebot import paths as _paths_mod
    dry_path = tmp_path / "dry" / "cycle_audit.csv"
    live_path = tmp_path / "live" / "cycle_audit.csv"
    monkeypatch.setattr(_paths_mod, "DRY_CYCLE_AUDIT_PATH", str(dry_path), raising=True)
    monkeypatch.setattr(_paths_mod, "LIVE_CYCLE_AUDIT_PATH", str(live_path), raising=True)

    for dry, path in ((True, dry_path), (False, live_path)):
        _CYCLE_AUDIT_HEADER_OK_PATHS.clear()
        closed = _ClosedState()
        _record_cycle_audit(
            _make_cfg(dry=dry), closed,
            current_epoch=486441, locked_epoch=486440, lock_ts=1780411538, cutoff_ts=1780411536,
            locked_price_bnbusd=600.0, action="BET", decision_stage="pipeline",
            open_round=None, bankroll_before_action_bnb=2.0, bankroll_after_action_bnb=2.0,
            decision=None, decision_latency_ms=274.0, pool_bull_bnb=3.0, pool_bear_bnb=2.5,
            wake_mode="dynamic",
            kline_fire_offset_before_lock_ms=927,      # COMPUTED dynamic wake
            t_features_start_offset_ms=1253.4,         # ACTUAL fire (Regime B: earlier)
            btc_fetch_ms=250, eth_fetch_ms=260, sol_fetch_ms=270,
            btc_fetch_result="ok", eth_fetch_result="ok", sol_fetch_result="ok",
        )
        with open(path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        assert row["action"] == "BET"
        assert row["t_features_start_offset_ms"] == "1253.4"
        assert row["kline_fire_offset_before_lock_ms"] == "927"
        # Regime B: actual fetch fired EARLIER than the computed wake target.
        assert float(row["t_features_start_offset_ms"]) > float(row["kline_fire_offset_before_lock_ms"])


# ---------------------------------------------------------------------------
# Live-mode BET audit row (the hoist fix: BET write moved out of the dry-only
# branch to a mode-agnostic site so live bets land an action=BET row).
# ---------------------------------------------------------------------------

class _StubDecision:
    """Minimal decision stub — record_cycle_audit reads bet_side / bet_size_bnb
    via getattr (audit.py:339-340)."""
    bet_side = "Bull"
    bet_size_bnb = 0.001
    skip_reason = None


def test_live_bet_writes_cycle_audit_row(tmp_path, monkeypatch):
    """A live-mode BET writes exactly one action='BET' row to
    LIVE_CYCLE_AUDIT_PATH. Drives the audit wrapper the way the engine's live
    bet path now does: cfg.dry=False, action='BET', live bankroll pair
    (pre-bet wallet read -> projected = wallet - stake - gas cap).

    Before the hoist, the BET ``_record_cycle_audit`` call lived only inside
    the dry ``else:`` branch, so live bets produced zero BET rows. The
    structural guarantee that the engine now reaches this call in BOTH modes
    is asserted separately in ``test_bet_audit_callsite_is_mode_agnostic``."""
    from pancakebot import paths as _paths_mod
    dry_path = tmp_path / "dry" / "cycle_audit.csv"
    live_path = tmp_path / "live" / "cycle_audit.csv"
    monkeypatch.setattr(_paths_mod, "DRY_CYCLE_AUDIT_PATH", str(dry_path), raising=True)
    monkeypatch.setattr(_paths_mod, "LIVE_CYCLE_AUDIT_PATH", str(live_path), raising=True)

    closed = _ClosedState()  # strategy_pipeline defaults to None (router_mode skipped)
    cfg = _make_cfg(dry=False)
    _record_cycle_audit(
        cfg, closed,
        current_epoch=999, locked_epoch=998, lock_ts=1300, cutoff_ts=1298,
        locked_price_bnbusd=600.0,
        action="BET", decision_stage="pipeline",
        open_round=None,
        bankroll_before_action_bnb=2.3471,   # live: pre-bet wallet read
        bankroll_after_action_bnb=2.3451,    # live: projected (wallet - stake - gas)
        decision=_StubDecision(),
        decision_latency_ms=281.0,
        pool_bull_bnb=3.0, pool_bear_bnb=2.5,
        btc_fetch_ms=270, eth_fetch_ms=265, sol_fetch_ms=275,
        wake_mode="dynamic", kline_fire_offset_before_lock_ms=990,
        btc_fetch_result="ok", eth_fetch_result="ok", sol_fetch_result="ok",
    )

    assert live_path.exists(), "live BET must write LIVE_CYCLE_AUDIT_PATH"
    assert not dry_path.exists(), "live BET must NOT touch DRY_CYCLE_AUDIT_PATH"
    with open(live_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f"expected exactly 1 BET row, got {len(rows)}"
    r = rows[0]
    assert r["action"] == "BET"
    assert r["bet_side"] == "Bull"
    assert float(r["bet_size_bnb"]) == 0.001
    assert float(r["bankroll_before_action_bnb"]) == 2.3471
    assert float(r["bankroll_after_action_bnb"]) == 2.3451


def test_bet_audit_callsite_is_mode_agnostic():
    """Structural guard for the hoist fix: in engine.py ``_run_one_iteration``
    there is exactly ONE ``_record_cycle_audit(action="BET")`` call, and it is
    NOT nested inside any ``cfg.dry``-conditioned ``if/else`` block — so both
    the live and dry bet branches reach it. This directly guards against the
    original regression (BET audit trapped in the dry-only branch) recurring."""
    import ast

    src = (_REPO_ROOT / "pancakebot" / "runtime" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_run_one_iteration"),
        None,
    )
    assert fn is not None, "engine._run_one_iteration not found"

    # Parent map for the function subtree.
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    def _is_bet_audit_call(node: ast.AST) -> bool:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_record_cycle_audit"):
            return False
        for kw in node.keywords:
            if (kw.arg == "action" and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "BET"):
                return True
        return False

    bet_calls = [n for n in ast.walk(fn) if _is_bet_audit_call(n)]
    assert len(bet_calls) == 1, (
        f"expected exactly 1 _record_cycle_audit(action='BET') call in "
        f"_run_one_iteration, found {len(bet_calls)}"
    )

    def _references_dry(test: ast.AST) -> bool:
        return any(isinstance(d, ast.Attribute) and d.attr == "dry"
                   for d in ast.walk(test))

    # Walk ancestors of the BET call; none may be an `if ... cfg.dry ...` block.
    node: ast.AST | None = bet_calls[0]
    while node is not None and id(node) in parents:
        parent = parents[id(node)]
        if isinstance(parent, ast.If) and _references_dry(parent.test):
            raise AssertionError(
                "BET cycle_audit call is nested inside a cfg.dry-conditioned "
                "if/else — it must be hoisted to a mode-agnostic site so live "
                "bets also write a BET row."
            )
        node = parent
