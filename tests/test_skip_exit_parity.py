"""Byte-level parity net for the SKIP-exit refactor (2026-06-11).

``engine._skip_round`` replaced six literal ``record_cycle_audit`` blocks in
``_run_one_iteration``. The GOLDEN dicts below are transcribed from the
pre-refactor call sites verbatim (same sentinel inputs): for each site, the
helper must hand the dry-mode wrapper exactly the same effective kwargs the
literal block did — any drift (a dropped column, a changed default) fails
here before it can silently skew cycle_audit.csv.

Also pins: site 1's operator log line (level + exact wording) and the
row-level byte-identity of the written CSV row for a representative site.

Run:
    python -m pytest tests/test_skip_exit_parity.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime import engine  # noqa: E402
from pancakebot.runtime.audit import _CYCLE_AUDIT_HEADER_OK_PATHS  # noqa: E402
from pancakebot.runtime.dry import RuntimeState, record_cycle_audit  # noqa: E402
from pancakebot.runtime.engine import _RoundSkipCtx, _kline_result_get, _kline_timing_get  # noqa: E402


# Sentinels shared by every site (mirror the engine's local names).
CURRENT_EPOCH = 124
LOCKED_EPOCH = 123
LOCK_TS = 1300
CUTOFF_TS = 1298
PRICE = 600.0
GATE = None  # gate-less pipeline path; both old and new code call the
             # same _kline_result_get/_kline_timing_get on it.
WAKE_MODE = "dynamic"
KLINE_FIRE = 1040
DECISION = object()  # captured, never written
LATENCY = 123.4
POOL_BULL = 2.5
POOL_BEAR = 1.5
BANKROLL = 5.0
STALE_BANKROLL = 0.0

_COMMON = dict(
    current_epoch=CURRENT_EPOCH,
    locked_epoch=LOCKED_EPOCH,
    lock_ts=LOCK_TS,
    cutoff_ts=CUTOFF_TS,
    locked_price_bnbusd=PRICE,
    action="SKIP",
    open_round=None,
    wake_mode=WAKE_MODE,
    kline_fire_offset_before_lock_ms=KLINE_FIRE,
    btc_fetch_result=_kline_result_get(GATE, "btc"),
    eth_fetch_result=_kline_result_get(GATE, "eth"),
    sol_fetch_result=_kline_result_get(GATE, "sol"),
)

_EXTRAS = dict(
    decision=DECISION,
    decision_latency_ms=LATENCY,
    pool_bull_bnb=POOL_BULL,
    pool_bear_bnb=POOL_BEAR,
    btc_fetch_ms=_kline_timing_get(GATE, "btc_ms"),
    eth_fetch_ms=_kline_timing_get(GATE, "eth_ms"),
    sol_fetch_ms=_kline_timing_get(GATE, "sol_ms"),
)

# The dry-wrapper's own defaults: pre-refactor sites OMITTED these keys
# where unused; the helper passes them explicitly. Omitted-vs-explicit-
# default is row-identical; this map is what "effectively equal" means.
_WRAPPER_DEFAULTS = dict(
    decision=None,
    skip_reason=None,
    decision_latency_ms=None,
    pool_bull_bnb=0.0,
    pool_bear_bnb=0.0,
    btc_fetch_ms=None,
    eth_fetch_ms=None,
    sol_fetch_ms=None,
    wake_mode="",
    kline_fire_offset_before_lock_ms=None,
    t_features_start_offset_ms=None,
    btc_fetch_result="not_fetched",
    eth_fetch_result="not_fetched",
    sol_fetch_result="not_fetched",
)

# (name, golden kwargs from the PRE-refactor literal block, helper args)
SITES = [
    (
        "risk_bankroll_stale",
        {**_COMMON, "decision_stage": "pipeline",
         "bankroll_before_action_bnb": STALE_BANKROLL,
         "bankroll_after_action_bnb": STALE_BANKROLL,
         "skip_reason": "risk_bankroll_stale"},
        dict(skip_reason="risk_bankroll_stale", decision_stage="pipeline",
             bankroll_bnb=STALE_BANKROLL),
    ),
    (
        "pool_not_ready",
        {**_COMMON, "decision_stage": "pipeline",
         "bankroll_before_action_bnb": BANKROLL,
         "bankroll_after_action_bnb": BANKROLL,
         "skip_reason": "pool_not_ready_cold_start_in_progress"},
        dict(skip_reason="pool_not_ready_cold_start_in_progress",
             decision_stage="pipeline", bankroll_bnb=BANKROLL),
    ),
    (
        "pipeline_decision_skip",
        {**_COMMON, **_EXTRAS, "decision_stage": "pipeline",
         "bankroll_before_action_bnb": BANKROLL,
         "bankroll_after_action_bnb": BANKROLL,
         "skip_reason": "gate_no_signal"},
        dict(skip_reason="gate_no_signal", decision_stage="pipeline",
             bankroll_bnb=BANKROLL, decision=DECISION,
             decision_latency_ms=LATENCY, pool_bull_bnb=POOL_BULL,
             pool_bear_bnb=POOL_BEAR, with_fetch_ms=True),
    ),
    (
        "timing_guard",
        {**_COMMON, **_EXTRAS, "decision_stage": "timing_guard",
         "bankroll_before_action_bnb": BANKROLL,
         "bankroll_after_action_bnb": BANKROLL,
         "skip_reason": "too_close_to_lock_for_bet"},
        dict(skip_reason="too_close_to_lock_for_bet",
             decision_stage="timing_guard", bankroll_bnb=BANKROLL,
             decision=DECISION, decision_latency_ms=LATENCY,
             pool_bull_bnb=POOL_BULL, pool_bear_bnb=POOL_BEAR,
             with_fetch_ms=True),
    ),
    (
        "send_cache_unready",
        {**_COMMON, **_EXTRAS, "decision_stage": "send_cache_check",
         "bankroll_before_action_bnb": BANKROLL,
         "bankroll_after_action_bnb": BANKROLL,
         "skip_reason": "risk_send_cache_unready"},
        dict(skip_reason="risk_send_cache_unready",
             decision_stage="send_cache_check", bankroll_bnb=BANKROLL,
             decision=DECISION, decision_latency_ms=LATENCY,
             pool_bull_bnb=POOL_BULL, pool_bear_bnb=POOL_BEAR,
             with_fetch_ms=True),
    ),
    (
        "gas_cap_breached",
        {**_COMMON, **_EXTRAS, "decision_stage": "gas_cap_check",
         "bankroll_before_action_bnb": BANKROLL,
         "bankroll_after_action_bnb": BANKROLL,
         "skip_reason": "gas_cap_breached"},
        dict(skip_reason="gas_cap_breached", decision_stage="gas_cap_check",
             bankroll_bnb=BANKROLL, decision=DECISION,
             decision_latency_ms=LATENCY, pool_bull_bnb=POOL_BULL,
             pool_bear_bnb=POOL_BEAR, with_fetch_ms=True),
    ),
]


def _ctx() -> _RoundSkipCtx:
    return _RoundSkipCtx(
        current_epoch=CURRENT_EPOCH, locked_epoch=LOCKED_EPOCH,
        lock_ts=LOCK_TS, cutoff_ts=CUTOFF_TS, bnbusd_price=PRICE,
        open_round=None, gate=GATE,
    )


def _capture_helper_kwargs(monkeypatch, helper_args) -> dict:
    captured: dict = {}

    def fake_record(cfg, closed, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(engine, "record_cycle_audit", fake_record)
    engine._skip_round(
        object(), object(), _ctx(),
        wake_mode=WAKE_MODE,
        kline_fire_offset_before_lock_ms=KLINE_FIRE,
        **helper_args,
    )
    return captured


@pytest.mark.parametrize("name,golden,helper_args", SITES,
                         ids=[s[0] for s in SITES])
def test_helper_kwargs_match_pre_refactor_literal(monkeypatch, name, golden,
                                                  helper_args):
    captured = _capture_helper_kwargs(monkeypatch, helper_args)
    # Every golden key (from the literal pre-refactor block) must match.
    for k, v in golden.items():
        assert k in captured, f"{name}: helper dropped audit kwarg {k!r}"
        assert captured[k] is v or captured[k] == v, (
            f"{name}: audit kwarg {k!r} drifted: {captured[k]!r} != {v!r}"
        )
    # Any key the helper adds beyond the literal block must equal the
    # wrapper's default (omitted-vs-explicit-default is row-identical).
    for k in set(captured) - set(golden):
        assert k in _WRAPPER_DEFAULTS, f"{name}: unexpected audit kwarg {k!r}"
        assert captured[k] == _WRAPPER_DEFAULTS[k], (
            f"{name}: extra kwarg {k!r}={captured[k]!r} is not the wrapper "
            f"default {_WRAPPER_DEFAULTS[k]!r} — would change the row"
        )


def test_site1_log_line_byte_identical(monkeypatch):
    """Site 1 is the only site whose log emission moved into the helper;
    its wording + level are watcher-visible and pinned here."""
    logs: list[tuple[str, str, str]] = []
    monkeypatch.setattr(engine, "record_cycle_audit", lambda *a, **k: None)
    monkeypatch.setattr(engine, "warn", lambda act, msg: logs.append(("warn", act, msg)))
    monkeypatch.setattr(engine, "info", lambda act, msg: logs.append(("info", act, msg)))
    engine._skip_round(
        object(), object(), _ctx(),
        skip_reason="risk_bankroll_stale", decision_stage="pipeline",
        bankroll_bnb=STALE_BANKROLL, wake_mode="",
        kline_fire_offset_before_lock_ms=None,
        log_level="warn",
        log_line=f"Skipped epoch {CURRENT_EPOCH}: bankroll stale",
    )
    assert logs == [("warn", "SKIP", f"Skipped epoch {CURRENT_EPOCH}: bankroll stale")]


def test_written_row_byte_identical(monkeypatch, tmp_path):
    """End-to-end: the row the dry wrapper writes from the helper's kwargs
    is byte-identical to the row written from the pre-refactor literal
    kwargs (site 3 shape, decision=None so the row is writable)."""
    from pancakebot import paths as _paths_mod
    monkeypatch.setattr(_paths_mod, "DRY_CYCLE_AUDIT_PATH",
                        str(tmp_path / "golden.csv"), raising=True)
    monkeypatch.setattr(_paths_mod, "LIVE_CYCLE_AUDIT_PATH",
                        str(tmp_path / "unused_live.csv"), raising=True)

    class _Cfg:
        dry = True

    golden = dict(SITES[2][1])
    golden["decision"] = None

    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()
    closed = RuntimeState()
    record_cycle_audit(_Cfg(), closed, **golden)
    golden_bytes = (tmp_path / "golden.csv").read_bytes()

    helper_args = dict(SITES[2][2])
    helper_args["decision"] = None
    captured = _capture_helper_kwargs(monkeypatch, helper_args)

    monkeypatch.setattr(_paths_mod, "DRY_CYCLE_AUDIT_PATH",
                        str(tmp_path / "helper.csv"), raising=True)
    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()
    record_cycle_audit(_Cfg(), closed, **captured)
    helper_bytes = (tmp_path / "helper.csv").read_bytes()

    assert helper_bytes == golden_bytes
    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()
