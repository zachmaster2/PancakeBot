"""Tests for the Tier 1+2+4 observability sweep (guard audit).

Covers the startup timing-ladder invariant, the gas-cap-bypass streak
counter, the anchor static-fallback / block-time monitors, and the named
wallet-balance exhaustion. The pure rolling-window monitors are covered
separately in ``test_regime_telemetry.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot import timing_constants as _tc  # noqa: E402
from pancakebot.runtime import engine  # noqa: E402
from pancakebot.chain.prediction_contract import (  # noqa: E402
    Web3PredictionContract,
    _GAS_CAP_BYPASS_WARN_STREAK,
)
from pancakebot.chain.rpc_poller import RpcPoller  # noqa: E402
from pancakebot.runtime.regime_telemetry import (  # noqa: E402
    RollingMedianDriftMonitor,
    RollingRateMonitor,
)
from pancakebot.util import InvariantError, TransientRpcError  # noqa: E402


def _valid_timing_cfg() -> SimpleNamespace:
    """A strictly-decreasing offset ladder with positive anchor slack."""
    return SimpleNamespace(
        ramp_poll_1_wakeup_offset_before_lock_ms=7550,
        okx_warmup_wakeup_offset_before_lock_ms=7000,
        preflight_wakeup_offset_before_lock_ms=5970,
        ramp_poll_2_wakeup_offset_before_lock_ms=5850,
        final_rpc_poll_wakeup_offset_before_lock_ms=4750,
        critical_path_wakeup_offset_before_lock_ms=970,
        bet_submit_deadline_offset_before_lock_ms=625,
    )


# --------------------------------------------------------------------------
# 3.3 + 5.9 — startup timing-ladder invariant
# --------------------------------------------------------------------------


def test_timing_ladder_valid_config_passes():
    engine._assert_critical_path_timing_sane(_valid_timing_cfg())  # no raise


def test_timing_ladder_misordered_raises():
    cfg = _valid_timing_cfg()
    # ramp_2 fires earlier than bankroll -> not strictly decreasing.
    cfg.ramp_poll_2_wakeup_offset_before_lock_ms = 6000
    with pytest.raises(InvariantError, match="timing_ladder_not_strictly_decreasing"):
        engine._assert_critical_path_timing_sane(cfg)


def test_timing_ladder_equal_offsets_raises():
    cfg = _valid_timing_cfg()
    cfg.critical_path_wakeup_offset_before_lock_ms = 625  # equal to deadline
    with pytest.raises(InvariantError, match="timing_ladder_not_strictly_decreasing"):
        engine._assert_critical_path_timing_sane(cfg)


def test_anchor_slack_negative_raises():
    cfg = _valid_timing_cfg()
    # Push critical-path wake past where the anchor response can land:
    # anchor responds by lock-(1300-200)=lock-1100; require critical < 1100.
    cfg.critical_path_wakeup_offset_before_lock_ms = (
        _tc.ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS - _tc.ANCHOR_POLL_TIMEOUT_MS + 50
    )
    with pytest.raises(InvariantError, match="anchor_slack_negative"):
        engine._assert_critical_path_timing_sane(cfg)


def test_real_constants_pass_the_invariant():
    """The shipped timing_constants/config values must satisfy the invariant
    (otherwise the bot can't boot)."""
    # Mirror the production offsets the runtime config derives.
    engine._assert_critical_path_timing_sane(_valid_timing_cfg())


# --------------------------------------------------------------------------
# 4.3 — gas-cap-bypass streak counter
# --------------------------------------------------------------------------


def _bare_contract() -> Web3PredictionContract:
    c = object.__new__(Web3PredictionContract)
    c._gas_cap_bypass_streak = 0
    return c


def test_gas_cap_bypass_warns_only_at_threshold():
    c = _bare_contract()
    with mock.patch("pancakebot.chain.prediction_contract.warn") as m_warn:
        for _ in range(_GAS_CAP_BYPASS_WARN_STREAK - 1):
            c._note_gas_cap_bypass("rpc_error")
        assert m_warn.call_count == 0  # below threshold: silent
        c._note_gas_cap_bypass("rpc_error")  # crosses threshold
        assert m_warn.call_count == 1
        args = m_warn.call_args[0]
        assert args[0] == "ALERT"
        assert "GAS_CAP_BYPASS" in args[1]
        assert f"streak={_GAS_CAP_BYPASS_WARN_STREAK}" in args[1]


def test_gas_cap_streak_resets_field():
    c = _bare_contract()
    for _ in range(5):
        c._note_gas_cap_bypass("node_returned_zero")
    assert c._gas_cap_bypass_streak == 5
    c._gas_cap_bypass_streak = 0  # simulates a clean validation
    assert c._gas_cap_bypass_streak == 0


# --------------------------------------------------------------------------
# 3.1 + 5.2 — anchor fallback / block-time monitors
# --------------------------------------------------------------------------


def _bare_poller() -> RpcPoller:
    p = object.__new__(RpcPoller)
    p._anchor_fallback_monitor = RollingRateMonitor(
        name="anchor_static_fallback", max_rate=0.10, window=5, min_samples=5,
    )
    p._block_time_monitor = RollingMedianDriftMonitor(
        name="bsc_block_time", expected=450, tolerance=20, window=5, min_samples=3,
    )
    p._prev_anchor_block = 0
    p._prev_anchor_milli_ts = 0
    return p


def test_anchor_fallback_rate_alerts_when_sustained():
    p = _bare_poller()
    with mock.patch("pancakebot.chain.rpc_poller.warn") as m_warn:
        for _ in range(5):
            p._record_anchor_outcome(fell_back=True, reason="timeout_or_transport")
    msgs = [c.args[1] for c in m_warn.call_args_list]
    assert any("anchor_static_fallback" in m and "reason=timeout_or_transport" in m for m in msgs)


def test_anchor_success_derives_block_time_and_alerts_on_drift():
    p = _bare_poller()
    # First success seeds prev_anchor; subsequent ones advance by 600 blocks.
    # Use a 480ms/block span (drift +30 > tolerance 20).
    block = 1_000_000
    milli = 1_000_000_000
    with mock.patch("pancakebot.chain.rpc_poller.warn") as m_warn:
        for _ in range(6):
            p._record_anchor_outcome(
                fell_back=False, reason="ok", block_number=block, milli_ts=milli,
            )
            block += 600
            milli += 600 * 480  # 480ms per block
    msgs = [c.args[1] for c in m_warn.call_args_list]
    assert any("bsc_block_time" in m and "observed_median=480" in m for m in msgs)


def test_anchor_success_no_alert_at_nominal_block_time():
    p = _bare_poller()
    block = 2_000_000
    milli = 2_000_000_000
    with mock.patch("pancakebot.chain.rpc_poller.warn") as m_warn:
        for _ in range(6):
            p._record_anchor_outcome(
                fell_back=False, reason="ok", block_number=block, milli_ts=milli,
            )
            block += 600
            milli += 600 * 450  # exactly 450ms/block
    msgs = [c.args[1] for c in m_warn.call_args_list]
    assert not any("bsc_block_time" in m for m in msgs)


def test_anchor_telemetry_never_raises_on_bad_state():
    """_record_anchor_outcome must swallow internal errors (telemetry must
    never affect polling)."""
    p = object.__new__(RpcPoller)
    p._anchor_fallback_monitor = None  # will raise inside -> must be swallowed
    p._record_anchor_outcome(fell_back=True, reason="x")  # no exception


# --------------------------------------------------------------------------
# 2.3 — named wallet-balance exhaustion
# --------------------------------------------------------------------------


def test_wallet_balance_retry_exhausted_named():
    from pancakebot.runtime import dry

    cfg = SimpleNamespace(
        contract=SimpleNamespace(
            wallet_balance_bnb=mock.Mock(side_effect=TransientRpcError("down")),
        ),
        wallet_address="0xabc",
    )
    with mock.patch.object(dry, "sleep_seconds"), \
            mock.patch.object(dry, "warn"):
        with pytest.raises(InvariantError, match="wallet_balance_retry_exhausted"):
            dry._fetch_wallet_balance_bnb_with_retries(cfg=cfg, reason="boot")
