"""Tests for the engine's NTP clock sync (Era 10 A4 architecture).

Covers ``engine._utc_now()`` reading the cached NTP offset from the
module-level ``NtpSync`` singleton. Per-round ntp_sync_wake behavior
is exercised end-to-end via NtpSync's own test file
(``tests/test_ntp_sync.py``); this file just tests the engine's
interaction surface.

Historical note: this file used to test the prior OKX-server-time
clock-skew compensation (Cristian's algorithm against OKX
``/api/v5/public/time``). That architecture was retired 2026-05-05
in favor of direct NTP queries; see
``project_pancakebot_timing_architecture_history.md`` Era 10.

Run:
    python -m pytest tests/test_engine_clock_skew.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime import engine  # noqa: E402
from pancakebot.runtime.ntp_sync import NtpSync, NtpSyncState  # noqa: E402


def _reset_engine_ntp_singleton():
    engine._ntp_sync = None


def _install_fake_ntp(*, offset_seconds: float = 0.0,
                     last_query_ts: float = 0.0,
                     consecutive_failures: int = 0,
                     last_server: str = "fake.test") -> NtpSync:
    fake = NtpSync.__new__(NtpSync)
    fake._servers = ("fake.test",)
    fake._timeout_s = 1.0
    fake._client = None
    fake._state = NtpSyncState(
        last_offset_seconds=offset_seconds,
        last_query_ts=last_query_ts,
        consecutive_failures=consecutive_failures,
        last_server=last_server,
        successful_queries=1 if last_query_ts > 0 else 0,
    )
    engine._ntp_sync = fake
    return fake


def test_utc_now_subtracts_ntp_offset():
    _reset_engine_ntp_singleton()
    _install_fake_ntp(offset_seconds=0.05)
    with mock.patch("pancakebot.runtime.engine.time.time", return_value=1000.0):
        assert engine._utc_now() == 1000.0 - 0.05
    _reset_engine_ntp_singleton()


def test_utc_now_with_zero_offset_equals_local():
    _reset_engine_ntp_singleton()
    _install_fake_ntp(offset_seconds=0.0)
    with mock.patch("pancakebot.runtime.engine.time.time", return_value=42.0):
        assert engine._utc_now() == 42.0
    _reset_engine_ntp_singleton()


def test_utc_now_handles_negative_offset():
    _reset_engine_ntp_singleton()
    _install_fake_ntp(offset_seconds=-0.025)
    with mock.patch("pancakebot.runtime.engine.time.time", return_value=100.0):
        assert engine._utc_now() == 100.025
    _reset_engine_ntp_singleton()


def test_get_ntp_sync_singleton_lazy_construction():
    _reset_engine_ntp_singleton()
    a = engine._get_ntp_sync()
    b = engine._get_ntp_sync()
    assert a is b
    assert isinstance(a, NtpSync)
    _reset_engine_ntp_singleton()


def test_get_ntp_sync_respects_pre_installed_fake():
    _reset_engine_ntp_singleton()
    fake = _install_fake_ntp()
    assert engine._get_ntp_sync() is fake
    _reset_engine_ntp_singleton()


def test_sleep_until_ts_uses_ntp_corrected_now_in_initial_check():
    _reset_engine_ntp_singleton()
    _install_fake_ntp(offset_seconds=0.5)
    time_returns = iter([
        1000.0,
        1000.0,
        1000.2,
        1000.2,
    ])
    sleep_called = []
    with mock.patch.object(engine.time, "time", side_effect=lambda: next(time_returns)):
        with mock.patch.object(engine, "sleep_seconds",
                               side_effect=lambda s: sleep_called.append(s)):
            engine._sleep_until_ts(999.7, reason="test", epoch=1)
    _reset_engine_ntp_singleton()


def test_sleep_until_ts_short_circuits_when_already_past_in_ntp_frame():
    _reset_engine_ntp_singleton()
    _install_fake_ntp(offset_seconds=0.05)
    with mock.patch("pancakebot.runtime.engine.time.time", return_value=1000.0):
        with mock.patch("pancakebot.runtime.engine.sleep_seconds") as mock_sleep:
            engine._sleep_until_ts(950.0, reason="test", epoch=1)
        assert mock_sleep.call_count == 0, "target in past should short-circuit"
    _reset_engine_ntp_singleton()


def _run_all() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
