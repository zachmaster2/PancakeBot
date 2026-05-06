"""Tests for the per-round NTP clock-sync manager.

Covers the NtpSync class that the engine consults at each ntp_sync_wake.
The class never crashes on network failure, rotates servers round-robin,
caps glitch offsets at +/-250ms, and exposes a healthy/unhealthy
predicate for the engine's skip-vs-bet decision.

Run::
    python -m pytest tests/test_ntp_sync.py -v
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest import mock

import ntplib

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime.ntp_sync import (  # noqa: E402
    NtpSync,
    _DEFAULT_MAX_CONSECUTIVE_FAILURES,
    _DEFAULT_SERVERS,
    _LAST_GOOD_MAX_AGE_SECONDS,
    _OFFSET_GLITCH_CAP_SECONDS,
)


def _make_response(offset_seconds: float):
    r = mock.MagicMock()
    r.offset = offset_seconds
    return r


def _patch_request(responses: list, /):
    iterator = iter(responses)

    def _side_effect(*args, **kwargs):
        item = next(iterator)
        if isinstance(item, Exception):
            raise item
        return item

    return mock.patch.object(
        ntplib.NTPClient, "request", side_effect=_side_effect,
    )


def test_constructor_rejects_empty_servers():
    raised = None
    try:
        NtpSync(servers=())
    except ValueError as e:
        raised = e
    assert raised is not None


def test_constructor_initial_state_zero():
    n = NtpSync(servers=("a.test", "b.test"))
    assert n.current_offset() == 0.0
    assert n.consecutive_failures() == 0
    assert n.successful_queries() == 0
    assert n.glitch_rejections() == 0
    assert n.last_query_age_seconds() == float("inf")


def test_force_resync_success_caches_offset():
    n = NtpSync(servers=("a.test",))
    with _patch_request([_make_response(0.05)]):
        ok = n.force_resync()
    assert ok is True
    assert n.current_offset() == 0.05
    assert n.successful_queries() == 1
    assert n.consecutive_failures() == 0
    assert n.last_server() == "a.test"


def test_force_resync_advances_server_index_round_robin():
    n = NtpSync(servers=("a.test", "b.test", "c.test"))
    with _patch_request([_make_response(0.01),
                         _make_response(0.02),
                         _make_response(0.03)]):
        n.force_resync()
        assert n.last_server() == "a.test"
        n.force_resync()
        assert n.last_server() == "b.test"
        n.force_resync()
        assert n.last_server() == "c.test"


def test_force_resync_wraps_around_after_last_server():
    n = NtpSync(servers=("a.test", "b.test"))
    with _patch_request([_make_response(0.0)] * 5):
        for _ in range(5):
            n.force_resync()
    assert n.last_server() == "a.test"


def test_force_resync_keeps_prior_offset_on_timeout():
    n = NtpSync(servers=("a.test", "b.test"))
    with _patch_request([_make_response(0.07)]):
        n.force_resync()
    assert n.current_offset() == 0.07
    with _patch_request([ntplib.NTPException("simulated timeout")]):
        ok = n.force_resync()
    assert ok is False
    assert n.current_offset() == 0.07
    assert n.consecutive_failures() == 1


def test_force_resync_keeps_prior_offset_on_arbitrary_exception():
    n = NtpSync(servers=("a.test",))
    with _patch_request([_make_response(0.03)]):
        n.force_resync()
    with _patch_request([RuntimeError("simulated socket error")]):
        ok = n.force_resync()
    assert ok is False
    assert n.current_offset() == 0.03


def test_force_resync_rejects_glitch_above_cap():
    n = NtpSync(servers=("a.test",))
    with _patch_request([_make_response(_OFFSET_GLITCH_CAP_SECONDS + 0.001)]):
        ok = n.force_resync()
    assert ok is False
    assert n.glitch_rejections() == 1
    assert n.consecutive_failures() == 1
    assert n.current_offset() == 0.0


def test_force_resync_rejects_glitch_below_negative_cap():
    n = NtpSync(servers=("a.test",))
    with _patch_request([_make_response(-(_OFFSET_GLITCH_CAP_SECONDS + 0.001))]):
        ok = n.force_resync()
    assert ok is False
    assert n.glitch_rejections() == 1


def test_force_resync_accepts_offset_at_exact_cap():
    n = NtpSync(servers=("a.test",))
    with _patch_request([_make_response(_OFFSET_GLITCH_CAP_SECONDS)]):
        ok = n.force_resync()
    assert ok is True
    assert n.current_offset() == _OFFSET_GLITCH_CAP_SECONDS


def test_consecutive_failures_resets_on_success_after_failure():
    n = NtpSync(servers=("a.test", "b.test"))
    with _patch_request([ntplib.NTPException("fail")]):
        n.force_resync()
    assert n.consecutive_failures() == 1
    with _patch_request([_make_response(0.0)]):
        n.force_resync()
    assert n.consecutive_failures() == 0


def test_is_healthy_false_when_no_query_has_succeeded():
    n = NtpSync(servers=("a.test",))
    assert n.is_healthy() is False


def test_is_healthy_true_after_successful_query():
    n = NtpSync(servers=("a.test",))
    with _patch_request([_make_response(0.0)]):
        n.force_resync()
    assert n.is_healthy() is True


def test_is_healthy_false_when_failure_streak_reaches_threshold():
    n = NtpSync(servers=("a.test", "b.test", "c.test", "d.test"))
    with _patch_request([_make_response(0.0)]):
        n.force_resync()
    failures = [ntplib.NTPException("fail")] * _DEFAULT_MAX_CONSECUTIVE_FAILURES
    with _patch_request(failures):
        for _ in range(_DEFAULT_MAX_CONSECUTIVE_FAILURES):
            n.force_resync()
    assert n.consecutive_failures() == _DEFAULT_MAX_CONSECUTIVE_FAILURES
    assert n.is_healthy() is False


def test_is_healthy_false_when_last_query_too_old():
    n = NtpSync(servers=("a.test",))
    with _patch_request([_make_response(0.01)]):
        n.force_resync()
    n._state.last_query_ts = time.time() - (_LAST_GOOD_MAX_AGE_SECONDS + 1)
    assert n.is_healthy() is False


def test_is_healthy_respects_custom_thresholds():
    n = NtpSync(servers=("a.test",))
    with _patch_request([_make_response(0.0)]):
        n.force_resync()
    assert n.is_healthy(max_age_seconds=1.0) is True
    n._state.last_query_ts = time.time() - 2.0
    assert n.is_healthy(max_age_seconds=1.0) is False


def test_bootstrap_returns_true_on_first_server_success():
    n = NtpSync(servers=("a.test", "b.test", "c.test"))
    with _patch_request([_make_response(0.0)]):
        assert n.bootstrap() is True


def test_bootstrap_falls_through_to_second_server():
    n = NtpSync(servers=("a.test", "b.test"))
    with _patch_request([
        ntplib.NTPException("a down"),
        _make_response(0.005),
    ]):
        assert n.bootstrap() is True
    assert n.last_server() == "b.test"


def test_bootstrap_returns_false_when_all_servers_fail():
    n = NtpSync(servers=("a.test", "b.test", "c.test"))
    with _patch_request([ntplib.NTPException("network down")] * 3):
        assert n.bootstrap() is False
    assert n.consecutive_failures() == 3


def test_bootstrap_treats_glitch_as_failure_not_success():
    n = NtpSync(servers=("a.test", "b.test"))
    with _patch_request([
        _make_response(_OFFSET_GLITCH_CAP_SECONDS + 0.5),
        _make_response(0.01),
    ]):
        assert n.bootstrap() is True
    assert n.last_server() == "b.test"
    assert n.glitch_rejections() == 1


def test_default_servers_includes_three_distinct_pools():
    assert len(_DEFAULT_SERVERS) == 3
    assert len(set(_DEFAULT_SERVERS)) == 3


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
