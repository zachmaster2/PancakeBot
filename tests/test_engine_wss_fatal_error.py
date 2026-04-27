"""Engine ``_check_wss_health`` raises on WSS daemon fatal-error.

Phase 2 spec item 17 part A (2026-04-27): without this housekeeping poll,
the bot would silently keep iterating after the WSS daemon escalates --
emitting ``risk_kline_wss_failure`` skips for every round forever, looking
healthy in normal logs while serving nothing. The poll fails loud
(InvariantError) so the supervisor restarts and alerts fire.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime.engine import _check_wss_health  # noqa: E402
from pancakebot.util import InvariantError  # noqa: E402


def _make_gate_with_wss(fatal_value: str | None):
    """Stub gate whose `_wss.fatal_error()` returns *fatal_value*."""
    fake_wss = mock.MagicMock()
    fake_wss.fatal_error = mock.MagicMock(return_value=fatal_value)
    fake_gate = mock.MagicMock()
    fake_gate._wss = fake_wss
    return fake_gate, fake_wss


def test_engine_raises_on_wss_fatal_error_during_housekeeping():
    """When the WSS daemon has set a fatal error, ``_check_wss_health``
    raises ``InvariantError`` with the fatal message embedded."""
    gate, _wss = _make_gate_with_wss(
        fatal_value=(
            "okx_wss_newest_lagging_unrecoverable: symbol=BTC-USDT "
            "3 consecutive newest_lagging without recovery"
        )
    )
    raised = False
    try:
        _check_wss_health(gate)
    except InvariantError as e:
        raised = True
        assert "okx_wss_fatal" in str(e), str(e)
        assert "okx_wss_newest_lagging_unrecoverable" in str(e), str(e)
        assert "BTC-USDT" in str(e), str(e)
    assert raised, "expected InvariantError when fatal_error is set"


def test_engine_no_op_when_wss_healthy():
    """When the WSS daemon is healthy (fatal_error is None), the
    housekeeping check is a silent no-op."""
    gate, _wss = _make_gate_with_wss(fatal_value=None)
    # Must not raise.
    _check_wss_health(gate)


def test_engine_no_op_when_gate_is_none():
    """Backtest / sync / test fixtures may pass gate=None. The check is
    a silent no-op for those paths."""
    _check_wss_health(None)


def test_engine_no_op_when_gate_has_no_wss():
    """A gate constructed without a wss client (test fixtures) is
    silently tolerated -- no crash."""
    bare_gate = mock.MagicMock(spec=[])  # no _wss attribute
    _check_wss_health(bare_gate)


def test_engine_no_op_when_wss_has_no_fatal_error_method():
    """Ducks-type guard: if `_wss` doesn't expose ``fatal_error()``
    (older test stub or unrelated object), the check is a no-op rather
    than crashing on AttributeError."""
    fake_wss_no_method = object()
    fake_gate = mock.MagicMock()
    fake_gate._wss = fake_wss_no_method
    _check_wss_health(fake_gate)


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
