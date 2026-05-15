"""Tests for ``engine._utc_now()``.

Bundle 5 v2 (2026-05-14): the application-level NTP sync layer
(``NtpSync`` in ``pancakebot/runtime/ntp_sync.py``) is retired. The bot
trusts the OS clock directly (Windows Time Service kept tight via
MaxPollInterval=5; see README "W32Time prerequisite"). ``_utc_now()``
is now a thin alias for ``time.time()``, preserved as a separate
function so call sites that compare local time against chain-anchored
values remain self-documenting.

This file used to test the NTP-corrected ``_utc_now()`` behavior;
those tests are obsolete and replaced by the two minimal contracts
below.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime import engine  # noqa: E402


def test_utc_now_returns_time_time():
    """_utc_now is a thin alias for time.time() — no offset applied."""
    with mock.patch("pancakebot.runtime.engine.time.time", return_value=12345.6):
        assert engine._utc_now() == 12345.6


def test_utc_now_tracks_time_advance():
    """Successive calls return successive ``time.time()`` values."""
    returns = iter([100.0, 100.5, 101.0])
    with mock.patch(
        "pancakebot.runtime.engine.time.time",
        side_effect=lambda: next(returns),
    ):
        assert engine._utc_now() == 100.0
        assert engine._utc_now() == 100.5
        assert engine._utc_now() == 101.0
