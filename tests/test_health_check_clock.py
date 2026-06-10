"""Clock-sync step of the post-install health check (bootstrap STEP-0,
2026-06-10): ``chronyc tracking`` must show a synchronized clock with the
offset inside the bot's documented +-250ms truth budget.

Tests target the pure parser (deterministic, no chronyc needed) plus the
skip path for hosts without chronyc (Windows operator desktop).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bootstrap.common.health_check import (  # noqa: E402
    _CLOCK_OFFSET_TOLERANCE_S,
    _clock_sync_ok,
    _parse_chronyc_tracking,
)


_HEALTHY = """\
Reference ID    : C0248D12 (node-1.infogral.is)
Stratum         : 3
Ref time (UTC)  : Tue Jun 10 14:00:00 2026
System time     : 0.000028204 seconds slow of NTP time
Last offset     : +0.000038768 seconds
RMS offset      : 0.000058721 seconds
Frequency       : 11.461 ppm fast
Residual freq   : +0.001 ppm
Skew            : 0.090 ppm
Root delay      : 0.004212 seconds
Root dispersion : 0.000482 seconds
Update interval : 64.2 seconds
Leap status     : Normal
"""


def test_parse_healthy_output_passes():
    ok, detail = _parse_chronyc_tracking(_HEALTHY)
    assert ok is True
    assert "synchronized" in detail
    assert "0.028ms" in detail  # 28.204us rendered in ms


def test_parse_not_synchronised_fails():
    """A freshly-booted / NTP-blocked host reports 'Not synchronised' —
    must FAIL the health check (the bot's wake schedule cannot trust the
    clock)."""
    text = _HEALTHY.replace("Leap status     : Normal",
                            "Leap status     : Not synchronised")
    ok, detail = _parse_chronyc_tracking(text)
    assert ok is False
    assert "NOT synchronized" in detail


def test_parse_offset_over_tolerance_fails():
    """Synchronized-but-way-off (e.g. mid-recovery after a VM pause):
    offset above the +-250ms budget must FAIL."""
    text = _HEALTHY.replace(
        "System time     : 0.000028204 seconds slow of NTP time",
        "System time     : 0.500000000 seconds slow of NTP time",
    )
    ok, detail = _parse_chronyc_tracking(text)
    assert ok is False
    assert "exceeds tolerance" in detail


def test_parse_offset_exactly_at_tolerance_passes():
    """Boundary: offset == tolerance is acceptable (<= semantics)."""
    text = _HEALTHY.replace(
        "System time     : 0.000028204 seconds slow of NTP time",
        f"System time     : {_CLOCK_OFFSET_TOLERANCE_S:.9f} seconds fast of NTP time",
    )
    ok, _ = _parse_chronyc_tracking(text)
    assert ok is True


def test_parse_garbage_output_fails():
    """Unparseable output must fail closed, not pass silently."""
    ok, detail = _parse_chronyc_tracking("chronyd not running\n")
    assert ok is False
    assert "unparseable" in detail


def test_clock_sync_skips_without_chronyc(monkeypatch):
    """Hosts without chronyc (Windows operator desktop) skip the check as
    healthy — the bot host is where it must hold, and install.sh
    guarantees chronyd there."""
    import bootstrap.common.health_check as hc
    monkeypatch.setattr(hc.shutil, "which", lambda name: None)
    ok, detail = _clock_sync_ok()
    assert ok is True
    assert "skipped" in detail


def test_run_refuses_to_start_when_conflicting_unit_is_running(monkeypatch):
    """Conflicts= guard (2026-06-10 incident): the live/dry units are
    mutually exclusive, so health-checking the STOPPED one on a box where
    the partner is RUNNING must FAIL EARLY — starting it would silently
    stop the running (production) bot."""
    import bootstrap.common.health_check as hc
    from pancakebot.service import ServiceState

    monkeypatch.setattr(hc, "_clock_sync_ok", lambda: (True, "test"))
    started: list[str] = []

    class FakePlatform:
        def service_status(self, name):
            return (
                ServiceState.RUNNING if name == "pancakebot-live"
                else ServiceState.STOPPED
            )

        def start_service(self, name):
            started.append(name)

    ok = hc.run(
        mode="dry", service_name="pancakebot-dry",
        start_timeout_s=1.0, ready_timeout_s=1.0,
        platform=FakePlatform(),
    )
    assert ok is False
    assert started == [], (
        "must refuse BEFORE start_service — starting dry would stop live"
    )
