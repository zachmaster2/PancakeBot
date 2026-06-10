"""Supervision package for the PancakeBot live/dry bots (Linux/systemd).

Two systemd units (installed by ``bootstrap/linux/setup_service.py``):
    pancakebot-live  -> python -m pancakebot.service.supervise --mode live
    pancakebot-dry   -> python -m pancakebot.service.supervise --mode dry

The supervisor spawns + monitors its bot child subprocess: polls
``Popen.poll()`` + crash.json every 1s, restarts the bot on CRASHED /
DOWN, sends Discord alerts on restart patterns, drains the bot cleanly on
SIGTERM.

Replaced the legacy one-shot ``scripts/supervisor.py`` (schtask-driven,
opt-in restart) on 2026-05-23. The OS-specific service operations are
abstracted behind ``ServicePlatform`` (``platform_base.py``); since the
Phase 3 Windows-host retirement (2026-06-10) the only adapter is
``LinuxServicePlatform`` (the pywin32/SCM ``WindowsServicePlatform`` and
its service-host shell live in the offline archive,
``Downloads/OLD/pancakebot_old/``). The adapter import stays lazy so
importing this package (e.g. on the Windows operator desktop, for tests)
never touches systemd.
"""
from __future__ import annotations

from pancakebot.service.platform_base import (  # noqa: F401  (re-export)
    HealthState,
    KillTree,
    ServicePlatform,
    ServiceSpec,
    ServiceState,
)

_PLATFORM: "ServicePlatform | None" = None


def get_platform(**kwargs) -> "ServicePlatform":
    """Return the ServicePlatform adapter (Linux-only since Phase 3).

    Cached after first construction (kwargs apply only to that first
    call). Constructing ``LinuxServicePlatform`` is OS-neutral — tests on
    the operator desktop may build it; only invoking systemctl-backed
    methods requires a Linux host.
    """
    global _PLATFORM
    if _PLATFORM is not None:
        return _PLATFORM
    from pancakebot.service.linux_platform import LinuxServicePlatform
    _PLATFORM = LinuxServicePlatform(**kwargs)
    return _PLATFORM


def reset_platform_cache() -> None:
    """Test hook: drop the cached platform so the next get_platform() rebuilds."""
    global _PLATFORM
    _PLATFORM = None
