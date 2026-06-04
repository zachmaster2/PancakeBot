"""Windows Service wrappers for PancakeBot live/dry supervision.

Two Windows Services (registered via pywin32 / SCM):
    PancakeBotLive  -> python -u run.py --live
    PancakeBotDry   -> python -u run.py --dry

Each service supervises its bot child subprocess: spawns it on service-start,
polls ``Popen.poll()`` + crash.json every 1s, restarts the bot on CRASHED /
DOWN, sends Discord alerts on restart patterns, drains the bot cleanly on
SvcStop.

Replaced the legacy one-shot ``scripts/supervisor.py`` (schtask-driven, opt-in
restart) on 2026-05-23 after a soak window confirmed the service architecture
was stable. See ``var/strategy_review/2026_05_22_supervisor_service_design.md``
for the full design rationale.

Cross-platform supervision (Phase 1, 2026-06-04): the OS-specific service
operations are abstracted behind ``ServicePlatform`` (``platform_base.py``),
with ``WindowsServicePlatform`` and ``LinuxServicePlatform`` adapters.
``get_platform()`` selects the adapter by ``sys.platform`` at call time. The
adapter modules are imported lazily so importing this package on Linux does not
pull in pywin32 (and vice-versa).
"""
from __future__ import annotations

import sys

from pancakebot.service.platform_base import (  # noqa: F401  (re-export)
    HealthState,
    KillTree,
    ServicePlatform,
    ServiceSpec,
    ServiceState,
)

_PLATFORM: "ServicePlatform | None" = None


def get_platform(**kwargs) -> "ServicePlatform":
    """Return the ServicePlatform adapter for the current OS.

    Cached after first construction (kwargs apply only to that first call —
    e.g. the Windows supervisor passing its ``status_reporter``/``log``).
    Lazy-imports the adapter so the other OS's deps are never touched.
    """
    global _PLATFORM
    if _PLATFORM is not None:
        return _PLATFORM
    if sys.platform == "win32":
        from pancakebot.service.windows_platform import WindowsServicePlatform
        _PLATFORM = WindowsServicePlatform(**kwargs)
    else:
        from pancakebot.service.linux_platform import LinuxServicePlatform
        _PLATFORM = LinuxServicePlatform(**kwargs)
    return _PLATFORM


def reset_platform_cache() -> None:
    """Test hook: drop the cached platform so the next get_platform() rebuilds."""
    global _PLATFORM
    _PLATFORM = None
