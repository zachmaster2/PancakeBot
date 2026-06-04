"""Windows Service hosts for PancakeBot Live / Dry supervision.

``_PancakeBotServiceBase`` is a thin pywin32 ServiceFramework shell that
delegates the entire supervision lifecycle to the OS-agnostic
``SupervisorCore`` (``supervisor_core.py``), constructed with the
``WindowsServicePlatform`` adapter. The per-mode subclasses
(``PancakeBotLiveService`` / ``PancakeBotDryService``) set ``_MODE`` and
``_OTHER_SERVICE``.

The Linux counterpart is ``pancakebot/service/supervise.py`` — the SAME
``SupervisorCore`` with the ``LinuxServicePlatform`` adapter, run under
systemd. Both produce identical Discord alerts for the same supervision events.
"""
from __future__ import annotations

import traceback
from pathlib import Path

# pywin32: this module is only imported in the Windows service host.
import servicemanager
import win32serviceutil

from pancakebot.service import notifications
from pancakebot.service.supervisor_core import SupervisorCore
from pancakebot.service.windows_platform import WindowsServicePlatform

_REPO_ROOT = Path(__file__).resolve().parents[2]

# venv interpreter for the bot child. CANNOT be ``sys.executable`` — under
# pythonservice.exe that resolves to the service host, which refuses run.py.
_VENV_PYTHON = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"


class _PancakeBotServiceBase(win32serviceutil.ServiceFramework):
    """Abstract base. Subclasses set ``_svc_name_``, ``_svc_display_name_``,
    ``_MODE`` (``"live"``/``"dry"``), and ``_OTHER_SERVICE`` (the other mode's
    SCM service name, for the mutex)."""

    _MODE: str = ""
    _OTHER_SERVICE: str = ""

    def __init__(self, args):
        super().__init__(args)
        platform = WindowsServicePlatform(
            status_reporter=self.ReportServiceStatus,
            log=lambda m: servicemanager.LogWarningMsg(f"{self._svc_name_}: {m}"),
        )
        self._core = SupervisorCore(
            mode=self._MODE,
            other_service=self._OTHER_SERVICE,
            platform=platform,
            repo_root=_REPO_ROOT,
            venv_python=_VENV_PYTHON,
            service_name=self._svc_name_,
            log=self._svc_log,
        )

    def _svc_log(self, level: str, msg: str) -> None:
        if level == "ERROR":
            servicemanager.LogErrorMsg(msg)
        elif level == "WARN":
            servicemanager.LogWarningMsg(msg)
        else:
            servicemanager.LogInfoMsg(msg)

    def SvcStop(self):
        """SCM-initiated stop -> delegate to the core (drains child, signals loop)."""
        self._core.request_stop()

    def SvcDoRun(self):
        """Service main entrypoint. Runs the supervision loop until SvcStop."""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )
        try:
            self._core.run()
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            servicemanager.LogErrorMsg(
                f"{self._svc_name_}: SvcDoRun raised {type(e).__name__}: {e}\n{tb}"
            )
            notifications.notify_service_error(mode=self._MODE, exc=e)
            # Re-raise so SCM marks the service failed and recovery kicks in.
            raise
