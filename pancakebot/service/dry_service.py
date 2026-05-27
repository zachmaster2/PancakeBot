"""Windows Service: ``PancakeBotDry`` — supervises ``run.py --dry``.

Install:    python -m pancakebot.service.dry_service install
Remove:     python -m pancakebot.service.dry_service remove
Start:      sc.exe start PancakeBotDry   (or scripts/enable_dry.ps1)
Stop:       sc.exe stop  PancakeBotDry   (or scripts/disable_dry.ps1)

When the service starts, it yields to live-priority by checking whether
``PancakeBotLive`` is running; if so, the dry service refuses to start
and exits cleanly without spawning a bot. Otherwise it spawns
``python -u run.py --dry`` as a detached subprocess and supervises it.
"""
from __future__ import annotations

# Path bootstrap — see live_service.py for rationale.
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENV_SITE = _REPO_ROOT / ".venv" / "Lib" / "site-packages"
for _p in (
    str(_REPO_ROOT),
    str(_VENV_SITE),
    str(_VENV_SITE / "win32"),
    str(_VENV_SITE / "win32" / "lib"),
    str(_VENV_SITE / "Pythonwin"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import win32serviceutil  # noqa: E402

from pancakebot.service.common import _PancakeBotServiceBase  # noqa: E402


class PancakeBotDryService(_PancakeBotServiceBase):
    _svc_name_ = "PancakeBotDry"
    _svc_display_name_ = "PancakeBot Dry Paper Trading"
    _svc_description_ = (
        "Supervises the PancakeBot dry-trading bot (python run.py --dry). "
        "Yields to live-priority: refuses to start if PancakeBotLive is running. "
        "Otherwise spawns the bot child on service start, monitors "
        "Popen.poll() + crash.json, restarts on CRASHED/DOWN, sends "
        "Discord alerts on restart patterns."
    )
    _MODE = "dry"
    _OTHER_SERVICE = "PancakeBotLive"


if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(PancakeBotDryService)
