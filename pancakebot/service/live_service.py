"""Windows Service: ``PancakeBotLive`` — supervises ``run.py --live``.

Install:    python -m pancakebot.service.live_service install
Remove:     python -m pancakebot.service.live_service remove
Start:      sc.exe start PancakeBotLive  (or scripts/enable_live.ps1)
Stop:       sc.exe stop  PancakeBotLive  (or scripts/disable_live.ps1)

When the service starts, it enforces live-priority by stopping the Dry
service if it is running, then spawns ``python -u run.py --live`` as a
detached subprocess and supervises it.
"""
from __future__ import annotations

# Path bootstrap — when pythonservice.exe hosts this module, the embedded
# interpreter does NOT honor pyvenv.cfg, so site-packages is missing from
# sys.path. We compute the venv's site-packages from __file__ and prepend
# everything needed (repo root for ``pancakebot.*``, venv site-packages for
# ``win32serviceutil``, ``servicemanager``, ``requests``, ``psutil``, ...
# and the ``win32`` subdir for the .pyd extension modules that live there).
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


class PancakeBotLiveService(_PancakeBotServiceBase):
    _svc_name_ = "PancakeBotLive"
    _svc_display_name_ = "PancakeBot Live Trading"
    _svc_description_ = (
        "Supervises the PancakeBot live-trading bot (python run.py --live). "
        "Spawns the bot child on service start, monitors Popen.poll() + "
        "crash.json, restarts on CRASHED/DOWN, sends Discord alerts on "
        "restart patterns. Enforces live-priority by stopping PancakeBotDry "
        "if it is running at service-start."
    )
    _MODE = "live"
    _OTHER_SERVICE = "PancakeBotDry"


if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(PancakeBotLiveService)
