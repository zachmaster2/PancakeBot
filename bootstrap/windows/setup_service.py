r"""Register the Windows live + dry services, then apply policy via the adapter.

Windows specialness: the services are hosted by ``pythonservice.exe`` (a
separate embedded interpreter that ignores ``pyvenv.cfg``), which requires DLL
relocation + per-service registry fixups (PythonClass + PYTHONPATH). That
host-specific registration is the proven ``scripts/install_services.ps1`` and
is NOT reinvented here. This script:

  1. (optionally) invokes ``scripts/install_services.ps1`` for the registration
     + DLL/registry fixups (skip with --skip-register if already registered),
  2. applies restart-on-failure + Dnscache/NlaSvc dependencies via the
     WindowsServicePlatform adapter (so policy lives in one cross-platform
     place), and
  3. leaves both services DISABLED — enabling is the explicit
     ``scripts/enable_live.ps1`` / ``enable_dry.ps1`` step.

Run from an elevated prompt. Usage:
    <venv>\Scripts\python.exe bootstrap\windows\setup_service.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bootstrap.common.service_specs import build_spec  # noqa: E402
from pancakebot.service import get_platform  # noqa: E402


def _log(msg: str) -> None:
    print(f"[windows/setup_service] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv-python", default=str(_REPO_ROOT / ".venv" / "Scripts" / "python.exe"))
    ap.add_argument("--skip-register", action="store_true",
                    help="services already registered (pythonservice host); only re-apply policy")
    args = ap.parse_args(argv)

    plat = get_platform()
    if plat.name != "windows":
        _log(f"refusing to run: platform is {plat.name}, not windows")
        return 1

    if not args.skip_register:
        ps1 = _REPO_ROOT / "scripts" / "install_services.ps1"
        _log(f"registering services via {ps1} (pythonservice host + DLL/registry fixups)")
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
            check=True,
        )

    # Re-apply restart/deps policy through the adapter (idempotent sc.exe config).
    venv_py = Path(args.venv_python)
    for mode in ("live", "dry"):
        spec = build_spec(mode=mode, repo_root=_REPO_ROOT, venv_python=venv_py)
        _log(f"applying policy to {spec.name}: restart={spec.restart_max_attempts}x, deps=network")
        plat.set_restart_on_failure(
            spec.name, max_attempts=spec.restart_max_attempts,
            reset_window_s=spec.restart_reset_window_s, delay_s=spec.restart_delay_s,
        )
        plat.set_service_dependencies(spec.name, requires_network=spec.requires_network)
        plat.disable_auto_start(spec.name)
    _log("services registered + policy applied (both left DISABLED; use enable_*.ps1 to start)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
