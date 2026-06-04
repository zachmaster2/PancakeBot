"""Install the systemd live + dry units via the LinuxServicePlatform adapter.

Idempotent: ``install_service`` overwrites the unit file and reloads systemd.
Leaves both units DISABLED (not auto-started) — enabling is an explicit
operator action (``systemctl enable``) so a fresh install never auto-starts a
live bot. Live/dry mutual exclusion is encoded as ``Conflicts=`` in the unit.

Run as root (writes to /etc/systemd/system). Usage:
    sudo <venv>/bin/python bootstrap/linux/setup_service.py [--enable-dry]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bootstrap.common.service_specs import build_spec  # noqa: E402
from pancakebot.service import get_platform  # noqa: E402


def _log(msg: str) -> None:
    print(f"[linux/setup_service] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv-python", default=str(_REPO_ROOT / ".venv" / "bin" / "python"))
    ap.add_argument("--enable-dry", action="store_true",
                    help="systemctl enable pancakebot-dry after install (default: leave disabled)")
    args = ap.parse_args(argv)

    plat = get_platform()
    if plat.name != "linux":
        _log(f"refusing to run: platform is {plat.name}, not linux")
        return 1

    venv_py = Path(args.venv_python)
    for mode in ("live", "dry"):
        spec = build_spec(mode=mode, repo_root=_REPO_ROOT, venv_python=venv_py)
        _log(f"installing unit {spec.name}.service -> {spec.exe_path} {' '.join(spec.args)}")
        plat.install_service(spec)
    _log("units installed (both left DISABLED; enable explicitly to auto-start)")

    if args.enable_dry:
        _log("enabling pancakebot-dry for the dry soak")
        plat.enable_auto_start("pancakebot-dry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
