"""Cross-platform venv creation + dependency install (idempotent).

Used by ``bootstrap/install.sh`` and ``bootstrap/install.ps1``. Creates
``<repo>/.venv`` with the given interpreter (must satisfy the minimum version)
and installs ``requirements.txt``. Re-running is safe: an existing venv with a
satisfactory interpreter is reused; ``pip install`` is naturally idempotent.

Usage:
    python bootstrap/common/python_setup.py --python /path/to/python3.13
    python bootstrap/common/python_setup.py          # uses sys.executable
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIN_PY = (3, 13)


def _log(msg: str) -> None:
    print(f"[python_setup] {msg}", flush=True)


def venv_python(venv_dir: Path) -> Path:
    """Interpreter path inside a venv, per-OS."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _interp_version(python_exe: str) -> tuple[int, int]:
    out = subprocess.check_output(
        [python_exe, "-c", "import sys;print(f'{sys.version_info[0]} {sys.version_info[1]}')"],
        text=True,
    )
    a, b = out.split()
    return int(a), int(b)


def ensure_venv(*, python_exe: str, venv_dir: Path, requirements: Path) -> Path:
    """Create the venv (if absent) and install requirements. Returns the venv
    interpreter path. Raises on a version shortfall or a failed install."""
    ver = _interp_version(python_exe)
    if ver < _MIN_PY:
        raise SystemExit(
            f"interpreter {python_exe} is {ver[0]}.{ver[1]}; need >= "
            f"{_MIN_PY[0]}.{_MIN_PY[1]} (matches the dev/runtime version)"
        )
    vpy = venv_python(venv_dir)
    if vpy.exists():
        existing = _interp_version(str(vpy))
        if existing >= _MIN_PY:
            _log(f"venv already present at {venv_dir} (python {existing[0]}.{existing[1]}); reusing")
        else:
            raise SystemExit(
                f"existing venv at {venv_dir} is python {existing[0]}.{existing[1]} "
                f"(< {_MIN_PY}); remove it and re-run"
            )
    else:
        _log(f"creating venv at {venv_dir} from {python_exe} ({ver[0]}.{ver[1]})")
        subprocess.run([python_exe, "-m", "venv", str(venv_dir)], check=True)
    _log(f"upgrading pip + installing {requirements.name}")
    subprocess.run([str(vpy), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(vpy), "-m", "pip", "install", "-r", str(requirements)], check=True)
    _log("dependency install complete")
    return vpy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create venv + install deps (idempotent).")
    ap.add_argument("--python", default=sys.executable, help="interpreter to build the venv from")
    ap.add_argument("--venv", default=str(_REPO_ROOT / ".venv"))
    ap.add_argument("--requirements", default=str(_REPO_ROOT / "requirements.txt"))
    args = ap.parse_args(argv)
    vpy = ensure_venv(
        python_exe=args.python,
        venv_dir=Path(args.venv),
        requirements=Path(args.requirements),
    )
    _log(f"OK: {vpy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
