"""Cross-platform supervisor entry point (primarily Linux / systemd).

    python -m pancakebot.service.supervise --mode live|dry

Constructs the OS ``ServicePlatform`` adapter + ``SupervisorCore`` and runs the
supervision loop, spawning + monitoring the bot child (``run.py``) and emitting
the full Discord alert taxonomy — identical to the Windows service host.

On Linux the systemd unit's ExecStart points here: systemd is the OUTER
supervisor that restarts THIS process (``Restart=on-failure``); SupervisorCore
handles the INNER bot restarts, crashloop limiting, and alerts. SIGTERM/SIGINT
request a graceful stop.
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from pancakebot.service import get_platform
from pancakebot.service.supervisor_core import SupervisorCore

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _service_names(mode: str) -> tuple[str, str]:
    """(this_service, other_service) per OS convention."""
    if sys.platform == "win32":
        live, dry = "PancakeBotLive", "PancakeBotDry"
    else:
        live, dry = "pancakebot-live", "pancakebot-dry"
    return (live, dry) if mode == "live" else (dry, live)


def _log(level: str, msg: str) -> None:
    # journald captures stdout/stderr under systemd.
    stream = sys.stderr if level in ("ERROR", "WARN") else sys.stdout
    print(f"[supervise:{level}] {msg}", file=stream, flush=True)


def build_core(mode: str, *, platform=None) -> SupervisorCore:
    name, other = _service_names(mode)
    return SupervisorCore(
        mode=mode,
        other_service=other,
        platform=platform if platform is not None else get_platform(),
        repo_root=_REPO_ROOT,
        # supervise.py runs under the venv interpreter directly (NOT a service
        # host), so sys.executable IS the venv python used to spawn the child.
        venv_python=Path(sys.executable),
        service_name=name,
        log=_log,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PancakeBot cross-platform supervisor.")
    ap.add_argument("--mode", choices=["live", "dry"], required=True)
    args = ap.parse_args(argv)

    core = build_core(args.mode)

    def _on_term(signum, _frame):
        _log("INFO", f"signal {signum} -> graceful stop")
        core.request_stop()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    core.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
