"""Shared ServiceSpec builders for the live + dry services.

The same declarative spec is rendered differently per OS by the adapter:
Windows -> sc.exe config / pythonservice host; Linux -> systemd unit. Service
NAMES differ by platform convention (SCM ``PancakeBotLive`` vs systemd
``pancakebot-live``); everything else is shared.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pancakebot.service import ServiceSpec

# Crashloop policy mirrors common.py's _FAST_RESTART_* and install_services.ps1.
_RESTART_MAX = 3
_RESTART_RESET_WINDOW_S = 86400
_RESTART_DELAY_S = 60


def _names(mode: str) -> tuple[str, str]:
    """(service_name, other_mode_service_name) per OS convention."""
    if sys.platform == "win32":
        live, dry = "PancakeBotLive", "PancakeBotDry"
    else:
        live, dry = "pancakebot-live", "pancakebot-dry"
    return (live, dry) if mode == "live" else (dry, live)


def build_spec(*, mode: str, repo_root: Path, venv_python: Path) -> ServiceSpec:
    if mode not in ("live", "dry"):
        raise ValueError(f"mode must be live|dry, got {mode!r}")
    name, other = _names(mode)
    return ServiceSpec(
        name=name,
        description=f"PancakeBot {'Live Trading' if mode == 'live' else 'Dry Paper Trading'}",
        exe_path=str(venv_python),
        # Run the cross-platform SUPERVISOR (spawns + monitors run.py, emits the
        # full Discord alert taxonomy). systemd is the OUTER supervisor.
        args=("-m", "pancakebot.service.supervise", "--mode", mode),
        working_dir=str(repo_root),
        # Discord webhooks come from machine env (Windows) / EnvironmentFile
        # (Linux); not embedded in the spec.
        env=None,
        requires_network=True,
        conflicts_with=other,
        restart_max_attempts=_RESTART_MAX,
        restart_reset_window_s=_RESTART_RESET_WINDOW_S,
        restart_delay_s=_RESTART_DELAY_S,
    )
