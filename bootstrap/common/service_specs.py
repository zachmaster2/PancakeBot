"""Shared ServiceSpec builders for the live + dry services.

The declarative spec is rendered into a systemd unit by
``LinuxServicePlatform`` (Linux-only since the Phase 3 Windows-host
retirement, 2026-06-10).
"""
from __future__ import annotations

from pathlib import Path

from pancakebot.service import ServiceSpec

# OUTER-supervisor restart policy (systemd StartLimit). Relaxed from 3/24h
# so intentional restarts (deploys, admin stop+start) don't exhaust it: the
# supervisor calls ``systemctl reset-failed`` after a start that followed an
# intentional stop (marker handshake). The INNER crashloop limiter
# (SupervisorCore _FAST_RESTART_MAX=3 / _SLOW_RESTART_MAX=8) remains the
# real crashloop guard.
_RESTART_MAX = 5
_RESTART_RESET_WINDOW_S = 3600
_RESTART_DELAY_S = 60


def _names(mode: str) -> tuple[str, str]:
    """(service_name, other_mode_service_name) — systemd unit names."""
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
        # Run the SUPERVISOR (spawns + monitors run.py, emits the full
        # Discord alert taxonomy). systemd is the OUTER supervisor.
        args=("-m", "pancakebot.service.supervise", "--mode", mode),
        working_dir=str(repo_root),
        # Discord webhooks come from the systemd EnvironmentFile
        # (/etc/pancakebot/pancakebot.env); not embedded in the spec.
        env=None,
        requires_network=True,
        conflicts_with=other,
        restart_max_attempts=_RESTART_MAX,
        restart_reset_window_s=_RESTART_RESET_WINDOW_S,
        restart_delay_s=_RESTART_DELAY_S,
    )
