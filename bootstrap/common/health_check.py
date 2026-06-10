"""Post-install validation: the host clock is sync'd, the systemd unit is
registered, starts, and the bot reaches READY.

Steps (each logged):
  0. clock synchronized?            (chronyc tracking: Leap status Normal +
                                     offset inside tolerance; Linux only —
                                     skipped where chronyc is unavailable)
  1. unit registered?               (systemctl show: LoadState=loaded)
  2. start it                       (systemctl start; refused if the
                                     Conflicts= partner unit is active;
                                     an ALREADY-active unit short-circuits
                                     to a runtime-log freshness check —
                                     its boot READY is long past)
  3. transitions to active          (poll ActiveState, timeout)
  4. bot emits READY                (tail var/<mode>/runtime.log for "READY",
                                     timeout)

Exit 0 = healthy. Intended to run at the end of install.sh; safe to run
standalone for a spot-check.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Max acceptable |local - NTP| offset. The bot's sub-second wake schedule
# documents a +-250ms clock-truth budget (engine.py clock-sync note); a
# chrony-disciplined host sits at ~30-60 MICROseconds, so tripping this
# means sync is genuinely broken, not merely loose.
_CLOCK_OFFSET_TOLERANCE_S = 0.25

# Proof-of-life bound for a unit that is ALREADY active when the check
# runs: the bot logs READY once at boot (nothing new to wait for), so
# health = the runtime log is being actively written. The bot logs many
# times per ~5-min round; 10 min of silence from an active unit is wrong.
_ACTIVE_LOG_FRESH_S = 600.0

# The live and dry systemd units declare mutual Conflicts= (one bot at a
# time): systemctl-starting one SILENTLY STOPS the other. Starting the
# checked service is part of this health check, so on a box where the
# partner unit is running, proceeding would take down a production bot
# (exactly what happened on 2026-06-10: a dry health check stopped
# pancakebot-live for ~77s). Refuse instead.
_CONFLICTING_SERVICE = {
    "pancakebot-live": "pancakebot-dry",
    "pancakebot-dry": "pancakebot-live",
}


def _log(msg: str) -> None:
    print(f"[health_check] {msg}", flush=True)


def _parse_chronyc_tracking(text: str) -> tuple[bool, str]:
    """Decide clock health from ``chronyc tracking`` output.

    Healthy = ``Leap status : Normal`` AND ``System time`` offset within
    ``_CLOCK_OFFSET_TOLERANCE_S``. Pure function for testability.
    """
    leap: str | None = None
    offset_s: float | None = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "leap status":
            leap = value
        elif key == "system time":
            # e.g. "0.000028204 seconds slow of NTP time"
            parts = value.split()
            try:
                offset_s = abs(float(parts[0]))
            except (IndexError, ValueError):
                offset_s = None
    if leap is None or offset_s is None:
        return False, f"unparseable chronyc tracking output: {text[:200]!r}"
    if leap.lower() != "normal":
        return False, f"clock NOT synchronized (Leap status: {leap})"
    if offset_s > _CLOCK_OFFSET_TOLERANCE_S:
        return False, (
            f"clock offset {offset_s:.3f}s exceeds tolerance "
            f"{_CLOCK_OFFSET_TOLERANCE_S}s (Leap status: {leap})"
        )
    return True, f"synchronized, offset {offset_s * 1000:.3f}ms"


def _clock_sync_ok() -> tuple[bool, str]:
    """Step 0: chrony-based clock-sync check. Skips (healthy) where
    chronyc is unavailable (Windows operator desktop, containers without
    chrony) — the bot HOST is where this must hold, and there
    bootstrap/install.sh guarantees chronyd."""
    if shutil.which("chronyc") is None:
        return True, "chronyc not available; clock check skipped"
    try:
        proc = subprocess.run(
            ["chronyc", "tracking"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"chronyc tracking failed to run: {e}"
    if proc.returncode != 0:
        return False, f"chronyc tracking exited {proc.returncode}: {proc.stderr.strip()}"
    return _parse_chronyc_tracking(proc.stdout)


def _run_systemctl(argv: list[str]) -> str:
    """Run ``systemctl <argv>`` and return stdout ('' on any failure).
    Module-level so tests monkeypatch it."""
    try:
        proc = subprocess.run(
            ["systemctl", *argv], capture_output=True, text=True, timeout=15,
        )
        return proc.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _unit_state(service_name: str) -> tuple[str, str]:
    """Return ``(load_state, active_state)`` for the unit.

    LoadState: ``loaded`` when registered, ``not-found`` otherwise.
    ActiveState: ``active`` / ``activating`` / ``inactive`` / ``failed``.
    """
    out = _run_systemctl(
        ["show", service_name, "-p", "LoadState", "-p", "ActiveState"],
    )
    parsed: dict[str, str] = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        parsed[key.strip()] = value.strip()
    return parsed.get("LoadState", "unknown"), parsed.get("ActiveState", "unknown")


def _runtime_log(mode: str) -> Path:
    return _REPO_ROOT / "var" / mode / "runtime.log"


def _tail_has_ready_since(path: Path, since_ts: float) -> bool:
    """True if a READY line appears in the log written after ``since_ts``.
    Uses file mtime as a coarse gate plus a content scan of the tail."""
    if not path.exists() or path.stat().st_mtime < since_ts:
        return False
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    return any(" READY " in ln for ln in lines[-40:])


def run(
    *, mode: str, service_name: str,
    start_timeout_s: float = 30.0, ready_timeout_s: float = 90.0,
) -> bool:
    clock_ok, clock_detail = _clock_sync_ok()
    if not clock_ok:
        _log(f"FAIL: clock sync — {clock_detail}")
        return False
    _log(f"clock sync ok ({clock_detail})")

    load_state, active_state = _unit_state(service_name)
    if load_state != "loaded":
        _log(f"FAIL: unit {service_name!r} is not registered "
             f"(LoadState={load_state})")
        return False
    _log(f"unit {service_name!r} registered (ActiveState={active_state})")

    if active_state == "active":
        # Already running — there is no boot READY to wait for (the bot
        # logs READY once at startup, long scrolled away by now). Health =
        # the runtime log is being actively written.
        log_path = _runtime_log(mode)
        try:
            age_s = time.time() - log_path.stat().st_mtime
        except OSError:
            _log(f"FAIL: unit is active but {log_path} is unreadable/missing")
            return False
        if age_s > _ACTIVE_LOG_FRESH_S:
            _log(f"FAIL: unit is active but {log_path} is stale "
                 f"({age_s:.0f}s > {_ACTIVE_LOG_FRESH_S:.0f}s)")
            return False
        _log(f"already active; runtime log fresh ({age_s:.0f}s old)")
        return True

    started_at = time.time()
    partner = _CONFLICTING_SERVICE.get(service_name)
    if partner is not None and _unit_state(partner)[1] == "active":
        _log(
            f"FAIL: refusing to start {service_name!r} — conflicting unit "
            f"{partner!r} is active (systemd Conflicts= would stop it). "
            f"Run the health check against {partner!r}, or stop it "
            f"explicitly first."
        )
        return False
    _log(f"starting {service_name!r}")
    _run_systemctl(["start", service_name])

    deadline = time.time() + start_timeout_s
    while time.time() < deadline:
        if _unit_state(service_name)[1] == "active":
            _log("unit is active")
            break
        time.sleep(1.0)
    else:
        _log(f"FAIL: {service_name!r} did not reach active within {start_timeout_s}s")
        return False

    log_path = _runtime_log(mode)
    deadline = time.time() + ready_timeout_s
    while time.time() < deadline:
        if _tail_has_ready_since(log_path, started_at):
            _log(f"bot reached READY (per {log_path})")
            return True
        time.sleep(2.0)
    _log(f"FAIL: no READY in {log_path} within {ready_timeout_s}s")
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Post-install health check.")
    ap.add_argument("--mode", choices=["live", "dry"], required=True)
    ap.add_argument("--service-name", required=True)
    args = ap.parse_args(argv)
    ok = run(mode=args.mode, service_name=args.service_name)
    _log("OK: healthy" if ok else "UNHEALTHY")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
