"""Post-install validation: the host clock is sync'd, the service is
registered, starts, and the bot reaches READY — driven through the
cross-platform ``ServicePlatform``.

Steps (each logged):
  0. clock synchronized?            (chronyc tracking: Leap status Normal +
                                     offset inside tolerance; Linux only —
                                     skipped where chronyc is unavailable)
  1. service registered?            (service_status != UNKNOWN)
  2. start it                       (platform.start_service)
  3. transitions to RUNNING         (poll service_status, timeout)
  4. bot emits READY                (tail var/<mode>/runtime.log for "READY",
                                     timeout)

Exit 0 = healthy. Intended to run at the end of install.{sh,ps1}; safe to run
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
    platform=None,
) -> bool:
    from pancakebot.service import ServiceState, get_platform
    p = platform if platform is not None else get_platform()

    clock_ok, clock_detail = _clock_sync_ok()
    if not clock_ok:
        _log(f"FAIL: clock sync — {clock_detail}")
        return False
    _log(f"clock sync ok ({clock_detail})")

    st = p.service_status(service_name)
    if st == ServiceState.UNKNOWN:
        _log(f"FAIL: service {service_name!r} is not registered")
        return False
    _log(f"service {service_name!r} registered (state={st})")

    started_at = time.time()
    if st != ServiceState.RUNNING:
        _log(f"starting {service_name!r}")
        p.start_service(service_name)

    deadline = time.time() + start_timeout_s
    while time.time() < deadline:
        if p.service_status(service_name) == ServiceState.RUNNING:
            _log("service is RUNNING")
            break
        time.sleep(1.0)
    else:
        _log(f"FAIL: {service_name!r} did not reach RUNNING within {start_timeout_s}s")
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
