"""Post-install validation: the service is registered, starts, and the bot
reaches READY — driven through the cross-platform ``ServicePlatform``.

Steps (each logged):
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
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _log(msg: str) -> None:
    print(f"[health_check] {msg}", flush=True)


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
