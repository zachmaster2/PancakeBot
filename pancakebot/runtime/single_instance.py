"""Detect other dry/live bot instances to prevent duplicate-process clashes."""
import os

_PATTERNS = ("run.py --dry", "run.py --live")


def find_duplicate_bots():
    """Return list of {pid, cmdline, started_at} for other python processes
    running `run.py --dry` or `run.py --live`, excluding self. Uses psutil."""
    import psutil
    self_pid = os.getpid()
    results = []
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            if proc.info["pid"] == self_pid:
                continue
            cmdline = proc.info.get("cmdline") or []
            if not cmdline:
                continue
            cmd_str = " ".join(cmdline)
            if any(pat in cmd_str for pat in _PATTERNS):
                results.append({
                    "pid": proc.info["pid"],
                    "cmdline": cmd_str,
                    "started_at": proc.info.get("create_time"),
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return results
