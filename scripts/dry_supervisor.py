"""
Dry-mode supervisor: checks whether the dry-mode process is alive and appends
a one-line status to var/dry/logs/supervisor.log.

Run manually, via cron, or via Windows Task Scheduler every 2-5 minutes.
Does NOT restart — observability only.

Usage:
    python scripts/dry_supervisor.py
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERVISOR_LOG = os.path.join(REPO_ROOT, "var", "dry", "logs", "supervisor.log")

# Signature that identifies the dry-mode process.
DRY_SIGNATURE = "run.py --dry"


def _find_dry_process() -> dict | None:
    """Return {pid, cmdline} if a dry-mode process is running, else None."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | "
             "Where-Object { $_.CommandLine -like '*run.py --dry*' } | "
             "Select-Object ProcessId, CommandLine | "
             "ConvertTo-Json"],
            capture_output=True, text=True, timeout=10,
        )
        out = result.stdout.strip()
        if not out or out == "null":
            return None
        import json
        data = json.loads(out)
        # May be a single object or a list.
        if isinstance(data, list):
            data = data[0]
        return {"pid": data.get("ProcessId"), "cmdline": data.get("CommandLine", "")}
    except Exception as e:
        return {"pid": None, "cmdline": f"ERROR checking: {e}"}


def _log(msg: str) -> None:
    os.makedirs(os.path.dirname(SUPERVISOR_LOG), exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts}  {msg}"
    print(line)
    with open(SUPERVISOR_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    proc = _find_dry_process()

    if proc is None:
        _log("STATUS=DOWN  pid=none  DRY MODE NOT RUNNING — manual restart required")
        sys.exit(1)
    elif proc.get("pid") is None:
        _log(f"STATUS=ERROR  check_error={proc.get('cmdline')}")
        sys.exit(2)
    else:
        _log(f"STATUS=UP  pid={proc['pid']}  dry mode running")
        sys.exit(0)


if __name__ == "__main__":
    main()
