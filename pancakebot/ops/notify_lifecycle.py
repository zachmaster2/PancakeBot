"""systemd lifecycle -> Discord notifier (Phase 3c-2 systemd-direct).

    python -m pancakebot.ops.notify_lifecycle <unit>-<event>
    # e.g. pancakebot-live-started, pancakebot-dry-stopped

Invoked by the ``pancakebot-notify@.service`` oneshot template, which the
bot units trigger via ExecStartPost/ExecStopPost (``--no-block``, so a slow
Discord POST never delays the bot's lifecycle). The template loads ONLY
``/etc/pancakebot/alerts.env`` (webhooks) — least privilege: the wallet key
never enters this process.

State source: ``systemctl show <unit> -p Result,ExecMainStatus,NRestarts``.
systemd RETAINS these on the unit object after the service exits, which is
what makes the separate-unit design work — ``$SERVICE_RESULT`` /
``$EXIT_STATUS`` exist only inside the MAIN unit's ExecStopPost environment
and do NOT propagate through ``systemctl start``. (They are still read
preferentially when present, so direct ExecStopPost invocation also works.)

Event -> kind mapping (reuses the existing notifications.py taxonomy +
SupervisorCore's thresholds verbatim):

  started, NRestarts==0 (fresh/manual start):
      uptime < 10min        -> REBOOTED
      crash.json present    -> RECOVERY_AFTER_CRASH
      else                  -> STARTED
  started, NRestarts>0 (systemd auto-restart after a failure):
      append to var/<mode>/restart_history.jsonl, then two-tier check:
      >=3 restarts/15min    -> SUPPRESSED_FAST_CRASHLOOP (one alert; the
                               CRASHED alert already fired per failure and
                               is 5-min rate-limited; systemd's
                               StartLimitBurst=5/900s is the actual brake)
      >=8 restarts/24h      -> SLOW_CRASHLOOP_WARNING
      else                  -> no alert (CRASHED covered the failure)
  stopped:
      Result=success         -> STOPPED (intentional)
      Result=start-limit-hit -> SUPPRESSED_FAST_CRASHLOOP (terminal: systemd
                                gave up restarting; manual intervention)
      anything else          -> CRASHED (cause + exit status + last journal
                                line in detail; crash.json traceback renders
                                via the existing build_message art path)

Never raises: an alert failure must not mark the notify unit failed in a
way that masks the underlying event (errors go to stderr -> journald).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

from pancakebot.service import notifications

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Thresholds — carried over from SupervisorCore verbatim.
_FAST_RESTART_MAX = 3
_FAST_RESTART_WINDOW_S = 15 * 60.0
_SLOW_RESTART_MAX = 8
_SLOW_RESTART_WINDOW_S = 24 * 3600.0
_REBOOT_DETECT_S = 10 * 60.0

_JOURNAL_TAIL_LINES = 30
_DETAIL_JOURNAL_CHARS = 160

RunCmd = Callable[[list[str]], str]


def _run_cmd(argv: list[str]) -> str:
    """Default command runner: stdout as text; '' on any failure."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return proc.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_instance(instance: str) -> tuple[str, str, str]:
    """``pancakebot-live-started`` -> (unit, mode, event).

    The event is the last dash-segment; the mode the one before it.
    Raises ValueError on anything else (the template should only ever be
    instantiated by the bot units' Exec hooks).
    """
    unit, _, event = instance.rpartition("-")
    if event not in ("started", "stopped") or not unit:
        raise ValueError(f"unrecognized notify instance: {instance!r}")
    mode = unit.rpartition("-")[2]
    if mode not in ("live", "dry", "test"):
        raise ValueError(f"unrecognized mode in instance: {instance!r}")
    return unit, mode, event


def artifacts(mode: str, repo_root: Path = _REPO_ROOT) -> dict[str, Path]:
    """Alert-relevant artifact paths (same layout the supervisor used)."""
    base = repo_root / "var" / mode
    return {
        "last_alert": base / "last_alert.json",
        "crash": base / "crash.json",
        "restart_history": base / "restart_history.jsonl",
        "logs_dir": base / "logs",
    }


def query_unit_state(
    unit: str, *, run_cmd: RunCmd = _run_cmd,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str, int]:
    """Return ``(result, exit_status, n_restarts)`` for the unit.

    Env vars win when present (direct ExecStopPost invocation); otherwise
    ``systemctl show`` — systemd retains Result/ExecMainStatus/NRestarts
    after exit, so the detached notify unit reads them reliably.
    """
    env = env or {}
    result = env.get("SERVICE_RESULT", "").strip()
    exit_status = env.get("EXIT_STATUS", "").strip()

    out = run_cmd([
        "systemctl", "show", unit,
        "-p", "Result", "-p", "ExecMainStatus", "-p", "NRestarts",
    ])
    parsed: dict[str, str] = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        parsed[key.strip()] = value.strip()
    if not result:
        result = parsed.get("Result", "unknown")
    if not exit_status:
        exit_status = parsed.get("ExecMainStatus", "?")
    try:
        n_restarts = int(parsed.get("NRestarts", "0"))
    except ValueError:
        n_restarts = 0
    return result, exit_status, n_restarts


def journal_tail(unit: str, *, run_cmd: RunCmd = _run_cmd) -> str:
    return run_cmd([
        "journalctl", "-u", unit, "-n", str(_JOURNAL_TAIL_LINES),
        "--no-pager", "-o", "cat",
    ])


def _last_journal_line(tail: str) -> str:
    for line in reversed(tail.splitlines()):
        if line.strip():
            return line.strip()[:_DETAIL_JOURNAL_CHARS]
    return ""


# -- restart history (same JSONL shape the supervisor wrote) ----------------

def read_history(path: Path) -> list[dict]:
    entries: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return entries
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("ts"), (int, float)):
            entries.append(obj)
    return entries


def write_history(path: Path, entries: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.name + ".tmp")
        body = "\n".join(
            json.dumps(e, sort_keys=True, separators=(",", ":")) for e in entries
        )
        tmp.write_text(body + ("\n" if body else ""), encoding="utf-8")
        tmp.replace(path)
    except Exception:  # noqa: BLE001 — history is best-effort telemetry
        print(f"notify_lifecycle: restart_history write failed: {path}",
              file=sys.stderr)


def _count_within(entries: list[dict], now: float, window_s: float) -> int:
    return sum(1 for e in entries if (now - float(e["ts"])) <= window_s)


# -- the decision table ------------------------------------------------------

def decide(
    *, event: str, result: str, exit_status: str, n_restarts: int,
    crash_exists: bool, uptime_s: float, history: list[dict], now: float,
    journal_line: str = "",
) -> tuple[list[tuple[str, dict, str | None]], list[dict] | None]:
    """Pure mapping: (event + unit state) -> alerts to send + new history.

    Returns ``(alerts, new_history)`` where alerts are ``(kind, fields,
    detail)`` tuples and ``new_history`` is None when the history file
    should not be touched.
    """
    if event == "started":
        if n_restarts <= 0:
            # Fresh/manual start — the supervisor's first-run classifier.
            if uptime_s < _REBOOT_DETECT_S:
                return [("REBOOTED", {}, None)], None
            if crash_exists:
                return [("RECOVERY_AFTER_CRASH", {}, None)], None
            return [("STARTED", {}, None)], None
        # systemd auto-restart after a failure: track + two-tier check.
        new_history = [
            e for e in history if (now - float(e["ts"])) <= _SLOW_RESTART_WINDOW_S
        ]
        new_history.append({"ts": now})
        alerts: list[tuple[str, dict, str | None]] = []
        fast = _count_within(new_history, now, _FAST_RESTART_WINDOW_S)
        slow = len(new_history)
        if fast >= _FAST_RESTART_MAX:
            alerts.append((
                "SUPPRESSED_FAST_CRASHLOOP", {},
                f"{fast} systemd auto-restarts in "
                f"{_FAST_RESTART_WINDOW_S / 60:.0f}min >= {_FAST_RESTART_MAX} "
                f"(StartLimitBurst halts restarts at 5/900s)",
            ))
        elif slow >= _SLOW_RESTART_MAX:
            alerts.append((
                "SLOW_CRASHLOOP_WARNING", {},
                f"{slow} restarts in {_SLOW_RESTART_WINDOW_S / 3600:.0f}h "
                f">= {_SLOW_RESTART_MAX}",
            ))
        return alerts, new_history

    # event == "stopped"
    if result == "success":
        return [("STOPPED", {"intentional": True}, None)], None
    if result == "start-limit-hit":
        return [(
            "SUPPRESSED_FAST_CRASHLOOP", {},
            "systemd start-limit-hit: restart budget exhausted "
            "(5 failures/900s) — NOT restarting; manual intervention required",
        )], None
    detail = f"{result} exit_status={exit_status} n_restarts={n_restarts}"
    if journal_line:
        detail += f" | journal: {journal_line}"
    return [("CRASHED", {}, detail)], None


def _system_uptime_s() -> float:
    try:
        import psutil
        return time.time() - psutil.boot_time()
    except Exception:  # noqa: BLE001 — conservative: assume warm system
        return float("inf")


def main(
    argv: list[str], *,
    run_cmd: RunCmd = _run_cmd,
    env: Mapping[str, str] | None = None,
    now: float | None = None,
    repo_root: Path = _REPO_ROOT,
) -> int:
    """CLI entry. Never raises; exit code 0 unless arguments are unusable."""
    if len(argv) != 1:
        print("usage: python -m pancakebot.ops.notify_lifecycle <unit>-<event>",
              file=sys.stderr)
        return 2
    try:
        unit, mode, event = parse_instance(argv[0])
    except ValueError as e:
        print(f"notify_lifecycle: {e}", file=sys.stderr)
        return 2

    try:
        now_ts = time.time() if now is None else now
        art = artifacts(mode, repo_root)
        result, exit_status, n_restarts = query_unit_state(
            unit, run_cmd=run_cmd, env=env,
        )
        tail = journal_tail(unit, run_cmd=run_cmd) if event == "stopped" else ""
        history = read_history(art["restart_history"])
        alerts, new_history = decide(
            event=event, result=result, exit_status=exit_status,
            n_restarts=n_restarts, crash_exists=art["crash"].exists(),
            uptime_s=_system_uptime_s(), history=history, now=now_ts,
            journal_line=_last_journal_line(tail),
        )
        if new_history is not None:
            write_history(art["restart_history"], new_history)
        for kind, fields, detail in alerts:
            outcome = notifications.notify(
                mode=mode, kind=kind, fields=fields, art=art, detail=detail,
            )
            print(f"notify_lifecycle: {unit} {event} -> {kind} ({outcome})")
        if not alerts:
            print(f"notify_lifecycle: {unit} {event} -> no alert "
                  f"(result={result} n_restarts={n_restarts})")
        return 0
    except Exception as e:  # noqa: BLE001 — alerting must never cascade
        print(f"notify_lifecycle: unexpected error: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
