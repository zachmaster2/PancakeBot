"""PancakeBot supervisor -- classify bot health, optionally alert + auto-restart.

Phase 2b: classify + log-only (default behaviour).
Phase 2c: Discord alerting, enabled with ``--alert``.
Phase 2d: auto-restart with fast/slow crashloop limits, enabled with ``--restart``.

One-shot invocation. Reads the Phase 2a health artifacts for one mode
(``var/<mode>/heartbeat.json``, ``bot.pid``, ``crash.json``), classifies state,
appends a single line to ``var/<mode>/supervisor.log``, and exits. Idempotent;
safe to invoke on a schedule (e.g. ``schtasks`` every 3 min).

Classification order (first match wins):
  UP            - fresh heartbeat (mtime < ``--stale-threshold``) + PID alive
  STARTING      - PID file alive, no/stale heartbeat, PID file age < ``--startup-grace``
  STALE         - PID alive but heartbeat stale past threshold (past grace)
  CRASHED       - ``crash.json`` present (regardless of process state)
  UNINSTRUMENTED- bot process alive (psutil match on ``run.py --<mode>``)
                  but no heartbeat AND no PID file (legacy pre-Phase-2a bot)
  DOWN          - none of the above

Exit codes:
  UP=0, STARTING=1, STALE=2, CRASHED=3, UNINSTRUMENTED=4, DOWN=5,
  SUPPRESSED_FAST_CRASHLOOP=6, supervisor error=99.

Actions (opt-in):
  --alert    POST a Discord message on STALE/CRASHED/UNINSTRUMENTED/DOWN.
             Webhook URL comes from env vars:
               DRY_DISCORD_ALERT_WEBHOOK_URL  (for --mode dry)
               LIVE_DISCORD_ALERT_WEBHOOK_URL (for --mode live)
             If the relevant env var is missing the alert is silently
             suppressed (one DISCORD_DISABLED note in supervisor.log) --
             the supervisor still classifies and logs as usual. A Discord
             send failure (HTTP error, timeout, bad URL) never crashes
             the supervisor; it logs DISCORD_SEND_FAILED and moves on.
             Rate limit: one alert per (mode, classification) per 5 min
             (tracked in ``var/<mode>/last_alert.json``).

  --restart  On STALE/CRASHED/DOWN, spawn a fresh ``python -u run.py --<mode>``
             detached from this supervisor (Windows: CREATE_NEW_PROCESS_GROUP).
             Never restarts on UP, STARTING, UNINSTRUMENTED.
             Each restart appends to ``var/<mode>/restart_history.jsonl``.
             Two-tier crashloop limiter:
               fast tier: ``--max-fast-restarts`` in ``--fast-window-min``
                          -> SUPPRESSED_FAST_CRASHLOOP, exit 6, alert if --alert
               slow tier: ``--max-slow-restarts`` in ``--slow-window-h``
                          -> restart proceeds BUT alert is escalated to
                          SLOW_CRASHLOOP_WARNING; exit code unchanged.
             restart_history.jsonl is pruned of entries older than
             --slow-window-h on every write.

**Real-money production: use ``--alert --restart`` in the schtasks invocation
AND set the mode's Discord webhook env var.** Without --restart the supervisor
is a detector only; without --alert the detection stays on disk until you
tail the log.

Usage:
    python scripts/supervisor.py --mode dry
    python scripts/supervisor.py --mode dry --alert
    python scripts/supervisor.py --mode dry --restart --alert
    python scripts/supervisor.py --mode live --restart --alert \\
        --max-fast-restarts 3 --fast-window-min 15 \\
        --max-slow-restarts 8 --slow-window-h 24
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure the repo root is importable when this script runs from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot import paths  # noqa: E402

# requests is in requirements.txt; always available. Imported lazily inside
# _send_discord to keep the import-free path clean in Phase 2b usage.


# -- Exit codes (per spec) ---------------------------------------------------

EXIT_UP = 0
EXIT_STARTING = 1
EXIT_STALE = 2
EXIT_CRASHED = 3
EXIT_UNINSTRUMENTED = 4
EXIT_DOWN = 5
EXIT_SUPPRESSED_FAST_CRASHLOOP = 6
EXIT_ERROR = 99

_STATUS_TO_EXIT = {
    "UP": EXIT_UP,
    "STARTING": EXIT_STARTING,
    "STALE": EXIT_STALE,
    "CRASHED": EXIT_CRASHED,
    "UNINSTRUMENTED": EXIT_UNINSTRUMENTED,
    "DOWN": EXIT_DOWN,
}

# Classifications that trigger the optional alert/restart actions.
_ALERT_STATES: tuple[str, ...] = ("STALE", "CRASHED", "UNINSTRUMENTED", "DOWN")
_RESTART_STATES: tuple[str, ...] = ("STALE", "CRASHED", "DOWN")

# Per-(mode, classification) alert cooldown. 5 min = 300 s.
_ALERT_COOLDOWN_S: float = 300.0


# -- Artifact resolution -----------------------------------------------------

def _artifacts_for_mode(mode: str) -> dict[str, Path]:
    """Return the full set of supervisor-visible paths for *mode*.

    All paths are resolved against ``_REPO_ROOT`` so the supervisor works
    correctly regardless of the caller's CWD (Windows Task Scheduler, an
    interactive shell in a different directory, etc.). The ``pancakebot.paths``
    constants are relative strings by convention -- resolving here keeps the
    supervisor independent of whatever chose to run it.
    """
    if mode == "dry":
        return {
            "heartbeat": _REPO_ROOT / paths.DRY_HEARTBEAT_PATH,
            "pid": _REPO_ROOT / paths.DRY_PID_PATH,
            "crash": _REPO_ROOT / paths.DRY_CRASH_PATH,
            "supervisor_log": _REPO_ROOT / "var/dry/supervisor.log",
            "trades": _REPO_ROOT / paths.DRY_TRADES_PATH,
            "last_alert": _REPO_ROOT / "var/dry/last_alert.json",
            "restart_history": _REPO_ROOT / "var/dry/restart_history.jsonl",
            "logs_dir": _REPO_ROOT / "var/dry/logs",
        }
    if mode == "live":
        return {
            "heartbeat": _REPO_ROOT / paths.LIVE_HEARTBEAT_PATH,
            "pid": _REPO_ROOT / paths.LIVE_PID_PATH,
            "crash": _REPO_ROOT / paths.LIVE_CRASH_PATH,
            "supervisor_log": _REPO_ROOT / "var/live/supervisor.log",
            "trades": _REPO_ROOT / paths.LIVE_TRADES_PATH,
            "last_alert": _REPO_ROOT / "var/live/last_alert.json",
            "restart_history": _REPO_ROOT / "var/live/restart_history.jsonl",
            "logs_dir": _REPO_ROOT / "var/live/logs",
        }
    raise ValueError(f"unknown_mode: {mode!r}")


def _env_var_for_mode(mode: str) -> str:
    return "DRY_DISCORD_ALERT_WEBHOOK_URL" if mode == "dry" else "LIVE_DISCORD_ALERT_WEBHOOK_URL"


# -- Safe reads (BLOCKER #2: atomic read, never hold handle) -----------------

def _safe_read_json(path: Path) -> dict | None:
    """Atomic open-read-close. Never holds a file handle. None on any error."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (PermissionError, OSError):
        return None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _safe_read_pid_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (PermissionError, OSError):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _safe_stat_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _safe_count_trades(path: Path) -> int | None:
    """Count non-header rows in trades.csv. Best-effort; None on any error."""
    try:
        with path.open(encoding="utf-8") as f:
            count = 0
            for i, _ in enumerate(f):
                count = i
            return count if count >= 0 else 0
    except FileNotFoundError:
        return None
    except (PermissionError, OSError):
        return None


# -- Process liveness -------------------------------------------------------

def _pid_is_our_bot(pid: int, mode: str) -> bool:
    """True iff PID is alive AND its cmdline looks like a ``run.py --<mode>`` process.

    We can't use plain ``psutil.pid_exists`` because Windows aggressively reuses
    PIDs -- a stale bot.pid could point at a PID now owned by svchost.exe or
    any other unrelated process. The cmdline check ensures we only report a
    PID as "our bot alive" when it actually is one.
    """
    try:
        import psutil
        if not psutil.pid_exists(int(pid)):
            return False
        proc = psutil.Process(int(pid))
        cmdline = " ".join(proc.cmdline() or [])
    except Exception:
        return False
    return f"run.py --{mode}" in cmdline


def _find_legacy_bot_pid(mode: str) -> int | None:
    """Scan process list for ``run.py --<mode>``. Excludes self.

    Used only for UNINSTRUMENTED detection -- finds bots that predate the
    Phase 2a heartbeat/PID-file instrumentation.
    """
    needle = f"run.py --{mode}"
    self_pid = os.getpid()
    try:
        import psutil
    except Exception:
        return None
    try:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                if proc.info["pid"] == self_pid:
                    continue
                cmdline = proc.info.get("cmdline") or []
                if not cmdline:
                    continue
                cmd = " ".join(cmdline)
                if needle in cmd:
                    return int(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        return None
    return None


# -- Classification ---------------------------------------------------------

def _classify(
    mode: str,
    *,
    stale_threshold_s: float,
    startup_grace_s: float,
) -> tuple[str, dict[str, Any]]:
    """Return (status, fields) for the given mode.

    Fields are all optional — callers format with "-" when absent.
    """
    art = _artifacts_for_mode(mode)
    now = time.time()
    fields: dict[str, Any] = {}

    hb = _safe_read_json(art["heartbeat"])
    hb_mtime = _safe_stat_mtime(art["heartbeat"])
    hb_age: float | None = (now - hb_mtime) if hb_mtime is not None else None

    pid_from_file = _safe_read_pid_file(art["pid"])
    pid_file_mtime = _safe_stat_mtime(art["pid"])
    pid_file_age: float | None = (now - pid_file_mtime) if pid_file_mtime is not None else None
    # A pid file with a dead PID is effectively absent -- stale leftovers from
    # old infrastructure or from a bot that crashed without atexit running
    # should not mask the UNINSTRUMENTED signal or wrongly imply STARTING/STALE.
    pid_file_is_live = pid_from_file is not None and _pid_is_our_bot(pid_from_file, mode)

    # Select the PID to check: prefer heartbeat.pid when fresh, else pid file.
    heartbeat_pid: int | None = None
    if hb is not None:
        raw = hb.get("pid")
        if isinstance(raw, int):
            heartbeat_pid = raw

    # Populate diagnostic fields from heartbeat (best-effort).
    if hb is not None:
        if heartbeat_pid is not None:
            fields["pid"] = heartbeat_pid
        br = hb.get("bankroll_bnb")
        if isinstance(br, (int, float)):
            fields["bankroll"] = f"{float(br):.4f}"
        ic = hb.get("iteration_count")
        if isinstance(ic, int):
            fields["iterations"] = ic
        le = hb.get("last_epoch")
        if isinstance(le, int):
            fields["last_epoch"] = le
    # Only surface pid_from_file when it's actually live, otherwise it's
    # misleading noise in the log line.
    if heartbeat_pid is None and pid_file_is_live:
        fields["pid"] = pid_from_file

    if hb_age is not None:
        fields["hb_age"] = f"{hb_age:.1f}s"

    # Bets count (best-effort, read trades.csv).
    bets = _safe_count_trades(art["trades"])
    if bets is not None:
        fields["bets"] = bets

    # -- State precedence (first match wins) --

    # 1. UP: fresh heartbeat + PID alive.
    if (
        hb is not None
        and hb_age is not None
        and hb_age <= stale_threshold_s
        and heartbeat_pid is not None
        and _pid_is_our_bot(heartbeat_pid, mode)
    ):
        return "UP", fields

    # 2. STARTING: pid file alive + fresh, no/stale heartbeat.
    if (
        pid_file_is_live
        and (hb is None or (hb_age is not None and hb_age > stale_threshold_s))
        and pid_file_age is not None
        and pid_file_age <= startup_grace_s
    ):
        fields["since_pid_ts"] = f"{pid_file_age:.0f}s"
        return "STARTING", fields

    # 3. STALE: PID alive but heartbeat stale past threshold + past grace.
    #    Prefer heartbeat.pid for the liveness probe; fall back to pid file.
    stale_pid_alive = (
        (heartbeat_pid is not None and _pid_is_our_bot(heartbeat_pid, mode))
        or pid_file_is_live
    )
    if (
        stale_pid_alive
        and hb is not None
        and hb_age is not None
        and hb_age > stale_threshold_s
        and (pid_file_age is None or pid_file_age > startup_grace_s)
    ):
        return "STALE", fields

    # 4. CRASHED: crash.json present (regardless of process state).
    crash = _safe_read_json(art["crash"])
    crash_mtime = _safe_stat_mtime(art["crash"])
    if crash is not None and crash_mtime is not None:
        crash_age = now - crash_mtime
        fields["crash_age"] = f"{crash_age:.1f}s"
        exc_type = crash.get("exc_type")
        if isinstance(exc_type, str):
            fields["exc"] = exc_type
        last_epoch = crash.get("last_epoch")
        if isinstance(last_epoch, int):
            fields["last_epoch"] = last_epoch
        return "CRASHED", fields

    # 5. UNINSTRUMENTED: bot process running but no heartbeat AND no valid
    #    pid file (a stale pid file with a dead PID is treated as absent).
    if hb is None and not pid_file_is_live:
        legacy_pid = _find_legacy_bot_pid(mode)
        if legacy_pid is not None:
            fields.clear()
            fields["pid"] = legacy_pid
            fields["note"] = "legacy_no_heartbeat"
            return "UNINSTRUMENTED", fields

    # 6. DOWN -- no useful context to report.
    fields.clear()
    return "DOWN", fields


# -- Logging ---------------------------------------------------------------

def _iso_utc_now() -> str:
    """2026-04-22T19:55:00Z-style ISO-8601 UTC timestamp."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_fields(fields: dict[str, Any]) -> str:
    # Deterministic field order -- readability + easy grep/parse. The
    # action/alert/new_pid/note-on-suppression keys land last so the common
    # UP-line stays short.
    order = (
        "pid", "hb_age", "bankroll", "bets", "iterations",
        "last_epoch", "since_pid_ts", "crash_age", "exc",
        "action", "new_pid", "alert", "note",
    )
    parts = []
    for k in order:
        if k in fields:
            parts.append(f"{k}={fields[k]}")
    # Any keys we didn't enumerate go last, sorted for stability.
    extras = sorted(k for k in fields if k not in order)
    for k in extras:
        parts.append(f"{k}={fields[k]}")
    return " ".join(parts)


def _write_supervisor_line(log_path: Path, mode: str, status: str, fields: dict[str, Any]) -> None:
    """Append one line. Creates parent dir if missing. Never raises."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = _iso_utc_now()
        body = _format_fields(fields)
        line = f"{ts} STATUS={status} mode={mode}"
        if body:
            line = f"{line} {body}"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        # Also mirror to stdout for interactive / schtasks visibility.
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception:
        # Last-ditch: if logging itself fails, at least stderr gets something.
        sys.stderr.write(f"supervisor_log_write_failed: {log_path}\n")


# -- Discord alerting (Phase 2c) -------------------------------------------

def _clip_text(text: str, max_lines: int, max_chars: int) -> str:
    """Clip *text* to the first ``max_lines`` lines, then to ``max_chars``."""
    lines = text.splitlines()
    result = "\n".join(lines[:max_lines])
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... [truncated]"
    return result


def _tail_latest_err_log(logs_dir: Path, max_lines: int = 20) -> str | None:
    """Return the last *max_lines* lines of the most-recent *_err.log file, or None."""
    if not logs_dir.exists():
        return None
    try:
        candidates = [p for p in logs_dir.glob("*_err.log") if p.is_file()]
    except OSError:
        return None
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        content = latest.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = content.splitlines()
    return "\n".join(lines[-max_lines:]) if lines else None


def _build_discord_message(
    *,
    mode: str,
    status: str,
    fields: dict[str, Any],
    art: dict[str, Path],
    escalation: str | None = None,
) -> str:
    """Build the Discord message body for the given classification."""
    hostname = socket.gethostname()
    ts = _iso_utc_now()

    header_emoji = {
        "STALE": ":warning:",
        "CRASHED": ":rotating_light:",
        "UNINSTRUMENTED": ":grey_question:",
        "DOWN": ":skull:",
    }.get(status, ":information_source:")

    header = f"{header_emoji} **{status}** `PancakeBot-{mode}` on `{hostname}` at `{ts}`"
    if escalation:
        header = f":fire: **{escalation}** -- {header}"
    lines: list[str] = [header]

    # Common context.
    for k in ("pid", "bankroll", "iterations", "last_epoch"):
        if k in fields:
            lines.append(f"{k}: `{fields[k]}`")

    if status == "CRASHED":
        crash = _safe_read_json(art["crash"])
        if crash is not None:
            exc_type = crash.get("exc_type", "?")
            exc_repr = crash.get("exc_repr", "?")
            lines.append(f"exc: `{exc_type}`")
            lines.append(f"repr: `{exc_repr}`")
            # last_epoch already rendered by the common-context loop above
            # when present in ``fields``; don't duplicate.
            tb_raw = str(crash.get("traceback_str", ""))
            tb = _clip_text(tb_raw, max_lines=20, max_chars=1500)
            if tb:
                lines.append("```\n" + tb + "\n```")
    elif status == "STALE":
        hb_age = fields.get("hb_age", "?")
        lines.append(f"heartbeat age: `{hb_age}`")
        tail = _tail_latest_err_log(art["logs_dir"], max_lines=20)
        if tail:
            clipped = _clip_text(tail, max_lines=20, max_chars=1500)
            lines.append("```\n" + clipped + "\n```")
    elif status == "UNINSTRUMENTED":
        lines.append("note: legacy bot pre-Phase-2a still running")

    return "\n".join(lines)


def _rate_limit_ok(last_alert_path: Path, status: str, now: float) -> bool:
    """True if we haven't alerted for (mode,status) within ``_ALERT_COOLDOWN_S``.

    Updates last_alert.json on every attempt (success OR failure) so a flapping
    Discord endpoint doesn't cause the supervisor to hammer it. If Discord is
    down, the next opportunity is one cooldown later.
    """
    data = _safe_read_json(last_alert_path) or {}
    last_ts = 0.0
    raw = data.get(status)
    if isinstance(raw, (int, float)):
        last_ts = float(raw)
    if (now - last_ts) < _ALERT_COOLDOWN_S:
        return False
    # Pre-write the timestamp so concurrent invocations don't both alert.
    data[status] = now
    try:
        last_alert_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = last_alert_path.parent / (last_alert_path.name + ".tmp")
        tmp.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        tmp.replace(last_alert_path)
    except Exception:
        # Best-effort; if we can't write state, still allow the send.
        pass
    return True


def _send_discord(webhook_url: str, mode: str, message: str) -> tuple[bool, str]:
    """POST a Discord message. Returns (ok, detail). Never raises."""
    try:
        import requests
    except Exception as e:
        return False, f"requests_import_failed:{e}"
    payload = {"content": message, "username": f"PancakeBot-{mode}"}
    try:
        r = requests.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        return False, f"post_exception:{type(e).__name__}:{e}"
    if 200 <= r.status_code < 300:
        return True, f"http_{r.status_code}"
    return False, f"http_{r.status_code}:{(r.text or '')[:200]}"


def _maybe_send_discord(
    *,
    mode: str,
    status: str,
    fields: dict[str, Any],
    art: dict[str, Path],
    escalation: str | None,
) -> str:
    """Dispatch an alert for *status*. Returns the outcome tag for the log line.

    Outcomes: SENT, DISABLED (env var unset), RATE_LIMITED, SEND_FAILED, NOT_APPLICABLE.
    """
    if status not in _ALERT_STATES and escalation is None:
        return "NOT_APPLICABLE"

    env_var = _env_var_for_mode(mode)
    webhook = os.environ.get(env_var, "").strip()
    if not webhook:
        return "DISABLED"

    # Rate limiting -- use the escalation tag as the "status" key when present
    # so SLOW_CRASHLOOP_WARNING gets its own cooldown bucket separate from the
    # underlying STALE/CRASHED classification.
    key = escalation or status
    if not _rate_limit_ok(art["last_alert"], key, time.time()):
        return "RATE_LIMITED"

    msg = _build_discord_message(
        mode=mode, status=status, fields=fields, art=art, escalation=escalation,
    )
    ok, detail = _send_discord(webhook, mode, msg)
    if ok:
        return "SENT"
    # Log the failure detail via stderr so the operator can see what happened
    # without needing to tail the log file.
    sys.stderr.write(f"discord_send_failed mode={mode} status={status} detail={detail}\n")
    return "SEND_FAILED"


# -- Auto-restart + crashloop limiter (Phase 2d) ---------------------------

def _read_restart_history(path: Path) -> list[dict]:
    entries: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return entries
    except (PermissionError, OSError):
        return entries
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _prune_history(entries: list[dict], now: float, slow_window_s: float) -> list[dict]:
    """Drop entries with ``ts_wall < now - slow_window_s``. Malformed entries dropped."""
    kept = []
    for e in entries:
        ts_wall = e.get("ts_wall")
        if isinstance(ts_wall, (int, float)) and (now - float(ts_wall)) <= slow_window_s:
            kept.append(e)
    return kept


def _count_within(entries: list[dict], now: float, window_s: float) -> int:
    n = 0
    for e in entries:
        ts_wall = e.get("ts_wall")
        if isinstance(ts_wall, (int, float)) and (now - float(ts_wall)) <= window_s:
            n += 1
    return n


def _write_restart_history(path: Path, entries: list[dict]) -> None:
    """Atomic write of JSONL. Parent dir created on demand. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.name + ".tmp")
        body = "\n".join(
            json.dumps(e, sort_keys=True, separators=(",", ":")) for e in entries
        )
        if body:
            body += "\n"
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        sys.stderr.write(f"restart_history_write_failed: {path}\n")


def _spawn_bot(mode: str, logs_dir: Path) -> tuple[int, Path]:
    """Launch a fresh ``python -u run.py --<mode>`` detached from this supervisor.

    Returns (pid, out_log_path). stdout/stderr redirected to timestamped files
    under *logs_dir*. Windows uses CREATE_NEW_PROCESS_GROUP so the child
    survives supervisor exit.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_log = logs_dir / f"{mode}-auto-{ts}.log"
    err_log = logs_dir / f"{mode}-auto-{ts}_err.log"

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )

    out_f = open(out_log, "w", encoding="utf-8")
    err_f = open(err_log, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", "run.py", f"--{mode}"],
            cwd=str(_REPO_ROOT),
            stdout=out_f,
            stderr=err_f,
            creationflags=creationflags,
        )
    finally:
        # The child owns the descriptors now; close ours so supervisor exit
        # doesn't keep them open.
        out_f.close()
        err_f.close()
    return int(proc.pid), out_log


def _do_restart(
    *,
    mode: str,
    status: str,
    art: dict[str, Path],
    max_fast: int,
    fast_window_s: float,
    max_slow: int,
    slow_window_s: float,
) -> dict[str, Any]:
    """Apply crashloop limits and (when allowed) spawn a fresh bot.

    Returns a dict describing the outcome; callers use ``action`` to populate
    the supervisor.log line and to decide exit code / alert escalation.
    Possible ``action`` values:
      RESTARTED, SUPPRESSED_FAST_CRASHLOOP, SLOW_CRASHLOOP_WARNING
    SLOW_CRASHLOOP_WARNING also carries new_pid -- the restart still happens.
    """
    now = time.time()
    entries = _read_restart_history(art["restart_history"])
    entries = _prune_history(entries, now, slow_window_s)

    fast_count = _count_within(entries, now, fast_window_s)
    if fast_count >= max_fast:
        # Suppress: don't spawn. Still write pruned history back so the file
        # doesn't grow unbounded when the bot is trapped in a fast loop.
        _write_restart_history(art["restart_history"], entries)
        return {
            "action": "SUPPRESSED_FAST_CRASHLOOP",
            "fast_count": fast_count,
            "max_fast": max_fast,
        }

    slow_count = _count_within(entries, now, slow_window_s)
    escalate = slow_count >= max_slow

    try:
        new_pid, out_log = _spawn_bot(mode, art["logs_dir"])
    except Exception as e:
        sys.stderr.write(f"spawn_bot_failed mode={mode}: {type(e).__name__}: {e}\n")
        sys.stderr.write(traceback.format_exc())
        return {"action": "SPAWN_FAILED", "detail": f"{type(e).__name__}: {e}"}

    entries.append({
        "ts": _iso_utc_now(),
        "ts_wall": now,
        "trigger": status,
        "new_pid": new_pid,
        "log_path": str(out_log),
    })
    _write_restart_history(art["restart_history"], entries)

    return {
        "action": "SLOW_CRASHLOOP_WARNING" if escalate else "RESTARTED",
        "new_pid": new_pid,
        "log_path": str(out_log),
        "fast_count": fast_count,
        "slow_count": slow_count,
        "escalated": escalate,
    }


# -- Entrypoint ------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="supervisor.py",
        description="PancakeBot health supervisor (classify + optional alert + optional restart).",
    )
    p.add_argument(
        "--mode", required=True, choices=("dry", "live"),
        help="Which bot mode to supervise.",
    )
    p.add_argument(
        "--stale-threshold", type=float, default=5.0,
        help="Seconds after which a heartbeat is considered stale (default: 5.0).",
    )
    p.add_argument(
        "--startup-grace", type=float, default=90.0,
        help="Seconds after PID-file write during which a missing or stale heartbeat "
             "is tolerated (STARTING state; default: 90.0).",
    )
    p.add_argument(
        "--alert", action="store_true",
        help="Send a Discord alert on STALE/CRASHED/UNINSTRUMENTED/DOWN and on "
             "SUPPRESSED_FAST_CRASHLOOP / SLOW_CRASHLOOP_WARNING. Webhook URL "
             "comes from the DRY_DISCORD_ALERT_WEBHOOK_URL or "
             "LIVE_DISCORD_ALERT_WEBHOOK_URL env var; missing env var is a soft "
             "fallback to log-only.",
    )
    p.add_argument(
        "--restart", action="store_true",
        help="Auto-restart the bot on STALE/CRASHED/DOWN (never on UP, STARTING, "
             "UNINSTRUMENTED). Appends to var/<mode>/restart_history.jsonl and "
             "enforces fast + slow crashloop limits.",
    )
    p.add_argument(
        "--max-fast-restarts", type=int, default=3,
        help="Max restarts within --fast-window-min before we SUPPRESS (default 3).",
    )
    p.add_argument(
        "--fast-window-min", type=float, default=15.0,
        help="Fast crashloop window in minutes (default 15).",
    )
    p.add_argument(
        "--max-slow-restarts", type=int, default=8,
        help="Max restarts within --slow-window-h before we WARN on every "
             "subsequent restart (default 8).",
    )
    p.add_argument(
        "--slow-window-h", type=float, default=24.0,
        help="Slow crashloop window in hours (default 24). Also the retention "
             "horizon for restart_history.jsonl (older entries are pruned).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        status, fields = _classify(
            args.mode,
            stale_threshold_s=float(args.stale_threshold),
            startup_grace_s=float(args.startup_grace),
        )
    except Exception as e:
        # Supervisor errored during classification -- log to stderr, exit 99.
        sys.stderr.write(f"supervisor_classify_failed: {type(e).__name__}: {e}\n")
        sys.stderr.write(traceback.format_exc())
        return EXIT_ERROR

    art = _artifacts_for_mode(args.mode)

    # -- Phase 2d: auto-restart (opt-in) --
    action_taken: str | None = None
    escalation: str | None = None
    new_pid_from_restart: int | None = None
    suppressed_fast = False
    if args.restart and status in _RESTART_STATES:
        restart_result = _do_restart(
            mode=args.mode,
            status=status,
            art=art,
            max_fast=int(args.max_fast_restarts),
            fast_window_s=float(args.fast_window_min) * 60.0,
            max_slow=int(args.max_slow_restarts),
            slow_window_s=float(args.slow_window_h) * 3600.0,
        )
        action_taken = restart_result.get("action")
        new_pid_from_restart = restart_result.get("new_pid")
        if action_taken == "SUPPRESSED_FAST_CRASHLOOP":
            suppressed_fast = True
            escalation = "SUPPRESSED_FAST_CRASHLOOP"
        elif action_taken == "SLOW_CRASHLOOP_WARNING":
            escalation = "SLOW_CRASHLOOP_WARNING"

    # -- Phase 2c: Discord alert (opt-in) --
    alert_outcome: str | None = None
    if args.alert:
        # Escalated actions force an alert even if the underlying classification
        # wouldn't normally trigger one (though it always will here, since
        # STALE/CRASHED/DOWN are both _RESTART_STATES and _ALERT_STATES).
        if status in _ALERT_STATES or escalation is not None:
            alert_outcome = _maybe_send_discord(
                mode=args.mode,
                status=status,
                fields=fields,
                art=art,
                escalation=escalation,
            )

    # Decorate the single supervisor.log line with action + alert details.
    if action_taken is not None:
        fields["action"] = action_taken
    if new_pid_from_restart is not None:
        fields["new_pid"] = new_pid_from_restart
    if alert_outcome is not None:
        fields["alert"] = alert_outcome

    _write_supervisor_line(art["supervisor_log"], args.mode, status, fields)

    if suppressed_fast:
        return EXIT_SUPPRESSED_FAST_CRASHLOOP
    return _STATUS_TO_EXIT.get(status, EXIT_ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
