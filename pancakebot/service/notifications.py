"""Discord notification state machine for the bot supervisor service.

Pure logic plus a single HTTP call (``requests.post``). Extracted from
``scripts/supervisor.py`` Phase 2c code, extended with the new states
introduced by the service refactor (REBOOTED, STOPPED, MODE_TRANSITION,
MODE_TRANSITION_REFUSED, STARTED, SERVICE_CRASHED).

Channels (env-var names unchanged from legacy):
  PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL  -> live mode events
  PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL   -> dry mode events
  PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL      -> cross-cutting (UNINSTRUMENTED,
                                                  SERVICE_CRASHED, supervisor-
                                                  self errors)

Rate limit: one alert per (mode, kind) per 5 minutes (matches legacy).
Unset env var: silent fallback (no HTTP, no crash). HTTP failure: logged
to stderr, never raises.

The mode mutex routes MODE_TRANSITION (Live started, Dry stopped) to the
**live** channel since that's the actionable event. MODE_TRANSITION_REFUSED
(Dry refused to start because Live is up) routes to the **dry** channel
since dry-watchers care that their start attempt was vetoed.
"""
from __future__ import annotations

import datetime
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants & routing tables
# ---------------------------------------------------------------------------

_ALERT_COOLDOWN_S: float = 300.0  # 5 min, matches legacy supervisor

LIVE_WEBHOOK_ENV = "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL"
DRY_WEBHOOK_ENV = "PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL"
GENERAL_WEBHOOK_ENV = "PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL"

# Channel routing per notification kind. "mode" = use mode-specific webhook;
# "general" = use general channel; "live" / "dry" = always that channel
# regardless of the firing service's mode.
_CHANNEL_BY_KIND: dict[str, str] = {
    # Bot-health states.
    "CRASHED": "mode",
    "DOWN": "mode",
    "UNINSTRUMENTED": "general",
    # Crashloop limiter outcomes.
    "SUPPRESSED_FAST_CRASHLOOP": "mode",
    "SLOW_CRASHLOOP_WARNING": "mode",
    "SPAWN_FAILED": "mode",
    # New service-only states.
    "STARTED": "mode",
    "REBOOTED": "mode",
    "RECOVERY_AFTER_CRASH": "mode",
    "STOPPED": "mode",
    "MODE_TRANSITION": "live",  # Live started, Dry was stopped
    "MODE_TRANSITION_REFUSED": "dry",  # Dry refused, Live is up
    "SERVICE_CRASHED": "general",
}

# Severity tag per kind. ASCII-only, monospace-friendly. Replaces the
# earlier emoji-shortcode header that rendered as Discord emoji glyphs
# (visually noisy and hard to filter / grep in alert pipelines).
_SEVERITY_BY_KIND: dict[str, str] = {
    "CRASHED": "CRIT",
    "DOWN": "CRIT",
    "STOPPED": "CRIT",
    "SPAWN_FAILED": "CRIT",
    "SERVICE_CRASHED": "CRIT",
    "UNINSTRUMENTED": "WARN",
    "MODE_TRANSITION_REFUSED": "WARN",
    "SUPPRESSED_FAST_CRASHLOOP": "WARN",
    "SLOW_CRASHLOOP_WARNING": "WARN",
    "STARTED": "INFO",
    "REBOOTED": "INFO",
    "RECOVERY_AFTER_CRASH": "INFO",
    "MODE_TRANSITION": "INFO",
}


def _local_time_str() -> str:
    """America/New_York wall time for Discord human-readability."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
        return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Channel resolution
# ---------------------------------------------------------------------------

def resolve_webhook_env(mode: str, kind: str) -> str:
    """Return the env-var name whose value is the target webhook URL."""
    channel = _CHANNEL_BY_KIND.get(kind, "mode")
    if channel == "general":
        return GENERAL_WEBHOOK_ENV
    if channel == "live":
        return LIVE_WEBHOOK_ENV
    if channel == "dry":
        return DRY_WEBHOOK_ENV
    # "mode" -> mode-specific
    if mode == "live":
        return LIVE_WEBHOOK_ENV
    if mode == "dry":
        return DRY_WEBHOOK_ENV
    raise ValueError(f"unknown_mode_for_routing: {mode!r}")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def _safe_read_json(path: Path) -> dict | None:
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
    return obj if isinstance(obj, dict) else None


def rate_limit_ok(last_alert_path: Path, key: str, now: float, cooldown_s: float = _ALERT_COOLDOWN_S) -> bool:
    """True if we haven't alerted for ``key`` within ``cooldown_s``.

    Updates last_alert.json on every attempt (success OR failure) so a flapping
    endpoint doesn't cause the supervisor to hammer it.
    """
    data = _safe_read_json(last_alert_path) or {}
    last_ts = 0.0
    raw = data.get(key)
    if isinstance(raw, (int, float)):
        last_ts = float(raw)
    if (now - last_ts) < cooldown_s:
        return False
    data[key] = now
    try:
        last_alert_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = last_alert_path.parent / (last_alert_path.name + ".tmp")
        tmp.write_text(
            json.dumps(data, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(last_alert_path)
    except Exception:
        # Best-effort. Still allow the send if we can't persist state.
        pass
    return True


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _clip_text(text: str, max_lines: int, max_chars: int) -> str:
    lines = text.splitlines()
    result = "\n".join(lines[:max_lines])
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... [truncated]"
    return result


def _tail_latest_err_log(logs_dir: Path, max_lines: int = 20) -> str | None:
    """Return the last *max_lines* of the most-recent ``*_err.log`` file."""
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


def build_message(
    *,
    mode: str,
    kind: str,
    fields: dict[str, Any] | None = None,
    art: dict[str, Path] | None = None,
    detail: str | None = None,
) -> str:
    """Compose a Discord message body for the given notification kind."""
    fields = fields or {}
    hostname = socket.gethostname()
    local = _local_time_str()

    severity = _SEVERITY_BY_KIND.get(kind, "INFO")
    # D4: STOPPED is INFO when intentional (admin/SCM stop, mode mutex, deploy)
    # and CRIT otherwise. The supervisor passes ``intentional=True`` on the clean
    # stop path; any future unintentional STOPPED stays CRIT.
    if kind == "STOPPED":
        severity = "INFO" if fields.get("intentional") else "CRIT"
    # D5: [MODE] prefix on EVERY alert (matches the bet-alert channel prefix) —
    # consistent visual scan + a guard against webhook misconfiguration (a
    # misrouted alert is obvious by its mode tag).
    mode_tag = f"[{mode.upper()}] "
    header = f"{mode_tag}[{severity}] **{kind}** `PancakeBot-{mode}` on `{hostname}` at `{local}`"
    lines: list[str] = [header]

    if detail:
        lines.append(f"detail: `{detail}`")

    # Common context fields (best-effort).
    for k in ("pid", "bankroll", "iterations", "last_epoch"):
        if k in fields:
            lines.append(f"{k}: `{fields[k]}`")

    if kind == "CRASHED" and art is not None:
        crash = _safe_read_json(art["crash"])
        if crash is not None:
            exc_type = crash.get("exc_type", "?")
            exc_repr = crash.get("exc_repr", "?")
            lines.append(f"exc: `{exc_type}`")
            lines.append(f"repr: `{exc_repr}`")
            tb_raw = str(crash.get("traceback_str", ""))
            tb = _clip_text(tb_raw, max_lines=20, max_chars=1500)
            if tb:
                lines.append("```\n" + tb + "\n```")
    elif kind == "UNINSTRUMENTED":
        lines.append("note: legacy bot detected outside service control")
    elif kind == "SPAWN_FAILED":
        spawn_err = fields.get("spawn_error")
        if isinstance(spawn_err, str) and spawn_err:
            lines.append(f"spawn_error: `{spawn_err}`")
        lines.append("note: service failed to spawn a bot child; manual intervention required")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discord HTTP send
# ---------------------------------------------------------------------------

def _send_discord(webhook_url: str, mode: str, message: str) -> tuple[bool, str]:
    """POST a Discord message. Returns ``(ok, detail)``. Never raises."""
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


# ---------------------------------------------------------------------------
# Public entry: notify(...)
# ---------------------------------------------------------------------------

def notify(
    *,
    mode: str,
    kind: str,
    fields: dict[str, Any] | None = None,
    art: dict[str, Path] | None = None,
    detail: str | None = None,
) -> str:
    """Dispatch a notification. Returns an outcome tag for the caller's log.

    Outcomes: SENT, DISABLED (env var unset), RATE_LIMITED, SEND_FAILED.

    Never raises. If ``art`` is None, rate-limit state is not persisted
    (used by the service crash path where we may not have an art dict
    available); the alert still tries to send.
    """
    env_var = resolve_webhook_env(mode, kind)
    webhook = os.environ.get(env_var, "").strip()
    if not webhook:
        return "DISABLED"

    now = time.time()
    if art is not None:
        if not rate_limit_ok(art["last_alert"], kind, now):
            return "RATE_LIMITED"

    msg = build_message(mode=mode, kind=kind, fields=fields, art=art, detail=detail)
    ok, send_detail = _send_discord(webhook, mode, msg)
    if ok:
        return "SENT"
    # safe_stderr_write handles sys.stderr=None when hosted by
    # pythonservice.exe (otherwise an AttributeError crashes the service —
    # caught 2026-05-23 post-reboot when first-run Discord POST failed on
    # not-yet-resolved DNS and the SEND_FAILED branch took down the supervisor).
    from pancakebot.service.supervision import safe_stderr_write
    safe_stderr_write(
        f"discord_send_failed mode={mode} kind={kind} detail={send_detail}"
    )
    return "SEND_FAILED"


def notify_service_error(*, mode: str, exc: BaseException) -> None:
    """Fire a best-effort general-channel alert when the service itself errors.

    Called from outer except-blocks in service SvcDoRun. No rate-limit
    persistence (no art dict available in a crash path). Never raises.
    """
    try:
        webhook = os.environ.get(GENERAL_WEBHOOK_ENV, "").strip()
        if not webhook:
            return
        hostname = socket.gethostname()
        local = _local_time_str()
        tb = _clip_text(traceback.format_exc(), max_lines=20, max_chars=1500)
        msg = "\n".join([
            f"[CRIT] **SERVICE_CRASHED** `PancakeBot-{mode}` on `{hostname}` at `{local}`",
            f"exc: `{type(exc).__name__}`",
            f"repr: `{exc!r}`",
            "```\n" + tb + "\n```" if tb else "",
        ]).strip()
        _send_discord(webhook, mode, msg)
    except Exception:
        # Last-ditch; never let an alert failure cascade from an already-failing context.
        pass
