"""Ping-fallback: send a Discord follow-up for a pending coordinator message.

Designed to be invoked by Windows Task Scheduler (via ``pythonw.exe`` so no
console window flashes) against a specific notification ``id``. If the user
has already answered (or a follow-up has already fired for this id) the
script is a silent no-op -- so it's safe to invoke via a scheduled task
that may fire after the user responded.

File layout (host-side, under ``var/notifications/``; the whole ``var/`` tree
is gitignored):

    pending.jsonl   one JSON object per line:
        {
          "id":       "<uuid4>",
          "ts_sent":  "2026-04-23T03:00:00Z",
          "subject":  "...",
          "summary":  "...",
          "answered": false,
          "fired":    false
        }

Usage:
    python tools/claude_desktop/notify_user_followup.py --id <uuid>

Exit codes:
    0 - done (alert sent, OR already answered/fired -> silent no-op)
    1 - entry not found
    2 - Discord send failure (env var missing or HTTP failure)
    3 - rewrite failure (best-effort post may still have landed)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
PENDING_PATH = _REPO / "var" / "notifications" / "pending.jsonl"
GENERAL_WEBHOOK_ENV = "PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL"


def _read_pending(path: Path) -> list[dict]:
    entries: list[dict] = []
    if not path.exists():
        return entries
    try:
        text = path.read_text(encoding="utf-8")
    except (PermissionError, OSError):
        return entries
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _atomic_write_jsonl(path: Path, entries: list[dict]) -> None:
    """tempfile + os.replace -- supervisor already uses this pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        json.dumps(e, sort_keys=True, separators=(",", ":")) for e in entries
    )
    if body:
        body += "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _format_ts_human(ts_sent: str) -> str:
    """Render ISO-8601 UTC -> '2026-04-23T03:00:00Z (3h ago)'."""
    try:
        parsed = datetime.strptime(ts_sent, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - parsed
        secs = int(delta.total_seconds())
        if secs < 60:
            ago = f"{secs}s ago"
        elif secs < 3600:
            ago = f"{secs // 60}m ago"
        elif secs < 86400:
            ago = f"{secs // 3600}h {(secs % 3600) // 60}m ago"
        else:
            ago = f"{secs // 86400}d {(secs % 86400) // 3600}h ago"
        return f"{ts_sent} ({ago})"
    except Exception:
        return ts_sent


def _local_time_str() -> str:
    """Human-readable local time in America/New_York for Discord messages."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _send_discord(webhook_url: str, content: str) -> tuple[bool, str]:
    try:
        import requests
    except Exception as e:
        return False, f"requests_import_failed:{e}"
    try:
        r = requests.post(
            webhook_url,
            json={"content": content, "username": "PancakeBot-coordinator"},
            timeout=15,
        )
    except Exception as e:
        return False, f"post_exception:{type(e).__name__}:{e}"
    if 200 <= r.status_code < 300:
        return True, f"http_{r.status_code}"
    return False, f"http_{r.status_code}:{(r.text or '')[:200]}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="notify_user_followup.py")
    p.add_argument("--id", required=True, help="uuid of the pending entry to follow up on")
    args = p.parse_args(argv)

    entries = _read_pending(PENDING_PATH)
    target_idx = None
    for i, e in enumerate(entries):
        if str(e.get("id", "")) == args.id:
            target_idx = i
            break
    if target_idx is None:
        sys.stderr.write(f"notify_followup: id not found: {args.id}\n")
        return 1

    entry = entries[target_idx]
    if bool(entry.get("answered")) or bool(entry.get("fired")):
        # Silent no-op -- the followup is no longer needed.
        return 0

    webhook = os.environ.get(GENERAL_WEBHOOK_ENV, "").strip()
    if not webhook:
        sys.stderr.write(
            f"notify_followup: {GENERAL_WEBHOOK_ENV} not set in env; cannot alert\n"
        )
        return 2

    subject = str(entry.get("subject", "(no subject)"))
    summary = str(entry.get("summary", "(no summary)"))
    ts_sent = str(entry.get("ts_sent", "?"))
    content = (
        f":bell: **Pending response from PancakeBot Coordinator**\n"
        f"**Subject:** {subject}\n"
        f"**Summary:** {summary}\n"
        f"**Sent:** {_format_ts_human(ts_sent)}\n"
        f"**Local now:** {_local_time_str()}\n"
        f"**ID:** `{entry.get('id')}`\n"
        f"\n"
        f"Reply in the Cowork app or reply here."
    )
    ok, detail = _send_discord(webhook, content)
    if not ok:
        sys.stderr.write(f"notify_followup: discord send failed detail={detail}\n")
        return 2

    # Mark fired=true and persist. If the rewrite fails the alert has already
    # landed -- flag that so the user / scheduler can decide what to do.
    entry["fired"] = True
    entries[target_idx] = entry
    try:
        _atomic_write_jsonl(PENDING_PATH, entries)
    except Exception as e:
        sys.stderr.write(
            f"notify_followup: alert sent but rewrite failed: {type(e).__name__}: {e}\n"
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
