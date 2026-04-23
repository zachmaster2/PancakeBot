"""Ping-fallback: mark notification entries as answered.

Called when the user responds so subsequent scheduled ``notify_user_followup``
invocations become silent no-ops.

Usage:
    python scripts/notify_user_mark_answered.py --id <uuid>
    python scripts/notify_user_mark_answered.py --all-unanswered

``--all-unanswered`` is the common case when the user replies to the running
Claude session: we don't know which specific pending id the reply is
addressing, so the safest interpretation is "everything outstanding is now
answered." Already-answered entries stay answered; this command is strictly
idempotent.

Exit codes:
    0 - done (any changes persisted)
    1 - entry not found (--id path only)
    2 - neither --id nor --all-unanswered given
    3 - rewrite failure
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
PENDING_PATH = _REPO / "var" / "notifications" / "pending.jsonl"


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="notify_user_mark_answered.py")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", help="mark a specific entry answered")
    g.add_argument(
        "--all-unanswered",
        action="store_true",
        help="mark every answered=false entry as answered (common case)",
    )
    args = p.parse_args(argv)

    entries = _read_pending(PENDING_PATH)
    changed = 0

    if args.id is not None:
        target_idx = None
        for i, e in enumerate(entries):
            if str(e.get("id", "")) == args.id:
                target_idx = i
                break
        if target_idx is None:
            sys.stderr.write(f"mark_answered: id not found: {args.id}\n")
            return 1
        if not bool(entries[target_idx].get("answered")):
            entries[target_idx]["answered"] = True
            changed = 1
    else:
        for e in entries:
            if not bool(e.get("answered")):
                e["answered"] = True
                changed += 1

    if changed == 0:
        # Idempotent no-op. Still succeeds.
        sys.stdout.write("mark_answered: no changes (already up to date)\n")
        return 0

    try:
        _atomic_write_jsonl(PENDING_PATH, entries)
    except Exception as e:
        sys.stderr.write(
            f"mark_answered: rewrite failed: {type(e).__name__}: {e}\n"
        )
        return 3

    sys.stdout.write(f"mark_answered: {changed} entry(ies) marked answered\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
