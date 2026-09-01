"""Turn an ABSENT sync into something Zach cannot miss, with no server.

THE REQUIREMENT. "The sync has not run" must look different from "nothing
happened", on a machine with no server, no cron mail, and nobody watching.
Every August failure became invisible by being an absence, and an absence
is not a signal.

WHY A FILE ON THE DESKTOP. The candidate surfaces on a personal Windows
machine are all worse:

  * A log file       -- requires knowing to look, which is the failure mode.
  * A toast          -- only fires while someone is logged in, and is gone
                        the moment it is dismissed. It cannot represent a
                        state that persists for four days.
  * Email/Discord    -- needs the network, which is one of the very things
                        that can be broken; and a push that fails is silent.
  * A tray app       -- a server by another name, and something else to die.

A file on the Desktop is passive, persistent, survives reboots, needs no
process to stay alive to keep being visible, and sits in the one place the
operator looks at every day without being asked to. It represents a STATE
rather than an EVENT, which is what a multi-day stall actually is.

The marker is named so the whole diagnosis is in the FILENAME, because a
file that must be opened to be understood is one step away from ignored.

DISCORD IS BEST-EFFORT AND SECONDARY. It is attempted when configured, but
the Desktop marker is the primary channel precisely because it does not
depend on the network being healthy.

THE IRREDUCIBLE BLIND SPOT, STATED PLAINLY. If the machine is off, nothing
here runs and nothing here notices. No local watchdog can observe its own
absence. What this guarantees is that the stall is visible the moment the
machine is used again -- not that it is detected while the machine is off.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.ops.store_guard import (  # noqa: E402
    evaluate,
    failure_tag,
    load_baseline,
    save_baseline,
    snapshot,
)
from pancakebot.ops.sync_health import assess, summary_line  # noqa: E402

# Resolved explicitly rather than via GetFolderPath: this runs as SYSTEM,
# whose own Desktop nobody ever looks at.
DEFAULT_DESKTOP = r"C:\Users\zking\Desktop"

_MARKER_PREFIX = "PANCAKEBOT_SYNC"


def _clear_markers(desktop: str) -> list[str]:
    removed = []
    try:
        for name in os.listdir(desktop):
            if name.startswith(_MARKER_PREFIX) and name.endswith(".txt"):
                try:
                    os.remove(os.path.join(desktop, name))
                    removed.append(name)
                except OSError:
                    pass
    except OSError:
        pass
    return removed


def _marker_name(a: dict, fails: list[str] | None = None) -> str:
    """The filename carries the diagnosis, so it needs no opening."""
    if fails:
        # Integrity beats staleness in the name: a store that is CORRUPTING
        # is a more urgent thing to read off a desktop than one that is
        # merely behind.
        return f"{_MARKER_PREFIX}_INTEGRITY_{failure_tag(fails)}.txt"
    st = a["status"]
    if st == "NEVER_RUN":
        return f"{_MARKER_PREFIX}_HAS_NEVER_RUN.txt"
    days = int((a["hours_since_success"] or 0) // 24)
    if days >= 1:
        return f"{_MARKER_PREFIX}_STOPPED_{days}_DAYS_AGO.txt"
    hours = int(a["hours_since_success"] or 0)
    return f"{_MARKER_PREFIX}_STOPPED_{hours}_HOURS_AGO.txt"


def _marker_body(a: dict, fails: list[str] | None = None) -> str:
    if fails:
        return (
            "PANCAKEBOT STORE INTEGRITY REGRESSED\n"
            "====================================\n\n"
            + "\n".join("  * " + f for f in fails) + "\n\n"
            f"Sync recency is separate and currently reads: {summary_line(a)}\n\n"
            "WHY THIS FILE EXISTS\n"
            "The daily sync may be running perfectly on schedule and still\n"
            "be corrupting or losing data. This file appears when the store\n"
            "files themselves failed a check, not when the sync stopped.\n\n"
            "WHAT IS AT RISK\n"
            "The five store files are the only surviving copy of data that\n"
            "cannot be refetched: OKX serves 1s klines for about 171.6 days,\n"
            "and anything older exists nowhere else.\n\n"
            "DO NOT let the sync keep running blindly if a store SHRANK or\n"
            "REWOUND. A backup exists at:\n"
            "  C:\\Users\\zking\\Downloads\\pancakebot_store_backup_20260829\\\n\n"
            "WHAT TO DO\n"
            "  1. Do not delete this file to make it go away; it clears\n"
            "     itself once the check passes.\n"
            "  2. Inspect:  .venv\\Scripts\\python.exe scripts\\store_integrity.py\n"
            "  3. Compare against var\\store_baseline.json, which holds the\n"
            "     previous run's numbers.\n"
        )
    return (
        "PANCAKEBOT DAILY SYNC IS NOT RUNNING\n"
        "====================================\n\n"
        f"{summary_line(a)}\n\n"
        "WHY THIS FILE EXISTS\n"
        "This file appears on the Desktop only when the daily market-data\n"
        "sync has stopped succeeding. It is deleted automatically on the\n"
        "next successful run, so its presence always means a live problem.\n\n"
        "WHAT IS AT RISK\n"
        "The five store files are the only surviving copy of data that\n"
        "cannot be refetched: OKX serves 1s klines for about 171.6 days,\n"
        "and anything older exists nowhere else. A sync that stops for\n"
        "longer than that horizon loses those epochs permanently.\n\n"
        "WHAT TO DO\n"
        "  1. Open a terminal in the repo.\n"
        "  2. Run:  .venv\\Scripts\\python.exe run.py --sync\n"
        "  3. If it fails, the error above is the place to start.\n\n"
        "Check the task itself with:\n"
        "  schtasks /query /tn PancakeBotDailySync /v /fo LIST\n"
    )


def _try_discord(text: str) -> str:
    """Best effort. Never raises, and never blocks the marker."""
    url = (os.environ.get("PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL")
           or os.environ.get("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL"))
    if not url:
        return "discord: not configured (no webhook in environment or .env)"
    try:
        import json
        import urllib.request
        req = urllib.request.Request(
            url,
            data=json.dumps({"content": text[:1900]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=15).read()
        return "discord: sent"
    except Exception as e:      # noqa: BLE001
        return f"discord: FAILED ({type(e).__name__})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desktop", default=DEFAULT_DESKTOP)
    ap.add_argument("--health", default="var/sync_health.json")
    ap.add_argument("--baseline", default="var/store_baseline.json")
    ap.add_argument("--no-discord", action="store_true")
    ap.add_argument("--staging-dir", default="staged_captures")
    ap.add_argument("--no-capture", action="store_true",
                    help="report gaps but do not fetch them into staging")
    args = ap.parse_args()

    a = assess(args.health)
    line = summary_line(a)
    print(line)

    # ---- integrity, evaluated INDEPENDENTLY of recency -------------------
    #
    # These answer different questions and either can fail alone. A sync
    # that runs every day on schedule while quietly losing records is
    # healthy by recency and broken by integrity, and that combination is
    # exactly the one nothing used to catch.
    #
    # Wrapped so a fault in the CHECK cannot be mistaken for a clean store:
    # an exception here is reported as a failure, never swallowed into a
    # pass. "The check could not run" and "the check found nothing" must
    # not look alike.
    fails: list[str] = []
    try:
        snap = snapshot(staging_dir=args.staging_dir)
        fails = evaluate(snap, load_baseline(args.baseline))
        for name, c in sorted((snap.get("stores") or {}).items()):
            if c.get("error"):
                print(f"  {name}: ERROR {c['error']}")
            else:
                print(f"  {name}: n={c['n']} latest={c['latest']} "
                      f"crlf={c['crlf']} missing={c['missing']} "
                      f"known_absent={c['known_absent']}")
        # Advance the baseline only when the stores are readable, so a
        # transient read failure cannot poison the comparison point. A
        # regression is deliberately still recorded: the next run compares
        # against today, which stops one bad day alarming forever, and the
        # marker for today has already been raised.
        if not any(c.get("error") for c in (snap.get("stores") or {}).values()):
            save_baseline(snap, args.baseline)
    except Exception as e:      # noqa: BLE001
        fails = [f"CHECKFAILED: the integrity check itself raised "
                 f"{type(e).__name__}: {e}"]

    # ---- capture: preserve gap records before the horizon takes them -----
    #
    # Runs only when a gap is actually present, and NEVER from the sync
    # task -- this is the watchdog, a separate task, so a failing or slow
    # capture cannot block or delay data collection. Collection always wins.
    #
    # Wrapped whole: a capture that explodes must not be able to suppress
    # the integrity findings that prompted it.
    if any(f.startswith("GAP") for f in fails) and not args.no_capture:
        try:
            from capture_gaps import run as capture_run
            cap = capture_run(staging_dir=args.staging_dir)
            print(f"  capture: captured={cap['captured']} "
                  f"failed={cap['failed']} "
                  f"unrecoverable={cap['unrecoverable']} "
                  f"blocked={cap['blocked']} "
                  f"already_staged={cap['already_staged']}")
            if cap["failed"]:
                # HIGH urgency and deliberately FIRST: the horizon clock is
                # running on records that are still fetchable today and will
                # not be forever.
                fails.insert(0, f"CAPTUREFAILED: {cap['failed']} gap record(s) "
                                f"could not be fetched -- still recoverable "
                                f"today, permanently lost after the "
                                f"171.6-day horizon")
            if cap["unrecoverable"]:
                # INFO. Nothing can be done, and saying so plainly is what
                # stops a lost cause training the operator to ignore the
                # states that DO need action.
                fails.append(f"UNRECOVERABLE: {cap['unrecoverable']} epoch(s) "
                             f"are past the OKX horizon and gone for good; "
                             f"recorded as permanently absent, no retry")
            # Re-read staging so a capture that just succeeded shows up as
            # CAPTURED (unmerged) in the SAME run rather than a day later.
            fails = [f for f in fails if not f.startswith("CAPTURED")]
            from pancakebot.ops.store_guard import _unmerged_staged_failures
            fails.extend(_unmerged_staged_failures(
                (snapshot(staging_dir=args.staging_dir).get("stores") or {})))
        except Exception as e:      # noqa: BLE001
            fails.insert(0, f"CAPTUREFAILED: capture itself raised "
                            f"{type(e).__name__}: {e}")

    for f in fails:
        print(f"  INTEGRITY FAIL: {f}")

    healthy = (a["status"] == "OK") and not fails

    if healthy:
        removed = _clear_markers(args.desktop)
        if removed:
            print(f"cleared stale marker(s): {removed}")
        return 0

    # Not OK: replace any existing marker so the filename reflects TODAY.
    _clear_markers(args.desktop)
    name = _marker_name(a, fails)
    try:
        os.makedirs(args.desktop, exist_ok=True)
        with open(os.path.join(args.desktop, name), "w",
                  encoding="utf-8", newline="") as f:
            f.write(_marker_body(a, fails))
        print(f"WROTE DESKTOP MARKER: {name}")
    except OSError as e:
        print(f"FAILED to write desktop marker: {e}", file=sys.stderr)

    if not args.no_discord:
        body = "\n".join(fails) if fails else line
        print(_try_discord(f"**PancakeBot sync problem**\n{body}"))

    # Non-zero so Task Scheduler's Last Result also carries the signal.
    return 1


if __name__ == "__main__":
    sys.exit(main())
