"""Read cycle_audit.csv with per-epoch dedup. Use this, not a bare csv scan.

WHY THIS EXISTS. On 2026-08-30 a stale-read fault made the engine round
loop free-run at ~1.4s/iteration for ~29 minutes. Every iteration wrote a
cycle_audit row, so the file gained 1,256 rows for a SINGLE round --
1,255 of them `risk_cooldown_active` on locked_epoch 511438. That is about
14% of the entire decision history to date.

THOSE ROWS ARE NOT JUNK TO BE DELETED. The engine really did make those
decisions; the file is a true record of what it did, and rewriting history
to make an analysis tidy is the opposite of what this project does. The
rows stay. The CONSUMERS have to stop treating "one row" as "one round".

Any per-round rate computed off a bare row count over that window is wrong
by ~200x. That includes the obvious ones (skip-reason rates, bet rates,
partial-kline rates) and the less obvious: reconstructing per-round state
by walking rows in order will replay one round hundreds of times.

DEDUP RULE: within a run of rows carrying the same locked_epoch, the LAST
row is authoritative. The engine overwrites its own view of a round as the
iteration progresses, so the final row is the one whose decision actually
stood.

Note the off-by-one that has cost time twice: `locked_epoch` is the epoch
that JUST LOCKED. The round being decided is locked_epoch + 1.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def read_rows(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dedup_by_epoch(rows: list[dict], *, key: str = "locked_epoch") -> list[dict]:
    """One row per DISTINCT epoch, keeping the last occurrence.

    Not a set-dedup: rows are kept in file order and a later re-appearance
    of an epoch (a genuinely re-decided round) replaces the earlier one.
    """
    out: dict[str, dict] = {}
    for r in rows:
        e = r.get(key)
        if e is None or e == "":
            continue
        out[e] = r
    return list(out.values())


def spin_report(rows: list[dict], *, key: str = "locked_epoch") -> list[tuple[str, int]]:
    """Epochs with more than one row, worst first -- the spin fingerprint."""
    c = Counter(r.get(key) for r in rows if r.get(key))
    return [(e, n) for e, n in c.most_common() if n > 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    rows = read_rows(args.path)
    ded = dedup_by_epoch(rows)
    dupes = spin_report(rows)
    inflated = len(rows) - len(ded)

    print("rows in file      : %d" % len(rows))
    print("distinct epochs   : %d" % len(ded))
    print("duplicate rows    : %d  (%.1f%% of the file)"
          % (inflated, 100.0 * inflated / len(rows) if rows else 0.0))
    if dupes:
        print("\nepochs with >1 row (worst %d):" % args.show)
        for e, n in dupes[:args.show]:
            print("  locked_epoch=%-9s %d rows" % (e, n))
        print("\nAny per-round rate computed WITHOUT dedup is wrong by up to")
        print("%.0fx over the affected window." % max(n for _, n in dupes))
    else:
        print("\nno duplicated epochs — one row per round throughout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
