"""Cheap whole-store epoch extraction by byte scan, without JSON parsing.

WHY BYTE SCAN. The five stores total ~4.5 GB. Parsing every line as JSON to
answer "which epochs are present" measured ~2 minutes for a four-store pass
on the VM, which is precisely why no whole-store contiguity report existed:
it was too slow to run routinely, so nobody ran it, so gaps stayed
invisible. A regex over raw bytes answers the same question fast enough to
run on every sync.

WHAT THIS IS NOT. This does not validate records. A line whose epoch field
scans cleanly but whose body is malformed will be counted as present. That
is the correct trade for a CONTIGUITY report, whose question is "which
epochs exist", not "is every record well-formed". Record validity is the
readers' job and they raise on it.

The scan is deliberately tolerant of both compact and spaced separators, so
it keeps working on records written before separators were pinned.
"""
from __future__ import annotations

import re

# Matches "epoch": 123 with or without whitespace, anywhere in the line.
# Anchored on the quoted key so a bare number elsewhere cannot match.
_EPOCH_RE = re.compile(rb'"epoch"\s*:\s*(\d+)')

_CHUNK = 4 * 1024 * 1024


def scan_epochs(path: str) -> list[int]:
    """Return every epoch in file order. Empty list if the file is absent.

    Reads in chunks aligned to line boundaries so a match can never be
    split across a chunk edge.
    """
    return _scan(path)[0]


def scan_epochs_and_crlf(path: str) -> tuple[list[int], int]:
    """Epochs plus a count of CRLF line terminators, in ONE pass.

    CRLF is counted here rather than in a separate read because the scan
    already has every byte in hand. The stores were normalised to LF and
    the writers pinned to it; a non-zero count means something wrote
    through a path that translates newlines, which is a silent corruption
    of an append-only file the project cannot refetch.
    """
    return _scan(path)


def _scan(path: str) -> tuple[list[int], int]:
    out: list[int] = []
    crlf = 0
    try:
        fh = open(path, "rb")
    except FileNotFoundError:
        return out, 0
    with fh:
        remainder = b""
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            buf = remainder + chunk
            cut = buf.rfind(b"\n")
            if cut == -1:
                remainder = buf
                continue
            head = buf[: cut + 1]
            crlf += head.count(b"\r\n")
            for m in _EPOCH_RE.finditer(head):
                out.append(int(m.group(1)))
            remainder = buf[cut + 1 :]
        if remainder:
            crlf += remainder.count(b"\r\n")
            for m in _EPOCH_RE.finditer(remainder):
                out.append(int(m.group(1)))
    return out, crlf


def gap_runs(epochs: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of MISSING epochs inside the observed span.

    Returns [(first_missing, last_missing), ...]. Empty when contiguous.
    Operates on the set, so duplicate or out-of-order records do not
    manufacture phantom gaps.
    """
    if not epochs:
        return []
    present = set(epochs)
    lo, hi = min(present), max(present)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for e in range(lo, hi + 1):
        if e in present:
            if start is not None:
                runs.append((start, e - 1))
                start = None
        elif start is None:
            start = e
    if start is not None:
        runs.append((start, hi))
    return runs


def contiguity(path: str, known_absent: frozenset[int] | None = None) -> dict:
    """One store's contiguity facts.

    Reports out_of_order and duplicates separately from gaps: an ascending
    append-only store should have neither, and conflating them with missing
    epochs would hide a different defect inside the gap count.

    ``known_absent`` epochs are counted and reported SEPARATELY rather than
    as gaps. Without that separation the same permanently-unfetchable
    epochs are flagged on every run forever, which turns a real finding into
    background noise. Defaults to the store's recorded set; pass an explicit
    frozenset() to see the raw picture with no exceptions applied.

    An epoch is only ever excused if it is BOTH absent AND listed. Anything
    absent and unlisted stays a gap.
    """
    if known_absent is None:
        from pancakebot.market_data.known_absent import known_absent_for
        known_absent = known_absent_for(path)

    epochs, crlf = scan_epochs_and_crlf(path)
    if not epochs:
        return {
            "path": path, "n": 0, "distinct": 0, "earliest": None,
            "latest": None, "span": 0, "missing": 0, "runs": [],
            "duplicates": 0, "out_of_order": 0, "known_absent": 0,
            "crlf": crlf,
        }
    distinct = set(epochs)
    lo, hi = min(distinct), max(distinct)
    span = hi - lo + 1

    # Only epochs that are actually missing AND inside the span can be
    # excused; a listed epoch outside the range explains nothing here.
    missing_set = {e for e in range(lo, hi + 1) if e not in distinct}
    excused = missing_set & set(known_absent)
    unexplained = missing_set - excused

    runs = _runs_from_missing(unexplained)
    out_of_order = sum(
        1 for a, b in zip(epochs, epochs[1:]) if b <= a
    )
    return {
        "path": path,
        "n": len(epochs),
        "distinct": len(distinct),
        "earliest": lo,
        "latest": hi,
        "span": span,
        "missing": len(unexplained),
        "runs": runs,
        "duplicates": len(epochs) - len(distinct),
        "out_of_order": out_of_order,
        "known_absent": len(excused),
        "crlf": crlf,
    }


def _runs_from_missing(missing: set[int]) -> list[tuple[int, int]]:
    """Collapse a set of missing epochs into contiguous runs."""
    if not missing:
        return []
    ordered = sorted(missing)
    runs: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for e in ordered[1:]:
        if e == prev + 1:
            prev = e
            continue
        runs.append((start, prev))
        start = prev = e
    runs.append((start, prev))
    return runs


def format_report(rows: list[dict]) -> list[str]:
    """Operator-facing lines. One per store, plus gap detail when present.

    States "contiguous" explicitly rather than printing nothing, so a clean
    store produces a POSITIVE signal. Silence is what let the last set of
    gaps hide; an empty report and a healthy report must not look alike.
    """
    lines: list[str] = []
    for r in rows:
        name = r["path"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if r["n"] == 0:
            lines.append(f"INTEGRITY {name}: ABSENT or empty")
            continue
        state = "contiguous" if r["missing"] == 0 else f"{r['missing']} MISSING"
        lines.append(
            f"INTEGRITY {name}: n={r['n']} distinct={r['distinct']} "
            f"span=[{r['earliest']}..{r['latest']}] ({r['span']}) {state}"
            + (f" known_absent={r['known_absent']}(expected)"
               if r.get("known_absent") else "")
            + (f" duplicates={r['duplicates']}" if r["duplicates"] else "")
            + (f" out_of_order={r['out_of_order']}" if r["out_of_order"] else "")
        )
        for a, b in r["runs"][:10]:
            lines.append(
                f"INTEGRITY   gap {a}..{b} ({b - a + 1} epoch(s))"
            )
        if len(r["runs"]) > 10:
            lines.append(
                f"INTEGRITY   ... and {len(r['runs']) - 10} more gap run(s)"
            )
    return lines
