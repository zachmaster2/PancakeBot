"""Full-scan integrity report for the JSONL stores.

REPLACES the EOL leg of the ad-hoc /tmp/vm_preflight.py used during the
2026-08-24 store repair. That checker sampled the first 1 MB and a 1 MB
window near the end, and reported anything where the two samples agreed as
"uniform". Two consequences, both reproduced against synthetic files:

  * A file whose terminator changes only in the MIDDLE is called uniform.
    That is not hypothetical -- eth/sol_spot_prices.jsonl are
    LF x39685, CRLF x17067, LF x15099, and both sampled windows land in
    LF regions. It reported "uniform LF" for files carrying 17,067 CRLF
    lines each, which normalisation then had to convert.
  * Worse: region() returns None when a sample is itself mixed, and the
    reporting line is "uniform {'CRLF' if head else 'LF'}". With head and
    tail both None, None == None takes the uniform branch and None is
    falsy, so a file it could not classify AT ALL is reported "uniform
    LF". "I cannot tell" is rendered as a confident clean answer.

Its other legs -- byte size, line count, first/last epoch, gap runs,
missing epochs -- were full scans and are sound; only the EOL leg was
sampled. This tool keeps those semantics and makes the EOL leg a full
scan that reports every run, so a multi-transition file describes itself
instead of being squeezed into a single-transition shape.

Read-only: opens every file "rb" and writes nothing.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

_KEY = b'"epoch":'


def line_epoch(raw: bytes) -> int | None:
    i = raw.find(_KEY)
    if i < 0:
        return None
    j = i + len(_KEY)
    while raw[j:j + 1] in (b" ", b"\t"):
        j += 1
    k = j
    while k < len(raw) and 48 <= raw[k] <= 57:
        k += 1
    return int(raw[j:k]) if k > j else None


def eol_runs(path) -> list[tuple[str, int, int]]:
    """[(kind, first_line, count)] over the WHOLE file.

    Full scan, never a sample. A single-element result means genuinely
    uniform; anything longer is a mixed file and says exactly where.
    """
    runs: list[list] = []
    prev = None
    n = 0
    with io.open(path, "rb") as f:
        for raw in f:
            n += 1
            if raw.endswith(b"\r\n"):
                kind = "CRLF"
            elif raw.endswith(b"\n"):
                kind = "LF"
            elif raw.endswith(b"\r"):
                kind = "CR"
            else:
                kind = "NONE"          # final line with no terminator
            if kind != prev:
                runs.append([kind, n, 1])
                prev = kind
            else:
                runs[-1][2] += 1
    return [(k, s, c) for k, s, c in runs]


def describe_eol(runs) -> str:
    if not runs:
        return "empty"
    if len(runs) == 1:
        return f"uniform {runs[0][0]}"
    return "; ".join(f"{k}x{c}@line{s}" for k, s, c in runs)


def scan(path) -> dict:
    p = Path(path)
    size = p.stat().st_size
    eps: list[int] = []
    unparsable = 0
    with io.open(p, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            e = line_epoch(raw)
            if e is None:
                unparsable += 1
            else:
                eps.append(e)
    runs = eol_runs(p)
    out = {
        "path": str(p), "bytes": size, "lines": sum(c for _, _, c in runs),
        "records": len(eps), "unparsable": unparsable,
        "eol": describe_eol(runs), "eol_runs": runs,
    }
    if eps:
        seen = set(eps)
        missing = [e for e in range(eps[0], eps[-1] + 1) if e not in seen]
        gap_runs = 0
        prev = None
        for e in missing:
            if prev is None or e != prev + 1:
                gap_runs += 1
            prev = e
        out.update(first=eps[0], last=eps[-1], gaps=gap_runs,
                   missing=len(missing),
                   ascending=all(b > a for a, b in zip(eps, eps[1:])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    print("%-26s %13s %8s %6s %6s %7s  %s"
          % ("file", "bytes", "records", "first", "last", "missing", "eol"))
    print("-" * 118)
    rc = 0
    for path in args.paths:
        r = scan(path)
        print("%-26s %13d %8d %6s %6s %7s  %s"
              % (Path(r["path"]).name, r["bytes"], r["records"],
                 r.get("first", "-"), r.get("last", "-"),
                 r.get("missing", "-"), r["eol"]))
        if r["unparsable"]:
            print("    !! %d unparsable line(s)" % r["unparsable"]); rc = 1
        if not r.get("ascending", True):
            print("    !! epochs are NOT strictly ascending"); rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
