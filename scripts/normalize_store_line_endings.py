"""Normalise a JSONL store to bare-LF terminators. Never in place.

THE RISK THIS CODE CARRIES. The five canonical stores total 4.37 GiB and
roughly 1.42 GiB of that -- every kline round older than the OKX 171.6-day
horizon -- cannot be re-obtained from anyone at any price. A rewrite of
every store file is exactly the operation that has already cost this
project canonical data once. So:

  * NEVER IN PLACE. Output goes to ``<path>.normalized``. This script does
    not swap; ``--swap`` is a separate, explicit step that refuses to run
    unless verification passed in the same invocation.
  * VERIFY BY CONTENT, NOT BY FILE BYTES. Whole-file bytes MUST change --
    that is the point. What must not change is the content of any line, so
    every line's body (the bytes before the terminator) is compared
    source-vs-output, and re-parsed as JSON to confirm it is still a valid
    record with the same epoch. Body-byte identity is STRICTLY STRONGER
    than comparing parsed objects: identical bytes parse identically, but
    two different byte strings can parse equal (key order, whitespace,
    float spelling). Bytes catch everything object comparison would, and
    more.
  * PREDICT THE DELTA. Each CRLF line loses exactly one byte. The expected
    size drop is computed BEFORE writing and the actual drop must match it
    exactly. A mismatch of one byte aborts.

RUN ORDER MATTERS. The writers must pin newline="" FIRST. Normalising
while an append path still emits CRLF on Windows means the next sync
re-mixes a file that was just rewritten -- a 4.37 GiB rewrite undone by
one append. See tests/test_store_line_endings_are_explicit.py.

Usage:
    python scripts/normalize_store_line_endings.py var/closed_rounds.jsonl
    python scripts/normalize_store_line_endings.py var/*.jsonl --swap
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path


class VerifyError(RuntimeError):
    pass


def census(path: Path) -> tuple[int, int, int]:
    """(lines, crlf, lf) by a full scan -- never a sample. vm_preflight.py
    samples head+tail and reported two of these files as 'uniform LF' when
    each in fact carries a 17,067-line CRLF block in the middle."""
    lines = crlf = lf = 0
    with io.open(path, "rb") as f:
        for raw in f:
            lines += 1
            if raw.endswith(b"\r\n"):
                crlf += 1
            elif raw.endswith(b"\n"):
                lf += 1
    return lines, crlf, lf


def normalize(src: Path, dst: Path) -> dict:
    """Stream src -> dst converting every terminator to a bare LF.

    Binary end to end. Text mode here would be self-defeating: it is the
    very translation this script exists to remove.
    """
    n = crlf = lf = noterm = 0
    with io.open(src, "rb") as fi, io.open(dst, "wb") as fo:
        for raw in fi:
            n += 1
            if raw.endswith(b"\r\n"):
                body = raw[:-2]; crlf += 1
            elif raw.endswith(b"\n"):
                body = raw[:-1]; lf += 1
            else:
                body = raw; noterm += 1
            fo.write(body + b"\n")
        fo.flush()
        os.fsync(fo.fileno())
    return {"lines": n, "crlf": crlf, "lf": lf, "no_terminator": noterm}


def verify(src: Path, dst: Path, expected_delta: int) -> dict:
    """Content identity, line by line, plus the predicted size delta."""
    s_sz, d_sz = src.stat().st_size, dst.stat().st_size
    actual_delta = s_sz - d_sz
    if actual_delta != expected_delta:
        raise VerifyError(
            f"size delta {actual_delta} != predicted {expected_delta} "
            f"({s_sz} -> {d_sz}). STOP: this is not a pure terminator change.")

    n = 0
    with io.open(src, "rb") as fi, io.open(dst, "rb") as fo:
        for a, b in zip(fi, fo):
            n += 1
            abody = a[:-2] if a.endswith(b"\r\n") else (
                a[:-1] if a.endswith(b"\n") else a)
            bbody = b[:-1] if b.endswith(b"\n") else b
            if abody != bbody:
                raise VerifyError(f"line {n}: body bytes differ")
            if b.endswith(b"\r\n") or not b.endswith(b"\n"):
                raise VerifyError(f"line {n}: output terminator is not a bare LF")
            if abody:
                try:
                    ja, jb = json.loads(abody), json.loads(bbody)
                except Exception as e:  # noqa: BLE001
                    raise VerifyError(f"line {n}: not valid JSON after: {e}") from e
                if ja != jb or ja.get("epoch") != jb.get("epoch"):
                    raise VerifyError(f"line {n}: parsed record differs")
        # length equality: both iterators must be exhausted together
        if any(True for _ in fi) or any(True for _ in fo):
            raise VerifyError("line COUNT differs between source and output")
    return {"lines_verified": n, "bytes_before": s_sz, "bytes_after": d_sz,
            "delta": actual_delta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--swap", action="store_true",
                    help="atomically replace the source AFTER verification passes")
    args = ap.parse_args()

    rc = 0
    for raw in args.paths:
        src = Path(raw)
        if not src.exists():
            print(f"!! {src}: missing"); rc = 1; continue
        dst = src.with_suffix(src.suffix + ".normalized")
        print(f"\n=== {src} ===")

        t0 = time.time()
        lines, crlf, lf = census(src)
        expected_delta = crlf          # one byte per CRLF line, exactly
        print(f"  census      lines={lines} CRLF={crlf} LF={lf} "
              f"({time.time()-t0:.1f}s)")
        print(f"  predicted   size drop = {expected_delta} bytes")
        if crlf == 0:
            print("  already uniform LF — nothing to do")
            continue

        t0 = time.time()
        stats = normalize(src, dst)
        print(f"  normalised  -> {dst.name} ({time.time()-t0:.1f}s)")
        if stats["no_terminator"]:
            print(f"  NOTE  {stats['no_terminator']} line(s) had no terminator; "
                  f"each gained one")

        t0 = time.time()
        try:
            v = verify(src, dst, expected_delta)
        except VerifyError as e:
            print(f"  !! VERIFY FAILED: {e}")
            print(f"  !! source untouched; inspect {dst}")
            rc = 1
            continue
        print(f"  verified    {v['lines_verified']} lines, "
              f"{v['bytes_before']} -> {v['bytes_after']} "
              f"(-{v['delta']}, matches prediction) ({time.time()-t0:.1f}s)")

        if args.swap:
            os.replace(dst, src)      # atomic within a filesystem
            print(f"  SWAPPED     {src} now uniform LF")
        else:
            print(f"  not swapped (pass --swap once the canonical hash is "
                  f"confirmed against the normalised copy)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
