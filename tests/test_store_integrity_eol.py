"""The EOL leg must describe the file, not a shape it assumed.

/tmp/vm_preflight.py -- the checker used as the integrity gate for the
2026-08-24 store repair -- sampled the first 1 MB and a 1 MB window near
the end and reported "uniform" whenever the two agreed. Reproduced against
synthetic files, it gets these wrong:

  * LF, CRLF block, LF        -> "uniform LF"   (THE REAL eth/sol shape;
                                 17,067 CRLF lines per file, invisible)
  * every line alternating    -> "uniform LF"   (both samples mixed ->
                                 region() returns None for each ->
                                 None == None takes the uniform branch,
                                 and None is falsy so it prints "LF")
  * LF block then CRLF        -> "CRLF->LF @byte 0" (direction inverted,
                                 location meaningless)
  * alternating blocks        -> one invented CRLF->LF transition

The second is the one that matters most: a file it could not classify at
all is reported as clean. That is the same failure shape as the two other
incidents this week -- a checker returning a confident clean answer about
something that is not clean.

Every case below is built with terminators chosen by construction, so the
expected answer is known independently of the code under test.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from store_integrity import describe_eol, eol_runs, scan  # noqa: E402


def _build(tmp_path, spec, name="s.jsonl"):
    """spec: [(terminator_bytes, n_lines)] -> a file with exactly that shape."""
    p = tmp_path / name
    with io.open(p, "wb") as f:
        for i, (term, n) in enumerate(spec):
            for j in range(n):
                f.write(b'{"epoch":%d}' % (i * 1000 + j + 1) + term)
    return p


def _kinds(p):
    return [k for k, _, _ in eol_runs(p)]


# ---- the shapes vm_preflight got wrong -----------------------------------

def test_the_real_eth_sol_shape_is_not_called_uniform(tmp_path):
    """THE regression. Both of vm_preflight's sample windows land in the
    leading and trailing LF runs; the CRLF block between them is never
    read. It reported uniform LF for a file with 17,067 CRLF lines."""
    p = _build(tmp_path, [(b"\n", 40), (b"\r\n", 17), (b"\n", 15)])
    runs = eol_runs(p)
    assert _kinds(p) == ["LF", "CRLF", "LF"]
    assert [c for _, _, c in runs] == [40, 17, 15]
    assert "uniform" not in describe_eol(runs)


def test_a_fully_alternating_file_is_not_called_uniform(tmp_path):
    """vm_preflight reports 'uniform LF' here because BOTH samples come
    back None (mixed), None == None takes the uniform branch, and None is
    falsy. 'I cannot tell' becomes 'it is clean'."""
    p = _build(tmp_path, [(b"\n", 1), (b"\r\n", 1)] * 30)
    assert "uniform" not in describe_eol(eol_runs(p))
    assert len(eol_runs(p)) == 60


def test_crlf_block_first(tmp_path):
    p = _build(tmp_path, [(b"\r\n", 20), (b"\n", 30)])
    assert _kinds(p) == ["CRLF", "LF"]


def test_lf_block_first_is_reported_in_the_right_direction(tmp_path):
    """vm_preflight hardcodes the label 'CRLF->LF' and a search that
    assumes CRLF precedes LF, so it inverts this case."""
    p = _build(tmp_path, [(b"\n", 30), (b"\r\n", 20)])
    assert _kinds(p) == ["LF", "CRLF"], "direction must follow the file"


def test_alternating_blocks_report_every_run(tmp_path):
    p = _build(tmp_path, [(b"\n", 5), (b"\r\n", 5), (b"\n", 5), (b"\r\n", 5)])
    assert _kinds(p) == ["LF", "CRLF", "LF", "CRLF"]


# ---- edges -------------------------------------------------------------

def test_single_line_file(tmp_path):
    p = _build(tmp_path, [(b"\n", 1)])
    assert eol_runs(p) == [("LF", 1, 1)]
    assert describe_eol(eol_runs(p)) == "uniform LF"


def test_file_with_no_trailing_newline(tmp_path):
    p = tmp_path / "notrail.jsonl"
    p.write_bytes(b'{"epoch":1}\n{"epoch":2}')
    assert _kinds(p) == ["LF", "NONE"], "a missing final terminator must show"


def test_single_line_with_no_terminator_at_all(tmp_path):
    p = tmp_path / "one.jsonl"
    p.write_bytes(b'{"epoch":1}')
    assert eol_runs(p) == [("NONE", 1, 1)]


def test_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_bytes(b"")
    assert eol_runs(p) == []
    assert describe_eol(eol_runs(p)) == "empty"


def test_genuinely_uniform_files_are_still_called_uniform(tmp_path):
    """The fix must not cry wolf: a real single-terminator file has to
    report as uniform or the signal is worthless."""
    for term, want in ((b"\n", "uniform LF"), (b"\r\n", "uniform CRLF")):
        p = _build(tmp_path, [(term, 50)], name=f"u{len(want)}.jsonl")
        assert describe_eol(eol_runs(p)) == want


# ---- the other legs, which vm_preflight got right ----------------------

def test_the_full_scan_legs_still_work(tmp_path):
    """bytes / records / first / last / gaps / missing were full scans in
    vm_preflight and are kept. Only the EOL leg was sampled."""
    p = tmp_path / "gapped.jsonl"
    p.write_bytes(b"".join(b'{"epoch":%d}\n' % e for e in (10, 11, 14, 15, 20)))
    r = scan(p)
    assert r["records"] == 5
    assert r["first"] == 10 and r["last"] == 20
    assert r["missing"] == 6          # 12,13,16,17,18,19
    assert r["gaps"] == 2             # two runs: 12-13 and 16-19
    assert r["ascending"] is True


def test_non_ascending_epochs_are_flagged(tmp_path):
    p = tmp_path / "unsorted.jsonl"
    p.write_bytes(b'{"epoch":5}\n{"epoch":3}\n')
    assert scan(p)["ascending"] is False
