"""Store gaps must be impossible or loud -- never silent.

REIMPLEMENTED FRESH, not cherry-picked. The behaviours below were approved
on their merits; the rejected design they originally shipped in is absent
by construction rather than by anyone remembering to leave it out. In
particular there is NO known-unusable ledger here and no round-page
classification -- those were not approved.

THE HISTORY. The round store held 23 missing epochs. All of them sat BELOW
the 35,000-round tail that the sync asserts on, so they were invisible by
construction for months. Nothing was broken; nothing was reported; the
absence of a signal read as health.

P1 -- CLOSURE. The old partition computed prepend (< earliest) and append
(> latest) and never named what sits between. An epoch inside the on-disk
range but absent from it fell into neither list and was dropped with no log
line, no error, and a total_to_fetch that under-reported it. The fix is
asserted as TOTALITY rather than as a test for the interior case, because
totality also catches partition shapes nobody has thought of yet.

P2 -- BATCH COMPLETENESS. `_fetch_and_append` filtered results to
`> prev_epoch` under a "skip gaps" comment. That comment encoded a false
assumption, and the filter is how a transient upstream omission became a
permanent hole in an append-only store.

P4/P5 -- WHOLE-STORE REPORT by byte scan, because a ~2 minute JSON parse is
why no such report ran routinely, and a report nobody runs is not a report.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data.epoch_scan import (  # noqa: E402
    contiguity,
    format_report,
    gap_runs,
    scan_epochs,
)

_SYNC = Path(_REPO_ROOT / "pancakebot" / "market_data" / "sync.py")


def _sync_src() -> str:
    return io.open(_SYNC, encoding="utf-8").read()


# ---- P1: closure -----------------------------------------------------------

def test_the_partition_is_asserted_total_not_merely_three_way():
    """The closure check is the load-bearing half.

    Adding an interior bucket without asserting totality would fix the one
    shape we know about and leave the next one silent.
    """
    src = _sync_src()
    assert "kline_partition_not_exhaustive" in src, (
        "no closure assertion -- an epoch in no bucket is dropped silently")
    i = src.index("kline_partition_not_exhaustive")
    window = src[max(0, i - 700):i]
    assert "interior_rounds" in window
    assert "!= len(remaining)" in window, (
        "closure must be arithmetic over the whole input, not a check for "
        "the interior case specifically")


def test_the_interior_bucket_exists_and_is_reported():
    src = _sync_src()
    assert "interior_rounds" in src
    assert "INTERIOR GAP" in src, "an interior gap that logs nothing is the bug"


def test_an_interior_gap_does_not_stop_forward_collection():
    """DELIBERATE DEVIATION, recorded here so it is visible rather than
    discovered. The original design raised on an interior gap. Raising
    would stop an unattended daily job permanently on a condition that does
    not block forward progress -- recreating the silent permanent stall the
    work exists to prevent."""
    src = _sync_src()
    i = src.index("INTERIOR GAP")
    window = src[max(0, i - 400):i + 700]
    assert "warn(" in window
    assert "raise" not in window, (
        "an interior gap must not abort the run; forward collection "
        "continues and the gap is reported every run until a human acts")


def test_closure_arithmetic_catches_an_epoch_in_no_bucket():
    """The property itself, independent of the source text."""
    earliest, latest = 100, 200
    remaining = [50, 150, 250]          # older, INTERIOR, newer
    prepend = [e for e in remaining if e < earliest]
    append = [e for e in remaining if e > latest]
    interior = [e for e in remaining if earliest <= e <= latest]
    assert len(prepend) + len(append) + len(interior) == len(remaining)

    # The OLD two-way split loses the interior epoch and the arithmetic says so.
    assert len(prepend) + len(append) == len(remaining) - 1


# ---- P2: batch completeness ------------------------------------------------

def test_a_short_batch_is_a_gap_not_a_result():
    src = _sync_src()
    assert "kline_batch_incomplete" in src, (
        "a fetch returning fewer records than requested is silently accepted")
    i = src.index("kline_batch_incomplete")
    window = src[max(0, i - 900):i]
    assert "requested_epochs" in window and "got_epochs" in window, (
        "completeness must be a set comparison against what was requested")


def test_the_skip_gaps_filter_is_gone():
    """THE regression. The filter is how a transient omission became a
    permanent hole."""
    src = _sync_src()
    assert "# Filter to only records strictly after prev_epoch (skip gaps)." not in src
    assert "kline_batch_not_after_cursor" in src, (
        "monotonicity must survive as an assertion, not as a filter")


def test_monotonicity_is_an_assertion_not_a_filter():
    src = _sync_src()
    i = src.index("kline_batch_not_after_cursor")
    window = src[max(0, i - 600):i]
    assert "_offenders" in window
    assert "appendable = results" in src, (
        "appendable must be the full fetched set once verified, not a "
        "filtered subset")


# ---- P4/P5: the byte-scan report -------------------------------------------

def test_scan_reads_epochs_without_parsing_json(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_bytes(
        b'{"epoch":10,"a":1}\n'
        b'{"epoch": 11 ,"a":2}\n'          # spaced separators still scan
        b'{"epoch":12,"junk":"not valid json at all\n'   # body malformed
    )
    assert scan_epochs(str(p)) == [10, 11, 12]


def test_scan_of_a_missing_file_is_empty_not_an_error():
    assert scan_epochs("does/not/exist.jsonl") == []


def test_gap_runs_finds_interior_runs_only():
    assert gap_runs([1, 2, 3]) == []
    assert gap_runs([1, 4]) == [(2, 3)]
    assert gap_runs([1, 2, 5, 6, 9]) == [(3, 4), (7, 8)]
    assert gap_runs([]) == []


def test_gap_runs_is_not_confused_by_duplicates_or_disorder():
    """Duplicates and out-of-order records are a different defect and must
    not manufacture phantom gaps in the contiguity count."""
    assert gap_runs([3, 1, 2, 3, 1]) == []


def test_contiguity_separates_gaps_from_duplicates_and_disorder(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_bytes(b"".join(
        b'{"epoch":%d}\n' % e for e in (1, 2, 2, 5, 4)
    ))
    r = contiguity(str(p))
    assert r["n"] == 5
    assert r["distinct"] == 4
    assert r["earliest"] == 1 and r["latest"] == 5
    assert r["missing"] == 1               # epoch 3
    assert r["runs"] == [(3, 3)]
    assert r["duplicates"] == 1
    assert r["out_of_order"] >= 1


def test_a_clean_store_produces_a_POSITIVE_line(tmp_path):
    """Silence is what let the gaps hide. An empty report and a healthy
    report must not look alike."""
    p = tmp_path / "s.jsonl"
    p.write_bytes(b"".join(b'{"epoch":%d}\n' % e for e in range(1, 6)))
    lines = format_report([contiguity(str(p))])
    assert any("contiguous" in l for l in lines)


def test_the_report_names_each_gap_run(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_bytes(b"".join(b'{"epoch":%d}\n' % e for e in (1, 5)))
    lines = format_report([contiguity(str(p))])
    assert any("MISSING" in l for l in lines)
    assert any("gap 2..4" in l for l in lines)


def test_an_absent_store_is_reported_not_skipped(tmp_path):
    lines = format_report([contiguity(str(tmp_path / "nope.jsonl"))])
    assert any("ABSENT" in l for l in lines)


def test_the_report_never_breaks_the_sync():
    """A report that can fail the run it reports on is worse than none."""
    src = io.open(Path(_REPO_ROOT / "pancakebot" / "app.py"),
                  encoding="utf-8").read()
    i = src.index("contiguity report failed")
    window = src[max(0, i - 700):i]
    assert "except Exception" in window, (
        "the contiguity report must not be able to fail the sync")


def test_the_report_covers_all_five_stores():
    src = io.open(Path(_REPO_ROOT / "pancakebot" / "app.py"),
                  encoding="utf-8").read()
    i = src.index("contiguity(p) for p in")
    window = src[i:i + 500]
    for name in ("CLOSED_ROUNDS_PATH", "BNB_SPOT_PRICES_PATH",
                 "BTC_SPOT_PRICES_PATH", "ETH_SPOT_PRICES_PATH",
                 "SOL_SPOT_PRICES_PATH"):
        assert name in window, f"the report omits {name}"


# ---- what was NOT built ----------------------------------------------------

def test_the_unapproved_parts_are_absent_by_construction():
    """P3 and the known-unusable ledger were explicitly not approved. This
    asserts they did not arrive by way of a cherry-pick."""
    assert not (_REPO_ROOT / "pancakebot" / "market_data"
                / "unusable_ledger.py").exists()
    src = _sync_src()
    assert "unusable" not in src.lower()
    assert "classify_page_gap" not in src
