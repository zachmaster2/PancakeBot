"""Cycle-audit CSV rotate-on-header-mismatch behavior.

Added 2026-05-12 alongside the per-symbol kline-fetch-timing columns.
Verifies:

1. When the existing file's header matches the current schema, no
   rotate happens.
2. When the existing file's header doesn't match, the file gets
   renamed with a timestamped suffix and a fresh file is written.
3. Malformed existing CSV (truncated / non-UTF / partial-write
   crash artifact) doesn't crash startup — the rotate path absorbs it.

Y6 reviewer coverage requirement.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime.audit import (  # noqa: E402
    ensure_cycle_audit_csv,
    _CYCLE_AUDIT_HEADER_OK_PATHS,
)


@pytest.fixture(autouse=True)
def _clear_header_cache():
    """The audit module caches header-validated paths in a process-local
    set. Each test must start with an empty cache so the rotate logic
    actually runs."""
    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()
    yield
    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()


def test_no_rotate_when_existing_header_matches(tmp_path: Path) -> None:
    """If the CSV already exists with a matching header, ensure_cycle_audit_csv
    is a no-op (no new file, no rotated file)."""
    csv_path = tmp_path / "cycle_audit.csv"
    # Write the current schema explicitly so the existing file is "matched".
    expected_header = ensure_cycle_audit_csv(str(csv_path))
    assert csv_path.exists()

    # Append a sentinel data row so we can verify the file isn't recreated.
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentinel"] + [""] * (len(expected_header) - 1))

    # Clear cache and re-call to force a fresh validation.
    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()
    ensure_cycle_audit_csv(str(csv_path))

    # File should still contain the sentinel row.
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    assert rows[0] == expected_header
    assert rows[1][0] == "sentinel"
    # No rotated sibling created.
    rotates = list(tmp_path.glob("cycle_audit.csv.pre-rotate-*"))
    assert rotates == []


def test_rotate_on_header_mismatch(tmp_path: Path) -> None:
    """If the CSV exists with a stale header (different columns), it
    gets renamed with a timestamped suffix and a fresh file is written
    with the current schema."""
    csv_path = tmp_path / "cycle_audit.csv"
    # Write a STALE header that doesn't match current.
    stale_header = ["cycle_ts", "epoch", "old_column_long_gone"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(stale_header)
        w.writerow(["1234567890", "100", "stale_value"])

    expected_header = ensure_cycle_audit_csv(str(csv_path))

    # Active file now has the new schema.
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == expected_header
    assert len(rows) == 1  # only header — no stale data leaked

    # Old file preserved with timestamped suffix.
    rotates = list(tmp_path.glob("cycle_audit.csv.pre-rotate-*"))
    assert len(rotates) == 1
    with open(rotates[0], newline="") as f:
        old_rows = list(csv.reader(f))
    assert old_rows[0] == stale_header
    assert old_rows[1][2] == "stale_value"


def test_rotate_absorbs_malformed_csv(tmp_path: Path) -> None:
    """A partially-corrupted CSV (e.g. truncated mid-row from a prior
    crash) must not crash ensure_cycle_audit_csv. The rotate path treats
    'no usable header' the same as 'mismatched header'.
    """
    csv_path = tmp_path / "cycle_audit.csv"
    # Write binary garbage that csv.reader will likely choke on.
    csv_path.write_bytes(b"\x00\x01\x02\xff garbage \"unbalanced\nquote\x00")

    # Should NOT raise.
    expected_header = ensure_cycle_audit_csv(str(csv_path))

    # Active file has the new schema.
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == expected_header
    # Old (garbage) file preserved.
    rotates = list(tmp_path.glob("cycle_audit.csv.pre-rotate-*"))
    assert len(rotates) == 1
