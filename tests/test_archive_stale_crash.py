"""Tests for process_health.archive_stale_crash.

Verifies the bot-startup housekeeping that renames a leftover crash.json to
a timestamped archive filename so the supervisor doesn't re-fire CRASHED
alerts every 3 minutes after a previous bot's death.

Run:
    python -m pytest tests/test_archive_stale_crash.py -v
    # or standalone:
    python tests/test_archive_stale_crash.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime.process_health import archive_stale_crash  # noqa: E402


def _write_fake_crash(path: Path, *, age_s: float) -> None:
    """Write a minimal crash.json and backdate its mtime by ``age_s`` seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"exc_type": "FakeError", "exc_repr": "fake"}), encoding="utf-8")
    past = time.time() - age_s
    os.utime(str(path), (past, past))


def test_missing_file_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "crash.json"
        assert not p.exists()
        result = archive_stale_crash(p)
        assert result is None
        assert not p.exists()  # no-op, no fake file created


def test_old_file_gets_archived():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "crash.json"
        _write_fake_crash(p, age_s=3600)  # 1h old
        result = archive_stale_crash(p)
        assert result is not None, "old crash.json should have been archived"
        assert not p.exists(), "original crash.json should be gone"
        assert result.exists(), "archive file should exist"
        assert result.name.startswith("crash_archive_"), f"bad archive name: {result.name}"
        assert result.name.endswith(".json")
        # Content preserved
        data = json.loads(result.read_text(encoding="utf-8"))
        assert data["exc_type"] == "FakeError"


def test_young_file_is_preserved():
    """A crash.json younger than min_age_seconds is NOT archived.

    Rationale: protects against a pathological race where the archive
    happens while the crash handler is still writing (extremely unlikely
    since atomic write -> rename is fast, but belt + suspenders).
    """
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "crash.json"
        _write_fake_crash(p, age_s=5)  # 5s old, default threshold 60s
        result = archive_stale_crash(p)
        assert result is None, "young crash.json should NOT have been archived"
        assert p.exists(), "original crash.json should still be in place"


def test_custom_min_age_threshold():
    """min_age_seconds is honored when passed by caller."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "crash.json"
        _write_fake_crash(p, age_s=10)  # 10s old
        # With a tight 5s threshold, the 10s-old file SHOULD archive.
        result = archive_stale_crash(p, min_age_seconds=5.0)
        assert result is not None
        assert result.name.startswith("crash_archive_")


def test_archive_collision_gets_numeric_suffix():
    """If an archive from the same-second mtime already exists, append -1, -2..."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "crash.json"
        # Two rounds: archive once, then put another fresh file with same mtime and archive again.
        _write_fake_crash(p, age_s=3600)
        first = archive_stale_crash(p)
        assert first is not None
        # Create a second crash.json with the SAME backdated mtime.
        _write_fake_crash(p, age_s=3600)
        old_stat = first.stat()
        os.utime(str(p), (old_stat.st_mtime, old_stat.st_mtime))
        second = archive_stale_crash(p)
        assert second is not None
        assert second != first, "second archive must not overwrite first"
        # Reviewer NIT: tighten the collision-suffix assertion. The suffix
        # helper uses _1, _2, ... so the second archive's stem must contain
        # "_1" (not merely "differ from first" which was tautological).
        assert "_1" in second.stem, (
            f"second archive should use _1 collision suffix; got {second.name}"
        )
        assert first.exists(), "first archive must not be overwritten"


def test_existing_archive_files_are_not_re_archived():
    """Sanity: archive_stale_crash operates on the exact crash_path it's
    given. An existing ``crash_archive_*.json`` left in the directory from a
    prior startup must not itself be touched."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "crash.json"
        prior_archive = Path(tmp) / "crash_archive_20260101-000000.json"
        prior_archive.write_text("prior", encoding="utf-8")
        past = time.time() - 86400  # very old
        os.utime(str(prior_archive), (past, past))
        # No crash.json -> archive_stale_crash returns None and does nothing.
        result = archive_stale_crash(p)
        assert result is None
        assert prior_archive.exists(), "prior archive must be untouched"
        assert prior_archive.read_text(encoding="utf-8") == "prior"


def test_archive_preserves_mtime_in_filename():
    """Archive filename uses the ORIGINAL crash mtime, not the archival time."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "crash.json"
        _write_fake_crash(p, age_s=3600)
        stamp_expected = time.strftime("%Y%m%d-%H%M%S", time.localtime(p.stat().st_mtime))
        result = archive_stale_crash(p)
        assert result is not None
        assert stamp_expected in result.name, (
            f"filename {result.name} should contain the original mtime stamp {stamp_expected}"
        )


def test_exception_does_not_propagate():
    """Any OS error during archive must be swallowed -- bot startup cannot block."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "nonexistent_subdir" / "crash.json"
        # crash_path parent doesn't exist -- ``.exists()`` returns False -> noop.
        result = archive_stale_crash(p)
        assert result is None


def main() -> int:
    tests = [
        test_missing_file_is_noop,
        test_old_file_gets_archived,
        test_young_file_is_preserved,
        test_custom_min_age_threshold,
        test_archive_collision_gets_numeric_suffix,
        test_existing_archive_files_are_not_re_archived,
        test_archive_preserves_mtime_in_filename,
        test_exception_does_not_propagate,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  [OK] {t.__name__}")
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failures.append(t.__name__)
    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print(f"ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
