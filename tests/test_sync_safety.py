"""Two guards an unattended daily sync cannot run without.

1. SINGLE-INSTANCE. `run.py --sync` had no guard at all -- the one in
   run.py covers --dry/--live only. On a daily schedule a long sync can
   still be appending when the next day fires. Two syncs read the same
   done-epoch set, fetch the same rounds and append them twice; the
   ascending validators catch it, but as an InvariantError MID-WRITE,
   which is exactly the state that leaves a torn line.

2. TORN TAIL. A process killed mid-append can leave a final line that is
   not valid JSON. Measured behaviour of that state before the fix:

       iter_closed_rounds()  raises InvariantError  (loud, good)
       load_latest_epoch()   raises JSONDecodeError (loud, but it is the
                             RESUME-POINT lookup, so the next sync dies
                             before it can repair anything)
       count_rounds()        returns the torn line as a RECORD (silent
                             disagreement with both of the above)

   So recovery was MANUAL -- unacceptable for an unattended job.

THE LOCK BUG THIS FILE EXISTS TO PIN DOWN. The first version of
process_lock called msvcrt.locking without seeking to 0. That locks a
range at the CURRENT position, and the lock file is opened "a+", which
positions at EOF. The first holder locked byte 0 of an empty file; every
later holder locked a byte past the note the first one had written.
Different ranges, no contention, a lock that never locked. It was not
caught by reasoning about it -- it was caught by running a second sync and
watching it sail through and append to the canonical stores. Hence
test_lock_still_contends_when_the_file_is_not_empty below, which is the
only one of these that would have failed.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data.round_store import (  # noqa: E402
    ClosedRoundsStore,
    repair_torn_tail,
)
from pancakebot.runtime.process_lock import (  # noqa: E402
    LockHeldError,
    exclusive_lock,
)
from pancakebot.types import Bet, Round  # noqa: E402
from pancakebot.util import InvariantError  # noqa: E402


def _round(e: int) -> Round:
    return Round(epoch=e, start_at=1787000000 + e, lock_at=1787000300 + e,
                 lock_price=700.0, close_price=701.0, position="Bull",
                 failed=False,
                 bets=[Bet(wallet_address="0xabc", amount_wei=10**15,
                           position="Bull", created_at=1787000010 + e)])


def _child(lock_path: Path) -> str:
    return textwrap.dedent(f'''
        import sys; sys.path.insert(0, r"{_REPO_ROOT}")
        from pancakebot.runtime.process_lock import exclusive_lock, LockHeldError
        try:
            with exclusive_lock(r"{lock_path}", label="sync"):
                sys.exit(0)          # acquired
        except LockHeldError:
            sys.exit(3)              # correctly refused
    ''')


def _run_child(lock_path: Path) -> int:
    return subprocess.run([sys.executable, "-c", _child(lock_path)],
                          capture_output=True, text=True, timeout=60).returncode


# ---- the lock -------------------------------------------------------------

def test_a_second_holder_is_refused(tmp_path):
    lock = tmp_path / "sync.lock"
    with exclusive_lock(lock, label="sync"):
        assert _run_child(lock) == 3, "a second holder acquired the lock"


def test_the_lock_is_released_on_normal_exit(tmp_path):
    lock = tmp_path / "sync.lock"
    with exclusive_lock(lock, label="sync"):
        pass
    assert _run_child(lock) == 0, "lock not released after a clean exit"


def test_lock_still_contends_when_the_file_is_not_empty(tmp_path):
    """THE regression. The holder writes a pid note, so the file is
    non-empty for every subsequent holder; opening "a+" then positions at
    EOF. Locking at the current position instead of byte 0 made every
    holder after the first lock a DIFFERENT byte and contend with nobody."""
    lock = tmp_path / "sync.lock"
    with exclusive_lock(lock, label="sync"):
        pass
    assert lock.stat().st_size > 0, "expected a non-empty lock file"
    with exclusive_lock(lock, label="sync"):
        assert _run_child(lock) == 3, (
            "lock is ineffective once the lock file has content — every "
            "holder must contend for the SAME byte")


def test_a_dead_holder_leaves_no_stale_lock(tmp_path):
    """The reason this is an OS lock and not a lockfile: no staleness
    question. The kernel drops it when the holder dies, however it dies."""
    lock = tmp_path / "sync.lock"
    killed = _child(lock).replace("sys.exit(0)", "import os; os._exit(0)")
    subprocess.run([sys.executable, "-c", killed], capture_output=True,
                   text=True, timeout=60)
    assert _run_child(lock) == 0, "a dead holder left the lock held"


def test_lock_held_error_is_raised_in_process(tmp_path):
    lock = tmp_path / "sync.lock"
    with exclusive_lock(lock, label="sync"):
        child = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(f'''
                import sys; sys.path.insert(0, r"{_REPO_ROOT}")
                from pancakebot.runtime.process_lock import exclusive_lock
                with exclusive_lock(r"{lock}", label="sync"):
                    pass
            ''')], capture_output=True, text=True, timeout=60)
        assert "LockHeldError" in child.stderr


# ---- the torn tail --------------------------------------------------------

def _store_with_torn_tail(tmp_path) -> tuple[ClosedRoundsStore, Path]:
    p = tmp_path / "torn.jsonl"
    s = ClosedRoundsStore(str(p))
    s.write_new_store([_round(1), _round(2)])
    full = json.dumps(_round(3).to_json(), separators=(",", ":"))
    with io.open(p, "ab") as f:
        f.write(full[: len(full) // 2].encode())     # killed mid-write
    return s, p


def test_a_torn_tail_is_repaired_and_the_store_reads_again(tmp_path):
    s, p = _store_with_torn_tail(tmp_path)
    with pytest.raises(Exception):
        s.load_latest_epoch()                        # resume point unreadable
    removed = repair_torn_tail(str(p))
    assert removed > 0
    assert s.load_latest_epoch() == 2, "repair did not restore the resume point"
    assert [r.epoch for r in s.iter_closed_rounds()] == [1, 2]


def test_repair_is_a_no_op_on_a_healthy_store(tmp_path):
    p = tmp_path / "ok.jsonl"
    s = ClosedRoundsStore(str(p))
    s.write_new_store([_round(1), _round(2)])
    before = p.read_bytes()
    assert repair_torn_tail(str(p)) == 0
    assert p.read_bytes() == before, "repair modified a healthy store"


def test_repair_refuses_when_the_damage_is_not_at_the_tail(tmp_path):
    """Body corruption is NOT a torn write. Silently deleting data to make
    a reader happy is the opposite of what this codebase does."""
    p = tmp_path / "mid.jsonl"
    s = ClosedRoundsStore(str(p))
    s.write_new_store([_round(1), _round(2), _round(3)])
    lines = p.read_bytes().split(b"\n")
    lines[1] = b'{"epoch":2,"broken'                  # corrupt a MIDDLE line
    p.write_bytes(b"\n".join(lines))
    with pytest.raises(InvariantError):
        repair_torn_tail(str(p))
    assert b'{"epoch":2,"broken' in p.read_bytes(), "refused repair still wrote"


def test_repair_handles_an_empty_or_missing_file(tmp_path):
    assert repair_torn_tail(str(tmp_path / "nope.jsonl")) == 0
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    assert repair_torn_tail(str(empty)) == 0


# ---- durability -----------------------------------------------------------

def test_append_rounds_after_flushes_per_record(tmp_path):
    """Without a per-record flush the whole batch sits in a buffer, so a
    kill mid-append can lose complete records AND leave a torn line. With
    it the exposure is one record."""
    src = (_REPO_ROOT / "pancakebot" / "market_data"
           / "round_store.py").read_text(encoding="utf-8")
    body = src[src.index("def append_rounds_after"):]
    body = body[:body.index("\n    def ", 10)]
    assert "f.flush()" in body, "append_rounds does not flush per record"
    assert "os.fsync" in body, "append_rounds does not fsync the batch"
