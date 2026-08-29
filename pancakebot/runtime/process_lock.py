"""An OS-held exclusive lock, for making a job single-instance.

WHY AN OS LOCK AND NOT A LOCKFILE. A lockfile whose presence means "held"
has to answer "is this stale?", and every answer is wrong somewhere: a PID
check races against PID reuse, a timestamp check guesses at how long the
job may legitimately run, and a crashed holder leaves a file that blocks
the next run until a human clears it. An OS-held byte-range lock has no
staleness question at all -- the kernel drops it when the holding process
dies, however it dies. Verified on Windows: a second holder gets
PermissionError while the first holds, and acquires immediately once the
first exits.

WHAT THIS IS FOR. ``run.py --sync`` had NO single-instance guard: the one
in run.py covers ``--dry``/``--live`` only. On an unattended daily
schedule a long sync can still be running when the next day fires, and two
syncs appending to the same append-only stores is a corruption path -- and
a quiet one, because both processes read the same "done epochs" set, fetch
the same rounds, and append them twice. The strictly-ascending validators
would likely catch it, but as an InvariantError mid-write, which is
exactly the state that leaves a torn line.

Not reentrant, and deliberately so: this exists to answer "is another
process doing this right now", and a process asking about itself is a bug.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


class LockHeldError(RuntimeError):
    """Another process holds the lock. The caller should exit, not wait."""


if os.name == "nt":
    import msvcrt

    def _try_lock(fh) -> bool:
        # SEEK TO 0 FIRST, and this line is the whole lock. msvcrt.locking
        # locks a byte range starting at the CURRENT file position, and this
        # file is opened "a+", which positions at EOF. Without the seek, the
        # first holder locks byte 0 of an empty file and every later holder
        # locks a byte past the note the first one wrote -- different ranges,
        # no contention, a lock that silently never locks. Measured: a second
        # `run.py --sync` sailed straight through and appended to the
        # canonical stores. Every holder must contend for the SAME byte.
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _try_lock(fh) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


@contextmanager
def exclusive_lock(path: str | os.PathLike, *, label: str = "job"):
    """Hold an exclusive lock on ``path`` for the duration of the block.

    Raises ``LockHeldError`` immediately if another process holds it --
    never blocks. A daily job that finds yesterday's run still going should
    say so and exit, not queue up behind it and double the overlap.

    The lock file itself is an artifact, not the lock. It is left on disk
    on purpose: deleting it would race another process that has it open,
    and its presence means nothing.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a+")
    try:
        if not _try_lock(fh):
            raise LockHeldError(
                f"another {label} is already running (lock held on {p}). "
                f"Exiting rather than running concurrently."
            )
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(f"pid={os.getpid()} label={label}\n")
            fh.flush()
        except OSError:
            pass          # the lock is the contract; the note is a courtesy
        try:
            yield
        finally:
            _unlock(fh)
    finally:
        fh.close()
