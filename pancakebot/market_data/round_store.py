"""Epoch-ascending JSONL store for closed rounds with append and prepend operations."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Sequence

from pancakebot.log import warn
from pancakebot.types import Round
from pancakebot.util import InvariantError


class ClosedRoundsStore:
    """Closed-round JSONL store (epoch-ascending).

    Format:
      - Flat JSON Lines: one round object per line.
      - epoch is a top-level key.

    Store invariant:
      - epochs are strictly increasing on disk.

    Note:
      - This store enforces ordering only. Usable-round invariants are enforced at ingestion.
    """

    def __init__(self, path_jsonl: str):
        if not path_jsonl:
            raise InvariantError("ClosedRoundsStore_requires_path")
        self._path = path_jsonl

    @property
    def path_jsonl(self) -> str:
        return self._path

    def exists(self) -> bool:
        return os.path.exists(self._path)

    # --------------------------
    # Reads
    # --------------------------

    def iter_closed_rounds(self) -> Iterator[Round]:
        if not self.exists():
            return iter(())
        return self._iter_impl()

    def _iter_impl(self) -> Iterator[Round]:
        with open(self._path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise InvariantError(f"jsonl_parse_failed: line={line_no} err={e}") from e
                yield Round.from_json(obj)

    def load_earliest_epoch(self) -> int | None:
        for r in self.iter_closed_rounds():
            return r.epoch
        return None

    def load_latest_epoch(self) -> int | None:
        if not self.exists():
            return None
        latest: int | None = None
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                latest = int(obj["epoch"])
        return latest

    def count_rounds(self) -> int:
        if not self.exists():
            return 0
        n = 0
        with open(self._path, "r") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    # --------------------------
    # Writes
    # --------------------------

    def write_new_store(self, rounds_asc: Sequence[Round]) -> None:
        """Create a new store file from scratch in epoch-ascending order."""
        if self.exists():
            raise InvariantError("write_new_store_store_already_exists")
        if not rounds_asc:
            raise InvariantError("write_new_store_empty")

        prev: int | None = None
        for idx, r in enumerate(rounds_asc):
            if prev is not None and r.epoch <= prev:
                raise InvariantError(f"new_store_not_strictly_increasing: idx={idx} got={r.epoch} prev={prev}")
            prev = r.epoch

        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8", newline="") as f:
            for r in rounds_asc:
                f.write(json.dumps(r.to_json(), separators=(",", ":")) + "\n")

    def append_rounds(self, rounds_asc: Sequence[Round]) -> None:
        """Append rounds to the end of the file.

        rounds_asc must be strictly increasing and start strictly after current latest epoch.

        NOTE: This method reads disk to discover the current latest epoch. In the live loop,
        prefer append_rounds_after(prev_epoch, ...), which avoids disk reads.
        """
        if not rounds_asc:
            return
        if not self.exists():
            raise InvariantError("append_requires_existing_store")

        latest = self.load_latest_epoch()
        if latest is None:
            raise InvariantError("store_latest_epoch_missing")
        self.append_rounds_after(latest, rounds_asc)

    def append_rounds_after(self, prev_epoch: int, rounds_asc: Sequence[Round]) -> int:
        """Append rounds without reading the file.

        Contract:
          - `prev_epoch` must be the current latest epoch already on disk.
          - `rounds_asc` must be strictly increasing and start strictly after `prev_epoch`.

        Returns:
          - the latest epoch after appending.
        """
        if not rounds_asc:
            return prev_epoch
        prev = prev_epoch
        for idx, r in enumerate(rounds_asc):
            if r.epoch <= prev:
                raise InvariantError(f"append_not_strictly_increasing: idx={idx} got={r.epoch} prev={prev}")
            prev = r.epoch

        # Use r+ so the file must already exist.
        # Flush per record and fsync once at the end. Without the
        # flush the whole batch sits in a buffer, so a kill mid-append
        # can lose complete records AND leave a torn line; with it the
        # window narrows to a single record. The fsync makes the batch
        # durable against power loss, not just process death.
        with open(self._path, "r+", encoding="utf-8", newline="") as f:
            f.seek(0, os.SEEK_END)
            for r in rounds_asc:
                f.write(json.dumps(r.to_json(), separators=(",", ":")) + "\n")
                f.flush()
            os.fsync(f.fileno())
        return prev

    def replace_with_prepended_chunk(self, chunk_asc: Sequence[Round], *, replace_path: str) -> None:
        """Prepend a strictly-older chunk by atomic replacement."""
        if not self.exists():
            raise InvariantError("replace_with_prepended_chunk_requires_existing_store")
        if not chunk_asc:
            return

        earliest = self.load_earliest_epoch()
        if earliest is None:
            raise InvariantError("store_earliest_epoch_missing")

        prev: int | None = None
        for idx, r in enumerate(chunk_asc):
            if prev is not None and r.epoch <= prev:
                raise InvariantError(f"prepend_chunk_not_increasing: idx={idx} got={r.epoch} prev={prev}")
            prev = r.epoch

        if prev is None or prev >= earliest:
            raise InvariantError("prepend_chunk_not_strictly_older_than_store")

        os.makedirs(os.path.dirname(replace_path) or ".", exist_ok=True)
        with open(replace_path, "w", encoding="utf-8", newline="") as out:
            for r in chunk_asc:
                out.write(json.dumps(r.to_json(), separators=(",", ":")) + "\n")
            with open(self._path, "r") as existing:
                for line in existing:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")

        os.replace(replace_path, self._path)

    # JSON codecs live on Round/Bet in pancakebot.types.

def repair_torn_tail(path: str) -> int:
    """Truncate a TRAILING partial line, if and only if there is one.

    A process killed mid-append (sleep, reboot, power) can leave a final
    line that is not valid JSON. Measured behaviour of that state:

        iter_closed_rounds()  raises InvariantError jsonl_parse_failed
        load_latest_epoch()   raises JSONDecodeError
        count_rounds()        returns the torn line as a RECORD

    The first two are the good failure -- loud, no silent corruption. But
    load_latest_epoch is how a sync finds its resume point, so the job dies
    before it can fix anything, and count_rounds quietly disagrees with
    both. Left alone that means an unattended daily job stops permanently
    on the first torn write and waits for a human to truncate a line.

    So: repair, but ONLY the tail, and only when the tail alone is bad.
      * If the file is empty or every line parses -> return 0, touch nothing.
      * If the LAST line does not parse and every earlier line does ->
        truncate exactly that line and return the bytes removed.
      * If any NON-final line does not parse -> raise. That is not a torn
        write, it is corruption in the body of the store, and silently
        deleting data to make a reader happy is the opposite of what this
        codebase does everywhere else.

    Returns the number of bytes truncated (0 if nothing was wrong).
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0

    # Walk in binary, tracking the byte offset each line starts at, so the
    # truncation point is exact and no decoding step can shift it.
    bad_offset = None
    offset = 0
    n = 0
    with open(p, "rb") as f:
        for raw in f:
            n += 1
            start = offset
            offset += len(raw)
            body = raw.strip()
            if not body:
                continue
            try:
                json.loads(body)
            except ValueError:
                if bad_offset is not None:
                    raise InvariantError(
                        f"jsonl_multiple_unparsable_lines: {path} line={n} "
                        f"-- body corruption, not a torn tail; refusing to "
                        f"truncate")
                bad_offset = (start, n)

    if bad_offset is None:
        return 0
    start, line_no = bad_offset
    if start + (offset - start) != offset or line_no != n:
        raise InvariantError(
            f"jsonl_unparsable_line_not_at_tail: {path} line={line_no} of "
            f"{n} -- body corruption, not a torn tail; refusing to truncate")

    removed = offset - start
    with open(p, "r+b") as f:
        f.truncate(start)
        f.flush()
        os.fsync(f.fileno())
    warn("STORE", f"repaired torn tail in {path}: dropped {removed} bytes "
                  f"(partial line {line_no}); a previous run was killed "
                  f"mid-append")
    return removed
