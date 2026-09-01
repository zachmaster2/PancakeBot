# staged_captures/

Gap records fetched automatically before OKX's ~171.6-day horizon can take
them, held here until a human merges them into the canonical stores.

**This directory is deliberately OUTSIDE `var/`.** `var/` is gitignored, so
anything under it is absent from git *and* from the backup archive. A staged
capture is the only copy of records that no longer exist upstream — putting
it there would make it a worse single point of failure than the gap it was
created to survive. Here it is committed and pushed with the repo: off
machine, versioned, automatic, no operator action required.

**Do not "tidy" this into `var/`.**

## Contents

- `<store>.staged.jsonl` — append-only, deduped by epoch, **never truncated**.
  There is no rewrite path in the capture code at all; not gated, absent.
- `permanently_absent.json` — epochs past the horizon, recorded once with a
  timestamp and reason so they stop being retried forever.

## Staging defers the dangerous write. It does not remove it.

Merging still calls `store.rewrite()` — a whole-file 4.5 GB replace, the
highest-risk write in the system, and the operation that mangled line
endings in May 2026. **Staged data is a preserved problem, not a solved
one.** A file sitting here means the store still has a hole in it.

To merge, deliberately:

    ALLOW_STORE_REWRITE=1 .venv\Scripts\python.exe scripts\merge_staged.py --dry-run
    ALLOW_STORE_REWRITE=1 .venv\Scripts\python.exe scripts\merge_staged.py --yes

The Desktop marker will keep saying `CAPTURED` until the records are
actually in the store. That is intentional: a successful capture downgrades
urgency, it does not clear the alarm.
