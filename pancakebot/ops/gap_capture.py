"""Capture gap records before the horizon takes them. Do NOT integrate them.

THE PROBLEM. A gap can be detected, still be fetchable, and be lost anyway,
because detection only helps if somebody looks. The posture this system now
runs in assumes months of nobody looking. A gap opening in November and
noticed in March would be reported accurately every single day and still be
permanently gone by the time anyone read the report.

THE DECOMPOSITION. Capture is time-critical and bounded by a hard external
deadline: OKX serves 1s klines for ~171.6 days and then the data ceases to
exist anywhere. Integration is risky but has NO deadline at all. Coupling
them forces the risky operation to inherit the deadline. So they are split:

    CAPTURE      automatic, append-only, never touches a canonical store
    INTEGRATION  deliberate, human-approved, gated, verified before and after

STAGING DEFERS THE DANGEROUS WRITE. IT DOES NOT REMOVE IT.
Merging staged records into a store still calls ``store.rewrite()`` -- a
whole-file 4.5 GB replace, the highest-risk write in the system, and the
operation that mangled line endings in May 2026. Staged data is NOT a
solved problem; it is a preserved one. Anyone reading this months from now
should understand that the merge is still ahead of them, and that the
staged file is a holding pen, not a repair.

WHY ONLY KLINES ARE TIME-CRITICAL. Closed rounds come from The Graph, which
the recorded research describes as contiguous with no horizon -- a missing
round can be refetched at any time, so there is no clock on it. The 171.6
day pressure is entirely on the four OKX kline stores. A closed_rounds gap
is still REPORTED, and it also BLOCKS kline capture for the same epochs,
because a kline fetch window is anchored to that round's lock_at and cannot
be built without it.

WHY CAPTURE MUST NOT USE THE SYNC'S WORKING SET. ``_sync_1s_klines`` only
ever sees ``rounds_all[-35000:]`` -- 124.0 days. OKX serves 171.6. The 47.6
day band between them, where a gap is detectable and fetchable but never
looked at, is the ENTIRE REASON this module exists. Capture is therefore
driven from the whole-store integrity report's gap list and never from the
sync's tail. A version of this that inherited the tail would address
nothing and would be an elaborate no-op.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

# OKX 1s kline retention, measured and recorded 2026. Epochs older than this
# cannot be fetched by anyone, so attempting them is pure waste and -- worse
# -- an indefinite daily failure that trains an operator to ignore the
# alarm. They are promoted to permanently-absent instead.
OKX_KLINE_HORIZON_DAYS = 171.6

# DELIBERATELY OUTSIDE var/. var/ is gitignored, so anything under it is
# absent from git AND from the 617MB backup archive. A staged capture is the
# ONLY copy of records that no longer exist upstream -- OKX serves 1s klines
# for ~171.6 days -- so storing it there would make it a worse single point
# of failure than the gap it was created to survive.
#
# At the repo root it is committed and pushed with everything else: off
# machine, versioned, automatic, and requiring no action from the operator.
# The files are small; only records that are not yet merged ever live here,
# and a realistic interior gap is tens of epochs.
#
# Do NOT "tidy" these into var/.
STAGING_DIR = "staged_captures"
PERMANENT_PATH = "staged_captures/permanently_absent.json"

# Bound the work a single run will attempt, so one enormous gap cannot turn
# into an hours-long unattended fetch storm against OKX. The remainder is
# picked up on the next run; the horizon is measured in months, so a cap of
# a few hundred per day still closes any realistic gap with room to spare.
MAX_CAPTURE_PER_RUN = 300


@dataclass
class StorePlan:
    """What to do about one store's gap, decided without any network I/O."""

    store: str
    fetchable: list[int] = field(default_factory=list)
    unrecoverable: list[int] = field(default_factory=list)
    already_staged: list[int] = field(default_factory=list)
    blocked: list[int] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return bool(self.fetchable)


def age_days(start_at: int, now: float) -> float:
    return (now - float(start_at)) / 86400.0


def plan_capture(
    *,
    store: str,
    missing: list[int],
    round_start_at: dict[int, int],
    staged: set[int],
    now: float | None = None,
    horizon_days: float = OKX_KLINE_HORIZON_DAYS,
    max_per_run: int = MAX_CAPTURE_PER_RUN,
) -> StorePlan:
    """Classify a store's missing epochs. PURE -- no I/O, no network.

    ``missing`` comes from the whole-store integrity report, NOT from the
    sync's working set. That distinction is the point of the module.

    Four outcomes:
      already_staged -- captured on a previous run; nothing to do. Checked
                        FIRST so a re-detection of the same gap can never
                        cause a refetch that might replace good data.
      blocked        -- no closed round on disk for this epoch, so no
                        lock_at, so no fetch window can be built.
      unrecoverable  -- older than the horizon; gone from OKX forever.
      fetchable      -- attempt it.
    """
    now = time.time() if now is None else now
    p = StorePlan(store=store)
    for e in sorted(missing):
        if e in staged:
            p.already_staged.append(e)
            continue
        sa = round_start_at.get(e)
        if sa is None:
            p.blocked.append(e)
            continue
        if age_days(sa, now) > horizon_days:
            p.unrecoverable.append(e)
            continue
        p.fetchable.append(e)
    # Oldest first: those are the ones closest to the horizon, so a capped
    # run spends its budget on the epochs with the least time remaining.
    p.fetchable = p.fetchable[:max_per_run]
    return p


# --------------------------------------------------------------------------
# Validation -- at CAPTURE time, never at merge time.
#
# Staging an unvalidated record means possibly preserving garbage and not
# discovering it until the merge, which may be months later and after the
# horizon has passed -- at which point the real data is gone and the staged
# copy is worthless. Validate while a refetch is still possible.
# --------------------------------------------------------------------------

EXPECTED_KLINE_COUNT = 300


def validate_kline_record(rec: dict, *, epoch: int, lock_at: int) -> None:
    """Raise ValueError if the record is not a usable kline capture."""
    if not isinstance(rec, dict):
        raise ValueError("record is not an object")
    if int(rec.get("epoch", -1)) != int(epoch):
        raise ValueError(f"epoch mismatch: record={rec.get('epoch')} wanted={epoch}")
    if int(rec.get("lock_at", -1)) != int(lock_at):
        raise ValueError(f"lock_at mismatch: record={rec.get('lock_at')} wanted={lock_at}")
    k = rec.get("klines_1s")
    if not isinstance(k, list):
        raise ValueError("klines_1s is not a list")
    if len(k) != EXPECTED_KLINE_COUNT:
        raise ValueError(f"expected {EXPECTED_KLINE_COUNT} candles, got {len(k)}")
    for row in k:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise ValueError("malformed candle row")


# --------------------------------------------------------------------------
# The staged file. APPEND-ONLY, deduped by epoch, NEVER truncated.
#
# Same rule as the canonical stores and for the same reason: a partial
# refetch must never be able to replace a complete capture with a worse one.
# There is no rewrite path here at all -- not gated, absent.
# --------------------------------------------------------------------------

def staging_path(store: str, staging_dir: str = STAGING_DIR) -> str:
    return os.path.join(staging_dir, store.replace(".jsonl", "") + ".staged.jsonl")


def staged_epochs(path: str) -> set[int]:
    """Epochs already held. A missing file is an empty set, not an error."""
    from pancakebot.market_data.epoch_scan import scan_epochs
    return set(scan_epochs(path))


def append_staged(path: str, records: list[dict]) -> int:
    """Append records, skipping any epoch already present. Returns count.

    Deliberately re-reads the held set immediately before writing rather
    than trusting a caller-supplied one: this is the last line of defence
    against a duplicate entering the only copy of irreplaceable data.
    """
    if not records:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    held = staged_epochs(path)
    new = [r for r in records if int(r["epoch"]) not in held]
    if not new:
        return 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        for r in sorted(new, key=lambda x: int(x["epoch"])):
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            f.flush()
        os.fsync(f.fileno())
    return len(new)


# --------------------------------------------------------------------------
# Permanently-absent promotion.
#
# An epoch past the horizon will fail capture every run, forever. Left
# alone that is an indefinite daily failure, and an alarm that always fires
# is an alarm nobody reads -- which would undermine every other check in
# the system. Promoting it records that it is gone, WHEN that was decided
# and WHY, and stops the retry.
#
# This is a suppression mechanism, so it is deliberately conservative: it
# records only epochs whose age exceeds the horizon, it never removes an
# entry, and each entry carries its own justification.
# --------------------------------------------------------------------------

def load_permanent(path: str | None = None) -> dict[str, dict]:
    # Resolved at CALL time, not bound as a default at def time: a default
    # argument freezes the module constant when the function is defined, so
    # any later override of PERMANENT_PATH would be silently ignored.
    path = PERMANENT_PATH if path is None else path
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("stores", {}) if isinstance(d, dict) else {}
    except Exception:
        return {}


def promote_permanent(
    store: str, epochs: list[int], *, reason: str,
    path: str | None = None,
) -> int:
    """Record epochs as permanently unfetchable. Never removes anything."""
    path = PERMANENT_PATH if path is None else path
    if not epochs:
        return 0
    stores = load_permanent(path)
    entry = stores.setdefault(store, {})
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    added = 0
    for e in epochs:
        k = str(int(e))
        if k in entry:
            continue
        entry[k] = {"promoted_utc": stamp, "reason": reason}
        added += 1
    if added:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            json.dump({"stores": stores}, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    return added


def permanent_epochs_for(store: str, path: str | None = None) -> frozenset[int]:
    path = PERMANENT_PATH if path is None else path
    return frozenset(int(k) for k in load_permanent(path).get(store, {}))
