"""Answer "did the sync run CORRECTLY", not just "did it run recently".

THE GAP THIS CLOSES. The watchdog measured recency: when did the sync last
succeed. That catches a stopped sync and nothing else. A sync that runs
happily every single day while silently regressing -- losing records,
picking up CRLF, developing a new gap -- reports success, refreshes the
health timestamp, and raises no marker. It looks perfect.

The integrity report was already EMITTED into the sync log, but nothing
EVALUATED it. A human reading those numbers daily was the only thing
standing between a silent regression and permanent, unrecoverable data
loss. That is not a layer, it is a person, and it is going away.

WHY A BASELINE FILE. A regression is a comparison, not a reading. "74268
records" is not information; "74268 today, 74300 yesterday" is an alarm.
Point-in-time integrity checks cannot detect shrinkage at all, so the
previous run's numbers are persisted and compared.

THE INVARIANTS, strongest first:

  * MONOTONICITY. These files are append-only and only ever grow. A record
    count or last epoch that DECREASES is the single strongest signal
    available, and it is immune to timing: no legitimate operation, at any
    point in a sync, ever makes a store smaller.
  * CRLF == 0. The stores were normalised to LF and the writers pinned to
    it. Any CRLF means something wrote through a newline-translating path.
  * CONTIGUITY. No missing epochs beyond the 8 permanently-unfetchable
    ones, which are named in known_absent and expected.
  * CO-ADVANCEMENT. The five stores should move together.

WHAT THIS MUST NEVER DO. It must never block or fail the sync. Collection
continuing matters more than the report, and a check that halts data
collection is strictly worse than the drift it was watching for. It runs in
the watchdog, which is a separate task from the sync and cannot affect it.

AND IT MUST NOT FAIL QUIET. If the stores cannot be read at all, that is
LOUD -- an unreadable store is a worse condition than a shrinking one, and
returning "no problems found" because nothing could be examined is the
precise failure this project keeps hitting.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from pancakebot import paths
from pancakebot.market_data.epoch_scan import contiguity

BASELINE_PATH = "var/store_baseline.json"

STORES = (
    paths.CLOSED_ROUNDS_PATH,
    paths.BNB_SPOT_PRICES_PATH,
    paths.BTC_SPOT_PRICES_PATH,
    paths.ETH_SPOT_PRICES_PATH,
    paths.SOL_SPOT_PRICES_PATH,
)

# Generous by design. The watchdog and the sync BOTH fire at boot -- observed
# starting within the same second -- so the watchdog can legitimately read
# the stores mid-sync, when closed_rounds has advanced and the klines have
# not yet caught up. A daily sync moves ~288 epochs; a stalled store drifts
# by thousands. This threshold separates those two cases without flagging
# the overlap, and monotonicity (which no timing can excuse) does the sharp
# work.
MAX_LAST_EPOCH_SPREAD = 1000


def _name(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def snapshot(stores: tuple[str, ...] = STORES) -> dict[str, Any]:
    """Current integrity facts for every store.

    An unreadable store is recorded as an explicit error entry rather than
    omitted -- a missing key would read downstream as "nothing to compare",
    which is indistinguishable from health.
    """
    out: dict[str, Any] = {"taken_ts": time.time(), "stores": {}}
    for p in stores:
        try:
            c = contiguity(p)
            out["stores"][_name(p)] = {
                "n": c["n"],
                "distinct": c["distinct"],
                "latest": c["latest"],
                "earliest": c["earliest"],
                "missing": c["missing"],
                "known_absent": c["known_absent"],
                "crlf": c["crlf"],
                "duplicates": c["duplicates"],
                "out_of_order": c["out_of_order"],
                "error": None,
            }
        except Exception as e:      # noqa: BLE001
            out["stores"][_name(p)] = {"error": f"{type(e).__name__}: {e}"}
    return out


def load_baseline(path: str = BASELINE_PATH) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_baseline(snap: dict[str, Any], path: str = BASELINE_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        json.dump(snap, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def evaluate(current: dict[str, Any],
             baseline: dict[str, Any] | None) -> list[str]:
    """Return a list of failures. Empty list means integrity holds.

    Ordered so the most serious condition appears first, because the marker
    filename is built from the first failure.
    """
    fails: list[str] = []
    cur = current.get("stores") or {}

    if not cur:
        return ["UNREADABLE: no stores could be examined at all"]

    # --- unreadable stores: loudest, and checked before anything else -----
    for name, c in sorted(cur.items()):
        if c.get("error"):
            fails.append(f"UNREADABLE {name}: {c['error']}")
    if fails:
        return fails

    for name, c in sorted(cur.items()):
        if c.get("n", 0) == 0:
            fails.append(f"EMPTY {name}: zero records")
    if fails:
        return fails

    # --- CRLF ------------------------------------------------------------
    for name, c in sorted(cur.items()):
        if c.get("crlf", 0):
            fails.append(f"CRLF {name}: {c['crlf']} CRLF line endings (must be 0)")

    # --- contiguity beyond the known-absent set ---------------------------
    for name, c in sorted(cur.items()):
        if c.get("missing", 0):
            fails.append(
                f"GAP {name}: {c['missing']} unexplained missing epoch(s) "
                f"beyond the {c.get('known_absent', 0)} known-absent")

    # --- duplicates and ordering ------------------------------------------
    #
    # These were CAPTURED in the snapshot and never EVALUATED -- the exact
    # "emitted but not asserted" defect this guard was written to close,
    # reproduced inside the guard itself. Recorded plainly so the lesson
    # survives: a field that is measured and not checked is indistinguishable
    # from one that is not measured at all.
    #
    # Both matter beyond tidiness. A duplicate epoch double-counts a round in
    # any per-round rate and silently biases a backtest. A non-ascending
    # store breaks the append cursor AND makes load_earliest/load_latest --
    # which return first/last in FILE ORDER, not min/max -- return values
    # that no longer bound the data, which is the one condition that can make
    # the sync's partition overlap instead of partition.
    for name, c in sorted(cur.items()):
        if c.get("duplicates", 0):
            fails.append(
                f"DUPLICATES {name}: {c['duplicates']} duplicate epoch "
                f"record(s) -- a round counted twice biases every per-round "
                f"rate computed from this store")
    for name, c in sorted(cur.items()):
        if c.get("out_of_order", 0):
            fails.append(
                f"DISORDER {name}: {c['out_of_order']} record(s) not in "
                f"ascending epoch order -- append-only stores are strictly "
                f"ascending, and first/last stop bounding the data")

    # --- co-advancement ---------------------------------------------------
    latests = [c["latest"] for c in cur.values() if c.get("latest") is not None]
    if latests:
        spread = max(latests) - min(latests)
        if spread > MAX_LAST_EPOCH_SPREAD:
            fails.append(
                f"DRIFT: stores disagree on last epoch by {spread} "
                f"(max {max(latests)}, min {min(latests)}, "
                f"limit {MAX_LAST_EPOCH_SPREAD}) -- a store has stopped advancing")

    # --- monotonicity: needs a baseline -----------------------------------
    base = (baseline or {}).get("stores") or {}
    if not base:
        # First ever run. Say so rather than reporting health -- "no
        # comparison was possible" and "nothing regressed" are different
        # statements and must not look alike.
        fails.append("NOBASELINE: first run, monotonicity not yet checkable "
                     "(this is expected exactly once)")
        return fails

    for name, c in sorted(cur.items()):
        b = base.get(name)
        if not b or b.get("error"):
            continue
        if c["n"] < b.get("n", 0):
            fails.append(
                f"SHRANK {name}: {b['n']} -> {c['n']} records "
                f"({b['n'] - c['n']} lost) -- append-only stores never shrink")
        if (c.get("latest") is not None and b.get("latest") is not None
                and c["latest"] < b["latest"]):
            fails.append(
                f"REWOUND {name}: last epoch {b['latest']} -> {c['latest']} "
                f"-- append-only stores never go backwards")

    return fails


def failure_tag(fails: list[str]) -> str:
    """A short slug for the marker FILENAME, from the first failure.

    The diagnosis belongs in the name: a file that must be opened to be
    understood is one step away from being ignored.
    """
    if not fails:
        return "OK"
    return fails[0].split(":", 1)[0].split(" ", 1)[0].upper()
