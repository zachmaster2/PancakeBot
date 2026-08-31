"""The watchdog must answer "did it run CORRECTLY", not just "recently".

THE GAP. Recency catches a stopped sync and nothing else. A sync that runs
happily every day while silently regressing -- losing records, picking up
CRLF, developing a gap -- refreshes the health timestamp, reports success,
and raises no marker. It looks perfect. The integrity report was already
emitted into the sync log, but nothing EVALUATED it; a human reading those
numbers daily was the only thing between a silent regression and permanent
loss of data that cannot be refetched.

EVERY CHECK HERE IS MUTATION-TESTED against real files on disk. A check
that has never fired is a check nobody has verified -- which is exactly the
shape of repair_torn_tail, which was correct, tested five times, and called
by nothing.

Mutations proven to raise: a store SHRINKS, a store REWINDS, CRLF appears,
a gap is manufactured, stores drift apart, a store becomes unreadable, a
store is empty, and no baseline exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from pancakebot.ops.store_guard import (  # noqa: E402
    MAX_LAST_EPOCH_SPREAD,
    evaluate,
    failure_tag,
    snapshot,
)


def _write(p: Path, epochs, *, crlf: bool = False) -> None:
    term = "\r\n" if crlf else "\n"
    with open(p, "w", encoding="utf-8", newline="") as f:
        for e in epochs:
            f.write(json.dumps({"epoch": e, "v": 1}, separators=(",", ":")) + term)


@pytest.fixture()
def store(tmp_path):
    """A healthy single-store world, plus its baseline."""
    p = tmp_path / "closed_rounds.jsonl"
    _write(p, range(1000, 1100))
    stores = (str(p),)
    base = snapshot(stores)
    return p, stores, base


def _fails(stores, base):
    return evaluate(snapshot(stores), base)


# ---- healthy baseline ------------------------------------------------------

def test_an_unchanged_store_passes(store):
    p, stores, base = store
    assert _fails(stores, base) == []


def test_growth_passes(store):
    """The normal case: the store got bigger."""
    p, stores, base = store
    _write(p, range(1000, 1150))
    assert _fails(stores, base) == []


# ---- MUTATION: the store shrinks ------------------------------------------

def test_a_shrinking_store_raises(store):
    """THE strongest invariant. Append-only files never lose records, at
    any point in any sync, so this needs no timing caveat."""
    p, stores, base = store
    _write(p, range(1000, 1050))          # 100 -> 50 records
    f = _fails(stores, base)
    assert any(x.startswith("SHRANK") for x in f), f
    assert "50 lost" in " ".join(f)
    assert failure_tag(f) == "SHRANK"


def test_a_rewound_store_raises(store):
    """Same count, but the last epoch went backwards."""
    p, stores, base = store
    _write(p, range(900, 1000))           # 100 records, but older
    f = _fails(stores, base)
    assert any(x.startswith("REWOUND") for x in f), f


# ---- MUTATION: CRLF --------------------------------------------------------

def test_crlf_raises(store):
    """The stores were normalised to LF and the writers pinned to it. Any
    CRLF means something wrote through a translating path."""
    p, stores, base = store
    _write(p, range(1000, 1100), crlf=True)
    f = _fails(stores, base)
    assert any(x.startswith("CRLF") for x in f), f
    assert "100 CRLF" in " ".join(f)


def test_even_one_crlf_line_raises(store):
    """Not a threshold -- zero is the only acceptable value."""
    p, stores, base = store
    with open(p, "a", encoding="utf-8", newline="") as fh:
        fh.write('{"epoch":1100,"v":1}\r\n')
    f = _fails(stores, base)
    assert any(x.startswith("CRLF") for x in f), f


# ---- MUTATION: a manufactured gap -----------------------------------------

def test_a_manufactured_gap_raises(store):
    p, stores, base = store
    holed = [e for e in range(1000, 1120) if e not in (1050, 1051, 1052)]
    _write(p, holed)
    f = _fails(stores, base)
    assert any(x.startswith("GAP") for x in f), f
    assert "3 unexplained" in " ".join(f)


def test_the_known_absent_epochs_do_not_raise(tmp_path):
    """The 8 permanently-unfetchable kline epochs must stay silent, or a
    real finding decays into daily noise and gets ignored."""
    p = tmp_path / "btc_spot_prices.jsonl"
    known = (445330, 445331, 447533, 447534, 449665, 449666, 452486, 452487)
    _write(p, [e for e in range(445000, 453000) if e not in known])
    stores = (str(p),)
    base = snapshot(stores)
    f = evaluate(snapshot(stores), base)
    assert f == [], f


# ---- MUTATION: stores drift apart -----------------------------------------

def test_stores_drifting_apart_raises(tmp_path):
    a = tmp_path / "closed_rounds.jsonl"
    b = tmp_path / "btc_spot_prices.jsonl"
    _write(a, range(1000, 1100))
    _write(b, range(1000, 1100))
    stores = (str(a), str(b))
    base = snapshot(stores)
    _write(a, range(1000, 1100 + MAX_LAST_EPOCH_SPREAD + 50))   # a races ahead
    f = evaluate(snapshot(stores), base)
    assert any(x.startswith("DRIFT") for x in f), f


def test_a_normal_sync_worth_of_lag_does_not_raise(tmp_path):
    """The watchdog and the sync BOTH fire at boot -- observed starting
    within the same second -- so reading mid-sync is normal and must not
    alarm."""
    a = tmp_path / "closed_rounds.jsonl"
    b = tmp_path / "btc_spot_prices.jsonl"
    _write(a, range(1000, 1400))          # 288-ish ahead, mid-sync
    _write(b, range(1000, 1100))
    stores = (str(a), str(b))
    f = evaluate(snapshot(stores), snapshot(stores))
    assert not any(x.startswith("DRIFT") for x in f), f


# ---- MUTATION: the check cannot silently pass -----------------------------

def test_an_unreadable_store_is_loud_not_a_silent_pass(tmp_path, monkeypatch):
    """'The check could not run' and 'the check found nothing' must never
    look alike. This is the failure mode the whole project keeps hitting."""
    p = tmp_path / "closed_rounds.jsonl"
    _write(p, range(1000, 1100))
    stores = (str(p),)
    base = snapshot(stores)

    import pancakebot.ops.store_guard as sg

    def boom(_path, *a, **k):
        raise OSError("disk went away")

    monkeypatch.setattr(sg, "contiguity", boom)
    f = evaluate(sg.snapshot(stores), base)
    assert any(x.startswith("UNREADABLE") for x in f), f
    assert failure_tag(f) == "UNREADABLE"


def test_a_missing_store_is_loud(tmp_path):
    stores = (str(tmp_path / "gone.jsonl"),)
    f = evaluate(snapshot(stores), {"stores": {"gone.jsonl": {"n": 5}}})
    assert any(x.startswith("EMPTY") for x in f), f


def test_no_stores_at_all_is_loud():
    assert evaluate({"stores": {}}, None) == [
        "UNREADABLE: no stores could be examined at all"]


def test_a_first_run_with_no_baseline_does_not_read_as_healthy(store):
    """No comparison was possible is NOT the same statement as nothing
    regressed, and they must not look alike."""
    p, stores, _ = store
    f = evaluate(snapshot(stores), None)
    assert any(x.startswith("NOBASELINE") for x in f), f


# ---- the marker filename carries the diagnosis ----------------------------

@pytest.mark.parametrize("fail,tag", [
    (["SHRANK x: ..."], "SHRANK"),
    (["CRLF x: ..."], "CRLF"),
    (["GAP x: ..."], "GAP"),
    (["DRIFT: ..."], "DRIFT"),
    (["UNREADABLE x: ..."], "UNREADABLE"),
    (["CHECKFAILED: ..."], "CHECKFAILED"),
])
def test_failure_tag_names_the_condition(fail, tag):
    assert failure_tag(fail) == tag


def test_the_marker_name_prefers_integrity_over_staleness():
    """A store that is CORRUPTING is more urgent to read off a desktop than
    one that is merely behind."""
    import sync_watchdog as W
    a = {"status": "OK", "hours_since_success": 1}
    assert W._marker_name(a, ["SHRANK x: ..."]) == \
        "PANCAKEBOT_SYNC_INTEGRITY_SHRANK.txt"
    assert W._marker_name({"status": "NEVER_RUN", "hours_since_success": None},
                          None) == "PANCAKEBOT_SYNC_HAS_NEVER_RUN.txt"


def test_the_integrity_marker_body_says_what_is_at_risk():
    import sync_watchdog as W
    body = W._marker_body({"status": "OK", "hours_since_success": 1,
                           "last_success_utc": "x", "consecutive_failures": 0,
                           "last_error": None},
                          ["SHRANK closed_rounds.jsonl: 10 -> 5 records"])
    assert "171.6 days" in body
    assert "SHRANK" in body
    assert "backup" in body.lower()


# ---- the sync must never be blocked by this -------------------------------

def test_the_guard_is_not_imported_by_the_sync_path():
    """Collection continuing matters more than the report. A check that can
    halt data collection is strictly worse than the drift it watches for."""
    app = (_REPO_ROOT / "pancakebot" / "app.py").read_text(encoding="utf-8")
    sync = (_REPO_ROOT / "pancakebot" / "market_data" / "sync.py").read_text(
        encoding="utf-8")
    assert "store_guard" not in app
    assert "store_guard" not in sync
