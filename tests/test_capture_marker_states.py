"""The three marker states, and the one requirement that decides the design.

CAPTURE SUCCESS MUST DOWNGRADE URGENCY, NOT CLEAR THE MARKER.

If a successful capture cleared the marker, the data would be safe forever
and merged never: the store keeps its hole, the backtest silently runs
against it, and the only signal that anything is outstanding has been
deleted by the thing that was supposed to help. That is not solving the
problem, it is moving it somewhere harder to see.

So "records staged but not yet merged" is a STANDING condition, recomputed
from the files every run, that clears only when the records are actually in
the store.

  CAPTUREFAILED  HIGH  -- still fetchable today, gone after the horizon
  CAPTURED       LOW   -- data safe, store still holed, merge outstanding
  UNRECOVERABLE  INFO  -- past the horizon, nothing can be done
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

from pancakebot.ops.gap_capture import (  # noqa: E402
    EXPECTED_KLINE_COUNT,
    append_staged,
    staging_path,
)
from pancakebot.ops.store_guard import (  # noqa: E402
    _unmerged_staged_failures,
    evaluate,
    failure_tag,
    snapshot,
)


def _store(tmp_path, name, epochs):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8", newline="") as f:
        for e in epochs:
            f.write(json.dumps({"epoch": e}, separators=(",", ":")) + "\n")
    return str(p)


def _rec(e):
    return {"epoch": e, "lock_at": e, "klines_1s": [[0] * 6] * EXPECTED_KLINE_COUNT}


# ---- THE requirement -------------------------------------------------------

def test_capture_success_does_NOT_clear_the_marker(tmp_path):
    """The single most important behaviour in the staging design."""
    holed = [e for e in range(1000, 1100) if e != 1050]
    sp = _store(tmp_path, "btc_spot_prices.jsonl", holed)
    sdir = str(tmp_path / "staged")

    # Before capture: a GAP.
    before = evaluate(snapshot((sp,), staging_dir=sdir), None)
    assert any(f.startswith("GAP") for f in before), before

    # Capture succeeds -- the record is now staged.
    append_staged(staging_path("btc_spot_prices.jsonl", sdir), [_rec(1050)])

    after = evaluate(snapshot((sp,), staging_dir=sdir), None)
    assert any(f.startswith("CAPTURED") for f in after), (
        "capture succeeded and the condition vanished -- the merge would "
        "never happen and the store would keep its hole silently")
    assert after != [], "a successful capture must NOT read as healthy"


def test_the_condition_clears_only_when_the_record_is_IN_the_store(tmp_path):
    sp = _store(tmp_path, "btc_spot_prices.jsonl", range(1000, 1100))
    sdir = str(tmp_path / "staged")
    append_staged(staging_path("btc_spot_prices.jsonl", sdir), [_rec(1050)])
    # 1050 is already in the store, so nothing is outstanding.
    f = evaluate(snapshot((sp,), staging_dir=sdir), None)
    assert not any(x.startswith("CAPTURED") for x in f), f


def test_it_is_recomputed_from_the_files_not_remembered(tmp_path):
    """No state file to lose, corrupt, or forget to update."""
    holed = [e for e in range(1000, 1100) if e != 1050]
    sp = _store(tmp_path, "btc_spot_prices.jsonl", holed)
    sdir = str(tmp_path / "staged")
    append_staged(staging_path("btc_spot_prices.jsonl", sdir), [_rec(1050)])
    assert any(f.startswith("CAPTURED")
               for f in evaluate(snapshot((sp,), staging_dir=sdir), None))
    # Merge it for real; the condition disappears with no bookkeeping.
    full = _store(tmp_path, "btc_spot_prices.jsonl", range(1000, 1100))
    assert not any(f.startswith("CAPTURED")
                   for f in evaluate(snapshot((full,), staging_dir=sdir), None))


# ---- ordering: CAPTURED must never mask a real fault ----------------------

def test_captured_never_wins_the_marker_name_over_a_real_fault():
    cur = {
        "btc_spot_prices.jsonl": {
            "n": 10, "distinct": 10, "latest": 10, "earliest": 1,
            "missing": 0, "known_absent": 0, "crlf": 4, "duplicates": 0,
            "out_of_order": 0, "unmerged_staged": 5, "error": None,
        }
    }
    base = {"stores": {"btc_spot_prices.jsonl": {"n": 10, "latest": 10}}}
    f = evaluate({"stores": cur}, base)
    assert failure_tag(f) == "CRLF", f
    assert any(x.startswith("CAPTURED") for x in f), "still reported, just last"


def test_captured_is_appended_last_even_with_no_baseline():
    cur = {
        "btc_spot_prices.jsonl": {
            "n": 10, "distinct": 10, "latest": 10, "earliest": 1,
            "missing": 0, "known_absent": 0, "crlf": 0, "duplicates": 0,
            "out_of_order": 0, "unmerged_staged": 5, "error": None,
        }
    }
    f = evaluate({"stores": cur}, None)
    assert failure_tag(f) == "NOBASELINE", f
    assert any(x.startswith("CAPTURED") for x in f), (
        "the first run must still report an outstanding merge")


def test_captured_alone_names_the_marker_captured():
    cur = {
        "btc_spot_prices.jsonl": {
            "n": 10, "distinct": 10, "latest": 10, "earliest": 1,
            "missing": 0, "known_absent": 0, "crlf": 0, "duplicates": 0,
            "out_of_order": 0, "unmerged_staged": 5, "error": None,
        }
    }
    base = {"stores": {"btc_spot_prices.jsonl": {"n": 10, "latest": 10}}}
    f = evaluate({"stores": cur}, base)
    assert failure_tag(f) == "CAPTURED", f


# ---- the three states in the watchdog -------------------------------------

def test_capturefailed_is_inserted_FIRST_as_the_urgent_state():
    """Still fetchable today, gone after the horizon. It must outrank
    everything, including a CAPTURED that succeeded for other epochs."""
    src = (_REPO_ROOT / "scripts" / "sync_watchdog.py").read_text(encoding="utf-8")
    i = src.index('fails.insert(0, f"CAPTUREFAILED')
    assert i > 0
    assert 'fails.append(f"UNRECOVERABLE' in src, "the INFO state is missing"
    j = src.index('fails.append(f"UNRECOVERABLE')
    assert i < j, "CAPTUREFAILED must be raised before UNRECOVERABLE"


def test_unrecoverable_is_informational_not_a_daily_failure():
    """A lost cause reported as a failure every day trains the operator to
    ignore the states that DO need action."""
    src = (_REPO_ROOT / "scripts" / "capture_gaps.py").read_text(encoding="utf-8")
    assert 'return 1 if s["failed"] else 0' in src, (
        "unrecoverable epochs must not make the run exit non-zero")


def test_all_three_states_produce_distinct_marker_names():
    import sync_watchdog as W
    a = {"status": "OK", "hours_since_success": 1}
    names = {
        W._marker_name(a, ["CAPTUREFAILED: ..."]),
        W._marker_name(a, ["CAPTURED x: ..."]),
        W._marker_name(a, ["UNRECOVERABLE: ..."]),
    }
    assert len(names) == 3, names
    assert "PANCAKEBOT_SYNC_INTEGRITY_CAPTUREFAILED.txt" in names
    assert "PANCAKEBOT_SYNC_INTEGRITY_CAPTURED.txt" in names
    assert "PANCAKEBOT_SYNC_INTEGRITY_UNRECOVERABLE.txt" in names


# ---- capture must not be able to suppress the findings --------------------

def test_a_raising_capture_becomes_a_failure_not_a_silence():
    src = (_REPO_ROOT / "scripts" / "sync_watchdog.py").read_text(encoding="utf-8")
    i = src.index("capture itself raised")
    window = src[max(0, i - 400):i]
    assert "except Exception" in window
    assert "fails.insert(0" in src[i - 200:i + 200], (
        "a capture that explodes must surface, not swallow the gap finding")


def test_capture_only_runs_when_a_gap_exists():
    """No gap, no fetching. The common case must cost nothing."""
    src = (_REPO_ROOT / "scripts" / "sync_watchdog.py").read_text(encoding="utf-8")
    assert 'if any(f.startswith("GAP") for f in fails) and not args.no_capture:' in src
