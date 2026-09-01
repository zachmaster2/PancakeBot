"""Capture gap records before the horizon takes them; never integrate them.

THE FEATURE ONLY JUSTIFIES ITSELF IN THE 47.6-DAY BAND. The sync's working
set is the last 35,000 rounds (124.0 days); OKX serves 1s klines for 171.6
days. Between them is a region where a gap is detectable by the whole-store
report, still fetchable from OKX, and NEVER LOOKED AT by the routine sync.
A capture implementation that inherited the sync's tail would address
nothing and would be an elaborate no-op, so the band case is tested first
and explicitly.

STAGING DEFERS THE DANGEROUS WRITE, IT DOES NOT REMOVE IT. Merging still
calls store.rewrite(). Staged data is a preserved problem, not a solved one.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from pancakebot.ops.gap_capture import (  # noqa: E402
    EXPECTED_KLINE_COUNT,
    OKX_KLINE_HORIZON_DAYS,
    append_staged,
    load_permanent,
    permanent_epochs_for,
    plan_capture,
    promote_permanent,
    staged_epochs,
    staging_path,
    validate_kline_record,
)

NOW = 1_800_000_000.0
DAY = 86400.0

SYNC_TAIL_DAYS = 124.0          # 35,000 rounds x ~306s


def _start_at(days_ago: float) -> int:
    return int(NOW - days_ago * DAY)


def _plan(missing, ages, staged=frozenset(), **kw):
    idx = {e: _start_at(a) for e, a in zip(missing, ages)}
    return plan_capture(store="btc_spot_prices.jsonl", missing=list(missing),
                        round_start_at=idx, staged=set(staged), now=NOW, **kw)


# ---- THE CASE THAT JUSTIFIES THE FEATURE ----------------------------------

def test_a_gap_OUTSIDE_the_sync_working_set_is_still_captured():
    """150 days old: beyond the sync's 124-day tail, inside OKX's 171.6-day
    horizon. This is the 47.6-day band, and it is the entire reason the
    module exists. If this epoch is not fetchable, capture is a no-op."""
    p = _plan([500000], [150.0])
    assert p.fetchable == [500000], (
        "an epoch in the 47.6-day band was not captured -- capture has "
        "inherited the sync's working set and addresses nothing")
    assert 150.0 > SYNC_TAIL_DAYS, "the fixture must sit outside the tail"
    assert 150.0 < OKX_KLINE_HORIZON_DAYS, "and inside the horizon"


def test_capture_does_not_read_the_syncs_tail():
    """Source-level: no cache_n / tail_rounds / backtest_round_count may
    appear in the capture path. Nothing must be able to reintroduce the
    124-day ceiling."""
    src = io.open(_REPO_ROOT / "scripts" / "capture_gaps.py",
                  encoding="utf-8").read()
    for forbidden in ("tail_rounds", "cache_n", "backtest_round_count", "[-35000:]"):
        assert forbidden not in src, f"capture path references {forbidden}"
    assert "contiguity(" in src, "capture must drive from the whole-store report"


def test_a_recent_gap_is_also_captured():
    p = _plan([500000], [3.0])
    assert p.fetchable == [500000]


# ---- past the horizon ------------------------------------------------------

def test_a_past_horizon_gap_is_unrecoverable_not_retried():
    """Left as fetchable it would fail every run forever, and an alarm that
    always fires is one nobody reads."""
    p = _plan([400000], [200.0])
    assert p.unrecoverable == [400000]
    assert p.fetchable == []


def test_the_horizon_boundary(monkeypatch):
    inside = _plan([1], [OKX_KLINE_HORIZON_DAYS - 0.5])
    outside = _plan([1], [OKX_KLINE_HORIZON_DAYS + 0.5])
    assert inside.fetchable == [1]
    assert outside.unrecoverable == [1]


def test_promotion_is_recorded_with_a_reason_and_never_removed(tmp_path):
    p = str(tmp_path / "perm.json")
    assert promote_permanent("btc_spot_prices.jsonl", [1, 2],
                             reason="past horizon", path=p) == 2
    # Re-promoting the same epochs adds nothing and loses nothing.
    assert promote_permanent("btc_spot_prices.jsonl", [1, 2],
                             reason="again", path=p) == 0
    assert promote_permanent("btc_spot_prices.jsonl", [3],
                             reason="past horizon", path=p) == 1
    d = load_permanent(p)["btc_spot_prices.jsonl"]
    assert set(d) == {"1", "2", "3"}
    assert all("promoted_utc" in v and "reason" in v for v in d.values())


def test_promoted_epochs_stop_being_reported_as_gaps(tmp_path, monkeypatch):
    """The promotion must actually reach known_absent, or the retry never
    stops."""
    p = str(tmp_path / "perm.json")
    promote_permanent("btc_spot_prices.jsonl", [77], reason="x", path=p)
    import pancakebot.ops.gap_capture as gc
    monkeypatch.setattr(gc, "PERMANENT_PATH", p)
    assert 77 in permanent_epochs_for("btc_spot_prices.jsonl", p)

    import pancakebot.market_data.known_absent as ka
    got = ka.known_absent_for("var/btc_spot_prices.jsonl")
    assert 77 in got
    assert 445330 in got, "the static known-absent set must survive the union"


def test_suppression_fails_closed_when_the_promotion_file_is_broken(tmp_path, monkeypatch):
    """A broken suppression file must REPORT MORE, never less."""
    bad = tmp_path / "perm.json"
    bad.write_text("{not json", encoding="utf-8")
    import pancakebot.ops.gap_capture as gc
    monkeypatch.setattr(gc, "PERMANENT_PATH", str(bad))
    import pancakebot.market_data.known_absent as ka
    got = ka.known_absent_for("var/btc_spot_prices.jsonl")
    assert got == ka.KNOWN_ABSENT_KLINE_EPOCHS


# ---- blocked ---------------------------------------------------------------

def test_an_epoch_with_no_round_on_disk_is_blocked_not_silently_skipped():
    """A kline window is anchored to the round's lock_at; without the round
    there is no window. Reported, not dropped."""
    p = plan_capture(store="x", missing=[1, 2], round_start_at={1: _start_at(5)},
                     staged=set(), now=NOW)
    assert p.fetchable == [1]
    assert p.blocked == [2]


# ---- re-detection of an already-captured gap -------------------------------

def test_an_already_staged_epoch_is_not_refetched():
    """THE requirement that stops a partial refetch replacing a complete
    capture. Checked BEFORE anything else."""
    p = _plan([500000], [10.0], staged={500000})
    assert p.already_staged == [500000]
    assert p.fetchable == []


def test_a_partial_capture_leaves_the_rest_fetchable():
    idx = {e: _start_at(10.0) for e in (1, 2, 3)}
    p = plan_capture(store="x", missing=[1, 2, 3], round_start_at=idx,
                     staged={2}, now=NOW)
    assert p.already_staged == [2]
    assert p.fetchable == [1, 3]


# ---- the staged file: append-only, dedup, never truncate ------------------

def _rec(e):
    return {"epoch": e, "lock_at": 1000 + e, "klines_1s": [[0] * 6] * EXPECTED_KLINE_COUNT}


def test_staging_appends_and_dedups(tmp_path):
    p = str(tmp_path / "s.staged.jsonl")
    assert append_staged(p, [_rec(1), _rec(2)]) == 2
    assert append_staged(p, [_rec(2), _rec(3)]) == 1      # 2 already held
    assert staged_epochs(p) == {1, 2, 3}


def test_a_repeat_capture_cannot_shrink_the_staged_file(tmp_path):
    """A partial refetch must never replace a complete capture."""
    p = str(tmp_path / "s.staged.jsonl")
    append_staged(p, [_rec(e) for e in range(1, 11)])
    before = Path(p).read_bytes()
    assert append_staged(p, [_rec(1)]) == 0
    assert Path(p).read_bytes() == before, "the staged file was rewritten"
    assert staged_epochs(p) == set(range(1, 11))


def test_there_is_no_rewrite_path_in_staging_at_all():
    """Not gated -- absent. The staged file is the only copy of records
    that exist nowhere else."""
    src = io.open(_REPO_ROOT / "pancakebot" / "ops" / "gap_capture.py",
                  encoding="utf-8").read()
    assert "def append_staged" in src
    assert '"w"' not in src.split("def append_staged")[1].split("def ")[0]
    assert "truncate" not in src.lower().split("def append_staged")[1]


def test_staged_epochs_of_a_missing_file_is_empty(tmp_path):
    assert staged_epochs(str(tmp_path / "nope.jsonl")) == set()


# ---- validation happens at CAPTURE time -----------------------------------

def test_a_good_record_validates():
    validate_kline_record(_rec(5), epoch=5, lock_at=1005)


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r.update({"epoch": 99}), "epoch mismatch"),
    (lambda r: r.update({"lock_at": 1}), "lock_at mismatch"),
    (lambda r: r.update({"klines_1s": [[0] * 6] * 299}), "expected 300"),
    (lambda r: r.update({"klines_1s": "nope"}), "not a list"),
    (lambda r: r.update({"klines_1s": [[0, 1]] * 300}), "malformed candle"),
])
def test_bad_records_are_refused_before_they_reach_staging(mutate, msg):
    """Staging unvalidated records means possibly preserving garbage and
    not finding out until the merge -- months later, past the horizon,
    when the real data is gone."""
    r = _rec(5)
    mutate(r)
    with pytest.raises(ValueError, match=msg):
        validate_kline_record(r, epoch=5, lock_at=1005)


def test_capture_validates_before_appending():
    src = io.open(_REPO_ROOT / "scripts" / "capture_gaps.py",
                  encoding="utf-8").read()
    assert src.index("validate_kline_record(") < src.index("append_staged("), (
        "records must be validated before they are staged")


# ---- capture must not be able to block the sync ---------------------------

def test_capture_is_not_reachable_from_the_sync_path():
    app = io.open(_REPO_ROOT / "pancakebot" / "app.py", encoding="utf-8").read()
    sync = io.open(_REPO_ROOT / "pancakebot" / "market_data" / "sync.py",
                   encoding="utf-8").read()
    assert "gap_capture" not in app and "capture_gaps" not in app
    assert "gap_capture" not in sync and "capture_gaps" not in sync


def test_a_capture_failure_is_not_a_run_failure_for_unrecoverable_epochs():
    """An unrecoverable epoch is a fact about the world, not a failure of
    today's run. Reporting it as a failure daily is what trains people to
    ignore alarms."""
    src = io.open(_REPO_ROOT / "scripts" / "capture_gaps.py",
                  encoding="utf-8").read()
    assert 'return 1 if s["failed"] else 0' in src


# ---- the backup discipline -------------------------------------------------

def test_staged_captures_is_NOT_gitignored():
    """REQUIREMENT. The staged file is the only copy of records that exist
    nowhere else. Gitignored and outside the archive, it would be a worse
    single point of failure than the gap it survives. Committed, it is
    off-machine and versioned with no operator action."""
    import subprocess
    probe = _REPO_ROOT / "staged_captures" / "_ignore_probe.jsonl"
    probe.parent.mkdir(exist_ok=True)
    probe.write_text('{"epoch":1}\n', encoding="utf-8")
    try:
        rc = subprocess.run(
            ["git", "check-ignore", "-q", str(probe)],
            cwd=str(_REPO_ROOT), capture_output=True).returncode
    finally:
        probe.unlink(missing_ok=True)
    assert rc != 0, (
        "staged_captures/ is gitignored -- captured records would exist "
        "only on this machine, which is the failure this design prevents")


def test_staging_does_not_live_under_the_ignored_var_tree():
    from pancakebot.ops.gap_capture import PERMANENT_PATH, STAGING_DIR
    assert not STAGING_DIR.startswith("var/"), (
        "staging moved under var/, which is gitignored and unbacked")
    assert not PERMANENT_PATH.startswith("var/")
