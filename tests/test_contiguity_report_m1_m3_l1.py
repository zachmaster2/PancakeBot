"""M1, M3, and the L1 reachability finding.

M1 -- known-absent epochs are reported as EXPECTED, not as gaps. Without
this the same eight permanently-unfetchable epochs are flagged every run
forever, which is how a real finding becomes background noise and then gets
ignored. A report nobody reads is worth less than no report, because it
also carries the false comfort of having one.

M3 -- the report emits on the FAILURE path. A store is most in question
exactly when the run that touched it went wrong; reporting only on success
means the report is absent precisely when it is needed.

L1 -- the closure assert is a TAUTOLOGY on a well-formed store. When
earliest <= latest the three predicates partition the integers exhaustively
and the branch is unreachable no matter what data arrives. It is NOT what
protects the normal path; the interior report is. It IS reachable on an
out-of-order store, because load_earliest_epoch/load_latest_epoch return
first/last in FILE ORDER rather than min/max -- then the predicates overlap
and an epoch is counted twice. These tests pin both halves of that, so the
guard is never again described as doing work it does not do.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data import sync as _sync  # noqa: E402
from pancakebot.market_data.epoch_scan import contiguity, format_report  # noqa: E402
from pancakebot.market_data.known_absent import (  # noqa: E402
    KNOWN_ABSENT_KLINE_EPOCHS,
    known_absent_for,
)
from pancakebot.util import InvariantError  # noqa: E402


class _R:
    __slots__ = ("epoch",)

    def __init__(self, e: int) -> None:
        self.epoch = int(e)


def _store(tmp_path, name, epochs):
    p = tmp_path / name
    p.write_bytes(b"".join(b'{"epoch":%d}\n' % e for e in epochs))
    return str(p)


# ---- M1 --------------------------------------------------------------------

def test_known_absent_epochs_are_reported_as_expected_not_as_gaps(tmp_path):
    present = [e for e in range(1, 11) if e not in (4, 5)]
    p = _store(tmp_path, "s.jsonl", present)
    r = contiguity(p, frozenset({4, 5}))
    assert r["missing"] == 0
    assert r["known_absent"] == 2
    assert r["runs"] == []
    line = format_report([r])[0]
    assert "contiguous" in line and "known_absent=2(expected)" in line


def test_an_unlisted_gap_is_still_a_gap(tmp_path):
    """The list is an explanation, never a blanket permission."""
    present = [e for e in range(1, 11) if e not in (4, 5, 8)]
    p = _store(tmp_path, "s.jsonl", present)
    r = contiguity(p, frozenset({4, 5}))
    assert r["missing"] == 1
    assert r["runs"] == [(8, 8)]
    assert r["known_absent"] == 2


def test_a_listed_epoch_that_is_present_excuses_nothing(tmp_path):
    """Only absent AND listed is excused."""
    p = _store(tmp_path, "s.jsonl", list(range(1, 11)))
    r = contiguity(p, frozenset({4, 5}))
    assert r["missing"] == 0 and r["known_absent"] == 0


def test_a_listed_epoch_outside_the_span_explains_nothing(tmp_path):
    p = _store(tmp_path, "s.jsonl", [10, 11, 13])
    r = contiguity(p, frozenset({99}))
    assert r["missing"] == 1 and r["known_absent"] == 0


def test_the_real_eight_are_the_repair_manifest_set():
    """Provenance: MANIFEST.json -> provenance.unfetchable_klines, and the
    byte scan independently rediscovered exactly this set."""
    assert KNOWN_ABSENT_KLINE_EPOCHS == frozenset({
        445330, 445331, 447533, 447534, 449665, 449666, 452486, 452487})
    for s in ("bnb", "btc", "eth", "sol"):
        assert known_absent_for(f"var/{s}_spot_prices.jsonl") == KNOWN_ABSENT_KLINE_EPOCHS


def test_closed_rounds_has_no_exceptions():
    """It is fully contiguous and must not be granted a blanket excuse."""
    assert known_absent_for("var/closed_rounds.jsonl") == frozenset()


def test_passing_an_explicit_empty_set_shows_the_raw_picture(tmp_path):
    """The operator must always be able to see what is being excused."""
    present = [e for e in range(1, 11) if e not in (4, 5)]
    p = _store(tmp_path, "s.jsonl", present)
    assert contiguity(p, frozenset())["missing"] == 2


# ---- M3 --------------------------------------------------------------------

def test_the_report_emits_when_the_sync_fails():
    """THE M3 regression: a store is most in question when the run that
    touched it went wrong."""
    from pancakebot import app

    with mock.patch.object(app, "_emit_contiguity_report") as m_report, \
         mock.patch.object(app, "choose_rpc_url", return_value="http://x"), \
         mock.patch.object(app, "Web3PredictionContract"), \
         mock.patch.object(app, "fetch_and_save_contract_constants"), \
         mock.patch.object(app, "load_env"), \
         mock.patch.object(app, "repair_torn_tail", return_value=0), \
         mock.patch.object(app, "require_env", return_value="k"), \
         mock.patch.object(app, "GraphClient"), \
         mock.patch.object(app, "ClosedRoundsStore"), \
         mock.patch.object(app, "OkxClient"), \
         mock.patch.object(app, "sync_runtime_market_data",
                           side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            app.run_from_config(config_path="config.toml", dry=False,
                                backtest=False, sync=True)

    assert m_report.called, (
        "the contiguity report did not run on the failure path -- it is "
        "absent exactly when it is most needed")


def test_a_failing_report_cannot_mask_the_original_sync_error():
    """Called from a `finally`, a raising report would replace the real
    diagnosis with its own."""
    from pancakebot import app

    with mock.patch.object(app, "contiguity", side_effect=OSError("disk")), \
         mock.patch.object(app, "warn") as m_warn:
        app._emit_contiguity_report()      # must not raise
    assert any("contiguity report failed" in str(c)
               for c in m_warn.call_args_list)


def test_the_report_still_emits_on_the_success_path():
    from pancakebot import app
    with mock.patch.object(app, "contiguity", return_value={
            "path": "var/x.jsonl", "n": 1, "distinct": 1, "earliest": 1,
            "latest": 1, "span": 1, "missing": 0, "runs": [],
            "duplicates": 0, "out_of_order": 0, "known_absent": 0}), \
         mock.patch.object(app, "info") as m_info:
        app._emit_contiguity_report()
    assert any("INTEGRITY" in str(c) for c in m_info.call_args_list)


# ---- L1: what the closure assert actually does ----------------------------

def test_closure_is_unreachable_on_a_wellformed_ascending_store():
    """L1, pinned. With earliest <= latest the three predicates partition
    the integers exhaustively, so the sum ALWAYS equals len(remaining).
    This is documentation of intent, not the guard that protects the
    normal path -- the interior report is."""
    earliest, latest = 100, 200
    for e in range(50, 260):
        buckets = ((e < earliest)
                   + (e > latest)
                   + (earliest <= e <= latest))
        assert buckets == 1, f"epoch {e} landed in {buckets} buckets"


class _OutOfOrderStore:
    """First record 200, last record 100 -- file order, not sorted."""
    path_jsonl = "var/f.jsonl"

    def exists(self): return True
    def load_done_epochs(self): return {200, 100}
    def load_earliest_epoch(self): return 200
    def load_latest_epoch(self): return 100


def test_closure_IS_reachable_on_an_out_of_order_store():
    """load_earliest/load_latest are FILE ORDER, not min/max. When
    earliest > latest the predicates OVERLAP and an epoch is counted twice
    -- which is the one real condition this assert catches."""
    with mock.patch.object(_sync, "warn"), \
         mock.patch.object(_sync, "_fetch_and_append", return_value=0):
        with pytest.raises(InvariantError) as ei:
            _sync._sync_1s_klines(
                rounds=[_R(150)], inst_id="X", store=_OutOfOrderStore(),
                label="X", okx_client=object(),
            )
    msg = str(ei.value)
    assert "kline_partition_not_exhaustive" in msg
    assert "TWO buckets" in msg, (
        "the message must name the over-count case; saying an epoch landed "
        "in NO bucket describes the opposite condition")
    assert "earliest=200" in msg and "latest=100" in msg
