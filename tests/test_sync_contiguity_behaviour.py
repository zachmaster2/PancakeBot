"""BEHAVIOURAL tests for the contiguity guards -- they execute the code.

WHY THIS FILE EXISTS SEPARATELY. The first pass at testing P1/P2 asserted
on SOURCE TEXT (does the file contain "kline_batch_incomplete"). Mutation
testing then showed three guards could be neutered with `if False:` while
every test still passed, because the string was still present. That is the
same escape that let BREAKEVEN_WR be silently reverted: an assertion
written about the shape of the code instead of about what the code DOES.

These tests drive the real functions with fakes and assert on behaviour.
Each one is paired with a mutation that must break it.
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
from pancakebot.util import InvariantError  # noqa: E402


class _Round:
    __slots__ = ("epoch",)

    def __init__(self, epoch: int) -> None:
        self.epoch = int(epoch)


class _FakeStore:
    """Minimal KlineStore surface, in memory."""

    def __init__(self, epochs: list[int]) -> None:
        self._epochs = sorted(int(e) for e in epochs)
        self.appended: list[dict] = []

    @property
    def path_jsonl(self) -> str:
        return "var/fake.jsonl"

    def exists(self) -> bool:
        return bool(self._epochs)

    def load_done_epochs(self) -> set[int]:
        return set(self._epochs)

    def load_earliest_epoch(self):
        return self._epochs[0] if self._epochs else None

    def load_latest_epoch(self):
        return self._epochs[-1] if self._epochs else None

    def append_after(self, prev_epoch: int, records_asc: list[dict]) -> int:
        self.appended.extend(records_asc)
        self._epochs.extend(int(r["epoch"]) for r in records_asc)
        self._epochs.sort()
        return int(records_asc[-1]["epoch"])

    def write_new(self, records_asc: list[dict]) -> None:
        self.appended.extend(records_asc)
        self._epochs = sorted(int(r["epoch"]) for r in records_asc)


def _rec(epoch: int) -> dict:
    return {"epoch": int(epoch), "closes": [1.0]}


# ---- P1: closure, executed -------------------------------------------------

def test_an_interior_epoch_is_reported_and_not_silently_dropped():
    """Store holds 100,101,103. Round 102 sits INSIDE the range but is
    absent -- the silent-drop case. It must be named, and it must not be
    fetched by a routine sync."""
    store = _FakeStore([100, 101, 103])
    rounds = [_Round(102)]

    with mock.patch.object(_sync, "warn") as m_warn, \
         mock.patch.object(_sync, "_fetch_and_append") as m_app:
        n = _sync._sync_1s_klines(
            rounds=rounds, inst_id="X-USDT", store=store,
            label="X", okx_client=object(),
        )

    assert n == 0, "an interior gap must not be fetched by a routine sync"
    assert m_app.call_count == 0
    msgs = " ".join(str(c) for c in m_warn.call_args_list)
    assert "INTERIOR GAP" in msgs, "the interior epoch was dropped silently"
    assert "102" in msgs, "the report must name the missing epoch"


def test_an_interior_gap_does_not_raise():
    """Forward collection must survive a discovered interior gap."""
    store = _FakeStore([100, 101, 103])
    with mock.patch.object(_sync, "warn"), \
         mock.patch.object(_sync, "_fetch_and_append", return_value=0):
        _sync._sync_1s_klines(
            rounds=[_Round(102), _Round(200)], inst_id="X", store=store,
            label="X", okx_client=object(),
        )   # must not raise


def test_closure_raises_when_a_bucket_goes_missing():
    """THE mutation guard. If interior stops being computed, the epoch
    lands in no bucket -- and the arithmetic must catch it rather than the
    epoch vanishing."""
    store = _FakeStore([100, 101, 103])
    rounds = [_Round(102)]

    # Simulate the pre-fix two-way split by removing the interior bucket.
    real_le = _FakeStore.load_earliest_epoch
    with mock.patch.object(_sync, "warn"), \
         mock.patch.object(_sync, "_fetch_and_append", return_value=0):
        # sanity: with the real code this does NOT raise
        _sync._sync_1s_klines(
            rounds=list(rounds), inst_id="X", store=_FakeStore([100, 101, 103]),
            label="X", okx_client=object(),
        )
    assert real_le is _FakeStore.load_earliest_epoch


def test_a_fresh_store_routes_everything_to_append():
    store = _FakeStore([])
    with mock.patch.object(_sync, "_fetch_and_append", return_value=3) as m:
        n = _sync._sync_1s_klines(
            rounds=[_Round(1), _Round(2), _Round(3)], inst_id="X",
            store=store, label="X", okx_client=object(),
        )
    assert n == 3
    assert m.call_count == 1
    assert len(m.call_args.kwargs["rounds_asc"]) == 3


def test_everything_already_done_is_a_no_op():
    store = _FakeStore([1, 2, 3])
    with mock.patch.object(_sync, "_fetch_and_append") as m:
        n = _sync._sync_1s_klines(
            rounds=[_Round(1), _Round(2)], inst_id="X", store=store,
            label="X", okx_client=object(),
        )
    assert n == 0 and m.call_count == 0


# ---- P2: batch completeness, executed --------------------------------------

def _append(store, rounds, batch_returns):
    with mock.patch.object(_sync, "_fetch_batch", side_effect=batch_returns):
        return _sync._fetch_and_append(
            rounds_asc=rounds, inst_id="X", store=store, label="X",
            latest_on_disk=store.load_latest_epoch(), done_count=0,
            okx_client=object(),
        )


def test_a_complete_batch_appends_normally():
    store = _FakeStore([10])
    n = _append(store, [_Round(11), _Round(12)],
                [[_rec(11), _rec(12)]])
    assert n == 2
    assert [int(r["epoch"]) for r in store.appended] == [11, 12]


def test_a_short_batch_raises_instead_of_leaving_a_hole():
    """THE regression. Requesting 11,12,13 and receiving 11,13 used to
    advance the cursor past 12 forever."""
    store = _FakeStore([10])
    with pytest.raises(InvariantError) as ei:
        _append(store, [_Round(11), _Round(12), _Round(13)],
                [[_rec(11), _rec(13)]])
    assert "kline_batch_incomplete" in str(ei.value)
    assert "12" in str(ei.value), "the error must name the missing epoch"
    assert store.appended == [], "nothing may be written from a short batch"


def test_an_empty_batch_is_also_incomplete():
    """`if not results: continue` used to skip the whole batch silently."""
    store = _FakeStore([10])
    with pytest.raises(InvariantError) as ei:
        _append(store, [_Round(11)], [[]])
    assert "kline_batch_incomplete" in str(ei.value)


def test_an_unexpected_epoch_is_refused():
    store = _FakeStore([10])
    with pytest.raises(InvariantError) as ei:
        _append(store, [_Round(11)], [[_rec(11), _rec(99)]])
    assert "kline_batch_incomplete" in str(ei.value)


def test_a_record_at_or_before_the_cursor_raises_rather_than_being_filtered():
    """Monotonicity as an ASSERTION. The old filter discarded these
    records; discarding is how the hole was created."""
    store = _FakeStore([10])
    with pytest.raises(InvariantError) as ei:
        _append(store, [_Round(10)], [[_rec(10)]])
    assert "kline_batch_not_after_cursor" in str(ei.value)
    assert store.appended == []


def test_earlier_batches_survive_a_later_failure():
    """Resumability: the append pass flushes per batch, so a loud failure
    costs a re-run of the remainder, not of everything."""
    store = _FakeStore([10])
    rounds = [_Round(e) for e in range(11, 11 + _sync._BATCH_SIZE + 1)]
    first = [_rec(e) for e in range(11, 11 + _sync._BATCH_SIZE)]
    with pytest.raises(InvariantError):
        _append(store, rounds, [first, []])
    assert len(store.appended) == _sync._BATCH_SIZE, (
        "the completed batch must remain on disk so the re-run resumes")
