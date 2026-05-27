"""Tests for ``engine._startup_handshake_with_retry`` and the
zero-state retry guards inside ``engine._epoch_handshake`` itself.

Step 27d: defensive retry wrapper that absorbs the fresh-spawn race
after a supervisor respawn during a round transition window. The chain
may briefly report ``locked.lock_price == 0`` or ``open.lock_at == 0``
before ``executeRound()`` settles the new values; without this wrapper
``_run_one_iteration`` would crash on the unsettled state.

Step 27e-D: defense-in-depth — the bare ``_epoch_handshake`` also
retries on the two non-``lock_ts`` zero-state conditions
(``locked.lock_price_usd <= 0`` and ``open.lock_ts <= 0``) within its
own RETRY_BACKOFF_SECONDS budget. This catches mid-iteration recurrences;
the startup wrapper handles the longer fresh-spawn window.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.chain.prediction_contract import RoundData  # noqa: E402
from pancakebot.runtime import engine  # noqa: E402
from pancakebot.types import Round  # noqa: E402
from pancakebot.util import InvariantError  # noqa: E402


def _make_round_data(
    *,
    epoch: int,
    start_ts: int = 1_700_000_000,
    lock_ts: int = 1_700_000_300,
    lock_price_usd: float = 350.0,
) -> RoundData:
    return RoundData(
        epoch=epoch,
        start_ts=start_ts,
        lock_ts=lock_ts,
        close_ts=lock_ts + 300,
        lock_price_usd=lock_price_usd,
        close_price_usd=0.0,
        bull_amount_wei=0,
        bear_amount_wei=0,
        oracle_called=False,
    )


def _valid_handshake_tuple(
    *,
    lock_price: float = 350.0,
    open_lock_at: int = 1_700_000_300,
) -> tuple[Round, Round, int, object]:
    locked = Round(
        epoch=100,
        start_at=1_700_000_000,
        lock_at=1_700_000_300,
        lock_price=lock_price,
        close_price=None,
        position=None,
        failed=False,
        bets=(),
    )
    open_round = Round(
        epoch=101,
        start_at=1_700_000_300,
        lock_at=open_lock_at,
        lock_price=None,
        close_price=None,
        position=None,
        failed=False,
        bets=(),
    )
    return locked, open_round, 101, object()


def _make_cfg(buffer_seconds: int = 30):
    return mock.Mock(buffer_seconds=buffer_seconds)


def test_startup_handshake_first_call_valid_returns_immediately():
    """When _epoch_handshake returns settled state on the first call,
    no retries occur and no sleep is incurred."""
    cfg = _make_cfg()
    valid = _valid_handshake_tuple()

    with mock.patch(
        "pancakebot.runtime.engine._epoch_handshake",
        return_value=valid,
    ) as m_hs, mock.patch(
        "pancakebot.runtime.engine.sleep_seconds",
    ) as m_sleep, mock.patch(
        "pancakebot.runtime.engine.time.time",
        return_value=1_000.0,
    ):
        engine._startup_handshake_with_retry(cfg)

    assert m_hs.call_count == 1
    assert m_sleep.call_count == 0


def test_startup_handshake_retries_then_succeeds():
    """Two unsettled responses followed by a settled one: the wrapper
    retries each unsettled response with sleep_seconds(2.0) and returns
    once the chain has settled."""
    cfg = _make_cfg()
    zero_price = _valid_handshake_tuple(lock_price=0.0)
    zero_lock_at = _valid_handshake_tuple(open_lock_at=0)
    valid = _valid_handshake_tuple()

    with mock.patch(
        "pancakebot.runtime.engine._epoch_handshake",
        side_effect=[zero_price, zero_lock_at, valid],
    ) as m_hs, mock.patch(
        "pancakebot.runtime.engine.sleep_seconds",
    ) as m_sleep, mock.patch(
        "pancakebot.runtime.engine.time.time",
        return_value=1_000.0,
    ):
        engine._startup_handshake_with_retry(cfg)

    assert m_hs.call_count == 3
    assert m_sleep.call_count == 2
    for call in m_sleep.call_args_list:
        assert call.args == (2.0,)


def test_startup_handshake_exhausts_deadline_raises():
    """When the chain never settles within buffer_seconds + padding,
    raise InvariantError with both observed values in the message."""
    cfg = _make_cfg(buffer_seconds=30)
    zero_price = _valid_handshake_tuple(lock_price=0.0)

    # time.time(): startup -> deadline = 1000 + 30 + 5 = 1035.
    # Subsequent calls: well past deadline so the first deadline check trips.
    time_values = iter([1_000.0, 1_100.0, 1_100.0])

    with mock.patch(
        "pancakebot.runtime.engine._epoch_handshake",
        return_value=zero_price,
    ), mock.patch(
        "pancakebot.runtime.engine.sleep_seconds",
    ) as m_sleep, mock.patch(
        "pancakebot.runtime.engine.time.time",
        side_effect=lambda: next(time_values),
    ):
        with pytest.raises(InvariantError) as exc_info:
            engine._startup_handshake_with_retry(cfg)

    msg = str(exc_info.value)
    assert "startup_handshake_exhausted_retries" in msg
    assert "lock_price=0.0" in msg
    assert "lock_at=" in msg
    # Deadline tripped on the first attempt's check, so no sleep occurred.
    assert m_sleep.call_count == 0


def test_startup_handshake_deadline_boundary():
    """Regression guard for deadline arithmetic: just-before-deadline still
    loops; reaching deadline (>=) trips. buffer_seconds=30 + padding=5 = 35s.
    """
    cfg = _make_cfg(buffer_seconds=30)
    zero_price = _valid_handshake_tuple(lock_price=0.0)

    # Startup t0=1000.0 → deadline = 1035.0.
    # After 1st handshake: t=1034.999 → < deadline → sleep + retry.
    # After 2nd handshake: t=1035.0 → >= deadline → raise.
    time_values = iter([1_000.0, 1_034.999, 1_035.0])

    with mock.patch(
        "pancakebot.runtime.engine._epoch_handshake",
        return_value=zero_price,
    ), mock.patch(
        "pancakebot.runtime.engine.sleep_seconds",
    ) as m_sleep, mock.patch(
        "pancakebot.runtime.engine.time.time",
        side_effect=lambda: next(time_values),
    ):
        with pytest.raises(InvariantError):
            engine._startup_handshake_with_retry(cfg)

    # First check at 1034.999 was below deadline → exactly one sleep occurred.
    assert m_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Step 27e-D: zero-state retry guards inside _epoch_handshake itself
# ---------------------------------------------------------------------------


def _make_handshake_cfg(contract: mock.Mock) -> mock.Mock:
    return mock.Mock(contract=contract)


def test_epoch_handshake_retries_on_locked_lock_price_zero():
    """First attempt: locked_rd.lock_price_usd == 0 → retry path triggers.
    Second attempt: full settled state → returns successfully."""
    locked_unsettled = _make_round_data(epoch=100, lock_price_usd=0.0)
    locked_settled = _make_round_data(epoch=100, lock_price_usd=350.0)
    open_settled = _make_round_data(
        epoch=101,
        start_ts=1_700_000_300,
        lock_ts=1_700_000_600,
    )

    contract = mock.Mock()
    contract.current_epoch.return_value = 101
    contract.round_data.side_effect = [
        locked_unsettled,  # attempt 1, locked
        open_settled,      # attempt 1, open
        locked_settled,    # attempt 2, locked
        open_settled,      # attempt 2, open
    ]
    cfg = _make_handshake_cfg(contract)

    with mock.patch("pancakebot.runtime.engine.sleep_seconds") as m_sleep:
        locked_r, open_r, ep, open_rd = engine._epoch_handshake(cfg)

    assert ep == 101
    assert locked_r.lock_price == 350.0
    assert int(open_r.lock_at) == 1_700_000_600
    assert open_rd is open_settled
    # Exactly one retry → one backoff sleep (RETRY_BACKOFF_SECONDS[0] = 2s).
    assert m_sleep.call_count == 1
    assert m_sleep.call_args_list[0].args == (2,)


def test_epoch_handshake_retries_on_open_lock_ts_zero():
    """First attempt: open_rd.lock_ts == 0 → retry path triggers.
    Second attempt: full settled state → returns successfully."""
    locked_settled = _make_round_data(epoch=100, lock_price_usd=350.0)
    open_unsettled = _make_round_data(
        epoch=101,
        start_ts=1_700_000_300,
        lock_ts=0,
    )
    open_settled = _make_round_data(
        epoch=101,
        start_ts=1_700_000_300,
        lock_ts=1_700_000_600,
    )

    contract = mock.Mock()
    contract.current_epoch.return_value = 101
    contract.round_data.side_effect = [
        locked_settled,    # attempt 1, locked
        open_unsettled,    # attempt 1, open
        locked_settled,    # attempt 2, locked
        open_settled,      # attempt 2, open
    ]
    cfg = _make_handshake_cfg(contract)

    with mock.patch("pancakebot.runtime.engine.sleep_seconds") as m_sleep:
        locked_r, open_r, ep, open_rd = engine._epoch_handshake(cfg)

    assert ep == 101
    assert locked_r.lock_price == 350.0
    assert int(open_r.lock_at) == 1_700_000_600
    assert open_rd is open_settled
    assert m_sleep.call_count == 1
    assert m_sleep.call_args_list[0].args == (2,)
