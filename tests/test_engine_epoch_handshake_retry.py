"""Tests for the zero-state retry guards inside ``engine._epoch_handshake``.

The bare ``_epoch_handshake`` retries on three zero-state conditions
within its RETRY_BACKOFF_SECONDS budget:
  - ``locked_rd.lock_ts <= 0``       (covered indirectly by integration paths)
  - ``locked_rd.lock_price_usd <= 0`` (this file)
  - ``open_rd.lock_ts <= 0``          (this file)

These appear during the fresh-spawn-during-round-transition window:
``executeRound()`` has incremented ``currentEpoch`` but not yet written
``lock_price`` for the new locked epoch / ``lock_ts`` for the new open
epoch. The RETRY_BACKOFF_SECONDS schedule is sized so the cumulative
wait crosses ``buffer_seconds + _RPC_ALIGNMENT_PADDING_SECONDS`` (~35s)
on the 5th retry, well past the chain-enforced settlement window.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.chain.prediction_contract import RoundData  # noqa: E402
from pancakebot.runtime import engine  # noqa: E402


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


def _make_handshake_cfg(contract: mock.Mock) -> mock.Mock:
    return mock.Mock(contract=contract)


def test_epoch_handshake_retries_on_locked_lock_price_zero():
    """First attempt: locked_rd.lock_price_usd == 0 -> retry path triggers.
    Second attempt: full settled state -> returns successfully."""
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
    # Exactly one retry -> one backoff sleep (RETRY_BACKOFF_SECONDS[0] = 2s).
    assert m_sleep.call_count == 1
    assert m_sleep.call_args_list[0].args == (2,)


def test_epoch_handshake_retries_on_open_lock_ts_zero():
    """First attempt: open_rd.lock_ts == 0 -> retry path triggers.
    Second attempt: full settled state -> returns successfully."""
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
