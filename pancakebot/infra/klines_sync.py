from __future__ import annotations

from pancakebot.infra.binance_us_client import BinanceUsClient
from pancakebot.infra.klines_store import KlinesStore
from pancakebot.domain.types import Kline
from pancakebot.core.errors import InvariantError


_ONE_MINUTE_MS = 60_000
_BINANCE_MAX_LIMIT = 1000


def sync_klines(
    *,
    client: BinanceUsClient,
    store: KlinesStore,
    symbol: str,
    end_time_ms: int,
) -> int:
    """Sync fully-closed 1m klines into the store up to end_time_ms.

    end_time_ms is treated as an exclusive upper bound on the *kline open time*.
    """

    if int(end_time_ms) <= 0:
        raise InvariantError("klines_sync_end_time_invalid")

    last_open = store.latest_open_time_ms()
    if last_open is None:
        # If empty, start from end_time_ms - 1000 minutes.
        # The caller is expected to run a bulk sync separately for deep history.
        start_open = int(end_time_ms) - int(_ONE_MINUTE_MS) * 1000
    else:
        start_open = int(last_open) + int(_ONE_MINUTE_MS)

    if int(start_open) >= int(end_time_ms):
        return 0

    appended_total = 0
    cursor = int(start_open)

    while cursor < int(end_time_ms):
        batch = client.fetch_1m_klines(
            symbol=str(symbol),
            start_time_ms=int(cursor),
            end_time_ms=int(end_time_ms),
            limit=int(_BINANCE_MAX_LIMIT),
        )
        if not batch:
            break

        # Binance US includes klines whose open_time >= startTime and <= endTime.
        # We ensure strict store append ordering by filtering anything <= store tail.
        tail = store.latest_open_time_ms()
        new_batch: list[Kline] = []
        for k in batch:
            if tail is not None and int(k.open_time_ms) <= int(tail):
                continue
            new_batch.append(k)

        if not new_batch:
            break

        appended_total += store.append_many(new_batch)
        cursor = int(new_batch[-1].open_time_ms) + int(_ONE_MINUTE_MS)

    return int(appended_total)


def backfill_klines(
    *,
    client: BinanceUsClient,
    store: KlinesStore,
    symbol: str,
    start_open_time_ms: int,
    end_open_time_ms: int,
) -> int:
    """Backfill closed klines from start_open_time_ms up to end_open_time_ms (exclusive)."""
    if int(start_open_time_ms) < 0 or int(end_open_time_ms) <= 0:
        raise InvariantError("klines_backfill_bounds_invalid")
    if int(end_open_time_ms) <= int(start_open_time_ms):
        return 0
    if store.latest_open_time_ms() is not None:
        raise InvariantError("klines_backfill_requires_empty_store")

    appended_total = 0
    cursor = int(start_open_time_ms)

    while cursor < int(end_open_time_ms):
        batch = client.fetch_1m_klines(
            symbol=str(symbol),
            start_time_ms=int(cursor),
            end_time_ms=int(end_open_time_ms),
            limit=int(_BINANCE_MAX_LIMIT),
        )
        if not batch:
            break

        appended_total += store.append_many(batch)
        cursor = int(batch[-1].open_time_ms) + int(_ONE_MINUTE_MS)

    return int(appended_total)


def ensure_klines_coverage(
    *,
    client: BinanceUsClient,
    store: KlinesStore,
    symbol: str,
    start_open_time_ms: int,
    end_open_time_ms: int,
) -> int:
    """Ensure [start_open_time_ms, end_open_time_ms) is present in the kline store."""
    if int(start_open_time_ms) < 0 or int(end_open_time_ms) <= 0:
        raise InvariantError("klines_coverage_bounds_invalid")
    if int(end_open_time_ms) <= int(start_open_time_ms):
        return 0

    changed = 0
    earliest_open = store.earliest_open_time_ms()
    if earliest_open is None:
        changed += backfill_klines(
            client=client,
            store=store,
            symbol=symbol,
            start_open_time_ms=int(start_open_time_ms),
            end_open_time_ms=int(end_open_time_ms),
        )
    elif int(earliest_open) > int(start_open_time_ms):
        changed += _prepend_older_klines(
            client=client,
            store=store,
            symbol=symbol,
            start_open_time_ms=int(start_open_time_ms),
            stop_open_time_ms=int(earliest_open),
        )

    changed += sync_klines(
        client=client,
        store=store,
        symbol=symbol,
        end_time_ms=int(end_open_time_ms),
    )
    return int(changed)


def _prepend_older_klines(
    *,
    client: BinanceUsClient,
    store: KlinesStore,
    symbol: str,
    start_open_time_ms: int,
    stop_open_time_ms: int,
) -> int:
    """Prepend older klines for [start_open_time_ms, stop_open_time_ms)."""
    if int(stop_open_time_ms) <= int(start_open_time_ms):
        return 0

    prepended_total = 0
    cursor_end = int(stop_open_time_ms)
    window_size_ms = int(_ONE_MINUTE_MS) * int(_BINANCE_MAX_LIMIT)

    while int(cursor_end) > int(start_open_time_ms):
        cursor_start = max(int(start_open_time_ms), int(cursor_end) - int(window_size_ms))

        batch = client.fetch_1m_klines(
            symbol=str(symbol),
            start_time_ms=int(cursor_start),
            end_time_ms=int(cursor_end),
            limit=int(_BINANCE_MAX_LIMIT),
        )

        filtered: list[Kline] = []
        for k in batch:
            ot = int(k.open_time_ms)
            if int(cursor_start) <= int(ot) < int(cursor_end):
                filtered.append(k)

        if filtered:
            filtered.sort(key=lambda x: int(x.open_time_ms))

            prev: int | None = None
            for idx, k in enumerate(filtered):
                ot = int(k.open_time_ms)
                if prev is not None and ot <= int(prev):
                    raise InvariantError(f"prepend_klines_not_strictly_increasing: idx={idx} got={ot} prev={prev}")
                prev = ot

            head = store.earliest_open_time_ms()
            if head is None:
                raise InvariantError("prepend_klines_requires_existing_store")
            if int(filtered[-1].open_time_ms) >= int(head):
                raise InvariantError("prepend_klines_overlaps_store_head")

            prepended_total += store.prepend_many(filtered)

        # Move to the next older non-overlapping window even if this window had no rows.
        cursor_end = int(cursor_start)

    return int(prepended_total)
