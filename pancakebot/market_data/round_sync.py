from __future__ import annotations

import math
from pathlib import Path

from pancakebot.market_data.graph_client import GraphClient
from pancakebot.market_data.round_store import ClosedRoundsStore
from pancakebot.errors import InvariantError
from pancakebot.log import info


def sync_closed_rounds(*, graph: GraphClient, store: ClosedRoundsStore, cache_n: int) -> None:
    """Ensure the closed rounds JSONL store contains a bounded window of usable closed rounds.

    The store is the on-disk persistence layer consumed by backtests.
    This sync ensures we have at least `cache_n` usable closed rounds on disk.
    """
    if cache_n <= 0:
        raise InvariantError("cache_n_nonpositive")

    # Determine the target epoch window (usable closed rounds only).
    end_epoch = graph.fetch_latest_usable_closed_epoch()
    if end_epoch <= 0:
        raise InvariantError("sync_end_epoch_nonpositive")

    page_size = 1000
    cache_n_page_ceil = math.ceil(cache_n / page_size) * page_size

    # Use an estimated minimum initial rounds needed (assuming all are usable), rounded up to the nearest page size multiple.
    start_epoch = max(end_epoch - cache_n_page_ceil + 1, 1)
    fetch_type = "initial"

    # If the store already exists, then append all rounds newer than our latest (to ensure continuity).
    if store.exists():
        latest_on_disk = store.load_latest_epoch()
        start_epoch = latest_on_disk + 1
        fetch_type = "newer"

    if start_epoch <= end_epoch:
        # Append the estimated minimum rounds needed (or all newer rounds).
        _fetch_and_append_range(graph=graph, store=store, start_epoch=start_epoch, end_epoch=end_epoch, fetch_type=fetch_type)

    stored_n = store.count_rounds()
    if stored_n >= cache_n:
        return
    if stored_n <= 0:
        raise InvariantError("sync_store_empty")

    older_end = store.load_earliest_epoch() - 1
    if older_end <= 0:
        raise InvariantError("sync_insufficient_closed_rounds")

    needed_n = cache_n - stored_n

    # Use an estimated minimum older rounds needed (assuming all are usable).
    older_start = older_end - needed_n + 1
    fetch_type = "older"

    # Use a temp file to store older rounds to minimize disk reads/writes.
    # The temp file will be prepended to the closed rounds file (atomically) at the end.
    store_path = Path(store.path_jsonl)
    tmp_path = store_path.with_suffix(store_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    tmp_store = ClosedRoundsStore(path_jsonl=str(tmp_path))

    # Append the estimated minimum older rounds needed to the temp file.
    _fetch_and_append_range(graph=graph, store=tmp_store, start_epoch=older_start, end_epoch=older_end, fetch_type=fetch_type)

    missing_n = needed_n - tmp_store.count_rounds()
    if missing_n > 0:
        info(
            "CORE",
            "STORE",
            "SYNC",
            msg=f"Ensuring store contains enough closed rounds: missing_n={missing_n}",
        )
        # Ensure that the desired cache size is covered.
        _ensure_min_count_by_scanning_older(graph=graph, store=tmp_store, needed_n=needed_n)

    # Stream-copy existing store to tmp.
    with tmp_path.open("ab") as out_f, store_path.open("rb") as in_f:
        while True:
            chunk = in_f.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)

    tmp_path.replace(store_path)


def _ensure_min_count_by_scanning_older(*, graph: GraphClient, store: ClosedRoundsStore, needed_n: int) -> None:
    """Ensure the store has at least cache_n usable closed rounds.

    The epoch-window estimate (end_epoch - cache_n + 1) assumes every epoch is usable. In reality,
    the Graph returns only usable closed rounds; a wide epoch range can contain far fewer usable
    rounds than its width. This function scans older epochs in fixed-size windows until the store
    reaches cache_n, or we hit epoch=1.

    No approximation: we never synthesize rounds; we only ingest returned usable rounds.
    """
    page_size = 1000
    stored_n = store.count_rounds()
    scan_end = store.load_earliest_epoch() - 1

    while stored_n < needed_n:
        if scan_end <= 0:
            raise InvariantError("sync_insufficient_closed_rounds")

        scan_start = max(scan_end - page_size + 1, 1)

        info(
            "CORE",
            "STORE",
            "SYNC",
            msg=f"Fetching additional older closed rounds: range=[{scan_start}..{scan_end}]",
        )
        rounds = graph.fetch_closed_rounds(
            order="asc",
            epoch_gte=scan_start,
            epoch_lte=scan_end,
            first=page_size,
            skip=0,
        )

        if rounds:
            earliest_on_disk = store.load_earliest_epoch()

            prev: int | None = None
            for idx, r in enumerate(rounds):
                if prev is not None and r.epoch <= prev:
                    raise InvariantError(f"older_scan_not_increasing: idx={idx} got={r.epoch} prev={prev}")
                if r.epoch >= earliest_on_disk:
                    raise InvariantError("older_scan_overlaps_store")
                prev = r.epoch

            replace_path = str(Path(store.path_jsonl).with_suffix(".prepend.tmp"))
            store.replace_with_prepended_chunk(rounds, replace_path=replace_path)
            stored_n = store.count_rounds()

        # Always advance the scan window, even if this window returned 0 usable rounds.
        scan_end = scan_start - 1


def _fetch_and_append_range(*, graph: GraphClient, store: ClosedRoundsStore, start_epoch: int, end_epoch: int, fetch_type: str) -> int:
    if start_epoch <= 0 or end_epoch <= 0 or start_epoch > end_epoch:
        raise InvariantError("fetch_append_range_invalid")

    # Deterministic epoch-window scanning.
    #
    # IMPORTANT: We do not use Graph first/skip pagination here. The Graph query includes a
    # server-side filter for "usable" rounds, which can cause pages to return fewer than the
    # requested size. Using skip/first over a filtered result set can re-emit earlier epochs and
    # break strict-ascending store appends. Scanning fixed epoch windows avoids duplicates.
    page_size = 1000
    fetched_total = 0

    is_new_store = not store.exists()
    prev_epoch_on_disk: int | None = None
    if not is_new_store:
        prev_epoch_on_disk = store.load_latest_epoch()
        if prev_epoch_on_disk is None:
            raise InvariantError("append_requires_existing_store")

    window_start = start_epoch
    while window_start <= end_epoch:
        window_end = window_start + page_size - 1
        if window_end > end_epoch:
            window_end = end_epoch

        info(
            "CORE",
            "STORE",
            "SYNC",
            msg=f"Fetching {fetch_type} closed rounds: range=[{window_start}..{window_end}]",
        )
        rounds = graph.fetch_closed_rounds(
            order="asc",
            epoch_gte=window_start,
            epoch_lte=window_end,
            first=page_size,
            skip=0,
        )

        if rounds:
            prev: int | None = None
            for idx, r in enumerate(rounds):
                if r.epoch < window_start or r.epoch > window_end:
                    raise InvariantError(
                        f"closed_rounds_epoch_out_of_requested_bounds: idx={idx} got={r.epoch} range=[{window_start}..{window_end}]"
                    )
                if prev is not None and r.epoch <= prev:
                    raise InvariantError(f"closed_rounds_not_strictly_increasing: idx={idx} got={r.epoch} prev={prev}")
                prev = r.epoch

            if prev_epoch_on_disk is None:
                filtered = rounds
            else:
                filtered = [r for r in rounds if r.epoch > prev_epoch_on_disk]

            if filtered:
                if is_new_store:
                    store.write_new_store(filtered)
                    is_new_store = False
                    prev_epoch_on_disk = filtered[-1].epoch
                else:
                    if prev_epoch_on_disk is None:
                        raise InvariantError("append_requires_existing_store")
                    prev_epoch_on_disk = store.append_rounds_after(prev_epoch_on_disk, filtered)
                fetched_total += len(filtered)

        window_start = window_end + 1

    return fetched_total
