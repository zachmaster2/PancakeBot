"""Fetch closed rounds from The Graph and BNB/BTC/ETH/SOL 1s klines from OKX.

Runs the four kline fetches in parallel under a shared rate limiter,
trims the round store and kline stores to their common epoch intersection,
and returns a SyncSummary of counts.
"""
from __future__ import annotations

import json
import os as _os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pancakebot import paths as _proj_paths
from pancakebot.config import AppConfig
from pancakebot.util import InvariantError, TransientGraphError
from pancakebot.log import info
from pancakebot.market_data.round_store import ClosedRoundsStore
from pancakebot.market_data.round_sync import sync_closed_rounds
from pancakebot.market_data.graph_client import GraphClient
from pancakebot.market_data.okx_client import OkxClient, RETRY_SYNC, okx_rate_acquire
from pancakebot.market_data.kline_store import KlineStore
from time import sleep as sleep_seconds

_TRANSIENT_NETWORK_DELAY_SECONDS = 10

# Per-round 1s candle window. Anchored to ``lock_at`` only — independent of
# ``kline_cutoff_seconds`` (which is a strategy/gate-only knob, not a fetch
# parameter). Matches the 2026-04-26 rebuild's on-disk shape exactly:
#   oldest open_ts = lock_at_ms - 301_000   (300 candles, oldest-first)
#   newest open_ts = lock_at_ms - 2_000     (one extra past gate window)
# 300 candles is the maximum OKX ``/history-candles`` accepts per request.
_HISTORY_OLDEST_OFFSET_MS = 301_000
_HISTORY_NEWEST_OFFSET_MS = 2_000
_FETCH_WORKERS = 4  # concurrent OKX fetches per batch

# OKX REST rate budget moved to pancakebot.market_data.okx_client
# (``okx_rate_acquire``). Sync's bulk fetch + WSS bootstrap + WSS gap-fill
# all share the same process-wide token bucket.

_BNB_KLINES_PATH = Path(_proj_paths.BNB_SPOT_PRICES_PATH)
_BTC_KLINES_PATH = Path(_proj_paths.BTC_SPOT_PRICES_PATH)
_ETH_KLINES_PATH = Path(_proj_paths.ETH_SPOT_PRICES_PATH)
_SOL_KLINES_PATH = Path(_proj_paths.SOL_SPOT_PRICES_PATH)


@dataclass(frozen=True, slots=True)
class SyncSummary:
    cache_n: int
    stored_closed_round_count: int
    earliest_closed_epoch: int
    latest_closed_epoch: int
    bnb_klines_synced: int
    btc_klines_synced: int
    eth_klines_synced: int
    sol_klines_synced: int


def sync_runtime_market_data(
    *,
    cfg: AppConfig,
    graph: GraphClient,
    round_store: ClosedRoundsStore,
    okx_client: OkxClient,
) -> SyncSummary:
    cache_n = int(cfg.backtest.simulation_size)

    info(
        "CORE",
        "SYNC",
        "START",
        msg=f"Sync setup: simulation_size={int(cfg.backtest.simulation_size)} closed_cache_needed={int(cache_n)}",
    )

    # Phase 1: Sync closed rounds from The Graph.
    while True:
        try:
            sync_closed_rounds(
                graph=graph,
                store=round_store,
                cache_n=int(cache_n),
            )
            break
        except TransientGraphError as e:
            info(
                "CORE",
                "SYNC",
                "RETRY",
                msg=(
                    "Caught TransientGraphError during sync-only closed-round sync: "
                    f"retrying after delay err={str(e)}"
                ),
            )
            sleep_seconds(int(_TRANSIENT_NETWORK_DELAY_SECONDS))

    rounds_all = list(round_store.iter_closed_rounds())
    stored_closed_round_count = int(len(rounds_all))
    if not rounds_all:
        raise InvariantError("closed_rounds_store_empty_after_sync")

    earliest_closed_epoch = int(rounds_all[0].epoch)
    latest_closed_epoch = int(rounds_all[-1].epoch)

    info(
        "CORE",
        "SYNC",
        "ROUNDS",
        msg=(
            f"Closed rounds synced: stored_n={int(stored_closed_round_count)} "
            f"epochs=[{int(earliest_closed_epoch)}..{int(latest_closed_epoch)}]"
        ),
    )

    # Phase 2: Sync BNB + BTC + ETH + SOL 1s klines in parallel (anchored at lockAt).
    tail_rounds = rounds_all[-cache_n:]

    bnb_store = KlineStore(str(_BNB_KLINES_PATH))
    btc_store = KlineStore(str(_BTC_KLINES_PATH))
    eth_store = KlineStore(str(_ETH_KLINES_PATH))
    sol_store = KlineStore(str(_SOL_KLINES_PATH))

    # All 4 pairs run in parallel -- the shared _rate_acquire() limiter
    # throttles total OKX requests to 8/s across all threads. Note: the
    # per-round window is now derived from lock_at only; cfg.kline_cutoff_seconds
    # is intentionally NOT used here (silent-corruption fix 2026-04-27).
    with ThreadPoolExecutor(max_workers=4) as pool:
        bnb_fut = pool.submit(
            _sync_1s_klines,
            rounds=tail_rounds, inst_id="BNB-USDT",
            store=bnb_store, label="BNB",
            okx_client=okx_client,
        )
        btc_fut = pool.submit(
            _sync_1s_klines,
            rounds=tail_rounds, inst_id="BTC-USDT",
            store=btc_store, label="BTC",
            okx_client=okx_client,
        )
        eth_fut = pool.submit(
            _sync_1s_klines,
            rounds=tail_rounds, inst_id="ETH-USDT",
            store=eth_store, label="ETH",
            okx_client=okx_client,
        )
        sol_fut = pool.submit(
            _sync_1s_klines,
            rounds=tail_rounds, inst_id="SOL-USDT",
            store=sol_store, label="SOL",
            okx_client=okx_client,
        )
        bnb_synced = bnb_fut.result()
        btc_synced = btc_fut.result()
        eth_synced = eth_fut.result()
        sol_synced = sol_fut.result()

    # Phase 3: Integrity assertion -- with centralized retry inside OkxClient,
    # a successful Phase 2 means every pair has the full expected epoch set by
    # construction. Any fetch failure would have raised InvariantError already.
    # We keep the assertion as a hard-stop tripwire for unexpected drift.
    all_stores = [
        ("BNB-USDT", bnb_store),
        ("BTC-USDT", btc_store),
        ("ETH-USDT", eth_store),
        ("SOL-USDT", sol_store),
    ]
    final_rounds = list(round_store.iter_closed_rounds())
    final_round_epochs = {int(r.epoch) for r in final_rounds}
    for inst_id, store in all_stores:
        store_epochs = store.load_done_epochs()
        if final_round_epochs != store_epochs:
            raise InvariantError(
                f"sync_integrity_mismatch: rounds={len(final_round_epochs)} "
                f"{inst_id}={len(store_epochs)}"
            )

    info(
        "CORE", "SYNC", "INTEG",
        msg=(
            f"Stores aligned: {len(final_round_epochs)} epochs "
            f"[{min(final_round_epochs)}..{max(final_round_epochs)}]"
        ),
    )

    stored_closed_round_count = len(final_rounds)
    earliest_closed_epoch = int(final_rounds[0].epoch)
    latest_closed_epoch = int(final_rounds[-1].epoch)

    return SyncSummary(
        cache_n=int(cache_n),
        stored_closed_round_count=int(stored_closed_round_count),
        earliest_closed_epoch=int(earliest_closed_epoch),
        latest_closed_epoch=int(latest_closed_epoch),
        bnb_klines_synced=int(bnb_synced),
        btc_klines_synced=int(btc_synced),
        eth_klines_synced=int(eth_synced),
        sol_klines_synced=int(sol_synced),
    )


_BATCH_SIZE = 50  # fetch+flush in small ordered batches for resumability


def _sync_1s_klines(
    *,
    rounds: list,
    inst_id: str,
    store: KlineStore,
    label: str,
    okx_client: OkxClient,
) -> int:
    """Fetch 1s OKX klines for rounds not yet in the store. Returns count synced.

    Split into two passes to avoid O(n2) merge-rewrites:
      1. **Append pass** -- epochs AFTER the store's latest: fetched in small
         ordered batches, each appended and flushed immediately (resumable).
      2. **Prepend pass** -- epochs BEFORE the store's earliest: fetched in
         batches into a staging file, then prepended atomically at the end.

    Both passes survive interruption: the append pass writes incrementally,
    and the prepend staging file is resumed on restart.
    """
    done_epochs = store.load_done_epochs()
    remaining = [r for r in rounds if int(r.epoch) not in done_epochs]
    if not remaining:
        info("SYNC", "1S_KL", label, msg=f"All {len(done_epochs)} epochs already synced")
        return 0

    remaining.sort(key=lambda r: int(r.epoch))

    earliest_on_disk = store.load_earliest_epoch()
    latest_on_disk = store.load_latest_epoch()

    # Split into prepend (older) and append (newer) groups.
    if latest_on_disk is not None:
        prepend_rounds = [r for r in remaining if int(r.epoch) < earliest_on_disk]
        append_rounds = [r for r in remaining if int(r.epoch) > latest_on_disk]
    else:
        # Fresh store -- everything goes into append.
        prepend_rounds = []
        append_rounds = remaining

    total_to_fetch = len(prepend_rounds) + len(append_rounds)
    info(
        "SYNC", "1S_KL", label,
        msg=f"Fetching {total_to_fetch} rounds ({len(done_epochs)} already done) "
            f"append={len(append_rounds)} prepend={len(prepend_rounds)} "
            f"workers={_FETCH_WORKERS} batch_size={_BATCH_SIZE}",
    )

    total_synced = 0

    # --- Pass 1: Append (epochs after existing store) ---
    # Raises InvariantError on unrecoverable OKX failure (caller's sync fails loud).
    if append_rounds:
        total_synced += _fetch_and_append(
            rounds_asc=append_rounds, inst_id=inst_id, store=store, label=label,
            latest_on_disk=latest_on_disk, done_count=len(done_epochs),
            okx_client=okx_client,
        )

    # --- Pass 2: Prepend (epochs before existing store) ---
    if prepend_rounds:
        staging_path = store.path_jsonl + ".prepend_staging"
        synced = _fetch_to_staging(
            rounds_asc=prepend_rounds, inst_id=inst_id,
            staging_path=staging_path, label=label,
            okx_client=okx_client,
        )
        if synced > 0:
            _prepend_staging_to_store(store=store, staging_path=staging_path, label=label)
            total_synced += synced

    info("SYNC", "1S_KL", label, msg=f"Done: {total_synced} synced")
    return total_synced


def _fetch_and_append(
    *, rounds_asc: list, inst_id: str, store: KlineStore, label: str,
    latest_on_disk: int | None, done_count: int,
    okx_client: OkxClient,
) -> int:
    """Fetch epochs in ordered batches and append each to store immediately.

    Raises InvariantError on unrecoverable OKX failure (any batch fetch that
    exhausts retries inside OkxClient).
    """
    synced = 0
    prev_epoch = latest_on_disk if latest_on_disk is not None else 0

    for batch_start in range(0, len(rounds_asc), _BATCH_SIZE):
        batch = rounds_asc[batch_start : batch_start + _BATCH_SIZE]
        if batch_start > 0:
            time.sleep(1.0)

        results = _fetch_batch(batch, inst_id, okx_client=okx_client)
        if not results:
            continue

        results.sort(key=lambda r: int(r["epoch"]))

        # Filter to only records strictly after prev_epoch (skip gaps).
        appendable = [r for r in results if int(r["epoch"]) > prev_epoch]
        if not appendable:
            continue

        if not store.exists():
            store.write_new(appendable)
        else:
            store.append_after(prev_epoch, appendable)

        prev_epoch = int(appendable[-1]["epoch"])
        synced += len(appendable)

        if (batch_start + _BATCH_SIZE) % 200 < _BATCH_SIZE:
            info("SYNC", "1S_KL", label,
                 msg=f"  append: {done_count + synced} done")

    return synced


def _fetch_to_staging(
    *, rounds_asc: list, inst_id: str, staging_path: str, label: str,
    okx_client: OkxClient,
) -> int:
    """Fetch older epochs into a staging file (resumable append-only).

    Raises InvariantError on unrecoverable OKX failure.
    """
    # Load what's already in the staging file from a prior interrupted run.
    staged_epochs: set[int] = set()
    if _os.path.exists(staging_path):
        with open(staging_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    staged_epochs.add(int(json.loads(line)["epoch"]))
        info("SYNC", "1S_KL", label,
             msg=f"  prepend staging: {len(staged_epochs)} already staged from prior run")

    still_needed = [r for r in rounds_asc if int(r.epoch) not in staged_epochs]
    if not still_needed:
        return len(staged_epochs)

    synced = len(staged_epochs)

    # Append to staging file incrementally (batch by batch).
    _os.makedirs(_os.path.dirname(staging_path) or ".", exist_ok=True)
    with open(staging_path, "a", encoding="utf-8") as staging_f:
        for batch_start in range(0, len(still_needed), _BATCH_SIZE):
            batch = still_needed[batch_start : batch_start + _BATCH_SIZE]
            if batch_start > 0:
                time.sleep(1.0)

            results = _fetch_batch(batch, inst_id, okx_client=okx_client)

            for rec in results:
                staging_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                staging_f.flush()
                synced += 1

            if (batch_start + _BATCH_SIZE) % 200 < _BATCH_SIZE:
                info("SYNC", "1S_KL", label,
                     msg=f"  prepend staging: {synced} staged")

    return synced


def _prepend_staging_to_store(*, store: KlineStore, staging_path: str, label: str) -> None:
    """Merge staging file (older epochs) in front of existing store atomically."""
    # Read staging records, sort by epoch.
    staging_records: list[dict] = []
    with open(staging_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                staging_records.append(json.loads(line))
    staging_records.sort(key=lambda r: int(r["epoch"]))

    # Read existing store.
    existing_records = list(store.iter_records())
    existing_epochs = {int(r["epoch"]) for r in existing_records}

    # Merge: staged (older) + existing, deduplicating.
    merged = [r for r in staging_records if int(r["epoch"]) not in existing_epochs]
    merged.extend(existing_records)
    merged.sort(key=lambda r: int(r["epoch"]))

    # Atomic rewrite.
    store.rewrite(merged)

    # Clean up staging file.
    _os.remove(staging_path)
    info("SYNC", "1S_KL", label,
         msg=f"  prepended {len(staging_records)} older epochs into store")


def _fetch_batch(batch: list, inst_id: str, okx_client: OkxClient) -> list[dict]:
    """Fetch a batch of rounds in parallel.

    Returns the list of kline records. Propagates any exception raised by
    `_fetch_one_kline` (OkxClient's retry exhaustion surfaces as InvariantError).
    """
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_kline, rnd, inst_id, okx_client): rnd for rnd in batch}
        for fut in as_completed(futures):
            rec = fut.result()  # propagates exception on failure
            results.append(rec)
    return results


def _fetch_one_kline(rnd, inst_id: str, okx_client: OkxClient) -> dict:
    """Fetch 1s klines for a single round via the canonical primitive.

    Window is anchored to ``lock_at`` only — strategy-side ``cutoff_seconds``
    does NOT enter the fetch math. Produces records bit-identical to the
    2026-04-26 rebuild's on-disk shape:
      newest open_ts = lock_at_ms - 2_000
      oldest open_ts = lock_at_ms - 301_000
      300 candles, oldest-first

    Raises InvariantError if the shared retry policy in
    ``OkxClient.fetch_kline_window`` exhausts.
    """
    epoch = int(rnd.epoch)
    lock_at = int(rnd.lock_at)
    lock_at_ms = lock_at * 1000
    klines = okx_client.fetch_kline_window(
        symbol=inst_id,
        oldest_open_ms=lock_at_ms - _HISTORY_OLDEST_OFFSET_MS,
        newest_open_ms_inclusive=lock_at_ms - _HISTORY_NEWEST_OFFSET_MS,
        retry_policy=RETRY_SYNC,
        rate_acquire_fn=okx_rate_acquire,
    )
    return {"epoch": epoch, "lock_at": lock_at, "klines_1s": klines}
