"""Sync runtime market data: closed rounds + 1s spot klines for backtest.

Fetches closed rounds from The Graph, then fetches BNB + BTC 1s klines
from OKX for any rounds not already present in the kline stores.

Kline fetching uses parallel workers within each asset and runs both
assets concurrently.  Records are collected, sorted by epoch, and
appended in strict ascending order — matching the closed rounds store
pattern.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pancakebot.config import AppConfig
from pancakebot.errors import InvariantError, TransientGraphError
from pancakebot.log import info
from pancakebot.market_data.round_store import ClosedRoundsStore
from pancakebot.market_data.round_sync import sync_closed_rounds
from pancakebot.market_data.graph_client import GraphClient
from pancakebot.market_data.kline_store import KlineStore
from pancakebot.runtime.config import required_runtime_sync_cache_n
from time import sleep as sleep_seconds

_TRANSIENT_NETWORK_DELAY_SECONDS = 10

_KLINES_PER_ROUND = 31  # Must match momentum_gate._CANDLE_COUNT
_FETCH_WORKERS = 4      # concurrent OKX fetches per batch
_FETCH_RETRIES = 3      # retry failed OKX requests
_RETRY_DELAY_S = 1.0    # base delay between retries (doubles each attempt)

# Global rate limiter: OKX allows 20 req/2s per endpoint per IP = 10/s.
# All sync threads share this to avoid 429 errors.
_OKX_RATE_LIMIT_PER_SEC = 8  # safely under 10/s
_rate_lock = threading.Lock()
_rate_last = 0.0


def _rate_acquire() -> None:
    """Block until we can make another OKX request without exceeding rate limit."""
    global _rate_last
    min_interval = 1.0 / _OKX_RATE_LIMIT_PER_SEC
    with _rate_lock:
        now = time.monotonic()
        wait = _rate_last + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _rate_last = time.monotonic()

_BNB_KLINES_PATH = Path("var/bnb_spot_prices.jsonl")
_BTC_KLINES_PATH = Path("var/btc_spot_prices.jsonl")
_ETH_KLINES_PATH = Path("var/eth_spot_prices.jsonl")
_SOL_KLINES_PATH = Path("var/sol_spot_prices.jsonl")


@dataclass(frozen=True, slots=True)
class SyncSummary:
    warmup_rounds: int
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
    okx_client: object,
) -> SyncSummary:
    warmup_rounds = int(required_runtime_sync_cache_n())
    cache_n = max(int(warmup_rounds), int(cfg.backtest.simulation_size))

    info(
        "CORE",
        "SYNC",
        "START",
        msg=f"Sync setup: warmup_rounds={int(warmup_rounds)} simulation_size={int(cfg.backtest.simulation_size)} closed_cache_needed={int(cache_n)}",
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

    # Phase 2: Sync BNB + BTC 1s klines in parallel (anchored at lockAt).
    tail_rounds = rounds_all[-cache_n:]

    bnb_store = KlineStore(str(_BNB_KLINES_PATH))
    btc_store = KlineStore(str(_BTC_KLINES_PATH))
    eth_store = KlineStore(str(_ETH_KLINES_PATH))
    sol_store = KlineStore(str(_SOL_KLINES_PATH))

    cutoff_s = int(cfg.cutoff_seconds)
    # All 4 pairs run in parallel — the shared _rate_acquire() limiter
    # throttles total OKX requests to 8/s across all threads.
    with ThreadPoolExecutor(max_workers=4) as pool:
        bnb_fut = pool.submit(
            _sync_1s_klines,
            rounds=tail_rounds, inst_id="BNB-USDT",
            store=bnb_store, label="BNB", cutoff_seconds=cutoff_s,
            okx_client=okx_client,
        )
        btc_fut = pool.submit(
            _sync_1s_klines,
            rounds=tail_rounds, inst_id="BTC-USDT",
            store=btc_store, label="BTC", cutoff_seconds=cutoff_s,
            okx_client=okx_client,
        )
        eth_fut = pool.submit(
            _sync_1s_klines,
            rounds=tail_rounds, inst_id="ETH-USDT",
            store=eth_store, label="ETH", cutoff_seconds=cutoff_s,
            okx_client=okx_client,
        )
        sol_fut = pool.submit(
            _sync_1s_klines,
            rounds=tail_rounds, inst_id="SOL-USDT",
            store=sol_store, label="SOL", cutoff_seconds=cutoff_s,
            okx_client=okx_client,
        )
        bnb_synced = bnb_fut.result()
        btc_synced = btc_fut.result()
        eth_synced = eth_fut.result()
        sol_synced = sol_fut.result()

    # Phase 3: Integrity — trim all stores to the exact intersection so
    # every closed round has klines in ALL 4 stores.  Retry transient
    # failures first, then trim any epochs that genuinely have no data.
    all_stores = [
        ("BNB-USDT", bnb_store, "BNB-retry"),
        ("BTC-USDT", btc_store, "BTC-retry"),
        ("ETH-USDT", eth_store, "ETH-retry"),
        ("SOL-USDT", sol_store, "SOL-retry"),
    ]
    _MAX_RETRY_PASSES = 3

    for retry_pass in range(_MAX_RETRY_PASSES):
        covered_epochs = set.intersection(*(s.load_done_epochs() for _, s, _ in all_stores))
        all_epochs = {int(r.epoch) for r in tail_rounds}
        uncovered = all_epochs - covered_epochs
        if not uncovered:
            break

        info(
            "CORE", "SYNC", "INTEG",
            msg=(
                f"Retry pass {retry_pass + 1}/{_MAX_RETRY_PASSES}: "
                f"{len(uncovered)} rounds still missing klines "
                f"(epochs {min(uncovered)}..{max(uncovered)})"
            ),
        )
        uncovered_rounds = [r for r in tail_rounds if int(r.epoch) in uncovered]
        for inst_id, store, label in all_stores:
            _sync_1s_klines(
                rounds=uncovered_rounds, inst_id=inst_id,
                store=store, label=label, cutoff_seconds=cutoff_s,
                okx_client=okx_client,
            )

    # After retries, trim stores to the exact 5-way intersection:
    # every round must have klines in ALL stores.
    all_round_epochs = {int(r.epoch) for r in rounds_all}
    all_kline_epoch_sets = [s.load_done_epochs() for _, s, _ in all_stores]
    valid_epochs = all_round_epochs & set.intersection(*all_kline_epoch_sets)

    trimmed_rounds = len(all_round_epochs) - len(valid_epochs)
    if trimmed_rounds > 0:
        _trim_closed_rounds(round_store, valid_epochs)
        info("CORE", "SYNC", "TRIM",
             msg=f"Trimmed closed_rounds: {len(all_round_epochs)} -> {len(valid_epochs)}")
    for inst_id, store, _ in all_stores:
        store_epochs = store.load_done_epochs()
        trimmed = len(store_epochs) - len(valid_epochs)
        if trimmed > 0:
            _trim_kline_store(store, valid_epochs)
            label = inst_id.split("-")[0]  # "BNB-USDT" -> "BNB"
            info("CORE", "SYNC", "TRIM",
                 msg=f"Trimmed {label} klines: {len(store_epochs)} -> {len(valid_epochs)}")

    # Re-read final state after any trimming.
    final_rounds = list(round_store.iter_closed_rounds())
    final_round_epochs = {int(r.epoch) for r in final_rounds}
    for inst_id, store, _ in all_stores:
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
            f"[{min(final_round_epochs)}..{max(final_round_epochs)}] "
            f"(trimmed {trimmed_rounds} rounds)"
        ),
    )

    stored_closed_round_count = len(final_rounds)
    earliest_closed_epoch = int(final_rounds[0].epoch)
    latest_closed_epoch = int(final_rounds[-1].epoch)

    return SyncSummary(
        warmup_rounds=int(warmup_rounds),
        cache_n=int(cache_n),
        stored_closed_round_count=int(stored_closed_round_count),
        earliest_closed_epoch=int(earliest_closed_epoch),
        latest_closed_epoch=int(latest_closed_epoch),
        bnb_klines_synced=int(bnb_synced),
        btc_klines_synced=int(btc_synced),
        eth_klines_synced=int(eth_synced),
        sol_klines_synced=int(sol_synced),
    )


def _trim_closed_rounds(store: ClosedRoundsStore, valid_epochs: set[int]) -> None:
    """Remove closed rounds whose epochs are not in valid_epochs (atomic)."""
    import os as _os

    all_rounds = list(store.iter_closed_rounds())
    kept = [r for r in all_rounds if int(r.epoch) in valid_epochs]
    kept.sort(key=lambda r: int(r.epoch))

    tmp = store._path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r.to_json(), separators=(",", ":")) + "\n")
    _os.replace(tmp, store._path)

    info("CORE", "SYNC", "TRIM",
         msg=f"Trimmed closed rounds: {len(all_rounds)} -> {len(kept)}")


def _trim_kline_store(store: KlineStore, valid_epochs: set[int]) -> None:
    """Remove kline records whose epochs are not in valid_epochs (atomic)."""
    all_records = list(store.iter_records())
    kept = [r for r in all_records if int(r["epoch"]) in valid_epochs]
    kept.sort(key=lambda r: int(r["epoch"]))
    store.rewrite(kept)

    info("CORE", "SYNC", "TRIM",
         msg=f"Trimmed kline store: {len(all_records)} -> {len(kept)}")


_BATCH_SIZE = 50  # fetch+flush in small ordered batches for resumability


def _sync_1s_klines(
    *,
    rounds: list,
    inst_id: str,
    store: KlineStore,
    label: str,
    cutoff_seconds: int = 2,
    okx_client: object,
) -> int:
    """Fetch 1s OKX klines for rounds not yet in the store. Returns count synced.

    Split into two passes to avoid O(n²) merge-rewrites:
      1. **Append pass** — epochs AFTER the store's latest: fetched in small
         ordered batches, each appended and flushed immediately (resumable).
      2. **Prepend pass** — epochs BEFORE the store's earliest: fetched in
         batches into a staging file, then prepended atomically at the end.

    Both passes survive interruption: the append pass writes incrementally,
    and the prepend staging file is resumed on restart.
    """
    # Purge any legacy error records so those epochs get retried.
    purged = store.purge_and_rewrite()
    if purged > 0:
        info("SYNC", "1S_KL", label, msg=f"Purged {purged} error records for retry")

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
        # Fresh store — everything goes into append.
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
    total_errors = 0

    # --- Pass 1: Append (epochs after existing store) ---
    if append_rounds:
        synced, errors = _fetch_and_append(
            rounds_asc=append_rounds, inst_id=inst_id, store=store, label=label,
            latest_on_disk=latest_on_disk, done_count=len(done_epochs),
            cutoff_seconds=cutoff_seconds, okx_client=okx_client,
        )
        total_synced += synced
        total_errors += errors

    # --- Pass 2: Prepend (epochs before existing store) ---
    if prepend_rounds:
        staging_path = store.path_jsonl + f".prepend_staging"
        synced, errors = _fetch_to_staging(
            rounds_asc=prepend_rounds, inst_id=inst_id,
            staging_path=staging_path, label=label,
            cutoff_seconds=cutoff_seconds, okx_client=okx_client,
        )
        total_errors += errors
        if synced > 0:
            _prepend_staging_to_store(store=store, staging_path=staging_path, label=label)
            total_synced += synced

    if total_errors > 0:
        info("SYNC", "1S_KL", label,
             msg=f"WARNING: {total_errors} epochs failed — re-run sync to retry them")

    info("SYNC", "1S_KL", label, msg=f"Done: {total_synced} synced, {total_errors} failed")
    return total_synced


def _fetch_and_append(
    *, rounds_asc: list, inst_id: str, store: KlineStore, label: str,
    latest_on_disk: int | None, done_count: int, cutoff_seconds: int = 2,
    okx_client: object,
) -> tuple[int, int]:
    """Fetch epochs in ordered batches and append each to store immediately."""
    synced = 0
    errors = 0
    prev_epoch = latest_on_disk if latest_on_disk is not None else 0

    for batch_start in range(0, len(rounds_asc), _BATCH_SIZE):
        batch = rounds_asc[batch_start : batch_start + _BATCH_SIZE]
        if batch_start > 0:
            time.sleep(1.0)

        results, batch_errors = _fetch_batch(batch, inst_id, cutoff_seconds=cutoff_seconds, okx_client=okx_client)
        errors += batch_errors
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
                 msg=f"  append: {done_count + synced} done, {errors} errors")

    return synced, errors


def _fetch_to_staging(
    *, rounds_asc: list, inst_id: str, staging_path: str, label: str,
    cutoff_seconds: int = 2, okx_client: object,
) -> tuple[int, int]:
    """Fetch older epochs into a staging file (resumable append-only)."""
    import os as _os

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
        return len(staged_epochs), 0

    synced = len(staged_epochs)
    errors = 0

    # Append to staging file incrementally (batch by batch).
    _os.makedirs(_os.path.dirname(staging_path) or ".", exist_ok=True)
    with open(staging_path, "a", encoding="utf-8") as staging_f:
        for batch_start in range(0, len(still_needed), _BATCH_SIZE):
            batch = still_needed[batch_start : batch_start + _BATCH_SIZE]
            if batch_start > 0:
                time.sleep(1.0)

            results, batch_errors = _fetch_batch(batch, inst_id, cutoff_seconds=cutoff_seconds, okx_client=okx_client)
            errors += batch_errors

            for rec in results:
                staging_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                staging_f.flush()
                synced += 1

            if (batch_start + _BATCH_SIZE) % 200 < _BATCH_SIZE:
                info("SYNC", "1S_KL", label,
                     msg=f"  prepend staging: {synced} staged, {errors} errors")

    return synced, errors


def _prepend_staging_to_store(*, store: KlineStore, staging_path: str, label: str) -> None:
    """Merge staging file (older epochs) in front of existing store atomically."""
    import os as _os

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


def _fetch_batch(batch: list, inst_id: str, okx_client: object, cutoff_seconds: int = 2) -> tuple[list[dict], int]:
    """Fetch a batch of rounds in parallel. Returns (results, error_count)."""
    results: list[dict] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_kline, rnd, inst_id, okx_client, cutoff_seconds): rnd for rnd in batch}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception:
                errors += 1
                continue
            if rec is None:
                errors += 1
                continue
            results.append(rec)
    return results, errors


def _fetch_one_kline(rnd, inst_id: str, okx_client: object, cutoff_seconds: int = 2) -> dict | None:
    """Fetch 1s klines for a single round. Returns record dict or None."""
    epoch = int(rnd.epoch)
    lock_at = rnd.lock_at
    if lock_at is None:
        return None
    cutoff_ms = int(lock_at) * 1000 - cutoff_seconds * 1000
    klines = _fetch_1s_klines(inst_id=inst_id, anchor_ms=cutoff_ms, okx_client=okx_client)
    if klines is None:
        return None
    return {"epoch": epoch, "lock_at": int(lock_at), "klines_1s": klines}


def _fetch_1s_klines(*, inst_id: str, anchor_ms: int, okx_client: object) -> list[list] | None:
    """Fetch 1s klines ending just before anchor_ms from OKX.

    Uses the shared OkxClient session (keep-alive) for fast fetches.
    Tries history-candles first, falls back to the live candles endpoint.
    Each endpoint is retried up to _FETCH_RETRIES times.

    Returns list of [ts_ms, open, high, low, close, volume] sorted
    oldest-first, or None on failure.
    """
    params = {
        "instId": inst_id,
        "bar": "1s",
        "limit": str(_KLINES_PER_ROUND),
        "after": str(anchor_ms),
    }
    for endpoint in ("history-candles", "candles"):
        for attempt in range(_FETCH_RETRIES):
            _rate_acquire()
            body = okx_client.fetch_raw(endpoint=endpoint, params=params)

            if body is None:
                time.sleep(_RETRY_DELAY_S * (2 ** attempt))
                continue

            if body.get("code") != "0" or not body.get("data"):
                break  # non-transient: try next endpoint

            rows = body["data"]  # newest first
            if len(rows) < _KLINES_PER_ROUND * 0.9:
                break  # insufficient data: try next endpoint

            out = []
            for row in reversed(rows):
                out.append([
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                ])
            return out
    return None
