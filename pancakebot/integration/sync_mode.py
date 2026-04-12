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
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pancakebot.config.app_config import AppConfig
from pancakebot.core.errors import InvariantError, TransientGraphError
from pancakebot.core.logging import info
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.closed_rounds_sync import sync_closed_rounds
from pancakebot.infra.graph_client import GraphClient
from pancakebot.infra.kline_store import KlineStore
from pancakebot.runtime.runtime_loop import required_runtime_sync_cache_n
from pancakebot.runtime.sleep import sleep_seconds

_TRANSIENT_NETWORK_DELAY_SECONDS = 10

_OKX_BASE = "https://www.okx.com"
_KLINES_PER_ROUND = 40  # Must match momentum_gate._CANDLE_COUNT
_FETCH_WORKERS = 2      # concurrent OKX fetches per asset (4 total with both assets)
_FETCH_RETRIES = 3      # retry failed OKX requests
_RETRY_DELAY_S = 1.0    # base delay between retries (doubles each attempt)

_SPOT_KLINES_PATH = Path("var/cutoff_spot_prices.jsonl")
_BTC_KLINES_PATH = Path("var/btc_spot_prices.jsonl")


@dataclass(frozen=True, slots=True)
class SyncSummary:
    warmup_rounds: int
    cache_n: int
    stored_closed_round_count: int
    earliest_closed_epoch: int
    latest_closed_epoch: int
    spot_klines_synced: int
    btc_klines_synced: int


def sync_runtime_market_data(
    *,
    cfg: AppConfig,
    graph: GraphClient,
    round_store: ClosedRoundsStore,
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

    spot_store = KlineStore(str(_SPOT_KLINES_PATH))
    btc_store = KlineStore(str(_BTC_KLINES_PATH))

    with ThreadPoolExecutor(max_workers=2) as pool:
        spot_fut = pool.submit(
            _sync_1s_klines,
            rounds=tail_rounds, inst_id="BNB-USDT",
            store=spot_store, label="BNB",
        )
        btc_fut = pool.submit(
            _sync_1s_klines,
            rounds=tail_rounds, inst_id="BTC-USDT",
            store=btc_store, label="BTC",
        )
        spot_synced = spot_fut.result()
        btc_synced = btc_fut.result()

    return SyncSummary(
        warmup_rounds=int(warmup_rounds),
        cache_n=int(cache_n),
        stored_closed_round_count=int(stored_closed_round_count),
        earliest_closed_epoch=int(earliest_closed_epoch),
        latest_closed_epoch=int(latest_closed_epoch),
        spot_klines_synced=int(spot_synced),
        btc_klines_synced=int(btc_synced),
    )


def _sync_1s_klines(
    *,
    rounds: list,
    inst_id: str,
    store: KlineStore,
    label: str,
) -> int:
    """Fetch 1s OKX klines for rounds not yet in the store. Returns count synced.

    Uses _FETCH_WORKERS concurrent threads for fetching, then sorts results
    by epoch and appends to the store in strict ascending order.
    Only successfully-fetched records are written; previous error records
    are purged on startup so they get retried.
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

    info(
        "SYNC",
        "1S_KL",
        label,
        msg=f"Fetching {len(remaining)} rounds ({len(done_epochs)} already done) workers={_FETCH_WORKERS}",
    )

    def _fetch_one(rnd) -> dict | None:
        epoch = int(rnd.epoch)
        lock_at = rnd.lock_at
        if lock_at is None:
            return None
        cutoff_ms = int(lock_at) * 1000 - 4000
        klines = _fetch_1s_klines(inst_id=inst_id, anchor_ms=cutoff_ms)
        if klines is None:
            return None
        return {"epoch": epoch, "lock_at": int(lock_at), "klines_1s": klines}

    # Fetch in parallel, collect all results.
    fetched: list[dict] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, rnd): rnd for rnd in remaining}
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            try:
                rec = fut.result()
            except Exception as exc:
                errors += 1
                info("SYNC", "1S_KL", label, msg=f"Worker error: {exc}")
                continue
            if rec is None:
                errors += 1
                continue
            fetched.append(rec)
            if completed % 200 == 0:
                info(
                    "SYNC", "1S_KL", label,
                    msg=f"  {completed}/{len(remaining)} fetched ({errors} errors)",
                )

    if not fetched:
        if errors > 0:
            info("SYNC", "1S_KL", label,
                 msg=f"WARNING: all {errors} epochs failed — re-run sync to retry")
        return 0

    # Sort by epoch for ordered insertion.
    fetched.sort(key=lambda r: int(r["epoch"]))

    latest_on_disk = store.load_latest_epoch()

    if latest_on_disk is None and not store.exists():
        # Fresh store — write all fetched records.
        store.write_new(fetched)
    elif all(int(r["epoch"]) > (latest_on_disk or 0) for r in fetched):
        # All new records are strictly after what's on disk — simple append.
        prev = latest_on_disk if latest_on_disk is not None else 0
        store.append_after(prev, fetched)
    else:
        # Some records fill gaps before latest_on_disk — merge and rewrite.
        existing = list(store.iter_records())
        existing_epochs = {int(r["epoch"]) for r in existing}
        merged = existing + [r for r in fetched if int(r["epoch"]) not in existing_epochs]
        merged.sort(key=lambda r: int(r["epoch"]))
        store.rewrite(merged)
        info("SYNC", "1S_KL", label, msg=f"Merged {len(fetched)} new records into store (gap-fill)")

    if errors > 0:
        info("SYNC", "1S_KL", label,
             msg=f"WARNING: {errors} epochs failed — re-run sync to retry them")

    info("SYNC", "1S_KL", label, msg=f"Done: {len(fetched)} synced, {errors} failed")
    return len(fetched)


def _fetch_1s_klines(*, inst_id: str, anchor_ms: int) -> list[list] | None:
    """Fetch 1s klines ending just before anchor_ms from OKX.

    Returns the *_KLINES_PER_ROUND* completed candles with open_time
    < anchor_ms, matching exactly what the live path fetches via the
    OKX ``after`` parameter.  Tries history-candles first, falls back
    to the live candles endpoint.  Each endpoint is retried up to
    _FETCH_RETRIES times with exponential backoff.

    Returns list of [ts_ms, open, high, low, close, volume] sorted
    oldest-first, or None on failure.
    """
    after_ms = anchor_ms
    for endpoint in ("history-candles", "candles"):
        url = (
            f"{_OKX_BASE}/api/v5/market/{endpoint}"
            f"?instId={inst_id}&bar=1s&limit={_KLINES_PER_ROUND}"
            f"&after={after_ms}"
        )
        for attempt in range(_FETCH_RETRIES):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PancakeBot/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError, OSError):
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
