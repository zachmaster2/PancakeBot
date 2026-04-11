"""Sync runtime market data: closed rounds + 1s spot klines for backtest.

Fetches closed rounds from The Graph, then fetches BNB + BTC 1s klines
from OKX (100 candles per round, anchored at lockAt) for any rounds
not already present in the output files.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from pancakebot.config.app_config import AppConfig
from pancakebot.core.errors import InvariantError, TransientGraphError
from pancakebot.core.logging import info
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.closed_rounds_sync import sync_closed_rounds
from pancakebot.infra.graph_client import GraphClient
from pancakebot.runtime.runtime_loop import required_runtime_sync_cache_n
from pancakebot.runtime.sleep import sleep_seconds

_TRANSIENT_NETWORK_DELAY_SECONDS = 10

_OKX_BASE = "https://www.okx.com"
_KLINES_PER_ROUND = 40  # lockAt-44 through lockAt-5 (40 completed candles)
_REQUEST_DELAY = 0.13  # rate limit between OKX requests

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

    # Phase 2: Sync BNB + BTC 1s klines (100 per round, anchored at lockAt).
    tail_rounds = rounds_all[-cache_n:]

    spot_synced = _sync_1s_klines(
        rounds=tail_rounds,
        inst_id="BNB-USDT",
        out_path=_SPOT_KLINES_PATH,
        label="BNB",
    )
    btc_synced = _sync_1s_klines(
        rounds=tail_rounds,
        inst_id="BTC-USDT",
        out_path=_BTC_KLINES_PATH,
        label="BTC",
    )

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
    out_path: Path,
    label: str,
) -> int:
    """Fetch 1s OKX klines for rounds not yet in out_path. Returns count synced."""
    done_epochs: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_epochs.add(int(json.loads(line)["epoch"]))

    remaining = [r for r in rounds if int(r.epoch) not in done_epochs]
    if not remaining:
        info("SYNC", "1S_KL", label, msg=f"All {len(done_epochs)} epochs already synced")
        return 0

    info(
        "SYNC",
        "1S_KL",
        label,
        msg=f"Fetching {len(remaining)} rounds ({len(done_epochs)} already done)",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fetched = 0
    errors = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, rnd in enumerate(remaining):
            epoch = int(rnd.epoch)
            lock_at = rnd.lock_at
            if lock_at is None:
                continue
            lock_at_ms = int(lock_at) * 1000

            cutoff_ms = lock_at_ms - 4000  # lockAt - cutoff_seconds
            klines = _fetch_1s_klines(inst_id=inst_id, anchor_ms=cutoff_ms)

            if klines is None:
                errors += 1
                rec = {"epoch": epoch, "lock_at": int(lock_at), "klines_1s": None, "error": True}
            else:
                rec = {"epoch": epoch, "lock_at": int(lock_at), "klines_1s": klines, "error": False}

            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fetched += 1

            if (i + 1) % 50 == 0:
                info(
                    "SYNC",
                    "1S_KL",
                    label,
                    msg=f"  {i + 1}/{len(remaining)} (errors={errors})",
                )

            time.sleep(_REQUEST_DELAY)

    info("SYNC", "1S_KL", label, msg=f"Done: {fetched} fetched, {errors} errors")
    return fetched


def _fetch_1s_klines(*, inst_id: str, anchor_ms: int) -> list[list] | None:
    """Fetch 1s klines ending just before anchor_ms from OKX.

    Returns the *_KLINES_PER_ROUND* completed candles with open_time
    < anchor_ms, matching exactly what the live path fetches via the
    OKX ``after`` parameter.  Tries history-candles first, falls back
    to the live candles endpoint.

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
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PancakeBot/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError):
            continue

        if body.get("code") != "0" or not body.get("data"):
            continue

        rows = body["data"]  # newest first
        if len(rows) < _KLINES_PER_ROUND * 0.9:
            continue

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
