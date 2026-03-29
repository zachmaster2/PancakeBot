from __future__ import annotations

from dataclasses import dataclass

from pancakebot.config.app_config import AppConfig
from pancakebot.core.errors import InvariantError, TransientGraphError
from pancakebot.core.logging import info
from pancakebot.domain.closed_rounds_cache import RollingClosedRoundsCache
from pancakebot.domain.features.schema import max_required_prior_context_rounds_size
from pancakebot.infra.binance_us_client import BinanceUsClient
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.closed_rounds_sync import sync_closed_rounds
from pancakebot.infra.graph_client import GraphClient
from pancakebot.infra.klines_store import KlinesStore
from pancakebot.infra.klines_sync import ensure_klines_coverage
from pancakebot.runtime.runtime_loop import (
    required_klines_window_for_closed_cache,
    required_runtime_sync_cache_n,
)
from pancakebot.runtime.sleep import sleep_seconds

_TRANSIENT_NETWORK_DELAY_SECONDS = 10


@dataclass(frozen=True, slots=True)
class SyncSummary:
    prior_context_rounds_required: int
    warmup_rounds: int
    cache_n: int
    stored_closed_round_count: int
    earliest_closed_epoch: int
    latest_closed_epoch: int
    kline_changed_count: int
    earliest_kline_open_time_ms: int | None
    latest_kline_open_time_ms: int | None


def sync_runtime_market_data(
    *,
    cfg: AppConfig,
    graph: GraphClient,
    round_store: ClosedRoundsStore,
    klines_store: KlinesStore,
    binance_us_client: BinanceUsClient,
    binance_us_symbol: str,
) -> SyncSummary:
    prior_context_rounds_required = int(max_required_prior_context_rounds_size())
    warmup_rounds = int(required_runtime_sync_cache_n(strategy_cfg=cfg.strategy))
    cache_n = int(warmup_rounds)

    context_desc = "target_only" if int(prior_context_rounds_required) <= 0 else f"prior_context_rounds[{int(prior_context_rounds_required)}]"
    info(
        "CORE",
        "SYNC",
        "START",
        msg=(
            f"Sync setup: prior_context_rounds_required={int(prior_context_rounds_required)} "
            f"context={context_desc} warmup_rounds={int(warmup_rounds)} "
            f"closed_cache_needed={int(cache_n)}"
        ),
    )

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

    cache_rounds = list(rounds_all[-int(cache_n) :]) if len(rounds_all) > int(cache_n) else list(rounds_all)
    closed_cache = RollingClosedRoundsCache(rounds=cache_rounds, capacity=int(cache_n))
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

    start_open_ms, end_open_ms = required_klines_window_for_closed_cache(
        closed_cache=closed_cache,
        cutoff_seconds=int(cfg.cutoff_seconds),
    )
    kline_changed_count = int(
        ensure_klines_coverage(
            client=binance_us_client,
            store=klines_store,
            symbol=str(binance_us_symbol),
            start_open_time_ms=int(start_open_ms),
            end_open_time_ms=int(end_open_ms),
        )
    )
    earliest_kline_open_time_ms = klines_store.earliest_open_time_ms()
    latest_kline_open_time_ms = klines_store.latest_open_time_ms()
    info(
        "CORE",
        "SYNC",
        "KLINES",
        msg=(
            f"Klines synced: changed_n={int(kline_changed_count)} "
            f"window=[{int(start_open_ms)}..{int(end_open_ms)}] "
            f"store=[{str(earliest_kline_open_time_ms)}..{str(latest_kline_open_time_ms)}]"
        ),
    )

    return SyncSummary(
        prior_context_rounds_required=int(prior_context_rounds_required),
        warmup_rounds=int(warmup_rounds),
        cache_n=int(cache_n),
        stored_closed_round_count=int(stored_closed_round_count),
        earliest_closed_epoch=int(earliest_closed_epoch),
        latest_closed_epoch=int(latest_closed_epoch),
        kline_changed_count=int(kline_changed_count),
        earliest_kline_open_time_ms=(
            None if earliest_kline_open_time_ms is None else int(earliest_kline_open_time_ms)
        ),
        latest_kline_open_time_ms=(
            None if latest_kline_open_time_ms is None else int(latest_kline_open_time_ms)
        ),
    )
