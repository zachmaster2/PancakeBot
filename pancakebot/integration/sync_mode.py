from __future__ import annotations

from dataclasses import dataclass

from pancakebot.config.app_config import AppConfig
from pancakebot.core.errors import InvariantError, TransientGraphError
from pancakebot.core.logging import info
from pancakebot.domain.closed_rounds_cache import RollingClosedRoundsCache
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.closed_rounds_sync import sync_closed_rounds
from pancakebot.infra.graph_client import GraphClient
from pancakebot.runtime.runtime_loop import required_runtime_sync_cache_n
from pancakebot.runtime.sleep import sleep_seconds

_TRANSIENT_NETWORK_DELAY_SECONDS = 10


@dataclass(frozen=True, slots=True)
class SyncSummary:
    warmup_rounds: int
    cache_n: int
    stored_closed_round_count: int
    earliest_closed_epoch: int
    latest_closed_epoch: int


def sync_runtime_market_data(
    *,
    cfg: AppConfig,
    graph: GraphClient,
    round_store: ClosedRoundsStore,
) -> SyncSummary:
    warmup_rounds = int(required_runtime_sync_cache_n())
    cache_n = int(warmup_rounds)

    info(
        "CORE",
        "SYNC",
        "START",
        msg=f"Sync setup: warmup_rounds={int(warmup_rounds)} closed_cache_needed={int(cache_n)}",
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

    return SyncSummary(
        warmup_rounds=int(warmup_rounds),
        cache_n=int(cache_n),
        stored_closed_round_count=int(stored_closed_round_count),
        earliest_closed_epoch=int(earliest_closed_epoch),
        latest_closed_epoch=int(latest_closed_epoch),
    )
