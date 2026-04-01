from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_dataset import (
    NeuralDirectionDataset,
    available_feature_groups,
    build_neural_direction_dataset,
    neural_direction_required_context_klines,
    neural_direction_required_history_rounds,
    select_feature_groups,
    tail_neural_direction_dataset,
)
from pancakebot.infra.feature_cache_store import FeatureCacheStore
from pancakebot.infra.market_data_db import MarketDataDb, SqliteKlinesStore


@dataclass(frozen=True, slots=True)
class NeuralDirectionEvalSlice:
    dataset: NeuralDirectionDataset
    loaded_round_count: int
    total_rounds_available: int


def parse_positive_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = int(text)
        if int(value) <= 0:
            raise InvariantError("neural_direction_eval_nonpositive_int")
        out.append(int(value))
    if not out:
        raise InvariantError("neural_direction_eval_empty_positive_int_list")
    return out


def parse_nonnegative_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = int(text)
        if int(value) < 0:
            raise InvariantError("neural_direction_eval_negative_offset")
        out.append(int(value))
    if not out:
        raise InvariantError("neural_direction_eval_empty_offset_list")
    return out


def parse_optional_str_list(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    out = tuple(str(token).strip() for token in str(raw).split(",") if str(token).strip() != "")
    if not out:
        return None
    return out


def feature_groups_help_text() -> str:
    return ",".join(available_feature_groups())


def load_recent_direction_eval_slice(
    *,
    config_path: str,
    required_examples: int,
    tail_offset_rounds: int,
    include_feature_groups: tuple[str, ...] | None = None,
    exclude_feature_groups: tuple[str, ...] | None = None,
) -> NeuralDirectionEvalSlice:
    if int(required_examples) <= 0:
        raise InvariantError("neural_direction_eval_required_examples_nonpositive")
    if int(tail_offset_rounds) < 0:
        raise InvariantError("neural_direction_eval_tail_offset_negative")

    cfg = load_app_config(str(config_path))
    market_data_store = MarketDataDb(str(cfg.market_data_db_path))
    feature_cache_store = FeatureCacheStore(str(cfg.feature_cache_path))
    try:
        market_data_store.ensure_sources_synced(
            rounds_jsonl_path=str(cfg.closed_rounds_path),
            klines_jsonl_path=str(cfg.klines_path),
        )
        klines_store = SqliteKlinesStore(market_data_db=market_data_store)
        total_rounds_available = int(market_data_store.count_rounds())
        history_n = int(neural_direction_required_history_rounds())
        load_n = min(
            int(total_rounds_available),
            int(history_n) + int(required_examples) + int(tail_offset_rounds) + 4096,
        )
        if int(load_n) <= int(history_n):
            raise InvariantError("neural_direction_eval_rounds_unavailable")

        while True:
            rounds = list(market_data_store.load_tail_rounds(n=int(load_n)))
            if int(tail_offset_rounds) > 0:
                if len(rounds) <= int(tail_offset_rounds):
                    raise InvariantError("neural_direction_eval_tail_offset_out_of_range")
                rounds = list(rounds[: -int(tail_offset_rounds)])
            earliest_kline_open_ms = klines_store.earliest_open_time_ms()
            if earliest_kline_open_ms is not None:
                min_anchor_ms = (
                    int(earliest_kline_open_ms)
                    + (int(neural_direction_required_context_klines()) - 1) * 60_000
                    + 59_999
                )
                first_target_idx_with_kline_coverage: int | None = None
                for idx, round_t in enumerate(rounds):
                    if round_t.lock_at is None:
                        continue
                    anchor_ms = (int(round_t.lock_at) - int(cfg.cutoff_seconds)) * 1000
                    if int(anchor_ms) >= int(min_anchor_ms):
                        first_target_idx_with_kline_coverage = int(idx)
                        break
                if first_target_idx_with_kline_coverage is None:
                    raise InvariantError("neural_direction_eval_kline_coverage_unavailable")
                trim_start_idx = max(
                    0,
                    int(first_target_idx_with_kline_coverage) - int(history_n),
                )
                rounds = list(rounds[int(trim_start_idx) :])
            dataset = build_neural_direction_dataset(
                rounds=rounds,
                klines_store_like=klines_store,
                cutoff_seconds=int(cfg.cutoff_seconds),
                feature_cache_store=feature_cache_store,
            )
            dataset = select_feature_groups(
                dataset=dataset,
                include_groups=include_feature_groups,
                exclude_groups=exclude_feature_groups,
            )
            if int(dataset.num_examples) >= int(required_examples):
                return NeuralDirectionEvalSlice(
                    dataset=tail_neural_direction_dataset(dataset=dataset, n=int(required_examples)),
                    loaded_round_count=int(load_n),
                    total_rounds_available=int(total_rounds_available),
                )
            if int(load_n) >= int(total_rounds_available):
                raise InvariantError("neural_direction_eval_dataset_insufficient_examples")
            load_n = min(
                int(total_rounds_available),
                int(load_n) + max(int(required_examples), 10_000),
            )
    finally:
        feature_cache_store.flush()
        feature_cache_store.close()
        market_data_store.close()


def summary_path(*, output_dir: Path, name_prefix: str, suffix: str) -> Path:
    return (output_dir / f"{name_prefix}_{suffix}.json").resolve()


def rows_path(*, output_dir: Path, name_prefix: str, suffix: str) -> Path:
    return (output_dir / f"{name_prefix}_{suffix}.csv").resolve()
