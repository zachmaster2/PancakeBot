from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_dataset import (
    NeuralDirectionDataset,
    build_neural_direction_dataset,
    neural_direction_required_history_rounds,
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


def load_recent_direction_eval_slice(
    *,
    config_path: str,
    required_examples: int,
    tail_offset_rounds: int,
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
            dataset = build_neural_direction_dataset(
                rounds=rounds,
                klines_store_like=klines_store,
                cutoff_seconds=int(cfg.cutoff_seconds),
                feature_cache_store=feature_cache_store,
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
