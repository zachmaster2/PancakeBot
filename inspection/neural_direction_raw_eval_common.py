from __future__ import annotations

from dataclasses import dataclass

from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_raw_sequence_dataset import (
    NeuralDirectionRawSequenceDataset,
    build_neural_direction_raw_sequence_dataset,
    tail_neural_direction_raw_sequence_dataset,
)
from pancakebot.domain.types import Round
from pancakebot.infra.market_data_db import MarketDataDb, SqliteKlinesStore


@dataclass(frozen=True, slots=True)
class NeuralDirectionRawEvalSlice:
    dataset: NeuralDirectionRawSequenceDataset
    target_rounds_by_epoch: dict[int, Round]
    loaded_round_count: int
    total_rounds_available: int


def load_recent_raw_direction_eval_slice(
    *,
    config_path: str,
    required_examples: int,
    tail_offset_rounds: int,
    settled_history_len: int,
    round_flow_bins: int,
    kline_seq_len: int,
) -> NeuralDirectionRawEvalSlice:
    if int(required_examples) <= 0:
        raise InvariantError("neural_direction_raw_eval_required_examples_nonpositive")
    if int(tail_offset_rounds) < 0:
        raise InvariantError("neural_direction_raw_eval_tail_offset_negative")
    if int(settled_history_len) <= 0:
        raise InvariantError("neural_direction_raw_eval_settled_history_nonpositive")
    if int(round_flow_bins) <= 0:
        raise InvariantError("neural_direction_raw_eval_round_flow_bins_nonpositive")
    if int(kline_seq_len) <= 0:
        raise InvariantError("neural_direction_raw_eval_kline_seq_nonpositive")

    cfg = load_app_config(str(config_path))
    market_data_store = MarketDataDb(str(cfg.market_data_db_path))
    try:
        market_data_store.ensure_sources_synced(
            rounds_jsonl_path=str(cfg.closed_rounds_path),
            klines_jsonl_path=str(cfg.klines_path),
        )
        klines_store = SqliteKlinesStore(market_data_db=market_data_store)
        total_rounds_available = int(market_data_store.count_rounds())
        required_prior_context_rounds = int(settled_history_len) + 1
        load_n = min(
            int(total_rounds_available),
            int(required_prior_context_rounds) + int(required_examples) + int(tail_offset_rounds) + 4096,
        )
        if int(load_n) <= int(required_prior_context_rounds):
            raise InvariantError("neural_direction_raw_eval_rounds_unavailable")

        while True:
            rounds = list(market_data_store.load_tail_rounds(n=int(load_n)))
            if int(tail_offset_rounds) > 0:
                if len(rounds) <= int(tail_offset_rounds):
                    raise InvariantError("neural_direction_raw_eval_tail_offset_out_of_range")
                rounds = list(rounds[: -int(tail_offset_rounds)])

            earliest_kline_open_ms = klines_store.earliest_open_time_ms()
            if earliest_kline_open_ms is not None:
                min_anchor_ms = int(earliest_kline_open_ms) + (int(kline_seq_len) - 1) * 60_000 + 59_999
                first_target_idx_with_kline_coverage: int | None = None
                for idx, round_t in enumerate(rounds):
                    if round_t.lock_at is None:
                        continue
                    anchor_ms = (int(round_t.lock_at) - int(cfg.cutoff_seconds)) * 1000
                    if int(anchor_ms) >= int(min_anchor_ms):
                        first_target_idx_with_kline_coverage = int(idx)
                        break
                if first_target_idx_with_kline_coverage is None:
                    raise InvariantError("neural_direction_raw_eval_kline_coverage_unavailable")
                trim_start_idx = max(
                    0,
                    int(first_target_idx_with_kline_coverage) - int(required_prior_context_rounds),
                )
                rounds = list(rounds[int(trim_start_idx) :])

            dataset = build_neural_direction_raw_sequence_dataset(
                rounds=rounds,
                klines_store_like=klines_store,
                cutoff_seconds=int(cfg.cutoff_seconds),
                settled_history_len=int(settled_history_len),
                round_flow_bins=int(round_flow_bins),
                kline_seq_len=int(kline_seq_len),
            )
            if int(dataset.num_examples) >= int(required_examples):
                tail_dataset = tail_neural_direction_raw_sequence_dataset(
                    dataset=dataset,
                    n=int(required_examples),
                )
                target_epoch_set = {int(epoch) for epoch in tail_dataset.target_epochs}
                target_rounds_by_epoch = {
                    int(round_t.epoch): round_t
                    for round_t in rounds
                    if int(round_t.epoch) in target_epoch_set
                }
                if len(target_rounds_by_epoch) != int(tail_dataset.num_examples):
                    raise InvariantError("neural_direction_raw_eval_target_rounds_len_mismatch")
                return NeuralDirectionRawEvalSlice(
                    dataset=tail_dataset,
                    target_rounds_by_epoch=target_rounds_by_epoch,
                    loaded_round_count=int(load_n),
                    total_rounds_available=int(total_rounds_available),
                )
            if int(load_n) >= int(total_rounds_available):
                raise InvariantError("neural_direction_raw_eval_dataset_insufficient_examples")
            load_n = min(
                int(total_rounds_available),
                int(load_n) + max(int(required_examples), 10_000),
            )
    finally:
        market_data_store.close()
