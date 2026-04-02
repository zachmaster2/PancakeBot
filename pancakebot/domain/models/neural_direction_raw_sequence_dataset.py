from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.flow_features import compute_late_phase_features
from pancakebot.domain.features.pool_amounts import compute_pool_amounts_wei_at_or_before
from pancakebot.domain.types import Bet, Kline, Round

_WEI_PER_BNB = 1_000_000_000_000_000_000
_LOG_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class NeuralDirectionRawSequenceDataset:
    target_epochs: tuple[int, ...]
    labels: np.ndarray
    round_sequence: np.ndarray
    kline_sequence: np.ndarray
    snapshot_matrix: np.ndarray
    round_feature_columns: tuple[str, ...]
    kline_feature_columns: tuple[str, ...]
    snapshot_feature_columns: tuple[str, ...]
    metadata: dict[str, object]

    @property
    def num_examples(self) -> int:
        return int(len(self.target_epochs))


def tail_neural_direction_raw_sequence_dataset(
    *,
    dataset: NeuralDirectionRawSequenceDataset,
    n: int,
) -> NeuralDirectionRawSequenceDataset:
    if int(n) <= 0:
        raise InvariantError("neural_direction_raw_tail_n_nonpositive")
    if int(dataset.num_examples) < int(n):
        raise InvariantError("neural_direction_raw_tail_n_exceeds_dataset")
    start = int(dataset.num_examples) - int(n)
    metadata = dict(dataset.metadata)
    metadata["tail_n"] = int(n)
    return NeuralDirectionRawSequenceDataset(
        target_epochs=tuple(int(epoch) for epoch in dataset.target_epochs[start:]),
        labels=np.asarray(dataset.labels[start:], dtype=np.int64),
        round_sequence=np.asarray(dataset.round_sequence[start:], dtype=np.float32),
        kline_sequence=np.asarray(dataset.kline_sequence[start:], dtype=np.float32),
        snapshot_matrix=np.asarray(dataset.snapshot_matrix[start:], dtype=np.float32),
        round_feature_columns=tuple(dataset.round_feature_columns),
        kline_feature_columns=tuple(dataset.kline_feature_columns),
        snapshot_feature_columns=tuple(dataset.snapshot_feature_columns),
        metadata=metadata,
    )


def select_raw_sequence_lengths(
    *,
    dataset: NeuralDirectionRawSequenceDataset,
    settled_history_len: int,
    kline_seq_len: int,
) -> NeuralDirectionRawSequenceDataset:
    max_settled_history_len = int(dataset.metadata["settled_history_len"])
    max_kline_seq_len = int(dataset.metadata["kline_seq_len"])
    if int(settled_history_len) <= 0:
        raise InvariantError("neural_direction_raw_settled_history_nonpositive")
    if int(kline_seq_len) <= 0:
        raise InvariantError("neural_direction_raw_kline_seq_nonpositive")
    if int(settled_history_len) > int(max_settled_history_len):
        raise InvariantError("neural_direction_raw_settled_history_exceeds_dataset")
    if int(kline_seq_len) > int(max_kline_seq_len):
        raise InvariantError("neural_direction_raw_kline_seq_exceeds_dataset")

    round_keep = int(settled_history_len) + 2
    round_sequence = np.asarray(dataset.round_sequence[:, -int(round_keep) :, :], dtype=np.float32)
    kline_sequence = np.asarray(dataset.kline_sequence[:, -int(kline_seq_len) :, :], dtype=np.float32)
    metadata = dict(dataset.metadata)
    metadata["selected_settled_history_len"] = int(settled_history_len)
    metadata["selected_round_seq_len"] = int(round_keep)
    metadata["selected_kline_seq_len"] = int(kline_seq_len)
    return NeuralDirectionRawSequenceDataset(
        target_epochs=tuple(int(epoch) for epoch in dataset.target_epochs),
        labels=np.asarray(dataset.labels, dtype=np.int64),
        round_sequence=round_sequence,
        kline_sequence=kline_sequence,
        snapshot_matrix=np.asarray(dataset.snapshot_matrix, dtype=np.float32),
        round_feature_columns=tuple(dataset.round_feature_columns),
        kline_feature_columns=tuple(dataset.kline_feature_columns),
        snapshot_feature_columns=tuple(dataset.snapshot_feature_columns),
        metadata=metadata,
    )


def build_raw_sequence_examples_for_target_epochs(
    *,
    dataset: NeuralDirectionRawSequenceDataset,
    target_epochs: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not target_epochs:
        raise InvariantError("neural_direction_raw_target_epochs_empty")
    index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
    round_rows: list[np.ndarray] = []
    kline_rows: list[np.ndarray] = []
    snapshot_rows: list[np.ndarray] = []
    labels: list[int] = []
    seen: set[int] = set()
    for epoch in target_epochs:
        key = int(epoch)
        if int(key) in seen:
            raise InvariantError("neural_direction_raw_target_epochs_duplicate")
        seen.add(int(key))
        idx = index_by_epoch.get(int(key))
        if idx is None:
            raise InvariantError("neural_direction_raw_target_epoch_missing")
        round_rows.append(np.asarray(dataset.round_sequence[int(idx)], dtype=np.float32))
        kline_rows.append(np.asarray(dataset.kline_sequence[int(idx)], dtype=np.float32))
        snapshot_rows.append(np.asarray(dataset.snapshot_matrix[int(idx)], dtype=np.float32))
        labels.append(int(dataset.labels[int(idx)]))
    return (
        np.asarray(round_rows, dtype=np.float32),
        np.asarray(kline_rows, dtype=np.float32),
        np.asarray(snapshot_rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def build_neural_direction_raw_sequence_dataset(
    *,
    rounds: Sequence[Round],
    klines_store_like: Any,
    cutoff_seconds: int,
    settled_history_len: int,
    round_flow_bins: int,
    kline_seq_len: int,
) -> NeuralDirectionRawSequenceDataset:
    if int(settled_history_len) <= 0:
        raise InvariantError("neural_direction_raw_settled_history_invalid")
    if int(round_flow_bins) <= 0:
        raise InvariantError("neural_direction_raw_round_flow_bins_invalid")
    if int(kline_seq_len) <= 0:
        raise InvariantError("neural_direction_raw_kline_seq_len_invalid")
    required_prior_context_rounds = int(settled_history_len) + 1
    if len(rounds) <= int(required_prior_context_rounds):
        raise InvariantError("neural_direction_raw_rounds_insufficient")
    if int(cutoff_seconds) <= 0:
        raise InvariantError("neural_direction_raw_cutoff_seconds_nonpositive")

    target_epochs: list[int] = []
    labels: list[int] = []
    round_rows: list[np.ndarray] = []
    kline_rows: list[np.ndarray] = []
    snapshot_rows: list[np.ndarray] = []

    skipped_house = 0
    skipped_failed = 0
    skipped_unusable = 0

    round_feature_columns = _build_round_feature_columns(round_flow_bins=int(round_flow_bins))
    kline_feature_columns = _build_kline_feature_columns()
    snapshot_feature_columns = _build_snapshot_feature_columns()

    for idx in range(int(required_prior_context_rounds), len(rounds)):
        target_round = rounds[int(idx)]
        position = str(target_round.position) if target_round.position is not None else ""

        if bool(target_round.failed):
            skipped_failed += 1
            continue
        if position == "House":
            skipped_house += 1
            continue
        if position not in ("Bull", "Bear"):
            skipped_unusable += 1
            continue
        if target_round.lock_at is None or target_round.close_at is None:
            skipped_unusable += 1
            continue
        if target_round.lock_price is None or target_round.close_price is None:
            skipped_unusable += 1
            continue
        if float(target_round.lock_price) <= 0.0 or float(target_round.close_price) <= 0.0:
            skipped_unusable += 1
            continue

        prior_context_rounds = list(rounds[int(idx) - int(required_prior_context_rounds) : int(idx)])
        if len(prior_context_rounds) != int(required_prior_context_rounds):
            raise InvariantError("neural_direction_raw_prior_context_len_mismatch")
        locked_round = prior_context_rounds[-1]
        anchor_close_time_ms = int(int(target_round.lock_at) - int(cutoff_seconds)) * 1000
        try:
            context_klines = list(
                klines_store_like.get_context_klines(
                    anchor_close_time_ms=int(anchor_close_time_ms),
                    size=int(kline_seq_len),
                )
            )
        except Exception as exc:
            raise InvariantError(
                f"neural_direction_raw_context_klines_unavailable: epoch={int(target_round.epoch)} err={exc}"
            ) from exc

        round_rows.append(
            _build_round_sequence(
                prior_context_rounds=prior_context_rounds,
                target_round=target_round,
                cutoff_seconds=int(cutoff_seconds),
                round_flow_bins=int(round_flow_bins),
            )
        )
        kline_rows.append(_build_kline_sequence(klines=context_klines))
        snapshot_rows.append(
            _build_snapshot_vector(
                target_round=target_round,
                locked_round=locked_round,
                cutoff_seconds=int(cutoff_seconds),
            )
        )
        target_epochs.append(int(target_round.epoch))
        labels.append(1 if position == "Bull" else 0)

    if not target_epochs:
        raise InvariantError("neural_direction_raw_dataset_empty")

    return NeuralDirectionRawSequenceDataset(
        target_epochs=tuple(int(epoch) for epoch in target_epochs),
        labels=np.asarray(labels, dtype=np.int64),
        round_sequence=np.asarray(round_rows, dtype=np.float32),
        kline_sequence=np.asarray(kline_rows, dtype=np.float32),
        snapshot_matrix=np.asarray(snapshot_rows, dtype=np.float32),
        round_feature_columns=tuple(round_feature_columns),
        kline_feature_columns=tuple(kline_feature_columns),
        snapshot_feature_columns=tuple(snapshot_feature_columns),
        metadata={
            "cutoff_seconds": int(cutoff_seconds),
            "settled_history_len": int(settled_history_len),
            "round_seq_len": int(settled_history_len) + 2,
            "round_flow_bins": int(round_flow_bins),
            "kline_seq_len": int(kline_seq_len),
            "num_examples": int(len(target_epochs)),
            "skipped_house": int(skipped_house),
            "skipped_failed": int(skipped_failed),
            "skipped_unusable": int(skipped_unusable),
        },
    )


def _build_round_feature_columns(*, round_flow_bins: int) -> list[str]:
    columns = [
        "state_is_settled",
        "state_is_locked",
        "state_is_target",
        "visible_bull_bnb",
        "visible_bear_bnb",
        "visible_total_bnb",
        "visible_bull_count",
        "visible_bear_count",
        "visible_total_count",
        "visible_bull_share",
        "visible_log_imb",
        "visible_window_fraction",
        "lock_price_rel_prev_close",
        "lock_price_known",
        "close_return_from_lock",
        "close_return_from_prev_close",
        "close_price_known",
        "outcome_is_bull",
        "outcome_is_bear",
        "outcome_known",
        "delta_total_bnb_vs_prev",
        "delta_bull_share_vs_prev",
        "delta_log_imb_vs_prev",
    ]
    for bin_idx in range(int(round_flow_bins)):
        prefix = f"bin_{int(bin_idx)}"
        columns.extend(
            [
                f"{prefix}_bull_bnb",
                f"{prefix}_bear_bnb",
                f"{prefix}_bull_count",
                f"{prefix}_bear_count",
                f"{prefix}_net_bnb",
            ]
        )
    return columns


def _build_kline_feature_columns() -> list[str]:
    return [
        "close_ret_prev",
        "open_ret_prev",
        "body_ret",
        "range_ret",
        "close_rel_anchor",
        "volume_log1p",
        "quote_volume_log1p",
        "taker_buy_base_share",
        "trades_log1p",
    ]


def _build_snapshot_feature_columns() -> list[str]:
    return [
        "target_bull_bnb",
        "target_bear_bnb",
        "target_total_bnb",
        "target_bull_count",
        "target_bear_count",
        "target_total_count",
        "target_bull_share",
        "target_log_imb",
        "locked_total_bnb",
        "locked_bull_share",
        "locked_late_bull_sum",
        "locked_late_bear_sum",
        "locked_late_total_sum",
        "locked_late_bull_n",
        "locked_late_bear_n",
        "locked_late_total_n",
        "locked_late_log_imb",
        "target_total_minus_locked_total",
        "target_bull_share_minus_locked_bull_share",
    ]


def _build_round_sequence(
    *,
    prior_context_rounds: Sequence[Round],
    target_round: Round,
    cutoff_seconds: int,
    round_flow_bins: int,
) -> np.ndarray:
    sequence_rounds = list(prior_context_rounds) + [target_round]
    target_cutoff_ts = int(target_round.lock_at) - int(cutoff_seconds)
    rows: list[dict[str, float]] = []
    for idx, round_t in enumerate(sequence_rounds):
        state = "target" if int(idx) == int(len(sequence_rounds) - 1) else ("locked" if int(idx) == int(len(sequence_rounds) - 2) else "settled")
        prev_close_price = _previous_close_price(sequence_rounds=sequence_rounds, idx=int(idx))
        decision_end_ts = _decision_end_ts(
            round_t=round_t,
            state=str(state),
            target_cutoff_ts=int(target_cutoff_ts),
        )
        rows.append(
            _build_round_timestep_features(
                round_t=round_t,
                state=str(state),
                decision_end_ts=int(decision_end_ts),
                prev_close_price=prev_close_price,
                round_flow_bins=int(round_flow_bins),
            )
        )
    for idx in range(len(rows)):
        prev_row = None if int(idx) == 0 else rows[int(idx) - 1]
        row = dict(rows[int(idx)])
        if prev_row is None:
            row["delta_total_bnb_vs_prev"] = 0.0
            row["delta_bull_share_vs_prev"] = 0.0
            row["delta_log_imb_vs_prev"] = 0.0
        else:
            row["delta_total_bnb_vs_prev"] = float(row["visible_total_bnb"] - prev_row["visible_total_bnb"])
            row["delta_bull_share_vs_prev"] = float(row["visible_bull_share"] - prev_row["visible_bull_share"])
            row["delta_log_imb_vs_prev"] = float(row["visible_log_imb"] - prev_row["visible_log_imb"])
        rows[int(idx)] = row
    columns = _build_round_feature_columns(round_flow_bins=int(round_flow_bins))
    return np.asarray(
        [[float(row[str(column)]) for column in columns] for row in rows],
        dtype=np.float32,
    )


def _build_round_timestep_features(
    *,
    round_t: Round,
    state: str,
    decision_end_ts: int,
    prev_close_price: float | None,
    round_flow_bins: int,
) -> dict[str, float]:
    if int(decision_end_ts) < int(round_t.start_at):
        decision_end_ts = int(round_t.start_at)
    visible_pool = compute_pool_amounts_wei_at_or_before(
        bets=round_t.bets,
        cutoff_ts=int(decision_end_ts),
    )
    visible_bull_bnb = _wei_to_bnb(int(visible_pool.bull_wei))
    visible_bear_bnb = _wei_to_bnb(int(visible_pool.bear_wei))
    visible_total_bnb = _wei_to_bnb(int(visible_pool.total_wei))
    visible_bull_count, visible_bear_count = _visible_bet_counts(
        bets=round_t.bets,
        decision_end_ts=int(decision_end_ts),
    )
    visible_total_count = float(visible_bull_count + visible_bear_count)
    visible_bull_share = 0.5 if float(visible_total_bnb) <= 0.0 else float(visible_bull_bnb / visible_total_bnb)
    visible_log_imb = _log_imb(bull=float(visible_bull_bnb), bear=float(visible_bear_bnb))
    full_decision_end_ts = int(round_t.lock_at) if round_t.lock_at is not None else int(decision_end_ts)
    full_span = max(1, int(full_decision_end_ts) - int(round_t.start_at))
    visible_span = max(0, int(decision_end_ts) - int(round_t.start_at))
    features: dict[str, float] = {
        "state_is_settled": 1.0 if str(state) == "settled" else 0.0,
        "state_is_locked": 1.0 if str(state) == "locked" else 0.0,
        "state_is_target": 1.0 if str(state) == "target" else 0.0,
        "visible_bull_bnb": float(visible_bull_bnb),
        "visible_bear_bnb": float(visible_bear_bnb),
        "visible_total_bnb": float(visible_total_bnb),
        "visible_bull_count": float(visible_bull_count),
        "visible_bear_count": float(visible_bear_count),
        "visible_total_count": float(visible_total_count),
        "visible_bull_share": float(visible_bull_share),
        "visible_log_imb": float(visible_log_imb),
        "visible_window_fraction": float(visible_span / float(full_span)),
        "lock_price_rel_prev_close": 0.0,
        "lock_price_known": 0.0,
        "close_return_from_lock": 0.0,
        "close_return_from_prev_close": 0.0,
        "close_price_known": 0.0,
        "outcome_is_bull": 0.0,
        "outcome_is_bear": 0.0,
        "outcome_known": 0.0,
        "delta_total_bnb_vs_prev": 0.0,
        "delta_bull_share_vs_prev": 0.0,
        "delta_log_imb_vs_prev": 0.0,
    }
    if str(state) != "target" and round_t.lock_price is not None and float(round_t.lock_price) > 0.0:
        features["lock_price_known"] = 1.0
        if prev_close_price is not None and float(prev_close_price) > 0.0:
            features["lock_price_rel_prev_close"] = float(float(round_t.lock_price) / float(prev_close_price) - 1.0)
    if str(state) == "settled" and round_t.close_price is not None and float(round_t.close_price) > 0.0:
        features["close_price_known"] = 1.0
        if round_t.lock_price is not None and float(round_t.lock_price) > 0.0:
            features["close_return_from_lock"] = float(float(round_t.close_price) / float(round_t.lock_price) - 1.0)
        if prev_close_price is not None and float(prev_close_price) > 0.0:
            features["close_return_from_prev_close"] = float(float(round_t.close_price) / float(prev_close_price) - 1.0)
    if str(state) == "settled" and str(round_t.position) == "Bull":
        features["outcome_is_bull"] = 1.0
        features["outcome_known"] = 1.0
    elif str(state) == "settled" and str(round_t.position) == "Bear":
        features["outcome_is_bear"] = 1.0
        features["outcome_known"] = 1.0
    features.update(
        _visible_flow_bin_features(
            bets=round_t.bets,
            start_ts=int(round_t.start_at),
            decision_end_ts=int(decision_end_ts),
            round_flow_bins=int(round_flow_bins),
        )
    )
    return features


def _build_kline_sequence(*, klines: Sequence[Kline]) -> np.ndarray:
    if not klines:
        raise InvariantError("neural_direction_raw_kline_sequence_empty")
    anchor_close = float(klines[-1].close_price)
    if float(anchor_close) <= 0.0:
        raise InvariantError("neural_direction_raw_kline_anchor_invalid")
    rows: list[list[float]] = []
    prev_close = float(klines[0].close_price)
    for idx, kline in enumerate(klines):
        open_price = max(float(kline.open_price), _LOG_EPS)
        close_price = max(float(kline.close_price), _LOG_EPS)
        high_price = max(float(kline.high_price), open_price)
        low_price = max(float(kline.low_price), _LOG_EPS)
        ref_close = max(float(prev_close), _LOG_EPS)
        volume = max(float(kline.volume), 0.0)
        quote_volume = max(float(kline.quote_asset_volume), 0.0)
        rows.append(
            [
                float(close_price / ref_close - 1.0) if int(idx) > 0 else 0.0,
                float(open_price / ref_close - 1.0) if int(idx) > 0 else 0.0,
                float(close_price / open_price - 1.0),
                float((high_price - low_price) / open_price),
                float(close_price / float(anchor_close) - 1.0),
                float(math.log1p(volume)),
                float(math.log1p(quote_volume)),
                float(float(kline.taker_buy_base_volume) / volume) if float(volume) > 0.0 else 0.5,
                float(math.log1p(max(float(kline.number_of_trades), 0.0))),
            ]
        )
        prev_close = float(close_price)
    return np.asarray(rows, dtype=np.float32)


def _build_snapshot_vector(
    *,
    target_round: Round,
    locked_round: Round,
    cutoff_seconds: int,
) -> np.ndarray:
    target_cutoff_ts = int(target_round.lock_at) - int(cutoff_seconds)
    target_pool = compute_pool_amounts_wei_at_or_before(
        bets=target_round.bets,
        cutoff_ts=int(target_cutoff_ts),
    )
    target_bull_bnb = _wei_to_bnb(int(target_pool.bull_wei))
    target_bear_bnb = _wei_to_bnb(int(target_pool.bear_wei))
    target_total_bnb = _wei_to_bnb(int(target_pool.total_wei))
    target_bull_count, target_bear_count = _visible_bet_counts(
        bets=target_round.bets,
        decision_end_ts=int(target_cutoff_ts),
    )
    target_total_count = float(target_bull_count + target_bear_count)
    target_bull_share = 0.5 if float(target_total_bnb) <= 0.0 else float(target_bull_bnb / target_total_bnb)
    target_log_imb = _log_imb(bull=float(target_bull_bnb), bear=float(target_bear_bnb))

    locked_pool = compute_pool_amounts_wei_at_or_before(
        bets=locked_round.bets,
        cutoff_ts=int(locked_round.lock_at),
    )
    locked_total_bnb = _wei_to_bnb(int(locked_pool.total_wei))
    locked_bull_bnb = _wei_to_bnb(int(locked_pool.bull_wei))
    locked_bull_share = 0.5 if float(locked_total_bnb) <= 0.0 else float(locked_bull_bnb / locked_total_bnb)

    late = compute_late_phase_features(
        bets=locked_round.bets,
        lock_ts=locked_round.lock_at,
        cutoff_seconds=int(cutoff_seconds),
    )

    return np.asarray(
        [
            float(target_bull_bnb),
            float(target_bear_bnb),
            float(target_total_bnb),
            float(target_bull_count),
            float(target_bear_count),
            float(target_total_count),
            float(target_bull_share),
            float(target_log_imb),
            float(locked_total_bnb),
            float(locked_bull_share),
            float(late["late_bull_sum"]),
            float(late["late_bear_sum"]),
            float(late["late_total_sum"]),
            float(late["late_bull_n"]),
            float(late["late_bear_n"]),
            float(late["late_total_n"]),
            float(late["late_log_imb"]),
            float(target_total_bnb - locked_total_bnb),
            float(target_bull_share - locked_bull_share),
        ],
        dtype=np.float32,
    )


def _visible_flow_bin_features(
    *,
    bets: Sequence[Bet],
    start_ts: int,
    decision_end_ts: int,
    round_flow_bins: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    span = max(1, int(decision_end_ts) - int(start_ts) + 1)
    bull_bnb = [0.0 for _ in range(int(round_flow_bins))]
    bear_bnb = [0.0 for _ in range(int(round_flow_bins))]
    bull_count = [0.0 for _ in range(int(round_flow_bins))]
    bear_count = [0.0 for _ in range(int(round_flow_bins))]
    for bet in bets:
        created_at = int(bet.created_at)
        if int(created_at) < int(start_ts) or int(created_at) > int(decision_end_ts):
            continue
        rel = int(created_at) - int(start_ts)
        bucket = min(int(round_flow_bins) - 1, int(rel * int(round_flow_bins) / int(span)))
        amount_bnb = _wei_to_bnb(int(bet.amount_wei))
        if str(bet.position) == "Bull":
            bull_bnb[int(bucket)] += float(amount_bnb)
            bull_count[int(bucket)] += 1.0
        elif str(bet.position) == "Bear":
            bear_bnb[int(bucket)] += float(amount_bnb)
            bear_count[int(bucket)] += 1.0
    for idx in range(int(round_flow_bins)):
        prefix = f"bin_{int(idx)}"
        out[f"{prefix}_bull_bnb"] = float(bull_bnb[int(idx)])
        out[f"{prefix}_bear_bnb"] = float(bear_bnb[int(idx)])
        out[f"{prefix}_bull_count"] = float(bull_count[int(idx)])
        out[f"{prefix}_bear_count"] = float(bear_count[int(idx)])
        out[f"{prefix}_net_bnb"] = float(bull_bnb[int(idx)] - bear_bnb[int(idx)])
    return out


def _visible_bet_counts(*, bets: Sequence[Bet], decision_end_ts: int) -> tuple[float, float]:
    bull = 0.0
    bear = 0.0
    for bet in bets:
        if int(bet.created_at) > int(decision_end_ts):
            continue
        if str(bet.position) == "Bull":
            bull += 1.0
        elif str(bet.position) == "Bear":
            bear += 1.0
    return float(bull), float(bear)


def _decision_end_ts(*, round_t: Round, state: str, target_cutoff_ts: int) -> int:
    if str(state) == "target":
        return int(target_cutoff_ts)
    if round_t.lock_at is None:
        raise InvariantError("neural_direction_raw_round_lock_missing")
    return int(round_t.lock_at)


def _previous_close_price(*, sequence_rounds: Sequence[Round], idx: int) -> float | None:
    for prev_idx in range(int(idx) - 1, -1, -1):
        prev_close = sequence_rounds[int(prev_idx)].close_price
        if prev_close is not None and float(prev_close) > 0.0:
            return float(prev_close)
    return None


def _wei_to_bnb(amount_wei: int) -> float:
    return float(int(amount_wei)) / float(_WEI_PER_BNB)


def _log_imb(*, bull: float, bear: float) -> float:
    return float(math.log((float(bull) + _LOG_EPS) / (float(bear) + _LOG_EPS)))
