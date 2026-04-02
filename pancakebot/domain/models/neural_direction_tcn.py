from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_dataset import NeuralDirectionDataset


@dataclass(frozen=True, slots=True)
class NeuralDirectionTcnConfig:
    seq_len: int = 16
    channels: tuple[int, ...] = (64, 64)
    kernel_size: int = 3
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 40
    patience_epochs: int = 6


@dataclass(frozen=True, slots=True)
class NeuralDirectionTcnBundle:
    feature_columns: tuple[str, ...]
    config: NeuralDirectionTcnConfig
    impute_values: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, object]


class _CausalConvBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pad = int(kernel_size - 1) * int(dilation)
        self._pad = nn.ConstantPad1d((int(pad), 0), 0.0)
        self._conv = nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._pad(x)
        out = self._conv(out)
        out = self._relu(out)
        out = self._dropout(out)
        return out


class NeuralDirectionTcn(nn.Module):
    def __init__(self, *, input_dim: int, config: NeuralDirectionTcnConfig) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise InvariantError("neural_direction_tcn_input_dim_invalid")
        if int(config.seq_len) <= 1:
            raise InvariantError("neural_direction_tcn_seq_len_invalid")
        if int(config.kernel_size) <= 1:
            raise InvariantError("neural_direction_tcn_kernel_size_invalid")
        blocks: list[nn.Module] = []
        in_channels = int(input_dim)
        for idx, out_channels_raw in enumerate(config.channels):
            out_channels = int(out_channels_raw)
            if int(out_channels) <= 0:
                raise InvariantError("neural_direction_tcn_channels_invalid")
            blocks.append(
                _CausalConvBlock(
                    in_channels=int(in_channels),
                    out_channels=int(out_channels),
                    kernel_size=int(config.kernel_size),
                    dilation=int(2**idx),
                    dropout=float(config.dropout),
                )
            )
            in_channels = int(out_channels)
        self._blocks = nn.ModuleList(blocks)
        self._head = nn.Linear(int(in_channels), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise InvariantError("neural_direction_tcn_forward_rank_invalid")
        out = torch.transpose(x, 1, 2)
        for block in self._blocks:
            out = block(out)
        last = out[:, :, -1]
        logits = self._head(last)
        return logits.squeeze(-1)


def default_neural_direction_tcn_config() -> NeuralDirectionTcnConfig:
    return NeuralDirectionTcnConfig()


def train_neural_direction_tcn(
    *,
    dataset: NeuralDirectionDataset,
    train_target_epochs: Sequence[int],
    valid_target_epochs: Sequence[int],
    random_seed: int,
    config: NeuralDirectionTcnConfig | None = None,
) -> NeuralDirectionTcnBundle:
    model_cfg = config or default_neural_direction_tcn_config()
    _validate_dataset_for_tcn(dataset=dataset)
    train_x, train_y = _sequence_rows_for_target_epochs(
        dataset=dataset,
        target_epochs=train_target_epochs,
        seq_len=int(model_cfg.seq_len),
    )
    valid_x, valid_y = _sequence_rows_for_target_epochs(
        dataset=dataset,
        target_epochs=valid_target_epochs,
        seq_len=int(model_cfg.seq_len),
    )
    if len(train_x) <= 0:
        raise InvariantError("neural_direction_tcn_train_empty")
    if len(valid_x) <= 0:
        raise InvariantError("neural_direction_tcn_valid_empty")

    impute_values, feature_means, feature_stds = _fit_preprocessor(train_x=train_x)
    train_x_proc = _apply_preprocessor(
        x=train_x,
        impute_values=impute_values,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    valid_x_proc = _apply_preprocessor(
        x=valid_x,
        impute_values=impute_values,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )

    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))

    model = NeuralDirectionTcn(input_dim=int(train_x_proc.shape[2]), config=model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.learning_rate),
        weight_decay=float(model_cfg.weight_decay),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.from_numpy(train_x_proc),
        torch.from_numpy(train_y.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    loader = DataLoader(
        train_dataset,
        batch_size=int(model_cfg.batch_size),
        shuffle=True,
        generator=generator,
    )
    valid_x_tensor = torch.from_numpy(valid_x_proc)
    valid_y_tensor = torch.from_numpy(valid_y.astype(np.float32))

    best_state: dict[str, torch.Tensor] | None = None
    best_valid_loss: float | None = None
    best_epoch = -1
    best_valid_win_rate = 0.0
    stale_epochs = 0

    for epoch_idx in range(int(model_cfg.max_epochs)):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_x_tensor)
            valid_loss = float(loss_fn(valid_logits, valid_y_tensor).item())
            valid_probs = torch.sigmoid(valid_logits).cpu().numpy()
        valid_preds = (valid_probs >= 0.5).astype(np.int64)
        valid_win_rate = float(np.mean(valid_preds == valid_y.astype(np.int64)))

        if best_valid_loss is None or float(valid_loss) < float(best_valid_loss):
            best_valid_loss = float(valid_loss)
            best_epoch = int(epoch_idx)
            best_valid_win_rate = float(valid_win_rate)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if int(stale_epochs) >= int(model_cfg.patience_epochs):
                break

    if best_state is None or best_valid_loss is None:
        raise InvariantError("neural_direction_tcn_best_state_missing")

    return NeuralDirectionTcnBundle(
        feature_columns=tuple(str(col) for col in dataset.feature_columns),
        config=model_cfg,
        impute_values=np.asarray(impute_values, dtype=np.float32),
        feature_means=np.asarray(feature_means, dtype=np.float32),
        feature_stds=np.asarray(feature_stds, dtype=np.float32),
        state_dict=best_state,
        metadata={
            "input_dim": int(train_x_proc.shape[2]),
            "seq_len": int(model_cfg.seq_len),
            "train_examples": int(len(train_x_proc)),
            "valid_examples": int(len(valid_x_proc)),
            "train_epoch_start": int(train_target_epochs[0]),
            "train_epoch_end": int(train_target_epochs[-1]),
            "valid_epoch_start": int(valid_target_epochs[0]),
            "valid_epoch_end": int(valid_target_epochs[-1]),
            "random_seed": int(random_seed),
            "best_epoch": int(best_epoch),
            "best_valid_loss": float(best_valid_loss),
            "best_valid_win_rate": float(best_valid_win_rate),
        },
    )


def predict_neural_direction_tcn_probabilities(
    *,
    bundle: NeuralDirectionTcnBundle,
    feature_sequences: np.ndarray,
) -> np.ndarray:
    x = np.asarray(feature_sequences, dtype=np.float32)
    if x.ndim != 3:
        raise InvariantError("neural_direction_tcn_predict_x_rank_invalid")
    x_proc = _apply_preprocessor(
        x=x,
        impute_values=np.asarray(bundle.impute_values, dtype=np.float32),
        feature_means=np.asarray(bundle.feature_means, dtype=np.float32),
        feature_stds=np.asarray(bundle.feature_stds, dtype=np.float32),
    )
    model = NeuralDirectionTcn(
        input_dim=int(x_proc.shape[2]),
        config=bundle.config,
    )
    model.load_state_dict(bundle.state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_proc))
    probs = torch.sigmoid(logits).cpu().numpy()
    return np.asarray(probs, dtype=np.float32)


def build_neural_direction_tcn_feature_sequences(
    *,
    dataset: NeuralDirectionDataset,
    target_epochs: Sequence[int],
    seq_len: int,
) -> np.ndarray:
    x, _ = _sequence_rows_for_target_epochs(
        dataset=dataset,
        target_epochs=target_epochs,
        seq_len=int(seq_len),
    )
    return np.asarray(x, dtype=np.float32)


def build_sequence_examples_for_target_epochs(
    *,
    dataset: NeuralDirectionDataset,
    target_epochs: Sequence[int],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _sequence_rows_for_target_epochs(
        dataset=dataset,
        target_epochs=target_epochs,
        seq_len=int(seq_len),
    )


def save_neural_direction_tcn_bundle(*, bundle: NeuralDirectionTcnBundle, path: str) -> None:
    out_path = Path(str(path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_columns": tuple(str(col) for col in bundle.feature_columns),
        "config": asdict(bundle.config),
        "impute_values": np.asarray(bundle.impute_values, dtype=np.float32),
        "feature_means": np.asarray(bundle.feature_means, dtype=np.float32),
        "feature_stds": np.asarray(bundle.feature_stds, dtype=np.float32),
        "state_dict": {
            str(key): value.detach().cpu()
            for key, value in bundle.state_dict.items()
        },
        "metadata": dict(bundle.metadata),
    }
    torch.save(payload, str(out_path))


def load_neural_direction_tcn_bundle(path: str) -> NeuralDirectionTcnBundle:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise InvariantError("neural_direction_tcn_payload_invalid")
    config_raw = payload.get("config")
    if not isinstance(config_raw, dict):
        raise InvariantError("neural_direction_tcn_config_missing")
    return NeuralDirectionTcnBundle(
        feature_columns=tuple(str(col) for col in payload["feature_columns"]),
        config=NeuralDirectionTcnConfig(
            seq_len=int(config_raw["seq_len"]),
            channels=tuple(int(v) for v in config_raw["channels"]),
            kernel_size=int(config_raw["kernel_size"]),
            dropout=float(config_raw["dropout"]),
            learning_rate=float(config_raw["learning_rate"]),
            weight_decay=float(config_raw["weight_decay"]),
            batch_size=int(config_raw["batch_size"]),
            max_epochs=int(config_raw["max_epochs"]),
            patience_epochs=int(config_raw["patience_epochs"]),
        ),
        impute_values=np.asarray(payload["impute_values"], dtype=np.float32),
        feature_means=np.asarray(payload["feature_means"], dtype=np.float32),
        feature_stds=np.asarray(payload["feature_stds"], dtype=np.float32),
        state_dict={
            str(key): value.detach().cpu()
            for key, value in dict(payload["state_dict"]).items()
        },
        metadata=dict(payload.get("metadata", {})),
    )


def _validate_dataset_for_tcn(*, dataset: NeuralDirectionDataset) -> None:
    if int(dataset.feature_matrix.ndim) != 2:
        raise InvariantError("neural_direction_tcn_feature_rank_invalid")
    if int(dataset.feature_matrix.shape[0]) != int(dataset.num_examples):
        raise InvariantError("neural_direction_tcn_feature_rows_len_mismatch")
    if len(dataset.labels) != int(dataset.num_examples):
        raise InvariantError("neural_direction_tcn_labels_len_mismatch")


def _sequence_rows_for_target_epochs(
    *,
    dataset: NeuralDirectionDataset,
    target_epochs: Sequence[int],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(seq_len) <= 1:
        raise InvariantError("neural_direction_tcn_seq_len_nonpositive")
    if not target_epochs:
        raise InvariantError("neural_direction_tcn_target_epochs_empty")
    index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    seen: set[int] = set()
    for epoch in target_epochs:
        key = int(epoch)
        if int(key) in seen:
            raise InvariantError("neural_direction_tcn_target_epochs_duplicate")
        seen.add(int(key))
        idx = index_by_epoch.get(int(key))
        if idx is None:
            raise InvariantError("neural_direction_tcn_target_epoch_missing")
        if int(idx) < int(seq_len) - 1:
            raise InvariantError("neural_direction_tcn_sequence_history_insufficient")
        x_rows.append(
            np.asarray(
                dataset.feature_matrix[int(idx) - int(seq_len) + 1 : int(idx) + 1],
                dtype=np.float32,
            )
        )
        y_rows.append(int(dataset.labels[int(idx)]))
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64)


def _fit_preprocessor(*, train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(train_x, dtype=np.float32)
    if x.ndim != 3:
        raise InvariantError("neural_direction_tcn_preprocessor_train_rank_invalid")
    flat = x.reshape(-1, int(x.shape[2]))
    impute_values = np.nanmean(flat, axis=0)
    impute_values = np.where(np.isnan(impute_values), 0.0, impute_values).astype(np.float32)
    flat_imputed = np.where(np.isnan(flat), impute_values, flat).astype(np.float32)
    feature_means = np.asarray(np.mean(flat_imputed, axis=0), dtype=np.float32)
    feature_stds = np.asarray(np.std(flat_imputed, axis=0), dtype=np.float32)
    feature_stds = np.where(feature_stds <= 1e-6, 1.0, feature_stds).astype(np.float32)
    return impute_values, feature_means, feature_stds


def _apply_preprocessor(
    *,
    x: np.ndarray,
    impute_values: np.ndarray,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(x, dtype=np.float32)
    if raw.ndim != 3:
        raise InvariantError("neural_direction_tcn_preprocessor_x_rank_invalid")
    x_imputed = np.where(np.isnan(raw), impute_values.reshape(1, 1, -1), raw).astype(np.float32)
    return ((x_imputed - feature_means.reshape(1, 1, -1)) / feature_stds.reshape(1, 1, -1)).astype(np.float32)
