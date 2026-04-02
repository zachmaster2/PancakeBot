from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_raw_sequence_dataset import (
    NeuralDirectionRawSequenceDataset,
    build_raw_sequence_examples_for_target_epochs,
)


@dataclass(frozen=True, slots=True)
class NeuralDirectionRawTcnConfig:
    round_channels: tuple[int, ...] = (64, 64)
    kline_channels: tuple[int, ...] = (64, 64)
    kernel_size: int = 3
    snapshot_hidden_sizes: tuple[int, ...] = (64,)
    fusion_hidden_sizes: tuple[int, ...] = (64,)
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 40
    patience_epochs: int = 6


@dataclass(frozen=True, slots=True)
class NeuralDirectionRawTcnBundle:
    round_feature_columns: tuple[str, ...]
    kline_feature_columns: tuple[str, ...]
    snapshot_feature_columns: tuple[str, ...]
    config: NeuralDirectionRawTcnConfig
    round_impute_values: np.ndarray
    round_feature_means: np.ndarray
    round_feature_stds: np.ndarray
    kline_impute_values: np.ndarray
    kline_feature_means: np.ndarray
    kline_feature_stds: np.ndarray
    snapshot_impute_values: np.ndarray
    snapshot_feature_means: np.ndarray
    snapshot_feature_stds: np.ndarray
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


class _TemporalEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        channels: Sequence[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise InvariantError("neural_direction_raw_tcn_input_dim_invalid")
        blocks: list[nn.Module] = []
        in_channels = int(input_dim)
        for idx, out_channels_raw in enumerate(channels):
            out_channels = int(out_channels_raw)
            if int(out_channels) <= 0:
                raise InvariantError("neural_direction_raw_tcn_channels_invalid")
            blocks.append(
                _CausalConvBlock(
                    in_channels=int(in_channels),
                    out_channels=int(out_channels),
                    kernel_size=int(kernel_size),
                    dilation=int(2**idx),
                    dropout=float(dropout),
                )
            )
            in_channels = int(out_channels)
        self._blocks = nn.ModuleList(blocks)
        self.output_dim = int(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise InvariantError("neural_direction_raw_tcn_encoder_rank_invalid")
        out = torch.transpose(x, 1, 2)
        for block in self._blocks:
            out = block(out)
        return out[:, :, -1]


def _build_mlp(
    *,
    input_dim: int,
    hidden_sizes: Sequence[int],
    dropout: float,
    output_dim: int | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for hidden_size_raw in hidden_sizes:
        hidden_size = int(hidden_size_raw)
        if int(hidden_size) <= 0:
            raise InvariantError("neural_direction_raw_tcn_mlp_hidden_invalid")
        layers.append(nn.Linear(int(in_dim), int(hidden_size)))
        layers.append(nn.ReLU())
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        in_dim = int(hidden_size)
    if output_dim is not None:
        layers.append(nn.Linear(int(in_dim), int(output_dim)))
    return nn.Sequential(*layers)


class NeuralDirectionRawTcn(nn.Module):
    def __init__(
        self,
        *,
        round_input_dim: int,
        kline_input_dim: int,
        snapshot_input_dim: int,
        config: NeuralDirectionRawTcnConfig,
    ) -> None:
        super().__init__()
        self._round_encoder = _TemporalEncoder(
            input_dim=int(round_input_dim),
            channels=config.round_channels,
            kernel_size=int(config.kernel_size),
            dropout=float(config.dropout),
        )
        self._kline_encoder = _TemporalEncoder(
            input_dim=int(kline_input_dim),
            channels=config.kline_channels,
            kernel_size=int(config.kernel_size),
            dropout=float(config.dropout),
        )
        if int(snapshot_input_dim) <= 0:
            raise InvariantError("neural_direction_raw_tcn_snapshot_dim_invalid")
        if config.snapshot_hidden_sizes:
            self._snapshot_encoder = _build_mlp(
                input_dim=int(snapshot_input_dim),
                hidden_sizes=config.snapshot_hidden_sizes,
                dropout=float(config.dropout),
            )
            snapshot_out_dim = int(config.snapshot_hidden_sizes[-1])
        else:
            self._snapshot_encoder = nn.Identity()
            snapshot_out_dim = int(snapshot_input_dim)
        fusion_input_dim = int(self._round_encoder.output_dim + self._kline_encoder.output_dim + snapshot_out_dim)
        fusion_layers: list[nn.Module] = []
        in_dim = int(fusion_input_dim)
        for hidden_size_raw in config.fusion_hidden_sizes:
            hidden_size = int(hidden_size_raw)
            if int(hidden_size) <= 0:
                raise InvariantError("neural_direction_raw_tcn_fusion_hidden_invalid")
            fusion_layers.append(nn.Linear(int(in_dim), int(hidden_size)))
            fusion_layers.append(nn.ReLU())
            if float(config.dropout) > 0.0:
                fusion_layers.append(nn.Dropout(float(config.dropout)))
            in_dim = int(hidden_size)
        fusion_layers.append(nn.Linear(int(in_dim), 1))
        self._head = nn.Sequential(*fusion_layers)

    def forward(
        self,
        round_x: torch.Tensor,
        kline_x: torch.Tensor,
        snapshot_x: torch.Tensor,
    ) -> torch.Tensor:
        round_vec = self._round_encoder(round_x)
        kline_vec = self._kline_encoder(kline_x)
        snapshot_vec = self._snapshot_encoder(snapshot_x)
        fused = torch.cat([round_vec, kline_vec, snapshot_vec], dim=1)
        logits = self._head(fused)
        return logits.squeeze(-1)


def default_neural_direction_raw_tcn_config() -> NeuralDirectionRawTcnConfig:
    return NeuralDirectionRawTcnConfig()


def train_neural_direction_raw_tcn(
    *,
    dataset: NeuralDirectionRawSequenceDataset,
    train_target_epochs: Sequence[int],
    valid_target_epochs: Sequence[int],
    random_seed: int,
    config: NeuralDirectionRawTcnConfig | None = None,
) -> NeuralDirectionRawTcnBundle:
    model_cfg = config or default_neural_direction_raw_tcn_config()
    _validate_dataset_for_raw_tcn(dataset=dataset)
    train_round_x, train_kline_x, train_snapshot_x, train_y = build_raw_sequence_examples_for_target_epochs(
        dataset=dataset,
        target_epochs=train_target_epochs,
    )
    valid_round_x, valid_kline_x, valid_snapshot_x, valid_y = build_raw_sequence_examples_for_target_epochs(
        dataset=dataset,
        target_epochs=valid_target_epochs,
    )
    if len(train_round_x) <= 0:
        raise InvariantError("neural_direction_raw_tcn_train_empty")
    if len(valid_round_x) <= 0:
        raise InvariantError("neural_direction_raw_tcn_valid_empty")

    round_impute_values, round_feature_means, round_feature_stds = _fit_sequence_preprocessor(train_x=train_round_x)
    kline_impute_values, kline_feature_means, kline_feature_stds = _fit_sequence_preprocessor(train_x=train_kline_x)
    snapshot_impute_values, snapshot_feature_means, snapshot_feature_stds = _fit_matrix_preprocessor(train_x=train_snapshot_x)

    train_round_proc = _apply_sequence_preprocessor(
        x=train_round_x,
        impute_values=round_impute_values,
        feature_means=round_feature_means,
        feature_stds=round_feature_stds,
    )
    valid_round_proc = _apply_sequence_preprocessor(
        x=valid_round_x,
        impute_values=round_impute_values,
        feature_means=round_feature_means,
        feature_stds=round_feature_stds,
    )
    train_kline_proc = _apply_sequence_preprocessor(
        x=train_kline_x,
        impute_values=kline_impute_values,
        feature_means=kline_feature_means,
        feature_stds=kline_feature_stds,
    )
    valid_kline_proc = _apply_sequence_preprocessor(
        x=valid_kline_x,
        impute_values=kline_impute_values,
        feature_means=kline_feature_means,
        feature_stds=kline_feature_stds,
    )
    train_snapshot_proc = _apply_matrix_preprocessor(
        x=train_snapshot_x,
        impute_values=snapshot_impute_values,
        feature_means=snapshot_feature_means,
        feature_stds=snapshot_feature_stds,
    )
    valid_snapshot_proc = _apply_matrix_preprocessor(
        x=valid_snapshot_x,
        impute_values=snapshot_impute_values,
        feature_means=snapshot_feature_means,
        feature_stds=snapshot_feature_stds,
    )

    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))

    model = NeuralDirectionRawTcn(
        round_input_dim=int(train_round_proc.shape[2]),
        kline_input_dim=int(train_kline_proc.shape[2]),
        snapshot_input_dim=int(train_snapshot_proc.shape[1]),
        config=model_cfg,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.learning_rate),
        weight_decay=float(model_cfg.weight_decay),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.from_numpy(train_round_proc),
        torch.from_numpy(train_kline_proc),
        torch.from_numpy(train_snapshot_proc),
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

    valid_round_tensor = torch.from_numpy(valid_round_proc)
    valid_kline_tensor = torch.from_numpy(valid_kline_proc)
    valid_snapshot_tensor = torch.from_numpy(valid_snapshot_proc)
    valid_y_tensor = torch.from_numpy(valid_y.astype(np.float32))

    best_state: dict[str, torch.Tensor] | None = None
    best_valid_loss: float | None = None
    best_epoch = -1
    best_valid_win_rate = 0.0
    stale_epochs = 0

    for epoch_idx in range(int(model_cfg.max_epochs)):
        model.train()
        for batch_round_x, batch_kline_x, batch_snapshot_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_round_x, batch_kline_x, batch_snapshot_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_round_tensor, valid_kline_tensor, valid_snapshot_tensor)
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
        raise InvariantError("neural_direction_raw_tcn_best_state_missing")

    return NeuralDirectionRawTcnBundle(
        round_feature_columns=tuple(str(col) for col in dataset.round_feature_columns),
        kline_feature_columns=tuple(str(col) for col in dataset.kline_feature_columns),
        snapshot_feature_columns=tuple(str(col) for col in dataset.snapshot_feature_columns),
        config=model_cfg,
        round_impute_values=np.asarray(round_impute_values, dtype=np.float32),
        round_feature_means=np.asarray(round_feature_means, dtype=np.float32),
        round_feature_stds=np.asarray(round_feature_stds, dtype=np.float32),
        kline_impute_values=np.asarray(kline_impute_values, dtype=np.float32),
        kline_feature_means=np.asarray(kline_feature_means, dtype=np.float32),
        kline_feature_stds=np.asarray(kline_feature_stds, dtype=np.float32),
        snapshot_impute_values=np.asarray(snapshot_impute_values, dtype=np.float32),
        snapshot_feature_means=np.asarray(snapshot_feature_means, dtype=np.float32),
        snapshot_feature_stds=np.asarray(snapshot_feature_stds, dtype=np.float32),
        state_dict=best_state,
        metadata={
            "round_input_dim": int(train_round_proc.shape[2]),
            "kline_input_dim": int(train_kline_proc.shape[2]),
            "snapshot_input_dim": int(train_snapshot_proc.shape[1]),
            "round_seq_len": int(train_round_proc.shape[1]),
            "kline_seq_len": int(train_kline_proc.shape[1]),
            "train_examples": int(len(train_round_proc)),
            "valid_examples": int(len(valid_round_proc)),
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


def predict_neural_direction_raw_tcn_probabilities(
    *,
    bundle: NeuralDirectionRawTcnBundle,
    round_sequence: np.ndarray,
    kline_sequence: np.ndarray,
    snapshot_matrix: np.ndarray,
) -> np.ndarray:
    round_x = np.asarray(round_sequence, dtype=np.float32)
    kline_x = np.asarray(kline_sequence, dtype=np.float32)
    snapshot_x = np.asarray(snapshot_matrix, dtype=np.float32)
    if round_x.ndim != 3 or kline_x.ndim != 3 or snapshot_x.ndim != 2:
        raise InvariantError("neural_direction_raw_tcn_predict_rank_invalid")
    round_proc = _apply_sequence_preprocessor(
        x=round_x,
        impute_values=np.asarray(bundle.round_impute_values, dtype=np.float32),
        feature_means=np.asarray(bundle.round_feature_means, dtype=np.float32),
        feature_stds=np.asarray(bundle.round_feature_stds, dtype=np.float32),
    )
    kline_proc = _apply_sequence_preprocessor(
        x=kline_x,
        impute_values=np.asarray(bundle.kline_impute_values, dtype=np.float32),
        feature_means=np.asarray(bundle.kline_feature_means, dtype=np.float32),
        feature_stds=np.asarray(bundle.kline_feature_stds, dtype=np.float32),
    )
    snapshot_proc = _apply_matrix_preprocessor(
        x=snapshot_x,
        impute_values=np.asarray(bundle.snapshot_impute_values, dtype=np.float32),
        feature_means=np.asarray(bundle.snapshot_feature_means, dtype=np.float32),
        feature_stds=np.asarray(bundle.snapshot_feature_stds, dtype=np.float32),
    )
    model = NeuralDirectionRawTcn(
        round_input_dim=int(round_proc.shape[2]),
        kline_input_dim=int(kline_proc.shape[2]),
        snapshot_input_dim=int(snapshot_proc.shape[1]),
        config=bundle.config,
    )
    model.load_state_dict(bundle.state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(round_proc),
            torch.from_numpy(kline_proc),
            torch.from_numpy(snapshot_proc),
        )
        probs = torch.sigmoid(logits).cpu().numpy()
    return np.asarray(probs, dtype=np.float32)


def save_neural_direction_raw_tcn_bundle(
    *,
    bundle: NeuralDirectionRawTcnBundle,
    path: str,
) -> None:
    out_path = Path(str(path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "round_feature_columns": tuple(str(col) for col in bundle.round_feature_columns),
        "kline_feature_columns": tuple(str(col) for col in bundle.kline_feature_columns),
        "snapshot_feature_columns": tuple(str(col) for col in bundle.snapshot_feature_columns),
        "config": asdict(bundle.config),
        "round_impute_values": np.asarray(bundle.round_impute_values, dtype=np.float32),
        "round_feature_means": np.asarray(bundle.round_feature_means, dtype=np.float32),
        "round_feature_stds": np.asarray(bundle.round_feature_stds, dtype=np.float32),
        "kline_impute_values": np.asarray(bundle.kline_impute_values, dtype=np.float32),
        "kline_feature_means": np.asarray(bundle.kline_feature_means, dtype=np.float32),
        "kline_feature_stds": np.asarray(bundle.kline_feature_stds, dtype=np.float32),
        "snapshot_impute_values": np.asarray(bundle.snapshot_impute_values, dtype=np.float32),
        "snapshot_feature_means": np.asarray(bundle.snapshot_feature_means, dtype=np.float32),
        "snapshot_feature_stds": np.asarray(bundle.snapshot_feature_stds, dtype=np.float32),
        "state_dict": {
            str(key): value.detach().cpu()
            for key, value in bundle.state_dict.items()
        },
        "metadata": dict(bundle.metadata),
    }
    torch.save(payload, str(out_path))


def load_neural_direction_raw_tcn_bundle(path: str) -> NeuralDirectionRawTcnBundle:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise InvariantError("neural_direction_raw_tcn_payload_invalid")
    config_raw = payload.get("config")
    if not isinstance(config_raw, dict):
        raise InvariantError("neural_direction_raw_tcn_config_missing")
    return NeuralDirectionRawTcnBundle(
        round_feature_columns=tuple(str(col) for col in payload["round_feature_columns"]),
        kline_feature_columns=tuple(str(col) for col in payload["kline_feature_columns"]),
        snapshot_feature_columns=tuple(str(col) for col in payload["snapshot_feature_columns"]),
        config=NeuralDirectionRawTcnConfig(
            round_channels=tuple(int(v) for v in config_raw["round_channels"]),
            kline_channels=tuple(int(v) for v in config_raw["kline_channels"]),
            kernel_size=int(config_raw["kernel_size"]),
            snapshot_hidden_sizes=tuple(int(v) for v in config_raw["snapshot_hidden_sizes"]),
            fusion_hidden_sizes=tuple(int(v) for v in config_raw["fusion_hidden_sizes"]),
            dropout=float(config_raw["dropout"]),
            learning_rate=float(config_raw["learning_rate"]),
            weight_decay=float(config_raw["weight_decay"]),
            batch_size=int(config_raw["batch_size"]),
            max_epochs=int(config_raw["max_epochs"]),
            patience_epochs=int(config_raw["patience_epochs"]),
        ),
        round_impute_values=np.asarray(payload["round_impute_values"], dtype=np.float32),
        round_feature_means=np.asarray(payload["round_feature_means"], dtype=np.float32),
        round_feature_stds=np.asarray(payload["round_feature_stds"], dtype=np.float32),
        kline_impute_values=np.asarray(payload["kline_impute_values"], dtype=np.float32),
        kline_feature_means=np.asarray(payload["kline_feature_means"], dtype=np.float32),
        kline_feature_stds=np.asarray(payload["kline_feature_stds"], dtype=np.float32),
        snapshot_impute_values=np.asarray(payload["snapshot_impute_values"], dtype=np.float32),
        snapshot_feature_means=np.asarray(payload["snapshot_feature_means"], dtype=np.float32),
        snapshot_feature_stds=np.asarray(payload["snapshot_feature_stds"], dtype=np.float32),
        state_dict={
            str(key): value.detach().cpu()
            for key, value in dict(payload["state_dict"]).items()
        },
        metadata=dict(payload.get("metadata", {})),
    )


def _validate_dataset_for_raw_tcn(*, dataset: NeuralDirectionRawSequenceDataset) -> None:
    if int(dataset.round_sequence.ndim) != 3:
        raise InvariantError("neural_direction_raw_tcn_round_rank_invalid")
    if int(dataset.kline_sequence.ndim) != 3:
        raise InvariantError("neural_direction_raw_tcn_kline_rank_invalid")
    if int(dataset.snapshot_matrix.ndim) != 2:
        raise InvariantError("neural_direction_raw_tcn_snapshot_rank_invalid")
    if int(dataset.round_sequence.shape[0]) != int(dataset.num_examples):
        raise InvariantError("neural_direction_raw_tcn_round_rows_len_mismatch")
    if int(dataset.kline_sequence.shape[0]) != int(dataset.num_examples):
        raise InvariantError("neural_direction_raw_tcn_kline_rows_len_mismatch")
    if int(dataset.snapshot_matrix.shape[0]) != int(dataset.num_examples):
        raise InvariantError("neural_direction_raw_tcn_snapshot_rows_len_mismatch")
    if len(dataset.labels) != int(dataset.num_examples):
        raise InvariantError("neural_direction_raw_tcn_labels_len_mismatch")


def _fit_sequence_preprocessor(*, train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(train_x, dtype=np.float32)
    if x.ndim != 3:
        raise InvariantError("neural_direction_raw_tcn_sequence_preprocessor_rank_invalid")
    flat = x.reshape(-1, int(x.shape[2]))
    impute_values = np.nanmean(flat, axis=0)
    impute_values = np.where(np.isnan(impute_values), 0.0, impute_values).astype(np.float32)
    flat_imputed = np.where(np.isnan(flat), impute_values, flat).astype(np.float32)
    feature_means = np.asarray(np.mean(flat_imputed, axis=0), dtype=np.float32)
    feature_stds = np.asarray(np.std(flat_imputed, axis=0), dtype=np.float32)
    feature_stds = np.where(feature_stds <= 1e-6, 1.0, feature_stds).astype(np.float32)
    return impute_values, feature_means, feature_stds


def _apply_sequence_preprocessor(
    *,
    x: np.ndarray,
    impute_values: np.ndarray,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(x, dtype=np.float32)
    if raw.ndim != 3:
        raise InvariantError("neural_direction_raw_tcn_sequence_apply_rank_invalid")
    x_imputed = np.where(np.isnan(raw), impute_values.reshape(1, 1, -1), raw).astype(np.float32)
    return ((x_imputed - feature_means.reshape(1, 1, -1)) / feature_stds.reshape(1, 1, -1)).astype(np.float32)


def _fit_matrix_preprocessor(*, train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(train_x, dtype=np.float32)
    if x.ndim != 2:
        raise InvariantError("neural_direction_raw_tcn_matrix_preprocessor_rank_invalid")
    impute_values = np.nanmean(x, axis=0)
    impute_values = np.where(np.isnan(impute_values), 0.0, impute_values).astype(np.float32)
    x_imputed = np.where(np.isnan(x), impute_values, x).astype(np.float32)
    feature_means = np.asarray(np.mean(x_imputed, axis=0), dtype=np.float32)
    feature_stds = np.asarray(np.std(x_imputed, axis=0), dtype=np.float32)
    feature_stds = np.where(feature_stds <= 1e-6, 1.0, feature_stds).astype(np.float32)
    return impute_values, feature_means, feature_stds


def _apply_matrix_preprocessor(
    *,
    x: np.ndarray,
    impute_values: np.ndarray,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(x, dtype=np.float32)
    if raw.ndim != 2:
        raise InvariantError("neural_direction_raw_tcn_matrix_apply_rank_invalid")
    x_imputed = np.where(np.isnan(raw), impute_values, raw).astype(np.float32)
    return ((x_imputed - feature_means) / feature_stds).astype(np.float32)
