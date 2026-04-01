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
class NeuralDirectionMlpConfig:
    hidden_sizes: tuple[int, ...] = (128, 64)
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 40
    patience_epochs: int = 6


@dataclass(frozen=True, slots=True)
class NeuralDirectionMlpBundle:
    feature_columns: tuple[str, ...]
    config: NeuralDirectionMlpConfig
    impute_values: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, object]


class NeuralDirectionMlp(nn.Module):
    def __init__(self, *, input_dim: int, config: NeuralDirectionMlpConfig) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise InvariantError("neural_direction_mlp_input_dim_invalid")
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_size in config.hidden_sizes:
            hidden = int(hidden_size)
            if int(hidden) <= 0:
                raise InvariantError("neural_direction_mlp_hidden_size_invalid")
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if float(config.dropout) > 0.0:
                layers.append(nn.Dropout(float(config.dropout)))
            in_dim = int(hidden)
        layers.append(nn.Linear(in_dim, 1))
        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._net(x)
        return logits.squeeze(-1)


def default_neural_direction_mlp_config() -> NeuralDirectionMlpConfig:
    return NeuralDirectionMlpConfig()


def train_neural_direction_mlp(
    *,
    dataset: NeuralDirectionDataset,
    train_target_epochs: Sequence[int],
    valid_target_epochs: Sequence[int],
    random_seed: int,
    config: NeuralDirectionMlpConfig | None = None,
    train_sample_weights: np.ndarray | None = None,
    initial_bundle: NeuralDirectionMlpBundle | None = None,
) -> NeuralDirectionMlpBundle:
    model_cfg = config or default_neural_direction_mlp_config()
    _validate_dataset_for_mlp(dataset=dataset)
    train_x, train_y = _rows_for_target_epochs(dataset=dataset, target_epochs=train_target_epochs)
    valid_x, valid_y = _rows_for_target_epochs(dataset=dataset, target_epochs=valid_target_epochs)
    if len(train_x) <= 0:
        raise InvariantError("neural_direction_mlp_train_empty")
    if len(valid_x) <= 0:
        raise InvariantError("neural_direction_mlp_valid_empty")

    train_weights = _prepare_train_sample_weights(
        train_sample_weights=train_sample_weights,
        train_example_count=int(len(train_x)),
    )
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

    model = NeuralDirectionMlp(input_dim=int(train_x_proc.shape[1]), config=model_cfg)
    if initial_bundle is not None:
        _load_initial_bundle(
            model=model,
            initial_bundle=initial_bundle,
            dataset=dataset,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.learning_rate),
        weight_decay=float(model_cfg.weight_decay),
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    train_dataset = TensorDataset(
        torch.from_numpy(train_x_proc),
        torch.from_numpy(train_y.astype(np.float32)),
        torch.from_numpy(train_weights),
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
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            losses = loss_fn(logits, batch_y)
            loss = torch.sum(losses * batch_w) / torch.clamp_min(torch.sum(batch_w), 1e-8)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_x_tensor)
            valid_loss = float(torch.mean(loss_fn(valid_logits, valid_y_tensor)).item())
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
        raise InvariantError("neural_direction_mlp_best_state_missing")

    return NeuralDirectionMlpBundle(
        feature_columns=tuple(str(col) for col in dataset.feature_columns),
        config=model_cfg,
        impute_values=np.asarray(impute_values, dtype=np.float32),
        feature_means=np.asarray(feature_means, dtype=np.float32),
        feature_stds=np.asarray(feature_stds, dtype=np.float32),
        state_dict=best_state,
        metadata={
            "input_dim": int(train_x_proc.shape[1]),
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
            "train_weight_min": float(np.min(train_weights)),
            "train_weight_max": float(np.max(train_weights)),
            "warm_started": bool(initial_bundle is not None),
        },
    )


def predict_neural_direction_probabilities(
    *,
    bundle: NeuralDirectionMlpBundle,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    x = np.asarray(feature_matrix, dtype=np.float32)
    if x.ndim != 2:
        raise InvariantError("neural_direction_mlp_predict_x_rank_invalid")
    x_proc = _apply_preprocessor(
        x=x,
        impute_values=np.asarray(bundle.impute_values, dtype=np.float32),
        feature_means=np.asarray(bundle.feature_means, dtype=np.float32),
        feature_stds=np.asarray(bundle.feature_stds, dtype=np.float32),
    )
    model = NeuralDirectionMlp(
        input_dim=int(x_proc.shape[1]),
        config=bundle.config,
    )
    model.load_state_dict(bundle.state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_proc))
        probs = torch.sigmoid(logits).cpu().numpy()
    return np.asarray(probs, dtype=np.float32)


def save_neural_direction_mlp_bundle(*, bundle: NeuralDirectionMlpBundle, path: str) -> None:
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


def load_neural_direction_mlp_bundle(path: str) -> NeuralDirectionMlpBundle:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise InvariantError("neural_direction_mlp_payload_invalid")
    config_raw = payload.get("config")
    if not isinstance(config_raw, dict):
        raise InvariantError("neural_direction_mlp_config_missing")
    bundle = NeuralDirectionMlpBundle(
        feature_columns=tuple(str(col) for col in payload["feature_columns"]),
        config=NeuralDirectionMlpConfig(
            hidden_sizes=tuple(int(v) for v in config_raw["hidden_sizes"]),
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
    return bundle


def _validate_dataset_for_mlp(*, dataset: NeuralDirectionDataset) -> None:
    if int(dataset.feature_matrix.ndim) != 2:
        raise InvariantError("neural_direction_mlp_feature_rank_invalid")
    if int(dataset.feature_matrix.shape[0]) != int(dataset.num_examples):
        raise InvariantError("neural_direction_mlp_feature_rows_len_mismatch")
    if len(dataset.labels) != int(dataset.num_examples):
        raise InvariantError("neural_direction_mlp_labels_len_mismatch")


def _rows_for_target_epochs(
    *,
    dataset: NeuralDirectionDataset,
    target_epochs: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not target_epochs:
        raise InvariantError("neural_direction_target_epochs_empty")
    index_by_epoch = {
        int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)
    }
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    seen: set[int] = set()
    for epoch in target_epochs:
        key = int(epoch)
        if int(key) in seen:
            raise InvariantError("neural_direction_target_epochs_duplicate")
        seen.add(int(key))
        idx = index_by_epoch.get(int(key))
        if idx is None:
            raise InvariantError("neural_direction_target_epoch_missing")
        x_rows.append(np.asarray(dataset.feature_matrix[int(idx)], dtype=np.float32))
        y_rows.append(int(dataset.labels[int(idx)]))
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.int64),
    )


def _fit_preprocessor(*, train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(train_x, dtype=np.float32)
    if x.ndim != 2:
        raise InvariantError("neural_direction_preprocessor_train_rank_invalid")
    impute_values = np.nanmean(x, axis=0)
    impute_values = np.where(np.isnan(impute_values), 0.0, impute_values).astype(np.float32)
    x_imputed = np.where(np.isnan(x), impute_values, x).astype(np.float32)
    feature_means = np.asarray(np.mean(x_imputed, axis=0), dtype=np.float32)
    feature_stds = np.asarray(np.std(x_imputed, axis=0), dtype=np.float32)
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
    if raw.ndim != 2:
        raise InvariantError("neural_direction_preprocessor_x_rank_invalid")
    x_imputed = np.where(np.isnan(raw), impute_values, raw).astype(np.float32)
    return ((x_imputed - feature_means) / feature_stds).astype(np.float32)


def build_exponential_recency_weights(
    *,
    num_examples: int,
    half_life_examples: float,
) -> np.ndarray:
    if int(num_examples) <= 0:
        raise InvariantError("neural_direction_mlp_recency_num_examples_nonpositive")
    if float(half_life_examples) <= 0.0:
        raise InvariantError("neural_direction_mlp_recency_half_life_nonpositive")
    age_from_newest = np.arange(int(num_examples) - 1, -1, -1, dtype=np.float32)
    raw = np.power(np.float32(0.5), age_from_newest / float(half_life_examples)).astype(np.float32)
    mean_value = float(np.mean(raw))
    if not np.isfinite(mean_value) or float(mean_value) <= 0.0:
        raise InvariantError("neural_direction_mlp_recency_weights_invalid")
    return np.asarray(raw / mean_value, dtype=np.float32)


def _prepare_train_sample_weights(
    *,
    train_sample_weights: np.ndarray | None,
    train_example_count: int,
) -> np.ndarray:
    if int(train_example_count) <= 0:
        raise InvariantError("neural_direction_mlp_train_example_count_nonpositive")
    if train_sample_weights is None:
        return np.ones(int(train_example_count), dtype=np.float32)
    weights = np.asarray(train_sample_weights, dtype=np.float32)
    if weights.ndim != 1:
        raise InvariantError("neural_direction_mlp_train_weights_rank_invalid")
    if int(len(weights)) != int(train_example_count):
        raise InvariantError("neural_direction_mlp_train_weights_len_mismatch")
    if bool(np.any(~np.isfinite(weights))):
        raise InvariantError("neural_direction_mlp_train_weights_nonfinite")
    if bool(np.any(weights < 0.0)):
        raise InvariantError("neural_direction_mlp_train_weights_negative")
    total_weight = float(np.sum(weights))
    if float(total_weight) <= 0.0:
        raise InvariantError("neural_direction_mlp_train_weights_zero_sum")
    return np.asarray(weights * (float(train_example_count) / float(total_weight)), dtype=np.float32)


def _load_initial_bundle(
    *,
    model: NeuralDirectionMlp,
    initial_bundle: NeuralDirectionMlpBundle,
    dataset: NeuralDirectionDataset,
) -> None:
    if tuple(str(col) for col in initial_bundle.feature_columns) != tuple(
        str(col) for col in dataset.feature_columns
    ):
        raise InvariantError("neural_direction_mlp_initial_bundle_feature_columns_mismatch")
    try:
        model.load_state_dict(initial_bundle.state_dict)
    except Exception as exc:
        raise InvariantError(
            f"neural_direction_mlp_initial_bundle_incompatible: {exc}"
        ) from exc
