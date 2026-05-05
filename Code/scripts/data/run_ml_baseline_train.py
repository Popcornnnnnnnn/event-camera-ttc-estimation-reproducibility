#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


INTERVAL_FIELDNAMES = [
    "sample_id",
    "sequence_id",
    "interval_idx",
    "gt_ttc_s",
    "ttc_est_s",
    "target_pred",
    "abs_err_s",
    "e_ttc_pct",
    "status",
    "cost_time_s",
    "fold_test_sequence",
    "train_sequences",
    "debug_ref",
]

ProgressCallback = Callable[[dict[str, object]], None]


def variant_label_for(model_type: str, feature_mode: str) -> str:
    if model_type == "ridge":
        if feature_mode == "stats_metadata":
            return "RidgeStatsMetaV1"
        if feature_mode == "pooled_image_stats_metadata":
            return "RidgePooledV1"
    if model_type == "mlp":
        if feature_mode == "stats_metadata":
            return "TinyMlpStatsMetaV1"
        if feature_mode == "pooled_image_stats_metadata":
            return "TinyMlpPooledV1"
    if model_type == "cnn":
        if feature_mode == "image_only":
            return "TinyCnnImageOnlyV1"
        if feature_mode == "image_stats_metadata":
            return "TinyCnnStatsMetaV1"
        if feature_mode == "image_late_fusion_stats_metadata":
            return "TinyCnnLateFusionV1"
        if feature_mode == "image_residual_stats_metadata":
            return "TinyCnnResidualFusionV1"
    raise ValueError(f"Unsupported model_type={model_type!r} feature_mode={feature_mode!r}")


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(f"Invalid hidden dims: {value!r}")
    return dims


def parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(channels) != 2 or any(channel <= 0 for channel in channels):
        raise argparse.ArgumentTypeError(f"Invalid cnn channels: {value!r}")
    return channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate grouped ML baselines on exported EvTTC data.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=CODE_ROOT / "Derived" / "MlBaselineV1" / "tau20_64x64_g3x3",
        help="Directory containing per-sequence exported dataset.npz files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_ROOT / "Experiments" / "MlBaseline",
        help="Root directory for ML baseline experiment outputs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional fixed output directory name under output-root. Enables stable resume targets.",
    )
    parser.add_argument(
        "--overwrite-existing-folds",
        action="store_true",
        help="Retrain folds even when fold outputs already look complete.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence IDs to include. Default: all sequences under dataset root.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=[
            "stats_metadata",
            "pooled_image_stats_metadata",
            "image_only",
            "image_stats_metadata",
            "image_late_fusion_stats_metadata",
            "image_residual_stats_metadata",
        ],
        default="pooled_image_stats_metadata",
        help="Feature construction mode.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=8,
        help="Target spatial size for pooled image features when feature-mode uses pooled image channels.",
    )
    parser.add_argument(
        "--model",
        choices=["ridge", "mlp", "cnn"],
        default="ridge",
        help="Regressor family.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="L2 regularization strength for ridge regression.",
    )
    parser.add_argument(
        "--target",
        choices=["log_ttc", "ttc"],
        default="log_ttc",
        help="Regression target.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for data shuffling and model initialization.",
    )
    parser.add_argument(
        "--mlp-hidden-dims",
        type=parse_hidden_dims,
        default=(64, 32),
        help="Comma-separated hidden layer dims for tiny MLP, e.g. 64,32.",
    )
    parser.add_argument(
        "--mlp-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for MLP Adam optimizer.",
    )
    parser.add_argument(
        "--mlp-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for MLP.",
    )
    parser.add_argument(
        "--mlp-batch-size",
        type=int,
        default=32,
        help="Mini-batch size for MLP.",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=400,
        help="Max epochs for MLP training.",
    )
    parser.add_argument(
        "--mlp-patience",
        type=int,
        default=40,
        help="Early stopping patience in epochs for MLP.",
    )
    parser.add_argument(
        "--mlp-val-ratio",
        type=float,
        default=0.2,
        help="Per-sequence validation ratio carved from training data for MLP.",
    )
    parser.add_argument(
        "--cnn-channels",
        type=parse_channels,
        default=(6, 10),
        help="Conv channel counts for the two-layer tiny CNN, e.g. 6,10.",
    )
    parser.add_argument(
        "--cnn-head-hidden-dim",
        type=int,
        default=32,
        help="Hidden dim of the CNN regression head.",
    )
    parser.add_argument(
        "--cnn-extra-hidden-dim",
        type=int,
        default=24,
        help="Hidden dim of the late-fusion stats/metadata branch.",
    )
    parser.add_argument(
        "--cnn-learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate for CNN Adam optimizer.",
    )
    parser.add_argument(
        "--cnn-weight-decay",
        type=float,
        default=3e-4,
        help="Weight decay for CNN.",
    )
    parser.add_argument(
        "--cnn-batch-size",
        type=int,
        default=16,
        help="Mini-batch size for CNN.",
    )
    parser.add_argument(
        "--cnn-epochs",
        type=int,
        default=120,
        help="Max epochs for CNN training.",
    )
    parser.add_argument(
        "--cnn-patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs for CNN.",
    )
    parser.add_argument(
        "--cnn-val-ratio",
        type=float,
        default=0.2,
        help="Per-sequence validation ratio carved from training data for CNN.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    valid_feature_modes = {
        "ridge": {"stats_metadata", "pooled_image_stats_metadata"},
        "mlp": {"stats_metadata", "pooled_image_stats_metadata"},
        "cnn": {"image_only", "image_stats_metadata", "image_late_fusion_stats_metadata", "image_residual_stats_metadata"},
    }
    if args.feature_mode not in valid_feature_modes[args.model]:
        raise SystemExit(
            f"feature-mode {args.feature_mode!r} is not supported for model {args.model!r}. "
            f"Allowed: {sorted(valid_feature_modes[args.model])}"
        )
    if args.pool_size <= 0:
        raise SystemExit("--pool-size must be positive")
    if args.cnn_head_hidden_dim <= 0:
        raise SystemExit("--cnn-head-hidden-dim must be positive")
    if args.cnn_extra_hidden_dim <= 0:
        raise SystemExit("--cnn-extra-hidden-dim must be positive")


def list_sequence_dirs(dataset_root: Path) -> list[str]:
    if not dataset_root.exists():
        return []
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir() and (path / "dataset.npz").exists())


def load_sequence_dataset(dataset_root: Path, sequence_id: str) -> dict[str, np.ndarray]:
    dataset_path = dataset_root / sequence_id / "dataset.npz"
    with np.load(dataset_path) as data:
        return {key: data[key] for key in data.files}


def mean_pool_image(image_nchw: np.ndarray, pool_size: int) -> np.ndarray:
    n, c, h, w = image_nchw.shape
    row_edges = np.linspace(0, h, pool_size + 1, dtype=np.int64)
    col_edges = np.linspace(0, w, pool_size + 1, dtype=np.int64)
    pooled = np.zeros((n, c, pool_size, pool_size), dtype=np.float32)
    for i in range(pool_size):
        r0, r1 = int(row_edges[i]), int(row_edges[i + 1])
        for j in range(pool_size):
            c0, c1 = int(col_edges[j]), int(col_edges[j + 1])
            cell = image_nchw[:, :, r0:r1, c0:c1]
            pooled[:, :, i, j] = cell.mean(axis=(2, 3))
    return pooled


def build_stats_metadata(dataset: dict[str, np.ndarray]) -> np.ndarray:
    stats = dataset["stats"].astype(np.float32)
    metadata = dataset["metadata"].astype(np.float32)
    return np.concatenate([stats, metadata], axis=1).astype(np.float32)


def build_tabular_features(dataset: dict[str, np.ndarray], *, feature_mode: str, pool_size: int) -> np.ndarray:
    stats_metadata = build_stats_metadata(dataset)
    if feature_mode == "stats_metadata":
        return stats_metadata

    image = dataset["image_nchw"].astype(np.float32)
    pooled = mean_pool_image(image, pool_size).reshape(image.shape[0], -1)
    return np.concatenate([pooled, stats_metadata], axis=1).astype(np.float32)


def select_target(dataset: dict[str, np.ndarray], target_name: str) -> np.ndarray:
    if target_name == "log_ttc":
        return dataset["gt_log_ttc_s"].astype(np.float64)
    return dataset["gt_ttc_s"].astype(np.float64)


def standardize_vector_features(
    train_x: np.ndarray,
    other_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (other_x - mean) / std, mean, std


def standardize_image_features(
    train_image: np.ndarray,
    other_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_image.mean(axis=(0, 2, 3), keepdims=True)
    std = train_image.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_image - mean) / std, (other_image - mean) / std, mean, std


def standardize_target(
    train_y: np.ndarray,
    other_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean())
    std = float(train_y.std())
    if std < 1e-6:
        std = 1.0
    return (train_y - mean) / std, (other_y - mean) / std, mean, std


def fit_ridge_regression(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x = train_x.astype(np.float64)
    y = train_y.astype(np.float64)
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    x_aug = np.concatenate([ones, x], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    bias = float(weights[0])
    coef = weights[1:]
    return coef, bias


def predict_ridge_regression(x: np.ndarray, coef: np.ndarray, bias: float) -> np.ndarray:
    return x.astype(np.float64) @ coef + bias


def split_train_val_indices(
    train_sequence_ids: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []

    for sequence_id in sorted(set(str(item) for item in train_sequence_ids.tolist())):
        seq_indices = np.flatnonzero(train_sequence_ids == sequence_id)
        shuffled = seq_indices.copy()
        rng.shuffle(shuffled)
        if val_ratio <= 0.0 or len(shuffled) < 2:
            train_indices.append(shuffled)
            continue
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1
        val_indices.append(shuffled[:val_count])
        train_indices.append(shuffled[val_count:])

    train_idx = np.concatenate(train_indices, axis=0)
    val_idx = np.concatenate(val_indices, axis=0) if val_indices else np.empty((0,), dtype=np.int64)
    return train_idx, val_idx


def init_mlp_layers(input_dim: int, hidden_dims: tuple[int, ...], seed: int) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    layer_dims = (input_dim, *hidden_dims, 1)
    layers: list[dict[str, np.ndarray]] = []
    for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
        weight_scale = np.sqrt(2.0 / float(in_dim))
        layers.append(
            {
                "weight": (rng.standard_normal((in_dim, out_dim)) * weight_scale).astype(np.float64),
                "bias": np.zeros((1, out_dim), dtype=np.float64),
            }
        )
    return layers


def copy_layers(layers: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    return [{"weight": layer["weight"].copy(), "bias": layer["bias"].copy()} for layer in layers]


def forward_mlp(
    x: np.ndarray,
    layers: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    activations = [x.astype(np.float64)]
    preactivations: list[np.ndarray] = []
    current = activations[0]
    for layer_idx, layer in enumerate(layers):
        z = current @ layer["weight"] + layer["bias"]
        preactivations.append(z)
        if layer_idx == len(layers) - 1:
            current = z
        else:
            current = np.maximum(z, 0.0)
        activations.append(current)
    return activations[-1].reshape(-1), activations, preactivations


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred - target
    return float(np.mean(diff * diff))


def backward_mlp(
    activations: list[np.ndarray],
    preactivations: list[np.ndarray],
    layers: list[dict[str, np.ndarray]],
    pred: np.ndarray,
    target: np.ndarray,
    *,
    weight_decay: float,
) -> list[dict[str, np.ndarray]]:
    batch_size = max(1, target.shape[0])
    grad = (2.0 / float(batch_size)) * (pred - target).reshape(-1, 1)
    grads: list[dict[str, np.ndarray]] = []

    for layer_idx in range(len(layers) - 1, -1, -1):
        layer = layers[layer_idx]
        a_prev = activations[layer_idx]
        grad_weight = a_prev.T @ grad + weight_decay * layer["weight"]
        grad_bias = grad.sum(axis=0, keepdims=True)
        grads.append({"weight": grad_weight, "bias": grad_bias})
        if layer_idx > 0:
            grad = grad @ layer["weight"].T
            grad = grad * (preactivations[layer_idx - 1] > 0.0)

    grads.reverse()
    return grads


def evaluate_mlp(x: np.ndarray, y: np.ndarray, layers: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, float]:
    pred, _, _ = forward_mlp(x, layers)
    return pred, mse_loss(pred, y)


def train_mlp_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    hidden_dims: tuple[int, ...],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, np.ndarray]], dict[str, float | int], list[dict[str, float | int]]]:
    layers = init_mlp_layers(train_x.shape[1], hidden_dims, seed=seed)
    best_layers = copy_layers(layers)
    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    history: list[dict[str, float | int]] = []

    adam_m = [
        {"weight": np.zeros_like(layer["weight"]), "bias": np.zeros_like(layer["bias"])}
        for layer in layers
    ]
    adam_v = [
        {"weight": np.zeros_like(layer["weight"]), "bias": np.zeros_like(layer["bias"])}
        for layer in layers
    ]
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    global_step = 0
    stale_epochs = 0
    rng = np.random.default_rng(seed)
    batch_count = int(np.ceil(train_x.shape[0] / float(batch_size)))
    last_progress_at = 0.0

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(train_x.shape[0])
        shuffled_x = train_x[permutation]
        shuffled_y = train_y[permutation]

        for batch_idx, start in enumerate(range(0, train_x.shape[0], batch_size), start=1):
            stop = min(start + batch_size, train_x.shape[0])
            batch_x = shuffled_x[start:stop]
            batch_y = shuffled_y[start:stop]

            pred_batch, activations, preactivations = forward_mlp(batch_x, layers)
            grads = backward_mlp(
                activations,
                preactivations,
                layers,
                pred_batch,
                batch_y,
                weight_decay=weight_decay,
            )
            global_step += 1

            for layer_idx, layer in enumerate(layers):
                for name in ("weight", "bias"):
                    adam_m[layer_idx][name] = beta1 * adam_m[layer_idx][name] + (1.0 - beta1) * grads[layer_idx][name]
                    adam_v[layer_idx][name] = beta2 * adam_v[layer_idx][name] + (1.0 - beta2) * (
                        grads[layer_idx][name] * grads[layer_idx][name]
                    )
                    m_hat = adam_m[layer_idx][name] / (1.0 - beta1**global_step)
                    v_hat = adam_v[layer_idx][name] / (1.0 - beta2**global_step)
                    layer[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            now = time.perf_counter()
            if progress_callback is not None and (
                now - last_progress_at >= 10.0 or batch_idx == batch_count
            ):
                progress_callback(
                    {
                        "phase": "batch",
                        "epoch": int(epoch),
                        "epochs": int(epochs),
                        "batch": int(batch_idx),
                        "batch_count": int(batch_count),
                        "global_step": int(global_step),
                    }
                )
                last_progress_at = now

        train_pred, train_loss = evaluate_mlp(train_x, train_y, layers)
        val_pred, val_loss = evaluate_mlp(val_x, val_y, layers)
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_pred_mean": float(train_pred.mean()),
            "val_pred_mean": float(val_pred.mean()),
        }
        history.append(history_row)

        improved = val_loss + 1e-9 < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_layers = copy_layers(layers)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "epoch",
                    **history_row,
                    "epochs": int(epochs),
                    "best_epoch": int(best_epoch),
                    "best_train_loss": float(best_train_loss),
                    "best_val_loss": float(best_val_loss),
                    "stale_epochs": int(stale_epochs),
                    "patience": int(patience),
                }
            )
        if not improved and stale_epochs >= patience:
            break

    summary = {
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "trained_epochs": int(len(history)),
    }
    return best_layers, summary, history


def avg_pool2x2(x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    n, c, h, w = x.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"avg_pool2x2 requires even spatial dims, got {x.shape}")
    reshaped = x.reshape(n, c, h // 2, 2, w // 2, 2)
    return reshaped.mean(axis=(3, 5)), x.shape


def avg_pool2x2_backward(grad_out: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    n, c, h, w = input_shape
    grad = np.repeat(np.repeat(grad_out, 2, axis=2), 2, axis=3)
    return grad / 4.0


def global_avg_pool(x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    return x.mean(axis=(2, 3)), x.shape


def global_avg_pool_backward(grad_out: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    n, c, h, w = input_shape
    scale = 1.0 / float(h * w)
    return np.broadcast_to(grad_out[:, :, None, None] * scale, input_shape).copy()


def conv2d_same_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    windows = sliding_window_view(padded, (kernel_h, kernel_w), axis=(2, 3))
    out = np.tensordot(windows, weight, axes=([1, 4, 5], [1, 2, 3]))
    out = out.transpose(0, 3, 1, 2)
    out += bias.reshape(1, -1, 1, 1)
    return out, (x, windows)


def conv2d_same_backward(
    grad_out: np.ndarray,
    weight: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, windows = cache
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    grad_bias = grad_out.sum(axis=(0, 2, 3))
    grad_weight = np.tensordot(grad_out, windows, axes=([0, 2, 3], [0, 2, 3]))

    n, c, h, w = x.shape
    grad_padded = np.zeros((n, c, h + 2 * pad_h, w + 2 * pad_w), dtype=np.float64)
    for kernel_i in range(kernel_h):
        for kernel_j in range(kernel_w):
            grad_padded[:, :, kernel_i : kernel_i + h, kernel_j : kernel_j + w] += np.einsum(
                "nohw,oc->nchw",
                grad_out,
                weight[:, :, kernel_i, kernel_j],
            )
    grad_input = grad_padded[:, :, pad_h : pad_h + h, pad_w : pad_w + w]
    return grad_input, grad_weight, grad_bias


def init_cnn_params(
    input_channels: int,
    extra_dim: int,
    conv_channels: tuple[int, int],
    head_hidden_dim: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    conv1_out, conv2_out = conv_channels
    gap_dim = conv2_out
    head_input_dim = gap_dim + extra_dim

    def he_weight(shape: tuple[int, ...], fan_in: int) -> np.ndarray:
        return (rng.standard_normal(shape) * np.sqrt(2.0 / float(fan_in))).astype(np.float64)

    return {
        "conv1_weight": he_weight((conv1_out, input_channels, 5, 5), input_channels * 5 * 5),
        "conv1_bias": np.zeros((conv1_out,), dtype=np.float64),
        "conv2_weight": he_weight((conv2_out, conv1_out, 3, 3), conv1_out * 3 * 3),
        "conv2_bias": np.zeros((conv2_out,), dtype=np.float64),
        "head1_weight": he_weight((head_input_dim, head_hidden_dim), head_input_dim),
        "head1_bias": np.zeros((1, head_hidden_dim), dtype=np.float64),
        "head2_weight": he_weight((head_hidden_dim, 1), head_hidden_dim),
        "head2_bias": np.zeros((1, 1), dtype=np.float64),
    }


def init_cnn_late_fusion_params(
    input_channels: int,
    extra_dim: int,
    conv_channels: tuple[int, int],
    image_hidden_dim: int,
    extra_hidden_dim: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    conv1_out, conv2_out = conv_channels
    gap_dim = conv2_out

    def he_weight(shape: tuple[int, ...], fan_in: int) -> np.ndarray:
        return (rng.standard_normal(shape) * np.sqrt(2.0 / float(fan_in))).astype(np.float64)

    return {
        "conv1_weight": he_weight((conv1_out, input_channels, 5, 5), input_channels * 5 * 5),
        "conv1_bias": np.zeros((conv1_out,), dtype=np.float64),
        "conv2_weight": he_weight((conv2_out, conv1_out, 3, 3), conv1_out * 3 * 3),
        "conv2_bias": np.zeros((conv2_out,), dtype=np.float64),
        "image_head_weight": he_weight((gap_dim, image_hidden_dim), gap_dim),
        "image_head_bias": np.zeros((1, image_hidden_dim), dtype=np.float64),
        "extra_head_weight": he_weight((extra_dim, extra_hidden_dim), extra_dim),
        "extra_head_bias": np.zeros((1, extra_hidden_dim), dtype=np.float64),
        "fusion_weight": he_weight((image_hidden_dim + extra_hidden_dim, 1), image_hidden_dim + extra_hidden_dim),
        "fusion_bias": np.zeros((1, 1), dtype=np.float64),
    }


def init_cnn_residual_fusion_params(
    input_channels: int,
    extra_dim: int,
    conv_channels: tuple[int, int],
    image_hidden_dim: int,
    extra_hidden_dim: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    conv1_out, conv2_out = conv_channels
    gap_dim = conv2_out

    def he_weight(shape: tuple[int, ...], fan_in: int) -> np.ndarray:
        return (rng.standard_normal(shape) * np.sqrt(2.0 / float(fan_in))).astype(np.float64)

    return {
        "conv1_weight": he_weight((conv1_out, input_channels, 5, 5), input_channels * 5 * 5),
        "conv1_bias": np.zeros((conv1_out,), dtype=np.float64),
        "conv2_weight": he_weight((conv2_out, conv1_out, 3, 3), conv1_out * 3 * 3),
        "conv2_bias": np.zeros((conv2_out,), dtype=np.float64),
        "image_head1_weight": he_weight((gap_dim, image_hidden_dim), gap_dim),
        "image_head1_bias": np.zeros((1, image_hidden_dim), dtype=np.float64),
        "image_head2_weight": he_weight((image_hidden_dim, 1), image_hidden_dim),
        "image_head2_bias": np.zeros((1, 1), dtype=np.float64),
        "extra_head1_weight": he_weight((extra_dim, extra_hidden_dim), extra_dim),
        "extra_head1_bias": np.zeros((1, extra_hidden_dim), dtype=np.float64),
        "extra_head2_weight": he_weight((extra_hidden_dim, 1), extra_hidden_dim),
        "extra_head2_bias": np.zeros((1, 1), dtype=np.float64),
        "fusion_weight": np.asarray([[1.0], [0.0]], dtype=np.float64),
        "fusion_bias": np.zeros((1, 1), dtype=np.float64),
    }


def copy_param_dict(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in params.items()}


def forward_tiny_cnn(
    image_x: np.ndarray,
    extra_x: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[int, ...]]]:
    image0, pool0_shape = avg_pool2x2(image_x.astype(np.float64))

    conv1, conv1_cache = conv2d_same_forward(image0, params["conv1_weight"], params["conv1_bias"])
    relu1 = np.maximum(conv1, 0.0)
    pool1, pool1_shape = avg_pool2x2(relu1)

    conv2, conv2_cache = conv2d_same_forward(pool1, params["conv2_weight"], params["conv2_bias"])
    relu2 = np.maximum(conv2, 0.0)
    pool2, pool2_shape = avg_pool2x2(relu2)

    gap, gap_shape = global_avg_pool(pool2)
    if extra_x.size > 0:
        head_input = np.concatenate([gap, extra_x.astype(np.float64)], axis=1)
    else:
        head_input = gap
    head1_pre = head_input @ params["head1_weight"] + params["head1_bias"]
    head1 = np.maximum(head1_pre, 0.0)
    pred = (head1 @ params["head2_weight"] + params["head2_bias"]).reshape(-1)

    cache: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[int, ...]] = {
        "image0": image0,
        "pool0_shape": pool0_shape,
        "conv1": conv1,
        "conv1_cache": conv1_cache,
        "relu1": relu1,
        "pool1_shape": pool1_shape,
        "conv2": conv2,
        "conv2_cache": conv2_cache,
        "relu2": relu2,
        "pool2": pool2,
        "pool2_shape": pool2_shape,
        "gap": gap,
        "gap_shape": gap_shape,
        "head_input": head_input,
        "head1_pre": head1_pre,
        "head1": head1,
        "extra_dim": np.asarray([extra_x.shape[1]], dtype=np.int64),
    }
    return pred, cache


def forward_tiny_cnn_late_fusion(
    image_x: np.ndarray,
    extra_x: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[int, ...]]]:
    image0, pool0_shape = avg_pool2x2(image_x.astype(np.float64))

    conv1, conv1_cache = conv2d_same_forward(image0, params["conv1_weight"], params["conv1_bias"])
    relu1 = np.maximum(conv1, 0.0)
    pool1, pool1_shape = avg_pool2x2(relu1)

    conv2, conv2_cache = conv2d_same_forward(pool1, params["conv2_weight"], params["conv2_bias"])
    relu2 = np.maximum(conv2, 0.0)
    pool2, pool2_shape = avg_pool2x2(relu2)

    gap, gap_shape = global_avg_pool(pool2)
    image_head_pre = gap @ params["image_head_weight"] + params["image_head_bias"]
    image_head = np.maximum(image_head_pre, 0.0)
    extra_head_pre = extra_x.astype(np.float64) @ params["extra_head_weight"] + params["extra_head_bias"]
    extra_head = np.maximum(extra_head_pre, 0.0)
    fusion_input = np.concatenate([image_head, extra_head], axis=1)
    pred = (fusion_input @ params["fusion_weight"] + params["fusion_bias"]).reshape(-1)

    cache: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[int, ...]] = {
        "pool0_shape": pool0_shape,
        "conv1": conv1,
        "conv1_cache": conv1_cache,
        "pool1_shape": pool1_shape,
        "conv2": conv2,
        "conv2_cache": conv2_cache,
        "pool2_shape": pool2_shape,
        "gap": gap,
        "gap_shape": gap_shape,
        "image_head_pre": image_head_pre,
        "image_head": image_head,
        "extra_head_pre": extra_head_pre,
        "extra_head": extra_head,
        "extra_x": extra_x.astype(np.float64),
        "fusion_input": fusion_input,
    }
    return pred, cache


def forward_tiny_cnn_residual_fusion(
    image_x: np.ndarray,
    extra_x: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[int, ...]]]:
    image0, pool0_shape = avg_pool2x2(image_x.astype(np.float64))

    conv1, conv1_cache = conv2d_same_forward(image0, params["conv1_weight"], params["conv1_bias"])
    relu1 = np.maximum(conv1, 0.0)
    pool1, pool1_shape = avg_pool2x2(relu1)

    conv2, conv2_cache = conv2d_same_forward(pool1, params["conv2_weight"], params["conv2_bias"])
    relu2 = np.maximum(conv2, 0.0)
    pool2, pool2_shape = avg_pool2x2(relu2)

    gap, gap_shape = global_avg_pool(pool2)
    image_head1_pre = gap @ params["image_head1_weight"] + params["image_head1_bias"]
    image_head1 = np.maximum(image_head1_pre, 0.0)
    image_scalar = image_head1 @ params["image_head2_weight"] + params["image_head2_bias"]

    extra_head1_pre = extra_x.astype(np.float64) @ params["extra_head1_weight"] + params["extra_head1_bias"]
    extra_head1 = np.maximum(extra_head1_pre, 0.0)
    extra_scalar = extra_head1 @ params["extra_head2_weight"] + params["extra_head2_bias"]

    fusion_input = np.concatenate([image_scalar, extra_scalar], axis=1)
    pred = (fusion_input @ params["fusion_weight"] + params["fusion_bias"]).reshape(-1)

    cache: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[int, ...]] = {
        "pool0_shape": pool0_shape,
        "conv1": conv1,
        "conv1_cache": conv1_cache,
        "pool1_shape": pool1_shape,
        "conv2": conv2,
        "conv2_cache": conv2_cache,
        "pool2_shape": pool2_shape,
        "gap": gap,
        "gap_shape": gap_shape,
        "image_head1_pre": image_head1_pre,
        "image_head1": image_head1,
        "image_scalar": image_scalar,
        "extra_head1_pre": extra_head1_pre,
        "extra_head1": extra_head1,
        "extra_scalar": extra_scalar,
        "extra_x": extra_x.astype(np.float64),
        "fusion_input": fusion_input,
    }
    return pred, cache


def backward_tiny_cnn(
    pred: np.ndarray,
    target: np.ndarray,
    cache: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[int, ...]],
    params: dict[str, np.ndarray],
    *,
    weight_decay: float,
) -> dict[str, np.ndarray]:
    batch_size = max(1, target.shape[0])
    grad_pred = (2.0 / float(batch_size)) * (pred - target).reshape(-1, 1)

    head1 = cache["head1"]
    head_input = cache["head_input"]
    head1_pre = cache["head1_pre"]

    grad_head2_weight = head1.T @ grad_pred + weight_decay * params["head2_weight"]
    grad_head2_bias = grad_pred.sum(axis=0, keepdims=True)
    grad_head1 = grad_pred @ params["head2_weight"].T
    grad_head1 = grad_head1 * (head1_pre > 0.0)

    grad_head1_weight = head_input.T @ grad_head1 + weight_decay * params["head1_weight"]
    grad_head1_bias = grad_head1.sum(axis=0, keepdims=True)
    grad_head_input = grad_head1 @ params["head1_weight"].T

    extra_dim = int(cache["extra_dim"][0])
    grad_gap = grad_head_input[:, :-extra_dim] if extra_dim > 0 else grad_head_input
    grad_pool2 = global_avg_pool_backward(grad_gap, cache["gap_shape"])

    grad_relu2 = avg_pool2x2_backward(grad_pool2, cache["pool2_shape"])
    grad_conv2 = grad_relu2 * (cache["conv2"] > 0.0)
    grad_pool1, grad_conv2_weight, grad_conv2_bias = conv2d_same_backward(
        grad_conv2,
        params["conv2_weight"],
        cache["conv2_cache"],
    )
    grad_conv2_weight += weight_decay * params["conv2_weight"]

    grad_relu1 = avg_pool2x2_backward(grad_pool1, cache["pool1_shape"])
    grad_conv1 = grad_relu1 * (cache["conv1"] > 0.0)
    grad_image0, grad_conv1_weight, grad_conv1_bias = conv2d_same_backward(
        grad_conv1,
        params["conv1_weight"],
        cache["conv1_cache"],
    )
    grad_conv1_weight += weight_decay * params["conv1_weight"]
    _ = avg_pool2x2_backward(grad_image0, cache["pool0_shape"])

    return {
        "conv1_weight": grad_conv1_weight,
        "conv1_bias": grad_conv1_bias,
        "conv2_weight": grad_conv2_weight,
        "conv2_bias": grad_conv2_bias,
        "head1_weight": grad_head1_weight,
        "head1_bias": grad_head1_bias,
        "head2_weight": grad_head2_weight,
        "head2_bias": grad_head2_bias,
    }


def backward_tiny_cnn_late_fusion(
    pred: np.ndarray,
    target: np.ndarray,
    cache: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[int, ...]],
    params: dict[str, np.ndarray],
    *,
    weight_decay: float,
) -> dict[str, np.ndarray]:
    batch_size = max(1, target.shape[0])
    grad_pred = (2.0 / float(batch_size)) * (pred - target).reshape(-1, 1)

    fusion_input = cache["fusion_input"]
    image_head_pre = cache["image_head_pre"]
    extra_head_pre = cache["extra_head_pre"]
    image_head = cache["image_head"]
    extra_x = cache["extra_x"]

    grad_fusion_weight = fusion_input.T @ grad_pred + weight_decay * params["fusion_weight"]
    grad_fusion_bias = grad_pred.sum(axis=0, keepdims=True)
    grad_fusion_input = grad_pred @ params["fusion_weight"].T

    image_hidden_dim = image_head.shape[1]
    grad_image_head = grad_fusion_input[:, :image_hidden_dim]
    grad_extra_head = grad_fusion_input[:, image_hidden_dim:]

    grad_image_head = grad_image_head * (image_head_pre > 0.0)
    grad_extra_head = grad_extra_head * (extra_head_pre > 0.0)

    gap = cache["gap"]
    grad_image_head_weight = gap.T @ grad_image_head + weight_decay * params["image_head_weight"]
    grad_image_head_bias = grad_image_head.sum(axis=0, keepdims=True)
    grad_gap = grad_image_head @ params["image_head_weight"].T

    grad_extra_head_weight = extra_x.T @ grad_extra_head + weight_decay * params["extra_head_weight"]
    grad_extra_head_bias = grad_extra_head.sum(axis=0, keepdims=True)

    grad_pool2 = global_avg_pool_backward(grad_gap, cache["gap_shape"])
    grad_relu2 = avg_pool2x2_backward(grad_pool2, cache["pool2_shape"])
    grad_conv2 = grad_relu2 * (cache["conv2"] > 0.0)
    grad_pool1, grad_conv2_weight, grad_conv2_bias = conv2d_same_backward(
        grad_conv2,
        params["conv2_weight"],
        cache["conv2_cache"],
    )
    grad_conv2_weight += weight_decay * params["conv2_weight"]

    grad_relu1 = avg_pool2x2_backward(grad_pool1, cache["pool1_shape"])
    grad_conv1 = grad_relu1 * (cache["conv1"] > 0.0)
    grad_image0, grad_conv1_weight, grad_conv1_bias = conv2d_same_backward(
        grad_conv1,
        params["conv1_weight"],
        cache["conv1_cache"],
    )
    grad_conv1_weight += weight_decay * params["conv1_weight"]
    _ = avg_pool2x2_backward(grad_image0, cache["pool0_shape"])

    return {
        "conv1_weight": grad_conv1_weight,
        "conv1_bias": grad_conv1_bias,
        "conv2_weight": grad_conv2_weight,
        "conv2_bias": grad_conv2_bias,
        "image_head_weight": grad_image_head_weight,
        "image_head_bias": grad_image_head_bias,
        "extra_head_weight": grad_extra_head_weight,
        "extra_head_bias": grad_extra_head_bias,
        "fusion_weight": grad_fusion_weight,
        "fusion_bias": grad_fusion_bias,
    }


def backward_tiny_cnn_residual_fusion(
    pred: np.ndarray,
    target: np.ndarray,
    cache: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[int, ...]],
    params: dict[str, np.ndarray],
    *,
    weight_decay: float,
) -> dict[str, np.ndarray]:
    batch_size = max(1, target.shape[0])
    grad_pred = (2.0 / float(batch_size)) * (pred - target).reshape(-1, 1)

    fusion_input = cache["fusion_input"]
    image_head1 = cache["image_head1"]
    image_head1_pre = cache["image_head1_pre"]
    extra_head1 = cache["extra_head1"]
    extra_head1_pre = cache["extra_head1_pre"]
    extra_x = cache["extra_x"]
    gap = cache["gap"]

    grad_fusion_weight = fusion_input.T @ grad_pred + weight_decay * params["fusion_weight"]
    grad_fusion_bias = grad_pred.sum(axis=0, keepdims=True)
    grad_fusion_input = grad_pred @ params["fusion_weight"].T

    grad_image_scalar = grad_fusion_input[:, :1]
    grad_extra_scalar = grad_fusion_input[:, 1:]

    grad_image_head2_weight = image_head1.T @ grad_image_scalar + weight_decay * params["image_head2_weight"]
    grad_image_head2_bias = grad_image_scalar.sum(axis=0, keepdims=True)
    grad_image_head1 = grad_image_scalar @ params["image_head2_weight"].T
    grad_image_head1 = grad_image_head1 * (image_head1_pre > 0.0)

    grad_image_head1_weight = gap.T @ grad_image_head1 + weight_decay * params["image_head1_weight"]
    grad_image_head1_bias = grad_image_head1.sum(axis=0, keepdims=True)
    grad_gap = grad_image_head1 @ params["image_head1_weight"].T

    grad_extra_head2_weight = extra_head1.T @ grad_extra_scalar + weight_decay * params["extra_head2_weight"]
    grad_extra_head2_bias = grad_extra_scalar.sum(axis=0, keepdims=True)
    grad_extra_head1 = grad_extra_scalar @ params["extra_head2_weight"].T
    grad_extra_head1 = grad_extra_head1 * (extra_head1_pre > 0.0)

    grad_extra_head1_weight = extra_x.T @ grad_extra_head1 + weight_decay * params["extra_head1_weight"]
    grad_extra_head1_bias = grad_extra_head1.sum(axis=0, keepdims=True)

    grad_pool2 = global_avg_pool_backward(grad_gap, cache["gap_shape"])
    grad_relu2 = avg_pool2x2_backward(grad_pool2, cache["pool2_shape"])
    grad_conv2 = grad_relu2 * (cache["conv2"] > 0.0)
    grad_pool1, grad_conv2_weight, grad_conv2_bias = conv2d_same_backward(
        grad_conv2,
        params["conv2_weight"],
        cache["conv2_cache"],
    )
    grad_conv2_weight += weight_decay * params["conv2_weight"]

    grad_relu1 = avg_pool2x2_backward(grad_pool1, cache["pool1_shape"])
    grad_conv1 = grad_relu1 * (cache["conv1"] > 0.0)
    grad_image0, grad_conv1_weight, grad_conv1_bias = conv2d_same_backward(
        grad_conv1,
        params["conv1_weight"],
        cache["conv1_cache"],
    )
    grad_conv1_weight += weight_decay * params["conv1_weight"]
    _ = avg_pool2x2_backward(grad_image0, cache["pool0_shape"])

    return {
        "conv1_weight": grad_conv1_weight,
        "conv1_bias": grad_conv1_bias,
        "conv2_weight": grad_conv2_weight,
        "conv2_bias": grad_conv2_bias,
        "image_head1_weight": grad_image_head1_weight,
        "image_head1_bias": grad_image_head1_bias,
        "image_head2_weight": grad_image_head2_weight,
        "image_head2_bias": grad_image_head2_bias,
        "extra_head1_weight": grad_extra_head1_weight,
        "extra_head1_bias": grad_extra_head1_bias,
        "extra_head2_weight": grad_extra_head2_weight,
        "extra_head2_bias": grad_extra_head2_bias,
        "fusion_weight": grad_fusion_weight,
        "fusion_bias": grad_fusion_bias,
    }


def evaluate_tiny_cnn(
    image_x: np.ndarray,
    extra_x: np.ndarray,
    target_y: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    pred, _ = forward_tiny_cnn(image_x, extra_x, params)
    return pred, mse_loss(pred, target_y)


def evaluate_tiny_cnn_late_fusion(
    image_x: np.ndarray,
    extra_x: np.ndarray,
    target_y: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    pred, _ = forward_tiny_cnn_late_fusion(image_x, extra_x, params)
    return pred, mse_loss(pred, target_y)


def evaluate_tiny_cnn_residual_fusion(
    image_x: np.ndarray,
    extra_x: np.ndarray,
    target_y: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    pred, _ = forward_tiny_cnn_residual_fusion(image_x, extra_x, params)
    return pred, mse_loss(pred, target_y)


def train_tiny_cnn_regressor(
    train_image: np.ndarray,
    train_extra: np.ndarray,
    train_y: np.ndarray,
    val_image: np.ndarray,
    val_extra: np.ndarray,
    val_y: np.ndarray,
    *,
    conv_channels: tuple[int, int],
    head_hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | int], list[dict[str, float | int]]]:
    params = init_cnn_params(
        input_channels=train_image.shape[1],
        extra_dim=train_extra.shape[1],
        conv_channels=conv_channels,
        head_hidden_dim=head_hidden_dim,
        seed=seed,
    )
    best_params = copy_param_dict(params)
    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    history: list[dict[str, float | int]] = []

    adam_m = {name: np.zeros_like(value) for name, value in params.items()}
    adam_v = {name: np.zeros_like(value) for name, value in params.items()}
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    global_step = 0
    stale_epochs = 0
    rng = np.random.default_rng(seed)
    batch_count = int(np.ceil(train_image.shape[0] / float(batch_size)))
    last_progress_at = 0.0

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(train_image.shape[0])
        shuffled_image = train_image[permutation]
        shuffled_extra = train_extra[permutation]
        shuffled_y = train_y[permutation]

        for batch_idx, start in enumerate(range(0, train_image.shape[0], batch_size), start=1):
            stop = min(start + batch_size, train_image.shape[0])
            batch_image = shuffled_image[start:stop]
            batch_extra = shuffled_extra[start:stop]
            batch_y = shuffled_y[start:stop]

            pred_batch, cache = forward_tiny_cnn(batch_image, batch_extra, params)
            grads = backward_tiny_cnn(
                pred_batch,
                batch_y,
                cache,
                params,
                weight_decay=weight_decay,
            )
            global_step += 1

            for name in params:
                adam_m[name] = beta1 * adam_m[name] + (1.0 - beta1) * grads[name]
                adam_v[name] = beta2 * adam_v[name] + (1.0 - beta2) * (grads[name] * grads[name])
                m_hat = adam_m[name] / (1.0 - beta1**global_step)
                v_hat = adam_v[name] / (1.0 - beta2**global_step)
                params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            now = time.perf_counter()
            if progress_callback is not None and (
                now - last_progress_at >= 10.0 or batch_idx == batch_count
            ):
                progress_callback(
                    {
                        "phase": "batch",
                        "epoch": int(epoch),
                        "epochs": int(epochs),
                        "batch": int(batch_idx),
                        "batch_count": int(batch_count),
                        "global_step": int(global_step),
                    }
                )
                last_progress_at = now

        train_pred, train_loss = evaluate_tiny_cnn(train_image, train_extra, train_y, params)
        val_pred, val_loss = evaluate_tiny_cnn(val_image, val_extra, val_y, params)
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_pred_mean": float(train_pred.mean()),
            "val_pred_mean": float(val_pred.mean()),
        }
        history.append(history_row)

        improved = val_loss + 1e-9 < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_params = copy_param_dict(params)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "epoch",
                    **history_row,
                    "epochs": int(epochs),
                    "best_epoch": int(best_epoch),
                    "best_train_loss": float(best_train_loss),
                    "best_val_loss": float(best_val_loss),
                    "stale_epochs": int(stale_epochs),
                    "patience": int(patience),
                }
            )
        if not improved and stale_epochs >= patience:
            break

    summary = {
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "trained_epochs": int(len(history)),
    }
    return best_params, summary, history


def train_tiny_cnn_residual_fusion_regressor(
    train_image: np.ndarray,
    train_extra: np.ndarray,
    train_y: np.ndarray,
    val_image: np.ndarray,
    val_extra: np.ndarray,
    val_y: np.ndarray,
    *,
    conv_channels: tuple[int, int],
    image_hidden_dim: int,
    extra_hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | int], list[dict[str, float | int]]]:
    params = init_cnn_residual_fusion_params(
        input_channels=train_image.shape[1],
        extra_dim=train_extra.shape[1],
        conv_channels=conv_channels,
        image_hidden_dim=image_hidden_dim,
        extra_hidden_dim=extra_hidden_dim,
        seed=seed,
    )
    best_params = copy_param_dict(params)
    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    history: list[dict[str, float | int]] = []

    adam_m = {name: np.zeros_like(value) for name, value in params.items()}
    adam_v = {name: np.zeros_like(value) for name, value in params.items()}
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    global_step = 0
    stale_epochs = 0
    rng = np.random.default_rng(seed)
    batch_count = int(np.ceil(train_image.shape[0] / float(batch_size)))
    last_progress_at = 0.0

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(train_image.shape[0])
        shuffled_image = train_image[permutation]
        shuffled_extra = train_extra[permutation]
        shuffled_y = train_y[permutation]

        for batch_idx, start in enumerate(range(0, train_image.shape[0], batch_size), start=1):
            stop = min(start + batch_size, train_image.shape[0])
            batch_image = shuffled_image[start:stop]
            batch_extra = shuffled_extra[start:stop]
            batch_y = shuffled_y[start:stop]

            pred_batch, cache = forward_tiny_cnn_residual_fusion(batch_image, batch_extra, params)
            grads = backward_tiny_cnn_residual_fusion(
                pred_batch,
                batch_y,
                cache,
                params,
                weight_decay=weight_decay,
            )
            global_step += 1

            for name in params:
                adam_m[name] = beta1 * adam_m[name] + (1.0 - beta1) * grads[name]
                adam_v[name] = beta2 * adam_v[name] + (1.0 - beta2) * (grads[name] * grads[name])
                m_hat = adam_m[name] / (1.0 - beta1**global_step)
                v_hat = adam_v[name] / (1.0 - beta2**global_step)
                params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            now = time.perf_counter()
            if progress_callback is not None and (
                now - last_progress_at >= 10.0 or batch_idx == batch_count
            ):
                progress_callback(
                    {
                        "phase": "batch",
                        "epoch": int(epoch),
                        "epochs": int(epochs),
                        "batch": int(batch_idx),
                        "batch_count": int(batch_count),
                        "global_step": int(global_step),
                    }
                )
                last_progress_at = now

        train_pred, train_loss = evaluate_tiny_cnn_residual_fusion(train_image, train_extra, train_y, params)
        val_pred, val_loss = evaluate_tiny_cnn_residual_fusion(val_image, val_extra, val_y, params)
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_pred_mean": float(train_pred.mean()),
            "val_pred_mean": float(val_pred.mean()),
        }
        history.append(history_row)

        improved = val_loss + 1e-9 < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_params = copy_param_dict(params)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "epoch",
                    **history_row,
                    "epochs": int(epochs),
                    "best_epoch": int(best_epoch),
                    "best_train_loss": float(best_train_loss),
                    "best_val_loss": float(best_val_loss),
                    "stale_epochs": int(stale_epochs),
                    "patience": int(patience),
                }
            )
        if not improved and stale_epochs >= patience:
            break

    summary = {
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "trained_epochs": int(len(history)),
    }
    return best_params, summary, history


def train_tiny_cnn_late_fusion_regressor(
    train_image: np.ndarray,
    train_extra: np.ndarray,
    train_y: np.ndarray,
    val_image: np.ndarray,
    val_extra: np.ndarray,
    val_y: np.ndarray,
    *,
    conv_channels: tuple[int, int],
    image_hidden_dim: int,
    extra_hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | int], list[dict[str, float | int]]]:
    params = init_cnn_late_fusion_params(
        input_channels=train_image.shape[1],
        extra_dim=train_extra.shape[1],
        conv_channels=conv_channels,
        image_hidden_dim=image_hidden_dim,
        extra_hidden_dim=extra_hidden_dim,
        seed=seed,
    )
    best_params = copy_param_dict(params)
    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    history: list[dict[str, float | int]] = []

    adam_m = {name: np.zeros_like(value) for name, value in params.items()}
    adam_v = {name: np.zeros_like(value) for name, value in params.items()}
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    global_step = 0
    stale_epochs = 0
    rng = np.random.default_rng(seed)
    batch_count = int(np.ceil(train_image.shape[0] / float(batch_size)))
    last_progress_at = 0.0

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(train_image.shape[0])
        shuffled_image = train_image[permutation]
        shuffled_extra = train_extra[permutation]
        shuffled_y = train_y[permutation]

        for batch_idx, start in enumerate(range(0, train_image.shape[0], batch_size), start=1):
            stop = min(start + batch_size, train_image.shape[0])
            batch_image = shuffled_image[start:stop]
            batch_extra = shuffled_extra[start:stop]
            batch_y = shuffled_y[start:stop]

            pred_batch, cache = forward_tiny_cnn_late_fusion(batch_image, batch_extra, params)
            grads = backward_tiny_cnn_late_fusion(
                pred_batch,
                batch_y,
                cache,
                params,
                weight_decay=weight_decay,
            )
            global_step += 1

            for name in params:
                adam_m[name] = beta1 * adam_m[name] + (1.0 - beta1) * grads[name]
                adam_v[name] = beta2 * adam_v[name] + (1.0 - beta2) * (grads[name] * grads[name])
                m_hat = adam_m[name] / (1.0 - beta1**global_step)
                v_hat = adam_v[name] / (1.0 - beta2**global_step)
                params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            now = time.perf_counter()
            if progress_callback is not None and (
                now - last_progress_at >= 10.0 or batch_idx == batch_count
            ):
                progress_callback(
                    {
                        "phase": "batch",
                        "epoch": int(epoch),
                        "epochs": int(epochs),
                        "batch": int(batch_idx),
                        "batch_count": int(batch_count),
                        "global_step": int(global_step),
                    }
                )
                last_progress_at = now

        train_pred, train_loss = evaluate_tiny_cnn_late_fusion(train_image, train_extra, train_y, params)
        val_pred, val_loss = evaluate_tiny_cnn_late_fusion(val_image, val_extra, val_y, params)
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_pred_mean": float(train_pred.mean()),
            "val_pred_mean": float(val_pred.mean()),
        }
        history.append(history_row)

        improved = val_loss + 1e-9 < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_params = copy_param_dict(params)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "epoch",
                    **history_row,
                    "epochs": int(epochs),
                    "best_epoch": int(best_epoch),
                    "best_train_loss": float(best_train_loss),
                    "best_val_loss": float(best_val_loss),
                    "stale_epochs": int(stale_epochs),
                    "patience": int(patience),
                }
            )
        if not improved and stale_epochs >= patience:
            break

    summary = {
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "trained_epochs": int(len(history)),
    }
    return best_params, summary, history


def fit_and_predict_tabular(
    model_type: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_sequence_ids: np.ndarray,
    test_x: np.ndarray,
    *,
    args: argparse.Namespace,
    fold_seed: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, float | int]]]:
    if model_type == "ridge":
        coef, bias = fit_ridge_regression(train_x, train_y, alpha=args.ridge_alpha)
        pred_target = predict_ridge_regression(test_x, coef, bias)
        return pred_target, {"ridge_alpha": float(args.ridge_alpha)}, []

    train_idx, val_idx = split_train_val_indices(
        train_sequence_ids,
        val_ratio=args.mlp_val_ratio,
        seed=fold_seed,
    )
    if val_idx.size == 0:
        raise ValueError("MLP requires a non-empty validation split.")

    core_train_x = train_x[train_idx]
    core_train_y = train_y[train_idx]
    val_x = train_x[val_idx]
    val_y = train_y[val_idx]

    core_train_y_norm, val_y_norm, target_mean, target_std = standardize_target(core_train_y, val_y)
    _, test_y_zero_norm, _, _ = standardize_target(core_train_y, np.zeros((test_x.shape[0],), dtype=np.float64))

    layers, training_summary, history = train_mlp_regressor(
        core_train_x,
        core_train_y_norm,
        val_x,
        val_y_norm,
        hidden_dims=args.mlp_hidden_dims,
        learning_rate=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
        batch_size=args.mlp_batch_size,
        epochs=args.mlp_epochs,
        patience=args.mlp_patience,
        seed=fold_seed,
        progress_callback=progress_callback,
    )
    test_pred_norm, _ = evaluate_mlp(test_x, test_y_zero_norm, layers)
    pred_target = test_pred_norm * target_std + target_mean

    return (
        pred_target,
        {
            "mlp_hidden_dims": list(args.mlp_hidden_dims),
            "mlp_learning_rate": float(args.mlp_learning_rate),
            "mlp_weight_decay": float(args.mlp_weight_decay),
            "mlp_batch_size": int(args.mlp_batch_size),
            "mlp_epochs": int(args.mlp_epochs),
            "mlp_patience": int(args.mlp_patience),
            "mlp_val_ratio": float(args.mlp_val_ratio),
            "seed": int(fold_seed),
            "train_count_after_val_split": int(core_train_x.shape[0]),
            "val_count": int(val_x.shape[0]),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            **training_summary,
        },
        history,
    )


def fit_and_predict_cnn(
    train_image: np.ndarray,
    train_extra: np.ndarray,
    train_y: np.ndarray,
    train_sequence_ids: np.ndarray,
    test_image: np.ndarray,
    test_extra: np.ndarray,
    *,
    args: argparse.Namespace,
    fold_seed: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, float | int]]]:
    train_idx, val_idx = split_train_val_indices(
        train_sequence_ids,
        val_ratio=args.cnn_val_ratio,
        seed=fold_seed,
    )
    if val_idx.size == 0:
        raise ValueError("CNN requires a non-empty validation split.")

    core_train_image = train_image[train_idx]
    core_train_extra = train_extra[train_idx]
    core_train_y = train_y[train_idx]
    val_image = train_image[val_idx]
    val_extra = train_extra[val_idx]
    val_y = train_y[val_idx]

    core_train_y_norm, val_y_norm, target_mean, target_std = standardize_target(core_train_y, val_y)
    _, test_y_zero_norm, _, _ = standardize_target(core_train_y, np.zeros((test_image.shape[0],), dtype=np.float64))

    if args.feature_mode == "image_late_fusion_stats_metadata":
        params, training_summary, history = train_tiny_cnn_late_fusion_regressor(
            core_train_image,
            core_train_extra,
            core_train_y_norm,
            val_image,
            val_extra,
            val_y_norm,
            conv_channels=args.cnn_channels,
            image_hidden_dim=args.cnn_head_hidden_dim,
            extra_hidden_dim=args.cnn_extra_hidden_dim,
            learning_rate=args.cnn_learning_rate,
            weight_decay=args.cnn_weight_decay,
            batch_size=args.cnn_batch_size,
            epochs=args.cnn_epochs,
            patience=args.cnn_patience,
            seed=fold_seed,
            progress_callback=progress_callback,
        )
        test_pred_norm, _ = evaluate_tiny_cnn_late_fusion(test_image, test_extra, test_y_zero_norm, params)
    elif args.feature_mode == "image_residual_stats_metadata":
        params, training_summary, history = train_tiny_cnn_residual_fusion_regressor(
            core_train_image,
            core_train_extra,
            core_train_y_norm,
            val_image,
            val_extra,
            val_y_norm,
            conv_channels=args.cnn_channels,
            image_hidden_dim=args.cnn_head_hidden_dim,
            extra_hidden_dim=args.cnn_extra_hidden_dim,
            learning_rate=args.cnn_learning_rate,
            weight_decay=args.cnn_weight_decay,
            batch_size=args.cnn_batch_size,
            epochs=args.cnn_epochs,
            patience=args.cnn_patience,
            seed=fold_seed,
            progress_callback=progress_callback,
        )
        test_pred_norm, _ = evaluate_tiny_cnn_residual_fusion(test_image, test_extra, test_y_zero_norm, params)
    else:
        params, training_summary, history = train_tiny_cnn_regressor(
            core_train_image,
            core_train_extra,
            core_train_y_norm,
            val_image,
            val_extra,
            val_y_norm,
            conv_channels=args.cnn_channels,
            head_hidden_dim=args.cnn_head_hidden_dim,
            learning_rate=args.cnn_learning_rate,
            weight_decay=args.cnn_weight_decay,
            batch_size=args.cnn_batch_size,
            epochs=args.cnn_epochs,
            patience=args.cnn_patience,
            seed=fold_seed,
            progress_callback=progress_callback,
        )
        test_pred_norm, _ = evaluate_tiny_cnn(test_image, test_extra, test_y_zero_norm, params)
    pred_target = test_pred_norm * target_std + target_mean

    return (
        pred_target,
        {
            "cnn_channels": list(args.cnn_channels),
            "cnn_head_hidden_dim": int(args.cnn_head_hidden_dim),
            "cnn_extra_hidden_dim": int(args.cnn_extra_hidden_dim),
            "cnn_learning_rate": float(args.cnn_learning_rate),
            "cnn_weight_decay": float(args.cnn_weight_decay),
            "cnn_batch_size": int(args.cnn_batch_size),
            "cnn_epochs": int(args.cnn_epochs),
            "cnn_patience": int(args.cnn_patience),
            "cnn_val_ratio": float(args.cnn_val_ratio),
            "seed": int(fold_seed),
            "train_count_after_val_split": int(core_train_image.shape[0]),
            "val_count": int(val_image.shape[0]),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            **training_summary,
        },
        history,
    )


def invert_target(pred: np.ndarray, target_name: str) -> np.ndarray:
    if target_name == "log_ttc":
        return np.exp(pred)
    return pred


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    gt = np.asarray([float(row["gt_ttc_s"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row["ttc_est_s"]) for row in rows], dtype=np.float64)
    mae_s = float(np.mean(np.abs(pred - gt)))
    e_ttc_pct = float(np.mean(np.abs(pred - gt) / np.clip(gt, 1e-6, None) * 100.0))
    return {
        "gt_valid_count": int(len(rows)),
        "est_valid_count": int(len(rows)),
        "failure_count": 0,
        "mae_s": mae_s,
        "e_ttc_pct": e_ttc_pct,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def is_complete_fold_dir(fold_dir: Path) -> bool:
    required = [fold_dir / "IntervalMetrics.csv", fold_dir / "Summary.json", fold_dir / "Summary.md"]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False
    try:
        summary = read_json(fold_dir / "Summary.json")
        rows = read_csv_rows(fold_dir / "IntervalMetrics.csv")
        return len(rows) > 0 and int(summary.get("test_count", 0)) == len(rows)
    except Exception:
        return False


def write_run_status(
    run_dir: Path,
    *,
    started_at: str,
    dataset_root: Path,
    variant_label: str,
    sequences: list[str],
    completed_folds: list[str],
    failed_folds: list[dict[str, str]],
    state: str,
    current_fold: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "state": state,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "dataset_root": str(dataset_root.resolve()),
        "variant_label": variant_label,
        "sequences": sequences,
        "total_fold_count": len(sequences),
        "completed_fold_count": len(completed_folds),
        "pending_fold_count": max(0, len(sequences) - len(completed_folds) - len(failed_folds)),
        "completed_folds": completed_folds,
        "failed_folds": failed_folds,
    }
    if current_fold is not None:
        payload["current_fold"] = current_fold
    if error is not None:
        payload["error"] = error
    write_json_atomic(
        run_dir / "RunStatus.json",
        payload,
    )


def format_number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    validate_args(args)
    sequences = args.sequences or list_sequence_dirs(args.dataset_root)
    if len(sequences) < 2:
        raise SystemExit(f"Need at least 2 sequences under {args.dataset_root}; got {sequences}")

    loaded = {sequence_id: load_sequence_dataset(args.dataset_root, sequence_id) for sequence_id in sequences}
    variant_label = variant_label_for(args.model, args.feature_mode)
    run_ts = args.run_name or datetime.now().strftime(f"{variant_label}_%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    combined_rows: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    completed_folds: list[str] = []
    failed_folds: list[dict[str, str]] = []
    current_fold_status: dict[str, object] | None = None

    def mark_run_stopped(state: str, error_type: str, message: str) -> None:
        fold_status = dict(current_fold_status or {})
        if fold_status:
            fold_status["status"] = state
        sequence_id = str(fold_status.get("sequence_id", "unknown"))
        failed_folds.append(
            {
                "sequence_id": sequence_id,
                "error_type": error_type,
                "message": message,
            }
        )
        write_run_status(
            run_dir,
            started_at=started_at,
            dataset_root=args.dataset_root,
            variant_label=variant_label,
            sequences=list(sequences),
            completed_folds=completed_folds,
            failed_folds=failed_folds,
            state=state,
            current_fold=fold_status or None,
            error=f"{error_type}: {message}",
        )

    write_run_status(
        run_dir,
        started_at=started_at,
        dataset_root=args.dataset_root,
        variant_label=variant_label,
        sequences=list(sequences),
        completed_folds=completed_folds,
        failed_folds=failed_folds,
        state="running",
    )

    for fold_idx, test_sequence_id in enumerate(sequences):
        fold_dir = run_dir / test_sequence_id
        current_fold_status = {
            "sequence_id": test_sequence_id,
            "fold_index": int(fold_idx + 1),
            "fold_count": int(len(sequences)),
            "status": "checking",
        }
        write_run_status(
            run_dir,
            started_at=started_at,
            dataset_root=args.dataset_root,
            variant_label=variant_label,
            sequences=list(sequences),
            completed_folds=completed_folds,
            failed_folds=failed_folds,
            state="running",
            current_fold=current_fold_status,
        )
        if not args.overwrite_existing_folds and is_complete_fold_dir(fold_dir):
            fold_summary = read_json(fold_dir / "Summary.json")
            rows = read_csv_rows(fold_dir / "IntervalMetrics.csv")
            fold_summaries.append(fold_summary)
            combined_rows.extend(rows)
            completed_folds.append(test_sequence_id)
            current_fold_status = {
                **current_fold_status,
                "status": "skipped_complete",
                "e_ttc_pct": float(fold_summary["e_ttc_pct"]),
            }
            write_run_status(
                run_dir,
                started_at=started_at,
                dataset_root=args.dataset_root,
                variant_label=variant_label,
                sequences=list(sequences),
                completed_folds=completed_folds,
                failed_folds=failed_folds,
                state="running",
                current_fold=current_fold_status,
            )
            print(
                f"[SkipFold] model={args.model} test={test_sequence_id} "
                f"e_ttc_pct={format_number(float(fold_summary['e_ttc_pct']), digits=3)}",
                flush=True,
            )
            continue

        train_sequence_ids = tuple(seq for seq in sequences if seq != test_sequence_id)
        train_parts = [loaded[seq] for seq in train_sequence_ids]
        test_data = loaded[test_sequence_id]
        train_target = np.concatenate([select_target(item, args.target) for item in train_parts], axis=0)
        train_sequence_array = np.concatenate([item["sequence_id"] for item in train_parts], axis=0)
        current_fold_status = {
            **current_fold_status,
            "status": "preparing",
            "train_sequence_count": int(len(train_sequence_ids)),
            "train_count": int(train_target.shape[0]),
            "test_count": int(test_data["gt_ttc_s"].shape[0]),
        }
        write_run_status(
            run_dir,
            started_at=started_at,
            dataset_root=args.dataset_root,
            variant_label=variant_label,
            sequences=list(sequences),
            completed_folds=completed_folds,
            failed_folds=failed_folds,
            state="running",
            current_fold=current_fold_status,
        )
        started_fold = time.perf_counter()

        def report_training_progress(progress: dict[str, object]) -> None:
            nonlocal current_fold_status
            elapsed_time_s = time.perf_counter() - started_fold
            current_fold_status = {
                **current_fold_status,
                "status": "training",
                "elapsed_time_s": round(elapsed_time_s, 3),
                **progress,
            }
            write_run_status(
                run_dir,
                started_at=started_at,
                dataset_root=args.dataset_root,
                variant_label=variant_label,
                sequences=list(sequences),
                completed_folds=completed_folds,
                failed_folds=failed_folds,
                state="running",
                current_fold=current_fold_status,
            )
            phase = str(progress.get("phase", "progress"))
            if phase == "batch":
                print(
                    f"[Progress] fold={fold_idx + 1}/{len(sequences)} test={test_sequence_id} "
                    f"epoch={progress.get('epoch')}/{progress.get('epochs')} "
                    f"batch={progress.get('batch')}/{progress.get('batch_count')} "
                    f"elapsed_s={elapsed_time_s:.1f}",
                    flush=True,
                )
            elif phase == "epoch":
                print(
                    f"[Epoch] fold={fold_idx + 1}/{len(sequences)} test={test_sequence_id} "
                    f"epoch={progress.get('epoch')}/{progress.get('epochs')} "
                    f"train_loss={format_number(float(progress['train_loss']))} "
                    f"val_loss={format_number(float(progress['val_loss']))} "
                    f"best_epoch={progress.get('best_epoch')} "
                    f"elapsed_s={elapsed_time_s:.1f}",
                    flush=True,
                )

        if args.model in {"ridge", "mlp"}:
            train_features = np.concatenate(
                [build_tabular_features(item, feature_mode=args.feature_mode, pool_size=args.pool_size) for item in train_parts],
                axis=0,
            )
            test_features = build_tabular_features(test_data, feature_mode=args.feature_mode, pool_size=args.pool_size)
            train_x, test_x, _, _ = standardize_vector_features(train_features, test_features)

            try:
                pred_target, train_meta, history_rows = fit_and_predict_tabular(
                    args.model,
                    train_x,
                    train_target,
                    train_sequence_array,
                    test_x,
                    args=args,
                    fold_seed=args.seed + fold_idx,
                    progress_callback=report_training_progress,
                )
            except KeyboardInterrupt:
                mark_run_stopped("interrupted", "KeyboardInterrupt", "training interrupted")
                raise SystemExit(130)
            except Exception as exc:
                mark_run_stopped("failed", type(exc).__name__, str(exc))
                raise
            cost_time_s = time.perf_counter() - started_fold
        else:
            train_image = np.concatenate([item["image_nchw"].astype(np.float32) for item in train_parts], axis=0)
            test_image = test_data["image_nchw"].astype(np.float32)
            train_image_x, test_image_x, _, _ = standardize_image_features(train_image, test_image)

            if args.feature_mode in {"image_stats_metadata", "image_late_fusion_stats_metadata", "image_residual_stats_metadata"}:
                train_extra = np.concatenate([build_stats_metadata(item) for item in train_parts], axis=0)
                test_extra = build_stats_metadata(test_data)
                train_extra_x, test_extra_x, _, _ = standardize_vector_features(train_extra, test_extra)
            else:
                train_extra_x = np.zeros((train_image_x.shape[0], 0), dtype=np.float64)
                test_extra_x = np.zeros((test_image_x.shape[0], 0), dtype=np.float64)

            try:
                pred_target, train_meta, history_rows = fit_and_predict_cnn(
                    train_image_x,
                    train_extra_x,
                    train_target,
                    train_sequence_array,
                    test_image_x,
                    test_extra_x,
                    args=args,
                    fold_seed=args.seed + fold_idx,
                    progress_callback=report_training_progress,
                )
            except KeyboardInterrupt:
                mark_run_stopped("interrupted", "KeyboardInterrupt", "training interrupted")
                raise SystemExit(130)
            except Exception as exc:
                mark_run_stopped("failed", type(exc).__name__, str(exc))
                raise
            cost_time_s = time.perf_counter() - started_fold

        pred_ttc_s = invert_target(pred_target, args.target)
        pred_ttc_s = np.clip(pred_ttc_s, 1e-6, None)
        gt_ttc_s = test_data["gt_ttc_s"].astype(np.float64)

        rows: list[dict[str, object]] = []
        for idx in range(len(gt_ttc_s)):
            abs_err_s = float(abs(pred_ttc_s[idx] - gt_ttc_s[idx]))
            e_ttc_pct = float(abs_err_s / max(1e-6, float(gt_ttc_s[idx])) * 100.0)
            rows.append(
                {
                    "sample_id": str(test_data["sample_id"][idx]),
                    "sequence_id": str(test_data["sequence_id"][idx]),
                    "interval_idx": int(test_data["interval_idx"][idx]),
                    "gt_ttc_s": float(gt_ttc_s[idx]),
                    "ttc_est_s": float(pred_ttc_s[idx]),
                    "target_pred": float(pred_target[idx]),
                    "abs_err_s": abs_err_s,
                    "e_ttc_pct": e_ttc_pct,
                    "status": "OK",
                    "cost_time_s": cost_time_s / max(1, len(gt_ttc_s)),
                    "fold_test_sequence": test_sequence_id,
                    "train_sequences": ",".join(train_sequence_ids),
                    "debug_ref": "",
                }
            )

        fold_summary = {
            "sequence_id": test_sequence_id,
            "method_family": "MlBaseline",
            "variant": variant_label,
            "model": args.model,
            "feature_mode": args.feature_mode,
            "pool_size": args.pool_size,
            "target": args.target,
            "train_sequences": list(train_sequence_ids),
            "train_count": int(train_target.shape[0]),
            "test_count": int(len(gt_ttc_s)),
            **train_meta,
            **summarize_rows(rows),
        }
        fold_summaries.append(fold_summary)
        combined_rows.extend(rows)

        write_csv(fold_dir / "IntervalMetrics.csv", rows, fieldnames=INTERVAL_FIELDNAMES)
        if history_rows:
            write_csv(
                fold_dir / "TrainHistory.csv",
                history_rows,
                fieldnames=["epoch", "train_loss", "val_loss", "train_pred_mean", "val_pred_mean"],
            )
        (fold_dir / "Summary.json").write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        fold_md = [
            f"# {test_sequence_id} / {variant_label}",
            "",
            f"- model: `{args.model}`",
            f"- train_sequences: `{', '.join(train_sequence_ids)}`",
            f"- train_count: `{fold_summary['train_count']}`",
            f"- test_count: `{fold_summary['test_count']}`",
            f"- mae_s: `{format_number(fold_summary['mae_s'])}`",
            f"- e_ttc_pct: `{format_number(fold_summary['e_ttc_pct'], digits=3)}`",
        ]
        if args.model == "ridge":
            fold_md.append(f"- ridge_alpha: `{fold_summary['ridge_alpha']}`")
        elif args.model == "mlp":
            fold_md.extend(
                [
                    f"- mlp_hidden_dims: `{fold_summary['mlp_hidden_dims']}`",
                    f"- train_count_after_val_split: `{fold_summary['train_count_after_val_split']}`",
                    f"- val_count: `{fold_summary['val_count']}`",
                    f"- best_epoch: `{fold_summary['best_epoch']}`",
                    f"- best_val_loss: `{format_number(float(fold_summary['best_val_loss']))}`",
                ]
            )
        else:
            fold_md.extend(
                [
                    f"- cnn_channels: `{fold_summary['cnn_channels']}`",
                    f"- cnn_head_hidden_dim: `{fold_summary['cnn_head_hidden_dim']}`",
                    f"- cnn_extra_hidden_dim: `{fold_summary['cnn_extra_hidden_dim']}`",
                    f"- train_count_after_val_split: `{fold_summary['train_count_after_val_split']}`",
                    f"- val_count: `{fold_summary['val_count']}`",
                    f"- best_epoch: `{fold_summary['best_epoch']}`",
                    f"- best_val_loss: `{format_number(float(fold_summary['best_val_loss']))}`",
                ]
            )
        fold_md.append("")
        (fold_dir / "Summary.md").write_text("\n".join(fold_md), encoding="utf-8")
        print(
            f"[Fold] model={args.model} test={test_sequence_id} train={','.join(train_sequence_ids)} "
            f"mae={format_number(fold_summary['mae_s'])} "
            f"e_ttc_pct={format_number(fold_summary['e_ttc_pct'], digits=3)}",
            flush=True,
        )
        completed_folds.append(test_sequence_id)
        current_fold_status = {
            **current_fold_status,
            "status": "completed",
            "mae_s": float(fold_summary["mae_s"]),
            "e_ttc_pct": float(fold_summary["e_ttc_pct"]),
            "elapsed_time_s": round(cost_time_s, 3),
        }
        write_run_status(
            run_dir,
            started_at=started_at,
            dataset_root=args.dataset_root,
            variant_label=variant_label,
            sequences=list(sequences),
            completed_folds=completed_folds,
            failed_folds=failed_folds,
            state="running",
            current_fold=current_fold_status,
        )

    if not combined_rows:
        write_run_status(
            run_dir,
            started_at=started_at,
            dataset_root=args.dataset_root,
            variant_label=variant_label,
            sequences=list(sequences),
            completed_folds=completed_folds,
            failed_folds=failed_folds,
            state="failed",
        )
        raise SystemExit("No fold rows available; cannot write overall summary.")

    combined_summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "method_family": "MlBaseline",
        "variant": run_ts,
        "variant_label": variant_label,
        "dataset_root": str(args.dataset_root.resolve()),
        "config": {
            "model": args.model,
            "feature_mode": args.feature_mode,
            "pool_size": args.pool_size,
            "ridge_alpha": args.ridge_alpha,
            "target": args.target,
            "seed": args.seed,
            "mlp_hidden_dims": list(args.mlp_hidden_dims),
            "mlp_learning_rate": args.mlp_learning_rate,
            "mlp_weight_decay": args.mlp_weight_decay,
            "mlp_batch_size": args.mlp_batch_size,
            "mlp_epochs": args.mlp_epochs,
            "mlp_patience": args.mlp_patience,
            "mlp_val_ratio": args.mlp_val_ratio,
            "cnn_channels": list(args.cnn_channels),
            "cnn_head_hidden_dim": args.cnn_head_hidden_dim,
            "cnn_extra_hidden_dim": args.cnn_extra_hidden_dim,
            "cnn_learning_rate": args.cnn_learning_rate,
            "cnn_weight_decay": args.cnn_weight_decay,
            "cnn_batch_size": args.cnn_batch_size,
            "cnn_epochs": args.cnn_epochs,
            "cnn_patience": args.cnn_patience,
            "cnn_val_ratio": args.cnn_val_ratio,
            "sequences": sequences,
        },
        "sequence_summaries": fold_summaries,
        **summarize_rows(combined_rows),
    }

    write_csv(run_dir / "IntervalMetrics.csv", combined_rows, fieldnames=INTERVAL_FIELDNAMES)
    (run_dir / "Config.json").write_text(
        json.dumps(
            {
                "started_at": started_at,
                "dataset_root": str(args.dataset_root.resolve()),
                "variant_label": variant_label,
                "config": combined_summary["config"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "Summary.json").write_text(
        json.dumps(combined_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_run_status(
        run_dir,
        started_at=started_at,
        dataset_root=args.dataset_root,
        variant_label=variant_label,
        sequences=list(sequences),
        completed_folds=completed_folds,
        failed_folds=failed_folds,
        state="complete",
    )

    summary_md = [
        f"# MlBaseline / {variant_label}",
        "",
        f"- started_at: `{combined_summary['started_at']}`",
        f"- finished_at: `{combined_summary['finished_at']}`",
        f"- dataset_root: `{combined_summary['dataset_root']}`",
        f"- model: `{args.model}`",
        f"- feature_mode: `{args.feature_mode}`",
        f"- pool_size: `{args.pool_size}`",
        f"- ridge_alpha: `{args.ridge_alpha}`",
        f"- target: `{args.target}`",
        f"- gt_valid_count: `{combined_summary['gt_valid_count']}`",
        f"- est_valid_count: `{combined_summary['est_valid_count']}`",
        f"- failure_count: `{combined_summary['failure_count']}`",
        f"- mae_s: `{format_number(combined_summary['mae_s'])}`",
        f"- e_ttc_pct: `{format_number(combined_summary['e_ttc_pct'], digits=3)}`",
    ]
    if args.model == "mlp":
        summary_md.extend(
            [
                f"- mlp_hidden_dims: `{list(args.mlp_hidden_dims)}`",
                f"- mlp_learning_rate: `{args.mlp_learning_rate}`",
                f"- mlp_weight_decay: `{args.mlp_weight_decay}`",
                f"- mlp_batch_size: `{args.mlp_batch_size}`",
                f"- mlp_epochs: `{args.mlp_epochs}`",
                f"- mlp_patience: `{args.mlp_patience}`",
                f"- mlp_val_ratio: `{args.mlp_val_ratio}`",
            ]
        )
    if args.model == "cnn":
        summary_md.extend(
            [
                f"- cnn_channels: `{list(args.cnn_channels)}`",
                f"- cnn_head_hidden_dim: `{args.cnn_head_hidden_dim}`",
                f"- cnn_extra_hidden_dim: `{args.cnn_extra_hidden_dim}`",
                f"- cnn_learning_rate: `{args.cnn_learning_rate}`",
                f"- cnn_weight_decay: `{args.cnn_weight_decay}`",
                f"- cnn_batch_size: `{args.cnn_batch_size}`",
                f"- cnn_epochs: `{args.cnn_epochs}`",
                f"- cnn_patience: `{args.cnn_patience}`",
                f"- cnn_val_ratio: `{args.cnn_val_ratio}`",
            ]
        )
    summary_md.extend(["", "## Sequence Summaries", ""])
    for item in fold_summaries:
        summary_md.extend(
            [
                f"### {item['sequence_id']}",
                f"- train_sequences: `{', '.join(item['train_sequences'])}`",
                f"- train_count: `{item['train_count']}`",
                f"- test_count: `{item['test_count']}`",
                f"- mae_s: `{format_number(item['mae_s'])}`",
                f"- e_ttc_pct: `{format_number(item['e_ttc_pct'], digits=3)}`",
                "",
            ]
        )
    (run_dir / "Summary.md").write_text("\n".join(summary_md), encoding="utf-8")
    print(
        f"[Done] model={args.model} overall mae={format_number(combined_summary['mae_s'])} "
        f"e_ttc_pct={format_number(combined_summary['e_ttc_pct'], digits=3)} "
        f"output={run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
