#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CODE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_ml_baseline_train as np_train  # noqa: E402


TRAIN_HISTORY_FIELDNAMES = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_pred_mean",
    "val_pred_mean",
    "elapsed_time_s",
]


def parse_channels(value: str) -> tuple[int, int]:
    channels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(channels) != 2 or any(channel <= 0 for channel in channels):
        raise argparse.ArgumentTypeError(f"Invalid cnn channels: {value!r}")
    return int(channels[0]), int(channels[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train grouped EvTTC tiny CNN baselines with PyTorch/MPS acceleration.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=CODE_ROOT / "Derived" / "MlBaselineV1" / "tau10_20_40_64x64_g3x3_polsplit_all",
        help="Directory containing per-sequence exported dataset.npz files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_ROOT / "Experiments" / "MlBaselineTorch",
        help="Root directory for PyTorch ML baseline experiment outputs.",
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
        choices=["image_only", "image_stats_metadata", "image_bbox_aux", "image_stats_metadata_bbox_aux"],
        default="image_only",
        help="Torch CNN feature construction mode.",
    )
    parser.add_argument(
        "--aux-metrics-csv",
        type=Path,
        default=None,
        help="Optional per-sample auxiliary metrics CSV, e.g. BBoxLoomingV7 IntervalMetrics.csv.",
    )
    parser.add_argument(
        "--target",
        choices=["log_ttc", "ttc"],
        default="log_ttc",
        help="Regression target.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["direct", "residual_to_aux"],
        default="direct",
        help=(
            "direct predicts the target directly; residual_to_aux predicts target - aux bbox log(TTC) "
            "and adds the aux prediction back at inference time."
        ),
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Torch device. auto prefers MPS when available.",
    )
    parser.add_argument(
        "--cnn-channels",
        type=parse_channels,
        default=(8, 12),
        help="Conv channel counts for the two-layer tiny CNN, e.g. 8,12.",
    )
    parser.add_argument(
        "--cnn-head-hidden-dim",
        type=int,
        default=32,
        help="Hidden dim of the CNN regression head.",
    )
    parser.add_argument(
        "--cnn-learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--cnn-weight-decay",
        type=float,
        default=3e-4,
        help="L2 weight decay for Adam.",
    )
    parser.add_argument(
        "--cnn-batch-size",
        type=int,
        default=64,
        help="Mini-batch size. Larger than the NumPy baseline by default to use MPS better.",
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
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--cnn-val-ratio",
        type=float,
        default=0.2,
        help="Per-sequence validation ratio carved from training data.",
    )
    parser.add_argument(
        "--status-interval-s",
        type=float,
        default=10.0,
        help="Minimum seconds between batch-level RunStatus.json updates.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.cnn_head_hidden_dim <= 0:
        raise SystemExit("--cnn-head-hidden-dim must be positive")
    if args.cnn_batch_size <= 0:
        raise SystemExit("--cnn-batch-size must be positive")
    if args.cnn_epochs <= 0:
        raise SystemExit("--cnn-epochs must be positive")
    if args.cnn_patience <= 0:
        raise SystemExit("--cnn-patience must be positive")
    if not 0.0 < args.cnn_val_ratio < 1.0:
        raise SystemExit("--cnn-val-ratio must be between 0 and 1")
    if "bbox_aux" in args.feature_mode and args.aux_metrics_csv is None:
        raise SystemExit(f"--feature-mode {args.feature_mode!r} requires --aux-metrics-csv")
    if args.prediction_mode == "residual_to_aux":
        if args.target != "log_ttc":
            raise SystemExit("--prediction-mode residual_to_aux currently requires --target log_ttc")
        if "bbox_aux" not in args.feature_mode:
            raise SystemExit("--prediction-mode residual_to_aux requires a bbox_aux feature mode")
        if args.aux_metrics_csv is None:
            raise SystemExit("--prediction-mode residual_to_aux requires --aux-metrics-csv")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Requested --device mps, but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def variant_label_for(feature_mode: str, prediction_mode: str) -> str:
    if prediction_mode == "residual_to_aux":
        if feature_mode == "image_bbox_aux":
            return "TorchTinyCnnBBoxResidualV1"
        if feature_mode == "image_stats_metadata_bbox_aux":
            return "TorchTinyCnnStatsMetaBBoxResidualV1"
    if feature_mode == "image_only":
        return "TorchTinyCnnImageOnlyV1"
    if feature_mode == "image_stats_metadata":
        return "TorchTinyCnnStatsMetaV1"
    if feature_mode == "image_bbox_aux":
        return "TorchTinyCnnBBoxAuxV1"
    if feature_mode == "image_stats_metadata_bbox_aux":
        return "TorchTinyCnnStatsMetaBBoxAuxV1"
    raise ValueError(f"Unsupported feature_mode={feature_mode!r}")


def load_aux_metrics(aux_metrics_csv: Path) -> dict[str, dict[str, str]]:
    with aux_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"Aux metrics CSV missing sample_id column: {aux_metrics_csv}")
        return {str(row["sample_id"]): row for row in reader}


def build_bbox_aux_arrays(
    dataset: dict[str, np.ndarray],
    aux_index: dict[str, dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    sample_ids = [str(item) for item in dataset["sample_id"].tolist()]
    features = np.zeros((len(sample_ids), 3), dtype=np.float32)
    base_log_ttc = np.zeros((len(sample_ids),), dtype=np.float64)
    for idx, sample_id in enumerate(sample_ids):
        row = aux_index.get(sample_id)
        if row is None:
            continue
        status = row.get("status", "")
        ttc_text = row.get("ttc_est_s", "")
        confidence_text = row.get("confidence", "")
        try:
            ttc_est_s = float(ttc_text) if ttc_text else float("nan")
        except ValueError:
            ttc_est_s = float("nan")
        try:
            confidence = float(confidence_text) if confidence_text else 0.0
        except ValueError:
            confidence = 0.0
        valid = bool(status.startswith("OK") and np.isfinite(ttc_est_s) and ttc_est_s > 0.0)
        log_ttc = float(np.log(max(ttc_est_s, 1e-6))) if valid else 0.0
        features[idx, 0] = log_ttc
        features[idx, 1] = 1.0 if valid else 0.0
        features[idx, 2] = float(confidence) if valid else 0.0
        base_log_ttc[idx] = log_ttc
    return features, base_log_ttc


def build_bbox_aux_features(dataset: dict[str, np.ndarray], aux_index: dict[str, dict[str, str]]) -> np.ndarray:
    return build_bbox_aux_arrays(dataset, aux_index)[0]


class TorchTinyCnnRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        extra_dim: int,
        conv_channels: tuple[int, int],
        head_hidden_dim: int,
    ) -> None:
        super().__init__()
        conv1_out, conv2_out = conv_channels
        self.extra_dim = int(extra_dim)
        self.conv1 = nn.Conv2d(input_channels, conv1_out, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.head1 = nn.Linear(conv2_out + self.extra_dim, head_hidden_dim)
        self.head2 = nn.Linear(head_hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        x = F.avg_pool2d(image, kernel_size=2)
        x = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = x.mean(dim=(2, 3))
        if self.extra_dim > 0:
            if extra is None:
                raise ValueError("extra features are required for this model")
            x = torch.cat([x, extra], dim=1)
        x = F.relu(self.head1(x))
        return self.head2(x).squeeze(1)


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def maybe_sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def predict_in_batches(
    model: nn.Module,
    image: torch.Tensor,
    extra: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    preds: list[torch.Tensor] = []
    for start in range(0, image.shape[0], batch_size):
        stop = min(start + batch_size, image.shape[0])
        preds.append(model(image[start:stop], extra[start:stop]))
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate_loss_and_mean(
    model: nn.Module,
    image: torch.Tensor,
    extra: torch.Tensor,
    target: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, float]:
    pred = predict_in_batches(model, image, extra, batch_size=batch_size)
    loss = F.mse_loss(pred, target).item()
    return float(loss), float(pred.mean().item())


def copy_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_cpu_state(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({name: value.to(device) for name, value in state.items()})


def train_torch_cnn(
    train_image: np.ndarray,
    train_extra: np.ndarray,
    train_y: np.ndarray,
    val_image: np.ndarray,
    val_extra: np.ndarray,
    val_y: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    fold_seed: int,
    progress_callback,
) -> tuple[nn.Module, dict[str, object], list[dict[str, float | int]]]:
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    train_image_t = to_tensor(train_image, device)
    train_extra_t = to_tensor(train_extra, device)
    train_y_t = to_tensor(train_y, device)
    val_image_t = to_tensor(val_image, device)
    val_extra_t = to_tensor(val_extra, device)
    val_y_t = to_tensor(val_y, device)

    model = TorchTinyCnnRegressor(
        input_channels=train_image.shape[1],
        extra_dim=train_extra.shape[1],
        conv_channels=args.cnn_channels,
        head_hidden_dim=args.cnn_head_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.cnn_learning_rate),
        weight_decay=float(args.cnn_weight_decay),
    )

    best_state = copy_state_to_cpu(model)
    best_epoch = 0
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    stale_epochs = 0
    global_step = 0
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()
    last_progress_at = 0.0
    batch_count = int(np.ceil(train_image.shape[0] / float(args.cnn_batch_size)))

    for epoch in range(1, args.cnn_epochs + 1):
        model.train()
        permutation = torch.randperm(train_image_t.shape[0], device=device)

        for batch_idx, start in enumerate(range(0, train_image_t.shape[0], args.cnn_batch_size), start=1):
            stop = min(start + args.cnn_batch_size, train_image_t.shape[0])
            batch_indices = permutation[start:stop]
            optimizer.zero_grad(set_to_none=True)
            pred = model(train_image_t[batch_indices], train_extra_t[batch_indices])
            loss = F.mse_loss(pred, train_y_t[batch_indices])
            loss.backward()
            optimizer.step()
            global_step += 1

            now = time.perf_counter()
            if now - last_progress_at >= args.status_interval_s or batch_idx == batch_count:
                maybe_sync(device)
                progress_callback(
                    {
                        "phase": "batch",
                        "epoch": int(epoch),
                        "epochs": int(args.cnn_epochs),
                        "batch": int(batch_idx),
                        "batch_count": int(batch_count),
                        "global_step": int(global_step),
                        "device": str(device),
                    }
                )
                last_progress_at = now

        maybe_sync(device)
        train_loss, train_pred_mean = evaluate_loss_and_mean(
            model,
            train_image_t,
            train_extra_t,
            train_y_t,
            batch_size=args.cnn_batch_size,
        )
        val_loss, val_pred_mean = evaluate_loss_and_mean(
            model,
            val_image_t,
            val_extra_t,
            val_y_t,
            batch_size=args.cnn_batch_size,
        )
        elapsed_time_s = time.perf_counter() - started_at
        history_row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_pred_mean": float(train_pred_mean),
            "val_pred_mean": float(val_pred_mean),
            "elapsed_time_s": float(elapsed_time_s),
        }
        history.append(history_row)

        improved = val_loss + 1e-9 < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = copy_state_to_cpu(model)
            stale_epochs = 0
        else:
            stale_epochs += 1

        progress_callback(
            {
                "phase": "epoch",
                **history_row,
                "epochs": int(args.cnn_epochs),
                "best_epoch": int(best_epoch),
                "best_train_loss": float(best_train_loss),
                "best_val_loss": float(best_val_loss),
                "stale_epochs": int(stale_epochs),
                "patience": int(args.cnn_patience),
                "device": str(device),
            }
        )
        if not improved and stale_epochs >= args.cnn_patience:
            break

    load_cpu_state(model, best_state, device)
    summary = {
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "trained_epochs": int(len(history)),
        "device": str(device),
    }
    return model, summary, history


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    sequences = args.sequences or np_train.list_sequence_dirs(args.dataset_root)
    if len(sequences) < 2:
        raise SystemExit(f"Need at least 2 sequences under {args.dataset_root}; got {sequences}")

    loaded = {sequence_id: np_train.load_sequence_dataset(args.dataset_root, sequence_id) for sequence_id in sequences}
    aux_index = load_aux_metrics(args.aux_metrics_csv) if args.aux_metrics_csv is not None else None
    variant_label = variant_label_for(args.feature_mode, args.prediction_mode)
    run_ts = args.run_name or datetime.now().strftime(f"{variant_label}_%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    combined_rows: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    completed_folds: list[str] = []
    failed_folds: list[dict[str, str]] = []
    current_fold_status: dict[str, object] | None = None

    def write_status(state: str, current_fold: dict[str, object] | None = None, error: str | None = None) -> None:
        np_train.write_run_status(
            run_dir,
            started_at=started_at,
            dataset_root=args.dataset_root,
            variant_label=variant_label,
            sequences=list(sequences),
            completed_folds=completed_folds,
            failed_folds=failed_folds,
            state=state,
            current_fold=current_fold,
            error=error,
        )

    def mark_stopped(state: str, error_type: str, message: str) -> None:
        fold_status = dict(current_fold_status or {})
        if fold_status:
            fold_status["status"] = state
        failed_folds.append(
            {
                "sequence_id": str(fold_status.get("sequence_id", "unknown")),
                "error_type": error_type,
                "message": message,
            }
        )
        write_status(state, current_fold=fold_status or None, error=f"{error_type}: {message}")

    write_status("running")
    print(f"[Torch] device={device} run_dir={run_dir}", flush=True)

    for fold_idx, test_sequence_id in enumerate(sequences):
        fold_dir = run_dir / test_sequence_id
        current_fold_status = {
            "sequence_id": test_sequence_id,
            "fold_index": int(fold_idx + 1),
            "fold_count": int(len(sequences)),
            "status": "checking",
            "device": str(device),
        }
        write_status("running", current_fold=current_fold_status)

        if not args.overwrite_existing_folds and np_train.is_complete_fold_dir(fold_dir):
            fold_summary = np_train.read_json(fold_dir / "Summary.json")
            rows = np_train.read_csv_rows(fold_dir / "IntervalMetrics.csv")
            fold_summaries.append(fold_summary)
            combined_rows.extend(rows)
            completed_folds.append(test_sequence_id)
            current_fold_status = {
                **current_fold_status,
                "status": "skipped_complete",
                "e_ttc_pct": float(fold_summary["e_ttc_pct"]),
            }
            write_status("running", current_fold=current_fold_status)
            print(
                f"[SkipFold] model=torch-cnn test={test_sequence_id} "
                f"e_ttc_pct={np_train.format_number(float(fold_summary['e_ttc_pct']), digits=3)}",
                flush=True,
            )
            continue

        train_sequence_ids = tuple(seq for seq in sequences if seq != test_sequence_id)
        train_parts = [loaded[seq] for seq in train_sequence_ids]
        test_data = loaded[test_sequence_id]
        train_target = np.concatenate([np_train.select_target(item, args.target) for item in train_parts], axis=0)
        train_sequence_array = np.concatenate([item["sequence_id"] for item in train_parts], axis=0)

        train_image = np.concatenate([item["image_nchw"].astype(np.float32) for item in train_parts], axis=0)
        test_image = test_data["image_nchw"].astype(np.float32)
        train_image_x, test_image_x, _, _ = np_train.standardize_image_features(train_image, test_image)
        train_aux_base_target = np.zeros_like(train_target, dtype=np.float64)
        test_aux_base_target = np.zeros((test_data["gt_ttc_s"].shape[0],), dtype=np.float64)

        train_extra_parts: list[np.ndarray] = []
        test_extra_parts: list[np.ndarray] = []
        if "stats_metadata" in args.feature_mode:
            train_extra_parts.append(np.concatenate([np_train.build_stats_metadata(item) for item in train_parts], axis=0))
            test_extra_parts.append(np_train.build_stats_metadata(test_data))
        if "bbox_aux" in args.feature_mode:
            if aux_index is None:
                raise ValueError("bbox_aux feature mode requires an aux metrics index")
            train_bbox_aux_arrays = [build_bbox_aux_arrays(item, aux_index) for item in train_parts]
            test_bbox_aux_features, test_aux_base_target = build_bbox_aux_arrays(test_data, aux_index)
            train_extra_parts.append(
                np.concatenate([item[0] for item in train_bbox_aux_arrays], axis=0)
            )
            train_aux_base_target = np.concatenate([item[1] for item in train_bbox_aux_arrays], axis=0)
            test_extra_parts.append(test_bbox_aux_features)
        if train_extra_parts:
            train_extra = np.concatenate(train_extra_parts, axis=1).astype(np.float32)
            test_extra = np.concatenate(test_extra_parts, axis=1).astype(np.float32)
            train_extra_x, test_extra_x, _, _ = np_train.standardize_vector_features(train_extra, test_extra)
        else:
            train_extra_x = np.zeros((train_image_x.shape[0], 0), dtype=np.float32)
            test_extra_x = np.zeros((test_image_x.shape[0], 0), dtype=np.float32)

        train_idx, val_idx = np_train.split_train_val_indices(
            train_sequence_array,
            val_ratio=args.cnn_val_ratio,
            seed=args.seed + fold_idx,
        )
        if val_idx.size == 0:
            raise ValueError("Torch CNN requires a non-empty validation split.")
        train_model_target = train_target
        if args.prediction_mode == "residual_to_aux":
            train_model_target = train_target - train_aux_base_target

        core_train_image = train_image_x[train_idx].astype(np.float32)
        core_train_extra = train_extra_x[train_idx].astype(np.float32)
        core_train_y = train_model_target[train_idx]
        val_image = train_image_x[val_idx].astype(np.float32)
        val_extra = train_extra_x[val_idx].astype(np.float32)
        val_y = train_model_target[val_idx]
        core_train_y_norm, val_y_norm, target_mean, target_std = np_train.standardize_target(core_train_y, val_y)

        current_fold_status = {
            **current_fold_status,
            "status": "preparing",
            "train_sequence_count": int(len(train_sequence_ids)),
            "train_count": int(train_target.shape[0]),
            "train_count_after_val_split": int(core_train_image.shape[0]),
            "val_count": int(val_image.shape[0]),
            "test_count": int(test_data["gt_ttc_s"].shape[0]),
        }
        write_status("running", current_fold=current_fold_status)
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
            write_status("running", current_fold=current_fold_status)
            if progress.get("phase") == "batch":
                print(
                    f"[Progress] torch fold={fold_idx + 1}/{len(sequences)} test={test_sequence_id} "
                    f"epoch={progress.get('epoch')}/{progress.get('epochs')} "
                    f"batch={progress.get('batch')}/{progress.get('batch_count')} "
                    f"elapsed_s={elapsed_time_s:.1f}",
                    flush=True,
                )
            elif progress.get("phase") == "epoch":
                print(
                    f"[Epoch] torch fold={fold_idx + 1}/{len(sequences)} test={test_sequence_id} "
                    f"epoch={progress.get('epoch')}/{progress.get('epochs')} "
                    f"train_loss={np_train.format_number(float(progress['train_loss']))} "
                    f"val_loss={np_train.format_number(float(progress['val_loss']))} "
                    f"best_epoch={progress.get('best_epoch')} "
                    f"elapsed_s={elapsed_time_s:.1f}",
                    flush=True,
                )

        try:
            model, training_summary, history_rows = train_torch_cnn(
                core_train_image,
                core_train_extra,
                core_train_y_norm.astype(np.float32),
                val_image,
                val_extra,
                val_y_norm.astype(np.float32),
                args=args,
                device=device,
                fold_seed=args.seed + fold_idx,
                progress_callback=report_training_progress,
            )
        except KeyboardInterrupt:
            mark_stopped("interrupted", "KeyboardInterrupt", "training interrupted")
            raise SystemExit(130)
        except Exception as exc:
            mark_stopped("failed", type(exc).__name__, str(exc))
            raise

        test_image_t = to_tensor(test_image_x.astype(np.float32), device)
        test_extra_t = to_tensor(test_extra_x.astype(np.float32), device)
        maybe_sync(device)
        pred_norm_t = predict_in_batches(model, test_image_t, test_extra_t, batch_size=args.cnn_batch_size)
        maybe_sync(device)
        pred_model_target = pred_norm_t.detach().cpu().numpy().astype(np.float64) * target_std + target_mean
        pred_target = pred_model_target
        if args.prediction_mode == "residual_to_aux":
            pred_target = test_aux_base_target + pred_model_target
        cost_time_s = time.perf_counter() - started_fold

        pred_ttc_s = np_train.invert_target(pred_target, args.target)
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
            "method_family": "MlBaselineTorch",
            "variant": variant_label,
            "model": "torch_cnn",
            "feature_mode": args.feature_mode,
            "target": args.target,
            "prediction_mode": args.prediction_mode,
            "train_sequences": list(train_sequence_ids),
            "train_count": int(train_target.shape[0]),
            "test_count": int(len(gt_ttc_s)),
            "cnn_channels": list(args.cnn_channels),
            "cnn_head_hidden_dim": int(args.cnn_head_hidden_dim),
            "cnn_learning_rate": float(args.cnn_learning_rate),
            "cnn_weight_decay": float(args.cnn_weight_decay),
            "cnn_batch_size": int(args.cnn_batch_size),
            "cnn_epochs": int(args.cnn_epochs),
            "cnn_patience": int(args.cnn_patience),
            "cnn_val_ratio": float(args.cnn_val_ratio),
            "seed": int(args.seed + fold_idx),
            "train_count_after_val_split": int(core_train_image.shape[0]),
            "val_count": int(val_image.shape[0]),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            **training_summary,
            **np_train.summarize_rows(rows),
        }
        fold_summaries.append(fold_summary)
        combined_rows.extend(rows)

        np_train.write_csv(fold_dir / "IntervalMetrics.csv", rows, fieldnames=np_train.INTERVAL_FIELDNAMES)
        np_train.write_csv(fold_dir / "TrainHistory.csv", history_rows, fieldnames=TRAIN_HISTORY_FIELDNAMES)
        (fold_dir / "Summary.json").write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        torch.save(
            {
                "model_state_dict": copy_state_to_cpu(model),
                "fold_summary": fold_summary,
            },
            fold_dir / "Model.pt",
        )
        write_markdown(
            fold_dir / "Summary.md",
            [
                f"# {test_sequence_id} / {variant_label}",
                "",
                "- model: `torch_cnn`",
                f"- device: `{device}`",
                f"- train_sequences: `{', '.join(train_sequence_ids)}`",
                f"- train_count: `{fold_summary['train_count']}`",
                f"- test_count: `{fold_summary['test_count']}`",
                f"- mae_s: `{np_train.format_number(fold_summary['mae_s'])}`",
                f"- e_ttc_pct: `{np_train.format_number(fold_summary['e_ttc_pct'], digits=3)}`",
                f"- trained_epochs: `{fold_summary['trained_epochs']}`",
                f"- best_epoch: `{fold_summary['best_epoch']}`",
                f"- best_val_loss: `{np_train.format_number(float(fold_summary['best_val_loss']))}`",
            ],
        )
        print(
            f"[Fold] torch test={test_sequence_id} train={','.join(train_sequence_ids)} "
            f"mae={np_train.format_number(fold_summary['mae_s'])} "
            f"e_ttc_pct={np_train.format_number(fold_summary['e_ttc_pct'], digits=3)} "
            f"elapsed_s={cost_time_s:.1f}",
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
        write_status("running", current_fold=current_fold_status)

        if device.type == "mps":
            torch.mps.empty_cache()

    if not combined_rows:
        write_status("failed", error="No fold rows available; cannot write overall summary.")
        raise SystemExit("No fold rows available; cannot write overall summary.")

    combined_summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "method_family": "MlBaselineTorch",
        "variant": run_ts,
        "variant_label": variant_label,
        "dataset_root": str(args.dataset_root.resolve()),
        "config": {
            "model": "torch_cnn",
            "feature_mode": args.feature_mode,
            "target": args.target,
            "prediction_mode": args.prediction_mode,
            "seed": args.seed,
            "device": str(device),
            "aux_metrics_csv": str(args.aux_metrics_csv.resolve()) if args.aux_metrics_csv else None,
            "cnn_channels": list(args.cnn_channels),
            "cnn_head_hidden_dim": args.cnn_head_hidden_dim,
            "cnn_learning_rate": args.cnn_learning_rate,
            "cnn_weight_decay": args.cnn_weight_decay,
            "cnn_batch_size": args.cnn_batch_size,
            "cnn_epochs": args.cnn_epochs,
            "cnn_patience": args.cnn_patience,
            "cnn_val_ratio": args.cnn_val_ratio,
            "sequences": sequences,
        },
        "sequence_summaries": fold_summaries,
        **np_train.summarize_rows(combined_rows),
    }

    np_train.write_csv(run_dir / "IntervalMetrics.csv", combined_rows, fieldnames=np_train.INTERVAL_FIELDNAMES)
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
    write_status("complete")

    summary_md = [
        f"# MlBaselineTorch / {variant_label}",
        "",
        f"- started_at: `{combined_summary['started_at']}`",
        f"- finished_at: `{combined_summary['finished_at']}`",
        f"- dataset_root: `{combined_summary['dataset_root']}`",
        f"- device: `{device}`",
        f"- model: `torch_cnn`",
        f"- feature_mode: `{args.feature_mode}`",
        f"- target: `{args.target}`",
        f"- prediction_mode: `{args.prediction_mode}`",
        f"- gt_valid_count: `{combined_summary['gt_valid_count']}`",
        f"- est_valid_count: `{combined_summary['est_valid_count']}`",
        f"- failure_count: `{combined_summary['failure_count']}`",
        f"- mae_s: `{np_train.format_number(combined_summary['mae_s'])}`",
        f"- e_ttc_pct: `{np_train.format_number(combined_summary['e_ttc_pct'], digits=3)}`",
        f"- cnn_channels: `{list(args.cnn_channels)}`",
        f"- cnn_head_hidden_dim: `{args.cnn_head_hidden_dim}`",
        f"- cnn_batch_size: `{args.cnn_batch_size}`",
        "",
        "## Sequence Summaries",
        "",
    ]
    for item in fold_summaries:
        summary_md.extend(
            [
                f"### {item['sequence_id']}",
                f"- train_sequences: `{', '.join(item['train_sequences'])}`",
                f"- train_count: `{item['train_count']}`",
                f"- test_count: `{item['test_count']}`",
                f"- mae_s: `{np_train.format_number(item['mae_s'])}`",
                f"- e_ttc_pct: `{np_train.format_number(item['e_ttc_pct'], digits=3)}`",
                "",
            ]
        )
    write_markdown(run_dir / "Summary.md", summary_md)
    print(
        f"[Done] torch overall mae={np_train.format_number(combined_summary['mae_s'])} "
        f"e_ttc_pct={np_train.format_number(combined_summary['e_ttc_pct'], digits=3)} "
        f"output={run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
