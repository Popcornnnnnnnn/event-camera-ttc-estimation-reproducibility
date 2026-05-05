#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[2]

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


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(f"Invalid hidden dims: {value!r}")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc calibrate frozen TinyCnnImageOnly predictions with stats/meta.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=CODE_ROOT / "Derived" / "MlBaselineV1" / "tau20_64x64_g3x3",
        help="Directory containing per-sequence exported dataset.npz files.",
    )
    parser.add_argument(
        "--base-run",
        type=Path,
        required=True,
        help="TinyCnnImageOnly experiment directory used as frozen base predictor.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_ROOT / "Experiments" / "MlBaseline",
        help="Root directory for post-hoc calibration outputs.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence IDs to include. Default: all sequences under dataset root.",
    )
    parser.add_argument(
        "--target",
        choices=["log_ttc"],
        default="log_ttc",
        help="Target to calibrate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for calibration head.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=(24,),
        help="Comma-separated hidden dims for the calibration MLP.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate for calibration head.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for calibration head.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for calibration head.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Max epochs for calibration head.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience for calibration head.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Per-sequence validation ratio from training folds.",
    )
    return parser.parse_args()


def list_sequence_dirs(dataset_root: Path) -> list[str]:
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir() and (path / "dataset.npz").exists())


def load_sequence_dataset(dataset_root: Path, sequence_id: str) -> dict[str, np.ndarray]:
    with np.load(dataset_root / sequence_id / "dataset.npz") as data:
        return {key: data[key] for key in data.files}


def load_base_predictions(base_run: Path, sequence_id: str) -> dict[str, dict[str, float | int | str]]:
    path = base_run / sequence_id / "IntervalMetrics.csv"
    rows: dict[str, dict[str, float | int | str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[str(row["sample_id"])] = row
    return rows


def build_stats_metadata(dataset: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([dataset["stats"].astype(np.float32), dataset["metadata"].astype(np.float32)], axis=1)


def standardize_features(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (other_x - mean) / std


def standardize_target(train_y: np.ndarray, other_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean())
    std = float(train_y.std())
    if std < 1e-6:
        std = 1.0
    return (train_y - mean) / std, (other_y - mean) / std, mean, std


def split_train_val_indices(train_sequence_ids: np.ndarray, *, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
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
    return np.concatenate(train_indices), np.concatenate(val_indices) if val_indices else np.empty((0,), dtype=np.int64)


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


def forward_mlp(x: np.ndarray, layers: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    activations = [x.astype(np.float64)]
    preactivations: list[np.ndarray] = []
    current = activations[0]
    for layer_idx, layer in enumerate(layers):
        z = current @ layer["weight"] + layer["bias"]
        preactivations.append(z)
        current = z if layer_idx == len(layers) - 1 else np.maximum(z, 0.0)
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


def train_calibrator(
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
) -> tuple[list[dict[str, np.ndarray]], dict[str, float | int], list[dict[str, float | int]]]:
    layers = init_mlp_layers(train_x.shape[1], hidden_dims, seed)
    best_layers = copy_layers(layers)
    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    history: list[dict[str, float | int]] = []

    adam_m = [{"weight": np.zeros_like(layer["weight"]), "bias": np.zeros_like(layer["bias"])} for layer in layers]
    adam_v = [{"weight": np.zeros_like(layer["weight"]), "bias": np.zeros_like(layer["bias"])} for layer in layers]
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    global_step = 0
    stale_epochs = 0
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(train_x.shape[0])
        shuffled_x = train_x[permutation]
        shuffled_y = train_y[permutation]

        for start in range(0, train_x.shape[0], batch_size):
            stop = min(start + batch_size, train_x.shape[0])
            batch_x = shuffled_x[start:stop]
            batch_y = shuffled_y[start:stop]
            pred_batch, activations, preactivations = forward_mlp(batch_x, layers)
            grads = backward_mlp(activations, preactivations, layers, pred_batch, batch_y, weight_decay=weight_decay)
            global_step += 1
            for layer_idx, layer in enumerate(layers):
                for name in ("weight", "bias"):
                    adam_m[layer_idx][name] = beta1 * adam_m[layer_idx][name] + (1.0 - beta1) * grads[layer_idx][name]
                    adam_v[layer_idx][name] = beta2 * adam_v[layer_idx][name] + (1.0 - beta2) * (grads[layer_idx][name] ** 2)
                    m_hat = adam_m[layer_idx][name] / (1.0 - beta1**global_step)
                    v_hat = adam_v[layer_idx][name] / (1.0 - beta2**global_step)
                    layer[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        train_pred, train_loss = evaluate_mlp(train_x, train_y, layers)
        val_pred, val_loss = evaluate_mlp(val_x, val_y, layers)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_pred_mean": float(train_pred.mean()),
                "val_pred_mean": float(val_pred.mean()),
            }
        )
        if val_loss + 1e-9 < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_layers = copy_layers(layers)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return best_layers, {
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "trained_epochs": int(len(history)),
    }, history


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


def format_number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    sequences = args.sequences or list_sequence_dirs(args.dataset_root)
    loaded = {sequence_id: load_sequence_dataset(args.dataset_root, sequence_id) for sequence_id in sequences}
    variant_label = "TinyCnnPostHocCalibV1"
    run_ts = datetime.now().strftime(f"{variant_label}_%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    combined_rows: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []

    for fold_idx, test_sequence_id in enumerate(sequences):
        train_sequence_ids = tuple(seq for seq in sequences if seq != test_sequence_id)
        train_parts = [loaded[seq] for seq in train_sequence_ids]
        test_data = loaded[test_sequence_id]
        base_rows = load_base_predictions(args.base_run, test_sequence_id)

        train_stats = np.concatenate([build_stats_metadata(item) for item in train_parts], axis=0)
        train_base_pred = []
        train_target = []
        train_sequence_array = np.concatenate([item["sequence_id"] for item in train_parts], axis=0)
        for item in train_parts:
            train_base_fold_rows = load_base_predictions(args.base_run, str(item["sequence_id"][0]))
            for sample_id, gt_log_ttc in zip(item["sample_id"], item["gt_log_ttc_s"]):
                base_row = train_base_fold_rows[str(sample_id)]
                train_base_pred.append(float(base_row["target_pred"]))
                train_target.append(float(gt_log_ttc))

        train_base_pred = np.asarray(train_base_pred, dtype=np.float64).reshape(-1, 1)
        train_target = np.asarray(train_target, dtype=np.float64)
        train_x = np.concatenate([train_base_pred, train_stats.astype(np.float64)], axis=1)

        test_stats = build_stats_metadata(test_data).astype(np.float64)
        test_base_pred = np.asarray([float(base_rows[str(sample_id)]["target_pred"]) for sample_id in test_data["sample_id"]], dtype=np.float64).reshape(-1, 1)
        test_x = np.concatenate([test_base_pred, test_stats], axis=1)

        train_idx, val_idx = split_train_val_indices(train_sequence_array, val_ratio=args.val_ratio, seed=args.seed + fold_idx)
        if val_idx.size == 0:
            raise ValueError("Post-hoc calibrator requires a non-empty validation split.")
        core_train_x = train_x[train_idx]
        core_train_y = train_target[train_idx]
        val_x = train_x[val_idx]
        val_y = train_target[val_idx]

        feature_mean = core_train_x.mean(axis=0, keepdims=True)
        feature_std = core_train_x.std(axis=0, keepdims=True)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        core_train_x = (core_train_x - feature_mean) / feature_std
        val_x = (val_x - feature_mean) / feature_std
        test_x = (test_x - feature_mean) / feature_std
        core_train_y_norm, val_y_norm, target_mean, target_std = standardize_target(core_train_y, val_y)
        _, test_y_zero_norm, _, _ = standardize_target(core_train_y, np.zeros((test_x.shape[0],), dtype=np.float64))

        started_fold = time.perf_counter()
        layers, training_summary, history_rows = train_calibrator(
            core_train_x,
            core_train_y_norm,
            val_x,
            val_y_norm,
            hidden_dims=args.hidden_dims,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed + fold_idx,
        )
        pred_norm, _ = evaluate_mlp(test_x, test_y_zero_norm, layers)
        pred_target = pred_norm * target_std + target_mean
        cost_time_s = time.perf_counter() - started_fold

        pred_ttc_s = np.exp(pred_target)
        pred_ttc_s = np.clip(pred_ttc_s, 1e-6, None)
        gt_ttc_s = test_data["gt_ttc_s"].astype(np.float64)

        rows: list[dict[str, object]] = []
        for idx, gt in enumerate(gt_ttc_s):
            abs_err_s = float(abs(pred_ttc_s[idx] - gt))
            e_ttc_pct = float(abs_err_s / max(1e-6, float(gt)) * 100.0)
            rows.append(
                {
                    "sample_id": str(test_data["sample_id"][idx]),
                    "sequence_id": str(test_data["sequence_id"][idx]),
                    "interval_idx": int(test_data["interval_idx"][idx]),
                    "gt_ttc_s": float(gt),
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
            "model": "cnn_posthoc_calib",
            "feature_mode": "frozen_image_only_plus_stats_meta_calib",
            "target": args.target,
            "train_sequences": list(train_sequence_ids),
            "train_count": int(train_x.shape[0]),
            "test_count": int(test_x.shape[0]),
            "hidden_dims": list(args.hidden_dims),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "val_ratio": args.val_ratio,
            "seed": args.seed + fold_idx,
            "train_count_after_val_split": int(core_train_x.shape[0]),
            "val_count": int(val_x.shape[0]),
            **training_summary,
            **summarize_rows(rows),
        }
        fold_summaries.append(fold_summary)
        combined_rows.extend(rows)

        fold_dir = run_dir / test_sequence_id
        write_csv(fold_dir / "IntervalMetrics.csv", rows, INTERVAL_FIELDNAMES)
        write_csv(fold_dir / "TrainHistory.csv", history_rows, ["epoch", "train_loss", "val_loss", "train_pred_mean", "val_pred_mean"])
        (fold_dir / "Summary.json").write_text(json.dumps(fold_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    combined_summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "method_family": "MlBaseline",
        "variant": run_ts,
        "variant_label": variant_label,
        "dataset_root": str(args.dataset_root.resolve()),
        "base_run": str(args.base_run.resolve()),
        "config": {
            "hidden_dims": list(args.hidden_dims),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "val_ratio": args.val_ratio,
            "sequences": sequences,
        },
        "sequence_summaries": fold_summaries,
        **summarize_rows(combined_rows),
    }

    write_csv(run_dir / "IntervalMetrics.csv", combined_rows, INTERVAL_FIELDNAMES)
    (run_dir / "Config.json").write_text(json.dumps({"started_at": started_at, "dataset_root": str(args.dataset_root.resolve()), "base_run": str(args.base_run.resolve()), "variant_label": variant_label, "config": combined_summary["config"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "Summary.json").write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = [
        f"# MlBaseline / {variant_label}",
        "",
        f"- started_at: `{combined_summary['started_at']}`",
        f"- finished_at: `{combined_summary['finished_at']}`",
        f"- dataset_root: `{combined_summary['dataset_root']}`",
        f"- base_run: `{combined_summary['base_run']}`",
        f"- gt_valid_count: `{combined_summary['gt_valid_count']}`",
        f"- est_valid_count: `{combined_summary['est_valid_count']}`",
        f"- failure_count: `{combined_summary['failure_count']}`",
        f"- mae_s: `{format_number(combined_summary['mae_s'])}`",
        f"- e_ttc_pct: `{format_number(combined_summary['e_ttc_pct'], digits=3)}`",
        f"- hidden_dims: `{list(args.hidden_dims)}`",
        f"- learning_rate: `{args.learning_rate}`",
        f"- weight_decay: `{args.weight_decay}`",
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
                f"- mae_s: `{format_number(item['mae_s'])}`",
                f"- e_ttc_pct: `{format_number(item['e_ttc_pct'], digits=3)}`",
                "",
            ]
        )
    (run_dir / "Summary.md").write_text("\n".join(summary_md), encoding="utf-8")
    print(
        f"[Done] model=cnn_posthoc_calib overall mae={format_number(combined_summary['mae_s'])} "
        f"e_ttc_pct={format_number(combined_summary['e_ttc_pct'], digits=3)} output={run_dir}"
    )


if __name__ == "__main__":
    main()
