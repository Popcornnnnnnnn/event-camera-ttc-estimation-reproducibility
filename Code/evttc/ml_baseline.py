from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .native_adapter import NativeSampleAdapter
from .representations import LocalStatsV1, LtsV1, build_local_stats_v1, build_lts_v1


@dataclass(frozen=True)
class MlBaselineExportConfig:
    image_height: int = 64
    image_width: int = 64
    tau_ms: float = 20.0
    pad_px: int = 0
    grid_rows: int = 3
    grid_cols: int = 3
    image_variant: str = "lts_v1"
    multi_tau_ms: tuple[float, ...] = (10.0, 20.0, 40.0)
    max_samples_per_sequence: int | None = None


@dataclass(frozen=True)
class MlBaselineSample:
    sample_id: str
    sequence_id: str
    interval_idx: int
    image_chw: np.ndarray
    stats_vector: np.ndarray
    metadata_vector: np.ndarray
    gt_ttc_s: float
    gt_log_ttc_s: float


@dataclass(frozen=True)
class MlBaselineSkipRecord:
    sample_id: str
    sequence_id: str
    interval_idx: int
    reason: str
    detail: str


@dataclass(frozen=True)
class MlBaselineSequenceExport:
    sequence_id: str
    config: MlBaselineExportConfig
    sample_count: int
    skip_count: int
    image_shape_nchw: tuple[int, int, int, int]
    stats_dim: int
    metadata_dim: int
    output_npz_path: str | None
    sample_ids: tuple[str, ...]
    interval_indices: tuple[int, ...]
    skip_records: tuple[MlBaselineSkipRecord, ...]


def _resize_2d_nearest(image_hw: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    src_h, src_w = image_hw.shape
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Cannot resize empty image of shape {image_hw.shape}")
    row_idx = np.linspace(0, src_h - 1, target_height).astype(np.int64)
    col_idx = np.linspace(0, src_w - 1, target_width).astype(np.int64)
    return image_hw[row_idx][:, col_idx]


def _normalize_event_polarity(p: np.ndarray) -> np.ndarray:
    return (p.astype(np.int64) > 0).astype(np.uint8)


def _build_image_tensor(lts: LtsV1, *, image_height: int, image_width: int) -> np.ndarray:
    time_surface = lts.time_surface.astype(np.float32)
    count_map = np.log1p(lts.count_map.astype(np.float32))
    occupancy_map = lts.occupancy_map.astype(np.float32)
    resized = [
        _resize_2d_nearest(channel, image_height, image_width)
        for channel in (time_surface, count_map, occupancy_map)
    ]
    return np.stack(resized, axis=0).astype(np.float32)


def _build_polarity_split_image_tensor(
    bundle: Any,
    lts: LtsV1,
    *,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    roi = lts.roi
    h, w = lts.shape
    x = bundle.event_slice.x.astype(np.int64)
    y = bundle.event_slice.y.astype(np.int64)
    t_us = bundle.event_slice.t_us.astype(np.int64)
    p = _normalize_event_polarity(bundle.event_slice.p)

    inside = (
        (x >= roi.x0)
        & (x < roi.x1)
        & (y >= roi.y0)
        & (y < roi.y1)
    )
    x_roi = x[inside] - roi.x0
    y_roi = y[inside] - roi.y0
    t_roi = t_us[inside]
    p_roi = p[inside]

    pos_mask = p_roi > 0
    neg_mask = ~pos_mask

    def build_maps(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count_map = np.zeros((h, w), dtype=np.int32)
        last_timestamp_us = np.full((h, w), -1, dtype=np.int64)
        if np.any(mask):
            x_part = x_roi[mask]
            y_part = y_roi[mask]
            t_part = t_roi[mask]
            np.add.at(count_map, (y_part, x_part), 1)
            linear_idx = y_part * w + x_part
            np.maximum.at(last_timestamp_us.reshape(-1), linear_idx, t_part)
        valid_mask = last_timestamp_us >= 0
        occupancy_map = valid_mask.astype(np.float32)
        time_surface = np.zeros((h, w), dtype=np.float32)
        if np.any(valid_mask):
            age_us = lts.reference_t_us - last_timestamp_us[valid_mask]
            time_surface[valid_mask] = np.exp(-age_us.astype(np.float32) / float(lts.tau_us))
        return time_surface, np.log1p(count_map.astype(np.float32)), occupancy_map

    pos_time_surface, pos_count_map, pos_occupancy_map = build_maps(pos_mask)
    neg_time_surface, neg_count_map, neg_occupancy_map = build_maps(neg_mask)
    resized = [
        _resize_2d_nearest(channel, image_height, image_width)
        for channel in (
            pos_time_surface,
            neg_time_surface,
            pos_count_map,
            neg_count_map,
            pos_occupancy_map,
            neg_occupancy_map,
        )
    ]
    return np.stack(resized, axis=0).astype(np.float32)


def _resolve_multi_tau_ms(config: MlBaselineExportConfig) -> tuple[float, ...]:
    values = tuple(float(value) for value in config.multi_tau_ms if float(value) > 0.0)
    if not values:
        raise ValueError("multi_tau_ms must contain at least one positive tau value")
    return values


def _build_multi_tau_polarity_split_image_tensor(
    bundle: Any,
    *,
    config: MlBaselineExportConfig,
) -> np.ndarray:
    tau_values = _resolve_multi_tau_ms(config)
    channels: list[np.ndarray] = []
    for tau_ms in tau_values:
        tau_lts = build_lts_v1(
            bundle,
            tau_ms=tau_ms,
            pad_px=config.pad_px,
        )
        channels.append(
            _build_polarity_split_image_tensor(
                bundle,
                tau_lts,
                image_height=config.image_height,
                image_width=config.image_width,
            )
        )
    return np.concatenate(channels, axis=0).astype(np.float32)


def _build_stats_vector(stats: LocalStatsV1) -> np.ndarray:
    parts = [
        np.log1p(stats.event_count_grid.astype(np.float32)).reshape(-1),
        stats.occupancy_ratio_grid.astype(np.float32).reshape(-1),
        stats.mean_surface_grid.astype(np.float32).reshape(-1),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


def _build_metadata_vector(lts: LtsV1) -> np.ndarray:
    roi_h, roi_w = lts.shape
    values = np.asarray(
        [
            float(roi_h),
            float(roi_w),
            float(lts.roi.area),
            float(lts.roi_event_count),
            float(lts.occupancy_ratio),
            float(lts.mean_surface),
        ],
        dtype=np.float32,
    )
    return values


def build_ml_baseline_sample(
    sequence_id: str,
    interval_idx: int,
    *,
    adapter: NativeSampleAdapter,
    config: MlBaselineExportConfig,
) -> MlBaselineSample:
    bundle = adapter.get_sample(interval_idx=interval_idx, include_events=True)
    if not bundle.interval_sample.has_valid_gt:
        raise ValueError(f"Sample {bundle.sample_id} has invalid GT: {bundle.interval_sample.status_gt}")

    lts = build_lts_v1(
        bundle,
        tau_ms=config.tau_ms,
        pad_px=config.pad_px,
    )
    stats = build_local_stats_v1(
        lts,
        grid_rows=config.grid_rows,
        grid_cols=config.grid_cols,
    )

    gt_ttc_s = float(bundle.interval_sample.gt_ttc_s)
    if gt_ttc_s <= 0.0:
        raise ValueError(f"Sample {bundle.sample_id} has non-positive GT TTC: {gt_ttc_s}")

    if config.image_variant == "lts_v1":
        image_chw = _build_image_tensor(
            lts,
            image_height=config.image_height,
            image_width=config.image_width,
        )
    elif config.image_variant == "polarity_split_v1":
        image_chw = _build_polarity_split_image_tensor(
            bundle,
            lts,
            image_height=config.image_height,
            image_width=config.image_width,
        )
    elif config.image_variant == "multi_tau_polarity_split_v1":
        image_chw = _build_multi_tau_polarity_split_image_tensor(
            bundle,
            config=config,
        )
    else:
        raise ValueError(f"Unsupported image_variant={config.image_variant!r}")

    return MlBaselineSample(
        sample_id=bundle.sample_id,
        sequence_id=sequence_id,
        interval_idx=interval_idx,
        image_chw=image_chw,
        stats_vector=_build_stats_vector(stats),
        metadata_vector=_build_metadata_vector(lts),
        gt_ttc_s=gt_ttc_s,
        gt_log_ttc_s=float(np.log(gt_ttc_s)),
    )


def export_ml_baseline_sequence(
    sequence_id: str,
    *,
    config: MlBaselineExportConfig,
    output_npz_path: Path | None = None,
    formal_root: Path | None = None,
) -> MlBaselineSequenceExport:
    samples: list[MlBaselineSample] = []
    skips: list[MlBaselineSkipRecord] = []

    with NativeSampleAdapter(sequence_id, formal_root=formal_root) as adapter:
        emitted = 0
        for bundle in adapter.iter_samples(include_events=False):
            sample = bundle.interval_sample
            if not sample.has_valid_gt:
                skips.append(
                    MlBaselineSkipRecord(
                        sample_id=sample.sample_id,
                        sequence_id=sequence_id,
                        interval_idx=sample.interval_idx,
                        reason="INVALID_GT",
                        detail=sample.status_gt,
                    )
                )
                continue

            try:
                sample_export = build_ml_baseline_sample(
                    sequence_id,
                    sample.interval_idx,
                    adapter=adapter,
                    config=config,
                )
            except Exception as exc:
                skips.append(
                    MlBaselineSkipRecord(
                        sample_id=sample.sample_id,
                        sequence_id=sequence_id,
                        interval_idx=sample.interval_idx,
                        reason="EXPORT_FAIL",
                        detail=str(exc),
                    )
                )
                continue

            samples.append(sample_export)
            emitted += 1
            if config.max_samples_per_sequence is not None and emitted >= config.max_samples_per_sequence:
                break

    if not samples:
        raise ValueError(f"No valid exportable samples found for sequence {sequence_id}")

    images = np.stack([item.image_chw for item in samples], axis=0).astype(np.float32)
    stats = np.stack([item.stats_vector for item in samples], axis=0).astype(np.float32)
    metadata = np.stack([item.metadata_vector for item in samples], axis=0).astype(np.float32)
    gt_ttc_s = np.asarray([item.gt_ttc_s for item in samples], dtype=np.float32)
    gt_log_ttc_s = np.asarray([item.gt_log_ttc_s for item in samples], dtype=np.float32)
    sample_ids = np.asarray([item.sample_id for item in samples])
    sequence_ids = np.asarray([item.sequence_id for item in samples])
    interval_indices = np.asarray([item.interval_idx for item in samples], dtype=np.int32)

    if output_npz_path is not None:
        output_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz_path,
            image_nchw=images,
            stats=stats,
            metadata=metadata,
            gt_ttc_s=gt_ttc_s,
            gt_log_ttc_s=gt_log_ttc_s,
            sample_id=sample_ids,
            sequence_id=sequence_ids,
            interval_idx=interval_indices,
        )

    return MlBaselineSequenceExport(
        sequence_id=sequence_id,
        config=config,
        sample_count=len(samples),
        skip_count=len(skips),
        image_shape_nchw=tuple(int(v) for v in images.shape),
        stats_dim=int(stats.shape[1]),
        metadata_dim=int(metadata.shape[1]),
        output_npz_path=None if output_npz_path is None else str(output_npz_path),
        sample_ids=tuple(str(v) for v in sample_ids.tolist()),
        interval_indices=tuple(int(v) for v in interval_indices.tolist()),
        skip_records=tuple(skips),
    )


def summarize_ml_baseline_exports(exports: Iterable[MlBaselineSequenceExport]) -> dict[str, Any]:
    items = list(exports)
    if not items:
        return {
            "sequence_count": 0,
            "total_sample_count": 0,
            "total_skip_count": 0,
            "sequences": [],
        }

    return {
        "sequence_count": len(items),
        "total_sample_count": int(sum(item.sample_count for item in items)),
        "total_skip_count": int(sum(item.skip_count for item in items)),
        "sequences": [
            {
                "sequence_id": item.sequence_id,
                "sample_count": item.sample_count,
                "skip_count": item.skip_count,
                "image_shape_nchw": list(item.image_shape_nchw),
                "stats_dim": item.stats_dim,
                "metadata_dim": item.metadata_dim,
                "output_npz_path": item.output_npz_path,
                "skip_records": [asdict(skip) for skip in item.skip_records],
            }
            for item in items
        ],
    }


__all__ = [
    "MlBaselineExportConfig",
    "MlBaselineSample",
    "MlBaselineSequenceExport",
    "MlBaselineSkipRecord",
    "build_ml_baseline_sample",
    "export_ml_baseline_sequence",
    "summarize_ml_baseline_exports",
]
