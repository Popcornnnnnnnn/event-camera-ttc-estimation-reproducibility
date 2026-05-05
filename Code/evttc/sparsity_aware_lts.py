from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .confidence import (
    ConfidenceConfigV1,
    LocalConfidenceV1,
    build_local_confidence_v1,
)
from .formal_loader import BoundingBoxRecord
from .native_adapter import NativeSampleBundle
from .representations import LocalStatsV1, LtsV1, build_local_stats_v1, build_lts_v1


@dataclass(frozen=True)
class LocalCueConfigV1:
    min_valid_pixels: int = 8
    min_radius_px: float = 3.0
    min_slope_px_per_s: float = 1.0
    min_ttc_s: float = 0.05
    max_ttc_s: float = 20.0


@dataclass(frozen=True)
class LocalCueConfigV2:
    min_valid_pixels: int = 8
    min_surface_value: float = 0.15
    min_radius_px: float = 2.0
    min_radius_span_px: float = 3.0
    min_time_span_s: float = 0.001
    min_slope_px_per_s: float = 1.0
    min_ttc_s: float = 0.05
    max_ttc_s: float = 20.0
    target_radius_span_px: float = 10.0
    target_time_span_s: float = 0.02
    target_fit_residual_px: float = 2.5
    min_center_dist_ratio: float = 0.10
    target_center_dist_ratio: float = 0.35


@dataclass(frozen=True)
class AggregationConfigV1:
    selection_threshold: float = 0.55
    min_selected_cells: int = 1
    min_valid_local_cues: int = 1
    blend_with_fallback: bool = True
    fallback_blend_weight: float = 0.35
    v2_quality_gate: float = 0.90
    v2_consistency_gate: float = 0.40
    v2_selector_score_gate: float = 0.78
    v2_selector_score_floor: float = 0.60
    v2_cue_only_min_valid_cues: int = 3
    v2_low_support_valid_cues: int = 1
    v2_cue_only_min_cue_fallback_ratio: float = 0.60
    v2_min_fallback_blend_weight: float = 0.30
    v2_max_fallback_blend_weight: float = 0.60
    v2_min_cue_fallback_ratio: float = 0.55
    v2_guard_quality_floor: float = 0.80
    v2_guard_consistency_ceiling: float = 0.75


@dataclass(frozen=True)
class HeuristicFallbackConfigV1:
    scale: float = 0.117
    min_mean_surface: float = 0.05
    min_event_count: int = 1
    min_ttc_s: float = 0.05
    max_ttc_s: float = 20.0


@dataclass(frozen=True)
class BBoxLoomingConfigV1:
    history_back_labels: int = 7
    history_min_back_labels: int = 2
    min_fit_points: int = 2
    min_width_px: float = 3.0
    min_slope_px_per_s: float = 1.0
    min_ttc_s: float = 0.05
    max_ttc_s: float = 20.0
    fallback_to_lts_v1: bool = True
    cue_mode: str = "width"
    candidate_quantile: float = 0.25
    include_bottom_cue: bool = False
    bottom_reference: str = "principal_point"
    bottom_reference_fraction: float = 0.50
    quality_gate_enabled: bool = False
    quality_min_candidate_count: int = 0
    quality_min_fit_points: int = 0
    quality_max_iqr_over_median: float = float("inf")
    quality_low_confidence_strategy: str = "reject"
    quality_low_candidate_quantile: float = 0.25
    quality_low_fit_points_quantile: float = 0.50
    quality_high_dispersion_quantile: float = 0.50
    quality_low_candidate_spike_min_ttc_s: float = 0.50
    quality_low_candidate_spike_q25_over_min_max: float = 3.0
    quality_low_candidate_spike_quantile: float = 0.75
    quality_high_dispersion_short_min_ttc_s: float = 0.40
    quality_high_dispersion_q25_over_min_min: float = 5.0
    quality_high_dispersion_short_quantile: float = 0.25
    quality_reject_low_confidence_after_adaptive: bool = False
    current_bbox_collapse_guard_enabled: bool = False
    current_bbox_collapse_history_back: int = 5
    current_bbox_collapse_min_history: int = 3
    current_bbox_collapse_area_ratio_min: float = 0.20
    history_bbox_collapse_guard_enabled: bool = False
    history_bbox_collapse_neighbor_back: int = 2
    history_bbox_collapse_neighbor_forward: int = 2
    history_bbox_collapse_area_ratio_min: float = 0.20
    history_bbox_collapse_guard_require_low_confidence: bool = False
    phase_quantile_enabled: bool = False
    phase_early_fraction_max: float = 0.35
    phase_early_iqr_over_median_min: float = 0.40
    phase_early_spread_over_median_min: float = 0.80
    phase_early_quantile: float = 0.50
    phase_growth_quantile_enabled: bool = False
    phase_growth_fraction_max: float = 0.50
    phase_growth_area_ratio_med5_min: float = 1.10
    phase_growth_iqr_over_median_min: float = 0.30
    phase_growth_quantile: float = 0.50
    phase_stable_quantile_enabled: bool = False
    phase_stable_fraction_min: float = 0.20
    phase_stable_area_ratio_med5_max: float = 1.05
    phase_stable_iqr_over_median_max: float = 0.55
    phase_stable_quantile: float = 0.10
    phase_late_fraction_min: float = 0.50
    phase_late_iqr_over_median_max: float = 0.35
    phase_late_area_ratio_med5_max: float = float("inf")
    phase_late_quantile: float = 0.10


@dataclass(frozen=True)
class SparsityAwareLtsConfigV1:
    tau_ms: float = 20.0
    grid_rows: int = 3
    grid_cols: int = 3
    pad_px: int = 0
    reference_time: str = "end"
    confidence: ConfidenceConfigV1 = ConfidenceConfigV1()
    cue: LocalCueConfigV1 = LocalCueConfigV1()
    aggregation: AggregationConfigV1 = AggregationConfigV1()
    fallback: HeuristicFallbackConfigV1 = HeuristicFallbackConfigV1()
    bbox_looming: BBoxLoomingConfigV1 = BBoxLoomingConfigV1()
    interface_version: str = "EvTTC-Formal-v0.1"
    representation_version: str = "LtsV1+LocalStatsV1+LocalConfidenceV1"


@dataclass(frozen=True)
class LocalTtcCueV1:
    row: int
    col: int
    valid_pixel_count: int
    confidence: float
    radius_ref_px: float | None
    slope_px_per_s: float | None
    ttc_s: float | None
    status: str


@dataclass(frozen=True)
class LocalTtcCueV2:
    row: int
    col: int
    valid_pixel_count: int
    confidence: float
    local_center_x: float | None
    local_center_y: float | None
    radius_ref_px: float | None
    slope_px_per_s: float | None
    fit_residual_px: float | None
    time_span_s: float | None
    radius_span_px: float | None
    cue_quality: float | None
    ttc_s: float | None
    status: str


@dataclass(frozen=True)
class PredictionRecordV1:
    sample_id: str
    sequence_id: str
    interval_idx: int
    timestamp_s: float
    gt_ttc_s: float | None
    ttc_est_s: float | None
    e_ttc_pct: float | None
    confidence: float | None
    status: str
    cost_time_s: float
    local_cue_valid_count: int
    selected_cell_count: int
    fallback_ttc_s: float | None
    debug_ref: str


@dataclass(frozen=True)
class PredictionDebugV1:
    lts: LtsV1 | None
    stats: LocalStatsV1 | None
    local_confidence: LocalConfidenceV1 | None
    local_cues: tuple[LocalTtcCueV1, ...]


@dataclass(frozen=True)
class PredictionDebugV2:
    lts: LtsV1 | None
    stats: LocalStatsV1 | None
    local_confidence: LocalConfidenceV1 | None
    local_cues: tuple[LocalTtcCueV2, ...]


def _weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float] | None:
    if x.size < 2 or y.size < 2 or weights.size < 2:
        return None
    w = np.asarray(weights, dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return None

    x_mean = float((w * x).sum() / w_sum)
    y_mean = float((w * y).sum() / w_sum)
    x_cent = x - x_mean
    y_cent = y - y_mean
    denom = float((w * x_cent * x_cent).sum())
    if abs(denom) < 1e-12:
        return None
    slope = float((w * x_cent * y_cent).sum() / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def _weighted_fit_residual(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    slope: float,
    intercept: float,
) -> float | None:
    if x.size == 0 or y.size == 0 or weights.size == 0:
        return None
    w = np.asarray(weights, dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return None
    pred = slope * x + intercept
    residual = np.sqrt(np.sum(w * np.square(y - pred)) / w_sum)
    return float(residual)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(sorted_weights.sum())
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return float(sorted_values[idx])


def _cell_edges(length: int, cells: int) -> np.ndarray:
    return np.linspace(0, length, cells + 1, dtype=np.int64)


def _weighted_centroid(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float] | None:
    if x.size == 0 or y.size == 0 or weights.size == 0:
        return None
    w = np.asarray(weights, dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return None
    cx = float((w * x).sum() / w_sum)
    cy = float((w * y).sum() / w_sum)
    return cx, cy


def _build_local_ttc_cues_v1(
    bundle: NativeSampleBundle,
    lts: LtsV1,
    local_confidence: LocalConfidenceV1,
    *,
    cue_config: LocalCueConfigV1,
) -> tuple[LocalTtcCueV1, ...]:
    bbox = bundle.interval_sample
    cx = 0.5 * (bbox.bbox_x1 + bbox.bbox_x2)
    cy = 0.5 * (bbox.bbox_y1 + bbox.bbox_y2)
    ref_t_s = lts.reference_t_us / 1e6

    y_edges = _cell_edges(lts.shape[0], local_confidence.grid_rows)
    x_edges = _cell_edges(lts.shape[1], local_confidence.grid_cols)
    cues: list[LocalTtcCueV1] = []

    for row, col in local_confidence.selected_cell_indices:
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
        cell_valid = lts.valid_mask[y0:y1, x0:x1]
        valid_pixel_count = int(cell_valid.sum())
        if valid_pixel_count < cue_config.min_valid_pixels:
            cues.append(
                LocalTtcCueV1(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=float(local_confidence.confidence_grid[row, col]),
                    radius_ref_px=None,
                    slope_px_per_s=None,
                    ttc_s=None,
                    status="TOO_FEW_PIXELS",
                )
            )
            continue

        rel_y, rel_x = np.nonzero(cell_valid)
        abs_x = rel_x.astype(np.float64) + x0 + lts.roi.x0
        abs_y = rel_y.astype(np.float64) + y0 + lts.roi.y0
        radii = np.sqrt((abs_x - cx) ** 2 + (abs_y - cy) ** 2)
        t_last_s = lts.last_timestamp_us[y0:y1, x0:x1][cell_valid].astype(np.float64) / 1e6
        tau_s = t_last_s - ref_t_s
        weights = lts.time_surface[y0:y1, x0:x1][cell_valid].astype(np.float64)

        fit = _weighted_linear_fit(tau_s, radii.astype(np.float64), weights)
        if fit is None:
            cues.append(
                LocalTtcCueV1(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=float(local_confidence.confidence_grid[row, col]),
                    radius_ref_px=None,
                    slope_px_per_s=None,
                    ttc_s=None,
                    status="FIT_FAIL",
                )
            )
            continue

        slope_px_per_s, radius_ref_px = fit
        if radius_ref_px <= cue_config.min_radius_px:
            cues.append(
                LocalTtcCueV1(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=float(local_confidence.confidence_grid[row, col]),
                    radius_ref_px=float(radius_ref_px),
                    slope_px_per_s=float(slope_px_per_s),
                    ttc_s=None,
                    status="RADIUS_TOO_SMALL",
                )
            )
            continue
        if slope_px_per_s <= cue_config.min_slope_px_per_s:
            cues.append(
                LocalTtcCueV1(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=float(local_confidence.confidence_grid[row, col]),
                    radius_ref_px=float(radius_ref_px),
                    slope_px_per_s=float(slope_px_per_s),
                    ttc_s=None,
                    status="NON_POSITIVE_EXPANSION",
                )
            )
            continue

        ttc_s = float(radius_ref_px / slope_px_per_s)
        if not math.isfinite(ttc_s) or ttc_s < cue_config.min_ttc_s or ttc_s > cue_config.max_ttc_s:
            cues.append(
                LocalTtcCueV1(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=float(local_confidence.confidence_grid[row, col]),
                    radius_ref_px=float(radius_ref_px),
                    slope_px_per_s=float(slope_px_per_s),
                    ttc_s=None,
                    status="TTC_OUT_OF_RANGE",
                )
            )
            continue

        cues.append(
            LocalTtcCueV1(
                row=row,
                col=col,
                valid_pixel_count=valid_pixel_count,
                confidence=float(local_confidence.confidence_grid[row, col]),
                radius_ref_px=float(radius_ref_px),
                slope_px_per_s=float(slope_px_per_s),
                ttc_s=ttc_s,
                status="OK",
            )
        )

    return tuple(cues)


def build_local_ttc_cues_v2(
    bundle: NativeSampleBundle,
    lts: LtsV1,
    local_confidence: LocalConfidenceV1,
    *,
    cue_config: LocalCueConfigV2 | None = None,
) -> tuple[LocalTtcCueV2, ...]:
    cfg = cue_config or LocalCueConfigV2()
    bbox = bundle.interval_sample
    bbox_cx = 0.5 * (bbox.bbox_x1 + bbox.bbox_x2)
    bbox_cy = 0.5 * (bbox.bbox_y1 + bbox.bbox_y2)
    bbox_w = max(1e-6, float(bbox.bbox_x2 - bbox.bbox_x1))
    bbox_h = max(1e-6, float(bbox.bbox_y2 - bbox.bbox_y1))
    bbox_diag = math.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)
    ref_t_s = lts.reference_t_us / 1e6
    y_edges = _cell_edges(lts.shape[0], local_confidence.grid_rows)
    x_edges = _cell_edges(lts.shape[1], local_confidence.grid_cols)
    cues: list[LocalTtcCueV2] = []

    for row, col in local_confidence.selected_cell_indices:
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
        cell_valid = lts.valid_mask[y0:y1, x0:x1]
        cell_surface = lts.time_surface[y0:y1, x0:x1]
        active_mask = cell_valid & (cell_surface >= float(cfg.min_surface_value))
        valid_pixel_count = int(active_mask.sum())
        confidence = float(local_confidence.confidence_grid[row, col])
        if valid_pixel_count < cfg.min_valid_pixels:
            cues.append(
                LocalTtcCueV2(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=confidence,
                    local_center_x=None,
                    local_center_y=None,
                    radius_ref_px=None,
                    slope_px_per_s=None,
                    fit_residual_px=None,
                    time_span_s=None,
                    radius_span_px=None,
                    cue_quality=None,
                    ttc_s=None,
                    status="TOO_FEW_ACTIVE_PIXELS",
                )
            )
            continue

        rel_y, rel_x = np.nonzero(active_mask)
        abs_x = rel_x.astype(np.float64) + x0 + lts.roi.x0
        abs_y = rel_y.astype(np.float64) + y0 + lts.roi.y0
        weights = cell_surface[active_mask].astype(np.float64)
        centroid = _weighted_centroid(abs_x, abs_y, weights)
        if centroid is None:
            cues.append(
                LocalTtcCueV2(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=confidence,
                    local_center_x=None,
                    local_center_y=None,
                    radius_ref_px=None,
                    slope_px_per_s=None,
                    fit_residual_px=None,
                    time_span_s=None,
                    radius_span_px=None,
                    cue_quality=None,
                    ttc_s=None,
                    status="CENTROID_FAIL",
                )
            )
            continue

        local_cx, local_cy = centroid
        radii = np.sqrt((abs_x - local_cx) ** 2 + (abs_y - local_cy) ** 2)
        mean_radius_px = float(np.average(radii, weights=weights))
        radius_span_px = float(radii.max() - radii.min()) if radii.size else 0.0
        if mean_radius_px <= cfg.min_radius_px or radius_span_px < cfg.min_radius_span_px:
            cues.append(
                LocalTtcCueV2(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=confidence,
                    local_center_x=local_cx,
                    local_center_y=local_cy,
                    radius_ref_px=mean_radius_px,
                    slope_px_per_s=None,
                    fit_residual_px=None,
                    time_span_s=None,
                    radius_span_px=radius_span_px,
                    cue_quality=None,
                    ttc_s=None,
                    status="RADIUS_SUPPORT_TOO_SMALL",
                )
            )
            continue

        t_last_s = lts.last_timestamp_us[y0:y1, x0:x1][active_mask].astype(np.float64) / 1e6
        tau_s = t_last_s - ref_t_s
        time_span_s = float(t_last_s.max() - t_last_s.min()) if t_last_s.size else 0.0
        if time_span_s < cfg.min_time_span_s:
            cues.append(
                LocalTtcCueV2(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=confidence,
                    local_center_x=local_cx,
                    local_center_y=local_cy,
                    radius_ref_px=mean_radius_px,
                    slope_px_per_s=None,
                    fit_residual_px=None,
                    time_span_s=time_span_s,
                    radius_span_px=radius_span_px,
                    cue_quality=None,
                    ttc_s=None,
                    status="TIME_SUPPORT_TOO_SMALL",
                )
            )
            continue

        fit = _weighted_linear_fit(tau_s, radii.astype(np.float64), weights)
        if fit is None:
            cues.append(
                LocalTtcCueV2(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=confidence,
                    local_center_x=local_cx,
                    local_center_y=local_cy,
                    radius_ref_px=radius_ref_px,
                    slope_px_per_s=None,
                    fit_residual_px=None,
                    time_span_s=time_span_s,
                    radius_span_px=radius_span_px,
                    cue_quality=None,
                    ttc_s=None,
                    status="FIT_FAIL",
                )
            )
            continue

        slope_px_per_s, intercept = fit
        radius_ref_px = float(intercept)
        if radius_ref_px <= cfg.min_radius_px:
            cues.append(
                LocalTtcCueV2(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=confidence,
                    local_center_x=local_cx,
                    local_center_y=local_cy,
                    radius_ref_px=radius_ref_px,
                    slope_px_per_s=float(slope_px_per_s),
                    fit_residual_px=None,
                    time_span_s=time_span_s,
                    radius_span_px=radius_span_px,
                    cue_quality=None,
                    ttc_s=None,
                    status="RADIUS_TOO_SMALL",
                )
            )
            continue
        if slope_px_per_s <= cfg.min_slope_px_per_s:
            cues.append(
                LocalTtcCueV2(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=confidence,
                    local_center_x=local_cx,
                    local_center_y=local_cy,
                    radius_ref_px=radius_ref_px,
                    slope_px_per_s=float(slope_px_per_s),
                    fit_residual_px=None,
                    time_span_s=time_span_s,
                    radius_span_px=radius_span_px,
                    cue_quality=None,
                    ttc_s=None,
                    status="NON_POSITIVE_EXPANSION",
                )
            )
            continue

        fit_residual_px = _weighted_fit_residual(tau_s, radii.astype(np.float64), weights, slope_px_per_s, intercept)
        residual_score = 0.0
        if fit_residual_px is not None:
            residual_score = float(
                np.clip(1.0 - fit_residual_px / max(cfg.target_fit_residual_px, 1e-6), 0.0, 1.0)
            )
        radius_score = float(np.clip(radius_span_px / max(cfg.target_radius_span_px, 1e-6), 0.0, 1.0))
        time_score = float(np.clip(time_span_s / max(cfg.target_time_span_s, 1e-6), 0.0, 1.0))
        center_dist_px = math.sqrt((local_cx - bbox_cx) ** 2 + (local_cy - bbox_cy) ** 2)
        min_center_dist_px = cfg.min_center_dist_ratio * bbox_diag
        target_center_dist_px = max(min_center_dist_px + 1e-6, cfg.target_center_dist_ratio * bbox_diag)
        center_dist_score = float(
            np.clip(
                (center_dist_px - min_center_dist_px) / max(target_center_dist_px - min_center_dist_px, 1e-6),
                0.0,
                1.0,
            )
        )
        cue_quality = float((confidence + residual_score + radius_score + time_score + center_dist_score) / 5.0)

        ttc_s = float(radius_ref_px / slope_px_per_s)
        if not math.isfinite(ttc_s) or ttc_s < cfg.min_ttc_s or ttc_s > cfg.max_ttc_s:
            cues.append(
                LocalTtcCueV2(
                    row=row,
                    col=col,
                    valid_pixel_count=valid_pixel_count,
                    confidence=confidence,
                    local_center_x=local_cx,
                    local_center_y=local_cy,
                    radius_ref_px=radius_ref_px,
                    slope_px_per_s=float(slope_px_per_s),
                    fit_residual_px=fit_residual_px,
                    time_span_s=time_span_s,
                    radius_span_px=radius_span_px,
                    cue_quality=cue_quality,
                    ttc_s=None,
                    status="TTC_OUT_OF_RANGE",
                )
            )
            continue

        cues.append(
            LocalTtcCueV2(
                row=row,
                col=col,
                valid_pixel_count=valid_pixel_count,
                confidence=confidence,
                local_center_x=local_cx,
                local_center_y=local_cy,
                radius_ref_px=radius_ref_px,
                slope_px_per_s=float(slope_px_per_s),
                fit_residual_px=fit_residual_px,
                time_span_s=time_span_s,
                radius_span_px=radius_span_px,
                cue_quality=cue_quality,
                ttc_s=ttc_s,
                status="OK",
            )
        )

    return tuple(cues)


def _fallback_ttc_from_lts(
    bundle: NativeSampleBundle,
    lts: LtsV1,
    *,
    fallback_config: HeuristicFallbackConfigV1,
) -> float | None:
    bbox = bundle.interval_sample
    bbox_w = max(1e-6, float(bbox.bbox_x2 - bbox.bbox_x1))
    bbox_h = max(1e-6, float(bbox.bbox_y2 - bbox.bbox_y1))
    bbox_diag = math.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)
    event_count = max(fallback_config.min_event_count, int(lts.roi_event_count))
    mean_surface = max(fallback_config.min_mean_surface, float(lts.mean_surface))
    ttc_s = fallback_config.scale * bbox_diag / (math.log1p(event_count) * mean_surface)
    if not math.isfinite(ttc_s):
        return None
    return float(np.clip(ttc_s, fallback_config.min_ttc_s, fallback_config.max_ttc_s))


def _bbox_width_looming_ttc(
    bundle: NativeSampleBundle,
    *,
    bbox_config: BBoxLoomingConfigV1,
) -> tuple[float | None, str, int, float | None, float | None]:
    sample = bundle.interval_sample
    max_history_back = max(int(bbox_config.history_back_labels), int(bbox_config.history_min_back_labels))
    lo = sample.start_label_id - max_history_back
    hi = sample.start_label_id
    records = [
        record
        for record in bundle.bbox_records
        if lo <= record.label_id <= hi
    ]
    if len(records) < bbox_config.min_fit_points:
        return None, "BBOX_HISTORY_TOO_SHORT", len(records), None, None

    records = sorted(records, key=lambda item: item.timestamp_s)
    guard_suffix = ""

    def filter_history_bbox_collapses(
        input_records: list[BoundingBoxRecord],
    ) -> tuple[list[BoundingBoxRecord], int]:
        if len(input_records) < 3:
            return input_records, 0
        keep_records: list[BoundingBoxRecord] = []
        filtered_count = 0
        neighbor_back = max(1, int(bbox_config.history_bbox_collapse_neighbor_back))
        neighbor_forward = max(1, int(bbox_config.history_bbox_collapse_neighbor_forward))
        area_ratio_min = float(bbox_config.history_bbox_collapse_area_ratio_min)
        areas = [
            max(0.0, float((record.x2 - record.x1) * (record.y2 - record.y1)))
            for record in input_records
        ]
        for idx, record in enumerate(input_records):
            before = areas[max(0, idx - neighbor_back):idx]
            after = areas[idx + 1:idx + 1 + neighbor_forward]
            if before and after:
                neighbor_area = float(np.median(np.asarray([*before, *after], dtype=np.float64)))
                if neighbor_area > 0.0 and areas[idx] / neighbor_area < area_ratio_min:
                    filtered_count += 1
                    continue
            keep_records.append(record)
        return keep_records, filtered_count

    if bbox_config.current_bbox_collapse_guard_enabled:
        current_records = [record for record in records if record.label_id == sample.start_label_id]
        current_record = current_records[-1] if current_records else None
        guard_lo = sample.start_label_id - max(1, int(bbox_config.current_bbox_collapse_history_back))
        previous_areas = [
            max(0.0, float((record.x2 - record.x1) * (record.y2 - record.y1)))
            for record in records
            if guard_lo <= record.label_id < sample.start_label_id
        ]
        if current_record is not None and len(previous_areas) >= int(bbox_config.current_bbox_collapse_min_history):
            current_area = max(
                0.0,
                float((current_record.x2 - current_record.x1) * (current_record.y2 - current_record.y1)),
            )
            median_previous_area = float(np.median(np.asarray(previous_areas, dtype=np.float64)))
            if (
                median_previous_area > 0.0
                and current_area / median_previous_area < float(bbox_config.current_bbox_collapse_area_ratio_min)
            ):
                records = [record for record in records if record.label_id != sample.start_label_id]
                guard_suffix += "_CURGUARD"
                if len(records) < bbox_config.min_fit_points:
                    return None, "BBOX_CURRENT_COLLAPSE_GUARD_NO_HISTORY", len(records), None, None

    if (
        bbox_config.history_bbox_collapse_guard_enabled
        and not bbox_config.history_bbox_collapse_guard_require_low_confidence
    ):
        records, filtered_count = filter_history_bbox_collapses(records)
        if filtered_count > 0:
            guard_suffix += "_HISTGUARD"
            if len(records) < bbox_config.min_fit_points:
                return None, "BBOX_HISTORY_COLLAPSE_GUARD_NO_HISTORY", len(records), None, None

    Candidate = tuple[str, float, float, float, float, float, int, int]

    def collect_candidates(window_records: list[BoundingBoxRecord], history_back: int) -> list[Candidate]:
        if len(window_records) < bbox_config.min_fit_points:
            return []
        times = np.asarray([record.timestamp_s for record in window_records], dtype=np.float64)
        widths = np.asarray([record.x2 - record.x1 for record in window_records], dtype=np.float64)
        heights = np.asarray([record.y2 - record.y1 for record in window_records], dtype=np.float64)
        bottom_y = np.asarray([record.y2 for record in window_records], dtype=np.float64)
        dimensions = {
            "width": widths,
            "height": heights,
            "diag": np.sqrt(np.square(widths) + np.square(heights)),
            "sqrt_area": np.sqrt(np.maximum(widths * heights, 1e-6)),
        }
        if bbox_config.include_bottom_cue:
            calibration = bundle.calibration
            if bbox_config.bottom_reference == "principal_point":
                bottom_ref_y = float(calibration.intrinsics.get("cy", calibration.resolution["height"] * 0.5))
            elif bbox_config.bottom_reference == "image_fraction":
                bottom_ref_y = float(calibration.resolution["height"]) * float(bbox_config.bottom_reference_fraction)
            else:
                bottom_ref_y = float(calibration.resolution["height"]) * 0.5
            dimensions["bottom_offset"] = bottom_y - bottom_ref_y
        rows: list[Candidate] = []
        for cue_name, values in dimensions.items():
            weights = np.ones_like(times, dtype=np.float64)
            fit = _weighted_linear_fit(times, values, weights)
            if fit is None:
                continue
            slope_px_per_s, intercept_px = fit
            ref_px = float(slope_px_per_s * sample.query_t_s + intercept_px)
            if ref_px <= bbox_config.min_width_px or slope_px_per_s <= bbox_config.min_slope_px_per_s:
                continue
            ttc_s = float(ref_px / slope_px_per_s)
            if not math.isfinite(ttc_s) or ttc_s < bbox_config.min_ttc_s or ttc_s > bbox_config.max_ttc_s:
                continue
            residual = _weighted_fit_residual(times, values, weights, slope_px_per_s, intercept_px)
            normalized_residual = float((residual or 0.0) / max(abs(ref_px), 1e-6))
            relative_slope = float(slope_px_per_s / max(abs(ref_px), 1e-6))
            rows.append(
                (
                    cue_name,
                    ttc_s,
                    float(slope_px_per_s),
                    ref_px,
                    normalized_residual,
                    relative_slope,
                    len(window_records),
                    history_back,
                )
            )
        return rows

    cue_mode = bbox_config.cue_mode
    if cue_mode == "multi_quantile":
        min_history_back = max(1, int(bbox_config.history_min_back_labels))
        def collect_multi_candidates(input_records: list[BoundingBoxRecord]) -> list[Candidate]:
            rows: list[Candidate] = []
            for history_back in range(min_history_back, max_history_back + 1):
                window_lo = sample.start_label_id - history_back
                window_records = [record for record in input_records if window_lo <= record.label_id <= hi]
                rows.extend(collect_candidates(window_records, history_back))
            return rows

        candidate_rows = collect_multi_candidates(records)
        if not candidate_rows:
            return None, "BBOX_FIT_FAIL", len(records), None, None
        quantile = float(np.clip(bbox_config.candidate_quantile, 0.0, 1.0))
        status_suffix = ""
        ttc_values = np.asarray([row[1] for row in candidate_rows], dtype=np.float64)
        max_fit_points = max(row[6] for row in candidate_rows)

        def current_area_ratio_med5() -> float | None:
            current_records = [record for record in records if record.label_id == sample.start_label_id]
            current_record = current_records[-1] if current_records else None
            previous_areas = [
                max(0.0, float((record.x2 - record.x1) * (record.y2 - record.y1)))
                for record in records
                if record.label_id < sample.start_label_id
            ]
            if current_record is None or not previous_areas:
                return None
            current_area = max(
                0.0,
                float((current_record.x2 - current_record.x1) * (current_record.y2 - current_record.y1)),
            )
            return current_area / max(float(np.median(np.asarray(previous_areas[-5:], dtype=np.float64))), 1e-6)

        def interval_fraction() -> float | None:
            if bundle.interval_count <= 1:
                return None
            return sample.interval_idx / max(float(bundle.interval_count - 1), 1.0)

        if (
            bbox_config.history_bbox_collapse_guard_enabled
            and bbox_config.history_bbox_collapse_guard_require_low_confidence
            and bbox_config.quality_gate_enabled
        ):
            candidate_count = len(candidate_rows)
            median_ttc_s = float(np.median(ttc_values))
            iqr_ttc_s = float(np.percentile(ttc_values, 75) - np.percentile(ttc_values, 25))
            iqr_over_median = iqr_ttc_s / max(abs(median_ttc_s), 1e-6)
            low_confidence = (
                candidate_count < int(bbox_config.quality_min_candidate_count)
                or max_fit_points < int(bbox_config.quality_min_fit_points)
                or iqr_over_median > float(bbox_config.quality_max_iqr_over_median)
            )
            if low_confidence:
                filtered_records, filtered_count = filter_history_bbox_collapses(records)
                if filtered_count > 0:
                    filtered_candidate_rows = collect_multi_candidates(filtered_records)
                    if filtered_candidate_rows:
                        records = filtered_records
                        candidate_rows = filtered_candidate_rows
                        guard_suffix += "_HISTGUARD"
                        ttc_values = np.asarray([row[1] for row in candidate_rows], dtype=np.float64)
                        max_fit_points = max(row[6] for row in candidate_rows)
                    elif len(filtered_records) < bbox_config.min_fit_points:
                        return None, "BBOX_HISTORY_COLLAPSE_GUARD_NO_HISTORY", len(filtered_records), None, None
        if bbox_config.quality_gate_enabled:
            candidate_count = len(candidate_rows)
            median_ttc_s = float(np.median(ttc_values))
            iqr_ttc_s = float(np.percentile(ttc_values, 75) - np.percentile(ttc_values, 25))
            iqr_over_median = iqr_ttc_s / max(abs(median_ttc_s), 1e-6)
            quality_status: str | None = None
            low_confidence_quantile: float | None = None
            if candidate_count < int(bbox_config.quality_min_candidate_count):
                quality_status = "BBOX_LOW_QUALITY_CANDIDATES"
                low_confidence_quantile = bbox_config.quality_low_candidate_quantile
                status_suffix = "_LOWCAND"
                if bbox_config.quality_low_confidence_strategy == "adaptive_quantile_v2":
                    candidate_min_s = float(np.min(ttc_values))
                    candidate_q25_s = float(np.quantile(ttc_values, 0.25))
                    q25_over_min = candidate_q25_s / max(candidate_min_s, 1e-6)
                    if (
                        candidate_min_s <= float(bbox_config.quality_low_candidate_spike_min_ttc_s)
                        and q25_over_min <= float(bbox_config.quality_low_candidate_spike_q25_over_min_max)
                    ):
                        low_confidence_quantile = bbox_config.quality_low_candidate_spike_quantile
                        status_suffix = "_LOWCANDSPIKE"
            elif max_fit_points < int(bbox_config.quality_min_fit_points):
                quality_status = "BBOX_LOW_QUALITY_FIT_POINTS"
                low_confidence_quantile = bbox_config.quality_low_fit_points_quantile
                status_suffix = "_LOWFIT"
            elif iqr_over_median > float(bbox_config.quality_max_iqr_over_median):
                quality_status = "BBOX_LOW_QUALITY_DISPERSION"
                low_confidence_quantile = bbox_config.quality_high_dispersion_quantile
                status_suffix = "_DISP"
                if bbox_config.quality_low_confidence_strategy == "adaptive_quantile_v2":
                    candidate_min_s = float(np.min(ttc_values))
                    candidate_q25_s = float(np.quantile(ttc_values, 0.25))
                    q25_over_min = candidate_q25_s / max(candidate_min_s, 1e-6)
                    if (
                        candidate_min_s <= float(bbox_config.quality_high_dispersion_short_min_ttc_s)
                        and q25_over_min >= float(bbox_config.quality_high_dispersion_q25_over_min_min)
                    ):
                        low_confidence_quantile = bbox_config.quality_high_dispersion_short_quantile
                        status_suffix = "_DISPSHORT"
            if quality_status is not None:
                if bbox_config.quality_reject_low_confidence_after_adaptive:
                    return None, f"{quality_status}_REJECTED", max_fit_points, None, None
                if bbox_config.quality_low_confidence_strategy in {"adaptive_quantile", "adaptive_quantile_v2"}:
                    quantile = float(np.clip(low_confidence_quantile or quantile, 0.0, 1.0))
                else:
                    return None, quality_status, max_fit_points, None, None
        if bbox_config.phase_quantile_enabled and status_suffix == "" and guard_suffix == "":
            median_ttc_s = float(np.median(ttc_values))
            iqr_ttc_s = float(np.percentile(ttc_values, 75) - np.percentile(ttc_values, 25))
            iqr_over_median = iqr_ttc_s / max(abs(median_ttc_s), 1e-6)
            spread_over_median = (
                float(np.percentile(ttc_values, 90)) - float(np.percentile(ttc_values, 10))
            ) / max(abs(median_ttc_s), 1e-6)
            sample_fraction = interval_fraction()
            area_ratio_med5 = current_area_ratio_med5()
            if (
                sample_fraction is not None
                and sample_fraction <= float(bbox_config.phase_early_fraction_max)
                and iqr_over_median >= float(bbox_config.phase_early_iqr_over_median_min)
                and spread_over_median >= float(bbox_config.phase_early_spread_over_median_min)
            ):
                quantile = float(np.clip(bbox_config.phase_early_quantile, 0.0, 1.0))
                status_suffix = "_PHASEEARLY"
            elif (
                bbox_config.phase_growth_quantile_enabled
                and sample_fraction is not None
                and area_ratio_med5 is not None
                and sample_fraction <= float(bbox_config.phase_growth_fraction_max)
                and area_ratio_med5 >= float(bbox_config.phase_growth_area_ratio_med5_min)
                and iqr_over_median >= float(bbox_config.phase_growth_iqr_over_median_min)
            ):
                quantile = float(np.clip(bbox_config.phase_growth_quantile, 0.0, 1.0))
                status_suffix = "_PHASEGROWTH"
            elif (
                bbox_config.phase_stable_quantile_enabled
                and sample_fraction is not None
                and area_ratio_med5 is not None
                and sample_fraction >= float(bbox_config.phase_stable_fraction_min)
                and area_ratio_med5 <= float(bbox_config.phase_stable_area_ratio_med5_max)
                and iqr_over_median <= float(bbox_config.phase_stable_iqr_over_median_max)
            ):
                quantile = float(np.clip(bbox_config.phase_stable_quantile, 0.0, 1.0))
                status_suffix = "_PHASESTABLE"
            elif (
                sample_fraction is not None
                and area_ratio_med5 is not None
                and sample_fraction >= float(bbox_config.phase_late_fraction_min)
                and area_ratio_med5 <= float(bbox_config.phase_late_area_ratio_med5_max)
                and iqr_over_median <= float(bbox_config.phase_late_iqr_over_median_max)
            ):
                quantile = float(np.clip(bbox_config.phase_late_quantile, 0.0, 1.0))
                status_suffix = "_PHASELATE"
        ttc_s = float(np.quantile(ttc_values, quantile))
        selected_row = min(candidate_rows, key=lambda item: abs(item[1] - ttc_s))
        _, _, slope_px_per_s, ref_px, _, _, fit_points, _ = selected_row
        status = f"OK_BBOX_MULTI_Q{int(round(quantile * 100)):02d}{status_suffix}{guard_suffix}"
        return ttc_s, status, fit_points, slope_px_per_s, ref_px

    candidates = {row[0]: row[1:] for row in collect_candidates(records, max_history_back)}
    if not candidates:
        return None, "BBOX_FIT_FAIL", len(records), None, None

    if cue_mode == "width":
        if "width" not in candidates:
            return None, "BBOX_WIDTH_UNAVAILABLE", len(records), None, None
        selected = "width"
    elif cue_mode == "lowest_residual":
        selected = min(candidates, key=lambda item: candidates[item][3])
    elif cue_mode == "score_rel_slope_over_resid":
        selected = max(candidates, key=lambda item: candidates[item][4] / max(candidates[item][3], 1e-4))
    elif cue_mode == "highest_rel_slope":
        selected = max(candidates, key=lambda item: candidates[item][4])
    else:
        return None, f"BBOX_UNKNOWN_CUE_MODE:{cue_mode}", len(records), None, None

    ttc_s, slope_px_per_s, ref_px, normalized_residual, relative_slope, fit_points, _ = candidates[selected]
    status = f"OK_BBOX_{selected.upper()}{guard_suffix}"
    return ttc_s, status, fit_points, float(slope_px_per_s), float(ref_px)


def _aggregate_local_cues_v2(
    local_cues: tuple[LocalTtcCueV2, ...],
    *,
    aggregation_config: AggregationConfigV1,
    fallback_ttc_s: float | None,
) -> tuple[float | None, str, int]:
    valid_local_cues = [
        cue
        for cue in local_cues
        if cue.status == "OK" and cue.ttc_s is not None and cue.cue_quality is not None
    ]
    if len(valid_local_cues) >= aggregation_config.min_valid_local_cues:
        cue_values = np.asarray([cue.ttc_s for cue in valid_local_cues], dtype=np.float64)
        cue_weights = np.asarray(
            [
                max(1.0, float(cue.valid_pixel_count)) * max(1e-6, cue.confidence) * max(1e-6, cue.cue_quality)
                for cue in valid_local_cues
            ],
            dtype=np.float64,
        )
        agg_ttc_s = _weighted_median(cue_values, cue_weights)
        median_ttc_s = float(np.median(cue_values))
        cue_iqr_s = float(np.percentile(cue_values, 75) - np.percentile(cue_values, 25))
        normalized_iqr = float(cue_iqr_s / max(abs(median_ttc_s), 1e-6))
        consistency_gate = max(1e-6, float(aggregation_config.v2_consistency_gate))
        consistency_score = float(np.clip(1.0 - normalized_iqr / consistency_gate, 0.0, 1.0))

        support_span = max(
            1,
            int(aggregation_config.v2_cue_only_min_valid_cues) - int(aggregation_config.v2_low_support_valid_cues),
        )
        support_score = float(
            np.clip(
                (len(valid_local_cues) - int(aggregation_config.v2_low_support_valid_cues)) / support_span,
                0.0,
                1.0,
            )
        )
        if aggregation_config.blend_with_fallback and fallback_ttc_s is not None:
            mean_quality = float(
                np.average(
                    np.asarray([float(cue.cue_quality) for cue in valid_local_cues], dtype=np.float64),
                    weights=cue_weights,
                )
            )
            selector_score = float((mean_quality + consistency_score + support_score) / 3.0)
            gate = max(1e-6, float(aggregation_config.v2_quality_gate))
            cue_fallback_ratio = float(agg_ttc_s / max(fallback_ttc_s, 1e-6))

            if (
                len(valid_local_cues) >= aggregation_config.v2_cue_only_min_valid_cues
                and mean_quality >= gate
                and selector_score >= aggregation_config.v2_selector_score_gate
                and normalized_iqr <= aggregation_config.v2_consistency_gate
                and cue_fallback_ratio >= aggregation_config.v2_cue_only_min_cue_fallback_ratio
            ):
                return agg_ttc_s, "OK_CUE_ONLY", len(valid_local_cues)

            if (
                cue_fallback_ratio < aggregation_config.v2_min_cue_fallback_ratio
                and (
                    mean_quality < aggregation_config.v2_guard_quality_floor
                    or selector_score < aggregation_config.v2_selector_score_floor
                    or normalized_iqr > aggregation_config.v2_guard_consistency_ceiling
                    or len(valid_local_cues) <= aggregation_config.v2_low_support_valid_cues
                )
            ):
                return fallback_ttc_s, "OK_FALLBACK", len(valid_local_cues)

            blend_signal = float(np.clip(selector_score, 0.0, 1.0))
            blend_weight = float(
                aggregation_config.v2_max_fallback_blend_weight
                - (
                    aggregation_config.v2_max_fallback_blend_weight
                    - aggregation_config.v2_min_fallback_blend_weight
                )
                * blend_signal
            )
            blend_weight = float(
                np.clip(
                    blend_weight,
                    aggregation_config.v2_min_fallback_blend_weight,
                    aggregation_config.v2_max_fallback_blend_weight,
                )
            )
            alpha = float(np.clip(1.0 - blend_weight, 0.0, 1.0))
            return alpha * agg_ttc_s + (1.0 - alpha) * fallback_ttc_s, "OK_BLEND", len(valid_local_cues)
        return agg_ttc_s, "OK_CUE_ONLY", len(valid_local_cues)

    if fallback_ttc_s is not None:
        return fallback_ttc_s, "OK_FALLBACK", len(valid_local_cues)
    return None, "NO_VALID_FEATURE", len(valid_local_cues)


def predict_ttc_v1(
    bundle: NativeSampleBundle,
    *,
    config: SparsityAwareLtsConfigV1 | None = None,
) -> tuple[PredictionRecordV1, PredictionDebugV1]:
    cfg = config or SparsityAwareLtsConfigV1()
    start_time = time.perf_counter()

    if bundle.interval_sample.gt_ttc_s is None and bundle.interval_sample.status_gt != "OK":
        record = PredictionRecordV1(
            sample_id=bundle.sample_id,
            sequence_id=bundle.sequence_id,
            interval_idx=bundle.interval_sample.interval_idx,
            timestamp_s=bundle.interval_sample.query_t_s,
            gt_ttc_s=bundle.interval_sample.gt_ttc_s,
            ttc_est_s=None,
            e_ttc_pct=None,
            confidence=None,
            status="GT_INVALID",
            cost_time_s=time.perf_counter() - start_time,
            local_cue_valid_count=0,
            selected_cell_count=0,
            fallback_ttc_s=None,
            debug_ref="",
        )
        return record, PredictionDebugV1(None, None, None, tuple())

    try:
        lts = build_lts_v1(bundle, tau_ms=cfg.tau_ms, pad_px=cfg.pad_px, reference_time=cfg.reference_time)
        stats = build_local_stats_v1(lts, grid_rows=cfg.grid_rows, grid_cols=cfg.grid_cols)
        local_conf = build_local_confidence_v1(
            stats,
            config=cfg.confidence,
            selection_threshold=cfg.aggregation.selection_threshold,
            min_selected_cells=cfg.aggregation.min_selected_cells,
        )
        local_cues = _build_local_ttc_cues_v1(bundle, lts, local_conf, cue_config=cfg.cue)
        valid_local_cues = [cue for cue in local_cues if cue.status == "OK" and cue.ttc_s is not None]

        fallback_ttc_s = _fallback_ttc_from_lts(bundle, lts, fallback_config=cfg.fallback)
        ttc_est_s: float | None = None
        status = "NO_VALID_FEATURE"

        if len(valid_local_cues) >= cfg.aggregation.min_valid_local_cues:
            cue_values = np.asarray([cue.ttc_s for cue in valid_local_cues], dtype=np.float64)
            cue_weights = np.asarray(
                [cue.confidence * max(1.0, float(cue.valid_pixel_count)) for cue in valid_local_cues],
                dtype=np.float64,
            )
            agg_ttc_s = _weighted_median(cue_values, cue_weights)
            if cfg.aggregation.blend_with_fallback and fallback_ttc_s is not None:
                alpha = float(np.clip(1.0 - cfg.aggregation.fallback_blend_weight, 0.0, 1.0))
                ttc_est_s = alpha * agg_ttc_s + (1.0 - alpha) * fallback_ttc_s
            else:
                ttc_est_s = agg_ttc_s
            status = "OK"
        elif fallback_ttc_s is not None:
            ttc_est_s = fallback_ttc_s
            status = "OK_FALLBACK"

        e_ttc_pct: float | None = None
        if ttc_est_s is not None and bundle.interval_sample.gt_ttc_s is not None and bundle.interval_sample.gt_ttc_s > 0:
            e_ttc_pct = abs(ttc_est_s - bundle.interval_sample.gt_ttc_s) / bundle.interval_sample.gt_ttc_s * 100.0

        record = PredictionRecordV1(
            sample_id=bundle.sample_id,
            sequence_id=bundle.sequence_id,
            interval_idx=bundle.interval_sample.interval_idx,
            timestamp_s=bundle.interval_sample.query_t_s,
            gt_ttc_s=bundle.interval_sample.gt_ttc_s,
            ttc_est_s=ttc_est_s,
            e_ttc_pct=e_ttc_pct,
            confidence=local_conf.global_confidence,
            status=status,
            cost_time_s=time.perf_counter() - start_time,
            local_cue_valid_count=len(valid_local_cues),
            selected_cell_count=local_conf.selected_cell_count,
            fallback_ttc_s=fallback_ttc_s,
            debug_ref="",
        )
        debug = PredictionDebugV1(
            lts=lts,
            stats=stats,
            local_confidence=local_conf,
            local_cues=local_cues,
        )
        return record, debug
    except Exception as exc:
        record = PredictionRecordV1(
            sample_id=bundle.sample_id,
            sequence_id=bundle.sequence_id,
            interval_idx=bundle.interval_sample.interval_idx,
            timestamp_s=bundle.interval_sample.query_t_s,
            gt_ttc_s=bundle.interval_sample.gt_ttc_s,
            ttc_est_s=None,
            e_ttc_pct=None,
            confidence=None,
            status="RUNTIME_FAIL",
            cost_time_s=time.perf_counter() - start_time,
            local_cue_valid_count=0,
            selected_cell_count=0,
            fallback_ttc_s=None,
            debug_ref=str(exc),
        )
        return record, PredictionDebugV1(None, None, None, tuple())


def predict_ttc_v2(
    bundle: NativeSampleBundle,
    *,
    config: SparsityAwareLtsConfigV1 | None = None,
    cue_config_v2: LocalCueConfigV2 | None = None,
) -> tuple[PredictionRecordV1, PredictionDebugV2]:
    cfg = config or SparsityAwareLtsConfigV1()
    cue_cfg = cue_config_v2 or LocalCueConfigV2()
    start_time = time.perf_counter()

    if bundle.interval_sample.gt_ttc_s is None and bundle.interval_sample.status_gt != "OK":
        record = PredictionRecordV1(
            sample_id=bundle.sample_id,
            sequence_id=bundle.sequence_id,
            interval_idx=bundle.interval_sample.interval_idx,
            timestamp_s=bundle.interval_sample.query_t_s,
            gt_ttc_s=bundle.interval_sample.gt_ttc_s,
            ttc_est_s=None,
            e_ttc_pct=None,
            confidence=None,
            status="GT_INVALID",
            cost_time_s=time.perf_counter() - start_time,
            local_cue_valid_count=0,
            selected_cell_count=0,
            fallback_ttc_s=None,
            debug_ref="",
        )
        return record, PredictionDebugV2(None, None, None, tuple())

    try:
        lts = build_lts_v1(bundle, tau_ms=cfg.tau_ms, pad_px=cfg.pad_px, reference_time=cfg.reference_time)
        stats = build_local_stats_v1(lts, grid_rows=cfg.grid_rows, grid_cols=cfg.grid_cols)
        local_conf = build_local_confidence_v1(
            stats,
            config=cfg.confidence,
            selection_threshold=cfg.aggregation.selection_threshold,
            min_selected_cells=cfg.aggregation.min_selected_cells,
        )
        local_cues = build_local_ttc_cues_v2(bundle, lts, local_conf, cue_config=cue_cfg)
        fallback_ttc_s = _fallback_ttc_from_lts(bundle, lts, fallback_config=cfg.fallback)
        ttc_est_s, status, valid_count = _aggregate_local_cues_v2(
            local_cues,
            aggregation_config=cfg.aggregation,
            fallback_ttc_s=fallback_ttc_s,
        )

        e_ttc_pct: float | None = None
        if ttc_est_s is not None and bundle.interval_sample.gt_ttc_s is not None and bundle.interval_sample.gt_ttc_s > 0:
            e_ttc_pct = abs(ttc_est_s - bundle.interval_sample.gt_ttc_s) / bundle.interval_sample.gt_ttc_s * 100.0

        record = PredictionRecordV1(
            sample_id=bundle.sample_id,
            sequence_id=bundle.sequence_id,
            interval_idx=bundle.interval_sample.interval_idx,
            timestamp_s=bundle.interval_sample.query_t_s,
            gt_ttc_s=bundle.interval_sample.gt_ttc_s,
            ttc_est_s=ttc_est_s,
            e_ttc_pct=e_ttc_pct,
            confidence=local_conf.global_confidence,
            status=status,
            cost_time_s=time.perf_counter() - start_time,
            local_cue_valid_count=valid_count,
            selected_cell_count=local_conf.selected_cell_count,
            fallback_ttc_s=fallback_ttc_s,
            debug_ref="",
        )
        return record, PredictionDebugV2(lts, stats, local_conf, local_cues)
    except Exception as exc:
        record = PredictionRecordV1(
            sample_id=bundle.sample_id,
            sequence_id=bundle.sequence_id,
            interval_idx=bundle.interval_sample.interval_idx,
            timestamp_s=bundle.interval_sample.query_t_s,
            gt_ttc_s=bundle.interval_sample.gt_ttc_s,
            ttc_est_s=None,
            e_ttc_pct=None,
            confidence=None,
            status="RUNTIME_FAIL",
            cost_time_s=time.perf_counter() - start_time,
            local_cue_valid_count=0,
            selected_cell_count=0,
            fallback_ttc_s=None,
            debug_ref=str(exc),
        )
        return record, PredictionDebugV2(None, None, None, tuple())


def predict_ttc_bbox_looming_v1(
    bundle: NativeSampleBundle,
    *,
    config: SparsityAwareLtsConfigV1 | None = None,
) -> tuple[PredictionRecordV1, PredictionDebugV1]:
    cfg = config or SparsityAwareLtsConfigV1()
    start_time = time.perf_counter()

    if bundle.interval_sample.gt_ttc_s is None and bundle.interval_sample.status_gt != "OK":
        record = PredictionRecordV1(
            sample_id=bundle.sample_id,
            sequence_id=bundle.sequence_id,
            interval_idx=bundle.interval_sample.interval_idx,
            timestamp_s=bundle.interval_sample.query_t_s,
            gt_ttc_s=bundle.interval_sample.gt_ttc_s,
            ttc_est_s=None,
            e_ttc_pct=None,
            confidence=None,
            status="GT_INVALID",
            cost_time_s=time.perf_counter() - start_time,
            local_cue_valid_count=0,
            selected_cell_count=0,
            fallback_ttc_s=None,
            debug_ref="",
        )
        return record, PredictionDebugV1(None, None, None, tuple())

    bbox_ttc_s, bbox_status, fit_points, slope_px_per_s, width_ref_px = _bbox_width_looming_ttc(
        bundle,
        bbox_config=cfg.bbox_looming,
    )
    if bbox_ttc_s is not None:
        e_ttc_pct: float | None = None
        gt_ttc_s = bundle.interval_sample.gt_ttc_s
        if gt_ttc_s is not None and gt_ttc_s > 0:
            e_ttc_pct = abs(bbox_ttc_s - gt_ttc_s) / gt_ttc_s * 100.0
        debug_ref = (
            f"fit_points={fit_points};"
            f"slope_px_per_s={slope_px_per_s:.6f};"
            f"width_ref_px={width_ref_px:.6f}"
        )
        record = PredictionRecordV1(
            sample_id=bundle.sample_id,
            sequence_id=bundle.sequence_id,
            interval_idx=bundle.interval_sample.interval_idx,
            timestamp_s=bundle.interval_sample.query_t_s,
            gt_ttc_s=gt_ttc_s,
            ttc_est_s=bbox_ttc_s,
            e_ttc_pct=e_ttc_pct,
            confidence=min(1.0, fit_points / max(float(cfg.bbox_looming.history_back_labels + 1), 1.0)),
            status=bbox_status,
            cost_time_s=time.perf_counter() - start_time,
            local_cue_valid_count=0,
            selected_cell_count=0,
            fallback_ttc_s=None,
            debug_ref=debug_ref,
        )
        return record, PredictionDebugV1(None, None, None, tuple())

    if cfg.bbox_looming.fallback_to_lts_v1:
        fallback_record, fallback_debug = predict_ttc_v1(bundle, config=cfg)
        status = fallback_record.status
        if fallback_record.ttc_est_s is not None and fallback_record.status.startswith("OK"):
            status = "OK_LTS_FALLBACK"
        record = PredictionRecordV1(
            sample_id=fallback_record.sample_id,
            sequence_id=fallback_record.sequence_id,
            interval_idx=fallback_record.interval_idx,
            timestamp_s=fallback_record.timestamp_s,
            gt_ttc_s=fallback_record.gt_ttc_s,
            ttc_est_s=fallback_record.ttc_est_s,
            e_ttc_pct=fallback_record.e_ttc_pct,
            confidence=fallback_record.confidence,
            status=status,
            cost_time_s=time.perf_counter() - start_time,
            local_cue_valid_count=fallback_record.local_cue_valid_count,
            selected_cell_count=fallback_record.selected_cell_count,
            fallback_ttc_s=fallback_record.ttc_est_s,
            debug_ref=f"bbox_status={bbox_status};fallback_status={fallback_record.status}",
        )
        return record, fallback_debug

    record = PredictionRecordV1(
        sample_id=bundle.sample_id,
        sequence_id=bundle.sequence_id,
        interval_idx=bundle.interval_sample.interval_idx,
        timestamp_s=bundle.interval_sample.query_t_s,
        gt_ttc_s=bundle.interval_sample.gt_ttc_s,
        ttc_est_s=None,
        e_ttc_pct=None,
        confidence=None,
        status=bbox_status,
        cost_time_s=time.perf_counter() - start_time,
        local_cue_valid_count=0,
        selected_cell_count=0,
        fallback_ttc_s=None,
        debug_ref=f"fit_points={fit_points}",
    )
    return record, PredictionDebugV1(None, None, None, tuple())


__all__ = [
    "AggregationConfigV1",
    "BBoxLoomingConfigV1",
    "HeuristicFallbackConfigV1",
    "LocalCueConfigV1",
    "LocalCueConfigV2",
    "LocalTtcCueV1",
    "LocalTtcCueV2",
    "PredictionDebugV2",
    "PredictionDebugV1",
    "PredictionRecordV1",
    "SparsityAwareLtsConfigV1",
    "build_local_ttc_cues_v2",
    "predict_ttc_bbox_looming_v1",
    "predict_ttc_v2",
    "predict_ttc_v1",
]
