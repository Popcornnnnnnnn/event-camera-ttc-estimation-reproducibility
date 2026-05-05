from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .representations import LocalStatsV1


@dataclass(frozen=True)
class ConfidenceConfigV1:
    target_events_per_cell: float = 200.0
    min_occupancy_ratio: float = 0.02
    good_occupancy_ratio: float = 0.20
    min_mean_surface: float = 0.10
    good_mean_surface: float = 0.50
    weight_event: float = 0.35
    weight_occupancy: float = 0.30
    weight_freshness: float = 0.35


@dataclass(frozen=True)
class LocalConfidenceV1:
    sequence_id: str
    sample_id: str
    interval_idx: int
    grid_rows: int
    grid_cols: int
    event_score_grid: np.ndarray
    occupancy_score_grid: np.ndarray
    freshness_score_grid: np.ndarray
    confidence_grid: np.ndarray
    active_mask: np.ndarray
    global_confidence: float
    selected_cell_indices: tuple[tuple[int, int], ...]

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.grid_rows, self.grid_cols)

    @property
    def selected_cell_count(self) -> int:
        return len(self.selected_cell_indices)


def _normalized_log_score(values: np.ndarray, target_value: float) -> np.ndarray:
    target = max(1e-6, float(target_value))
    return np.clip(np.log1p(values.astype(np.float32)) / np.log1p(target), 0.0, 1.0)


def _linear_band_score(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError(f"Invalid score band: low={low}, high={high}")
    scores = (values.astype(np.float32) - float(low)) / float(high - low)
    return np.clip(scores, 0.0, 1.0)


def build_local_confidence_v1(
    stats: LocalStatsV1,
    *,
    config: ConfidenceConfigV1 | None = None,
    selection_threshold: float = 0.55,
    min_selected_cells: int = 1,
) -> LocalConfidenceV1:
    cfg = config or ConfidenceConfigV1()

    event_score = _normalized_log_score(
        stats.event_count_grid,
        target_value=cfg.target_events_per_cell,
    )
    occupancy_score = _linear_band_score(
        stats.occupancy_ratio_grid,
        low=cfg.min_occupancy_ratio,
        high=cfg.good_occupancy_ratio,
    )
    freshness_score = _linear_band_score(
        stats.mean_surface_grid,
        low=cfg.min_mean_surface,
        high=cfg.good_mean_surface,
    )

    total_weight = cfg.weight_event + cfg.weight_occupancy + cfg.weight_freshness
    if total_weight <= 0:
        raise ValueError("Confidence weights must sum to a positive value")

    confidence_grid = (
        cfg.weight_event * event_score
        + cfg.weight_occupancy * occupancy_score
        + cfg.weight_freshness * freshness_score
    ) / total_weight

    active_mask = stats.event_count_grid > 0
    candidate_mask = active_mask & (confidence_grid >= float(selection_threshold))

    selected: list[tuple[int, int]] = [
        (int(row), int(col))
        for row, col in np.argwhere(candidate_mask)
    ]

    if len(selected) < min_selected_cells:
        active_positions = np.argwhere(active_mask)
        if active_positions.size > 0:
            order = np.argsort(confidence_grid[active_mask])[::-1]
            top_positions = active_positions[order[:min_selected_cells]]
            selected = [(int(row), int(col)) for row, col in top_positions]

    global_confidence = 0.0
    if np.any(active_mask):
        weights = np.maximum(stats.event_count_grid.astype(np.float32), 1.0)
        global_confidence = float(
            (confidence_grid * weights * active_mask.astype(np.float32)).sum()
            / (weights * active_mask.astype(np.float32)).sum()
        )

    return LocalConfidenceV1(
        sequence_id=stats.sequence_id,
        sample_id=stats.sample_id,
        interval_idx=stats.interval_idx,
        grid_rows=stats.grid_rows,
        grid_cols=stats.grid_cols,
        event_score_grid=event_score,
        occupancy_score_grid=occupancy_score,
        freshness_score_grid=freshness_score,
        confidence_grid=confidence_grid.astype(np.float32),
        active_mask=active_mask,
        global_confidence=global_confidence,
        selected_cell_indices=tuple(selected),
    )


def summarize_local_confidence(
    confidence: LocalConfidenceV1,
    *,
    include_component_grids: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sequence_id": confidence.sequence_id,
        "sample_id": confidence.sample_id,
        "interval_idx": confidence.interval_idx,
        "grid_shape": [confidence.grid_rows, confidence.grid_cols],
        "global_confidence": confidence.global_confidence,
        "selected_cell_indices": [list(item) for item in confidence.selected_cell_indices],
        "selected_cell_count": confidence.selected_cell_count,
        "confidence_grid": confidence.confidence_grid.tolist(),
    }
    if include_component_grids:
        payload["event_score_grid"] = confidence.event_score_grid.tolist()
        payload["occupancy_score_grid"] = confidence.occupancy_score_grid.tolist()
        payload["freshness_score_grid"] = confidence.freshness_score_grid.tolist()
        payload["active_mask"] = confidence.active_mask.astype(np.uint8).tolist()
    return payload


__all__ = [
    "ConfidenceConfigV1",
    "LocalConfidenceV1",
    "build_local_confidence_v1",
    "summarize_local_confidence",
]
