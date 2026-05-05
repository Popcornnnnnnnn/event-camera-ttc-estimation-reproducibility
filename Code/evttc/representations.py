from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .native_adapter import NativeSampleBundle


@dataclass(frozen=True)
class RoiBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class LtsV1:
    sequence_id: str
    sample_id: str
    interval_idx: int
    roi: RoiBox
    reference_t_us: int
    tau_us: int
    roi_event_count: int
    count_map: np.ndarray
    occupancy_map: np.ndarray
    last_timestamp_us: np.ndarray
    age_us: np.ndarray
    time_surface: np.ndarray
    valid_mask: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.time_surface.shape

    @property
    def active_pixel_count(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def occupancy_ratio(self) -> float:
        if self.roi.area == 0:
            return 0.0
        return float(self.active_pixel_count) / float(self.roi.area)

    @property
    def mean_surface(self) -> float:
        if not np.any(self.valid_mask):
            return 0.0
        return float(self.time_surface[self.valid_mask].mean())


@dataclass(frozen=True)
class LocalStatsV1:
    sequence_id: str
    sample_id: str
    interval_idx: int
    grid_rows: int
    grid_cols: int
    event_count_grid: np.ndarray
    occupied_pixel_grid: np.ndarray
    occupancy_ratio_grid: np.ndarray
    mean_surface_grid: np.ndarray
    max_surface_grid: np.ndarray
    min_surface_grid: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.grid_rows, self.grid_cols)


def _clip_bbox_to_roi(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    pad_px: int = 0,
) -> RoiBox:
    x1, y1, x2, y2 = bbox_xyxy
    x0_i = max(0, int(np.floor(x1)) - pad_px)
    y0_i = max(0, int(np.floor(y1)) - pad_px)
    x1_i = min(width, int(np.ceil(x2)) + pad_px)
    y1_i = min(height, int(np.ceil(y2)) + pad_px)
    return RoiBox(x0=x0_i, y0=y0_i, x1=x1_i, y1=y1_i)


def build_lts_v1(
    bundle: NativeSampleBundle,
    *,
    tau_ms: float = 20.0,
    pad_px: int = 0,
    reference_time: str = "end",
) -> LtsV1:
    if bundle.event_slice is None:
        raise ValueError("build_lts_v1 requires bundle.event_slice; call adapter with include_events=True")

    resolution = bundle.calibration.resolution
    roi = _clip_bbox_to_roi(
        bundle.interval_sample.bbox_xyxy,
        width=int(resolution["width"]),
        height=int(resolution["height"]),
        pad_px=pad_px,
    )
    h, w = roi.height, roi.width
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid ROI for {bundle.sample_id}: {roi.xyxy}")

    x = bundle.event_slice.x
    y = bundle.event_slice.y
    t_us = bundle.event_slice.t_us
    inside = (
        (x >= roi.x0)
        & (x < roi.x1)
        & (y >= roi.y0)
        & (y < roi.y1)
    )

    x_roi = x[inside].astype(np.int64) - roi.x0
    y_roi = y[inside].astype(np.int64) - roi.y0
    t_roi = t_us[inside].astype(np.int64)

    count_map = np.zeros((h, w), dtype=np.int32)
    last_timestamp_us = np.full((h, w), -1, dtype=np.int64)
    if x_roi.size:
        np.add.at(count_map, (y_roi, x_roi), 1)
        linear_idx = y_roi * w + x_roi
        np.maximum.at(last_timestamp_us.reshape(-1), linear_idx, t_roi)

    valid_mask = last_timestamp_us >= 0
    occupancy_map = valid_mask.astype(np.uint8)

    if reference_time == "end":
        reference_t_us = int(round(bundle.interval_sample.t_end_s * 1e6))
    elif reference_time == "last_event":
        reference_t_us = int(t_roi.max()) if t_roi.size else int(round(bundle.interval_sample.t_end_s * 1e6))
    else:
        raise ValueError(f"Unsupported reference_time={reference_time!r}")

    age_us = np.zeros((h, w), dtype=np.int64)
    age_us[valid_mask] = reference_t_us - last_timestamp_us[valid_mask]

    tau_us = max(1, int(round(tau_ms * 1000.0)))
    time_surface = np.zeros((h, w), dtype=np.float32)
    if np.any(valid_mask):
        time_surface[valid_mask] = np.exp(-age_us[valid_mask].astype(np.float32) / float(tau_us))

    return LtsV1(
        sequence_id=bundle.sequence_meta.sequence_id,
        sample_id=bundle.sample_id,
        interval_idx=bundle.interval_sample.interval_idx,
        roi=roi,
        reference_t_us=reference_t_us,
        tau_us=tau_us,
        roi_event_count=int(t_roi.size),
        count_map=count_map,
        occupancy_map=occupancy_map,
        last_timestamp_us=last_timestamp_us,
        age_us=age_us,
        time_surface=time_surface,
        valid_mask=valid_mask,
    )


def build_local_stats_v1(
    lts: LtsV1,
    *,
    grid_rows: int = 2,
    grid_cols: int = 2,
) -> LocalStatsV1:
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")

    h, w = lts.shape
    y_edges = np.linspace(0, h, grid_rows + 1, dtype=np.int64)
    x_edges = np.linspace(0, w, grid_cols + 1, dtype=np.int64)

    event_count_grid = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    occupied_pixel_grid = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    occupancy_ratio_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    mean_surface_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    max_surface_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    min_surface_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    for row in range(grid_rows):
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        for col in range(grid_cols):
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            cell_count_map = lts.count_map[y0:y1, x0:x1]
            cell_valid = lts.valid_mask[y0:y1, x0:x1]
            cell_surface = lts.time_surface[y0:y1, x0:x1]
            cell_area = max(1, (y1 - y0) * (x1 - x0))
            active_pixels = int(cell_valid.sum())

            event_count_grid[row, col] = int(cell_count_map.sum())
            occupied_pixel_grid[row, col] = active_pixels
            occupancy_ratio_grid[row, col] = float(active_pixels) / float(cell_area)

            if active_pixels > 0:
                active_surface = cell_surface[cell_valid]
                mean_surface_grid[row, col] = float(active_surface.mean())
                max_surface_grid[row, col] = float(active_surface.max())
                min_surface_grid[row, col] = float(active_surface.min())

    return LocalStatsV1(
        sequence_id=lts.sequence_id,
        sample_id=lts.sample_id,
        interval_idx=lts.interval_idx,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        event_count_grid=event_count_grid,
        occupied_pixel_grid=occupied_pixel_grid,
        occupancy_ratio_grid=occupancy_ratio_grid,
        mean_surface_grid=mean_surface_grid,
        max_surface_grid=max_surface_grid,
        min_surface_grid=min_surface_grid,
    )


def summarize_representation(lts: LtsV1, stats: LocalStatsV1 | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sequence_id": lts.sequence_id,
        "sample_id": lts.sample_id,
        "interval_idx": lts.interval_idx,
        "roi_xyxy": list(lts.roi.xyxy),
        "roi_shape_hw": list(lts.shape),
        "roi_area": lts.roi.area,
        "roi_event_count": lts.roi_event_count,
        "active_pixel_count": lts.active_pixel_count,
        "occupancy_ratio": lts.occupancy_ratio,
        "mean_surface": lts.mean_surface,
        "tau_us": lts.tau_us,
        "reference_t_us": lts.reference_t_us,
    }
    if stats is not None:
        payload["local_stats"] = {
            "grid_shape": [stats.grid_rows, stats.grid_cols],
            "event_count_grid": stats.event_count_grid.tolist(),
            "occupancy_ratio_grid": stats.occupancy_ratio_grid.tolist(),
            "mean_surface_grid": stats.mean_surface_grid.tolist(),
        }
    return payload


__all__ = [
    "LocalStatsV1",
    "LtsV1",
    "RoiBox",
    "build_local_stats_v1",
    "build_lts_v1",
    "summarize_representation",
]
