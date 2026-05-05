from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


def _code_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_float(value: str) -> float:
    return float(value)


def _parse_int(value: str) -> int:
    return int(value)


@dataclass(frozen=True)
class SequenceMeta:
    sequence_id: str
    source_root: str
    raw_h5_path: str
    event_group: str
    frame_ts_path: str
    label_dir: str
    ttc_csv_path: str
    gt_h5_path: str
    bbox_source: str
    gt_ttc_source: str
    label_frame_index_mode: str
    label_id_range: tuple[int, int]
    label_count: int
    ttc_row_count: int
    has_depth: bool
    has_pose: bool
    frame_delta_us_stats: dict[str, float | int | None]

    @property
    def raw_h5(self) -> Path:
        return Path(self.raw_h5_path)


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_name: str
    protocol_version: str
    sequence_time_source: str
    label_time_source: str
    label_frame_index_mode: str
    interval_mode: str
    interval_definition: str
    bbox_mode: str
    bbox_definition: str
    gt_query_mode: str
    gt_query_definition: str
    event_group: str
    event_slice_policy: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Calibration:
    camera_model: str
    distortion_model: str
    intrinsics: dict[str, float]
    resolution: dict[str, int]
    distortion_coeffs: tuple[float, ...]


@dataclass(frozen=True)
class BoundingBoxRecord:
    label_id: int
    frame_idx: int
    timestamp_s: float
    x1: float
    y1: float
    x2: float
    y2: float
    area: float
    category: str
    object_index: int
    json_path: str

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class GtTtcRecord:
    index: int
    time_s: float
    distance_m: float
    velocity_mps: float
    ttc_s: float


@dataclass(frozen=True)
class IntervalSample:
    sample_id: str
    sequence_id: str
    interval_idx: int
    start_label_id: int
    end_label_id: int
    start_frame_idx: int
    end_frame_idx: int
    t_start_s: float
    t_end_s: float
    dt_s: float
    query_t_s: float
    bbox_t_s: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    bbox_area: float
    event_idx_lo: int
    event_idx_hi: int
    event_count: int
    gt_ttc_s: float | None
    status_gt: str
    segmentation_json: str
    category: str

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2)

    @property
    def has_valid_gt(self) -> bool:
        return self.gt_ttc_s is not None and self.status_gt == "OK"


@dataclass(frozen=True)
class EventSlice:
    sequence_id: str
    sample_id: str
    event_idx_lo: int
    event_idx_hi: int
    x: np.ndarray
    y: np.ndarray
    t_us: np.ndarray
    p: np.ndarray

    @property
    def count(self) -> int:
        return int(self.event_idx_hi - self.event_idx_lo)

    @property
    def t_s(self) -> np.ndarray:
        return self.t_us.astype(np.float64) / 1e6

    @property
    def t0_us(self) -> int | None:
        if self.t_us.size == 0:
            return None
        return int(self.t_us[0])

    @property
    def t1_us(self) -> int | None:
        if self.t_us.size == 0:
            return None
        return int(self.t_us[-1])


@dataclass(frozen=True)
class FormalSequencePaths:
    sequence_dir: Path
    sequence_meta: Path
    protocol: Path
    calibration: Path
    bbox_csv: Path
    gt_ttc_csv: Path
    interval_index_csv: Path


def list_formal_sequences(formal_root: Path | None = None) -> list[str]:
    root = (formal_root or (_code_root_from_here() / "DatasetFormal")).resolve()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


class FormalSequenceReader:
    def __init__(
        self,
        sequence_id: str,
        *,
        formal_root: Path | None = None,
    ) -> None:
        self.sequence_id = sequence_id
        self.formal_root = (formal_root or (_code_root_from_here() / "DatasetFormal")).resolve()
        self.paths = self._build_paths(sequence_id)
        self._sequence_meta: SequenceMeta | None = None
        self._protocol: ProtocolSpec | None = None
        self._calibration: Calibration | None = None
        self._bbox_records: list[BoundingBoxRecord] | None = None
        self._gt_ttc_records: list[GtTtcRecord] | None = None
        self._interval_samples: list[IntervalSample] | None = None
        self._event_handle: h5py.File | None = None

    @classmethod
    def from_sequence_dir(cls, sequence_dir: Path) -> FormalSequenceReader:
        return cls(sequence_dir.name, formal_root=sequence_dir.parent)

    def __enter__(self) -> FormalSequenceReader:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._event_handle is not None:
            self._event_handle.close()
            self._event_handle = None

    def _build_paths(self, sequence_id: str) -> FormalSequencePaths:
        sequence_dir = self.formal_root / sequence_id
        return FormalSequencePaths(
            sequence_dir=sequence_dir,
            sequence_meta=sequence_dir / "SequenceMeta.json",
            protocol=sequence_dir / "Protocol.json",
            calibration=sequence_dir / "Calib.json",
            bbox_csv=sequence_dir / "bbox.csv",
            gt_ttc_csv=sequence_dir / "gt_ttc.csv",
            interval_index_csv=sequence_dir / "IntervalIndex.csv",
        )

    def _require_exists(self, path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Missing required formal artifact: {path}")
        return path

    def _load_json(self, path: Path) -> dict[str, Any]:
        return json.loads(self._require_exists(path).read_text(encoding="utf-8"))

    @property
    def sequence_meta(self) -> SequenceMeta:
        if self._sequence_meta is None:
            payload = self._load_json(self.paths.sequence_meta)
            self._sequence_meta = SequenceMeta(
                sequence_id=str(payload["sequence_id"]),
                source_root=str(payload["source_root"]),
                raw_h5_path=str(payload["raw_h5_path"]),
                event_group=str(payload["event_group"]),
                frame_ts_path=str(payload["frame_ts_path"]),
                label_dir=str(payload["label_dir"]),
                ttc_csv_path=str(payload["ttc_csv_path"]),
                gt_h5_path=str(payload["gt_h5_path"]),
                bbox_source=str(payload["bbox_source"]),
                gt_ttc_source=str(payload["gt_ttc_source"]),
                label_frame_index_mode=str(payload["label_frame_index_mode"]),
                label_id_range=(
                    int(payload["label_id_range"][0]),
                    int(payload["label_id_range"][1]),
                ),
                label_count=int(payload["label_count"]),
                ttc_row_count=int(payload["ttc_row_count"]),
                has_depth=_parse_bool(payload["has_depth"]),
                has_pose=_parse_bool(payload["has_pose"]),
                frame_delta_us_stats=dict(payload["frame_delta_us_stats"]),
            )
        return self._sequence_meta

    @property
    def protocol(self) -> ProtocolSpec:
        if self._protocol is None:
            payload = self._load_json(self.paths.protocol)
            self._protocol = ProtocolSpec(
                protocol_name=str(payload["protocol_name"]),
                protocol_version=str(payload["protocol_version"]),
                sequence_time_source=str(payload["sequence_time_source"]),
                label_time_source=str(payload["label_time_source"]),
                label_frame_index_mode=str(payload["label_frame_index_mode"]),
                interval_mode=str(payload["interval_mode"]),
                interval_definition=str(payload["interval_definition"]),
                bbox_mode=str(payload["bbox_mode"]),
                bbox_definition=str(payload["bbox_definition"]),
                gt_query_mode=str(payload["gt_query_mode"]),
                gt_query_definition=str(payload["gt_query_definition"]),
                event_group=str(payload["event_group"]),
                event_slice_policy=str(payload["event_slice_policy"]),
                notes=tuple(str(item) for item in payload.get("notes", [])),
            )
        return self._protocol

    @property
    def calibration(self) -> Calibration:
        if self._calibration is None:
            payload = self._load_json(self.paths.calibration)
            self._calibration = Calibration(
                camera_model=str(payload["camera_model"]),
                distortion_model=str(payload["distortion_model"]),
                intrinsics={key: float(value) for key, value in payload["intrinsics"].items()},
                resolution={key: int(value) for key, value in payload["resolution"].items()},
                distortion_coeffs=tuple(float(value) for value in payload["distortion_coeffs"]),
            )
        return self._calibration

    @property
    def bbox_records(self) -> list[BoundingBoxRecord]:
        if self._bbox_records is None:
            rows: list[BoundingBoxRecord] = []
            with self._require_exists(self.paths.bbox_csv).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows.append(
                        BoundingBoxRecord(
                            label_id=_parse_int(row["label_id"]),
                            frame_idx=_parse_int(row["frame_idx"]),
                            timestamp_s=_parse_float(row["timestamp_s"]),
                            x1=_parse_float(row["x1"]),
                            y1=_parse_float(row["y1"]),
                            x2=_parse_float(row["x2"]),
                            y2=_parse_float(row["y2"]),
                            area=_parse_float(row["area"]),
                            category=str(row["category"]),
                            object_index=_parse_int(row["object_index"]),
                            json_path=str(row["json_path"]),
                        )
                    )
            self._bbox_records = rows
        return self._bbox_records

    @property
    def gt_ttc_records(self) -> list[GtTtcRecord]:
        if self._gt_ttc_records is None:
            rows: list[GtTtcRecord] = []
            with self._require_exists(self.paths.gt_ttc_csv).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows.append(
                        GtTtcRecord(
                            index=_parse_int(row["index"]),
                            time_s=_parse_float(row["time_s"]),
                            distance_m=_parse_float(row["distance_m"]),
                            velocity_mps=_parse_float(row["velocity_mps"]),
                            ttc_s=_parse_float(row["ttc_s"]),
                        )
                    )
            self._gt_ttc_records = rows
        return self._gt_ttc_records

    @property
    def interval_samples(self) -> list[IntervalSample]:
        if self._interval_samples is None:
            rows: list[IntervalSample] = []
            with self._require_exists(self.paths.interval_index_csv).open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    gt_ttc_s_raw = row["gt_ttc_s"].strip()
                    rows.append(
                        IntervalSample(
                            sample_id=str(row["sample_id"]),
                            sequence_id=str(row["sequence_id"]),
                            interval_idx=_parse_int(row["interval_idx"]),
                            start_label_id=_parse_int(row["start_label_id"]),
                            end_label_id=_parse_int(row["end_label_id"]),
                            start_frame_idx=_parse_int(row["start_frame_idx"]),
                            end_frame_idx=_parse_int(row["end_frame_idx"]),
                            t_start_s=_parse_float(row["t_start_s"]),
                            t_end_s=_parse_float(row["t_end_s"]),
                            dt_s=_parse_float(row["dt_s"]),
                            query_t_s=_parse_float(row["query_t_s"]),
                            bbox_t_s=_parse_float(row["bbox_t_s"]),
                            bbox_x1=_parse_float(row["bbox_x1"]),
                            bbox_y1=_parse_float(row["bbox_y1"]),
                            bbox_x2=_parse_float(row["bbox_x2"]),
                            bbox_y2=_parse_float(row["bbox_y2"]),
                            bbox_area=_parse_float(row["bbox_area"]),
                            event_idx_lo=_parse_int(row["event_idx_lo"]),
                            event_idx_hi=_parse_int(row["event_idx_hi"]),
                            event_count=_parse_int(row["event_count"]),
                            gt_ttc_s=None if not gt_ttc_s_raw else _parse_float(gt_ttc_s_raw),
                            status_gt=str(row["status_gt"]),
                            segmentation_json=str(row["segmentation_json"]),
                            category=str(row["category"]),
                        )
                    )
            self._interval_samples = rows
        return self._interval_samples

    def iter_interval_samples(self) -> Iterator[IntervalSample]:
        yield from self.interval_samples

    def get_interval_sample(
        self,
        *,
        interval_idx: int | None = None,
        sample_id: str | None = None,
    ) -> IntervalSample:
        if interval_idx is None and sample_id is None:
            raise ValueError("Provide either interval_idx or sample_id")
        for sample in self.interval_samples:
            if interval_idx is not None and sample.interval_idx == interval_idx:
                return sample
            if sample_id is not None and sample.sample_id == sample_id:
                return sample
        raise KeyError(
            f"Interval sample not found for sequence={self.sequence_id}, "
            f"interval_idx={interval_idx}, sample_id={sample_id}"
        )

    def _event_group(self) -> h5py.Group:
        if self._event_handle is None:
            raw_h5_path = Path(self.sequence_meta.raw_h5_path)
            if not raw_h5_path.is_absolute():
                raw_h5_path = self.paths.sequence_dir / raw_h5_path
            self._event_handle = h5py.File(raw_h5_path, "r")
        return self._event_handle[self.sequence_meta.event_group]

    def read_event_slice(
        self,
        *,
        interval_idx: int | None = None,
        sample_id: str | None = None,
        sample: IntervalSample | None = None,
    ) -> EventSlice:
        if sample is None:
            sample = self.get_interval_sample(interval_idx=interval_idx, sample_id=sample_id)
        group = self._event_group()
        lo = sample.event_idx_lo
        hi = sample.event_idx_hi
        return EventSlice(
            sequence_id=self.sequence_id,
            sample_id=sample.sample_id,
            event_idx_lo=lo,
            event_idx_hi=hi,
            x=group["x"][lo:hi],
            y=group["y"][lo:hi],
            t_us=group["t"][lo:hi],
            p=group["p"][lo:hi],
        )

    def summary(self) -> dict[str, Any]:
        samples = self.interval_samples
        total_events = sum(sample.event_count for sample in samples)
        valid_gt = sum(1 for sample in samples if sample.has_valid_gt)
        return {
            "sequence_id": self.sequence_id,
            "formal_root": str(self.formal_root),
            "interval_count": len(samples),
            "valid_gt_count": valid_gt,
            "label_count": self.sequence_meta.label_count,
            "bbox_count": len(self.bbox_records),
            "gt_ttc_count": len(self.gt_ttc_records),
            "event_count_total": total_events,
            "protocol_version": self.protocol.protocol_version,
            "resolution": self.calibration.resolution,
        }


__all__ = [
    "BoundingBoxRecord",
    "Calibration",
    "EventSlice",
    "FormalSequencePaths",
    "FormalSequenceReader",
    "GtTtcRecord",
    "IntervalSample",
    "ProtocolSpec",
    "SequenceMeta",
    "list_formal_sequences",
]
