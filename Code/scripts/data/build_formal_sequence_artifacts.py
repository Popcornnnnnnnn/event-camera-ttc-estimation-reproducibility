#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REQUIRED_OUTPUT_FILES = {
    "json": ["SequenceMeta.json", "Protocol.json", "Calib.json"],
    "csv": ["bbox.csv", "gt_ttc.csv", "IntervalIndex.csv"],
}


@dataclass
class LabelRecord:
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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Build unified Phase 2 artifacts for EvTTC formal sequences: "
            "SequenceMeta.json, Protocol.json, Calib.json, bbox.csv, gt_ttc.csv, IntervalIndex.csv."
        )
    )
    parser.add_argument(
        "--dataset-full-root",
        type=Path,
        default=repo_root / "Dataset-Full",
        help="Root directory containing raw full EvTTC sequences.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "Code" / "DatasetFormal",
        help="Directory for generated formal sequence artifacts.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional list of sequence names. Default: all subdirectories in dataset root.",
    )
    parser.add_argument(
        "--event-group",
        default="prophesee/event_cam_left",
        help="Event group path in source HDF5.",
    )
    parser.add_argument(
        "--frame-ts-path",
        default="blackflys/left/ts",
        help="Left image frame timestamp dataset path in source HDF5.",
    )
    parser.add_argument(
        "--gt-query-mode",
        choices=["start"],
        default="start",
        help="GT TTC query time policy. v0.1 only supports start.",
    )
    parser.add_argument(
        "--bbox-mode",
        choices=["start"],
        default="start",
        help="BBox selection policy. v0.1 only supports start.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Rebuild sequence outputs even when an existing artifact set looks complete.",
    )
    return parser.parse_args()


def lower_bound(ds: h5py.Dataset, target: int, lo: int = 0, hi: int | None = None) -> int:
    if hi is None:
        hi = ds.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if int(ds[mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def find_raw_h5(sequence_root: Path) -> Path:
    candidates = sorted(
        path
        for path in sequence_root.glob("*.hdf5")
        if path.name != "gt.hdf5"
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"{sequence_root.name}: expected exactly one raw .hdf5 excluding gt.hdf5, got {len(candidates)}"
        )
    return candidates[0]


def load_label_records(label_dir: Path, frame_ts_us: np.ndarray) -> list[LabelRecord]:
    records: list[LabelRecord] = []
    label_files = sorted(path for path in label_dir.glob("*.json") if path.name != "isat.yaml")
    if not label_files:
        raise RuntimeError(f"{label_dir}: no label json files found")

    for json_path in label_files:
        frame_idx = int(json_path.stem)
        if frame_idx < 0 or frame_idx >= len(frame_ts_us):
            raise RuntimeError(
                f"{json_path.name}: frame index {frame_idx} out of range for left frame ts length {len(frame_ts_us)}"
            )

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        objects = payload.get("objects", [])
        if not objects:
            raise RuntimeError(f"{json_path.name}: no objects found")

        pick_idx = next((idx for idx, obj in enumerate(objects) if obj.get("category") == "car"), 0)
        obj = objects[pick_idx]
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError(f"{json_path.name}: invalid bbox field")

        x1, y1, x2, y2 = map(float, bbox)
        area = float(obj.get("area", max(0.0, x2 - x1) * max(0.0, y2 - y1)))
        records.append(
            LabelRecord(
                label_id=frame_idx,
                frame_idx=frame_idx,
                timestamp_s=float(frame_ts_us[frame_idx]) / 1e6,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                area=area,
                category=str(obj.get("category", "unknown")),
                object_index=pick_idx,
                json_path=str(json_path.resolve()),
            )
        )

    return records


def load_ttc_rows(ttc_csv: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with ttc_csv.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for raw in reader:
            if len(raw) < 5:
                continue
            rows.append(
                {
                    "index": int(raw[0]),
                    "time_s": float(raw[1]),
                    "distance_m": float(raw[2]),
                    "velocity_mps": float(raw[3]),
                    "ttc_s": float(raw[4]),
                }
            )
    if not rows:
        raise RuntimeError(f"{ttc_csv}: no valid TTC rows found")
    return rows


def interpolate_ttc(
    query_t_s: float,
    ttc_times: np.ndarray,
    ttc_values: np.ndarray,
) -> tuple[float | None, str]:
    if query_t_s < float(ttc_times[0]) or query_t_s > float(ttc_times[-1]):
        return None, "OUT_OF_RANGE"

    idx = int(np.searchsorted(ttc_times, query_t_s))
    if idx == 0:
        return float(ttc_values[0]), "OK"
    if idx >= len(ttc_times):
        return float(ttc_values[-1]), "OK"

    t0 = float(ttc_times[idx - 1])
    t1 = float(ttc_times[idx])
    v0 = float(ttc_values[idx - 1])
    v1 = float(ttc_values[idx])
    if abs(t1 - t0) < 1e-12:
        return v0, "OK"

    alpha = (query_t_s - t0) / (t1 - t0)
    value = v0 + alpha * (v1 - v0)
    return value, "OK"


def build_sequence_meta(
    sequence_name: str,
    sequence_root: Path,
    raw_h5_path: Path,
    event_group: str,
    frame_ts_path: str,
    label_records: list[LabelRecord],
    ttc_rows: list[dict[str, float | int]],
    frame_deltas_us: np.ndarray,
) -> dict[str, Any]:
    return {
        "sequence_id": sequence_name,
        "source_root": str(sequence_root.resolve()),
        "raw_h5_path": str(raw_h5_path.resolve()),
        "event_group": event_group,
        "frame_ts_path": frame_ts_path,
        "label_dir": str((sequence_root / "leftlabel").resolve()),
        "ttc_csv_path": str((sequence_root / "ttc.csv").resolve()),
        "gt_h5_path": str((sequence_root / "gt.hdf5").resolve()),
        "bbox_source": "leftlabel/*.json bbox field on left blackfly frames",
        "gt_ttc_source": "ttc.csv linear interpolation",
        "label_frame_index_mode": "label_stem_as_blackfly_left_index",
        "label_id_range": [label_records[0].label_id, label_records[-1].label_id],
        "label_count": len(label_records),
        "ttc_row_count": len(ttc_rows),
        "has_depth": True,
        "has_pose": True,
        "frame_delta_us_stats": {
            "mean": float(frame_deltas_us.mean()) if len(frame_deltas_us) else None,
            "min": int(frame_deltas_us.min()) if len(frame_deltas_us) else None,
            "max": int(frame_deltas_us.max()) if len(frame_deltas_us) else None,
        },
    }


def build_protocol(
    event_group: str,
    frame_ts_path: str,
    gt_query_mode: str,
    bbox_mode: str,
) -> dict[str, Any]:
    return {
        "protocol_name": "EvTTC-Formal-v0.1",
        "protocol_version": "v0.1.0",
        "sequence_time_source": frame_ts_path,
        "label_time_source": "blackflys/left/ts[label_id]",
        "label_frame_index_mode": "label_stem_as_blackfly_left_index",
        "interval_mode": "adjacent_label_frames",
        "interval_definition": "[t_i, t_{i+1})",
        "bbox_mode": bbox_mode,
        "bbox_definition": "start label bbox",
        "gt_query_mode": gt_query_mode,
        "gt_query_definition": "query TTC at t_start_s using linear interpolation over ttc.csv",
        "event_group": event_group,
        "event_slice_policy": "store event index range only, lazy load on read",
        "notes": [
            "leftlabel json filenames align with blackfly left frame indices",
            "blackfly left frame interval is approximately 50 ms with small jitter",
        ],
    }


def build_calib(handle: h5py.File, event_group: str) -> dict[str, Any]:
    group = handle[event_group]
    calib = group["calib"]
    intrinsics = calib["intrinsics"][:].tolist()
    resolution = calib["resolution"][:].tolist()
    distortion = calib["distortion_coeffs"][:].tolist()
    return {
        "camera_model": calib["camera_model"][()].decode() if hasattr(calib["camera_model"][()], "decode") else str(calib["camera_model"][()]),
        "distortion_model": calib["distortion_model"][()].decode() if hasattr(calib["distortion_model"][()], "decode") else str(calib["distortion_model"][()]),
        "intrinsics": {
            "fx": float(intrinsics[0]),
            "fy": float(intrinsics[1]),
            "cx": float(intrinsics[2]),
            "cy": float(intrinsics[3]),
        },
        "resolution": {
            "width": int(resolution[0]),
            "height": int(resolution[1]),
        },
        "distortion_coeffs": [float(value) for value in distortion],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def is_nonempty_json(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return True


def is_nonempty_csv(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        line_count = sum(1 for _ in handle)
    return line_count >= 2


def inspect_sequence_output(sequence_out: Path) -> tuple[str, list[str]]:
    invalid: list[str] = []
    for name in REQUIRED_OUTPUT_FILES["json"]:
        if not is_nonempty_json(sequence_out / name):
            invalid.append(name)
    for name in REQUIRED_OUTPUT_FILES["csv"]:
        if not is_nonempty_csv(sequence_out / name):
            invalid.append(name)

    if not invalid:
        return "complete", []
    if sequence_out.exists():
        return "partial", invalid
    return "missing", invalid


def build_sequence(sequence_root: Path, output_root: Path, args: argparse.Namespace) -> None:
    sequence_name = sequence_root.name
    raw_h5_path = find_raw_h5(sequence_root)
    label_dir = sequence_root / "leftlabel"
    ttc_csv = sequence_root / "ttc.csv"

    if not label_dir.exists():
        raise RuntimeError(f"{sequence_name}: missing leftlabel directory")
    if not ttc_csv.exists():
        raise RuntimeError(f"{sequence_name}: missing ttc.csv")

    sequence_out = output_root / sequence_name
    state, invalid = inspect_sequence_output(sequence_out)
    if state == "complete" and not args.overwrite_existing:
        print(f"[Skip] {sequence_name}: existing artifacts already complete -> {sequence_out}")
        return

    if state == "partial":
        print(
            f"[Resume] {sequence_name}: found incomplete artifacts "
            f"({', '.join(invalid)}), rebuilding -> {sequence_out}"
        )
    elif state == "missing":
        print(f"[Build] {sequence_name}: generating new artifacts -> {sequence_out}")
    else:
        print(f"[Rebuild] {sequence_name}: overwrite requested -> {sequence_out}")

    sequence_out.mkdir(parents=True, exist_ok=True)

    with h5py.File(raw_h5_path, "r") as handle:
        frame_ts_us = handle[args.frame_ts_path][:].reshape(-1)
        event_t_ds = handle[args.event_group]["t"]

        label_records = load_label_records(label_dir, frame_ts_us)
        if len(label_records) < 2:
            raise RuntimeError(f"{sequence_name}: need at least 2 labels to build intervals")

        ttc_rows = load_ttc_rows(ttc_csv)
        ttc_times = np.asarray([row["time_s"] for row in ttc_rows], dtype=np.float64)
        ttc_values = np.asarray([row["ttc_s"] for row in ttc_rows], dtype=np.float64)

        bbox_rows = [asdict(record) for record in label_records]
        gt_rows = ttc_rows

        interval_rows: list[dict[str, Any]] = []
        search_lo = 0
        for interval_idx in range(len(label_records) - 1):
            cur = label_records[interval_idx]
            nxt = label_records[interval_idx + 1]
            t_start_s = cur.timestamp_s
            t_end_s = nxt.timestamp_s
            start_us = int(round(t_start_s * 1e6))
            end_us = int(round(t_end_s * 1e6))
            event_idx_lo = lower_bound(event_t_ds, start_us, lo=search_lo)
            event_idx_hi = lower_bound(event_t_ds, end_us, lo=event_idx_lo)
            search_lo = event_idx_hi

            gt_ttc_s, status_gt = interpolate_ttc(
                query_t_s=t_start_s,
                ttc_times=ttc_times,
                ttc_values=ttc_values,
            )

            interval_rows.append(
                {
                    "sample_id": f"{sequence_name}_{interval_idx:04d}",
                    "sequence_id": sequence_name,
                    "interval_idx": interval_idx,
                    "start_label_id": cur.label_id,
                    "end_label_id": nxt.label_id,
                    "start_frame_idx": cur.frame_idx,
                    "end_frame_idx": nxt.frame_idx,
                    "t_start_s": f"{t_start_s:.6f}",
                    "t_end_s": f"{t_end_s:.6f}",
                    "dt_s": f"{(t_end_s - t_start_s):.6f}",
                    "query_t_s": f"{t_start_s:.6f}",
                    "bbox_t_s": f"{t_start_s:.6f}",
                    "bbox_x1": f"{cur.x1:.3f}",
                    "bbox_y1": f"{cur.y1:.3f}",
                    "bbox_x2": f"{cur.x2:.3f}",
                    "bbox_y2": f"{cur.y2:.3f}",
                    "bbox_area": f"{cur.area:.3f}",
                    "event_idx_lo": event_idx_lo,
                    "event_idx_hi": event_idx_hi,
                    "event_count": max(0, event_idx_hi - event_idx_lo),
                    "gt_ttc_s": "" if gt_ttc_s is None else f"{gt_ttc_s:.6f}",
                    "status_gt": status_gt,
                    "segmentation_json": cur.json_path,
                    "category": cur.category,
                }
            )

        frame_deltas_us = frame_ts_us[1:] - frame_ts_us[:-1]
        sequence_meta = build_sequence_meta(
            sequence_name=sequence_name,
            sequence_root=sequence_root,
            raw_h5_path=raw_h5_path,
            event_group=args.event_group,
            frame_ts_path=args.frame_ts_path,
            label_records=label_records,
            ttc_rows=ttc_rows,
            frame_deltas_us=frame_deltas_us,
        )
        protocol = build_protocol(
            event_group=args.event_group,
            frame_ts_path=args.frame_ts_path,
            gt_query_mode=args.gt_query_mode,
            bbox_mode=args.bbox_mode,
        )
        calib = build_calib(handle, args.event_group)

    write_json_atomic(sequence_out / "SequenceMeta.json", sequence_meta)
    write_json_atomic(sequence_out / "Protocol.json", protocol)
    write_json_atomic(sequence_out / "Calib.json", calib)

    write_csv_atomic(
        sequence_out / "bbox.csv",
        bbox_rows,
        fieldnames=[
            "label_id",
            "frame_idx",
            "timestamp_s",
            "x1",
            "y1",
            "x2",
            "y2",
            "area",
            "category",
            "object_index",
            "json_path",
        ],
    )
    write_csv_atomic(
        sequence_out / "gt_ttc.csv",
        gt_rows,
        fieldnames=["index", "time_s", "distance_m", "velocity_mps", "ttc_s"],
    )
    write_csv_atomic(
        sequence_out / "IntervalIndex.csv",
        interval_rows,
        fieldnames=[
            "sample_id",
            "sequence_id",
            "interval_idx",
            "start_label_id",
            "end_label_id",
            "start_frame_idx",
            "end_frame_idx",
            "t_start_s",
            "t_end_s",
            "dt_s",
            "query_t_s",
            "bbox_t_s",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "bbox_area",
            "event_idx_lo",
            "event_idx_hi",
            "event_count",
            "gt_ttc_s",
            "status_gt",
            "segmentation_json",
            "category",
        ],
    )

    print(
        f"[Done] {sequence_name}: labels={len(label_records)} "
        f"intervals={len(interval_rows)} -> {sequence_out}"
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_full_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.sequences:
        sequence_roots = [dataset_root / name for name in args.sequences]
    else:
        sequence_roots = sorted(path for path in dataset_root.iterdir() if path.is_dir())

    for sequence_root in sequence_roots:
        build_sequence(sequence_root=sequence_root, output_root=output_root, args=args)


if __name__ == "__main__":
    main()
