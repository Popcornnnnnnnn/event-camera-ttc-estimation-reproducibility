#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_formal_sequence_artifacts as formal_builder


REQUIRED_ROOT_FILES = ("gt.hdf5", "ttc.csv")
DOWNLOAD_PATTERNS = ("drive-download-*.zip", "*.zip", "*.part", "*.tmp", ".DS_Store")


@dataclass
class SequenceInventoryRecord:
    sequence_id: str
    source_root: str
    status: str
    formal_status: str
    raw_hdf5_count: int
    gt_hdf5_count: int
    ttc_csv_count: int
    bag_count: int
    mp4_count: int
    leftlabel_json_count: int
    has_nested_dcv: bool
    has_nested_gvt: bool
    formal_interval_count: int | None
    cleanup_notes: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Dataset-Full sequences and build resumable EvTTC formal artifacts.",
    )
    parser.add_argument(
        "--dataset-full-root",
        type=Path,
        default=REPO_ROOT / "Dataset-Full",
        help="Root containing raw full sequence directories.",
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=REPO_ROOT / "Code" / "DatasetFormal",
        help="Root where formal sequence artifacts are stored.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "Code" / "Derived" / "DataIngestion",
        help="Root for inventory and batch status artifacts.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional output run tag. Default: timestamped PrepareFormalBatch run.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional sequence IDs. Default: all directories under Dataset-Full.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect only; do not move files, delete leftovers, or build formal artifacts.",
    )
    parser.add_argument(
        "--keep-download-artifacts",
        action="store_true",
        help="Keep zip/part/tmp browser download artifacts after flattening.",
    )
    parser.add_argument(
        "--keep-gvt",
        action="store_true",
        help="Keep nested GVT directories after flattening DCV assets.",
    )
    parser.add_argument(
        "--overwrite-formal",
        action="store_true",
        help="Force rebuild formal artifacts even when existing outputs are complete.",
    )
    parser.add_argument("--event-group", default="prophesee/event_cam_left")
    parser.add_argument("--frame-ts-path", default="blackflys/left/ts")
    return parser.parse_args()


def count_paths(paths: list[Path]) -> int:
    return len(paths)


def list_sequence_roots(dataset_full_root: Path, sequence_ids: list[str] | None) -> list[Path]:
    if sequence_ids:
        return [dataset_full_root / sequence_id for sequence_id in sequence_ids]
    return sorted(path for path in dataset_full_root.iterdir() if path.is_dir())


def merge_or_move(src: Path, dst: Path, *, dry_run: bool, notes: list[str]) -> None:
    if src.is_dir():
        if dst.exists() and not dst.is_dir():
            notes.append(f"conflict_dir_to_file:{src.name}->{dst.name}")
            return
        if not dst.exists():
            notes.append(f"move_dir:{src.relative_to(src.parents[1])}->{dst.name}")
            if not dry_run:
                shutil.move(str(src), str(dst))
            return
        for child in sorted(src.iterdir()):
            merge_or_move(child, dst / child.name, dry_run=dry_run, notes=notes)
        if not dry_run:
            try:
                src.rmdir()
            except OSError:
                pass
        return

    if dst.exists():
        if dst.is_file() and src.stat().st_size == dst.stat().st_size:
            notes.append(f"duplicate_file:{src.name}")
            if not dry_run:
                src.unlink()
        else:
            notes.append(f"conflict_file:{src.name}")
        return

    notes.append(f"move_file:{src.name}")
    if not dry_run:
        shutil.move(str(src), str(dst))


def flatten_sequence_root(sequence_root: Path, *, dry_run: bool, keep_gvt: bool, keep_download_artifacts: bool) -> str:
    notes: list[str] = []
    dcv_root = sequence_root / "DCV"
    if dcv_root.exists():
        for child in sorted(dcv_root.iterdir()):
            merge_or_move(child, sequence_root / child.name, dry_run=dry_run, notes=notes)
        if not dry_run:
            try:
                dcv_root.rmdir()
            except OSError:
                pass

    gvt_root = sequence_root / "GVT"
    if gvt_root.exists() and not keep_gvt:
        notes.append("remove_gvt")
        if not dry_run:
            shutil.rmtree(gvt_root)

    if not keep_download_artifacts:
        for pattern in DOWNLOAD_PATTERNS:
            for artifact in sequence_root.glob(pattern):
                notes.append(f"remove_artifact:{artifact.name}")
                if not dry_run:
                    if artifact.is_dir():
                        shutil.rmtree(artifact)
                    else:
                        artifact.unlink()

    return ";".join(notes)


def inspect_sequence_root(sequence_root: Path) -> dict[str, Any]:
    leftlabel_root = sequence_root / "leftlabel"
    raw_hdf5 = sorted(path for path in sequence_root.glob("*.hdf5") if path.name != "gt.hdf5")
    return {
        "raw_hdf5_count": count_paths(raw_hdf5),
        "gt_hdf5_count": count_paths(list(sequence_root.glob("gt.hdf5"))),
        "ttc_csv_count": count_paths(list(sequence_root.glob("ttc.csv"))),
        "bag_count": count_paths(list(sequence_root.glob("*.bag"))),
        "mp4_count": count_paths(list(sequence_root.glob("*.mp4"))),
        "leftlabel_json_count": count_paths(sorted(leftlabel_root.glob("*.json"))) if leftlabel_root.exists() else 0,
        "has_nested_dcv": (sequence_root / "DCV").exists(),
        "has_nested_gvt": (sequence_root / "GVT").exists(),
    }


def classify_root(stats: dict[str, Any]) -> tuple[str, str]:
    missing: list[str] = []
    if stats["raw_hdf5_count"] != 1:
        missing.append(f"raw_hdf5_count={stats['raw_hdf5_count']}")
    if stats["gt_hdf5_count"] != 1:
        missing.append("gt.hdf5")
    if stats["ttc_csv_count"] != 1:
        missing.append("ttc.csv")
    if stats["leftlabel_json_count"] < 2:
        missing.append(f"leftlabel_json_count={stats['leftlabel_json_count']}")
    if missing:
        return "missing_required", ",".join(missing)
    return "ready", ""


def formal_interval_count(sequence_out: Path) -> int | None:
    interval_path = sequence_out / "IntervalIndex.csv"
    if not interval_path.exists():
        return None
    with interval_path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def build_formal(sequence_root: Path, formal_root: Path, args: argparse.Namespace) -> tuple[str, int | None, str]:
    builder_args = SimpleNamespace(
        event_group=args.event_group,
        frame_ts_path=args.frame_ts_path,
        gt_query_mode="start",
        bbox_mode="start",
        overwrite_existing=args.overwrite_formal,
    )
    try:
        formal_builder.build_sequence(sequence_root=sequence_root, output_root=formal_root, args=builder_args)
    except Exception as exc:
        return "formal_failed", formal_interval_count(formal_root / sequence_root.name), str(exc)

    sequence_out = formal_root / sequence_root.name
    state, invalid = formal_builder.inspect_sequence_output(sequence_out)
    if state == "complete":
        return "formal_complete", formal_interval_count(sequence_out), ""
    return "formal_partial", formal_interval_count(sequence_out), ",".join(invalid)


def write_csv_atomic(path: Path, rows: list[SequenceInventoryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SequenceInventoryRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def summarize(records: list[SequenceInventoryRecord]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record.status}:{record.formal_status}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "sequence_count": len(records),
        "status_counts": counts,
        "formal_complete_count": sum(1 for item in records if item.formal_status == "formal_complete"),
        "total_intervals": sum(item.formal_interval_count or 0 for item in records),
    }


def main() -> None:
    args = parse_args()
    dataset_full_root = args.dataset_full_root.resolve()
    formal_root = args.formal_root.resolve()
    run_tag = args.run_tag or datetime.now().strftime("PrepareFormalBatch_%Y%m%d_%H%M%S")
    run_root = (args.output_root / run_tag).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    formal_root.mkdir(parents=True, exist_ok=True)

    csv_path = run_root / "SequenceInventory.csv"
    summary_path = run_root / "Summary.json"
    records: list[SequenceInventoryRecord] = []

    sequence_roots = list_sequence_roots(dataset_full_root, args.sequences)
    for sequence_root in sequence_roots:
        cleanup_notes = ""
        error = ""
        formal_status = "not_attempted"
        formal_count: int | None = None
        status = "missing_root"
        stats = {
            "raw_hdf5_count": 0,
            "gt_hdf5_count": 0,
            "ttc_csv_count": 0,
            "bag_count": 0,
            "mp4_count": 0,
            "leftlabel_json_count": 0,
            "has_nested_dcv": False,
            "has_nested_gvt": False,
        }

        if sequence_root.exists():
            try:
                cleanup_notes = flatten_sequence_root(
                    sequence_root,
                    dry_run=args.dry_run,
                    keep_gvt=args.keep_gvt,
                    keep_download_artifacts=args.keep_download_artifacts,
                )
                stats = inspect_sequence_root(sequence_root)
                status, error = classify_root(stats)
                if status == "ready":
                    if args.dry_run:
                        formal_status = "dry_run"
                    else:
                        formal_status, formal_count, formal_error = build_formal(sequence_root, formal_root, args)
                        if formal_error:
                            error = formal_error
            except Exception as exc:
                status = "sequence_failed"
                error = str(exc)
        else:
            error = f"missing sequence root: {sequence_root}"

        record = SequenceInventoryRecord(
            sequence_id=sequence_root.name,
            source_root=str(sequence_root),
            status=status,
            formal_status=formal_status,
            raw_hdf5_count=int(stats["raw_hdf5_count"]),
            gt_hdf5_count=int(stats["gt_hdf5_count"]),
            ttc_csv_count=int(stats["ttc_csv_count"]),
            bag_count=int(stats["bag_count"]),
            mp4_count=int(stats["mp4_count"]),
            leftlabel_json_count=int(stats["leftlabel_json_count"]),
            has_nested_dcv=bool(stats["has_nested_dcv"]),
            has_nested_gvt=bool(stats["has_nested_gvt"]),
            formal_interval_count=formal_count,
            cleanup_notes=cleanup_notes,
            error=error,
        )
        records.append(record)
        write_csv_atomic(csv_path, records)
        write_json_atomic(
            summary_path,
            {
                "run_tag": run_tag,
                "started_or_updated_at": datetime.now().isoformat(timespec="seconds"),
                "dataset_full_root": str(dataset_full_root),
                "formal_root": str(formal_root),
                "dry_run": bool(args.dry_run),
                "summary": summarize(records),
            },
        )
        print(
            f"[{record.status}/{record.formal_status}] {record.sequence_id} "
            f"labels={record.leftlabel_json_count} intervals={record.formal_interval_count} "
            f"error={record.error}"
        )

    print(f"[Inventory] {csv_path}")
    print(f"[Summary] {summary_path}")


if __name__ == "__main__":
    main()
