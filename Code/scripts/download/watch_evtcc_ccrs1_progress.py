#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Code.scripts.download.download_evtcc_ccrs1_batch import (  # noqa: E402
    BIG_FILE_MIN_BYTES,
    TEXT_FILE_MIN_BYTES,
    MANIFEST,
    SequenceAsset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch real-time download progress for CCRs-1 EvTTC full sequences.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "Dataset-Full",
        help="Root directory containing raw full sequences.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=list(MANIFEST.keys()),
        help="Optional subset of sequence ids to display.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Refresh interval in seconds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render one snapshot and exit.",
    )
    return parser.parse_args()


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{value}B"


def find_part_file(sequence_root: Path, final_name: str) -> Path | None:
    matches = sorted(sequence_root.glob(f"{final_name}*.part"))
    if not matches:
        return None
    return matches[0]


def describe_file(sequence_root: Path, final_name: str, *, min_bytes: int) -> tuple[str, float]:
    final_path = sequence_root / final_name
    if final_path.exists() and final_path.stat().st_size >= min_bytes:
        return f"OK {format_bytes(final_path.stat().st_size)}", 1.0

    part_path = find_part_file(sequence_root, final_name)
    if part_path is not None and part_path.exists():
        return f"DL {format_bytes(part_path.stat().st_size)}", 0.5

    return "WAIT", 0.0


def describe_leftlabel(sequence_root: Path, asset: SequenceAsset) -> tuple[str, float]:
    leftlabel_dir = sequence_root / "leftlabel"
    count = len(list(leftlabel_dir.glob("*.json"))) if leftlabel_dir.is_dir() else 0
    ratio = min(1.0, count / float(asset.label_json_count))
    if count >= asset.label_json_count:
        return f"OK {count}/{asset.label_json_count}", 1.0
    if count > 0:
        return f"DL {count}/{asset.label_json_count}", ratio
    return f"WAIT 0/{asset.label_json_count}", 0.0


def overall_percent(parts: list[float]) -> str:
    percent = sum(parts) / max(1, len(parts)) * 100.0
    return f"{percent:5.1f}%"


def list_download_pids() -> list[str]:
    completed = subprocess.run(
        [
            "sh",
            "-lc",
            "ps -Ao pid=,command= | grep 'download_evtcc_ccrs1_batch.py' | grep -v grep || true",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines


def build_rows(dataset_root: Path, sequence_ids: list[str]) -> list[str]:
    rows: list[str] = []
    header = (
        f"{'Sequence':<18} {'Overall':>7}  {'HDF5':<14} {'MP4':<14} "
        f"{'BAG':<14} {'GT':<14} {'TTC':<12} {'LeftLabel':<14}"
    )
    rows.append(header)
    rows.append("-" * len(header))

    for sequence_id in sequence_ids:
        asset = MANIFEST[sequence_id]
        sequence_root = dataset_root / sequence_id
        hdf5_text, hdf5_ratio = describe_file(sequence_root, asset.raw_h5_name, min_bytes=BIG_FILE_MIN_BYTES)
        mp4_text, mp4_ratio = describe_file(sequence_root, asset.video_name, min_bytes=BIG_FILE_MIN_BYTES)
        bag_text, bag_ratio = describe_file(sequence_root, asset.bag_name, min_bytes=BIG_FILE_MIN_BYTES)
        gt_text, gt_ratio = describe_file(sequence_root, "gt.hdf5", min_bytes=BIG_FILE_MIN_BYTES)
        ttc_text, ttc_ratio = describe_file(sequence_root, "ttc.csv", min_bytes=TEXT_FILE_MIN_BYTES)
        leftlabel_text, leftlabel_ratio = describe_leftlabel(sequence_root, asset)
        rows.append(
            f"{sequence_id:<18} {overall_percent([hdf5_ratio, mp4_ratio, bag_ratio, gt_ratio, ttc_ratio, leftlabel_ratio]):>7}  "
            f"{hdf5_text:<14} {mp4_text:<14} {bag_text:<14} {gt_text:<14} {ttc_text:<12} {leftlabel_text:<14}"
        )
    return rows


def render(dataset_root: Path, sequence_ids: list[str]) -> str:
    lines: list[str] = []
    lines.append(f"Dataset Root: {dataset_root}")
    pids = list_download_pids()
    lines.append("Downloader: " + ("running" if pids else "not running"))
    if pids:
        lines.extend(f"  {line}" for line in pids)
    lines.append("")
    lines.extend(build_rows(dataset_root, sequence_ids))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    sequence_ids = args.sequences

    while True:
        print("\033[2J\033[H", end="")
        print(render(dataset_root, sequence_ids))
        if args.once:
            break
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    main()
