#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def count_bbox_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = sum(1 for _ in handle)
    return max(0, rows - 1)


def probe_hdf5(file_path: Path) -> dict:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"h5py_available": False, "error": str(exc)}

    try:
        with h5py.File(file_path, "r") as handle:
            keys = list(handle.keys())
            result = {"h5py_available": True, "root_keys": keys}
            # Try common path for EvTTC events
            group_path = "prophesee/event_cam_left"
            if group_path in handle:
                group = handle[group_path]
                for field in ("x", "y", "t", "p"):
                    if field in group:
                        result[f"{group_path}/{field}_shape"] = list(group[field].shape)
            return result
    except Exception as exc:  # pragma: no cover - malformed file
        return {"h5py_available": True, "error": str(exc)}


def build_manifest(dataset_root: Path) -> dict:
    sequences = []
    for seq_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        item: dict = {
            "name": seq_dir.name,
            "path": str(seq_dir),
            "bbox_csv": None,
            "hdf5": None,
            "bag_files": [],
            "bbox_rows": 0,
        }

        bbox = seq_dir / "bbox.csv"
        if bbox.exists():
            item["bbox_csv"] = str(bbox)
            item["bbox_rows"] = count_bbox_rows(bbox)

        h5 = seq_dir / "data.hdf5"
        if h5.exists():
            item["hdf5"] = str(h5)
            item["hdf5_probe"] = probe_hdf5(h5)

        bag_files = sorted(seq_dir.glob("*.bag"))
        item["bag_files"] = [str(path) for path in bag_files]

        if item["hdf5"]:
            item["data_type"] = "hdf5"
        elif item["bag_files"]:
            item["data_type"] = "bag"
        else:
            item["data_type"] = "unknown"

        sequences.append(item)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "sequence_count": len(sequences),
        "sequences": sequences,
    }


def to_markdown(manifest: dict) -> str:
    lines = [
        "# Dataset Manifest",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- dataset_root: `{manifest['dataset_root']}`",
        f"- sequence_count: `{manifest['sequence_count']}`",
        "",
        "| sequence | type | bbox_rows | hdf5 | bag_count |",
        "|---|---:|---:|---:|---:|",
    ]
    for seq in manifest["sequences"]:
        lines.append(
            f"| {seq['name']} | {seq['data_type']} | {seq['bbox_rows']} | "
            f"{'yes' if seq['hdf5'] else 'no'} | {len(seq['bag_files'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    code_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build a manifest for local EvTTC sample dataset.")
    parser.add_argument(
        "--dataset-root",
        default=code_root / "Dataset",
        type=Path,
        help="Path to dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=code_root / "results" / "small_sample",
        type=Path,
        help="Directory for generated manifest files.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(dataset_root)
    json_path = output_dir / "dataset_manifest.json"
    md_path = output_dir / "dataset_manifest.md"

    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(to_markdown(manifest), encoding="utf-8")

    print(f"[Done] manifest json: {json_path}")
    print(f"[Done] manifest md:   {md_path}")


if __name__ == "__main__":
    main()
