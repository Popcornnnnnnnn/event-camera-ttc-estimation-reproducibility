#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evttc import list_formal_sequences
from evttc.ml_baseline import (
    MlBaselineExportConfig,
    MlBaselineSequenceExport,
    export_ml_baseline_sequence,
    summarize_ml_baseline_exports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export formal EvTTC samples into a lightweight ML baseline dataset.",
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=CODE_ROOT / "DatasetFormal",
        help="Root directory containing formal sequence artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_ROOT / "Derived" / "MlBaselineV1",
        help="Root directory for exported ML baseline artifacts.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Sequence IDs to export. Default: all formal sequences.",
    )
    parser.add_argument("--tag", default="default", help="Subdirectory tag under output root.")
    parser.add_argument("--image-height", type=int, default=64, help="Target image height.")
    parser.add_argument("--image-width", type=int, default=64, help="Target image width.")
    parser.add_argument("--tau-ms", type=float, default=20.0, help="Tau used by LtsV1.")
    parser.add_argument("--pad-px", type=int, default=0, help="Optional bbox padding.")
    parser.add_argument("--grid-rows", type=int, default=3, help="Rows used by LocalStatsV1.")
    parser.add_argument("--grid-cols", type=int, default=3, help="Cols used by LocalStatsV1.")
    parser.add_argument(
        "--image-variant",
        choices=["lts_v1", "polarity_split_v1", "multi_tau_polarity_split_v1"],
        default="lts_v1",
        help="Image tensor variant to export.",
    )
    parser.add_argument(
        "--multi-tau-ms",
        nargs="+",
        type=float,
        default=(10.0, 20.0, 40.0),
        help="Tau list used by multi-tau image variants.",
    )
    parser.add_argument(
        "--max-samples-per-sequence",
        type=int,
        default=None,
        help="Optional export cap per sequence for smoke or prototyping.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Build exports in memory without writing per-sequence npz files.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Re-export sequences even when an existing dataset.npz is readable.",
    )
    return parser.parse_args()


def is_readable_dataset_npz(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    required = {
        "image_nchw",
        "stats",
        "metadata",
        "gt_ttc_s",
        "gt_log_ttc_s",
        "sample_id",
        "sequence_id",
        "interval_idx",
    }
    try:
        with np.load(path) as data:
            if not required.issubset(set(data.files)):
                return False
            n = int(data["image_nchw"].shape[0])
            return (
                n > 0
                and int(data["stats"].shape[0]) == n
                and int(data["metadata"].shape[0]) == n
                and int(data["gt_ttc_s"].shape[0]) == n
            )
    except Exception:
        return False


def summarize_existing_npz(sequence_id: str, config: MlBaselineExportConfig, path: Path) -> MlBaselineSequenceExport:
    with np.load(path) as data:
        image_shape = tuple(int(v) for v in data["image_nchw"].shape)
        return MlBaselineSequenceExport(
            sequence_id=sequence_id,
            config=config,
            sample_count=image_shape[0],
            skip_count=0,
            image_shape_nchw=image_shape,
            stats_dim=int(data["stats"].shape[1]),
            metadata_dim=int(data["metadata"].shape[1]),
            output_npz_path=str(path),
            sample_ids=tuple(str(v) for v in data["sample_id"].tolist()),
            interval_indices=tuple(int(v) for v in data["interval_idx"].tolist()),
            skip_records=tuple(),
        )


def write_csv_atomic(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def write_summary_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    sequences = args.sequences or list_formal_sequences(args.formal_root)
    if not sequences:
        raise SystemExit(f"No formal sequences found under {args.formal_root}")

    config = MlBaselineExportConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        tau_ms=args.tau_ms,
        pad_px=args.pad_px,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        image_variant=args.image_variant,
        multi_tau_ms=tuple(args.multi_tau_ms),
        max_samples_per_sequence=args.max_samples_per_sequence,
    )

    run_root = args.output_root / args.tag
    exports = []
    status_rows: list[dict[str, object]] = []
    status_path = run_root / "ExportStatus.csv"
    summary_path = run_root / "Summary.json"

    for sequence_id in sequences:
        output_npz_path = None if args.summary_only else run_root / sequence_id / "dataset.npz"
        try:
            if (
                output_npz_path is not None
                and not args.overwrite_existing
                and is_readable_dataset_npz(output_npz_path)
            ):
                export = summarize_existing_npz(sequence_id, config, output_npz_path)
                action = "skipped_existing"
            else:
                export = export_ml_baseline_sequence(
                    sequence_id,
                    config=config,
                    output_npz_path=output_npz_path,
                    formal_root=args.formal_root,
                )
                action = "exported"
            exports.append(export)
            status_rows.append(
                {
                    "sequence_id": sequence_id,
                    "status": action,
                    "sample_count": export.sample_count,
                    "skip_count": export.skip_count,
                    "image_shape_nchw": "x".join(str(v) for v in export.image_shape_nchw),
                    "output_npz_path": export.output_npz_path or "",
                    "error": "",
                }
            )
            print(
                f"[{action}] {sequence_id} samples={export.sample_count} "
                f"skips={export.skip_count} output={export.output_npz_path}"
            )
        except Exception as exc:
            status_rows.append(
                {
                    "sequence_id": sequence_id,
                    "status": "failed",
                    "sample_count": 0,
                    "skip_count": 0,
                    "image_shape_nchw": "",
                    "output_npz_path": "" if output_npz_path is None else str(output_npz_path),
                    "error": str(exc),
                }
            )
            print(f"[Failed] {sequence_id}: {exc}", file=sys.stderr)

        if not args.summary_only:
            write_csv_atomic(
                status_path,
                status_rows,
                fieldnames=[
                    "sequence_id",
                    "status",
                    "sample_count",
                    "skip_count",
                    "image_shape_nchw",
                    "output_npz_path",
                    "error",
                ],
            )
            partial_summary = summarize_ml_baseline_exports(exports)
            write_summary_atomic(
                summary_path,
                {
                    "tag": args.tag,
                    "formal_root": str(args.formal_root.resolve()),
                    "output_root": str(run_root.resolve()),
                    "config": asdict(config),
                    "summary": partial_summary,
                    "status_rows": status_rows,
                },
            )

    summary = summarize_ml_baseline_exports(exports)
    summary_payload = {
        "tag": args.tag,
        "formal_root": str(args.formal_root.resolve()),
        "output_root": str(run_root.resolve()),
        "config": asdict(config),
        "summary": summary,
        "status_rows": status_rows,
    }

    if not args.summary_only:
        run_root.mkdir(parents=True, exist_ok=True)
        write_summary_atomic(summary_path, summary_payload)
        print(f"[Summary] {summary_path}")
    else:
        print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
