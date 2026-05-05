#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evttc.formal_loader import FormalSequenceReader, list_formal_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Phase 2 formal loader on Code/DatasetFormal sequences.",
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=CODE_ROOT / "DatasetFormal",
        help="Root directory containing formal sequence artifacts.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Sequence ID to inspect. Default: first available sequence.",
    )
    parser.add_argument(
        "--interval-idx",
        type=int,
        default=0,
        help="Interval index used for event-slice smoke verification.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print summary as JSON only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = list_formal_sequences(args.formal_root)
    if not sequences:
        raise SystemExit(f"No formal sequences found under {args.formal_root}")

    sequence_id = args.sequence or sequences[0]
    if sequence_id not in sequences:
        raise SystemExit(f"Sequence {sequence_id!r} not found under {args.formal_root}")

    with FormalSequenceReader(sequence_id, formal_root=args.formal_root) as reader:
        summary = reader.summary()
        sample = reader.get_interval_sample(interval_idx=args.interval_idx)
        event_slice = reader.read_event_slice(sample=sample)

    payload = {
        "summary": summary,
        "sample": {
            "sample_id": sample.sample_id,
            "interval_idx": sample.interval_idx,
            "t_start_s": sample.t_start_s,
            "t_end_s": sample.t_end_s,
            "bbox_xyxy": sample.bbox_xyxy,
            "event_idx_lo": sample.event_idx_lo,
            "event_idx_hi": sample.event_idx_hi,
            "event_count": sample.event_count,
            "gt_ttc_s": sample.gt_ttc_s,
            "status_gt": sample.status_gt,
        },
        "event_slice": {
            "count": event_slice.count,
            "t0_us": event_slice.t0_us,
            "t1_us": event_slice.t1_us,
            "x_dtype": str(event_slice.x.dtype),
            "y_dtype": str(event_slice.y.dtype),
            "t_dtype": str(event_slice.t_us.dtype),
            "p_dtype": str(event_slice.p.dtype),
        },
    }

    if args.output_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"[Sequence] {sequence_id}")
    print(f"[Summary] intervals={summary['interval_count']} valid_gt={summary['valid_gt_count']}")
    print(
        f"[Sample] {sample.sample_id} "
        f"t=[{sample.t_start_s:.6f}, {sample.t_end_s:.6f}) "
        f"bbox={sample.bbox_xyxy} gt_ttc={sample.gt_ttc_s}"
    )
    print(
        f"[Events] count={event_slice.count} "
        f"idx=[{event_slice.event_idx_lo}, {event_slice.event_idx_hi}) "
        f"t_us=[{event_slice.t0_us}, {event_slice.t1_us}]"
    )


if __name__ == "__main__":
    main()
