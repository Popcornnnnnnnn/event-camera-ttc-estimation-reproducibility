#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evttc import NativeSampleAdapter
from evttc.confidence import build_local_confidence_v1, summarize_local_confidence
from evttc.representations import build_local_stats_v1, build_lts_v1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test LocalConfidenceV1 on a formal EvTTC sequence.",
    )
    parser.add_argument("--sequence", default="CCRs-1-low-100%", help="Formal sequence ID.")
    parser.add_argument("--interval-idx", type=int, default=0, help="Interval index to inspect.")
    parser.add_argument("--tau-ms", type=float, default=20.0, help="Tau used by LtsV1.")
    parser.add_argument("--grid-rows", type=int, default=2, help="Rows used by LocalStatsV1.")
    parser.add_argument("--grid-cols", type=int, default=2, help="Cols used by LocalStatsV1.")
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=0.55,
        help="Confidence threshold used to keep local cells.",
    )
    parser.add_argument("--output-json", action="store_true", help="Print full JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with NativeSampleAdapter(args.sequence) as adapter:
        bundle = adapter.get_sample(interval_idx=args.interval_idx, include_events=True)
        lts = build_lts_v1(bundle, tau_ms=args.tau_ms)
        stats = build_local_stats_v1(lts, grid_rows=args.grid_rows, grid_cols=args.grid_cols)
        confidence = build_local_confidence_v1(
            stats,
            selection_threshold=args.selection_threshold,
        )

    payload = summarize_local_confidence(confidence)
    if args.output_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"[Sequence] {payload['sequence_id']}")
    print(f"[Sample] {payload['sample_id']} interval_idx={payload['interval_idx']}")
    print(f"[Confidence] global_confidence={payload['global_confidence']:.6f}")
    print(f"[Confidence] selected_cell_count={payload['selected_cell_count']}")
    print(f"[Confidence] selected_cell_indices={payload['selected_cell_indices']}")
    print(f"[Confidence] confidence_grid={payload['confidence_grid']}")


if __name__ == "__main__":
    main()
