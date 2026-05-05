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
from evttc.representations import build_local_stats_v1, build_lts_v1, summarize_representation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test LtsV1 and LocalStatsV1 on a formal EvTTC sequence.",
    )
    parser.add_argument(
        "--sequence",
        default="CCRs-1-low-100%",
        help="Formal sequence ID.",
    )
    parser.add_argument(
        "--interval-idx",
        type=int,
        default=0,
        help="Interval index to build representation for.",
    )
    parser.add_argument(
        "--tau-ms",
        type=float,
        default=20.0,
        help="Exponential decay constant used by LtsV1.",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=2,
        help="Rows used by LocalStatsV1.",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=2,
        help="Columns used by LocalStatsV1.",
    )
    parser.add_argument(
        "--pad-px",
        type=int,
        default=0,
        help="Optional bbox padding before ROI clipping.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print full JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with NativeSampleAdapter(args.sequence) as adapter:
        bundle = adapter.get_sample(interval_idx=args.interval_idx, include_events=True)
        lts = build_lts_v1(bundle, tau_ms=args.tau_ms, pad_px=args.pad_px)
        stats = build_local_stats_v1(lts, grid_rows=args.grid_rows, grid_cols=args.grid_cols)

    payload = summarize_representation(lts, stats)
    if args.output_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"[Sequence] {payload['sequence_id']}")
    print(f"[Sample] {payload['sample_id']} interval_idx={payload['interval_idx']}")
    print(
        f"[ROI] xyxy={tuple(payload['roi_xyxy'])} "
        f"shape={tuple(payload['roi_shape_hw'])} area={payload['roi_area']}"
    )
    print(
        f"[LTS] roi_event_count={payload['roi_event_count']} "
        f"active_pixel_count={payload['active_pixel_count']} "
        f"occupancy_ratio={payload['occupancy_ratio']:.6f} "
        f"mean_surface={payload['mean_surface']:.6f}"
    )
    grid_shape = tuple(payload["local_stats"]["grid_shape"])
    print(f"[LocalStats] grid_shape={grid_shape}")
    print(f"[LocalStats] event_count_grid={payload['local_stats']['event_count_grid']}")
    print(f"[LocalStats] occupancy_ratio_grid={payload['local_stats']['occupancy_ratio_grid']}")


if __name__ == "__main__":
    main()
