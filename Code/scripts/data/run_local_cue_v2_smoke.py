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
from evttc.confidence import build_local_confidence_v1
from evttc.representations import build_local_stats_v1, build_lts_v1
from evttc.sparsity_aware_lts import LocalCueConfigV2, build_local_ttc_cues_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test LocalTtcCueV2 on a formal EvTTC sequence.",
    )
    parser.add_argument("--sequence", default="CCRs-1-low-100%", help="Formal sequence ID.")
    parser.add_argument("--interval-idx", type=int, default=0, help="Interval index to inspect.")
    parser.add_argument("--tau-ms", type=float, default=20.0, help="Tau used by LtsV1.")
    parser.add_argument("--grid-rows", type=int, default=3, help="Rows used by LocalStatsV1.")
    parser.add_argument("--grid-cols", type=int, default=3, help="Cols used by LocalStatsV1.")
    parser.add_argument("--selection-threshold", type=float, default=0.55, help="Confidence threshold.")
    parser.add_argument("--min-surface-value", type=float, default=0.15, help="Minimum time surface value.")
    parser.add_argument("--output-json", action="store_true", help="Print full JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with NativeSampleAdapter(args.sequence) as adapter:
        bundle = adapter.get_sample(interval_idx=args.interval_idx, include_events=True)
        lts = build_lts_v1(bundle, tau_ms=args.tau_ms)
        stats = build_local_stats_v1(lts, grid_rows=args.grid_rows, grid_cols=args.grid_cols)
        conf = build_local_confidence_v1(
            stats,
            selection_threshold=args.selection_threshold,
        )
        cues = build_local_ttc_cues_v2(
            bundle,
            lts,
            conf,
            cue_config=LocalCueConfigV2(min_surface_value=args.min_surface_value),
        )

    payload = {
        "sequence_id": bundle.sequence_id,
        "sample_id": bundle.sample_id,
        "interval_idx": bundle.interval_sample.interval_idx,
        "selected_cell_count": conf.selected_cell_count,
        "cue_count": len(cues),
        "ok_count": sum(1 for cue in cues if cue.status == "OK"),
        "cues": [
            {
                "row": cue.row,
                "col": cue.col,
                "status": cue.status,
                "valid_pixel_count": cue.valid_pixel_count,
                "confidence": cue.confidence,
                "cue_quality": cue.cue_quality,
                "radius_ref_px": cue.radius_ref_px,
                "radius_span_px": cue.radius_span_px,
                "time_span_s": cue.time_span_s,
                "fit_residual_px": cue.fit_residual_px,
                "ttc_s": cue.ttc_s,
            }
            for cue in cues
        ],
    }

    if args.output_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"[Sequence] {payload['sequence_id']}")
    print(f"[Sample] {payload['sample_id']} interval_idx={payload['interval_idx']}")
    print(f"[CueV2] selected_cell_count={payload['selected_cell_count']} cue_count={payload['cue_count']} ok_count={payload['ok_count']}")
    for cue in payload["cues"]:
        print(
            "[CueV2] "
            f"cell=({cue['row']}, {cue['col']}) "
            f"status={cue['status']} "
            f"quality={cue['cue_quality']} "
            f"ttc_s={cue['ttc_s']}"
        )


if __name__ == "__main__":
    main()
