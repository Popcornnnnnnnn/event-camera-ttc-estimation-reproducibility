#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evttc import NativeSampleAdapter
from evttc.sparsity_aware_lts import predict_ttc_v1, predict_ttc_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SparsityAwareLTS V1 and V2 on a single formal sample.",
    )
    parser.add_argument("--sequence", default="CCRs-1-low-100%", help="Formal sequence ID.")
    parser.add_argument("--interval-idx", type=int, default=0, help="Interval index to inspect.")
    parser.add_argument("--output-json", action="store_true", help="Print full JSON payload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with NativeSampleAdapter(args.sequence) as adapter:
        bundle = adapter.get_sample(interval_idx=args.interval_idx, include_events=True)
        record_v1, _ = predict_ttc_v1(bundle)
        record_v2, debug_v2 = predict_ttc_v2(bundle)

    payload = {
        "sequence_id": bundle.sequence_id,
        "sample_id": bundle.sample_id,
        "interval_idx": bundle.interval_sample.interval_idx,
        "v1": asdict(record_v1),
        "v2": asdict(record_v2),
        "v2_ok_cue_count": sum(1 for cue in debug_v2.local_cues if cue.status == "OK"),
        "v2_statuses": [cue.status for cue in debug_v2.local_cues],
    }

    if args.output_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"[Sequence] {payload['sequence_id']}")
    print(f"[Sample] {payload['sample_id']} interval_idx={payload['interval_idx']}")
    print(
        "[V1] "
        f"status={payload['v1']['status']} "
        f"ttc_est_s={payload['v1']['ttc_est_s']} "
        f"local_cue_valid_count={payload['v1']['local_cue_valid_count']} "
        f"fallback_ttc_s={payload['v1']['fallback_ttc_s']}"
    )
    print(
        "[V2] "
        f"status={payload['v2']['status']} "
        f"ttc_est_s={payload['v2']['ttc_est_s']} "
        f"local_cue_valid_count={payload['v2']['local_cue_valid_count']} "
        f"fallback_ttc_s={payload['v2']['fallback_ttc_s']}"
    )
    print(f"[V2] ok_cue_count={payload['v2_ok_cue_count']}")
    print(f"[V2] statuses={payload['v2_statuses']}")


if __name__ == "__main__":
    main()
