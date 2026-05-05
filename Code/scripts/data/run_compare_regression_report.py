#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the biggest regressions and improvements from a V1/V2 compare CSV.",
    )
    parser.add_argument(
        "--compare-csv",
        type=Path,
        required=True,
        help="Path to CompareIntervalMetrics.csv",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Optional sequence filter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of rows to print for regressions/improvements.",
    )
    return parser.parse_args()


def parse_optional_float(value: str) -> float | None:
    raw = value.strip()
    if not raw:
        return None
    return float(raw)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    with args.compare_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if args.sequence is not None and row["sequence_id"] != args.sequence:
                continue
            delta = parse_optional_float(row["delta_e_ttc_pct"])
            if delta is None:
                continue
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "sequence_id": row["sequence_id"],
                    "interval_idx": int(row["interval_idx"]),
                    "v1_status": row["v1_status"],
                    "v2_status": row["v2_status"],
                    "v1_e_ttc_pct": parse_optional_float(row["v1_e_ttc_pct"]),
                    "v2_e_ttc_pct": parse_optional_float(row["v2_e_ttc_pct"]),
                    "delta_e_ttc_pct": delta,
                }
            )

    regressions = sorted(rows, key=lambda item: float(item["delta_e_ttc_pct"]), reverse=True)[: args.top_k]
    improvements = sorted(rows, key=lambda item: float(item["delta_e_ttc_pct"]))[: args.top_k]

    print(f"[Report] source={args.compare_csv}")
    print(f"[Report] sequence_filter={args.sequence or 'ALL'} rows={len(rows)}")
    print("")
    print("[Top Regressions]")
    for row in regressions:
        print(
            f"{row['sample_id']} "
            f"v1={row['v1_e_ttc_pct']:.3f} "
            f"v2={row['v2_e_ttc_pct']:.3f} "
            f"delta={row['delta_e_ttc_pct']:.3f} "
            f"statuses={row['v1_status']}->{row['v2_status']}"
        )
    print("")
    print("[Top Improvements]")
    for row in improvements:
        print(
            f"{row['sample_id']} "
            f"v1={row['v1_e_ttc_pct']:.3f} "
            f"v2={row['v2_e_ttc_pct']:.3f} "
            f"delta={row['delta_e_ttc_pct']:.3f} "
            f"statuses={row['v1_status']}->{row['v2_status']}"
        )


if __name__ == "__main__":
    main()
