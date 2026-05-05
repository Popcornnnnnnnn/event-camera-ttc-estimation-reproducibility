#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bucket compare rows into actionable diagnostic categories.",
    )
    parser.add_argument(
        "--compare-csv",
        type=Path,
        required=True,
        help="Path to CompareIntervalMetrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: compare CSV parent.",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=15.0,
        help="Delta e_ttc_pct threshold for a big improvement.",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=15.0,
        help="Delta e_ttc_pct threshold for a big regression.",
    )
    parser.add_argument(
        "--cue-underestimate-ratio",
        type=float,
        default=0.55,
        help="Threshold for v1_ttc_est_s / v1_fallback_ttc_s to mark cue underestimation.",
    )
    parser.add_argument(
        "--low-support-max-valid-cues",
        type=int,
        default=1,
        help="Max valid cue count to mark a sample as low support.",
    )
    parser.add_argument(
        "--stable-delta-threshold",
        type=float,
        default=5.0,
        help="Absolute delta threshold for stable fallback samples.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Rows to print in top regressions/improvements sections.",
    )
    return parser.parse_args()


def parse_optional_float(value: str) -> float | None:
    raw = value.strip()
    if not raw:
        return None
    return float(raw)


def parse_int(value: str) -> int:
    return int(value.strip())


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def classify_row(row: dict[str, object], args: argparse.Namespace) -> tuple[str, list[str]]:
    flags: list[str] = []

    delta = row["delta_e_ttc_pct"]
    v1_status = str(row["v1_status"])
    v2_status = str(row["v2_status"])
    v1_ratio = row["v1_cue_fallback_ratio"]
    low_support = (
        int(row["v1_local_cue_valid_count"]) <= args.low_support_max_valid_cues
        and int(row["v2_local_cue_valid_count"]) <= args.low_support_max_valid_cues
    )

    if v1_status == "OK" and v2_status == "OK_FALLBACK":
        flags.append("GUARD_SWITCH")
    if delta is not None and delta <= -args.improvement_threshold:
        flags.append("BIG_IMPROVEMENT")
    if delta is not None and delta >= args.regression_threshold:
        flags.append("BIG_REGRESSION")
    if v1_ratio is not None and v1_ratio < args.cue_underestimate_ratio:
        flags.append("CUE_UNDERESTIMATE")
    if low_support:
        flags.append("LOW_SUPPORT")

    if v1_status == "OK" and v2_status == "OK_FALLBACK" and delta is not None and delta >= args.regression_threshold:
        return "GUARD_FALSE_REJECT", flags
    if v2_status == "OK_FALLBACK" and delta is not None and delta <= -args.improvement_threshold:
        return "FALLBACK_WIN_BIG", flags
    if v1_ratio is not None and v1_ratio < args.cue_underestimate_ratio:
        return "CUE_UNDERESTIMATE", flags
    if low_support:
        return "LOW_SUPPORT", flags
    if (
        v1_status == "OK_FALLBACK"
        and v2_status == "OK_FALLBACK"
        and delta is not None
        and abs(delta) <= args.stable_delta_threshold
    ):
        return "STABLE_FALLBACK", flags
    return "OTHER", flags


def load_rows(path: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, object] = {
                **raw,
                "interval_idx": parse_int(raw["interval_idx"]),
                "gt_ttc_s": parse_optional_float(raw["gt_ttc_s"]),
                "v1_ttc_est_s": parse_optional_float(raw["v1_ttc_est_s"]),
                "v1_e_ttc_pct": parse_optional_float(raw["v1_e_ttc_pct"]),
                "v1_local_cue_valid_count": parse_int(raw["v1_local_cue_valid_count"]),
                "v1_fallback_ttc_s": parse_optional_float(raw["v1_fallback_ttc_s"]),
                "v2_ttc_est_s": parse_optional_float(raw["v2_ttc_est_s"]),
                "v2_e_ttc_pct": parse_optional_float(raw["v2_e_ttc_pct"]),
                "v2_local_cue_valid_count": parse_int(raw["v2_local_cue_valid_count"]),
                "v2_fallback_ttc_s": parse_optional_float(raw["v2_fallback_ttc_s"]),
                "delta_ttc_s": parse_optional_float(raw["delta_ttc_s"]),
                "delta_e_ttc_pct": parse_optional_float(raw["delta_e_ttc_pct"]),
            }
            v1_ratio = None
            if row["v1_ttc_est_s"] is not None and row["v1_fallback_ttc_s"] is not None:
                v1_ratio = float(row["v1_ttc_est_s"]) / max(float(row["v1_fallback_ttc_s"]), 1e-6)
            row["v1_cue_fallback_ratio"] = v1_ratio
            primary_bucket, flags = classify_row(row, args)
            row["primary_bucket"] = primary_bucket
            row["flags"] = "|".join(flags)
            rows.append(row)
    return rows


def summarize_counts(rows: list[dict[str, object]]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    overall = Counter[str]()
    by_sequence: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        bucket = str(row["primary_bucket"])
        sequence = str(row["sequence_id"])
        overall[bucket] += 1
        by_sequence[sequence][bucket] += 1
    return overall, by_sequence


def summarize_flags(rows: list[dict[str, object]]) -> Counter[str]:
    counter = Counter[str]()
    for row in rows:
        raw_flags = str(row["flags"]).strip()
        if not raw_flags:
            continue
        for flag in raw_flags.split("|"):
            if flag:
                counter[flag] += 1
    return counter


def top_rows(rows: list[dict[str, object]], *, descending: bool, limit: int) -> list[dict[str, object]]:
    valid_rows = [row for row in rows if row["delta_e_ttc_pct"] is not None]
    return sorted(
        valid_rows,
        key=lambda item: float(item["delta_e_ttc_pct"]),
        reverse=descending,
    )[:limit]


def build_summary(rows: list[dict[str, object]], args: argparse.Namespace, compare_csv: Path) -> str:
    overall_counts, by_sequence = summarize_counts(rows)
    flag_counts = summarize_flags(rows)
    regressions = top_rows(rows, descending=True, limit=args.top_k)
    improvements = top_rows(rows, descending=False, limit=args.top_k)

    lines = [
        "# Compare Bucket Summary",
        "",
        f"- source: `{compare_csv}`",
        f"- improvement_threshold: `{args.improvement_threshold:.1f}` pct points",
        f"- regression_threshold: `{args.regression_threshold:.1f}` pct points",
        f"- cue_underestimate_ratio: `{args.cue_underestimate_ratio:.2f}`",
        f"- low_support_max_valid_cues: `{args.low_support_max_valid_cues}`",
        "",
        "## Primary Bucket Rules",
        "",
        "- `GUARD_FALSE_REJECT`: `v1=OK` but `v2=OK_FALLBACK`, and error gets worse by at least the regression threshold.",
        "- `FALLBACK_WIN_BIG`: `v2=OK_FALLBACK`, and error improves by at least the improvement threshold.",
        "- `CUE_UNDERESTIMATE`: `v1_ttc_est / v1_fallback_ttc < cue_underestimate_ratio`.",
        "- `LOW_SUPPORT`: both `v1` and `v2` have very few valid local cues.",
        "- `STABLE_FALLBACK`: both versions are fallback and change is small.",
        "- `OTHER`: rows that do not fall into the action buckets above.",
        "",
        "## Overall Counts",
        "",
        "| bucket | count |",
        "|---|---:|",
    ]
    for bucket, count in sorted(overall_counts.items()):
        lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## Flag Counts",
            "",
            "| flag | count |",
            "|---|---:|",
        ]
    )
    for flag, count in sorted(flag_counts.items()):
        lines.append(f"| {flag} | {count} |")

    lines.extend(
        [
            "",
            "## Per-Sequence Counts",
            "",
            "| sequence | bucket | count |",
            "|---|---|---:|",
        ]
    )
    for sequence, counts in sorted(by_sequence.items()):
        for bucket, count in sorted(counts.items()):
            lines.append(f"| {sequence} | {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## Top Regressions",
            "",
            "| sample | bucket | delta_e_ttc_pct | v1_status | v2_status | v1_ratio | v1_valid | v2_valid |",
            "|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in regressions:
        lines.append(
            f"| {row['sample_id']} | {row['primary_bucket']} | {format_number(row['delta_e_ttc_pct'])} | "
            f"{row['v1_status']} | {row['v2_status']} | {format_number(row['v1_cue_fallback_ratio'])} | "
            f"{row['v1_local_cue_valid_count']} | {row['v2_local_cue_valid_count']} |"
        )

    lines.extend(
        [
            "",
            "## Top Improvements",
            "",
            "| sample | bucket | delta_e_ttc_pct | v1_status | v2_status | v1_ratio | v1_valid | v2_valid |",
            "|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in improvements:
        lines.append(
            f"| {row['sample_id']} | {row['primary_bucket']} | {format_number(row['delta_e_ttc_pct'])} | "
            f"{row['v1_status']} | {row['v2_status']} | {format_number(row['v1_cue_fallback_ratio'])} | "
            f"{row['v1_local_cue_valid_count']} | {row['v2_local_cue_valid_count']} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    compare_csv = args.compare_csv.resolve()
    output_dir = (args.output_dir or compare_csv.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(compare_csv, args)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_dir / "BucketedCompare.csv", rows, fieldnames)
    summary = build_summary(rows, args, compare_csv)
    (output_dir / "BucketSummary.md").write_text(summary, encoding="utf-8")

    overall_counts, _ = summarize_counts(rows)
    print(f"[Done] source={compare_csv}")
    print(f"[Done] output_dir={output_dir}")
    for bucket, count in sorted(overall_counts.items()):
        print(f"[Bucket] {bucket}={count}")


if __name__ == "__main__":
    main()
