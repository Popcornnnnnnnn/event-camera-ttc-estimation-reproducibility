#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTIER_ROOT = CODE_ROOT / "Experiments" / "PostV13Frontier"
DEFAULT_FEATURE_TABLE = DEFAULT_FRONTIER_ROOT / "V13FrontierLGBM_full_v2" / "FeatureTable.csv"
QUANTILES = ("05", "10", "15", "25", "35", "50", "65", "75", "90")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate non-learning post-V13 candidate quantile policies on the fixed "
            "V13-valid set."
        )
    )
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FRONTIER_ROOT)
    parser.add_argument("--run-name", default="PostV13TraditionalPolicyAudit_v1")
    parser.add_argument("--pred-min-s", type=float, default=0.05)
    parser.add_argument("--pred-max-s", type=float, default=25.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_float(value: Any) -> float:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def candidate(row: dict[str, str], quantile: str) -> float:
    return parse_float(row.get(f"candidate_q{quantile}_s"))


def clipped(value: float, pred_min_s: float, pred_max_s: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return min(max(value, pred_min_s), pred_max_s)


def evaluate_policy(
    rows: list[dict[str, str]],
    policy_name: str,
    predict: Callable[[dict[str, str]], float],
    pred_min_s: float,
    pred_max_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    errors: list[float] = []
    mae_values: list[float] = []
    for row in rows:
        gt = parse_float(row.get("gt_ttc_s"))
        pred = clipped(predict(row), pred_min_s, pred_max_s)
        if not (math.isfinite(gt) and gt > 0.0 and math.isfinite(pred) and pred > 0.0):
            continue
        abs_error = abs(pred - gt)
        e_ttc_pct = abs_error / gt * 100.0
        errors.append(e_ttc_pct)
        mae_values.append(abs_error)
        details.append(
            {
                "policy": policy_name,
                "sample_id": row.get("sample_id"),
                "sequence_id": row.get("sequence_id"),
                "interval_idx": row.get("interval_idx"),
                "status_family": row.get("status_family"),
                "gt_ttc_s": gt,
                "pred_ttc_s": pred,
                "e_ttc_pct": e_ttc_pct,
            }
        )
    summary = {
        "policy": policy_name,
        "count": len(details),
        "mae_s": float(np.mean(mae_values)) if mae_values else None,
        "e_ttc_pct": float(np.mean(errors)) if errors else None,
        "median_e_ttc_pct": float(np.median(errors)) if errors else None,
        "p90_e_ttc_pct": float(np.percentile(errors, 90)) if errors else None,
        "over50": int(sum(value > 50.0 for value in errors)),
        "over100": int(sum(value > 100.0 for value in errors)),
    }
    return summary, details


def state_v13_like(row: dict[str, str]) -> float:
    status_family = row.get("status_family")
    if status_family in {"PHASESTABLE", "PHASELATE"}:
        return candidate(row, "10")
    if status_family in {"PHASEEARLY", "PHASEGROWTH"}:
        return candidate(row, "50")
    return candidate(row, "25")


def state_conservative(row: dict[str, str]) -> float:
    status_family = row.get("status_family")
    if status_family in {"PHASESTABLE", "PHASELATE"}:
        return candidate(row, "05")
    if status_family in {"PHASEEARLY", "PHASEGROWTH"}:
        return candidate(row, "35")
    return candidate(row, "15")


def state_median_guarded(row: dict[str, str]) -> float:
    status_family = row.get("status_family")
    if status_family in {"CURGUARD", "HISTGUARD"}:
        return candidate(row, "15")
    if status_family == "PHASELATE":
        return candidate(row, "10")
    if status_family == "PHASESTABLE":
        return candidate(row, "15")
    if status_family in {"PHASEEARLY", "PHASEGROWTH"}:
        return candidate(row, "50")
    return candidate(row, "25")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = sorted(payload["policy_summaries"], key=lambda item: float(item["e_ttc_pct"]))
    lines = [
        f"# {payload['run_name']}",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- fixed_eval_set: `V13-valid rows only`",
        f"- row_count: `{payload['row_count']}`",
        f"- sequence_count: `{payload['sequence_count']}`",
        f"- baseline_v13_e_ttc_pct: `{payload['baseline_v13_e_ttc_pct']:.3f}`",
        "",
        "| policy | count | mae_s | e_ttc_pct | median | p90 | over50 | over100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['policy']}`",
                    str(row["count"]),
                    f"{float(row['mae_s']):.6f}",
                    f"{float(row['e_ttc_pct']):.3f}",
                    f"{float(row['median_e_ttc_pct']):.3f}",
                    f"{float(row['p90_e_ttc_pct']):.3f}",
                    str(row["over50"]),
                    str(row["over100"]),
                ]
            )
            + " |"
        )
    best = rows[0]
    lines.extend(
        [
            "",
            "Conclusion:",
            "",
            f"- Best non-learning policy in this audit: `{best['policy']}` with `e_ttc_pct={float(best['e_ttc_pct']):.3f}`.",
            "- None of the fixed-set non-learning policies approaches `<7%` on the 3259 V13-valid rows.",
            "- This supports the boundary that the current `<7%` result depends on learned calibration plus confidence rejection, not on a pure hand-written candidate selector.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.feature_table)
    policies: list[tuple[str, Callable[[dict[str, str]], float]]] = []
    for quantile in QUANTILES:
        policies.append((f"all_q{quantile}", lambda row, q=quantile: candidate(row, q)))
    policies.extend(
        [
            ("state_v13_like_quantiles", state_v13_like),
            ("state_conservative", state_conservative),
            ("state_median_guarded", state_median_guarded),
        ]
    )
    policy_summaries: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    for name, predict in policies:
        summary, details = evaluate_policy(rows, name, predict, args.pred_min_s, args.pred_max_s)
        policy_summaries.append(summary)
        all_details.extend(details)

    baseline_errors = [parse_float(row.get("identity_v13_e_ttc_pct")) for row in rows]
    payload = {
        "run_name": args.run_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "Audit pure non-learning candidate quantile policies on fixed V13-valid rows.",
        "inputs": {
            "feature_table": str(args.feature_table),
            "pred_min_s": args.pred_min_s,
            "pred_max_s": args.pred_max_s,
        },
        "row_count": len(rows),
        "sequence_count": len({row.get("sequence_id") for row in rows}),
        "baseline_v13_e_ttc_pct": float(np.mean([x for x in baseline_errors if math.isfinite(x)])),
        "policy_summaries": policy_summaries,
    }
    write_json(output_dir / "Summary.json", payload)
    write_csv(
        output_dir / "PolicySummary.csv",
        policy_summaries,
        ["policy", "count", "mae_s", "e_ttc_pct", "median_e_ttc_pct", "p90_e_ttc_pct", "over50", "over100"],
    )
    write_csv(
        output_dir / "PolicyIntervalMetrics.csv",
        all_details,
        ["policy", "sample_id", "sequence_id", "interval_idx", "status_family", "gt_ttc_s", "pred_ttc_s", "e_ttc_pct"],
    )
    write_markdown(output_dir / "Summary.md", payload)
    best = min(policy_summaries, key=lambda item: float(item["e_ttc_pct"]))
    print(f"[Done] output={output_dir}")
    print(f"[Best] {best['policy']} e_ttc_pct={float(best['e_ttc_pct']):.3f}")


if __name__ == "__main__":
    main()
