#!/usr/bin/env python3
"""Audit paper-result traceability against local full-run artifacts.

This is a release audit, not a large rerun harness. It verifies that every row
in the paper manifest has source code, config, a local original run directory,
and a summary metric record matching the lightweight paper result numbers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "paper_results_manifest.json"
DEFAULT_FORMAL_ROOT = REPO_ROOT / "Code" / "DatasetFormal"
DEFAULT_REPORT_JSON = REPO_ROOT / "results_summaries" / "full_reproduction_audit.json"
DEFAULT_REPORT_MD = REPO_ROOT / "FULL_REPRODUCTION_AUDIT.md"


@dataclass
class RowAudit:
    paper_result_id: str
    family: str
    method_name: str
    variant_or_protocol: str
    code_entry_exists: bool
    config_exists: bool
    raw_run_directory_exists: bool
    summary_file: str | None
    summary_exists: bool
    metric_match: bool
    matched_metrics: dict[str, Any] | None
    status: str
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--metric-tolerance", type=float, default=0.0015)
    parser.add_argument("--pct-tolerance", type=float, default=0.01)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sequence_count(formal_root: Path) -> int:
    return len(list(formal_root.glob("*/SequenceMeta.json")))


def locate_summary(raw_run_dir: Path) -> Path | None:
    candidates = [raw_run_dir / "Summary.json", raw_run_dir / "summary.json"]
    candidates.extend(raw_run_dir.glob("*Summary*.json"))
    for parent in list(raw_run_dir.parents)[:4]:
        candidates.append(parent / "Summary.json")
        candidates.append(parent / "summary.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def metric_value(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def iter_metric_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        has_metric = any(
            key in value
            for key in (
                "est_valid",
                "est_valid_count",
                "e_ttc_pct",
                "mae_s",
                "failure",
                "failure_count",
            )
        )
        if has_metric:
            records.append(value)
        for child in value.values():
            records.extend(iter_metric_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(iter_metric_records(child))
    return records


def close_float(actual: Any, expected: Any, tolerance: float) -> bool:
    if actual is None or expected is None:
        return False
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def close_int(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return False
    try:
        return int(actual) == int(expected)
    except (TypeError, ValueError):
        return False


def match_metrics(row: dict[str, Any], summary: Any, metric_tol: float, pct_tol: float) -> dict[str, Any] | None:
    expected = {
        "gt_valid": row.get("gt_valid"),
        "est_valid": row.get("est_valid"),
        "failure": row.get("failure"),
        "mae_s": row.get("mae_s"),
        "e_ttc_pct": row.get("e_ttc_pct"),
    }
    for record in iter_metric_records(summary):
        actual = {
            "gt_valid": metric_value(record, "gt_valid", "gt_valid_count"),
            "est_valid": metric_value(record, "est_valid", "est_valid_count", "count"),
            "failure": metric_value(record, "failure", "failure_count", "total_failure_if_counting_rejection"),
            "mae_s": metric_value(record, "mae_s"),
            "e_ttc_pct": metric_value(record, "e_ttc_pct"),
        }
        count_ok = close_int(actual["est_valid"], expected["est_valid"])
        failure_ok = actual["failure"] is None or close_int(actual["failure"], expected["failure"])
        mae_ok = actual["mae_s"] is None or close_float(actual["mae_s"], expected["mae_s"], metric_tol)
        pct_ok = close_float(actual["e_ttc_pct"], expected["e_ttc_pct"], pct_tol)
        gt_ok = actual["gt_valid"] is None or close_int(actual["gt_valid"], expected["gt_valid"])
        if count_ok and failure_ok and mae_ok and pct_ok and gt_ok:
            return actual
    return None


def audit_row(row: dict[str, Any], args: argparse.Namespace) -> RowAudit:
    notes: list[str] = []
    code_entry = REPO_ROOT / row["code_entry"]
    config = REPO_ROOT / row["config"]
    raw_run_dir = REPO_ROOT / row["raw_run_directory"]
    summary_path = locate_summary(raw_run_dir)
    summary_exists = summary_path is not None and summary_path.exists()
    matched = None

    if not code_entry.exists():
        notes.append(f"missing code entry: {row['code_entry']}")
    if not config.exists():
        notes.append(f"missing config: {row['config']}")
    if not raw_run_dir.exists():
        notes.append(f"missing raw run directory: {row['raw_run_directory']}")
    if not summary_exists:
        notes.append("missing Summary.json near raw run directory")
    else:
        try:
            matched = match_metrics(row, load_json(summary_path), args.metric_tolerance, args.pct_tolerance)
        except Exception as exc:  # pragma: no cover - audit output should preserve unexpected parser failures.
            notes.append(f"summary parse/match error: {exc}")
        if matched is None:
            notes.append("no metric record matched manifest numbers within tolerance")

    ok = code_entry.exists() and config.exists() and raw_run_dir.exists() and summary_exists and matched is not None
    return RowAudit(
        paper_result_id=row["paper_result_id"],
        family=row["family"],
        method_name=row["method_name"],
        variant_or_protocol=row["variant_or_protocol"],
        code_entry_exists=code_entry.exists(),
        config_exists=config.exists(),
        raw_run_directory_exists=raw_run_dir.exists(),
        summary_file=str(summary_path.relative_to(REPO_ROOT)) if summary_path else None,
        summary_exists=summary_exists,
        metric_match=matched is not None,
        matched_metrics=matched,
        status="PASS" if ok else "FAIL",
        notes=notes,
    )


def write_reports(payload: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = payload["rows"]
    lines = [
        "# Full Reproduction Audit",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- repository_name: `{payload['repository_name']}`",
        f"- formal_sequence_count: `{payload['formal_sequence_count']}`",
        f"- manifest_rows: `{payload['manifest_rows']}`",
        f"- passed_rows: `{payload['passed_rows']}`",
        f"- failed_rows: `{payload['failed_rows']}`",
        "",
        "This audit checks row-level traceability against local full-run artifacts. It does not commit large datasets or generated experiment directories.",
        "",
        "## Row Status",
        "",
        "| id | family | status | summary | notes |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        notes = "; ".join(row["notes"]) if row["notes"] else "-"
        lines.append(
            f"| {row['paper_result_id']} | {row['family']} | {row['status']} | "
            f"{row['summary_file'] or '-'} | {notes} |"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    row_audits = [audit_row(row, args) for row in manifest]
    passed = sum(1 for row in row_audits if row.status == "PASS")
    try:
        formal_root_display = str(args.formal_root.resolve().relative_to(REPO_ROOT))
    except ValueError:
        formal_root_display = "${EVTTC_FORMAL_ROOT}"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repository_name": "event-camera-ttc-estimation-reproducibility",
        "thesis_title": "Time-To-Collision (TTC) estimation with event-based cameras",
        "formal_root": formal_root_display,
        "formal_sequence_count": sequence_count(args.formal_root),
        "manifest_rows": len(manifest),
        "passed_rows": passed,
        "failed_rows": len(row_audits) - passed,
        "rows": [asdict(row) for row in row_audits],
    }
    if args.write_report:
        write_reports(payload, args.report_json, args.report_md)
    print(
        "[FullReproductionAudit] "
        f"formal_sequences={payload['formal_sequence_count']} "
        f"manifest_rows={payload['manifest_rows']} "
        f"passed_rows={payload['passed_rows']} "
        f"failed_rows={payload['failed_rows']}"
    )
    if payload["failed_rows"]:
        for row in row_audits:
            if row.status != "PASS":
                print(f"[FAIL] {row.paper_result_id}: {'; '.join(row.notes)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
