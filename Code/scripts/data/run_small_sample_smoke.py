#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SmokeResult:
    sequence: str
    data_type: str
    command: list[str]
    return_code: int
    status: str
    output_tail: str


def read_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def run_cmd(command: list[str], cwd: Path, timeout_sec: int) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    combined = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode, combined


def tail_text(text: str, max_chars: int = 1000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def build_hdf5_command(python_bin: str, scripts_dir: Path, h5_path: str, windows: int) -> list[str]:
    return [
        python_bin,
        str(scripts_dir / "preview_event_windows.py"),
        "--h5",
        h5_path,
        "--event-path",
        "prophesee/event_cam_left",
        "--window-ms",
        "20",
        "--max-windows",
        str(windows),
        "--preview-count",
        "200000",
        "--no-display",
    ]


def build_bag_command(python_bin: str, scripts_dir: Path, bag_path: str, windows: int) -> list[str]:
    return [
        python_bin,
        str(scripts_dir / "preview_slider_bag.py"),
        "--bag",
        bag_path,
        "--window-ms",
        "20",
        "--max-windows",
        str(windows),
        "--no-display",
    ]


def to_markdown(results: list[SmokeResult], output_dir: Path, started_at: str) -> str:
    lines = [
        "# Small Sample Smoke Report",
        "",
        f"- started_at: `{started_at}`",
        f"- output_dir: `{output_dir}`",
        "",
        "| sequence | type | status | return_code |",
        "|---|---|---|---:|",
    ]
    for item in results:
        lines.append(f"| {item.sequence} | {item.data_type} | {item.status} | {item.return_code} |")

    lines.append("")
    for item in results:
        lines.extend(
            [
                f"## {item.sequence}",
                "",
                f"- type: `{item.data_type}`",
                f"- status: `{item.status}`",
                f"- command: `{' '.join(item.command)}`",
                "",
                "```text",
                item.output_tail or "(no output)",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    code_root = Path(__file__).resolve().parents[2]
    scripts_root = Path(__file__).resolve().parents[1]
    preview_dir = scripts_root / "preview"
    parser = argparse.ArgumentParser(description="Run smoke tests for local small-sample EvTTC sequences.")
    parser.add_argument(
        "--manifest",
        default=code_root / "results" / "small_sample" / "dataset_manifest.json",
        type=Path,
        help="Path to dataset manifest JSON.",
    )
    parser.add_argument(
        "--output-root",
        default=code_root / "results" / "small_sample",
        type=Path,
        help="Root directory for smoke outputs.",
    )
    parser.add_argument(
        "--python-bin",
        default="python3",
        help="Python executable used to run preview scripts.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=4,
        help="Run at most N sequences from manifest.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=3,
        help="Max windows per sequence for smoke run.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Timeout (seconds) for each sequence run.",
    )
    args = parser.parse_args()

    project_root = code_root.parent
    manifest = read_manifest(args.manifest.resolve())

    started_at = datetime.now().isoformat(timespec="seconds")
    run_dir = args.output_root.resolve() / datetime.now().strftime("smoke_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[SmokeResult] = []
    sequences = manifest.get("sequences", [])[: args.max_sequences]
    for seq in sequences:
        sequence_name = seq["name"]
        data_type = seq.get("data_type", "unknown")
        command: list[str] = []

        if data_type == "hdf5" and seq.get("hdf5"):
            command = build_hdf5_command(args.python_bin, preview_dir, seq["hdf5"], args.max_windows)
        elif data_type == "bag" and seq.get("bag_files"):
            command = build_bag_command(args.python_bin, preview_dir, seq["bag_files"][0], args.max_windows)
        else:
            results.append(
                SmokeResult(
                    sequence=sequence_name,
                    data_type=data_type,
                    command=[],
                    return_code=0,
                    status="SKIPPED_UNSUPPORTED",
                    output_tail="No runnable hdf5/bag source found.",
                )
            )
            continue

        try:
            return_code, output = run_cmd(command, cwd=project_root, timeout_sec=args.timeout_sec)
            status = "OK" if return_code == 0 else "FAILED"
            results.append(
                SmokeResult(
                    sequence=sequence_name,
                    data_type=data_type,
                    command=command,
                    return_code=return_code,
                    status=status,
                    output_tail=tail_text(output),
                )
            )
        except subprocess.TimeoutExpired as exc:
            results.append(
                SmokeResult(
                    sequence=sequence_name,
                    data_type=data_type,
                    command=command,
                    return_code=124,
                    status="TIMEOUT",
                    output_tail=tail_text((exc.stdout or "") + "\n" + (exc.stderr or "")),
                )
            )

    json_path = run_dir / "smoke_report.json"
    md_path = run_dir / "smoke_report.md"
    json_path.write_text(
        json.dumps(
            {
                "started_at": started_at,
                "output_dir": str(run_dir),
                "results": [asdict(item) for item in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md_path.write_text(to_markdown(results, run_dir, started_at), encoding="utf-8")

    print(f"[Done] smoke json: {json_path}")
    print(f"[Done] smoke md:   {md_path}")
    ok_count = sum(1 for item in results if item.status == "OK")
    print(f"[Summary] ok={ok_count} total={len(results)}")


if __name__ == "__main__":
    main()
