#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the public reproducibility smoke test on the synthetic formal fixture.",
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=REPO_ROOT / "fixtures" / "synthetic_formal",
        help="Formal fixture root. Defaults to fixtures/synthetic_formal.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "tmp" / "smoke_repro",
        help="Temporary output root for smoke-test experiment outputs.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("[Run]", " ".join(str(part) for part in command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    formal_root = args.formal_root.resolve()
    output_root = args.output_root.resolve()
    if not formal_root.exists():
        raise SystemExit(f"Missing formal fixture root: {formal_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            sys.executable,
            str(CODE_ROOT / "scripts" / "data" / "run_formal_loader_smoke.py"),
            "--formal-root",
            str(formal_root),
            "--sequence",
            "SYNTH-LOOMING-001",
            "--interval-idx",
            "0",
        ]
    )
    run_command(
        [
            sys.executable,
            str(CODE_ROOT / "scripts" / "data" / "run_sparsity_aware_lts_eval.py"),
            "--formal-root",
            str(formal_root),
            "--output-root",
            str(output_root),
            "--sequences",
            "SYNTH-LOOMING-001",
            "--variant",
            "BBoxLoomingV13",
            "--disable-lts-fallback",
        ]
    )

    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    if not run_dirs:
        raise SystemExit(f"No smoke run directory produced under {output_root}")
    summary_path = run_dirs[-1] / "Summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    est_valid = int(summary["est_valid_count"])
    e_ttc_pct = float(summary["e_ttc_pct"])
    if est_valid <= 0:
        raise SystemExit(f"Smoke failed: no valid predictions in {summary_path}")
    if e_ttc_pct > 35.0:
        raise SystemExit(f"Smoke failed: synthetic BBoxLooming error is too high: {e_ttc_pct:.3f}%")

    print(
        "[SmokePass] formal loader, event slice, BBoxLoomingV13 no-fallback, "
        f"and metric calculation passed: est_valid={est_valid}, e_ttc_pct={e_ttc_pct:.3f}%"
    )


if __name__ == "__main__":
    main()
