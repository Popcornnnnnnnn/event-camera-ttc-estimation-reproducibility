#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VENV_PYTHON = REPO_ROOT / "Code" / ".venv" / "bin" / "python"

BIG_FILE_MIN_BYTES = 1 * 1024 * 1024
TEXT_FILE_MIN_BYTES = 512


@dataclass(frozen=True)
class SequenceAsset:
    sequence_id: str
    raw_h5_name: str
    raw_h5_url: str
    video_name: str
    video_url: str
    bag_name: str
    bag_url: str
    gt_h5_url: str
    ttc_url: str
    leftlabel_folder_url: str
    label_json_count: int = 50


MANIFEST: dict[str, SequenceAsset] = {
    "CCRs-1-low-50%": SequenceAsset(
        sequence_id="CCRs-1-low-50%",
        raw_h5_name="2024-07-15-14-59-46.hdf5",
        raw_h5_url="https://drive.google.com/file/d/1aVrZlgUG9_XF0jxyLNpbfzTR6-2EaEtp/view?usp=drive_link",
        video_name="2024-07-15-14-59-46.mp4",
        video_url="https://drive.google.com/file/d/10hbjTI79ygY-IZ17axDrHoPHp3VygObw/view?usp=drive_link",
        bag_name="2024-07-15-14-59-46.bag",
        bag_url="https://drive.google.com/file/d/1hQtxl6UOT9KnlT32KzAl1eLjrMd1XeVL/view?usp=drive_link",
        gt_h5_url="https://drive.google.com/file/d/1ct9GvcEx22kbc5GKDd_6Kx7ovLidXL0o/view?usp=drive_link",
        ttc_url="https://drive.google.com/file/d/1zEn5STm-3g4R0tv5PWfDddTRrJ61Gc0M/view?usp=drive_link",
        leftlabel_folder_url="https://drive.google.com/drive/folders/1A63K2HV6JKM0j7CVlPBBO9Jm5ahtbcCC?usp=drive_link",
    ),
    "CCRs-1-medium-50%": SequenceAsset(
        sequence_id="CCRs-1-medium-50%",
        raw_h5_name="2024-07-15-15-03-00.hdf5",
        raw_h5_url="https://drive.google.com/file/d/1hrm_6cFoq-Axh6r_ej5qCjCN2aEhEZQC/view?usp=drive_link",
        video_name="2024-07-15-15-03-00.mp4",
        video_url="https://drive.google.com/file/d/1Kos9mJQKmFpapjSHsIOCu8m6CQhAAqrh/view?usp=drive_link",
        bag_name="2024-07-15-15-03-00.bag",
        bag_url="https://drive.google.com/file/d/1MNiihnoFSHc2toP5a5awSUz3gj8MVytG/view?usp=drive_link",
        gt_h5_url="https://drive.google.com/file/d/1g_Njqjz_ux2LfgMlbbl_ebFBhKIDlZyl/view?usp=drive_link",
        ttc_url="https://drive.google.com/file/d/17O7JDpDes0_f7MVvYWvsBfOaaxKF63LU/view?usp=drive_link",
        leftlabel_folder_url="https://drive.google.com/drive/folders/1f9C44byF5FU5TDbA6nqN6sJsu9eqSvuH?usp=drive_link",
    ),
    "CCRs-1-high-50%": SequenceAsset(
        sequence_id="CCRs-1-high-50%",
        raw_h5_name="2024-09-25-15-53-47.hdf5",
        raw_h5_url="https://drive.google.com/file/d/1iYp9zNcXtY2ciqbR2XUZhDKT5s9mP-LQ/view?usp=drive_link",
        video_name="2024-09-25-15-53-47.mp4",
        video_url="https://drive.google.com/file/d/1rzoMpjeMCXf8nYOY1kULpHTWPZZp5rHd/view?usp=drive_link",
        bag_name="2024-09-25-15-53-47.bag",
        bag_url="https://drive.google.com/file/d/10CGEmjCxJCSVVoUsWi6gnXHbjzw78_Hh/view?usp=drive_link",
        gt_h5_url="https://drive.google.com/file/d/1V7sZkaI-PsNgLWi0fHHUq7mqf-uH74Hg/view?usp=drive_link",
        ttc_url="https://drive.google.com/file/d/1WZsF3vOUpozFj3Zl6b0mdZEOSAbbYf2e/view?usp=drive_link",
        leftlabel_folder_url="https://drive.google.com/drive/folders/1dRjdfgmgmZylSf9xhlmC4_AvHjv6J86d?usp=drive_link",
    ),
    "CCRs-1-low-0%": SequenceAsset(
        sequence_id="CCRs-1-low-0%",
        raw_h5_name="2024-07-15-15-06-08.hdf5",
        raw_h5_url="https://drive.google.com/file/d/1iY0RIMoTPj0QV7_i4wy29dIyqCr1jQpo/view?usp=drive_link",
        video_name="2024-07-15-15-06-08.mp4",
        video_url="https://drive.google.com/file/d/1dXHORmwo5EZrgzpvIaKvc-Krqes3RA_k/view?usp=drive_link",
        bag_name="2024-07-15-15-06-08.bag",
        bag_url="https://drive.google.com/file/d/10B-fLhkz4PVq_JEefqHkT2wy9rOJZVLV/view?usp=drive_link",
        gt_h5_url="https://drive.google.com/file/d/1TnWdOdKtrVg-YE-dceo5d4fLF3mEwK-b/view?usp=drive_link",
        ttc_url="https://drive.google.com/file/d/1maTtgLAlCAIs7HtN_aoCbgqHg6_STeOM/view?usp=drive_link",
        leftlabel_folder_url="https://drive.google.com/drive/folders/1VPSqjQfEq8qjki6kq3yHOlwBDmzb9t-B?usp=drive_link",
    ),
    "CCRs-1-medium-0%": SequenceAsset(
        sequence_id="CCRs-1-medium-0%",
        raw_h5_name="2024-07-15-15-09-14.hdf5",
        raw_h5_url="https://drive.google.com/file/d/15FePYRua1ZApfD3_XDMJzvqYA-9a3fc-/view?usp=drive_link",
        video_name="2024-07-15-15-09-14.mp4",
        video_url="https://drive.google.com/file/d/1T3bai33SXEmKJ_016x-HN2Z80-W69yU4/view?usp=drive_link",
        bag_name="2024-07-15-15-09-14.bag",
        bag_url="https://drive.google.com/file/d/11rsYTjC1l8J9slx9vmnUGC3JJO2PyXEI/view?usp=drive_link",
        gt_h5_url="https://drive.google.com/file/d/1MC6yqX2fMLR9U5riWjHrOf0yQGKKD5z-/view?usp=drive_link",
        ttc_url="https://drive.google.com/file/d/1oIt07qqtrhqQ-bk9CJ4H36ZRPF8ayzB1/view?usp=drive_link",
        leftlabel_folder_url="https://drive.google.com/drive/folders/19ZEP4fEDbgVBilJDOdKJ72qpx2bOh0X2?usp=drive_link",
    ),
    "CCRs-1-high-0%": SequenceAsset(
        sequence_id="CCRs-1-high-0%",
        raw_h5_name="2024-09-25-16-00-39.hdf5",
        raw_h5_url="https://drive.google.com/file/d/19AXe6M_on8-6-Ma1DA61D7VDBgjM7Oj0/view?usp=drive_link",
        video_name="2024-09-25-16-00-39.mp4",
        video_url="https://drive.google.com/file/d/1TzJgljGqrfWzJ7za7E4X6TRuDaFL3fYF/view?usp=drive_link",
        bag_name="2024-09-25-16-00-39.bag",
        bag_url="https://drive.google.com/file/d/1nOaD743x562ADwhvQHU1yCY0vUJdUZdA/view?usp=drive_link",
        gt_h5_url="https://drive.google.com/file/d/1ysS10heTOZL7Y2UQW4EF9jTlB1aBx-Ly/view?usp=drive_link",
        ttc_url="https://drive.google.com/file/d/1JM7hJ9_kmVb540SO_kt73iueKr437xe0/view?usp=drive_link",
        leftlabel_folder_url="https://drive.google.com/drive/folders/1h-5U8tLtch6KpK8YQdRVIEXsa2jSZB7L?usp=drive_link",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download missing CCRs-1 EvTTC sequences with resume support.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "Dataset-Full",
        help="Destination root for raw full sequences.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=list(MANIFEST.keys()),
        help="Optional subset of sequence ids to download.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Max retry attempts per asset.",
    )
    return parser.parse_args()


def mobile_folder_url(url: str) -> str:
    folder_id = re.search(r"/folders/([A-Za-z0-9_-]+)", url)
    if not folder_id:
        raise ValueError(f"Could not parse folder id from {url}")
    return f"https://drive.google.com/drive/mobile/folders/{folder_id.group(1)}"


def fetch_text(url: str) -> str:
    completed = subprocess.run(
        [
            "curl",
            "-L",
            "--retry",
            "8",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "120",
            "-A",
            "Mozilla/5.0",
            url,
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to fetch url: {url}\n{completed.stderr}")
    return completed.stdout


def list_mobile_folder_files(folder_url: str) -> list[tuple[str, str]]:
    html_text = fetch_text(mobile_folder_url(folder_url))
    rows = re.findall(
        r'<tr[^>]+data-id="([A-Za-z0-9_-]+)"[^>]*>.*?<strong class="DNoYtb">([^<]+)</strong>',
        html_text,
        flags=re.DOTALL,
    )
    files: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for file_id, raw_name in rows:
        name = html.unescape(raw_name).strip()
        if not name or name in seen_names:
            continue
        if not (name.endswith(".json") or name == "isat.yaml"):
            continue
        seen_names.add(name)
        files.append((file_id, name))
    return files


def is_valid_file(path: Path, *, min_bytes: int) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def remove_if_invalid(path: Path, *, min_bytes: int) -> None:
    if path.exists() and path.is_file() and path.stat().st_size < min_bytes:
        path.unlink()


def run_command(args: list[str]) -> int:
    completed = subprocess.run(args, cwd=str(REPO_ROOT))
    return completed.returncode


def download_big_file(url: str, output: Path, *, min_bytes: int, max_attempts: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    remove_if_invalid(output, min_bytes=min_bytes)
    if is_valid_file(output, min_bytes=min_bytes):
        print(f"[Skip] {output.name} already exists")
        return

    for attempt in range(1, max_attempts + 1):
        print(f"[Download] {output.name} attempt={attempt}/{max_attempts}")
        exit_code = run_command(
            [
                str(VENV_PYTHON),
                "-m",
                "gdown",
                "--fuzzy",
                "--continue",
                url,
                "-O",
                str(output),
            ]
        )
        if exit_code == 0 and is_valid_file(output, min_bytes=min_bytes):
            print(f"[Done] {output}")
            return
        time.sleep(min(30, attempt * 5))
    raise RuntimeError(f"Failed to download file: {output}")


def download_leftlabel(folder_url: str, target_dir: Path, *, max_attempts: int) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    files = list_mobile_folder_files(folder_url)
    if not files:
        raise RuntimeError(f"No files discovered in folder: {folder_url}")

    for file_id, name in files:
        output = target_dir / name
        remove_if_invalid(output, min_bytes=TEXT_FILE_MIN_BYTES)
        if is_valid_file(output, min_bytes=TEXT_FILE_MIN_BYTES):
            continue
        file_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        for attempt in range(1, max_attempts + 1):
            print(f"[LeftLabel] {name} attempt={attempt}/{max_attempts}")
            exit_code = run_command(
                [
                    "curl",
                    "-L",
                    "--retry",
                    "5",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "30",
                    file_url,
                    "-o",
                    str(output),
                ]
            )
            if exit_code == 0 and is_valid_file(output, min_bytes=TEXT_FILE_MIN_BYTES):
                break
            remove_if_invalid(output, min_bytes=TEXT_FILE_MIN_BYTES)
            time.sleep(min(10, attempt * 2))
        else:
            raise RuntimeError(f"Failed to download leftlabel file: {name}")


def sequence_complete(sequence_root: Path, asset: SequenceAsset) -> bool:
    expected = [
        (sequence_root / asset.raw_h5_name, BIG_FILE_MIN_BYTES),
        (sequence_root / asset.video_name, BIG_FILE_MIN_BYTES),
        (sequence_root / asset.bag_name, BIG_FILE_MIN_BYTES),
        (sequence_root / "gt.hdf5", BIG_FILE_MIN_BYTES),
        (sequence_root / "ttc.csv", TEXT_FILE_MIN_BYTES),
    ]
    if not all(is_valid_file(path, min_bytes=min_bytes) for path, min_bytes in expected):
        return False
    label_dir = sequence_root / "leftlabel"
    if not label_dir.is_dir():
        return False
    return len(list(label_dir.glob("*.json"))) >= asset.label_json_count


def main() -> None:
    args = parse_args()
    for sequence_id in args.sequences:
        if sequence_id not in MANIFEST:
            raise SystemExit(f"Unknown sequence id: {sequence_id}")

    dataset_root = args.dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    for sequence_id in args.sequences:
        asset = MANIFEST[sequence_id]
        sequence_root = dataset_root / sequence_id
        sequence_root.mkdir(parents=True, exist_ok=True)
        if sequence_complete(sequence_root, asset):
            print(f"[Skip] {sequence_id} already complete")
            continue

        print(f"\n=== {sequence_id} ===")
        try:
            download_big_file(asset.raw_h5_url, sequence_root / asset.raw_h5_name, min_bytes=BIG_FILE_MIN_BYTES, max_attempts=args.max_attempts)
            download_big_file(asset.video_url, sequence_root / asset.video_name, min_bytes=BIG_FILE_MIN_BYTES, max_attempts=args.max_attempts)
            download_big_file(asset.bag_url, sequence_root / asset.bag_name, min_bytes=BIG_FILE_MIN_BYTES, max_attempts=args.max_attempts)
            download_big_file(asset.gt_h5_url, sequence_root / "gt.hdf5", min_bytes=BIG_FILE_MIN_BYTES, max_attempts=args.max_attempts)
            download_big_file(asset.ttc_url, sequence_root / "ttc.csv", min_bytes=TEXT_FILE_MIN_BYTES, max_attempts=args.max_attempts)
            download_leftlabel(asset.leftlabel_folder_url, sequence_root / "leftlabel", max_attempts=args.max_attempts)
            print(f"[Sequence Done] {sequence_id}")
        except Exception as exc:
            failures.append(f"{sequence_id}: {exc}")
            print(f"[Sequence Failed] {sequence_id}: {exc}")

    if failures:
        print("\nCompleted with failures:")
        for item in failures:
            print(f"- {item}")
    else:
        print("\nAll requested sequences completed.")


if __name__ == "__main__":
    main()
