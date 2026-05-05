#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

try:
    import cv2
except ImportError:  # allow headless no-display runs without OpenCV
    cv2 = None


def lower_bound(ds: h5py.Dataset, target: int, lo: int = 0, hi: int | None = None) -> int:
    """Binary search on an HDF5 1D sorted dataset."""
    if hi is None:
        hi = ds.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if int(ds[mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def get_resolution(group: h5py.Group, x_ds: h5py.Dataset, y_ds: h5py.Dataset) -> Tuple[int, int]:
    calib_res = group.get("calib/resolution")
    if calib_res is not None:
        width, height = map(int, calib_res[:])
        return width, height

    # Fallback when calibration resolution is missing.
    width = int(np.max(x_ds[:100000])) + 1
    height = int(np.max(y_ds[:100000])) + 1
    return width, height


def main() -> None:
    code_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Preview event windows from an HDF5 event stream.")
    parser.add_argument("--h5", default=str(code_root / "Dataset" / "CCRs1-low" / "data.hdf5"), help="Path to HDF5 file.")
    parser.add_argument(
        "--event-path",
        default="prophesee/event_cam_left",
        help="Group path that contains x/y/t/p datasets.",
    )
    parser.add_argument("--preview-count", type=int, default=100000, help="Preview first N events.")
    parser.add_argument("--window-ms", type=float, default=20.0, help="Time window in milliseconds.")
    parser.add_argument("--max-windows", type=int, default=0, help="Stop after N windows (0 = all).")
    parser.add_argument("--wait-ms", type=int, default=1, help="cv2.waitKey delay per frame.")
    parser.add_argument("--realtime", action="store_true", help="Play with timestamp-based pacing.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier for --realtime.")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow for headless runs.")
    parser.add_argument(
        "--render-mode",
        choices=["occupancy", "count"],
        default="occupancy",
        help="occupancy: binary hit map, count: per-pixel event count map.",
    )
    parser.add_argument("--count-thresh", type=int, default=0, help="Zero out pixels with count <= threshold.")
    parser.add_argument(
        "--count-map",
        choices=["linear", "log"],
        default="log",
        help="Brightness mapping for count mode.",
    )
    parser.add_argument("--hold-end", action="store_true", default=True, help="Hold on last frame when playback ends.")
    parser.add_argument("--no-hold-end", dest="hold_end", action="store_false", help="Exit immediately after playback.")
    parser.add_argument("--flip-x", action="store_true", help="Flip image horizontally.")
    parser.add_argument("--flip-y", action="store_true", default=False, help="Flip image vertically.")
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false", help="Disable vertical flip.")
    args = parser.parse_args()

    if not args.no_display and cv2 is None:
        raise ImportError("OpenCV (cv2) is required when --no-display is not set.")

    with h5py.File(args.h5, "r") as f:
        group = f[args.event_path]
        x_ds = group["x"]
        y_ds = group["y"]
        t_ds = group["t"]
        p_ds = group["p"]

        total = int(t_ds.shape[0])
        preview_n = min(args.preview_count, total)

        x0 = x_ds[:preview_n]
        y0 = y_ds[:preview_n]
        t0 = t_ds[:preview_n]
        p0 = p_ds[:preview_n]
        print(f"[Preview] events: {preview_n}/{total}")
        print(f"[Preview] x range: {int(x0.min())}..{int(x0.max())}")
        print(f"[Preview] y range: {int(y0.min())}..{int(y0.max())}")
        print(f"[Preview] t range: {int(t0.min())}..{int(t0.max())}")
        print(f"[Preview] p unique: {np.unique(p0)}")

        width, height = get_resolution(group, x_ds, y_ds)
        print(f"[Info] resolution: {width}x{height}")

        t_start = int(t_ds[0])
        t_last = int(t_ds[-1])
        dt = int(args.window_ms * 1000.0)  # dataset timestamps are in us
        print(f"[Info] window: {args.window_ms} ms ({dt} us)")
        print(f"[Info] timeline: {t_start}..{t_last}")
        print(f"[Info] flips: flip_x={args.flip_x}, flip_y={args.flip_y}")
        print(f"[Info] render: mode={args.render_mode}, count_thresh={args.count_thresh}, count_map={args.count_map}")
        if args.realtime:
            print(f"[Info] playback: realtime, speed={args.speed}x")
        else:
            print(f"[Info] playback: fixed wait, wait_ms={args.wait_ms}")

        if dt <= 0:
            raise ValueError("window size must be > 0")
        if args.speed <= 0:
            raise ValueError("speed must be > 0")
        if args.count_thresh < 0:
            raise ValueError("count-thresh must be >= 0")

        window_count = 0
        idx_lo = 0
        t_cursor = t_start
        wall_start = time.perf_counter()

        while t_cursor < t_last:
            t_next = t_cursor + dt
            start_idx = lower_bound(t_ds, t_cursor, lo=idx_lo)
            end_idx = lower_bound(t_ds, t_next, lo=start_idx)
            idx_lo = start_idx

            x_win = x_ds[start_idx:end_idx]
            y_win = y_ds[start_idx:end_idx]
            _p_win = p_ds[start_idx:end_idx]

            x_plot = x_win
            y_plot = y_win
            if args.flip_x:
                x_plot = (width - 1) - x_plot
            if args.flip_y:
                y_plot = (height - 1) - y_plot

            valid = (x_plot >= 0) & (x_plot < width) & (y_plot >= 0) & (y_plot < height)
            x_valid = x_plot[valid]
            y_valid = y_plot[valid]

            if args.render_mode == "occupancy":
                frame = np.zeros((height, width), dtype=np.uint8)
                frame[y_valid, x_valid] = 255
            else:
                count = np.zeros((height, width), dtype=np.uint32)
                np.add.at(count, (y_valid, x_valid), 1)
                if args.count_thresh > 0:
                    count[count <= args.count_thresh] = 0

                max_count = int(count.max())
                if max_count == 0:
                    frame = np.zeros((height, width), dtype=np.uint8)
                elif args.count_map == "log":
                    mapped = np.log1p(count.astype(np.float32)) / np.log1p(float(max_count))
                    frame = (mapped * 255.0).astype(np.uint8)
                else:
                    mapped = count.astype(np.float32) / float(max_count)
                    frame = (mapped * 255.0).astype(np.uint8)

            if not args.no_display:
                cv2.imshow("Event accumulation", frame)
                if args.realtime:
                    target_elapsed = ((t_next - t_start) / 1e6) / args.speed
                    now_elapsed = time.perf_counter() - wall_start
                    remain_s = max(0.0, target_elapsed - now_elapsed)
                    wait_ms = max(1, int(round(remain_s * 1000.0)))
                else:
                    wait_ms = args.wait_ms
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (27, ord("q")):
                    break
            else:
                print(
                    f"[Window {window_count:05d}] t=[{t_cursor},{t_next}) "
                    f"idx=[{start_idx},{end_idx}) events={end_idx - start_idx}"
                )

            window_count += 1
            if args.max_windows > 0 and window_count >= args.max_windows:
                break
            t_cursor = t_next

    if not args.no_display and args.hold_end and window_count > 0:
        print("[Info] playback ended. Press any key in the OpenCV window to exit.")
        cv2.waitKey(0)

    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"[Done] processed windows: {window_count}")


if __name__ == "__main__":
    main()
