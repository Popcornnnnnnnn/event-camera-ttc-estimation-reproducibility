#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

try:
    import cv2
except ImportError:  # allow headless no-display runs without OpenCV
    cv2 = None


def ns_from_ros_time(t) -> int:
    return int(t.sec) * 1_000_000_000 + int(t.nanosec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Slider *.bag event windows.")
    parser.add_argument("--bag", required=True, help="Path to slider *.bag file.")
    parser.add_argument("--topic", default="/capture_node/events", help="EventArray topic.")
    parser.add_argument("--window-ms", type=float, default=20.0, help="Window size in milliseconds.")
    parser.add_argument("--realtime", action="store_true", help="Play with timestamp-based pacing.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier for --realtime.")
    parser.add_argument("--wait-ms", type=int, default=1, help="cv2.waitKey delay when not realtime.")
    parser.add_argument("--max-windows", type=int, default=0, help="Stop after N windows (0 = all).")
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
    parser.add_argument("--flip-x", action="store_true", help="Flip image horizontally.")
    parser.add_argument("--flip-y", action="store_true", default=False, help="Flip image vertically.")
    parser.add_argument("--hold-end", action="store_true", default=True, help="Hold on last frame at end.")
    parser.add_argument("--no-hold-end", dest="hold_end", action="store_false", help="Exit immediately.")
    args = parser.parse_args()

    if not args.no_display and cv2 is None:
        raise ImportError("OpenCV (cv2) is required when --no-display is not set.")

    if args.window_ms <= 0:
        raise ValueError("window-ms must be > 0")
    if args.speed <= 0:
        raise ValueError("speed must be > 0")
    if args.count_thresh < 0:
        raise ValueError("count-thresh must be >= 0")

    bag_path = Path(args.bag)
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == args.topic]
        if not conns:
            topics = ", ".join(sorted({c.topic for c in reader.connections}))
            raise ValueError(f"Topic not found: {args.topic}. Available topics: {topics}")
        conn = conns[0]

        width = None
        height = None
        frame = None
        count_frame = None
        window_count = 0
        dt_ns = int(args.window_ms * 1_000_000.0)
        first_event_ns = None
        window_start_ns = None
        window_end_ns = None
        wall_start = None

        for c, _, raw in reader.messages(connections=[conn]):
            msg = reader.deserialize(raw, c.msgtype)

            if width is None:
                width = int(msg.width)
                height = int(msg.height)
                frame = np.zeros((height, width), dtype=np.uint8)
                count_frame = np.zeros((height, width), dtype=np.uint32)
                print(f"[Info] resolution: {width}x{height}")
                print(f"[Info] window: {args.window_ms} ms")
                print(
                    f"[Info] render: mode={args.render_mode}, "
                    f"count_thresh={args.count_thresh}, count_map={args.count_map}"
                )
                if args.realtime:
                    print(f"[Info] playback: realtime, speed={args.speed}x")
                else:
                    print(f"[Info] playback: fixed wait, wait_ms={args.wait_ms}")

            for ev in msg.events:
                ev_ns = ns_from_ros_time(ev.ts)
                if first_event_ns is None:
                    first_event_ns = ev_ns
                    window_start_ns = ev_ns
                    window_end_ns = window_start_ns + dt_ns
                    wall_start = time.perf_counter()

                while ev_ns >= window_end_ns:
                    if not args.no_display:
                        if args.render_mode == "count":
                            if args.count_thresh > 0:
                                masked = count_frame.copy()
                                masked[masked <= args.count_thresh] = 0
                            else:
                                masked = count_frame

                            max_count = int(masked.max())
                            if max_count == 0:
                                frame.fill(0)
                            elif args.count_map == "log":
                                mapped = np.log1p(masked.astype(np.float32)) / np.log1p(float(max_count))
                                frame = (mapped * 255.0).astype(np.uint8)
                            else:
                                mapped = masked.astype(np.float32) / float(max_count)
                                frame = (mapped * 255.0).astype(np.uint8)

                        if args.flip_x and args.flip_y:
                            show = cv2.flip(frame, -1)
                        elif args.flip_x:
                            show = cv2.flip(frame, 1)
                        elif args.flip_y:
                            show = cv2.flip(frame, 0)
                        else:
                            show = frame
                        cv2.imshow("Slider event accumulation", show)
                        if args.realtime:
                            target_elapsed = ((window_end_ns - first_event_ns) / 1e9) / args.speed
                            now_elapsed = time.perf_counter() - wall_start
                            remain_s = max(0.0, target_elapsed - now_elapsed)
                            wait_ms = max(1, int(round(remain_s * 1000.0)))
                        else:
                            wait_ms = args.wait_ms
                        key = cv2.waitKey(wait_ms) & 0xFF
                        if key in (27, ord("q")):
                            if not args.no_display:
                                cv2.destroyAllWindows()
                            print(f"[Done] processed windows: {window_count}")
                            return
                    else:
                        print(
                            f"[Window {window_count:05d}] "
                            f"t=[{window_start_ns},{window_end_ns})"
                        )

                    window_count += 1
                    if args.max_windows > 0 and window_count >= args.max_windows:
                        if args.hold_end and not args.no_display:
                            print("[Info] playback ended. Press any key in the OpenCV window to exit.")
                            cv2.waitKey(0)
                        if not args.no_display:
                            cv2.destroyAllWindows()
                        print(f"[Done] processed windows: {window_count}")
                        return

                    frame.fill(0)
                    count_frame.fill(0)
                    window_start_ns = window_end_ns
                    window_end_ns += dt_ns

                x = int(ev.x)
                y = int(ev.y)
                if 0 <= x < width and 0 <= y < height:
                    if args.render_mode == "occupancy":
                        frame[y, x] = 255
                    else:
                        count_frame[y, x] += 1

        if frame is not None:
            if not args.no_display:
                if args.render_mode == "count":
                    if args.count_thresh > 0:
                        masked = count_frame.copy()
                        masked[masked <= args.count_thresh] = 0
                    else:
                        masked = count_frame
                    max_count = int(masked.max())
                    if max_count == 0:
                        frame.fill(0)
                    elif args.count_map == "log":
                        mapped = np.log1p(masked.astype(np.float32)) / np.log1p(float(max_count))
                        frame = (mapped * 255.0).astype(np.uint8)
                    else:
                        mapped = masked.astype(np.float32) / float(max_count)
                        frame = (mapped * 255.0).astype(np.uint8)

                if args.flip_x and args.flip_y:
                    show = cv2.flip(frame, -1)
                elif args.flip_x:
                    show = cv2.flip(frame, 1)
                elif args.flip_y:
                    show = cv2.flip(frame, 0)
                else:
                    show = frame
                cv2.imshow("Slider event accumulation", show)
            window_count += 1

        if args.hold_end and not args.no_display:
            print("[Info] playback ended. Press any key in the OpenCV window to exit.")
            cv2.waitKey(0)
        if not args.no_display:
            cv2.destroyAllWindows()
        print(f"[Done] processed windows: {window_count}")


if __name__ == "__main__":
    main()
