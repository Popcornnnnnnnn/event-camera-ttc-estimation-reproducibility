#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Code.scripts.download.download_evtcc_ccrs1_batch import (  # noqa: E402
    BIG_FILE_MIN_BYTES,
    TEXT_FILE_MIN_BYTES,
    MANIFEST,
)
from Code.scripts.download.watch_evtcc_ccrs1_progress import (  # noqa: E402
    describe_file,
    describe_leftlabel,
    list_download_pids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a live CCRs-1 download dashboard over HTTP.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "Dataset-Full",
        help="Root directory containing raw full sequences.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=list(MANIFEST.keys()),
        help="Optional subset of sequence ids to display.",
    )
    return parser.parse_args()


def format_size(status_text: str) -> str:
    return status_text.split(" ", 1)[1] if " " in status_text else status_text


def build_snapshot(dataset_root: Path, sequence_ids: list[str]) -> dict[str, object]:
    sequences: list[dict[str, object]] = []
    for sequence_id in sequence_ids:
        asset = MANIFEST[sequence_id]
        sequence_root = dataset_root / sequence_id
        hdf5_text, hdf5_ratio = describe_file(sequence_root, asset.raw_h5_name, min_bytes=BIG_FILE_MIN_BYTES)
        mp4_text, mp4_ratio = describe_file(sequence_root, asset.video_name, min_bytes=BIG_FILE_MIN_BYTES)
        bag_text, bag_ratio = describe_file(sequence_root, asset.bag_name, min_bytes=BIG_FILE_MIN_BYTES)
        gt_text, gt_ratio = describe_file(sequence_root, "gt.hdf5", min_bytes=BIG_FILE_MIN_BYTES)
        ttc_text, ttc_ratio = describe_file(sequence_root, "ttc.csv", min_bytes=TEXT_FILE_MIN_BYTES)
        leftlabel_text, leftlabel_ratio = describe_leftlabel(sequence_root, asset)
        parts = [hdf5_ratio, mp4_ratio, bag_ratio, gt_ratio, ttc_ratio, leftlabel_ratio]
        percent = round(sum(parts) / max(1, len(parts)) * 100.0, 1)
        sequences.append(
            {
                "sequence_id": sequence_id,
                "overall_percent": percent,
                "assets": {
                    "hdf5": hdf5_text,
                    "mp4": mp4_text,
                    "bag": bag_text,
                    "gt": gt_text,
                    "ttc": ttc_text,
                    "leftlabel": leftlabel_text,
                },
            }
        )
    return {
        "dataset_root": str(dataset_root),
        "downloader_running": bool(list_download_pids()),
        "pids": list_download_pids(),
        "sequences": sequences,
    }


def render_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CCRs-1 下载进度</title>
  <style>
    :root {
      --bg: #f5f0e8;
      --panel: #fffaf4;
      --ink: #1f2a37;
      --muted: #6b7280;
      --ok: #0f766e;
      --wait: #9a6700;
      --dl: #1d4ed8;
      --down: #b91c1c;
      --line: #d7cbb9;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #efe7db 0%, var(--bg) 100%);
      color: var(--ink);
      font: 15px/1.4 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 18px;
    }
    .title {
      font-size: 28px;
      font-weight: 700;
    }
    .sub {
      color: var(--muted);
      margin-top: 4px;
    }
    .badge {
      padding: 10px 14px;
      border-radius: 999px;
      font-weight: 700;
      background: #e7f7f4;
      color: var(--ok);
      border: 1px solid #bde8e1;
    }
    .badge.down {
      background: #fdeaea;
      color: var(--down);
      border-color: #f5bcbc;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
      box-shadow: 0 10px 30px rgba(64, 46, 19, 0.06);
    }
    .meta { margin-bottom: 14px; color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }
    .seq {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }
    .seq-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 10px;
    }
    .seq-name { font-weight: 700; font-size: 17px; }
    .pct { font-weight: 700; }
    .bar {
      height: 10px;
      background: #eee4d6;
      border-radius: 999px;
      overflow: hidden;
      margin-bottom: 12px;
    }
    .fill {
      height: 100%;
      background: linear-gradient(90deg, #2563eb 0%, #0f766e 100%);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    td {
      padding: 4px 0;
      vertical-align: top;
      border-top: 1px dashed #efe6da;
    }
    td:first-child {
      width: 84px;
      color: var(--muted);
    }
    .ok { color: var(--ok); }
    .wait { color: var(--wait); }
    .dl { color: var(--dl); }
    .footer {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <div class="title">CCRs-1 下载面板</div>
        <div class="sub">浏览器会每 3 秒自动刷新一次，不需要再问后端有没有挂。</div>
      </div>
      <div id="badge" class="badge">检查中</div>
    </div>
    <div class="card">
      <div id="meta" class="meta">加载中...</div>
      <div id="grid" class="grid"></div>
      <div class="footer">如果下载器崩了，这里会明确显示 `not running`。</div>
    </div>
  </div>
  <script>
    function klass(text) {
      if (text.startsWith('OK')) return 'ok';
      if (text.startsWith('DL')) return 'dl';
      return 'wait';
    }
    function renderSequence(item) {
      const rows = Object.entries(item.assets).map(([key, value]) => `
        <tr><td>${key}</td><td class="${klass(value)}">${value}</td></tr>
      `).join('');
      return `
        <div class="seq">
          <div class="seq-head">
            <div class="seq-name">${item.sequence_id}</div>
            <div class="pct">${item.overall_percent.toFixed(1)}%</div>
          </div>
          <div class="bar"><div class="fill" style="width:${item.overall_percent}%;"></div></div>
          <table>${rows}</table>
        </div>
      `;
    }
    async function refresh() {
      const res = await fetch('/status.json?ts=' + Date.now(), { cache: 'no-store' });
      const data = await res.json();
      const badge = document.getElementById('badge');
      badge.textContent = data.downloader_running ? 'Downloader Running' : 'Downloader Not Running';
      badge.className = 'badge' + (data.downloader_running ? '' : ' down');
      document.getElementById('meta').textContent =
        `Dataset: ${data.dataset_root} | PIDs: ${data.pids.length ? data.pids.join(' ; ') : 'none'}`;
      document.getElementById('grid').innerHTML = data.sequences.map(renderSequence).join('');
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>"""


def make_handler(dataset_root: Path, sequence_ids: list[str]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                payload = render_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if parsed.path == "/status.json":
                snapshot = build_snapshot(dataset_root, sequence_ids)
                payload = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            payload = b"Not Found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return Handler


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    sequence_ids = args.sequences
    server = ThreadingHTTPServer((args.host, args.port), make_handler(dataset_root, sequence_ids))
    print(f"http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
