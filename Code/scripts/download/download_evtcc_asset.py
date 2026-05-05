#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import http.cookiejar
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024


def extract_drive_file_id(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return None

    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)

    query = urllib.parse.parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return query["id"][0]
    return None


def build_download_url(url: str) -> str:
    file_id = extract_drive_file_id(url)
    if file_id is None:
        return url
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def build_opener() -> urllib.request.OpenerDirector:
    cookie_jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))


def maybe_get_confirm_token(body: str) -> str | None:
    patterns = [
        r'name="confirm"\s+value="([^"]+)"',
        r"confirm=([0-9A-Za-z_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, body)
        if match:
            return html.unescape(match.group(1))
    return None


def request_with_optional_resume(
    opener: urllib.request.OpenerDirector,
    url: str,
    offset: int,
) -> urllib.response.addinfourl:
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "Mozilla/5.0")
    if offset > 0:
        request.add_header("Range", f"bytes={offset}-")
    return opener.open(request)


def resolve_google_drive_response(
    opener: urllib.request.OpenerDirector,
    url: str,
    offset: int,
) -> urllib.response.addinfourl:
    response = request_with_optional_resume(opener, url, offset)
    content_type = response.headers.get("Content-Type", "")
    disposition = response.headers.get("Content-Disposition", "")
    if "drive.google.com" not in url or disposition:
        return response
    if "text/html" not in content_type:
        return response

    body = response.read().decode("utf-8", errors="ignore")
    confirm_token = maybe_get_confirm_token(body)
    if confirm_token is None:
        raise RuntimeError("Google Drive confirm token not found. Open the link in browser and verify access.")

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    query["confirm"] = [confirm_token]
    confirmed = urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
    )
    return request_with_optional_resume(opener, confirmed, offset)


def format_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size}B"


def download(url: str, output: Path, resume: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_suffix(output.suffix + ".part")
    if resume and output.exists():
        print(f"[Skip] already exists: {output}")
        return

    offset = temp_path.stat().st_size if resume and temp_path.exists() else 0
    opener = build_opener()
    download_url = build_download_url(url)
    response = resolve_google_drive_response(opener, download_url, offset)

    status = getattr(response, "status", None)
    append_mode = offset > 0 and status == 206
    if offset > 0 and not append_mode:
        print("[Info] server did not honor resume; restarting download from 0")
        offset = 0

    total_size = None
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total_size = int(content_range.rsplit("/", 1)[1])
    else:
        content_length = response.headers.get("Content-Length")
        if content_length:
            total_size = offset + int(content_length)

    mode = "ab" if append_mode else "wb"
    downloaded = offset if append_mode else 0
    with temp_path.open(mode) as handle:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded / total_size * 100.0
                print(
                    f"\r[Downloading] {output.name}: {percent:6.2f}% "
                    f"({format_size(downloaded)}/{format_size(total_size)})",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\r[Downloading] {output.name}: {format_size(downloaded)}",
                    end="",
                    flush=True,
                )
    print()
    temp_path.replace(output)
    print(f"[Done] saved to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a single EvTTC / Google Drive asset with resume support."
    )
    parser.add_argument("--url", required=True, help="Direct URL or Google Drive share link.")
    parser.add_argument("--output", type=Path, required=True, help="Output file path.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Restart from zero even if a partial .part file exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        download(url=args.url, output=args.output.resolve(), resume=not args.no_resume)
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
