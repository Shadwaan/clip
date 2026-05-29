"""
Video ingestion from URLs.

Two paths:
  1. Direct media URL (.mp4, .mov, etc.) → stream via httpx
  2. Anything else (YouTube, Vimeo, X, TikTok, ...) → yt-dlp

yt-dlp handles ~1,000 sites and is actively maintained (the youtube-dl fork
that actually works). It's a heavy import (~100ms) so we lazy-load it.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

# Direct-fetchable media extensions (skip yt-dlp for these — faster)
DIRECT_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

# Hard limits to prevent abuse
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
MAX_DURATION_SEC = 2 * 3600                  # 2 hours
DOWNLOAD_TIMEOUT = 600                       # 10 min


class DownloadError(Exception):
    """Could not retrieve video from URL."""


class DownloadTooLarge(DownloadError):
    pass


class DownloadTooLong(DownloadError):
    pass


def _is_direct_media_url(url: str) -> bool:
    """Heuristic: does the URL end in a known media extension?"""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in DIRECT_EXTS)


def _safe_filename(stem: str) -> str:
    """Strip non-alphanumeric chars to make a safe filename stem."""
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem)[:50]
    return stem or "video"


def fetch_url(url: str, dest_dir: Path) -> tuple[Path, dict]:
    """
    Fetch a video from any URL.

    Returns (downloaded_path, metadata_dict).
    metadata_dict may include: title, duration_sec, uploader, source_url, ext.

    Raises DownloadError on any failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if _is_direct_media_url(url):
        return _fetch_direct(url, dest_dir)
    return _fetch_via_ytdlp(url, dest_dir)


def _fetch_direct(url: str, dest_dir: Path) -> tuple[Path, dict]:
    """Stream a direct media URL to disk."""
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower() or ".mp4"
    stem = _safe_filename(Path(parsed.path).stem)
    dest = dest_dir / f"{uuid.uuid4().hex}_{stem}{ext}"

    log.info(f"Downloading direct: {url}")
    size = 0
    try:
        with httpx.stream("GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as r:
            r.raise_for_status()
            content_length = int(r.headers.get("content-length", 0))
            if content_length > MAX_DOWNLOAD_BYTES:
                raise DownloadTooLarge(
                    f"Content-length {content_length} exceeds limit {MAX_DOWNLOAD_BYTES}"
                )
            with dest.open("wb") as f:
                for chunk in r.iter_bytes(8 * 1024 * 1024):
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        f.close()
                        dest.unlink(missing_ok=True)
                        raise DownloadTooLarge(f"Download exceeded {MAX_DOWNLOAD_BYTES} bytes")
                    f.write(chunk)
    except httpx.HTTPError as e:
        dest.unlink(missing_ok=True)
        raise DownloadError(f"HTTP error fetching {url}: {e}") from e

    return dest, {
        "title": stem,
        "source_url": url,
        "ext": ext.lstrip("."),
        "source_type": "direct",
        "size_bytes": size,
    }


def _fetch_via_ytdlp(url: str, dest_dir: Path) -> tuple[Path, dict]:
    """Use yt-dlp to fetch from YouTube / Vimeo / etc."""
    # Lazy import — yt-dlp is heavy
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError as YTDLError
    except ImportError as e:
        raise DownloadError(
            "yt-dlp not installed. `pip install yt-dlp` to enable URL downloads."
        ) from e

    # --- Step 1: probe metadata only (no download) ---
    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    try:
        with YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except YTDLError as e:
        raise DownloadError(f"yt-dlp probe failed: {e}") from e

    duration = info.get("duration") or 0
    if duration > MAX_DURATION_SEC:
        raise DownloadTooLong(
            f"Video is {duration}s, exceeds limit {MAX_DURATION_SEC}s"
        )

    title = _safe_filename(info.get("title", "video"))
    out_template = str(dest_dir / f"{uuid.uuid4().hex}_{title}.%(ext)s")

    # --- Step 2: actually download ---
    # Prefer mp4 ≤720p to keep things reasonable; fall back to best available.
    download_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "outtmpl": out_template,
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        # Don't write description / thumbnail / subs files
        "writethumbnail": False,
        "writeinfojson": False,
    }
    log.info(f"Downloading via yt-dlp: {url} (title={title!r}, duration={duration}s)")
    try:
        with YoutubeDL(download_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = Path(ydl.prepare_filename(info))
    except YTDLError as e:
        raise DownloadError(f"yt-dlp download failed: {e}") from e

    if not downloaded_path.exists():
        raise DownloadError(f"yt-dlp reported success but file missing: {downloaded_path}")

    size = downloaded_path.stat().st_size
    if size > MAX_DOWNLOAD_BYTES:
        downloaded_path.unlink(missing_ok=True)
        raise DownloadTooLarge(f"Downloaded file {size} bytes exceeds limit")

    return downloaded_path, {
        "title": info.get("title", title),
        "uploader": info.get("uploader"),
        "source_url": url,
        "ext": downloaded_path.suffix.lstrip("."),
        "duration_hint": duration,
        "source_type": "yt-dlp",
        "size_bytes": size,
    }
