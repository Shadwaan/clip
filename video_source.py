"""
Shared video-input helper for the Clip Modal apps (modal_app.py, modal_ocr.py).

`_materialize_video(video_bytes, video_url, video_ext)` writes the incoming video
to a tempfile on the container's disk and returns its path — the single thing all
the GPU methods need before they can decode. It supports two mutually-exclusive
input modes:

  • video_bytes  — the existing path. The gateway / scaffold / CI ship raw bytes
    over the Modal wire; we write them to a tempfile exactly as before. ZERO
    behaviour change for this path.
  • video_url    — new, additive. Stream-download an http(s) URL straight to the
    tempfile (never holding the whole video in memory), so callers (e.g. the
    Vercel app) can hand Modal a URL instead of multi-MB payloads.

Stdlib only — urllib.request + shutil. No new image dependency.

SSRF hardening (this code runs with Modal credentials; a URL it fetches must
never be an internal address):
  • http/https schemes only — file://, ftp://, etc. are rejected.
  • The hostname is RESOLVED and every resulting IP is checked; private (RFC1918),
    loopback (127.*/::1), link-local (169.254.* / fe80::), and other non-global
    ranges are rejected. We check the IP, not the string, so DNS names that point
    at internal hosts are caught.
  • Redirects are followed only after re-validating each hop's destination IP, so
    a public URL can't 302 us onto an internal address.
Residual: DNS rebinding between our resolution and urllib's own connect is not
defended (would require pinning the socket to the validated IP). Out of scope.
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlparse

# Stream the download in chunks so a large video never lands fully in memory.
_DOWNLOAD_CHUNK = 1024 * 1024          # 1 MiB
_MIN_VIDEO_BYTES = 1024                # < 1 KB is almost certainly an error page
_HTTP_TIMEOUT = 30                     # seconds


def _check_public_ip(host: str, scheme: str) -> None:
    """Resolve `host` and raise ValueError if ANY resolved IP is non-global
    (private / loopback / link-local / reserved / multicast / unspecified)."""
    port = 443 if scheme == "https" else 80
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"video_url host {host!r} did not resolve: {e}") from e

    for info in infos:
        ip_str = info[4][0]
        ip = ipaddress.ip_address(ip_str)
        # is_global is False for private/loopback/link-local/reserved/etc.
        # Check the specific flags too for a clearer error message.
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified
                or not ip.is_global):
            raise ValueError(
                f"video_url host {host!r} resolves to non-public address "
                f"{ip_str} — refusing to fetch (SSRF guard)."
            )


def _assert_safe_url(url: str) -> None:
    """Validate scheme (http/https) and that the host resolves to a public IP.
    Reused for the initial URL and for every redirect hop."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"video_url scheme must be http or https, got {parsed.scheme!r} "
            f"(in {url!r})."
        )
    if not parsed.hostname:
        raise ValueError(f"video_url has no host: {url!r}")
    _check_public_ip(parsed.hostname, parsed.scheme)


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate each redirect target before following it, so a public URL
    can't bounce us onto file:// or an internal address."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _assert_safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_GuardedRedirectHandler())


def _download_to(path: str, video_url: str) -> None:
    """Stream `video_url` into the file at `path`. Validates scheme/IP first,
    enforces 2xx + a sane minimum body size, and surfaces the status code and a
    snippet of the body in errors so failures are diagnosable."""
    _assert_safe_url(video_url)
    req = urllib.request.Request(
        video_url, headers={"User-Agent": "clip-modal/1.0"}
    )
    try:
        resp = _opener.open(req, timeout=_HTTP_TIMEOUT)
    except urllib.error.HTTPError as e:
        # Non-2xx (urllib raises here). Surface status + first 200 chars of body.
        try:
            body = e.read(2048).decode("utf-8", "replace")
        except Exception:
            body = "<unreadable body>"
        raise ValueError(
            f"video_url returned HTTP {e.code}; first 200 chars of body: "
            f"{body[:200]!r}"
        ) from e
    except urllib.error.URLError as e:
        raise ValueError(f"video_url could not be fetched: {e.reason}") from e

    with resp:
        status = getattr(resp, "status", None) or resp.getcode()
        if not (200 <= int(status) < 300):
            head = resp.read(2048).decode("utf-8", "replace")
            raise ValueError(
                f"video_url returned HTTP {status}; first 200 chars of body: "
                f"{head[:200]!r}"
            )
        with open(path, "wb") as out:
            shutil.copyfileobj(resp, out, _DOWNLOAD_CHUNK)

    size = os.path.getsize(path)
    if size < _MIN_VIDEO_BYTES:
        with open(path, "rb") as f:
            head = f.read(200).decode("utf-8", "replace")
        raise ValueError(
            f"video_url body is only {size} bytes (< {_MIN_VIDEO_BYTES}); "
            f"likely an error page, not a video. First 200 chars: {head!r}"
        )


def _materialize_video(
    video_bytes: bytes | None = None,
    video_url: str | None = None,
    video_ext: str = "mp4",
) -> str:
    """Write the video to a tempfile and return its path. EXACTLY ONE of
    `video_bytes` or `video_url` must be provided.

    Caller owns cleanup of the returned path (the existing try/finally
    os.unlink in each method already does this). On any failure the tempfile is
    removed here so we never leak a partial download.
    """
    if (video_bytes is None) == (video_url is None):
        raise ValueError(
            "Provide exactly one of video_bytes or video_url "
            f"(got video_bytes={'set' if video_bytes is not None else 'None'}, "
            f"video_url={'set' if video_url is not None else 'None'})."
        )

    fd, path = tempfile.mkstemp(suffix=f".{video_ext}")
    os.close(fd)
    try:
        if video_url is not None:
            _download_to(path, video_url)
        else:
            with open(path, "wb") as f:
                f.write(video_bytes)
        return path
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
