"""
Modal deployment for Clip's on-screen URL extraction (OCR pipeline).

Why this file exists
--------------------
This is a SEPARATE Modal app from `clip-marlin` (modal_app.py). The OCR
layer reads URLs that are visually present in video frames (screenshares,
lower-thirds, slides, browser address bars) and returns them with the
timestamps where they appeared. It is deliberately decoupled from the VLM
captioning app so it can be iterated and redeployed without any risk to the
production `clip-marlin` deployment.

See docs/ONSCREEN_LINKS.md for the full design rationale (why OCR instead of
prompting the VLM, why EasyOCR, why fixed 2 fps for v0, what's out of scope).

How it works
------------
- `OCRModel` is a Modal class on a GPU container. It loads an EasyOCR reader
  once (via `@modal.enter`) and keeps it warm for `scaledown_window` seconds.
- `extract_links(video_bytes, video_ext)` is the remote method:
    1. write bytes to a tempfile,
    2. ffmpeg-sample frames at 2 fps to JPGs (timestamp = frame_index / 2),
    3. EasyOCR each frame → detected text strings,
    4. URL regex over each frame's text,
    5. dedupe identical URLs across frames into one entry with
       first_seen / last_seen / occurrences,
    6. return {"links": [...], "frames_processed": int}.

Deploy
------
    modal deploy modal_ocr.py

Smoke test (after deploy):
    curl https://<your-workspace>--clip-ocr-healthz.modal.run

EasyOCR's detection + recognition models (~100 MB for English) download on
the first cold start into a persistent Modal Volume, so subsequent cold
starts reuse them.
"""

import modal

APP_NAME = "clip-ocr"

GPU_TYPE = "A10G"           # 24 GB; EasyOCR is light, but GPU makes 2 fps sampling tractable
CACHE_DIR = "/cache"        # EasyOCR model_storage_directory (persisted on the volume)
SAMPLE_FPS = 2              # v0: fixed 2 fps sampling (see docs §Sampling)
OCR_LANGS = ["en"]          # v0: English only

# --- Container image ---
# ffmpeg for frame extraction; easyocr pulls its own torch/opencv/numpy stack.
# Baked at build time so cold starts don't pip-install.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        # EasyOCR brings torch, torchvision, opencv-python-headless, numpy,
        # Pillow, scikit-image, python-bidi, shapely, pyclipper, etc. as deps.
        "easyocr>=1.7.0",
        # Required by `@modal.fastapi_endpoint` (healthz). Modal no longer
        # installs FastAPI automatically.
        "fastapi[standard]>=0.115.0",
    )
)

app = modal.App(APP_NAME, image=image)

# Dedicated volume for EasyOCR weights — kept separate from clip-marlin's
# `clip-model-cache` so the two apps share no state and can be wiped/iterated
# independently.
ocr_cache = modal.Volume.from_name("clip-ocr-cache", create_if_missing=True)


@app.cls(
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: ocr_cache},
    # OCR over many frames at 2 fps is slow on long videos (a 10-min clip is
    # ~1200 frames). Budget generously, same ceiling as the captioning app.
    timeout=1800,
    scaledown_window=120,   # keep warm 2 min after last request
    retries=0,
)
class OCRModel:
    """Loaded once per container; reused for every extract_links call."""

    @modal.enter()
    def load(self):
        """Run on container startup. Builds the EasyOCR reader (downloads
        weights into the volume on first cold start)."""
        import easyocr

        print(f"[modal_ocr] Building EasyOCR reader (langs={OCR_LANGS}, gpu=True)...")
        # model_storage_directory points EasyOCR at the persistent volume so
        # the ~100 MB English models download once across cold starts.
        self.reader = easyocr.Reader(
            OCR_LANGS,
            gpu=True,
            model_storage_directory=CACHE_DIR,
            download_enabled=True,
        )
        ocr_cache.commit()  # persist any newly-downloaded weights
        print("[modal_ocr] EasyOCR reader ready.")

    @modal.method()
    def extract_links(
        self,
        video_bytes: bytes,
        video_ext: str = "mp4",
    ) -> dict:
        """
        Extract on-screen URLs with timestamps from a video.

        Returns:
            {
              "links": [
                {"url": str, "first_seen": float, "last_seen": float,
                 "occurrences": int},
                ...
              ],
              "frames_processed": int,
            }

        `links` is sorted by first_seen. `occurrences` counts the number of
        sampled frames a URL appeared in (multiple hits within one frame
        count once). See docs/ONSCREEN_LINKS.md for the URL-detection and
        dedupe rules.
        """
        import os
        import re
        import subprocess
        import tempfile
        from urllib.parse import urlparse

        # ---- URL detection (v0) -------------------------------------------
        # Anchor on an explicit http(s):// scheme OR a leading "www." — both
        # are unambiguous URL signals. We deliberately do NOT match bare
        # domains (e.g. "main.py", "index.io") because OCR of code/terminals
        # produces filenames whose extensions collide with real ccTLDs
        # (.py, .io, .sh, .rs). Requiring the scheme/www anchor keeps
        # precision high — a missed URL is a v0 quality note; a wrong
        # clickable URL is worse. Known limitation: scheme-less on-screen
        # URLs ("github.com/foo") are skipped in v0.
        url_re = re.compile(
            r"(?i)\b(?:https?://|www\.)"   # required anchor: scheme or www.
            r"[a-z0-9\-._~%]+"             # host (labels + dots)
            r"\.[a-z]{2,24}"               # final dot + alphabetic TLD (rejects 1.2.3, v4.46.0)
            r"(?::\d{2,5})?"               # optional :port
            r"(?:/[^\s]*)?"                # optional /path?query#fragment
        )
        # Trailing punctuation that OCR / sentence context can glue onto a URL.
        trailing = ".,;:!?\"')]}>«»"

        def canonicalize(raw_url: str) -> str:
            """Dedupe key + display form. Lowercase scheme+host, strip trailing
            slashes; preserve path/query/fragment case (paths are case-sensitive).
            A leading 'www.' (no scheme) is promoted to http:// so the result is
            always a clickable absolute URL."""
            u = raw_url.rstrip(trailing)
            if u.lower().startswith("www."):
                u = "http://" + u
            p = urlparse(u)
            scheme = (p.scheme or "http").lower()
            host = p.netloc.lower()
            tail = p.path.rstrip("/")
            if p.query:
                tail += "?" + p.query
            if p.fragment:
                tail += "#" + p.fragment
            return f"{scheme}://{host}{tail}"

        # ---- write bytes to disk for ffmpeg --------------------------------
        with tempfile.NamedTemporaryFile(
            suffix=f".{video_ext}", delete=False
        ) as f:
            f.write(video_bytes)
            video_path = f.name

        frame_dir = tempfile.mkdtemp(prefix="ocr_frames_")
        # clusters: canonical_url -> {"url", "first_seen", "last_seen", "occurrences"}
        clusters: dict[str, dict] = {}
        frames_processed = 0

        try:
            # Sample frames at SAMPLE_FPS. The fps filter emits frames at
            # t = 0, 1/fps, 2/fps, ...; the k-th output file (0-indexed after
            # sorting) is therefore at k / fps seconds.
            frame_pattern = os.path.join(frame_dir, "frame_%06d.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 "-vf", f"fps={SAMPLE_FPS}", "-q:v", "2", frame_pattern],
                capture_output=True, check=True,
            )

            frame_files = sorted(
                fn for fn in os.listdir(frame_dir) if fn.endswith(".jpg")
            )
            frames_processed = len(frame_files)
            print(f"[modal_ocr] Sampled {frames_processed} frames @ {SAMPLE_FPS} fps")

            for idx, fn in enumerate(frame_files):
                timestamp = idx / float(SAMPLE_FPS)
                fpath = os.path.join(frame_dir, fn)

                # detail=0 → list of detected text strings (no bboxes/scores).
                try:
                    texts = self.reader.readtext(fpath, detail=0, paragraph=False)
                except Exception as e:
                    # A single unreadable frame shouldn't abort the whole job.
                    print(f"[modal_ocr] OCR failed on {fn}: {e}")
                    continue

                blob = " ".join(texts)
                # Unique canonical URLs in THIS frame (so multiple hits in one
                # frame count as a single occurrence).
                seen_in_frame = {canonicalize(m) for m in url_re.findall(blob)}
                for url in seen_in_frame:
                    entry = clusters.get(url)
                    if entry is None:
                        clusters[url] = {
                            "url": url,
                            "first_seen": timestamp,
                            "last_seen": timestamp,
                            "occurrences": 1,
                        }
                    else:
                        entry["last_seen"] = timestamp
                        entry["occurrences"] += 1

            links = sorted(clusters.values(), key=lambda e: e["first_seen"])
            print(f"[modal_ocr] Found {len(links)} unique URL(s)")
            return {"links": links, "frames_processed": frames_processed}
        finally:
            # Clean up the tempfile + all extracted frames.
            try:
                os.unlink(video_path)
            except OSError:
                pass
            for fn in os.listdir(frame_dir):
                try:
                    os.unlink(os.path.join(frame_dir, fn))
                except OSError:
                    pass
            try:
                os.rmdir(frame_dir)
            except OSError:
                pass


# Healthcheck — does NOT load the OCR model (no GPU), free to curl after deploy.
@app.function()
@modal.fastapi_endpoint(method="GET")
def healthz():
    return {
        "status": "ok",
        "app": APP_NAME,
        "gpu": GPU_TYPE,
        "sample_fps": SAMPLE_FPS,
        "langs": OCR_LANGS,
    }


# Optional local smoke test:
#     modal run modal_ocr.py --video-path some_clip.mp4
# Runs the full extract_links pipeline against the deployed container.
@app.local_entrypoint()
def main(video_path: str = ""):
    if not video_path:
        print("Pass --video-path <file> to run an end-to-end OCR smoke test.")
        return
    import pathlib
    p = pathlib.Path(video_path)
    ext = p.suffix.lstrip(".") or "mp4"
    result = OCRModel().extract_links.remote(
        video_bytes=p.read_bytes(), video_ext=ext
    )
    print(f"frames_processed={result['frames_processed']}")
    for link in result["links"]:
        print(f"  [{link['first_seen']:.1f}s - {link['last_seen']:.1f}s] "
              f"x{link['occurrences']}  {link['url']}")
