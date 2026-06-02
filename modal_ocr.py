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
prompting the VLM, sampling/cost guards, what's out of scope).

OCR engine: PaddleOCR (switched from EasyOCR 2026-06-02). EasyOCR read the
on-screen URLs but mangled dense-path *punctuation* — "/" as "l"/"f", "=" as
"-" — so the recovered URLs 404'd. PaddleOCR benchmarks materially better on
dense screen text; we also add a post-correction pass (_postcorrect) that
repairs the residual character substitutions before URL parsing.

How it works
------------
- `OCRModel` is a Modal class on a GPU container. It loads a PaddleOCR reader
  once (via `@modal.enter`) and keeps it warm for `scaledown_window` seconds.
- `extract_links(video_bytes, video_ext)` is the remote method:
    1. write bytes to a tempfile,
    2. ffmpeg-sample frames (up to SAMPLE_FPS, capped to MAX_FRAMES total;
       optionally downscaled to DOWNSCALE_WIDTH) to JPGs, recording timestamps,
    3. PaddleOCR each frame → detected text strings,
    4. post-correct OCR character substitutions, then URL-extract per string,
    5. dedupe identical URLs across frames into one entry with
       first_seen / last_seen / occurrences,
    6. return {"links": [...], "frames_processed": int}.

Deploy
------
    modal deploy modal_ocr.py

Smoke test (after deploy):
    curl https://<your-workspace>--clip-ocr-healthz.modal.run

PaddleOCR's detection + recognition + angle-cls models (~10-20 MB) download on
the first cold start into a persistent Modal Volume (HOME=CACHE_DIR points
~/.paddleocr there), so subsequent cold starts reuse them.
"""

import modal
import re

APP_NAME = "clip-ocr"

GPU_TYPE = "A10G"           # 24 GB; OCR is light, GPU makes dense-frame OCR tractable
CACHE_DIR = "/cache"        # HOME → here so PaddleOCR caches models on the volume
SAMPLE_FPS = 2              # nominal max sampling rate (used for short videos)
OCR_LANG = "en"             # v0: English only

# Cost/latency guards (added 2026-06-02 after a 15-min meeting recording produced
# ~1,815 frames at 2 fps and ran 40+ min). Dense screen-recording frames are slow
# because EasyOCR runs recognition once per detected text box, and meeting UIs
# have hundreds per frame — so we (a) CAP the total frames OCR'd regardless of
# duration, and (b) DOWNSCALE wide frames before OCR. Both are tunable: raise
# MAX_FRAMES / DOWNSCALE_WIDTH if on-screen URLs are being missed; lower for speed.
MAX_FRAMES = 300            # hard cap on frames OCR'd per video (effective fps = MAX_FRAMES/duration, capped at SAMPLE_FPS)
DOWNSCALE_WIDTH = 0         # downscale width (px) before OCR; 0 = OFF. 1280 was too aggressive
                            # for high-DPI small-text sources — a 2560px meeting recording lost
                            # its chat URL after the 2x shrink. MAX_FRAMES is the main speed lever,
                            # so we keep full resolution for URL legibility (recall > a bit of speed).
OCR_PROGRESS_EVERY = 25     # heartbeat: log "frame N/total" every N frames

# Diagnostic: surface raw URL-ish OCR text even when the regex rejects it, so we can
# tell "reader didn't see the URL" (resolution) from "reader garbled it" (regex). Cheap
# to keep on; it only collects strings that already look link-ish.
OCR_DEBUG_CANDIDATES = True
_CANDIDATE_TOKENS = ("http", "://", "www.", ".com", ".live", ".net", ".org", ".io", "meet", "teams")


# --- OCR character-substitution post-correction --------------------------------
# OCR (EasyOCR especially) mis-reads narrow / ambiguous glyphs in dense URL paths.
# Validated 2026-06-02 against the EasyOCR baseline (links_meeting_771436c4.txt):
# the rules below fix every observed mangle while leaving correct URLs, path
# hyphens, hash values, and prose untouched. Applied per OCR string BEFORE URL
# parsing. Each rule is tightly constrained — under-correct rather than mangle
# (see the no-regression note on '&'→'8' below). NOTE: PaddleOCR's output is clean
# enough that these rarely fire (0 hits on the meeting A/B) — kept as a safety net.
_URLISH = re.compile(
    r"(?i)(https?\s*[:/]"
    # space-mangled host: a TLD word, then port digits, then a slash
    r"|\b(?:com|net|org|gov|edu|live|io|bd|co)\b\s*[.\s:]*\d{2,5}\s*/"
    r"|\.(?:com|net|org|gov|edu|live|io|bd|co)\b)"
)


def _postcorrect(text: str) -> str:
    """Repair OCR character substitutions inside URL-shaped strings only."""
    if not _URLISH.search(text):
        return text
    t = text
    # NOTE: the '&' -> '8' substitution is deliberately NOT corrected. A real
    # trailing digit '8' before a param ("70028&p3309_id") is indistinguishable
    # from a substituted '&' ("7002&..."), so reversing it corrupts genuine
    # values. Backed out per the no-regression rule — under-correct, don't mangle.
    #
    # (1) '=' read as '-' in query params: "&session-123" -> "&session=123",
    #     "?p3309_loi_id-70028" -> "...=70028". Fires only when a '?'/'&'-led
    #     param token is followed by '-' then a DIGIT — so it never touches path
    #     hyphens ("purchase-requisition") or hash hyphens ("cs=3viHxGZn-ZfEmV",
    #     where '-' precedes a letter).
    t = re.sub(r"([?&][A-Za-z0-9_]+)-(?=\d)", r"\1=", t)
    # (2) '/' read as 'l'/'f' right after the ERP/APEX path keyword "erp":
    #     "erplinternal"/"erpfinternal" -> "erp/internal", "erpllogin" ->
    #     "erp/login". Can't fire on a clean "erp/..." (no l/f there) nor on the
    #     host "erpdev" ('d' isn't l/f).
    t = re.sub(r"(?i)\berp([lf])(?=[a-z])", "erp/", t)
    # (3) '/' read as 'l' in the two observed APEX route prefixes:
    #     "dev/r" -> "devlr", "ords/r/" -> "ords/rl".
    t = re.sub(r"(?i)\bdevlr(?=[/a-z])", "dev/r", t)
    t = re.sub(r"(?i)\bords/rl(?=[a-z])", "ords/r/", t)
    return t


# --- Container image ---
# Build on PaddlePaddle's OFFICIAL GPU image: paddle 2.6.2 + CUDA + cuDNN are
# pre-installed and matched, with the loader paths configured so cuDNN actually
# loads. Our first attempt (nvidia/cuda base + pip paddlepaddle-gpu) built fine
# but crashed every frame at inference: "(PreconditionNotMet) Cannot load cudnn
# shared library" — paddle couldn't find/load cuDNN on that base. The official
# image fixes that by construction.
#
# Tag note: the requested cuda11.8-cudnn8.6 tag does not exist on Docker Hub;
# the nearest 11.x GPU tag is cuda11.7-cudnn8.4-trt8.4 (verified against
# hub.docker.com/r/paddlepaddle/paddle/tags). We deliberately do NOT pass
# add_python — that would create a fresh interpreter that can't see the image's
# pre-installed paddle (which we no longer pip-install), reintroducing the very
# cuDNN problem we're fixing. We use the image's own Python instead.
image = (
    modal.Image.from_registry("paddlepaddle/paddle:2.6.2-gpu-cuda11.7-cudnn8.4-trt8.4")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        # paddlepaddle-gpu is already in the base image — only add PaddleOCR + deps.
        # PaddleOCR pulls shapely, pyclipper, opencv-python, lmdb, etc.
        "paddleocr==2.7.3",
        # Keep NumPy <2: paddle 2.6 is built against the NumPy 1.x C ABI, and
        # paddleocr's deps would otherwise pull 2.x (runtime ImportError:
        # "numpy.core.multiarray failed to import").
        "numpy<2",
        # Required by `@modal.fastapi_endpoint` (healthz).
        "fastapi[standard]>=0.115.0",
    )
)

app = modal.App(APP_NAME, image=image)

# Dedicated volume for PaddleOCR weights — kept separate from clip-marlin's
# `clip-model-cache` so the two apps share no state and can be wiped/iterated
# independently.
ocr_cache = modal.Volume.from_name("clip-ocr-cache", create_if_missing=True)


@app.cls(
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: ocr_cache},
    # OCR is frame-bound; even with the MAX_FRAMES cap, dense frames can take
    # a few seconds each. Budget generously, same ceiling as the captioning app.
    timeout=1800,
    scaledown_window=120,   # keep warm 2 min after last request
    retries=0,
)
class OCRModel:
    """Loaded once per container; reused for every extract_links call."""

    @modal.enter()
    def load(self):
        """Run on container startup. Builds the PaddleOCR reader (downloads
        weights into the volume on first cold start)."""
        import os
        # PaddleOCR caches its models under ~/.paddleocr; point HOME at the
        # persistent volume so the download happens once across cold starts.
        os.environ["HOME"] = CACHE_DIR

        import paddle
        from paddleocr import PaddleOCR

        # Confirm GPU explicitly. PaddleOCR silently falls back to (very slow)
        # CPU if use_gpu=True but CUDA isn't usable — that kind of ambiguity
        # cost us a 40-min detour on EasyOCR, so log it loudly and pass through.
        gpu_ok = bool(paddle.is_compiled_with_cuda()) and paddle.device.cuda.device_count() > 0
        print(f"[modal_ocr] paddle.is_compiled_with_cuda()={paddle.is_compiled_with_cuda()} "
              f"cuda_device_count={paddle.device.cuda.device_count()} -> use_gpu={gpu_ok}")

        print(f"[modal_ocr] Building PaddleOCR reader (lang={OCR_LANG}, gpu={gpu_ok})...")
        # use_angle_cls handles rotated text; for screen captures it's rarely
        # needed but cheap. show_log=False keeps the per-call paddle spam down.
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=OCR_LANG,
            use_gpu=gpu_ok,
            show_log=False,
        )
        ocr_cache.commit()  # persist any newly-downloaded weights
        print("[modal_ocr] PaddleOCR reader ready.")

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

        # ---- URL detection -------------------------------------------------
        # Two passes, because real OCR mangles URL punctuation badly (diagnosed
        # 2026-06-02 on a meeting recording: "https://erpdev.hameemgroup.com:8443/…"
        # was read as "https /lerpdev hameemgroup com.8443/…" — the "://" became a
        # space, domain dots became spaces). EasyOCR gets the *letters* right but
        # the *punctuation* wrong, so a rigid syntax match finds nothing.
        #
        #   STRICT: clean, well-formed URLs (http(s):// or www.). High precision.
        #   FUZZY:  tolerates OCR noise — optional scheme, host labels separated
        #           by '.' OR spaces ending in a known TLD, mangled port, then path.
        #           Reconstructs scheme://host[:port]/path. Requires a strong URL
        #           signal (scheme OR port OR a "/path") so prose like "see us com"
        #           and bare emails ("erp@hameemgroup.com") don't become links.
        #
        # Caveat: FUZZY recovers the host+port reliably; the path can still carry
        # OCR letter errors (e.g. "erp/internal" read as "erpfinternal") — we keep
        # the raw OCR string in url_candidates for ground truth.
        KNOWN_TLDS = ("com", "net", "org", "io", "gov", "edu", "live", "info",
                      "biz", "co", "us", "uk", "bd", "app", "dev", "ai", "xyz",
                      "me", "tv")
        _tld_alt = "|".join(KNOWN_TLDS)
        strict_re = re.compile(
            r"(?i)\b(?:https?://|www\.)[a-z0-9\-._~%]+\.[a-z]{2,24}"
            r"(?::\d{2,5})?(?:/[^\s]*)?"
        )
        fuzzy_re = re.compile(
            r"(?i)(?:(https?)://)?"
            r"([a-z0-9][a-z0-9\-]*(?:[.\s]+[a-z0-9\-]+)*[.\s]+(?:" + _tld_alt + r"))"
            r"(?:[.\s:]+(\d{2,5}))?"
            r"((?:\s*/\s*)[^\n]*)?"
        )
        # Trailing punctuation that OCR / sentence context can glue onto a URL.
        trailing = ".,;:!?\"')]}>«» "

        def _norm_scheme(text: str) -> str:
            # Repair OCR-mangled scheme separators: "https /l", "https //",
            # "https:/i", "http : / /" -> "https://" / "http://".
            text = re.sub(r"(?i)\b(https?)\s*[:;]?\s*/\s*[/l|i]\s*", r"\1://", text)
            text = re.sub(r"(?i)\b(https?)\s*[:;]?\s*//", r"\1://", text)
            return text

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

        def urls_in_text(text: str) -> set:
            """Return canonical URLs found in a single OCR string, via the strict
            pass then the OCR-tolerant fuzzy pass. Run per-detected-string (not on
            the joined frame blob) so the space-tolerant fuzzy matcher can't stitch
            a fake URL across unrelated text boxes."""
            found = set()
            for m in strict_re.finditer(text):
                found.add(canonicalize(m.group(0)))
            for m in fuzzy_re.finditer(_norm_scheme(text)):
                scheme, host_raw, port, path = m.groups()
                has_path = bool(path and "/" in path)
                # Require a strong URL signal to avoid prose/email false positives.
                if not (scheme or port or has_path):
                    continue
                host = re.sub(r"[.\s]+", ".", host_raw).strip(".").lower()
                if host.count(".") < 1:
                    continue
                sch = scheme.lower() if scheme else ("https" if port in ("8443", "443") else "http")
                url = f"{sch}://{host}"
                if port:
                    url += f":{port}"
                if has_path:
                    pth = re.sub(r"\s+", "", path)   # OCR sprinkles spaces into the path
                    if not pth.startswith("/"):
                        pth = "/" + pth
                    url += pth.rstrip(trailing)
                found.add(canonicalize(url))
            return found

        # ---- write bytes to disk for ffmpeg --------------------------------
        with tempfile.NamedTemporaryFile(
            suffix=f".{video_ext}", delete=False
        ) as f:
            f.write(video_bytes)
            video_path = f.name

        frame_dir = tempfile.mkdtemp(prefix="ocr_frames_")
        # clusters: canonical_url -> {"url", "first_seen", "last_seen", "occurrences"}
        clusters: dict[str, dict] = {}
        # Diagnostic: raw OCR strings that look link-ish but may not pass the regex.
        url_candidates: list[dict] = []
        frames_processed = 0

        try:
            # Probe duration + width to (a) cap the total frame count and
            # (b) decide whether to downscale. ffprobe ships with ffmpeg.
            def _probe(entries: str, stream: bool = False) -> str:
                cmd = ["ffprobe", "-v", "error"]
                if stream:
                    cmd += ["-select_streams", "v:0"]
                cmd += ["-show_entries", entries,
                        "-of", "default=noprint_wrappers=1:nokey=1", video_path]
                out = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""

            try:
                duration = float(_probe("format=duration"))
            except Exception:
                duration = 0.0
            try:
                width = int(_probe("stream=width", stream=True))
            except Exception:
                width = 0

            # Effective fps: enough to stay under MAX_FRAMES, never above SAMPLE_FPS.
            # Short videos keep the full SAMPLE_FPS; long ones sample sparsely
            # (e.g. a 907s video → 300/907 ≈ 0.33 fps). The k-th output frame
            # (0-indexed after sorting) is at k / effective_fps seconds.
            if duration > 0:
                effective_fps = min(float(SAMPLE_FPS), MAX_FRAMES / duration)
            else:
                effective_fps = float(SAMPLE_FPS)

            vf = f"fps={effective_fps:.6f}"
            if DOWNSCALE_WIDTH and width and width > DOWNSCALE_WIDTH:
                # Only ever downscale (never upscale); -2 keeps aspect, even height.
                vf += f",scale={DOWNSCALE_WIDTH}:-2"

            frame_pattern = os.path.join(frame_dir, "frame_%06d.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 "-vf", vf, "-q:v", "2", frame_pattern],
                capture_output=True, check=True,
            )

            frame_files = sorted(
                fn for fn in os.listdir(frame_dir) if fn.endswith(".jpg")
            )
            frames_processed = len(frame_files)
            print(f"[modal_ocr] duration={duration:.1f}s width={width} -> "
                  f"effective_fps={effective_fps:.4f}, vf='{vf}', "
                  f"sampled {frames_processed} frames (cap {MAX_FRAMES})")

            for idx, fn in enumerate(frame_files):
                timestamp = idx / effective_fps if effective_fps > 0 else 0.0
                fpath = os.path.join(frame_dir, fn)

                # Heartbeat so a long run is observably progressing (the silence
                # here is what made the 1,815-frame run look hung).
                if (idx + 1) % OCR_PROGRESS_EVERY == 0:
                    print(f"[modal_ocr] OCR {idx + 1}/{frames_processed} frames, "
                          f"{len(clusters)} url(s) so far")

                # PaddleOCR returns a per-image list: result[0] is a list of
                # [box, (text, confidence)] entries, or None for a blank frame.
                try:
                    result = self.ocr.ocr(fpath, cls=True)
                except Exception as e:
                    # A single unreadable frame shouldn't abort the whole job.
                    print(f"[modal_ocr] OCR failed on {fn}: {e}")
                    continue
                texts = []
                if result and result[0]:
                    for line in result[0]:
                        try:
                            texts.append(line[1][0])
                        except (IndexError, TypeError):
                            continue

                # Unique canonical URLs in THIS frame (so multiple hits in one
                # frame count as a single occurrence). Post-correct OCR character
                # substitutions first, then extract per detected string so the
                # fuzzy matcher can't span unrelated text boxes.
                seen_in_frame = set()
                for s in texts:
                    corrected = _postcorrect(s)
                    seen_in_frame |= urls_in_text(corrected)

                    # Diagnostic: capture any link-ish OCR string (raw + corrected
                    # when they differ). Lets us A/B raw vs corrected and tell
                    # "didn't see the URL" from "read it garbled".
                    if OCR_DEBUG_CANDIDATES:
                        low = s.lower()
                        if any(tok in low for tok in _CANDIDATE_TOKENS):
                            note = f" -> {corrected!r}" if corrected != s else ""
                            print(f"[modal_ocr] url-ish @ {timestamp:.1f}s: {s!r}{note}")
                            if len(url_candidates) < 120:
                                entry = {"t": round(timestamp, 1), "text": s}
                                if corrected != s:
                                    entry["corrected"] = corrected
                                url_candidates.append(entry)
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
            print(f"[modal_ocr] Found {len(links)} unique URL(s); "
                  f"{len(url_candidates)} url-ish OCR string(s) captured")
            return {
                "links": links,
                "frames_processed": frames_processed,
                "url_candidates": url_candidates,
            }
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
        "engine": "paddleocr",
        "lang": OCR_LANG,
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
