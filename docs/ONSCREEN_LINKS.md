# Design: On-Screen URL Extraction (`/links`)

Captured 2026-06-02. Status: **v0 in progress** on branch `feat/onscreen-links`.

> Sequencing note: this is being built **before** the NEXT_STEPS.md §7 (artifact-folder
> migration) and §8 (`/find` horizon fix) work, by deliberate decision. The OCR layer
> will need light refactoring to compose cleanly with the shared-artifact layout once
> §7 lands (its raw URL list is exactly the kind of output §7 wants to archive). That
> reconciliation is accepted debt, not an oversight.

## Goal

Extract URLs that are **visually present in the video frames** (screenshares, lower-thirds,
slides, "link in bio" cards, terminal sessions, browser address bars) and surface them in
the frontend as a list of clickable links, each anchored to the playback position(s) where
it appeared.

Output per URL: `{url, first_seen, last_seen, occurrences}` where `first_seen` / `last_seen`
are timestamps in seconds and `occurrences` is the number of sampled frames the URL appeared in.

## Why a separate OCR pipeline, not a Marlin/Qwen prompt

The obvious-looking shortcut — "just ask the VLM to read the URLs" — does not work for this task:

1. **Frame density.** Marlin's native `caption()` and Qwen's `ask()` sample on the order of
   ~32 frames across the *entire* video (Qwen is explicitly capped at `num_frames=32`; Marlin's
   internal sampler has the ~130s horizon documented in `modal_app.py`). A URL that's on screen
   for 3 seconds may fall entirely between sampled frames. OCR over a dense 2 fps sample sees
   ~60× more frames and actually catches transient on-screen text.
2. **Hallucination breaks clickability.** VLMs paraphrase and "autocorrect" text. For prose
   that's fine; for a URL, a single wrong character (`rn`→`m`, dropped query param, invented TLD)
   produces a link that 404s or, worse, points somewhere unintended. OCR transcribes glyphs
   rather than "understanding" them, so it preserves the exact string far more reliably. A URL
   only has value if it's *exactly* right.

So OCR runs as its own pipeline, **parallel to** (not nested inside) the VLM path.

## Architecture

```
video bytes ──▶ ffmpeg sample @ 2 fps ──▶ [frame_0.jpg @ t0, frame_1.jpg @ t1, ...]
                                                │
                                                ▼  (per frame)
                                          EasyOCR → list[str] detected text
                                                │
                                                ▼
                                    URL regex over concatenated frame text
                                                │
                                                ▼
                              dedupe / cluster across frames by normalized URL
                                                │
                                                ▼
            {"links": [{url, first_seen, last_seen, occurrences}], "frames_processed": N}
```

- **Frame sampling:** fixed **2 fps** for v0 via ffmpeg (`-vf fps=2`). Each output frame's
  timestamp is `frame_index / 2.0`. Simple, deterministic, codec-agnostic.
- **OCR:** **EasyOCR** runs on each JPG, returns all detected text strings for that frame.
- **URL detection:** a robust regex (see below) runs over the frame's concatenated text.
- **Dedupe:** identical URLs across frames collapse into one entry. Two URLs are "identical"
  if they match after **lowercasing scheme+host and stripping trailing slashes**
  (`HTTPS://Example.com/` ≡ `https://example.com`). `first_seen` = earliest timestamp,
  `last_seen` = latest, `occurrences` = count of frames it appeared in.

### URL detection (strict + OCR-tolerant fuzzy)

Two passes, run **per detected OCR string** (not the joined frame text, so the
space-tolerant fuzzy matcher can't stitch a fake URL across unrelated boxes):

1. **Strict** — clean, well-formed `http(s)://` / `www.` URLs with common TLDs, query
   strings, fragments. High precision; rejects file paths, version numbers (`1.2.3`,
   `v4.46.0`), and bare dotted words.
2. **Fuzzy (OCR-tolerant)** — added 2026-06-02 after diagnosing real output: EasyOCR gets
   URL *letters* right but mangles *punctuation*. `https://erpdev.hameemgroup.com:8443/…`
   came out as `https /lerpdev hameemgroup com.8443/…` — `://` read as a space, domain dots
   read as spaces. The fuzzy pass repairs the scheme (`https /l` → `https://`), accepts host
   labels separated by `.` **or spaces** ending in a known TLD, tolerates a mangled `:port`,
   then reconstructs `scheme://host[:port]/path`. To avoid prose/email false positives it
   requires a strong signal — a scheme **or** a port **or** a `/path` — so `see us com` and
   `erp@hameemgroup.com` don't become links.

**Known limitation:** fuzzy recovers the **host + port reliably**, but the **path can still
carry OCR letter errors** (`erp/internal` → `erpfinternal`) that aren't recoverable without a
dictionary. The raw OCR strings are kept in `raw_output.url_candidates` as ground truth so a
human can verify/repair a path. A missed URL is a quality note; a *wrong* clickable host is
worse, hence the strong-signal gate.

## Modal deployment

- **New app: `clip-ocr`**, deployed **separately** from `clip-marlin`. Same project, distinct
  Modal app, so the production captioning app is completely unaffected by OCR iteration and
  redeploys. Lives in its own file `modal_ocr.py` (keeps the already-large `modal_app.py` readable).
- Single GPU class `OCRModel` with one method:
  `extract_links(video_bytes, video_ext) -> {"links": [...], "frames_processed": int}`.
- Generous timeout (**1800s**): OCR over many frames on a long video at 2 fps is slow
  (a 10-min video = ~1200 frames).
- The gateway reaches it via `modal.Cls.from_name("clip-ocr", "OCRModel")`, mirroring how
  `ModalVLM` reaches `clip-marlin`. The async path reuses the existing Modal-native job queue
  (`.spawn()` → `FunctionCall.object_id` → poll `GET /v1/jobs/{id}`).

## OCR library choice

- **v0: EasyOCR.** Pip-installable, no system packages beyond what's already in the image,
  GPU-acceleratable on Modal (`Reader(['en'], gpu=True)`). Good enough to validate the pipeline.
- **Upgrade path: PaddleOCR** if EasyOCR accuracy on small / stylized fonts proves insufficient.
  Swapping it is isolated to the OCR step inside `OCRModel`; nothing else changes.

## Sampling choice

- **v0: up to 2 fps, capped to `MAX_FRAMES` (300) total.** Short videos sample at the full
  2 fps; long videos drop to `MAX_FRAMES / duration` (e.g. a 15-min video → ~0.33 fps). This
  cap was added 2026-06-02 after a 15-min meeting recording produced ~1,815 frames and ran 40+
  min — dense screen-recording frames are slow because EasyOCR runs recognition once per detected
  text box, and meeting UIs have hundreds per frame. Frames are also downscaled to
  `DOWNSCALE_WIDTH` (1280px) before OCR to cut spurious tiny-text boxes. Both knobs are tunable:
  raise them if URLs are missed, lower them for speed. A per-`OCR_PROGRESS_EVERY`-frame heartbeat
  log makes long runs observably progressing, and `torch.cuda.is_available()` is logged at load
  so a silent CPU fallback can't masquerade as "just slow."
- **Trade-off:** sparser sampling can miss a URL that flashes on screen for only a second or two.
  For the meeting use-case the URLs sit persistently in the chat, so the cap is safe; for
  fast-cut content, lower the sampling interval or move to the follow-up below.
- **Follow-up: PySceneDetect.** Scene-change-based sampling would OCR one representative frame
  per shot instead of fixed-interval sampling — far fewer frames on static content, better
  coverage of rapid cuts. Deferred; the frame cap is the pragmatic v0 guard.

## Out of scope for v0

- **Cross-referencing OCR hits with Marlin scenes** (e.g. "this URL was shown *during scene X*").
  v0 emits the raw URL list with timestamps only. Correlating the two timelines is a follow-up
  and is exactly the kind of thing the §7 shared-artifact layout will make natural.
- **Non-URL entities:** emails, phone numbers, physical addresses. **URLs only** for v0.
- **Timestamp → player seek** in the frontend (the URL list renders, but clicking a timestamp
  won't scrub the video yet — that's a polish pass).

## API surface (v0)

```
POST /v1/videos/{video_id}/links   → { job_id, status, status_url }   (enqueue, async)
GET  /v1/jobs/{job_id}             → SUCCESS → LinksResponse:
       {
         video_id: str,
         links: [ { url, first_seen, last_seen, occurrences } ],
         frames_processed: int,
         raw_output?: <debug>
       }
```

No `routing.py` changes: OCR has a single backend and does not participate in the VLM model
selector. No `constants.py` changes: OCR is not a `ModelChoice`.
