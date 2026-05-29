# Clip — Build Log

Running log of build activity, fixes, decisions, dead ends, and known issues.
Append chronologically. PRD remains authoritative for product spec; this file
captures the actual build journey — including reverts and operational lessons —
so future sessions don't have to rediscover what was already learned.

---

## 2026-05-26 / 2026-05-27 — Milestone 0 spike

### Setup completed

- **Modal account + CLI** installed; `modal setup` ran one-time browser auth.
- **HF token** created at https://huggingface.co/settings/tokens.
- **Marlin-2B gated-repo access** approved on HF (visit the model page, accept terms).
- **Modal secret** `huggingface-secret` created with `HF_TOKEN=hf_xxx`.
- **Persistent Modal app** `clip-marlin` deployed (gpu=A10G, scaledown_window=120s).
- **Persistent volume** `clip-model-cache` created for HF weights (~5 GB after first fetch).
- **Local FastAPI** runs via `uvicorn main:app --host 0.0.0.0 --port 8000` with `CLIP_BACKEND=modal`.

### Fixes applied (kept)

| File | Change | Why |
|---|---|---|
| `modal_app.py` | Added explicit HF auth probe (`whoami`, `HfApi().model_info`, pre-fetch `config.json`) before model load | Original code surfaced HF auth/network errors as generic transformers `OSError`. New probe raises specific, actionable errors at each step (bad token vs. no gated-repo access vs. network problem) |
| `modal_app.py` | Added `torchvision>=0.19.0` to image `.pip_install(...)` | Marlin's `AutoProcessor.from_pretrained` eagerly loads `Qwen3VLVideoProcessor`, which hard-requires `torchvision`. Without it, container crash-loops on lifecycle hook |
| `modal_app.py` | Set `retries=0` on `@app.cls` | Prevents Modal's container-start retry storm flooding logs while iterating on fixes |
| `sampling.py` | Lowered `MAX_FRAMES` 256 → 32 | Original 256 cap (with misleading "Marlin's sweet spot" comment) not feasible on A10G — visual-token cost is ~200-400 tokens/frame, so 64+ frames push inference past 1 min and can OOM. 32 is the practical ceiling for this hardware |
| `routing.py` | Rewrote `parse_describe` to handle Marlin's native `Scene: ... Events: <a: b> desc` output | Original parser only matched `[HH:MM:SS]` format, but Marlin emits its own `<start: end>` decimal-second tags. Result was empty `Scenes[]` / `Events[]` in API responses, falling back to dumping raw text into Summary. Fallback path for `[HH:MM:SS]` preserved for TimeLens and prompted-format responses |

### Tried and reverted

| Attempt | Result | Why reverted |
|---|---|---|
| Put `"fps": value` inside the video content dict | Silently ignored by `apply_chat_template(tokenize=True, return_dict=True)` code path | No behavioral change; timestamps stayed wrong |
| Pass `video_metadata=[{"fps": ..., "duration": ..., "total_num_frames": ...}]` as kwarg to `apply_chat_template` | Improved Summary detail noticeably (more accurate scene description, caught DAW channel names, hoodie vs. t-shirt), BUT broke Events generation — every tag started `<0.0:` with no end timestamp, model entered repetition loop ("The man speaks and gestures" repeated 20+ times), output truncated mid-stream | Reverted 2026-05-27. Trade was net negative: regression was user-visible degradation, summary gain wasn't worth degenerate events |

Both reverts left the codebase in a clean state matching original docs — no fps / video_metadata plumbing anywhere.

### Known issues (deferred)

**Hallucinated timestamps — PRD §9 known risk.** Marlin assumes `fps=24` when no source-fps is provided. With our 32-frame sampling, every video appears ~1.33s long to the model, so Events tags compress into ~0-2s regardless of true duration. The Summary paragraph remains accurate; only the timestamps are off.

A proper fix requires reading Marlin's `modeling_marlin.py` (custom code downloaded via `trust_remote_code=True`, cached on the Modal volume at `/cache/models--NemoStation--Marlin-2B/snapshots/<sha>/`) to find the correct kwarg pathway. Two casual attempts (fps in content dict; `video_metadata` kwarg) both failed; the third attempt requires source-code reading, not guessing.

PRD §9 mitigation already in spec: "Validate timestamps against video duration; flag low-confidence in API response." That's the path forward when this is picked back up.

### Operational notes for future sessions

- **Modal apps come in two flavors:**
  - `modal run modal_app.py` = ephemeral one-off run (executes the file's `@app.local_entrypoint()`).
  - `modal deploy modal_app.py` = updates the persistent `clip-marlin` app. `inference.py` calls into the *deployed* app via `modal.Cls.from_name("clip-marlin", ...)`.
- **Activation matrix after a code change:**
  - `modal_app.py` → requires `modal deploy modal_app.py` to push to the deployed app.
  - `inference.py` / `routing.py` / `sampling.py` / `main.py` → requires `uvicorn` restart (Python module cache holds the old code otherwise).
  - `index.html` → just browser refresh.
- **Diagnostic terminal:** `modal app logs clip-marlin` in a second PowerShell window tails Modal container stdout in real-time. Invaluable for debugging cold-starts and lifecycle hook failures.
- **First cold-start after deploy** = ~30-60s (container boot + GPU attach + model load). Subsequent calls within `scaledown_window=120s` are warm and start instantly.

### Current state

End-to-end pipeline works: file upload OR URL ingest → frame sampling (32 frames) → Modal A10G inference → parse (`Scene:` paragraph + `Events: <a: b>` tags) → `DescribeResponse` JSON → HTML render with Summary + Events + Scenes cards.

Tested against:
- `Sirocco Final Project.MOV` (92.43s local file, montage with two distinct shoot setups).
- YouTube Shorts URL `https://www.youtube.com/shorts/bY8A66LjGBg` (149s, NOISIA producer talking + DAW screen capture).

Both produced sensible Summary paragraphs and structured Events / Scenes — with the known timestamp-compression caveat.

**Milestone 0 (PRD §8) status:** functionally complete with the timestamp deferral noted. Ready to move to Milestone 1 (async queue + S3 + remaining endpoint testing + Next.js frontend).

---

---

### 2026-05-27 (later) — Found the actual fix for timestamps

User pushed back on the "defer #8" framing: timestamps are core (PRD §1, §5), not polish, and deferring them was lazy. Went back and actually read the Marlin model card on HuggingFace, which revealed the root cause that two prior fix attempts had missed.

**What the model card actually documents:**

Marlin's custom modeling code exposes two convenience methods directly on the model object — `marlin.caption(video_path)` and `marlin.find(video_path, event=query)`. The chat-template + pre-sampled-frames path we'd been using all along is explicitly labeled **"Advanced — raw inference"** in the docs. It's the escape hatch, not the canonical path.

`marlin.caption()` handles video decoding (via torchcodec), frame sampling, AND timestamp tracking internally — all of which are exactly what we kept trying (and failing) to thread through `apply_chat_template` kwargs.

**Why our prior attempts failed:**

- `"fps": value` in the video content dict — chat-template path doesn't read it.
- `video_metadata=[{"fps": ..., "duration": ..., "total_num_frames": ...}]` kwarg — broke generation into a repetition loop because Marlin's custom code wasn't built around those kwargs.

Both attempts were dead ends because we were trying to fix the wrong code path. The fix was to switch to the canonical path entirely.

**New fix applied:**

| File | Change |
|---|---|
| `modal_app.py` | Added `torchcodec` to image `pip_install` (model card requirement we'd missed). |
| `modal_app.py` | Added new `@modal.method() def caption(self, video_bytes, video_ext, ...)`. Writes bytes to a tempfile and calls `self.model.caption(video_path)`. Returns `{"raw": result["caption"]}`. |
| `inference.py` | Added `caption(self, video_path)` method to both `VideoVLM` and `ModalVLM`. The Modal version reads file bytes and ships them to the new remote method. |
| `main.py` | `/describe` endpoint now calls `m.caption(video_path)` when the backend exposes it (Marlin does, TimeLens won't). Other backends fall back to the existing prompted `run()` path. |

`generate()` in `modal_app.py` is kept unchanged for `/find`, `/summarise`, `/ask` — those use custom prompts and have to go through the raw path. Their timestamps will still be compressed for now; same model-card lesson applies and they can be moved to `model.find()` etc. in a follow-up.

**Activation required:**
- Restart uvicorn (picks up `inference.py` + `main.py` changes).
- `modal deploy modal_app.py` (pushes new image with torchcodec + new `caption` method).

**Net cost:** bigger payload to Modal (whole video bytes instead of 32 sampled PIL frames). For short clips this is comparable; for long clips it grows with duration × bitrate. Worth it for correctness.

**Open follow-ups:**
- `/find` → switch to `marlin.find(video_path, event=...)` (same lesson, native pathway).
- `/summarise` and `/ask` → no native method exists; either keep on raw `generate()` or build prompted variants of caption-mode.
- Bump `transformers` if `model.caption()` errors at runtime (model card says >= 5.7.0; we have >= 4.46.0; haven't confirmed compatibility yet).

**Result after deploy: it worked.**

Tested against the same NOISIA YouTube short (149s) that previously returned 3 events compressed into 0-2s. Now returns 25+ events spanning 0:00 → 2:31, with a much richer Summary paragraph (identifies "Under Armour" hoodie, audio-editing context, gesture-based explanation, etc.). This is the temporal grounding Marlin was actually built for; we were just routing around it.

One small artifact: the *last* Scene end-time runs 12 seconds past actual video duration (2:41 vs. 149s = 0:02:29). Probably Marlin extrapolating final-frame timing; easy to clamp downstream in `routing.parse_describe` if it matters. Not blocking.

`transformers >= 4.46.0` was sufficient — `marlin.caption()` worked without any version bump. Adding `torchcodec` was the actual unblock on the dependency side.

Milestone 0 (PRD §8) now fully complete with correct timestamps. No deferrals.

---

### 2026-05-27 (later) — Polish pass: clamping + /find

After confirming `/describe` worked end-to-end, two follow-up fixes landed:

**1. Timestamp clamping (`routing.py`).** Implementing PRD §9's stated mitigation — "validate timestamps against video duration; flag low-confidence."

Added `_clamp_to_duration()` helper to `routing.py`. Policy: drop entries whose start exceeds duration + 2s grace (treats as hallucination); clamp the rest's start/end to `[0, duration]`. Also drops zero-length scenes (artifact of clamping a hallucinated range to a point — meaningless to render).

Wired into `/describe` via a new `duration` kwarg on `parse_describe()`. Tested against the NOISIA short: Marlin emitted `<151.0 - 161.0>` as its final entry (12s past actual 149s duration); clamping dropped the scene cleanly and pinned the corresponding event timestamp to 2:29. No invented time anywhere in the structured output.

**2. `/find` switched to Marlin's native `.find()` (`modal_app.py`, `inference.py`, `main.py`).** Same model-card lesson as caption(): Marlin has a separately-trained find-mode that emits `From X.X to Y.Y.` and the custom modeling code parses it into a `(start, end)` tuple via `model.find(video_path, event=query)`. The original `/find` was using a custom prompt + `parse_find()` regex on raw generate output — same anti-pattern that broke timestamps in `/describe`.

Added:
- `@modal.method() def find(self, video_bytes, event, video_ext)` in `modal_app.py`.
- `find(self, video_path, query) -> dict` methods on `VideoVLM` and `ModalVLM` in `inference.py`.
- `/find` in `main.py` now uses `m.find(video_path, query)` when the backend exposes it; clamps the returned span to `[0, duration]`; wraps as a one-element `Match` list to match `FindResponse` schema.

Tested with query "when does the mixing panel appear" against NOISIA short → Marlin returned span 135-145s (2:15 — 2:25), which lines up with one of the dense panel-focus events from Describe. Single-best-match behavior is a v0.1 limitation per Marlin's design; multi-match would need a different prompting strategy and is out of scope.

**Milestone 0 status: complete with all polish.**

Untouched in this session (still on the prompted-frames raw path, will need similar refactor later if their outputs matter for the demo):
- `/summarise` — no native Marlin mode exists, will need a prompted variant.
- `/ask` — same.
- `generate()` in `modal_app.py` — kept for the fallback path used by the above two.

---

### 2026-05-27 (later) — /summarise via caption() post-processing

PRD §2 / §5.2 wants a real bullet summary. Marlin has no native summarize mode (only caption + find), so the right v0.1 answer is post-processing: call `marlin.caption()` to get the dense events, then cluster them.

Added `summarise_from_events()` in `routing.py`. Strategy:
1. Cluster events by normalized description (lowercase, strip leading articles/punctuation).
2. Build a "chapter" per cluster: `start` = first occurrence, `end` = last occurrence (or +5s synthetic span for single-occurrence clusters).
3. Rank chapters by frequency (most-recurring topics first).
4. Take top N (default 5), re-sort chronologically.
5. Format as `[M:SS - M:SS] description` bullets.

Wired into `main.py /summarise` via the same `if hasattr(m, "caption")` pattern used for `/describe` and `/find`. The original prompted path is preserved as fallback for non-Marlin backends. No `modal_app.py` change needed.

**Result on the NOISIA short:** 5 clean chronological bullets (man speaking, laptop screen, man continues, finger pointing at DAW spanning 0:30-2:11, camera panning across DAW spanning 0:36-2:21). Captures the dominant visual themes well.

**Observation worth recording for future awareness:** the underlying Marlin caption output was much messier on this run than earlier `/describe` runs — Marlin emitted events well past the video duration (up to ~291s for a 149s video) and eventually entered a repetition loop (`A finger points at theDAW interface` × 60+ lines) before hitting max_new_tokens=2048. The structured `/summarise` response was unaffected because:
- `_clamp_to_duration` dropped every event past `duration + 2s grace`.
- `summarise_from_events` clustering collapsed the repetition into one bullet.

Generation variance run-to-run is expected with VLMs even at `do_sample=False` (GPU non-determinism, container state, slight input differences if yt-dlp picks a different stream). Not actionable right now but worth knowing — both the clamp and the cluster step earn their keep on degenerate runs.

**Performance footnote:** each `/summarise` call currently triggers a fresh `caption()` inference (~20-40s warm, ~60s cold). Caching the `DescribeResponse` per video_id and reusing it for `/summarise` is a natural Milestone 1 optimisation once Postgres / Redis come in.

---

### Outstanding for PRD compliance after this session

- **`/ask`** still falls back to Marlin's caption output. PRD §5.4 routes `/ask` to `Qwen3-VL-8B-Instruct`, which isn't deployed. That's Milestone 2 work (multi-model deployment). Marlin literally can't answer arbitrary questions — it can only caption — so getting `/ask` to actually answer questions requires deploying a chat-tuned VLM. **[Resolved 2026-05-27 — see entry below.]**
- **`/find` is single-match** per query. PRD §5.3's schema envisions multi-match. Adding multi-match requires either issuing multiple `model.find()` calls with re-framings or building a different prompting strategy. Out of v0.1 scope.
- **Caching `/describe` output for `/summarise` reuse** — natural Postgres/Redis optimization, Milestone 1.

---

### 2026-05-27 (later) — Qwen3-VL-8B-Instruct deployed; /ask now routes per PRD §5.4

PRD §5.4 sends "open-ended reasoning" to `Qwen/Qwen3-VL-8B-Instruct`. Marlin literally can't answer questions (captioner only), so `/ask` was returning caption-style output regardless of the prompt. This session deploys the second model and wires `/ask` to it, closing the 4-endpoint contract.

**Design choices (locked after reading the [Qwen3-VL GitHub README](https://github.com/QwenLM/Qwen3-VL) — not the HF model card, which is sparser):**

- **Loader**: `AutoModelForImageTextToText.from_pretrained(..., dtype=torch.bfloat16, device_map={"":"cuda"})`. The HF model card showed `Qwen3VLForConditionalGeneration` but Qwen's own README uses the Auto class — going with the README.
- **`trust_remote_code`**: NOT needed. Qwen3-VL is native in `transformers >= 4.57.0`. Different from Marlin which still needs it.
- **Video input format**: `{"type": "video", "video": "file:///{tmppath}"}` directly in the chat template. Same tempfile-on-container pattern Marlin uses for `caption()` / `find()`. Avoids the `qwen_vl_utils.process_vision_info` path, which the README marks as optional (needed only for fine-grained pixel control on long videos).
- **Frame budget**: `apply_chat_template(..., num_frames=32, fps=None)`. Caps video tokens to keep A10G memory in check.
- **Hardware**: A10G (24GB), same as Marlin. 9B params bf16 ≈ 18GB weights + video tokens. Tight but feasible with the 32-frame cap. Flagged in code that an A100 bump is the obvious upgrade path if OOM appears in testing.
- **`attn_implementation="flash_attention_2"`**: skipped for v0.1. Recommended-not-required in the README; building flash-attn drags `nvcc` into the image. Easy follow-up later.
- **Generation**: `do_sample=False` (greedy). Qwen's README recommends sampling defaults (temp=0.7, top_p=0.8) for Instruct variants, but greedy is deterministic and matches Marlin's path — easier to debug and benchmark. Revisit when we have eval coverage.

**Hosting layout: second class, same app.**

Rather than spinning up a new Modal app (`clip-qwen` etc.), `QwenVL` is added as a second `@app.cls` inside the existing `clip-marlin` app. Both classes share the volume cache. Modal scales them independently regardless, so no functional cost to colocating; ops surface stays small (one `modal deploy`, one `modal app logs` stream).

**Separate `image` objects per class, though.** Qwen3-VL needs `transformers >= 4.57.0`; Marlin runs on `>= 4.46.0` with `trust_remote_code` against custom Marlin modeling code that wasn't tested against the newer transformers line. Sharing one image would force a bump on Marlin's side that could regress the working `/describe`, `/find`, `/summarise` pathways. Per-class images keep both blast radii minimal.

### Changes applied

| File | Change |
|---|---|
| `constants.py` | Added `ModelChoice.QWEN3_VL_8B = "qwen3-vl-8b"` + `MODEL_REPOS` entry pointing at `Qwen/Qwen3-VL-8B-Instruct`. |
| `schemas.py` | Extended the `Literal["marlin-2b", "timelens-8b"]` type on `AskRequest.model`, `FindRequest.model`, `SummariseRequest.model` to include `"qwen3-vl-8b"`. |
| `modal_app.py` | Added a second `qwen_image` (separate from Marlin's) with `transformers>=4.57.0`, torchcodec, torchvision, qwen-vl-utils. Added `QwenVL` class with `@modal.method() ask(video_bytes, question, video_ext, num_frames=32) -> {"raw": str}`. Updated `healthz()` to report both repos. |
| `inference.py` | `ModalVLM.__init__` now dispatches `(app_name, class_name)` via a `_MODAL_CLASSES` map (`MARLIN_2B → MarlinModel`, `QWEN3_VL_8B → QwenVL`). Added `_require_marlin()` guard at the top of `run()`/`caption()`/`find()` so misbinding raises a clear Python error instead of failing deep inside a `.remote()` wire call. Added `ask(video_path, question)` that requires the Qwen binding and ships video bytes + question to `QwenVL.ask.remote(...)`. |
| `main.py` | `/ask` defaults to `ModelChoice.QWEN3_VL_8B` (PRD §5.4) unless the caller forces a model. On Qwen, calls `m.ask(video_path, question)`; on `NotImplementedError` / `RuntimeError` falls back to Marlin's prompted `run()` path so `/ask` still returns something usable when the Qwen container isn't reachable. Response `model` field exposes which model actually answered, so callers can detect the degradation. Imported `ModelChoice` from `constants`. |

### Tried and reverted

| Attempt | Result | Why reverted |
|---|---|---|
| Pass video as `f"file://{video_path}"` URI in the chat-template content dict | First post-deploy `/ask` call hit `TypeError: Incorrect format used for video. Should be an url linking to an video or a local path.` deep in `transformers/video_utils.py:load_video` | The Qwen3-VL README's `file:///` form is documented under the `qwen_vl_utils.process_vision_info` path — that helper has its own URI parser. When messages go directly to `apply_chat_template`, transformers' internal `load_video` only accepts http(s) URLs or **raw absolute paths**. Fix was one line: drop the `file://` prefix and pass `str(video_path)` directly. Module-level comment added at the call site so the next person doesn't reintroduce it. |

Lesson, same shape as the previous Marlin session: the README is authoritative *for the path it's documenting*, not for all paths. The two video-input forms (direct vs. via `process_vision_info`) have different URI conventions, and I conflated them.

### Result after deploy: it worked.

Re-deployed (`modal deploy` again, no uvicorn restart needed). Tested against the same NOISIA YouTube short (149s) with the question "why is the man using a laptop". Response:

- Model badge: `qwen3-vl-8b` (routing landed where PRD §5.4 specifies).
- `Modal backend ready (clip-marlin / QwenVL)` in uvicorn logs — confirms `ModalVLM` dispatched to the new class, not `MarlinModel`.
- 11 MB of video bytes shipped to the container; total `/ask` round-trip ≈ 80s on cold start (container boot + 18 GB weight pull + ViT load + inference).
- Answer is doing actual reasoning, not captioning: identifies the software category ("Ableton Live or Logic Pro"), references the on-screen text "NOISIA & CAMO & KROOKED" (Qwen OCR'd it from a frame), explains the workflow rationale rather than just describing the frame. This is content Marlin literally cannot produce — confirms the routing change is real and not cosmetic.

**Milestone 2 (PRD §8) partial:** the multi-model routing piece for `/ask` is done. TimeLens-8B, webhooks, API key auth, and the HF Spaces gated demo are still open under Milestone 2.

### Updated outstanding for PRD compliance

- **`/find` is still single-match per query.** Out of v0.1 scope per prior session.
- **`/summarise` triggers a fresh `caption()` each call.** Postgres/Redis cache is Milestone 1.
- **Qwen3-VL flash-attn**: skipped for v0.1; ~1.5-2x throughput available once nvcc-in-image build cost is absorbed.
- **PRD §5.4 routing for long videos under `/ask`**: deliberately diverges from the PRD text (the PRD says ">10min → TimeLens" but Qwen3-VL is the one with 256K context tuned for long-video Q&A). Worth updating the PRD on this when the next pass happens — note in the next session.
- **A10G memory headroom for Qwen3-VL**: tested fine on 149s NOISIA short with `num_frames=32`. Untested on multi-minute videos where the temporal tokens balloon; bump to A100-40GB is the documented upgrade path.

### Smoke-test recipe (hand-off — needs the user's Modal-authenticated terminal)

Sandbox doesn't have the user's Modal token, so deploy and smoke test happen on the user's box. Two terminals:

```powershell
# Terminal A (deploy) — from clip/
modal deploy modal_app.py

# Terminal B (live logs in a second window — recommended)
modal app logs clip-marlin
```

Then restart uvicorn so `inference.py` / `main.py` changes pick up:

```powershell
# Terminal C (gateway)
$env:CLIP_BACKEND="modal"
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then re-upload a known video (the NOISIA short or `Sirocco Final Project.MOV`) and hit `/ask`:

```powershell
# Curl example — replace {video_id} with the one from the upload response.
curl -X POST "http://localhost:8000/v1/videos/{video_id}/ask" `
  -H "Content-Type: application/json" `
  -d '{"question": "Why is the person at the laptop?"}'
```

**Expected:**
- First call cold-starts the `QwenVL` container — ~60-90s (boot + 18 GB weight pull on first run, then volume-cached).
- Response `model` field is `"qwen3-vl-8b"`.
- `answer` reads like an answer to the question, not a description of the scene. Compare to a forced Marlin call (`{"question": "...", "model": "marlin-2b"}`) — Marlin should still give the old caption-style output, confirming routing actually changed something.

**If the QwenVL container OOMs on A10G:**
- Log will say something like "CUDA out of memory" deep in `model.generate`.
- First lever: drop `num_frames` from 32 → 16 in `modal_app.py` `QwenVL.ask`.
- If that's not enough: bump `QWEN_GPU = "A100-40GB"` in `modal_app.py` (line near the top). A100-40GB is the next standard tier on Modal; A100-80GB is overkill for 9B bf16.
- Either change is a `modal deploy modal_app.py` away — no client-side restart needed.

### Updated outstanding for PRD compliance

- **`/find` is still single-match per query.** PRD §5.3's schema envisions multi-match. Out of v0.1 scope per prior session.
- **`/summarise` triggers a fresh `caption()` each call.** Caching `DescribeResponse` per video_id and reusing it is a natural Milestone 1 / Postgres-Redis optimization.
- **Qwen3-VL flash-attn**: documented as recommended-not-required, skipped for v0.1. Picks up 1.5-2x throughput and lower memory once the build cost is absorbed. Easy follow-up.
- **Routing semantics for long videos under `/ask`**: PRD §5.4 says ">10min → TimeLens" but Qwen3-VL is the model with the 256K context made for long-video Q&A, and TimeLens isn't deployed. Current code intentionally skips the long-video TimeLens upgrade for `/ask`. May want to revisit the PRD text on this — the rule was written before Qwen3-VL was final.

---

### 2026-05-29 — Milestone 1 chunk (a): Celery + Redis async job queue

Picked chunk (a) from the three-way choice at session start. Rationale recorded
up front: PRD §5.2 explicitly says "sync if <60s else async" and every endpoint
currently violates that; the job-polling contract (`POST → job_id`,
`GET /jobs/{id}`) reshapes both the API and the frontend, so doing it before
chunk (c) avoids rewriting client code; and Redis comes in as a side effect
(Celery broker + result backend), which gives a durable home for the videos
metadata dict and kills the "uvicorn restart loses everything" bug as a freebie
without inventing a new dependency.

### What shipped

| File | Change |
|---|---|
| `schemas.py` | Added `JobStatus` literal mirroring Celery's built-in state strings (`PENDING`, `STARTED`, `SUCCESS`, `FAILURE`, `RETRY`, `REVOKED`). Added `JobEnqueueResponse` (returned by every inference endpoint) and `JobStatusResponse` (returned by the new poll endpoint). |
| `store.py` (new) | Redis-backed key/value store for video metadata. One key per video at `clip:video:{id}` → JSON. Optional TTL via `CLIP_VIDEO_TTL_SECONDS`. Replaces the Milestone-0 `_videos: dict` that lived in main.py and died on uvicorn restart. Same interface as Postgres will eventually expose — chunk (b) swaps the backend without touching callers. |
| `tasks.py` (new) | Celery app + four inference tasks (`clip.describe`, `clip.find`, `clip.summarise`, `clip.ask`). Broker + result backend both Redis on `/0`. `task_track_started=True` so the gateway can distinguish "queued" from "running" during polling. `task_time_limit=30 * 60` as a hard ceiling so a wedged inference can't lock a worker forever. JSON serializer; pydantic responses are `.model_dump()`'d before return because Celery's JSON path can't serialize pydantic instances directly. |
| `main.py` | The four inference endpoints (`/describe`, `/find`, `/summarise`, `/ask`) no longer execute inference inline — they call `task.delay(...)` and return `JobEnqueueResponse{job_id, status, status_url}`. `_videos` dict removed; all reads/writes go through `store.put_video` / `store.get_video`. New `GET /v1/jobs/{job_id}` translates Celery `AsyncResult` to `JobStatusResponse`. |
| `index.html` | New `pollJob(statusUrl, onTick)` helper polls every 1.5s with a 30-min hard ceiling (matches worker `task_time_limit`). `runOp()` now reads `{job_id, status_url}` from the enqueue response, then polls until `SUCCESS`/`FAILURE`/`REVOKED`. Status label updates distinguish `PENDING` (queued behind a busy worker) from `STARTED` (worker actively running inference) — Celery only reports `STARTED` because we set `task_track_started=True`. |

### How to run after this change (three terminals)

```powershell
# Terminal A — Redis (Docker, Windows host)
docker run -d --name clip-redis -p 6379:6379 redis:7-alpine
# Sanity: docker exec clip-redis redis-cli ping  → PONG

# Terminal B — Celery worker
cd clip
$env:CLIP_BACKEND="modal"
# --pool=solo is the Windows-friendly choice (Celery's prefork pool needs fork()).
# Concurrency is 1 with solo; for parallelism on Linux drop --pool=solo.
celery -A tasks worker --loglevel=info --pool=solo

# Terminal C — FastAPI gateway
cd clip
$env:CLIP_BACKEND="modal"
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then the existing upload + /describe etc. flow from the browser works the same
way — the page now shows a job-state spinner while polling instead of holding
the request open.

### Activation matrix (updated)

| File touched | Restart needed |
|---|---|
| `modal_app.py` | `modal deploy modal_app.py` |
| `inference.py`, `routing.py`, `sampling.py`, `constants.py`, `schemas.py`, `store.py` | uvicorn restart **AND** celery worker restart (both processes import these) |
| `main.py` | uvicorn restart only |
| `tasks.py` | celery worker restart only |
| `index.html` | browser refresh |

Celery worker holds its own Python module cache and won't pick up code changes
to `tasks.py` or any module it imports until the `celery worker` process is
killed and restarted. Easy to forget; symptom is "endpoint enqueues a job, job
keeps running old behavior."

### Env vars added

- `CLIP_REDIS_URL` (default `redis://localhost:6379/0`) — Celery broker + result backend.
- `CLIP_STORE_REDIS_URL` (default `redis://localhost:6379/1`) — video metadata store. Separate DB so Celery's keyspace and our metadata don't share churn patterns; one Redis instance still serves both. Override to use a different host entirely (e.g. managed Redis in prod).
- `CLIP_VIDEO_TTL_SECONDS` (default unset = no expiry) — optional TTL on metadata entries. PRD §6 says 7-day retention; that policy ultimately needs to cover the video bytes too, which is chunk (b) territory.

### Tried and reverted

Nothing in this session — the cuts landed first-try after reading the Celery 5.6
docs (broker URLs, AsyncResult state strings, `task_track_started`). One
discipline note worth recording: PENDING is overloaded — Celery returns
PENDING for *any* unknown task UUID, not just "queued and waiting." This means
`GET /v1/jobs/{wrong-id}` returns 200 with status=PENDING rather than a 404.
For v0.1 we accept this; the gateway-issued job_ids are the only ones clients
should ever poll, and we don't have a separate job-existence index to check
against. Documented in `get_job`'s docstring. Worth revisiting once Postgres
arrives — the job index could live there.

### Verification

Did a smoke test in an isolated sandbox with stubbed `inference.get_model` and
Celery `task_always_eager=True`:

- `POST /v1/videos/{id}/describe` returns 200 with valid `JobEnqueueResponse` (job_id is a UUID; status_url is absolute and ends with `/v1/jobs/{job_id}`).
- The eager-mode worker executed the describe task and returned a correctly-shaped `DescribeResponse` dict (summary, scenes, events all parsed from a synthetic Marlin-format input).
- `POST /v1/videos/{nonexistent}/describe` returns 404 — store check fires before the enqueue, so we don't burn worker capacity on bad video IDs.
- `/find`, `/summarise`, `/ask` all enqueue and return job envelopes with the correct shape.

The GET-job-status round-trip isn't verifiable in the sandbox (Celery's result
backend needs real Redis; the metadata-store stub doesn't intercept Celery's
own connection pool). That path needs to be confirmed by the user on the real
stack — see smoke recipe below.

### Smoke recipe — user terminal

After running the three startup commands above, in a fourth terminal:

```powershell
# 1. Ingest the known-good NOISIA short
curl -X POST "http://localhost:8000/v1/videos/from-url" `
  -H "Content-Type: application/json" `
  -d '{"url": "https://www.youtube.com/watch?v=bY8A66LjGBg"}'
# → {"video_id": "...", "duration_seconds": 149.x, ...}

# 2. Enqueue describe
curl -X POST "http://localhost:8000/v1/videos/{video_id}/describe"
# → {"job_id": "<uuid>", "status": "PENDING", "status_url": "http://.../v1/jobs/<uuid>"}

# 3. Poll
curl "http://localhost:8000/v1/jobs/{job_id}"
# → first hit: {"status": "PENDING" | "STARTED", "result": null, ...}
# → after worker finishes (~30-90s cold start, ~10-20s warm):
#     {"status": "SUCCESS", "result": {"summary": "...", "scenes": [...], "events": [...], ...}, ...}
```

Frontend equivalent: open `index.html`, upload or URL-ingest, hit Describe.
Spinner should now show "Waiting for a worker…" → "Running inference …"
instead of the old hang-then-payload behavior.

### Known issues / open follow-ups

- **Video bytes still on local disk** under `/tmp/clip-storage` (or `CLIP_STORAGE_DIR`). uvicorn restart preserves them (disk is durable), but worker scale-out is still single-box only — a worker on a different host can't reach the file path. Chunk (b) (S3 + Postgres) is the proper fix and was deliberately deferred for this session.
- **No webhook on job completion.** PRD §5.2 lists `POST /v1/webhooks` as v0.1; not built. Polling works for the demo flow. Webhook delivery layers cleanly on top of `tasks.app.signals.task_postrun` later.
- **PENDING ambiguity** (see Tried-and-reverted above) — gateway can't distinguish "queued" from "unknown id." Acceptable for v0.1; revisit with job index in Postgres.
- **Celery worker on Windows requires `--pool=solo`** (single-concurrency). Linux/macOS users can drop `--pool=solo` for multi-process concurrency. Documented in `tasks.py` docstring and run procedure above.
- **`request.url_for(...)` returns the testclient-relative `http://testserver/...` under TestClient.** In prod behind a reverse proxy, it'll respect the `Host` header. If we ever sit behind a path-stripping proxy, may need `app.root_path` set; flagging for when deployment becomes real (not v0.1).

### Updated outstanding for PRD compliance after this session

Milestone 1 progress:
- [x] Async job queue (Celery + Redis)
- [ ] S3-compatible storage  ← chunk (b), next
- [x] All 4 endpoints functional (already done in Milestone 0; now async)
- [ ] Next.js frontend with video player + timestamp jump  ← chunk (c)
- [ ] Deployed on a single A10 (Modal backend already covers this, but no production deploy pipeline yet)

Milestone 2 carry-overs (unchanged from prior sessions):
- `/find` still single-match per query.
- Qwen3-VL flash-attn skipped for v0.1.
- TimeLens-8B never deployed; routing rule that targets it under `/ask` for long videos diverges from current reality (Qwen handles long-video Q&A).
- API key auth, webhooks, HF Spaces gated demo all pending.

---

### 2026-05-29 (later) — Pivot: Modal-native job queue replaces Celery+Redis

User pivoted mid-session after consulting another Cowork session on
Celery vs. Modal-native queuing. Verdict: for v0.1 — where every
inference call already crosses the Modal boundary — running a separate
Celery worker + Docker Redis + uvicorn buys modularity we won't
exercise. Modal already IS a queue: `Function.spawn()` returns a
`FunctionCall`; `modal.FunctionCall.from_id(object_id).get(timeout=0)`
is the same submit/poll/fetch shape Celery gives us, one less process
to keep alive on Windows, and zero new infrastructure beyond what's
already running.

This session's earlier Celery work isn't a full revert — most of it was
queue-agnostic. The endpoint-contract reshape (POST returns
JobEnqueueResponse; GET /v1/jobs/{id} returns JobStatusResponse), the
schema additions, the index.html polling loop, and the Redis metadata
store all stayed put. Only the worker layer changed.

### What survived from the Celery cut

| File | Status |
|---|---|
| `schemas.py` | Unchanged. `JobStatus`, `JobEnqueueResponse`, `JobStatusResponse` are queue-agnostic. |
| `store.py` | Extended (see below); the videos KV layer is unchanged. |
| `index.html` | Unchanged. The polling loop is just an HTTP poll — doesn't care what runs the job. |
| Endpoint-contract shape in `main.py` | Unchanged. POST returns `JobEnqueueResponse`; GET `/v1/jobs/{job_id}` is the named polling route. Only the bodies changed. |

### What changed

| File | Change |
|---|---|
| `tasks.py` | **Deleted.** All Celery task definitions gone. |
| `requirements.txt` | Dropped `celery>=5.4.0`. Kept `redis>=5.2.0` — store.py still uses it. |
| `inference.py` | Added `submit_caption`, `submit_find`, `submit_run`, `submit_ask` to `ModalVLM`. Each is a 1:1 mirror of the corresponding sync method (same guards, same payload prep) but calls `.spawn(...)` on the remote method handle and returns the resulting `FunctionCall.object_id` string. Sync methods (`caption`, `find`, `run`, `ask`) are kept — useful for testing and any caller that wants to block. |
| `store.py` | Added `put_job(job_id, meta)`, `get_job(job_id)`, `delete_job(job_id)`. New key prefix `clip:job:`. 7-day TTL on job records — aligns with Modal's own output-retention window so we don't keep stale references to FunctionCalls Modal has already discarded. Existing video KV unchanged. |
| `main.py` | Dropped `from celery.result import AsyncResult` and `import tasks`. Each inference endpoint now: (1) routes via `routing.choose_model(...)`, (2) calls `m.submit_<x>(...)` to get a Modal `object_id`, (3) generates a UUID `job_id`, (4) stashes the job record via `store.put_job(...)` with all context the parser will need (`endpoint`, `video_id`, `duration`, `model`, plus per-endpoint fields like `query`/`question`/`style`), (5) returns `JobEnqueueResponse`. GET `/v1/jobs/{job_id}` looks up the job, reconstructs the `FunctionCall` via `modal.FunctionCall.from_id(...)`, tries `.get(timeout=0)`, and dispatches the raw result to the right parser based on the stashed `endpoint`. App version bumped to `0.1.2`. |
| `main.py` (parsers) | Added module-level `_parse_describe`, `_parse_find`, `_parse_summarise`, `_parse_ask` and a `_PARSERS` dispatch dict. Same parsing logic that used to live inside the synchronous endpoint bodies — `routing.parse_describe` with duration-clamping for describe/summarise, span→Match builder with clamping for find, `_strip_think` for ask. Moved server-side of the queue; nothing new logically. |

### Run procedure (supersedes the Celery entry above — only two terminals now)

```powershell
# Terminal A — Redis (Docker, Windows host). Same as before.
docker run -d --name clip-redis -p 6379:6379 redis:7-alpine

# Terminal B — FastAPI gateway. No more celery worker process.
cd clip
$env:CLIP_BACKEND="modal"
uvicorn main:app --host 0.0.0.0 --port 8000
```

The Celery worker terminal from the previous entry is **gone**. The
Windows `--pool=solo` workaround is no longer relevant — we don't run
Celery at all.

### Activation matrix (updated, supersedes the Celery entry)

| File touched | Restart needed |
|---|---|
| `modal_app.py` | `modal deploy modal_app.py` |
| `inference.py`, `routing.py`, `sampling.py`, `constants.py`, `schemas.py`, `store.py`, `main.py` | uvicorn restart |
| `index.html` | browser refresh |

No `celery worker` process exists, so there's no second restart to forget.

### Tried and reverted (within the pivot)

| Attempt | Result | Why reverted |
|---|---|---|
| Initially caught only built-in `TimeoutError` in `get_job`, per Modal's `doc_ocr_webapp.py` example | Verified at session start that `modal.exception.TimeoutError` does **not** inherit from `builtins.TimeoutError` (MRO checked against installed `modal>=0.66`). The example would have left modal-typed timeouts uncaught. | Switched to `except (TimeoutError, modal.exception.TimeoutError)`. Robust to either class being raised; harmless if they converge. |
| Caught `TimeoutError` BEFORE `OutputExpiredError` (natural order: most-common case first) | Verified MRO: `OutputExpiredError → modal.exception.TimeoutError → modal.exception.Error → Exception`. A first-match-wins `except TimeoutError:` clause therefore swallows expired-output as STARTED, never reaching the FAILURE branch. **Visible bug:** expired jobs would have polled forever. | Reordered: `OutputExpiredError` first, then `(TimeoutError, modal.exception.TimeoutError)`, then bare `Exception`. Verified via emulated branch dispatch that all four paths now route correctly. Documented why in a comment so the next person doesn't "fix" the order back. |

Same lesson as previous sessions: read the docs example, then verify
against the installed library. The doc example was a starting point,
not the final answer.

### Verification

Two checks done locally (sandbox doesn't have Modal credentials, so no
end-to-end against a real deployed app):

1. **API surface confirmed against installed `modal>=0.66`:**
   - `modal.FunctionCall.from_id(function_call_id: str, ...)` is the right signature.
   - `modal.FunctionCall.get(timeout: Optional[float] = None, *, index: int = 0)` takes the `timeout=0` form documented in the job-queue guide.
   - `modal.exception.OutputExpiredError` exists.
   - MRO check exposed the OutputExpiredError-subclasses-TimeoutError relationship → drove the reorder above.
2. **Exception-handling emulation:** scripted `try/except` mirroring `get_job`'s clauses against each raise type. Confirmed dispatch:
   - built-in `TimeoutError` → STARTED
   - `modal.exception.TimeoutError` → STARTED
   - `modal.exception.OutputExpiredError` → FAILURE (expired)
   - arbitrary inference exception → FAILURE (repr)
   - clean return → SUCCESS

What's **not** verified in the sandbox (deliberate — would need a real
Modal deploy + credentials):

- `m.submit_caption(...)` actually returns a populated `object_id` round-trippable through `from_id`. The shape match is in the docs and the canonical `doc_ocr_webapp.py` example uses exactly this pattern (`call.object_id` → poll), but it needs to run on the user's box to confirm end-to-end.
- The previous Celery cut was also never end-to-end smoke-tested against a real Redis — so we're not abandoning verified-working code. We're moving to a path that needs fewer processes to verify in the first place (one Docker container + one uvicorn process, vs. three terminals).

### Smoke recipe — user terminal (supersedes Celery recipe)

After running the two startup commands above:

```powershell
# 1. Ingest the known-good NOISIA short
curl -X POST "http://localhost:8000/v1/videos/from-url" `
  -H "Content-Type: application/json" `
  -d '{"url": "https://www.youtube.com/watch?v=bY8A66LjGBg"}'
# → {"video_id": "...", "duration_seconds": 149.x, ...}

# 2. Enqueue describe
curl -X POST "http://localhost:8000/v1/videos/{video_id}/describe"
# → {"job_id": "<uuid>", "status": "PENDING", "status_url": "http://.../v1/jobs/<uuid>"}
# Note: status is the initial-envelope PENDING; the GET handler reports
# STARTED while Modal runs and SUCCESS / FAILURE on completion.

# 3. Poll
curl "http://localhost:8000/v1/jobs/{job_id}"
# → first hit: {"status": "STARTED", "result": null, ...}
# → after Modal returns (~30-90s cold start, ~10-20s warm):
#     {"status": "SUCCESS", "result": {"summary": "...", "scenes": [...], "events": [...]}, ...}
```

Frontend equivalent: open `index.html`, upload or URL-ingest, hit
Describe. Same polling-loop UX as the Celery cut — the loop polls a
URL, doesn't care what's underneath.

### Modal-side notes

- `modal_app.py` is **not modified** this session. The deployed classes
  (`MarlinModel`, `QwenVL`) already expose `caption`, `find`, `generate`,
  `ask` as `@modal.method()` decorated functions. Any `@modal.method`
  picks up `.remote(...)` and `.spawn(...)` for free — no server-side
  change required to support spawn.
- No `modal deploy` needed for this pivot. uvicorn restart picks up
  `inference.py` + `main.py` + `store.py` + `schemas.py` changes.

### Known issues / open follow-ups

- **PENDING in the enqueue envelope vs. STARTED on first poll** — minor cosmetic. POST returns `status: "PENDING"` (initial state in our envelope) but the very next GET will almost always report STARTED because Modal accepts the spawn synchronously. Consistent with the polling-state machine: PENDING is the "just enqueued, not yet polled" state. The frontend doesn't actually act on the enqueue-response status; it just kicks the polling loop.
- **Video bytes still on local disk** under `/tmp/clip-storage`. Same as the Celery cut. uvicorn restart preserves them but multi-host scale-out is still single-box. Chunk (b) (S3 + Postgres) is the proper fix and remains deferred.
- **No webhook on job completion.** PRD §5.2 lists it; Modal supports webhooks per Function but we haven't wired one. Polling works for the demo flow.
- **Spawn-only on Modal backend.** The endpoint bodies raise 501 if the chosen backend isn't Modal (`CLIP_BACKEND=local` would need its own spawn shim — Celery-style — to support async). Acceptable for v0.1 since local-backend was only ever meant for "dev machine has a GPU"; the prod-ish path is always Modal.
- **`/find` still single-match per query.** Carry-over from previous sessions; out of v0.1 scope.

### Updated outstanding for PRD compliance after this session

Milestone 1 progress (same as after the Celery cut — the async-queue
checkbox is done, just via a different mechanism):

- [x] Async job queue (Modal-native: `FunctionCall.spawn` / `from_id` / `get(timeout=0)`)
- [ ] S3-compatible storage  ← chunk (b), next
- [x] All 4 endpoints functional and async
- [ ] Next.js frontend with video player + timestamp jump  ← chunk (c)
- [ ] Deployed on a single A10 (Modal backend already covers this; no production deploy pipeline yet)

Milestone 2 carry-overs unchanged from prior sessions.

---

### 2026-05-29 (later) — Browser "Failed to fetch" diagnosed: uvicorn dying on terminal focus, not a code bug

Resumed to chase the browser-side "X Failed to fetch" that appeared a few
seconds after clicking "Describe video," with the uvicorn terminal showing
200 OK responses and then the PowerShell prompt reappearing (uvicorn EXITED).
Two competing hypotheses going in: (a) Windows-terminal focus interrupt
killing uvicorn, or (b) a real gateway crash in the GET /v1/jobs path.

**Resolution: (a). No code bug.** The server path is verified good end-to-end
on the current stack. The fix is purely operational — run uvicorn detached so
terminal focus can't signal it.

#### Environment delta worth recording

- **Modal client is now `1.3.5`, not the `0.66` the pivot entry assumed.** Major
  version jump (0.x → 1.x). Re-verified the two API facts the gateway leans on,
  and **both still hold in 1.3.5**:
  - Exception MRO unchanged: `OutputExpiredError → modal.exception.TimeoutError
    → modal.exception.Error → Exception`. `modal.exception.TimeoutError` is
    still NOT a subclass of builtins `TimeoutError`; `OutputExpiredError` IS a
    subclass of `modal.exception.TimeoutError`. So `get_job`'s except ordering
    (OutputExpiredError first, then the TimeoutError pair) is still correct —
    do not reorder.
  - `FunctionCall.from_id(function_call_id: str, client=None)` and
    `FunctionCall.get(timeout: Optional[float] = None, *, index: int = 0)` —
    both signatures intact; `get(timeout=0)` is still the right poll form.

#### Why it can't be a get_job crash

An unhandled exception inside a FastAPI request handler returns a 500 (caught
by `@app.exception_handler(Exception)`) — it does **not** terminate the uvicorn
process. The observed symptom is the *process exiting* (prompt returns), which
only an external signal produces. The chain is: uvicorn exits (terminal
quirk) → the browser's next 1.5s poll can't connect → `fetch()` throws the
transport-level `TypeError` that surfaces as "Failed to fetch" (NOT an HTTP
error). The 200 OKs in the terminal right up until the exit are consistent
with this — the server was healthy until it was killed.

The likely mechanism on Windows Terminal / conhost is QuickEdit mode: clicking
into the window to read logs enters text-selection mode and can pause or, on a
stray Ctrl+C / double-interaction, terminate the foreground process. The user
was clicking the uvicorn window during the 30-90s Marlin cold start to watch
progress — exactly when the kill landed.

#### What was tested (all against modal 1.3.5, Memurai on :6379, Modal authed)

1. **GET path, independent of spawn** — polled the known-good pre-spawned job
   `cc33533dee674a45b37151a09807d12e` directly. Returned `SUCCESS` with a fully
   shaped `DescribeResponse` (summary + 26 scenes + 27 events, duration-clamp
   applied: Marlin's `<151-161>` tail dropped, last entry pinned to 149.03s).
   Confirms `from_id` + `get(timeout=0)` + parser dispatch all work on 1.3.5.

2. **Full fresh round-trip through the spawn path** — on the detached uvicorn:
   - `POST /v1/videos/from-url` (NOISIA short) → `video_id=6c137f2a…`, 149.0s,
     source `yt-dlp`.
   - `POST /v1/videos/{id}/describe` → `job_id=1752fb1c…`, status PENDING,
     valid status_url. Logs show `submit_caption` spawning 11,140,991 bytes →
     `modal=fc-01KSSEQG6598FC85HQ3HK785ZD`.
   - Polled every 3s: `STARTED` for ~52s (cold start) → `SUCCESS` with a valid
     `DescribeResponse` (626-char summary, 26 scenes, 27 events, marlin-2b).
   - **The detached uvicorn stayed alive through the entire 52s poll** — the
     exact window that previously killed the foreground instance.

#### The fix — focus-immune run procedure (supersedes the pivot entry's run procedure)

Launch uvicorn as a detached background process via `Start-Process`, with
stdout/stderr redirected to log files. There is no interactive console window
to QuickEdit-pause, and the process is not attached to the shell that launched
it, so clicking around in any terminal cannot signal it. uvicorn writes its
access/app logs to **stderr**, so the err log is the live request log.

```powershell
# Terminal A — Memurai (Windows-native Redis). Already installed + autostart.
& "C:\Program Files\Memurai\memurai-cli.exe" ping   # -> PONG

# Launch detached, focus-immune uvicorn (run once from clip/):
$dir = "C:\Users\Shadwaan\OneDrive\Documents\Claude\Projects\Video Transcription\clip"
$env:CLIP_BACKEND = "modal"
$proc = Start-Process -FilePath "python" `
  -ArgumentList "-m","uvicorn","main:app","--host","0.0.0.0","--port","8000" `
  -WorkingDirectory $dir `
  -RedirectStandardOutput (Join-Path $dir "uvicorn.out.log") `
  -RedirectStandardError  (Join-Path $dir "uvicorn.err.log") `
  -WindowStyle Hidden -PassThru
$proc.Id | Out-File (Join-Path $dir "uvicorn.pid") -Encoding ascii

# Watch live logs from any terminal (does NOT affect the process):
Get-Content (Join-Path $dir "uvicorn.err.log") -Wait -Tail 20

# Stop it when needed (PID is in uvicorn.pid):
Stop-Process -Id (Get-Content (Join-Path $dir "uvicorn.pid")) -Force
```

Restart semantics are unchanged from the pivot entry: code changes to
`main.py` / `inference.py` / `routing.py` / `sampling.py` / `constants.py` /
`schemas.py` / `store.py` still require killing and relaunching this process
(no `--reload`); `index.html` is browser-refresh only. `--reload` was
deliberately NOT used — it adds a file-watcher and a reloader-parent process,
extra surface for marginal benefit on a box where edits are deliberate.

**Operational rule going forward: never click into the uvicorn window to read
logs.** Tail `uvicorn.err.log` from a separate terminal (or
`modal app logs clip-marlin` for the container side) instead. The whole point
of the detached launch is that the server's lifetime no longer depends on
where the mouse is.

#### What changed in the repo this session

- **No source changes.** `main.py`, `inference.py`, `store.py`, `schemas.py`,
  `index.html`, `modal_app.py` all untouched — the diagnosis was operational,
  not a code defect. New runtime artifacts only: `uvicorn.out.log`,
  `uvicorn.err.log`, `uvicorn.pid` (candidates for `.gitignore`).

#### Known cosmetic staleness (not fixed — flagged for a cleanup pass)

`index.html` still carries Celery-era language from before the Modal-native
pivot: the poll-timeout error reads "Check Celery worker logs" and a comment
says "endpoints now enqueue Celery jobs." Harmless (the polling loop is
queue-agnostic and worked fine here) but misleading if a real timeout ever
fires. Worth a one-line scrub next time `index.html` is open.

---

## Next session — append below using this template:

```
### YYYY-MM-DD — <topic>

- What we did
- What we tried (kept + reverted)
- What's left
- Any new known issues / decisions
```
