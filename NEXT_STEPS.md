# Next Steps — Clip Build

You have the scaffold. To get from "code on disk" to "running demo", in priority order:

## 1. Get a GPU (30 min)

Marlin-2B needs ~5 GB VRAM, so you have options:

- **Free / cheapest**: Google Colab T4 (free tier) — fine for testing. Spin up a notebook, `pip install` requirements, run the FastAPI server with `pyngrok` for a public URL.
- **Pay-as-you-go**: [Modal](https://modal.com) (~$0.60/hr A10), [RunPod](https://runpod.io) (~$0.30/hr A10 spot), [Vast.ai](https://vast.ai) (~$0.20/hr 3090 spot). All take 10 min to set up.
- **Managed inference**: [HF Inference Endpoints](https://ui.endpoints.huggingface.co) — point at `NemoStation/Marlin-2B`, get a URL. Costs more (~$0.50/hr A10) but zero ops. Best for production once you validate.
- **Local**: if you have a 4090/3090 or M-series Mac with 16 GB unified memory, it'll run.

For Bangladesh (no local Anthropic data center, latency to US/EU), Modal's Asia-Pacific region or Singapore-hosted RunPod gives the best UX.

## 2. Spike test (15 min)

```bash
cd backend
pip install -r requirements.txt
python test_smoke.py /path/to/short_video.mp4 --probe-only   # sanity-check sampling, no GPU needed
python test_smoke.py /path/to/short_video.mp4 marlin-2b      # first time downloads ~5GB
```

If the smoke test prints sensible output, you're done with Milestone 0.

## 3. Boot the API (5 min)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `frontend/index.html` in a browser, upload a clip, hit Describe. The first call loads the model (~30s); subsequent calls reuse it.

## 4. Things that will probably break first

In rough order of likelihood:

1. **Model loading OOM** → drop to fp16 instead of bf16 (older GPUs); or use 4-bit quant via `bitsandbytes` (add to requirements, use `load_in_4bit=True`)
2. **Frame sampling hangs on weird codecs** → `pip install imageio-ffmpeg`, force ffmpeg backend
3. **Marlin output doesn't parse cleanly** → the `<think>` strip + `parse_describe` regex are conservative; check `raw_output` in the response, adjust prompts in `inference.py` if scenes/events come out empty
4. **Long videos (>5 min) timeout the sync endpoint** → either bump uvicorn's `--timeout-keep-alive`, or skip ahead to Milestone 1 (Celery queue)
5. **CORS errors in the browser** → main.py allows `*` for dev; fine

## 5. Decision points before Milestone 1

- **Stay self-hosted or go managed?** If you want devs to use this, run it on HF Inference Endpoints + put up a HF Space as the demo. That gets you SEO and discovery for free.
- **OSS licence?** Apache 2.0 to match Marlin. Avoid GPL (deters commercial use). MIT also fine.
- **Domain?** Skip the SaaS landing page until you have ~50 GitHub stars; just keep it on `github.com/<you>/clip` with a great README + Loom demo.
- **Bangla market angle?** Your home advantage. KriShop CCTV footage analysis, agri-machinery video monitoring (the thresher pilot), or Folon UGC moderation are all legitimate v1 customers. Could be the wedge: prove the platform on iFarmer's own video corpus, then open-source it.

## 6. Things to read while the model downloads

- Marlin paper: [arxiv:2501.00513](https://arxiv.org/abs/2501.00513) — the architecture and training recipe
- TimeLens paper: [arxiv:2512.14698](https://arxiv.org/abs/2512.14698) — the temporal grounding methodology, plus the dataset quality argument (relevant if you later finetune)
- Qwen3-VL technical report — the base model both fine-tunes from; explains the frame sampling and patching scheme
- Pricing comparison: Gemini 2.5 Flash video pricing vs. Modal A10 hourly — confirms the 10x cost arb is real

## 7. Architectural roadmap — artifact storage + transcriber merge

Captured 2026-05-30, deferred behind the timestamp-wrap fix. Two related shifts that should land together once the parser quality is solid.

**Why this matters now.** Currently Clip treats videos as ephemeral: video → Modal call → outputs returned in JSON → forgotten. `raw_output` lives only in the response body. Nothing is archived. Two emerging needs break this model:

1. **Eval archive.** Every job's `raw_output` + parsed result is dev-time gold for catching regressions when the parser, prompts, or models change. Easy to add (append a JSONL line on job success), expensive to retrofit when we already have a quality bug we can't replay.
2. **Convergence with the audio transcriber.** The longer-term goal is meeting-video analysis (Google Meet exports, etc.) where visual captions and audio transcripts feed an LLM that needs both streams temporally aligned. The audio transcriber already writes files locally; for the two tools to compose, they need a shared on-disk artifact layout — not two parallel APIs the downstream consumer has to glue together.

**Proposed artifact folder layout.** A single per-media-id directory that any tool in the family writes into:

```
artifacts/{media_id}/
    source.{mov,mp4,webm}     # original file (was CLIP_STORAGE_DIR/{video_id}.{ext})
    metadata.json             # duration, source_url, ingest time, filename
    describe.json             # latest visual caption + scenes + events
    find/{query_hash}.json    # cached find results
    summarise.json
    transcript.json           # written by the audio transcriber (future)
    runs.jsonl                # append-only eval log: every job's raw + parsed output + model + prompt
```

`media_id` replaces the current `video_id` as the cross-tool key. Both Clip and the transcriber write into the same root, keyed by the same id, and the downstream LLM reads the folder — not the two services separately. That's what makes audio↔visual temporal alignment cheap to use.

**What this changes for Clip:**
- `CLIP_STORAGE_DIR` semantics shift from "scratch space" to "shared artifact root."
- `store.py` either grows an artifact concept alongside `video`/`job`, or videos become one artifact type. Redis stays the fast index; disk is canonical.
- Job-completion path writes `{endpoint}.json` and appends a `runs.jsonl` line. Outputs stop being Modal-7-day-only.
- The eval-archive idea (item 1 above) drops in for free.

**What we are NOT doing yet:**
- Building the transcriber integration. The audio tool isn't merging in this milestone; we only need the storage layout to be compatible.
- Defining the downstream-LLM contract. That comes after both tools share an artifact root.
- Migrating off Redis as the metadata index.

**Sequencing:** fix the timestamp-wrap bug first (without trustworthy timestamps, archived outputs are archives of broken data). Then this artifact-folder migration. Roughly one focused day of work; no architectural risk because both directions of the migration are obvious file moves + a `store.py` extension.

## 8. Follow-up: `/find` shares the same ~130s horizon (FIX NEEDED before long-video search)

Captured 2026-05-30, right after the `/describe` chunked-caption fix landed (modal_app.py `caption()` now splits >130s videos into ≤`CHUNK_SEC` windows, captions each, and re-offsets timestamps).

**The problem.** Marlin's native `caption()` and `find()` both decode through the same internal frame sampler, which caps total frames and so only "sees" the first ~129.5s of any video (diagnosed on a 907.3s webm: timestamps climbed to 129.5s, then reset to 30.0 and looped). We fixed `caption()` by chunking. **`find()` was NOT fixed** — it has the identical horizon, so temporal search on anything over ~2 min silently misses everything past ~130s. `Find` + long-video `Ask`/`Summarise` (which lean on this) are unreliable until this is addressed.

**Why it's a different shape than the caption fix.** `caption()` returns a *stream* of `<a-b>` events, so chunking is "caption each window, offset, concatenate" — order-preserving and additive. `find()` returns a *single best-match span* (`{"span": (s, e), "format_ok": bool}`) for a query. You can't just concatenate; you have to:
- run `find()` on each ≤`CHUNK_SEC` chunk,
- offset each chunk's returned span by the chunk start,
- then **pick across chunks** — either the highest-confidence hit (Marlin's find mode doesn't currently surface a score, so we'd need one), or return *all* per-chunk hits as a ranked list and change the `/find` response contract from one span to many.

**Decision to make first:** does `/find` stay single-span (needs a cross-chunk scoring/selection rule) or become multi-span (API contract change, but arguably more useful — "show me every time X happens")? Lean multi-span; it's the honest answer for long videos and matches how `/describe` already returns many events.

**Watch-outs inherited from the caption fix:**
- Stream-copy (`-c copy`) chunk cuts land on keyframes, so chunk boundaries drift by ±keyframe-interval. On the 907s webm this produced ~5–20s **overlaps** at seams (a chunk's content ran past its nominal 120s window, so adjacent chunks overlapped in source and timestamps stepped backward at the seam). For `find` this means a match near a chunk boundary can appear in two chunks — dedupe overlapping spans.
- Per-chunk sequential calls multiply latency (~8× for a 15-min video); the Modal `timeout` was bumped to 1800s for `caption` and `find`'s class shares it, but confirm before shipping.

## 9. Open questions you should think about

1. **What's the wedge?** "General video analysis API" is a crowded market (Twelve Labs, Marqo, OpenAI). What can Clip do that they don't? My guesses: **self-hostable** (their videos never leave your infra) + **cheaper** (10x) + **structured outputs** (their APIs are chat-style). Pick one and lead with it.
2. **Distribution?** Reddit (r/LocalLLaMA), Hacker News (Show HN once Milestone 1 is done), Twitter/X with side-by-side demo videos vs. Gemini, blog post benchmarking Marlin vs. GPT-4o on a specific use case (sports? CCTV?).
3. **What's the moat?** Not the models (Tencent/NemoStation own those). The moat is: (a) the orchestration layer, (b) the routing rules tuned across many real video types, (c) the fine-tunes you'll do on top, (d) the customer relationships.
