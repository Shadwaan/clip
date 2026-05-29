# Clip — Video Analysis Platform
## Product Requirements Document (v0.1)

**Owner:** Zayan Khan
**Date:** May 2026
**Status:** Draft for build

---

## 1. Problem & Opportunity

Anyone with a video — security operators, content teams, sports analysts, lawyers reviewing CCTV, journalists, ML engineers building downstream pipelines — needs to **ask questions of video** instead of scrubbing through it. The status quo is either:

- **Manual review** (slow, doesn't scale past ~1hr of footage)
- **Frontier APIs** (Gemini 2.5 Pro, GPT-5 video) — capable but expensive (~$0.30–$1.00/min of video), latency-bound, and not deployable on private footage
- **Single-purpose tools** (object detection libraries, OCR-on-frames) — narrow, no natural-language interface

The 2026 open-weights VLM stack closed the gap. Marlin-2B and TimeLens-8B now match or exceed frontier proprietary models on the two questions that matter — **what is happening** (dense captioning) and **when** (temporal grounding) — at a fraction of the cost.

**Opportunity:** a self-hostable, API-first video analysis platform built on these open models, priced 10x below Gemini, with deployment options spanning a developer laptop, a single GPU VM, and managed HF Inference Endpoints.

---

## 2. Users & Use Cases

### Primary personas

| Persona | Job-to-be-done | Volume |
|---|---|---|
| **Dev/Indie hacker** | Bolt video understanding into my product without paying frontier API rates | 1–100 videos/day |
| **Ops analyst** | Search internal footage ("find every time someone enters the loading bay after 9pm") | 10–500 videos/day, mostly long |
| **Content team** | Auto-summarise + chapter long-form video for distribution | 5–50 videos/day, 10–60 min each |
| **ML engineer** | Generate labelled video data for fine-tuning | Bulk batch jobs |

### Core jobs

1. **Describe** — "What's happening in this video?" → multi-paragraph dense caption
2. **Find** — "When does the dog jump?" → list of `(start, end)` timestamps
3. **Summarise** — long video → bullet-point or chaptered summary
4. **Ask** — open-ended Q&A grounded in the video
5. **Extract** — structured output (objects, actions, scene changes, text overlays) as JSON

---

## 3. Non-goals (for v0.1)

- Video **generation** (text→video). Out of scope.
- **Audio-only** understanding (use Whisper separately for now; audio integration in v0.3).
- **Real-time streaming inference** (live cameras). v0.4.
- **Multi-tenant SaaS billing.** v0.5+.
- **Mobile clients.** Web first.

---

## 4. Solution Overview

Clip is an inference platform with three layers:

1. **Model layer** — Marlin-2B (default), TimeLens-8B (precision), Qwen3-VL-8B (general reasoning fallback). All Qwen-family, same toolchain.
2. **Orchestration layer** — FastAPI service that handles upload, frame sampling, prompt routing, model selection, and structured output parsing.
3. **Client layer** — REST API, plus a minimal web UI for non-developers.

### Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Default model | **Marlin-2B** | Best quality/cost ratio at 2B; runs on a single consumer GPU; native temporal grounding |
| Precision model | **TimeLens-8B** | SOTA temporal grounding when "when did X happen" needs to be exact |
| Inference runtime | **Transformers + bf16 → vLLM later** | Get to working first; vLLM port once latency matters |
| Frame sampling | **FPS-adaptive (0.5–2 fps)** | Match Marlin's training distribution; cap at 512 frames for long videos |
| Storage | **S3-compatible (local: MinIO, prod: R2/S3)** | Standard, cheap egress |
| Queue | **Celery + Redis** | Inference is async by default; sync endpoint only for <30s clips |
| API style | **REST + webhook for async jobs** | Simplest integration for v1; GraphQL later if needed |

---

## 5. Functional Requirements

### 5.1 Ingest
- Accept `.mp4`, `.mov`, `.mkv`, `.webm`, up to 2 GB / 2 hr
- Pre-signed upload URLs to object storage
- ffmpeg-based normalisation (re-encode to H.264 720p if needed)
- Frame sampling: adaptive FPS based on duration (longer → lower FPS; cap at 512 frames)

### 5.2 Endpoints (v0.1)

```
POST /v1/videos                    upload (returns video_id)
POST /v1/videos/{id}/describe      dense caption, sync if <60s else async
POST /v1/videos/{id}/find          natural-language temporal grounding
POST /v1/videos/{id}/summarise     chaptered or bullet summary
POST /v1/videos/{id}/ask           open-ended Q&A
GET  /v1/jobs/{id}                 async job status + result
POST /v1/webhooks                  register webhook for job completion
```

### 5.3 Output shape

All endpoints return JSON. Structured outputs use a strict schema; the model layer parses Marlin's `<scene>`/`<event>` tags into the schema before responding.

```json
// /describe
{
  "video_id": "...",
  "summary": "A person walks into a kitchen and prepares coffee.",
  "scenes": [
    {"start": 0.0, "end": 12.4, "caption": "Empty kitchen, morning light."},
    {"start": 12.4, "end": 41.0, "caption": "Person enters, opens cupboard..."}
  ],
  "events": [
    {"timestamp": 14.2, "event": "person enters frame"},
    {"timestamp": 33.8, "event": "kettle turned on"}
  ]
}

// /find
{
  "query": "when does the dog jump",
  "matches": [
    {"start": 4.2, "end": 5.1, "confidence": 0.91},
    {"start": 12.8, "end": 14.0, "confidence": 0.74}
  ]
}
```

### 5.4 Model routing rules

- Default → Marlin-2B
- Query contains explicit temporal language ("exactly when", "precise timestamp") OR video > 10 min → upgrade to TimeLens-8B
- Open-ended reasoning ("why did the person do X") → Qwen3-VL-8B-Instruct
- User can force model via `?model=` param

---

## 6. Non-functional Requirements

| Dimension | Target |
|---|---|
| **P50 latency (60s video, Marlin-2B, A10 GPU)** | < 8s |
| **P95 latency (5min video)** | < 45s |
| **Cost target (per minute of video, self-hosted A10)** | < $0.01 |
| **Cost target (managed, HF Inference Endpoint)** | < $0.05 |
| **Concurrency** | 4 GPU workers, autoscaling to 20 |
| **Storage retention** | Videos kept 7 days by default; results indefinitely |
| **Auth** | API key (v0.1); OAuth later |

---

## 7. Architecture

```
┌──────────┐     ┌─────────────┐     ┌──────────────┐
│  Client  │────▶│   FastAPI   │────▶│  PostgreSQL  │
│  (web,   │     │   Gateway   │     │  (jobs, meta)│
│  curl)   │◀────│             │◀────│              │
└──────────┘     └─────┬───────┘     └──────────────┘
                       │
                       ▼
                ┌─────────────┐
                │    Redis    │  (queue)
                └─────┬───────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
     ┌──────────────┐  ┌──────────────┐
     │  Celery      │  │  Celery      │
     │  Worker      │  │  Worker      │
     │  (GPU 0)     │  │  (GPU 1)     │
     │              │  │              │
     │  Marlin-2B   │  │ TimeLens-8B  │
     │  bf16        │  │  bf16        │
     └──────┬───────┘  └──────┬───────┘
            │                 │
            └────────┬────────┘
                     ▼
            ┌────────────────┐
            │  S3 / MinIO    │ (raw video + sampled frames)
            └────────────────┘
```

---

## 8. Build Plan

### Milestone 0 — Spike (this week)
- [ ] Marlin-2B inference works end-to-end on a 30s clip via FastAPI
- [ ] Frame sampling pipeline (ffmpeg + PIL)
- [ ] Sync `/describe` and `/find` endpoints
- [ ] Minimal HTML upload form

### Milestone 1 — Useful demo (week 2)
- [ ] Async job queue (Celery + Redis)
- [ ] S3-compatible storage
- [ ] All 4 endpoints (`describe`, `find`, `summarise`, `ask`)
- [ ] Next.js frontend with video player + timestamp jump on result click
- [ ] Deployed on a single A10 (Modal or RunPod)

### Milestone 2 — Multi-model (week 3)
- [ ] TimeLens-8B added with auto-routing rules
- [ ] Webhook support for async completion
- [ ] API key auth
- [ ] HF Spaces public demo (gated, Pro tier ZeroGPU)

### Milestone 3 — Production-ish (week 4–5)
- [ ] vLLM serving instead of raw Transformers (3–5x throughput)
- [ ] Batch endpoint for bulk processing
- [ ] Postgres-backed metadata + result cache
- [ ] Prometheus metrics + Grafana
- [ ] Rate limiting

### Milestone 4 — Distribution (week 6+)
- [ ] Docs site (Mintlify or Nextra)
- [ ] Python + TypeScript SDKs
- [ ] Pricing page if going SaaS

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Marlin-2B `<think>` token artifact pollutes output | High | Low | Strip with regex post-processing; documented in model card |
| GPU costs spike with long videos | Medium | High | Hard cap at 512 sampled frames; warn user; switch to TimeLens for >10min |
| Hallucinated timestamps | Medium | Medium | Validate timestamps against video duration; flag low-confidence in API response |
| HF model gets pulled or licence-changed | Low | High | Mirror weights to private S3 on first download; pin model revision SHA |
| New SOTA model drops (high) | High | Low | Architecture is model-agnostic; routing layer makes swap a config change |

---

## 10. Open Questions

1. **Go SaaS or stay OSS?** — OSS-first gets developer adoption; SaaS only on infra ops. Default: OSS core + paid hosted tier.
2. **Audio?** — Whisper integration in v0.3, fused captions via prompt engineering, or wait for omni models like Qwen2.5-Omni-7B?
3. **Localisation?** — Marlin is English-only. Bangla market exists (your home turf). Defer until v0.5; finetune Marlin on Bangla captions if signal is there.
4. **Pricing model if hosted?** — Per-minute, per-token, or flat tier? Per-minute is easiest to explain; per-token aligns cost.

---

## 11. Success Metrics

| Metric | 30-day target | 90-day target |
|---|---|---|
| API calls / week | 1,000 | 10,000 |
| Self-hosted installs (Docker pulls) | 200 | 2,000 |
| Active developers (≥1 call/week) | 25 | 200 |
| P50 latency (60s video) | < 10s | < 5s |
| GitHub stars | 100 | 1,000 |
| Cost/minute (hosted) | < $0.10 | < $0.03 |
