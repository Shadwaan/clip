"""
Clip — Video Analysis Platform
FastAPI gateway (v0.1, Milestone 1 — Modal-native job queue)

Endpoints:
  POST /v1/videos                 upload a video
  POST /v1/videos/from-url        ingest from URL (YouTube / direct media)
  GET  /v1/videos/{id}            video metadata
  POST /v1/videos/{id}/describe   enqueue describe job → {job_id}
  POST /v1/videos/{id}/find       enqueue find job     → {job_id}
  POST /v1/videos/{id}/summarise  enqueue summarise job→ {job_id}
  POST /v1/videos/{id}/ask        enqueue ask job      → {job_id}
  GET  /v1/jobs/{id}              poll job status + result
  GET  /healthz

Async architecture (PRD §5.2):
  • Each inference endpoint calls ModalVLM.submit_<x>(...) which spawns a
    Modal FunctionCall and returns its object_id.
  • The gateway stashes a job record in Redis keyed by a UUID job_id —
    the Modal object_id plus enough context for the eventual response
    parser (endpoint name, model that ran, video duration, per-endpoint
    args like find query / ask question).
  • Clients poll GET /v1/jobs/{job_id}. The handler looks up the job
    record, reconstructs the FunctionCall via FunctionCall.from_id,
    and tries .get(timeout=0). TimeoutError → STARTED; result → parse
    and return SUCCESS; modal.exception.OutputExpiredError → FAILURE.

Why Modal-native and not Celery+Redis: every inference call already
crosses the Modal boundary, so making Modal also be the queue saves a
process (the Celery worker) and a piece of infrastructure that mostly
duplicates what Modal does for us anyway.

Milestone 1 still pending: S3-backed video bytes (currently still on
local disk), Postgres for durable video + job metadata (Redis is a
stopgap), webhooks, Next.js frontend with real video player.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import downloader
import inference
import routing
import store
from constants import ModelChoice
from inference import _strip_think  # parsers re-use the same Marlin <think> stripper
from sampling import probe_duration
from schemas import (
    AskRequest,
    AskResponse,
    DescribeResponse,
    FindRequest,
    FindResponse,
    IngestFromURLRequest,
    JobEnqueueResponse,
    JobStatusResponse,
    Match,
    SummariseRequest,
    SummariseResponse,
    UploadResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("clip")

# ---------- Config ----------
STORAGE_DIR = Path(os.environ.get("CLIP_STORAGE_DIR", "/tmp/clip-storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = int(os.environ.get("CLIP_MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024))  # 2 GB
ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


app = FastAPI(
    title="Clip — Video Analysis Platform",
    version="0.1.2",
    description="Ask questions of any video. Powered by Marlin-2B and Qwen3-VL-8B.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Helpers ----------

def _require_video(video_id: str) -> dict:
    meta = store.get_video(video_id)
    if meta is None:
        raise HTTPException(404, f"Video {video_id} not found")
    return meta


def _job_envelope(request: Request, job_id: str) -> JobEnqueueResponse:
    """Build the enqueue response with a self-referencing status URL."""
    status_url = str(request.url_for("get_job", job_id=job_id))
    return JobEnqueueResponse(
        job_id=job_id,
        status="PENDING",
        status_url=status_url,
    )


def _stash_job(
    job_id: str,
    modal_object_id: str,
    endpoint: str,
    video_id: str,
    duration: float,
    model: str,
    **extras: Any,
) -> None:
    """One-liner wrapper around store.put_job so endpoint bodies stay tight."""
    record = {
        "modal_object_id": modal_object_id,
        "endpoint": endpoint,
        "video_id": video_id,
        "duration": duration,
        "model": model,
        **extras,
    }
    store.put_job(job_id, record)


# ---------- Routes ----------

@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": app.version}


@app.post("/v1/videos", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video. Returns a video_id you'll use for subsequent calls.
    Bytes still land on local disk in Milestone 1; S3 swap is chunk (b).
    Metadata lives in Redis so it survives uvicorn restart.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            400,
            f"Unsupported extension {ext!r}. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    video_id = uuid.uuid4().hex
    dest = STORAGE_DIR / f"{video_id}{ext}"

    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413,
                    f"File exceeds max size of {MAX_UPLOAD_BYTES // (1024 ** 3)} GB",
                )
            f.write(chunk)

    try:
        duration = probe_duration(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read video: {e}")

    store.put_video(video_id, {
        "path": str(dest),
        "duration": duration,
        "size": size,
        "filename": file.filename,
        "source_type": "upload",
        "title": file.filename,
        "source_url": None,
    })

    return UploadResponse(
        video_id=video_id,
        duration_seconds=duration,
        size_bytes=size,
        source_type="upload",
        title=file.filename,
    )


@app.post("/v1/videos/from-url", response_model=UploadResponse)
def ingest_from_url(req: IngestFromURLRequest):
    """
    Ingest a video from a URL (yt-dlp or direct media).
    LEGAL NOTE: You must have rights to analyze the content; YouTube ToS
    restricts downloading, so use accordingly.
    """
    try:
        path, meta = downloader.fetch_url(req.url, STORAGE_DIR)
    except downloader.DownloadTooLarge as e:
        raise HTTPException(413, str(e))
    except downloader.DownloadTooLong as e:
        raise HTTPException(413, str(e))
    except downloader.DownloadError as e:
        raise HTTPException(400, str(e))

    try:
        duration = probe_duration(path)
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read downloaded video: {e}")

    video_id = uuid.uuid4().hex
    store.put_video(video_id, {
        "path": str(path),
        "duration": duration,
        "size": meta["size_bytes"],
        "filename": path.name,
        "source_type": meta["source_type"],
        "title": meta.get("title"),
        "source_url": meta.get("source_url"),
    })

    log.info(f"Ingested from URL: video_id={video_id} title={meta.get('title')!r} duration={duration:.1f}s")

    return UploadResponse(
        video_id=video_id,
        duration_seconds=duration,
        size_bytes=meta["size_bytes"],
        source_type=meta["source_type"],
        title=meta.get("title"),
        source_url=meta.get("source_url"),
    )


@app.get("/v1/videos/{video_id}")
def get_video(video_id: str):
    meta = _require_video(video_id)
    return {
        "video_id": video_id,
        "duration_seconds": meta["duration"],
        "size_bytes": meta["size"],
        "filename": meta["filename"],
        "source_type": meta.get("source_type", "upload"),
        "title": meta.get("title"),
        "source_url": meta.get("source_url"),
    }


# ---------- Inference endpoints — spawn, don't execute ----------
#
# Per PRD §5.2 each endpoint spawns a Modal FunctionCall and returns the
# job envelope. Clients poll GET /v1/jobs/{id} for the result.

@app.post("/v1/videos/{video_id}/describe", response_model=JobEnqueueResponse)
def describe(
    request: Request,
    video_id: str,
    model: str | None = Query(None, description="Force a model: marlin-2b, timelens-8b, qwen3-vl-8b"),
):
    meta = _require_video(video_id)
    duration = meta["duration"]
    choice = routing.choose_model(model, duration, None)
    log.info(f"video={video_id} model={choice.value} duration={duration:.1f}s")

    m = inference.get_model(choice)
    if not hasattr(m, "submit_caption"):
        # Backend without the Modal spawn shim (e.g. local VideoVLM).
        # Modal-native job queue only meaningful when backend is Modal.
        raise HTTPException(
            501,
            f"Backend for {choice.value} does not support async spawn yet "
            f"(only the Modal backend does). Run with CLIP_BACKEND=modal.",
        )

    object_id = m.submit_caption(meta["path"])
    job_id = uuid.uuid4().hex
    _stash_job(job_id, object_id, "describe", video_id, duration, choice.value)
    log.info(f"spawned describe job={job_id} modal={object_id}")
    return _job_envelope(request, job_id)


@app.post("/v1/videos/{video_id}/find", response_model=JobEnqueueResponse)
def find(request: Request, video_id: str, req: FindRequest):
    meta = _require_video(video_id)
    duration = meta["duration"]
    choice = routing.choose_model(req.model, duration, req.query)
    log.info(f"video={video_id} model={choice.value} query={req.query!r}")

    m = inference.get_model(choice)
    if not hasattr(m, "submit_find"):
        raise HTTPException(
            501,
            f"Backend for {choice.value} does not support async spawn yet "
            f"(only the Modal backend does). Run with CLIP_BACKEND=modal.",
        )

    object_id = m.submit_find(meta["path"], req.query)
    job_id = uuid.uuid4().hex
    _stash_job(
        job_id, object_id, "find", video_id, duration, choice.value,
        query=req.query,
    )
    log.info(f"spawned find job={job_id} modal={object_id}")
    return _job_envelope(request, job_id)


@app.post("/v1/videos/{video_id}/summarise", response_model=JobEnqueueResponse)
def summarise(request: Request, video_id: str, req: SummariseRequest):
    meta = _require_video(video_id)
    duration = meta["duration"]
    choice = routing.choose_model(req.model, duration, None)
    log.info(f"video={video_id} model={choice.value} duration={duration:.1f}s")

    m = inference.get_model(choice)
    if not hasattr(m, "submit_caption"):
        raise HTTPException(
            501,
            f"Backend for {choice.value} does not support async spawn yet "
            f"(only the Modal backend does). Run with CLIP_BACKEND=modal.",
        )

    # /summarise rides on the same caption() output as /describe; the
    # parser-side dispatch (on `endpoint`) is what makes the response
    # come back as bullets instead of the full DescribeResponse.
    object_id = m.submit_caption(meta["path"])
    job_id = uuid.uuid4().hex
    _stash_job(
        job_id, object_id, "summarise", video_id, duration, choice.value,
        style=req.style,
    )
    log.info(f"spawned summarise job={job_id} modal={object_id}")
    return _job_envelope(request, job_id)


@app.post("/v1/videos/{video_id}/ask", response_model=JobEnqueueResponse)
def ask(request: Request, video_id: str, req: AskRequest):
    """
    Open-ended Q&A. Defaults to Qwen3-VL-8B per PRD §5.4. Falls back to
    Marlin's prompted-generate path if the Qwen backend can't be bound
    (e.g. the QwenVL class wasn't deployed, OOM, etc.) — the job record
    notes `used_fallback=True` so the GET handler reports the actually-
    used model on the response.
    """
    meta = _require_video(video_id)
    duration = meta["duration"]
    video_path = meta["path"]

    if req.model:
        choice = routing.choose_model(req.model, duration, req.question)
    else:
        choice = ModelChoice.QWEN3_VL_8B

    log.info(f"video={video_id} model={choice.value} question={req.question!r}")

    used_fallback = False
    if choice == ModelChoice.QWEN3_VL_8B:
        try:
            m = inference.get_model(choice)
            object_id = m.submit_ask(video_path, req.question)
        except (NotImplementedError, RuntimeError) as e:
            log.warning(
                f"Qwen3-VL backend unavailable ({e}); falling back to "
                f"Marlin generate path for /ask."
            )
            choice = ModelChoice.MARLIN_2B
            used_fallback = True
            m = inference.get_model(choice)
            prompt = inference.ASK_PROMPT_TEMPLATE.format(question=req.question)
            object_id = m.submit_run(video_path, prompt)
    else:
        # Explicit Marlin/TimeLens request → prompted path
        m = inference.get_model(choice)
        if not hasattr(m, "submit_run"):
            raise HTTPException(
                501,
                f"Backend for {choice.value} does not support async spawn yet.",
            )
        prompt = inference.ASK_PROMPT_TEMPLATE.format(question=req.question)
        object_id = m.submit_run(video_path, prompt)

    job_id = uuid.uuid4().hex
    _stash_job(
        job_id, object_id, "ask", video_id, duration, choice.value,
        question=req.question,
        used_fallback=used_fallback,
    )
    log.info(f"spawned ask job={job_id} modal={object_id} used_fallback={used_fallback}")
    return _job_envelope(request, job_id)


# ---------- Result parsing on GET /v1/jobs/{id} ----------
#
# Each parser takes the job record + the raw Modal return value and
# shapes a response that matches the original endpoint's schema.
# Same parsing logic that lived inline in the Milestone-0 endpoints,
# just moved server-side of the queue.

def _parse_describe(job: dict, raw_result: dict) -> dict:
    duration = job["duration"]
    raw = raw_result["raw"]
    cleaned = _strip_think(raw)
    summary, scenes, events = routing.parse_describe(cleaned, duration=duration)
    return DescribeResponse(
        video_id=job["video_id"],
        model=job["model"],
        summary=summary or cleaned,
        scenes=scenes,
        events=events,
        raw_output=raw,
    ).model_dump()


def _parse_find(job: dict, raw_result: dict) -> dict:
    duration = job["duration"]
    query = job["query"]
    raw = raw_result["raw"]
    matches: list[Match] = []
    span = raw_result.get("span")
    if span:
        s, e = span
        s = max(0.0, min(s, duration))
        e = max(s, min(e, duration))
        matches.append(Match(start=s, end=e, description=query))
    return FindResponse(
        video_id=job["video_id"],
        model=job["model"],
        query=query,
        matches=matches,
        raw_output=raw,
    ).model_dump()


def _parse_summarise(job: dict, raw_result: dict) -> dict:
    duration = job["duration"]
    raw = raw_result["raw"]
    cleaned = _strip_think(raw)
    _, _, events = routing.parse_describe(cleaned, duration=duration)
    bullets = routing.summarise_from_events(events, duration=duration)
    return SummariseResponse(
        video_id=job["video_id"],
        model=job["model"],
        style=job.get("style", "bullets"),
        bullets=bullets,
        raw_output=raw,
    ).model_dump()


def _parse_ask(job: dict, raw_result: dict) -> dict:
    raw = raw_result["raw"]
    cleaned = _strip_think(raw)
    return AskResponse(
        video_id=job["video_id"],
        model=job["model"],
        question=job["question"],
        answer=cleaned,
        raw_output=raw,
    ).model_dump()


_PARSERS = {
    "describe": _parse_describe,
    "find": _parse_find,
    "summarise": _parse_summarise,
    "ask": _parse_ask,
}


# ---------- Job polling ----------

@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, name="get_job")
def get_job(job_id: str):
    """
    Poll job status and (if ready) result.

    States we surface:
      STARTED  — Modal still running the call. We can't distinguish
                 "queued behind a busy container" from "actively
                 executing" via FunctionCall.get(timeout=0); both
                 raise TimeoutError. For v0.1 we report STARTED for
                 either — the actual moment-by-moment distinction
                 isn't meaningful to the user.
      SUCCESS  — Modal returned a result; we dispatch to the matching
                 parser based on the stashed endpoint name.
      FAILURE  — Modal raised on .get() (the wrapped task exception),
                 or the output expired (>7 days since spawn).

    PENDING is never returned for known job_ids — Modal accepts the
    spawn synchronously, so the moment we have an object_id the call
    is at least "queued at Modal." For an unknown job_id we return
    404 outright (unlike Celery's PENDING-means-anything ambiguity).
    """
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")

    fn_call = modal.FunctionCall.from_id(job["modal_object_id"])

    try:
        raw_result = fn_call.get(timeout=0)
    except modal.exception.OutputExpiredError as e:
        # MUST come before the TimeoutError catch below. OutputExpiredError
        # is a subclass of modal.exception.TimeoutError (verified at session
        # start: MRO is OutputExpiredError → modal.exception.TimeoutError →
        # Error → Exception). If TimeoutError caught first, expired results
        # would silently report STARTED forever.
        return JobStatusResponse(
            job_id=job_id,
            status="FAILURE",
            error=f"Output expired (Modal retains spawned results for 7 days): {e}",
        )
    except (TimeoutError, modal.exception.TimeoutError):
        # Not yet done. Modal's own doc-ocr example catches the built-in
        # TimeoutError (modal.com/docs/examples/doc_ocr_webapp), but
        # modal.exception.TimeoutError does NOT inherit from
        # builtins.TimeoutError as of modal 0.66 — verified at session
        # start. Catch both so behavior is robust to whichever Modal
        # version is installed; if they ever converge, the redundant
        # catch is harmless.
        return JobStatusResponse(job_id=job_id, status="STARTED")
    except Exception as e:
        # Anything else: the inference itself raised, or there was a
        # transport-level failure. Surface repr() — we don't expose the
        # traceback over the API (use Modal logs for that).
        log.exception(f"Job {job_id} failed during result fetch")
        return JobStatusResponse(job_id=job_id, status="FAILURE", error=repr(e))

    endpoint = job["endpoint"]
    parser = _PARSERS.get(endpoint)
    if parser is None:
        # Defensive — only happens if a future endpoint name slipped in
        # without a parser entry.
        return JobStatusResponse(
            job_id=job_id,
            status="FAILURE",
            error=f"No parser registered for endpoint {endpoint!r}",
        )

    shaped = parser(job, raw_result)
    return JobStatusResponse(job_id=job_id, status="SUCCESS", result=shaped)


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
