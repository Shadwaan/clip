"""
Video + job metadata store — Redis-backed key/value (Milestone 1).

Two record types live here:

  • Video metadata. Replaces the Milestone-0 in-memory `_videos` dict.
    Same shape main.py was using (path/duration/size/filename/
    source_type/title/source_url). Optional TTL via
    CLIP_VIDEO_TTL_SECONDS (default: none — PRD §6's 7-day retention
    policy will land at the deletion layer, not here).

  • Job records. Stash the Modal FunctionCall.object_id under a
    UUID job_id, alongside the context the GET /v1/jobs/{id} handler
    needs to shape the response correctly (endpoint name, video_id,
    duration, model that ran, plus per-endpoint context like find
    query / ask question / summarise style). TTL on job records is
    7 days to match Modal's own output-expiry window — anything past
    that can't be retrieved from Modal anyway.

When chunk (b) lands (Postgres + S3), this module's interface stays
the same and the implementation becomes SQL-backed; callers don't
notice the swap.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import redis

log = logging.getLogger(__name__)

# Redis URL convention matches Celery's (so the same env var works for
# both, though we use a different DB to avoid stomping on Celery's task
# queue / result keyspace).
#
# DB layout (single Redis instance, easier dev):
#   /0 → Celery broker + result backend
#   /1 → app data (video metadata)
#
# Override entirely with CLIP_REDIS_URL if you prefer a different host.
_DEFAULT_URL = "redis://localhost:6379/1"
REDIS_URL = os.environ.get("CLIP_STORE_REDIS_URL", _DEFAULT_URL)

_VIDEO_TTL = os.environ.get("CLIP_VIDEO_TTL_SECONDS")
VIDEO_TTL_SECONDS: Optional[int] = int(_VIDEO_TTL) if _VIDEO_TTL else None

_VIDEO_KEY_PREFIX = "clip:video:"
_JOB_KEY_PREFIX = "clip:job:"

# Modal keeps spawned outputs available for 7 days (per their docs); after
# that FunctionCall.get raises OutputExpiredError. There's no point holding
# our job records longer than the underlying outputs.
JOB_TTL_SECONDS = 7 * 24 * 60 * 60


# Singleton client. redis-py is fork-safe via its connection pool, so the
# same module-level client works for both uvicorn and Celery worker procs.
_client: Optional[redis.Redis] = None


def client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        # Connect now so misconfiguration surfaces at boot, not at first request.
        _client.ping()
        log.info(f"Connected to video metadata store at {REDIS_URL}")
    return _client


def _video_key(video_id: str) -> str:
    return f"{_VIDEO_KEY_PREFIX}{video_id}"


def _job_key(job_id: str) -> str:
    return f"{_JOB_KEY_PREFIX}{job_id}"


# ---------- Videos ----------

def put_video(video_id: str, meta: dict) -> None:
    """
    Persist video metadata. `meta` is the same shape main.py has been
    using all along — path, duration, size, filename, source_type,
    title, source_url. No schema enforcement here; the gateway is the
    source of truth on shape.
    """
    payload = json.dumps(meta)
    c = client()
    if VIDEO_TTL_SECONDS:
        c.setex(_video_key(video_id), VIDEO_TTL_SECONDS, payload)
    else:
        c.set(_video_key(video_id), payload)


def get_video(video_id: str) -> Optional[dict]:
    raw = client().get(_video_key(video_id))
    if raw is None:
        return None
    return json.loads(raw)


def delete_video(video_id: str) -> bool:
    return bool(client().delete(_video_key(video_id)))


def exists(video_id: str) -> bool:
    return bool(client().exists(_video_key(video_id)))


# ---------- Jobs ----------

def put_job(job_id: str, meta: dict) -> None:
    """
    Persist a job record.

    Expected shape (enforced by callers, not here):
      {
        "modal_object_id": str,           # FunctionCall.object_id
        "endpoint":   "describe"|"find"|"summarise"|"ask"|"links",
        "video_id":   str,
        "duration":   float,
        "model":      str,                # ModelChoice.value, or "clip-ocr/easyocr" for links
        # Per-endpoint extras needed by the parser:
        "query":     str,                 # find only
        "style":     str,                 # summarise only
        "question":  str,                 # ask only
        "used_fallback": bool,            # ask only — Marlin used when Qwen unreachable
        # "links" needs no extras — the OCR payload is self-describing.
      }

    7-day TTL aligns with Modal's output retention; expired jobs match
    the modal.exception.OutputExpiredError path on the read side.
    """
    client().setex(_job_key(job_id), JOB_TTL_SECONDS, json.dumps(meta))


def get_job(job_id: str) -> Optional[dict]:
    raw = client().get(_job_key(job_id))
    if raw is None:
        return None
    return json.loads(raw)


def delete_job(job_id: str) -> bool:
    return bool(client().delete(_job_key(job_id)))
