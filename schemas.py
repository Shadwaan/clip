"""Request and response schemas."""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


# Mirror of Celery's task state strings. We don't import celery.states here
# so this module stays import-cheap and usable from frontends / tests that
# don't need the celery dependency.
JobStatus = Literal["PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY", "REVOKED"]


# ---------- Requests ----------

class FindRequest(BaseModel):
    query: str = Field(..., description='Natural-language description of the moment to find, e.g. "when the dog jumps"')
    model: Optional[Literal["marlin-2b", "timelens-8b", "qwen3-vl-8b"]] = None


class IngestFromURLRequest(BaseModel):
    url: str = Field(..., description="YouTube URL, social media link, or direct media URL")


class AskRequest(BaseModel):
    question: str
    model: Optional[Literal["marlin-2b", "timelens-8b", "qwen3-vl-8b"]] = None


class SummariseRequest(BaseModel):
    style: Literal["bullets", "chapters"] = "bullets"
    model: Optional[Literal["marlin-2b", "timelens-8b", "qwen3-vl-8b"]] = None


# ---------- Responses ----------

class UploadResponse(BaseModel):
    video_id: str
    duration_seconds: float
    size_bytes: int
    source_type: Literal["upload", "direct", "yt-dlp"] = "upload"
    title: Optional[str] = None
    source_url: Optional[str] = None


class Scene(BaseModel):
    start: float
    end: float
    caption: str


class Event(BaseModel):
    timestamp: float
    event: str


class DescribeResponse(BaseModel):
    video_id: str
    model: str
    summary: str
    scenes: List[Scene] = []
    events: List[Event] = []
    raw_output: Optional[str] = None  # for debugging


class Match(BaseModel):
    start: float
    end: float
    description: Optional[str] = None
    confidence: Optional[float] = None


class FindResponse(BaseModel):
    video_id: str
    model: str
    query: str
    matches: List[Match]
    raw_output: Optional[str] = None


class SummariseResponse(BaseModel):
    video_id: str
    model: str
    style: str
    bullets: List[str] = []
    raw_output: Optional[str] = None


class AskResponse(BaseModel):
    video_id: str
    model: str
    question: str
    answer: str
    raw_output: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str


# ---------- Async job envelopes (Milestone 1, PRD §5.2) ----------
#
# Per PRD §5.2 each inference endpoint becomes "enqueue a job → poll".
# /describe, /find, /summarise, /ask all return JobEnqueueResponse
# instead of their structured payload directly. The structured payload
# is then exposed under JobStatusResponse.result once status == SUCCESS.
#
# We don't validate the shape of `result` here — it varies by endpoint
# (DescribeResponse / FindResponse / SummariseResponse / AskResponse).
# Callers can detect which by the original endpoint they POSTed to.

class JobEnqueueResponse(BaseModel):
    job_id: str
    status: JobStatus = "PENDING"
    status_url: str = Field(..., description="GET this URL to poll job status + result")


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
