"""
Model routing + output parsing.

Routing rules (v0.1):
  - User-forced via API param → use that
  - "Find" with explicit precision language → TimeLens-8B
  - Video > 10 min → TimeLens-8B
  - Otherwise → Marlin-2B
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from constants import ModelChoice
from schemas import Event, Match, Scene

PRECISION_KEYWORDS = (
    "exactly", "precisely", "exact time", "exact moment",
    "specific timestamp", "to the second",
)

LONG_VIDEO_THRESHOLD_SEC = 10 * 60  # 10 minutes


def choose_model(
    forced: Optional[str],
    duration_sec: float,
    query: Optional[str] = None,
) -> ModelChoice:
    if forced:
        return ModelChoice(forced)
    if duration_sec > LONG_VIDEO_THRESHOLD_SEC:
        return ModelChoice.TIMELENS_8B
    if query and any(kw in query.lower() for kw in PRECISION_KEYWORDS):
        return ModelChoice.TIMELENS_8B
    return ModelChoice.MARLIN_2B


# ---------- Output parsers ----------
#
# Models emit semi-structured text. We extract timestamps + segments
# with regex. Fragile by design — we keep `raw_output` in the response
# so callers can fall back to it if parsing fails.

_TIMESTAMP_PAT = re.compile(r"\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?")
_RANGE_PAT = re.compile(
    r"\[?\s*(\d{1,3}(?:\.\d+)?)\s*[,\-–]\s*(\d{1,3}(?:\.\d+)?)\s*\]?"
)

# Marlin's native output uses "<start: end> description" tags inside an
# Events: block. Different shape from the [HH:MM:SS] format above — angle
# brackets, decimal seconds, two timestamps per tag, free-text after.
_MARLIN_RANGE_PAT = re.compile(
    r"<\s*(\d+(?:\.\d+)?)\s*[:,\-–]\s*(\d+(?:\.\d+)?)\s*>"
)
_MARLIN_SCENE_HEADER = re.compile(r"^\s*Scene:\s*", re.IGNORECASE | re.MULTILINE)
_MARLIN_EVENTS_HEADER = re.compile(r"^\s*Events:\s*", re.IGNORECASE | re.MULTILINE)


def _hms_to_seconds(match: re.Match) -> float:
    g = match.groups()
    if g[2] is not None:
        h, m, s = int(g[0]), int(g[1]), int(g[2])
        return h * 3600 + m * 60 + s
    # mm:ss
    m, s = int(g[0]), int(g[1])
    return m * 60 + s


def _clamp_to_duration(
    scenes: List[Scene],
    events: List[Event],
    duration: float,
    start_grace: float = 2.0,
) -> Tuple[List[Scene], List[Event]]:
    """
    PRD §9 mitigation: validate timestamps against video duration.

    Marlin (and any temporal-grounding VLM) can emit timestamps slightly
    past the actual video — sometimes by rounding the last frame,
    sometimes by extrapolating beyond what the model actually saw. Two
    policies:

      • If `start > duration + start_grace` → drop the entry. The model
        claimed something happened at a time the video doesn't have;
        treat as hallucination.
      • Otherwise → clamp start and end to [0, duration] so the response
        never references nonexistent time.

    `start_grace` (default 2s) absorbs harmless rounding (e.g. event
    timestamp 151.0 in a 149.0s video stays at 149.0 rather than being
    dropped).
    """
    max_start = duration + start_grace

    kept_scenes: List[Scene] = []
    for s in scenes:
        if s.start > max_start:
            continue
        start = min(s.start, duration)
        end = min(s.end, duration)
        if end < start:
            end = start
        # Drop zero-length (or near-zero) scenes — these are clamp
        # artifacts where the original entry referenced time past the
        # video and both endpoints collapsed to `duration`. A real
        # Marlin scene is ≥0.5s based on the model's training-time
        # output granularity, so 0.1s is a safe lower bound.
        if end - start < 0.1:
            continue
        kept_scenes.append(Scene(start=start, end=end, caption=s.caption))

    kept_events: List[Event] = []
    for e in events:
        if e.timestamp > max_start:
            continue
        ts = min(e.timestamp, duration)
        kept_events.append(Event(timestamp=ts, event=e.event))

    return kept_scenes, kept_events


def parse_describe(
    raw: str,
    duration: float | None = None,
) -> Tuple[str, List[Scene], List[Event]]:
    """
    Marlin emits one of two formats:
      (A) Native:  "Scene: <paragraph>\n\nEvents:\n<a: b> desc\n..."
      (B) Prompted [HH:MM:SS] inline timestamps, when the prompt is followed.

    We try (A) first since Marlin nearly always emits its native format;
    fall back to (B) for other model responses (e.g. TimeLens) or when
    Marlin actually obeys the prompt's [HH:MM:SS] instruction.

    If `duration` (video duration in seconds) is provided, parsed
    timestamps are validated and clamped via _clamp_to_duration — drops
    pure hallucinations and pins boundary-rounding cases to duration.

    Returns (summary, scenes, events).
    """
    # --- (A) Marlin native format ---
    scene_h = _MARLIN_SCENE_HEADER.search(raw)
    events_h = _MARLIN_EVENTS_HEADER.search(raw)
    if scene_h and events_h and events_h.start() > scene_h.start():
        summary = raw[scene_h.end(): events_h.start()].strip()
        events_block = raw[events_h.end():]
        events_a: List[Event] = []
        scenes_a: List[Scene] = []
        for line in events_block.splitlines():
            m = _MARLIN_RANGE_PAT.search(line)
            if not m:
                continue
            start, end = float(m.group(1)), float(m.group(2))
            desc = line[m.end():].strip(" -:.,;")
            events_a.append(Event(timestamp=start, event=desc))
            scenes_a.append(Scene(start=start, end=end, caption=desc))
        if scenes_a or events_a:
            if duration is not None:
                scenes_a, events_a = _clamp_to_duration(
                    scenes_a, events_a, duration
                )
            return summary, scenes_a, events_a

    # --- (B) Fallback: [HH:MM:SS]-style parser (original behaviour) ---
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    events: List[Event] = []
    summary_lines: List[str] = []
    scenes: List[Scene] = []

    last_ts: Optional[float] = None
    scene_start = 0.0
    scene_buf: List[str] = []

    for ln in lines:
        ts_match = _TIMESTAMP_PAT.search(ln)
        if ts_match:
            ts = _hms_to_seconds(ts_match)
            text = _TIMESTAMP_PAT.sub("", ln).strip(" -:.,;")
            events.append(Event(timestamp=ts, event=text))
            if last_ts is not None and scene_buf:
                scenes.append(
                    Scene(
                        start=scene_start,
                        end=ts,
                        caption=" ".join(scene_buf).strip(),
                    )
                )
                scene_buf = []
            scene_start = ts
            last_ts = ts
        else:
            scene_buf.append(ln)
            summary_lines.append(ln)

    # Flush trailing scene
    if scene_buf and last_ts is not None:
        scenes.append(
            Scene(
                start=scene_start,
                end=last_ts,  # we don't know the true end; use last ts
                caption=" ".join(scene_buf).strip(),
            )
        )

    summary = " ".join(summary_lines).strip()
    if not summary and scenes:
        summary = scenes[0].caption

    if duration is not None:
        scenes, events = _clamp_to_duration(scenes, events, duration)
    return summary, scenes, events


def parse_find(raw: str) -> List[Match]:
    """
    Expect lines like:
      [4.2, 5.1] dog jumps over the fence
      [12.8, 14.0] another jump

    Or 'NO_MATCHES' sentinel.
    """
    if "NO_MATCHES" in raw.upper():
        return []
    matches: List[Match] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rng = _RANGE_PAT.search(line)
        if not rng:
            continue
        start, end = float(rng.group(1)), float(rng.group(2))
        # Description = everything after the range
        desc = line[rng.end():].strip(" -:.,;")
        matches.append(Match(start=start, end=end, description=desc or None))
    return matches


def parse_bullets(raw: str) -> List[str]:
    """Extract bullet-style lines from a summary."""
    bullets: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match common bullet prefixes
        if line.startswith(("-", "*", "•")) or re.match(r"^\d+[\.\)]", line):
            bullets.append(re.sub(r"^[-*•\d\.\)]+\s*", "", line))
    # Fallback: split on newlines if no bullets detected
    if not bullets:
        bullets = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return bullets


# ---------- Summary synthesis (post-processing over /describe events) ----------
#
# Marlin has no native summarize mode (only caption + find). To honour
# PRD §2 / §5.2 we synthesize a real bullet summary by clustering the
# dense events from caption() output. No second model call — pure
# post-processing.


def summarise_from_events(
    events: List[Event],
    duration: float,
    target_bullets: int = 5,
) -> List[str]:
    """
    Build a compact bullet summary from Marlin's dense event list.

    Marlin emits very fine-grained events (often every 0.5-5s). For a
    /summarise response that's actually useful, we cluster events with
    identical normalized descriptions into "chapters" (so 9 occurrences
    of "camera focuses on theDAW's mixing panel" become one bullet that
    spans first→last occurrence), then rank by frequency to surface the
    most-recurring topics, finally re-sort chronologically.

    Returns up to `target_bullets` strings of the form
    "[M:SS - M:SS] description".
    """
    if not events:
        return []

    # Group events by normalized description.
    clusters: dict = {}
    for ev in events:
        key = _normalize_desc(ev.event)
        clusters.setdefault(key, []).append(ev)

    # Build chapter records.
    chapters = []
    for group in clusters.values():
        start = min(e.timestamp for e in group)
        end = max(e.timestamp for e in group)
        if end <= start:
            # Single occurrence — synthesize a small span so the bullet
            # has a visible range rather than [0:30 - 0:30].
            end = min(start + 5.0, duration)
        chapters.append({
            "start": start,
            "end": end,
            "count": len(group),
            "desc": group[0].event,
        })

    # Rank by frequency (most-recurring topics first), then take top N,
    # then re-sort chronologically so the summary reads in video order.
    chapters.sort(key=lambda c: c["count"], reverse=True)
    top = chapters[:target_bullets]
    top.sort(key=lambda c: c["start"])

    return [
        f"[{_fmt_time(c['start'])} - {_fmt_time(c['end'])}] {c['desc']}"
        for c in top
    ]


def _normalize_desc(s: str) -> str:
    """
    Normalize an event description for clustering: lowercase, strip
    leading articles and trailing punctuation. Keeps verb/noun structure
    intact so e.g. 'pans across' and 'focuses on' remain distinct topics.
    """
    s = s.lower().strip()
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            s = s[len(art):]
            break
    return s.rstrip(" .,;:")


def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS for human-readable summary bullets."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"
