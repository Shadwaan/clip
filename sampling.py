"""
Frame sampling for video models.

Strategy: adaptive FPS based on duration so we always end up with
between MIN_FRAMES and MAX_FRAMES sampled frames. This matches the
training distribution of Qwen3-VL family models (Marlin, TimeLens).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import av
import numpy as np
from PIL import Image

MIN_FRAMES = 16          # don't go lower; model needs temporal context
# 32 is the practical ceiling for Marlin-2B on a single A10G (24 GB VRAM).
# The model's theoretical context allows more, but visual-token cost is
# ~200-400 tokens per frame, so 64+ frames push inference past 1 min and
# can OOM on long-form clips. Bump this up only if you move to A100/H100.
MAX_FRAMES = 32
TARGET_RES = 448         # short-edge resize; Qwen3-VL native is 448 or 768


@dataclass
class SampledVideo:
    frames: List[Image.Image]   # PIL RGB frames in temporal order
    timestamps: List[float]     # seconds, aligned with `frames`
    duration: float             # full video duration in seconds
    fps_sampled: float          # effective sampling FPS used


def probe_duration(path: str | Path) -> float:
    """Return video duration in seconds."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        # Fallback: container duration is in microseconds
        return container.duration / av.time_base if container.duration else 0.0


def _resize_short_edge(img: Image.Image, target: int = TARGET_RES) -> Image.Image:
    w, h = img.size
    if min(w, h) <= target:
        return img
    if w < h:
        new_w = target
        new_h = int(h * target / w)
    else:
        new_h = target
        new_w = int(w * target / h)
    return img.resize((new_w, new_h), Image.BICUBIC)


def sample_video(
    path: str | Path,
    max_frames: int = MAX_FRAMES,
    min_frames: int = MIN_FRAMES,
    resize: bool = True,
) -> SampledVideo:
    """
    Uniformly sample frames from a video.

    For a duration D and target N frames, we sample at FPS = N/D, evenly spaced.
    Returns PIL RGB frames + their timestamps in seconds.
    """
    path = Path(path)
    duration = probe_duration(path)
    if duration <= 0:
        raise ValueError(f"Could not determine duration for {path}")

    # Adaptive frame budget: longer video → still cap at max_frames
    n_frames = min(max_frames, max(min_frames, int(duration * 1.0)))  # 1 fps baseline
    fps_sampled = n_frames / duration

    # Target timestamps, evenly spaced
    targets = np.linspace(0, duration, n_frames, endpoint=False) + (duration / n_frames) / 2
    target_set = list(targets)

    frames: List[Image.Image] = []
    timestamps: List[float] = []

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base

        next_target_idx = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if next_target_idx >= len(target_set):
                    break
                t = float(frame.pts * time_base) if frame.pts is not None else None
                if t is None:
                    continue
                # Take this frame if it has passed the next target timestamp
                if t >= target_set[next_target_idx]:
                    img = frame.to_image()  # PIL RGB
                    if resize:
                        img = _resize_short_edge(img)
                    frames.append(img)
                    timestamps.append(t)
                    next_target_idx += 1
            if next_target_idx >= len(target_set):
                break

    if len(frames) < min_frames:
        raise RuntimeError(
            f"Only extracted {len(frames)} frames from {path} (need {min_frames}). "
            "Video may be corrupt or shorter than expected."
        )

    return SampledVideo(
        frames=frames,
        timestamps=timestamps,
        duration=duration,
        fps_sampled=fps_sampled,
    )
