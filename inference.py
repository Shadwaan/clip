"""
Video VLM inference wrappers.

Two models, one interface:
  - Marlin-2B (default)  — fast, dense captioning + temporal grounding
  - TimeLens-8B          — SOTA temporal grounding, slower

Both are Qwen3-VL family, so we use the same Transformers loader pattern.

The Marlin quickstart pattern (AutoModelForCausalLM + AutoProcessor + chat
template with a "video" content type) is taken directly from the model card:
https://huggingface.co/NemoStation/Marlin-2B

Marlin emits a `<think>...</think>` prefix on every response (training
artifact). We strip it in `_strip_think`.

Backends
--------
Two execution backends, selected by env var CLIP_BACKEND:
  - "local" (default): load model into local GPU/CPU. Needs NVIDIA + CUDA.
  - "modal":           call the Modal deployment in modal_app.py. Use this
                       when you don't have an NVIDIA GPU locally.

The frame-sampling step (sampling.py) always runs locally regardless of
backend — it's CPU-only ffmpeg work, no GPU needed. Only the model forward
pass moves to Modal.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from sampling import SampledVideo, sample_video
from constants import ModelChoice, MODEL_REPOS

log = logging.getLogger(__name__)

# Pick backend at import time. Heavy deps (torch, transformers) are loaded
# lazily inside the local-backend class so importing this module on a
# machine without CUDA / PyTorch installed doesn't crash.
BACKEND = os.environ.get("CLIP_BACKEND", "local").lower()
if BACKEND not in ("local", "modal"):
    raise ValueError(f"CLIP_BACKEND must be 'local' or 'modal', got {BACKEND!r}")

# Singletons — one model per process, lazily loaded
_loaded: dict[ModelChoice, "VideoVLM | ModalVLM"] = {}
_load_lock = threading.Lock()


@dataclass
class InferenceResult:
    text: str               # cleaned model output
    raw: str                # raw output incl. <think> tags
    model: ModelChoice
    sampled: SampledVideo   # what we actually fed the model


def _strip_think(text: str) -> str:
    """Marlin emits <think>...</think> prefix. Remove it."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ============================================================
# Local backend — runs the model on this machine (needs CUDA)
# ============================================================

class VideoVLM:
    """One instance = one loaded model on one local device. Local backend only."""

    def __init__(
        self,
        choice: ModelChoice,
        device: str = "cuda",
        dtype=None,           # torch.dtype; resolved inside to avoid top-level import
    ):
        # Heavy imports live inside the class so the Modal backend never
        # needs torch / transformers installed locally.
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.choice = choice
        self.device = device
        self.dtype = dtype or torch.bfloat16
        repo = MODEL_REPOS[choice]

        log.info(f"Loading {repo} on {device} ({self.dtype})...")
        # `trust_remote_code=True` is required by Marlin (custom modeling code)
        # and by Qwen3-VL family in general.
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            dtype=self.dtype,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            repo,
            trust_remote_code=True,
        )
        log.info(f"Loaded {repo}.")

    def run(
        self,
        video_path: str | Path,
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> InferenceResult:
        import torch

        sampled = sample_video(video_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sampled.frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        cleaned = _strip_think(raw)

        return InferenceResult(
            text=cleaned,
            raw=raw,
            model=self.choice,
            sampled=sampled,
        )

    def caption(self, video_path: str | Path) -> InferenceResult:
        """
        Marlin's native caption pathway. Bypasses our local sampling step —
        Marlin's custom modeling code does its own decoding + sampling +
        timestamp tracking. Output text has correct real-world seconds.
        See modal_app.py:caption for the full rationale.

        We still call sample_video() to populate InferenceResult.sampled
        for API-level metadata (frame count, duration), but those frames
        are not used by the model.
        """
        video_path = Path(video_path)
        sampled = sample_video(video_path)  # metadata only
        result = self.model.caption(str(video_path))
        raw = result["caption"]
        cleaned = _strip_think(raw)
        return InferenceResult(
            text=cleaned,
            raw=raw,
            model=self.choice,
            sampled=sampled,
        )

    def find(self, video_path: str | Path, query: str) -> dict:
        """
        Marlin's native find pathway (local backend). Calls
        self.model.find(path, event=query); Marlin returns a single
        best-match span. See modal_app.py:find for the rationale.

        Returns a dict (not InferenceResult) because the find shape is
        different — there's a parsed span rather than free text. Keys:
        raw, span, format_ok, model, sampled.
        """
        video_path = Path(video_path)
        sampled = sample_video(video_path)  # metadata only
        result = self.model.find(str(video_path), event=query)
        return {
            "raw": result["raw"],
            "span": result.get("span"),
            "format_ok": result.get("format_ok", False),
            "model": self.choice,
            "sampled": sampled,
        }


# ============================================================
# Modal backend — runs the model on a rented GPU via Modal
# ============================================================

class ModalVLM:
    """
    Mirror of VideoVLM's interface, but offloads the model forward pass to a
    Modal deployment (see modal_app.py).

    Sampling still happens locally (CPU, ffmpeg). Only the GPU work runs on
    Modal. We pass the sampled PIL frames over the wire as pickled Python
    objects — Modal handles that transparently.
    """

    # Map ModelChoice → (app name, class name) on Modal. Both classes live
    # in the same app (clip-marlin) for ops simplicity; Modal scales them
    # independently regardless. TimeLens isn't deployed today; if asked we
    # raise NotImplementedError rather than guessing a class name.
    _MODAL_CLASSES = {
        ModelChoice.MARLIN_2B: ("clip-marlin", "MarlinModel"),
        ModelChoice.QWEN3_VL_8B: ("clip-marlin", "QwenVL"),
    }

    def __init__(self, choice: ModelChoice):
        if choice not in self._MODAL_CLASSES:
            # TimeLens would need a third class — not deployed in v0.1.
            raise NotImplementedError(
                f"Modal backend has no deployed class for {choice.value}. "
                f"Available: {[c.value for c in self._MODAL_CLASSES]}"
            )
        self.choice = choice
        app_name, cls_name = self._MODAL_CLASSES[choice]

        # Look up the deployed Modal class. This is cheap — it doesn't spin
        # up a container; that happens on the first .remote() call.
        import modal
        try:
            self._remote_cls = modal.Cls.from_name(app_name, cls_name)
        except Exception as e:
            raise RuntimeError(
                f"Could not find deployed Modal class {app_name}/{cls_name}. "
                f"Run `modal deploy modal_app.py` first."
            ) from e

        # An instance to call methods on. Modal manages container lifecycle.
        self._remote = self._remote_cls()
        log.info(f"Modal backend ready ({app_name} / {cls_name})")

    # The methods below dispatch to MarlinModel.* on the remote side. They
    # only make sense when this ModalVLM is bound to MARLIN_2B. The QwenVL
    # remote class defines only `ask`, not `generate`/`caption`/`find`, so
    # we guard at the Python boundary rather than letting Modal raise a
    # "method not found" deep inside the wire call.
    def _require_marlin(self, method_name: str) -> None:
        if self.choice != ModelChoice.MARLIN_2B:
            raise NotImplementedError(
                f"ModalVLM.{method_name}() requires the Marlin backend; "
                f"this instance is bound to {self.choice.value}. Use the "
                f"matching endpoint for that model instead (e.g. /ask for "
                f"Qwen3-VL)."
            )

    def run(
        self,
        video_path: str | Path,
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> InferenceResult:
        self._require_marlin("run")
        # Sample frames locally — no GPU needed.
        sampled = sample_video(video_path)

        # Ship frames + prompt to the Modal container, get raw text back.
        log.info(f"Calling Modal generate ({len(sampled.frames)} frames)...")
        result = self._remote.generate.remote(
            frames=sampled.frames,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        raw = result["raw"]
        cleaned = _strip_think(raw)

        return InferenceResult(
            text=cleaned,
            raw=raw,
            model=self.choice,
            sampled=sampled,
        )

    def caption(self, video_path: str | Path) -> InferenceResult:
        """
        Marlin native caption pathway via Modal. Ships raw video bytes to
        the container (which writes them to a tempfile for Marlin's
        internal torchcodec decoder). Bigger payload than pre-sampled
        frames, but Marlin handles its own decoding + sampling + timestamp
        tracking, so output has correct real-world seconds.
        See modal_app.py:caption for the full rationale.
        """
        self._require_marlin("caption")
        video_path = Path(video_path)
        video_bytes = video_path.read_bytes()
        video_ext = video_path.suffix.lstrip(".") or "mp4"
        sampled = sample_video(video_path)  # metadata only

        log.info(
            f"Calling Modal caption ({len(video_bytes):,} bytes, "
            f"ext={video_ext}, duration={sampled.duration:.1f}s)..."
        )
        result = self._remote.caption.remote(
            video_bytes=video_bytes,
            video_ext=video_ext,
        )
        raw = result["raw"]
        cleaned = _strip_think(raw)

        return InferenceResult(
            text=cleaned,
            raw=raw,
            model=self.choice,
            sampled=sampled,
        )

    def find(self, video_path: str | Path, query: str) -> dict:
        """
        Marlin native find pathway via Modal. Ships video bytes to the
        container; Marlin's torchcodec decoder + trained find-mode handle
        the rest. Returns dict with raw text, parsed span (or None on
        parse failure), and format_ok flag.
        See modal_app.py:find for the full rationale.
        """
        self._require_marlin("find")
        video_path = Path(video_path)
        video_bytes = video_path.read_bytes()
        video_ext = video_path.suffix.lstrip(".") or "mp4"
        sampled = sample_video(video_path)  # metadata only

        log.info(
            f"Calling Modal find (query={query!r}, "
            f"{len(video_bytes):,} bytes, ext={video_ext})..."
        )
        result = self._remote.find.remote(
            video_bytes=video_bytes,
            event=query,
            video_ext=video_ext,
        )
        return {
            "raw": result["raw"],
            "span": result.get("span"),
            "format_ok": result.get("format_ok", False),
            "model": self.choice,
            "sampled": sampled,
        }

    # ============================================================
    # Spawn variants — Modal-native job queue (Milestone 1).
    # ============================================================
    #
    # Each `.remote()` call above has a `.spawn()` companion below. The
    # difference: `.remote()` blocks until the container returns a result;
    # `.spawn()` enqueues the call and returns a `modal.FunctionCall`
    # immediately. The caller stashes `fn_call.object_id` (a string) and
    # later reconstructs the call via `modal.FunctionCall.from_id(...)`
    # to poll for the result via `.get(timeout=0)`.
    #
    # This is Modal's native job-queue pattern (modal.com/docs/guide/job-queue).
    # We use it to fulfil PRD §5.2's "async if >60s" requirement without
    # standing up a separate Celery worker process.
    #
    # The submit_* methods do the exact same payload prep as their
    # synchronous counterparts; only the final dispatch line changes
    # (`.spawn(...)` instead of `.remote(...)`). Sync methods are kept
    # for testing and the rare case where a caller wants to block.

    def submit_caption(self, video_path: str | Path) -> str:
        """Marlin native caption pathway, spawned. Returns FunctionCall.object_id."""
        self._require_marlin("submit_caption")
        video_path = Path(video_path)
        video_bytes = video_path.read_bytes()
        video_ext = video_path.suffix.lstrip(".") or "mp4"
        log.info(f"Spawning Modal caption ({len(video_bytes):,} bytes, ext={video_ext})...")
        fn_call = self._remote.caption.spawn(
            video_bytes=video_bytes,
            video_ext=video_ext,
        )
        return fn_call.object_id

    def submit_find(self, video_path: str | Path, query: str) -> str:
        """Marlin native find pathway, spawned. Returns FunctionCall.object_id."""
        self._require_marlin("submit_find")
        video_path = Path(video_path)
        video_bytes = video_path.read_bytes()
        video_ext = video_path.suffix.lstrip(".") or "mp4"
        log.info(f"Spawning Modal find (query={query!r}, {len(video_bytes):,} bytes)...")
        fn_call = self._remote.find.spawn(
            video_bytes=video_bytes,
            event=query,
            video_ext=video_ext,
        )
        return fn_call.object_id

    def submit_run(
        self,
        video_path: str | Path,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Marlin prompted-generate pathway, spawned. Returns
        FunctionCall.object_id.

        Sampling happens inside the Modal container's `generate` method —
        wait, no: in the sync `.run()` path, we sample locally and ship
        frames. For spawn we do the same to keep behaviour identical, even
        though for very long videos this means the CPU sampling step still
        blocks the FastAPI process briefly. Acceptable for v0.1 — frame
        sampling is fast (a few seconds at most) compared to the inference
        itself.
        """
        self._require_marlin("submit_run")
        sampled = sample_video(video_path)
        log.info(f"Spawning Modal generate ({len(sampled.frames)} frames)...")
        fn_call = self._remote.generate.spawn(
            frames=sampled.frames,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        return fn_call.object_id

    def submit_ask(self, video_path: str | Path, question: str) -> str:
        """
        Qwen3-VL ask pathway, spawned. Returns FunctionCall.object_id.
        Only valid on a ModalVLM bound to QWEN3_VL_8B — same guard as
        the sync ask() method.
        """
        if self.choice != ModelChoice.QWEN3_VL_8B:
            raise NotImplementedError(
                f"submit_ask() is only implemented for the Qwen backend; "
                f"this ModalVLM is bound to {self.choice.value}."
            )
        video_path = Path(video_path)
        video_bytes = video_path.read_bytes()
        video_ext = video_path.suffix.lstrip(".") or "mp4"
        log.info(f"Spawning Modal QwenVL.ask ({len(video_bytes):,} bytes)...")
        fn_call = self._remote.ask.spawn(
            video_bytes=video_bytes,
            question=question,
            video_ext=video_ext,
        )
        return fn_call.object_id

    def ask(self, video_path: str | Path, question: str) -> InferenceResult:
        """
        Open-ended Q&A pathway via Modal's QwenVL class (Qwen3-VL-8B-
        Instruct). PRD §5.4 routes /ask to this model — Marlin is a
        captioner and can't actually answer questions, so this is the
        only real route for that endpoint.

        Only valid when self.choice == ModelChoice.QWEN3_VL_8B; main.py
        is responsible for routing /ask to a ModalVLM bound to that
        choice. Wrong-binding calls raise rather than silently issuing
        an ask().remote on a class (MarlinModel) that doesn't define it.

        Same payload pattern as caption() / find(): ship raw video bytes
        + question, container writes a tempfile and lets Qwen3-VL's chat
        template handle decoding and frame sampling.
        """
        if self.choice != ModelChoice.QWEN3_VL_8B:
            raise NotImplementedError(
                f"ask() is only implemented for the Qwen backend; this "
                f"ModalVLM is bound to {self.choice.value}."
            )

        video_path = Path(video_path)
        video_bytes = video_path.read_bytes()
        video_ext = video_path.suffix.lstrip(".") or "mp4"
        sampled = sample_video(video_path)  # metadata only

        log.info(
            f"Calling Modal QwenVL.ask ({len(video_bytes):,} bytes, "
            f"ext={video_ext}, duration={sampled.duration:.1f}s)..."
        )
        result = self._remote.ask.remote(
            video_bytes=video_bytes,
            question=question,
            video_ext=video_ext,
        )
        raw = result["raw"]
        cleaned = _strip_think(raw)
        return InferenceResult(
            text=cleaned,
            raw=raw,
            model=self.choice,
            sampled=sampled,
        )


# ============================================================
# Backend selector
# ============================================================

def get_model(choice: ModelChoice):
    """Lazy, thread-safe singleton accessor. Returns VideoVLM or ModalVLM."""
    if choice in _loaded:
        return _loaded[choice]
    with _load_lock:
        if choice in _loaded:
            return _loaded[choice]

        if BACKEND == "modal":
            _loaded[choice] = ModalVLM(choice)
        else:
            import torch
            device = os.environ.get("CLIP_DEVICE", "cuda")
            if device == "cuda" and not torch.cuda.is_available():
                log.warning("CUDA requested but not available; falling back to CPU (slow!)")
                device = "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            _loaded[choice] = VideoVLM(choice, device=device, dtype=dtype)

    return _loaded[choice]


# ---------- Task-specific prompts ----------
#
# These are tuned for Marlin's output style (scene/event tags with timestamps).
# TimeLens tends to be terser; same prompts work, parse layer handles both.

DESCRIBE_PROMPT = (
    "Describe this video in detail. Break it down into scenes and key events "
    "with timestamps in [HH:MM:SS] format. Be concrete and avoid speculation."
)

SUMMARISE_PROMPT = (
    "Summarise this video in 3-5 bullet points covering the most important "
    "moments. Include timestamps for each bullet in [HH:MM:SS] format."
)

FIND_PROMPT_TEMPLATE = (
    'In this video, find all moments matching: "{query}". '
    "For each match, output a line in the format: "
    "[start_seconds, end_seconds] brief_description. "
    "If there are no matches, respond exactly: NO_MATCHES."
)

ASK_PROMPT_TEMPLATE = (
    "{question}\n\nAnswer based on what you see in the video. "
    "Reference timestamps in [HH:MM:SS] when relevant."
)
