"""
Modal deployment for Clip's Marlin-2B inference.

Why this file exists
--------------------
Your local AMD GPU can't run Marlin-2B (CUDA-only, plus only 6 GB VRAM).
This Modal app hosts Marlin on a rented NVIDIA A10G and exposes it as
something the local FastAPI gateway (main.py) can call over the network.

How it works
------------
- `MarlinModel` is a Modal class. Modal spins up a container with a GPU
  when needed, loads the model weights once (via `@modal.enter`), and
  keeps it warm for `scaledown_window` seconds after each request.
- `generate(frames, prompt)` is the remote method. Local code calls it
  with `MarlinModel().generate.remote(frames, prompt)`.
- Containers scale to zero when idle, so you only pay for actual
  inference seconds (~$0.60/hr while running an A10G).

Setup
-----
1. `pip install modal`
2. `modal setup` (one-time browser auth)

Deploy
------
    modal deploy modal_app.py

Smoke test (after deploy):
    curl https://<your-workspace>--clip-marlin-healthz.modal.run

First call downloads the model weights into a Modal Volume (~5 GB).
Subsequent cold starts reuse those weights, so they're much faster.
"""

import modal

# Shared helper: writes video_bytes OR a downloaded video_url to a tempfile.
# Added to each image below via .add_local_python_source so it's importable in
# the container. The bytes path is unchanged; video_url is additive.
from video_source import _materialize_video

APP_NAME = "clip-marlin"
MODEL_REPO = "NemoStation/Marlin-2B"

# Marlin's native caption() has a ~130s temporal horizon: its internal frame
# sampler caps total frames, so for any video longer than ~130s it only "sees"
# the first ~130s, then degenerates into a repetition loop with wrapped
# timestamps (diagnosed on a 907.3s webm — job 07e1a7fc…, where timestamps
# climbed to 129.5s then reset to 30.0 and looped). We stay safely under that
# horizon by captioning the video in <=CHUNK_SEC windows and re-offsetting each
# window's chunk-relative timestamps by its start.
#
# Accuracy note: chunk boundaries are cut with ffmpeg stream-copy (-c copy),
# which can only cut on keyframes. Each chunk is offset by its nominal
# i*CHUNK_SEC source position, and caption() drops each chunk's overshoot
# events (relative start >= CHUNK_SEC) so adjacent stream-copy segments — which
# can run a few seconds past their nominal window — don't overlap. The result
# is a strictly monotonic timeline bounded by the real video duration.
# Absolute timestamps can still be off by up to ±keyframe-interval (typically
# 2-10s) where a segment's first keyframe precedes its nominal start. Frame-
# accurate cuts would require re-encoding each segment (much slower); the
# ±keyframe drift is acceptable for captioning.
CHUNK_SEC = 120
QWEN_REPO = "Qwen/Qwen3-VL-8B-Instruct"   # Open-ended reasoning model (PRD §5.4 → /ask)
GPU_TYPE = "A10G"      # 24 GB; comfortable for Marlin-2B (~5 GB needed)
QWEN_GPU = "A10G"      # 24 GB; tight for Qwen3-VL-8B-Instruct in bf16 (~18 GB weights).
                       # If OOM in testing, bump to "A100-40GB" — model card gives no
                       # explicit A10G guidance, so we run there to mirror Marlin and
                       # cap video tokens with num_frames=32 + total_pixels.
CACHE_DIR = "/cache"   # where HF caches downloaded weights inside the container

# --- Container image ---
# We bake the Python deps into the image at build time so cold starts don't
# pip-install every time. ffmpeg is included because PyAV (used by sampling.py)
# links against it at runtime in some paths.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.4.0",
        # Marlin-2B is built on Qwen3-VL. `AutoProcessor.from_pretrained`
        # eagerly loads a Qwen3VLVideoProcessor sub-processor, which
        # hard-requires torchvision even when we only feed it stills.
        # Without this line, container load crashes at processor init.
        "torchvision>=0.19.0",
        # Marlin's native .caption() method (canonical inference pathway
        # per the model card) decodes video files via torchcodec. Required
        # for the caption-mode path that gives us correct timestamps.
        "torchcodec",
        "transformers>=4.46.0",
        "accelerate>=1.0.0",
        "qwen-vl-utils>=0.0.10",
        "pillow>=10.4.0",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26.0",
        # Required by `@modal.fastapi_endpoint` (healthz). Modal used to
        # install FastAPI automatically but no longer does.
        "fastapi[standard]>=0.115.0",
    )
    # Ship the shared video-input helper so the container can import it.
    .add_local_python_source("video_source")
)

app = modal.App(APP_NAME, image=image)

# --- Container image for QwenVL (Qwen3-VL-8B-Instruct) ---
#
# Why a separate image (rather than reusing the Marlin image):
#   - Qwen3-VL is supported natively in transformers >= 4.57.0 (no
#     trust_remote_code needed). Marlin uses trust_remote_code=True with
#     custom modeling code from NemoStation/Marlin-2B's repo, and that code
#     was tested against the older 4.46.x line. Bumping Marlin to 4.57.x
#     would change its dependency surface; we don't want to regress Marlin
#     just to add Qwen.
#   - Keeping per-class images means each class has the minimal deps it
#     needs and we can iterate on one without redeploying the other from
#     scratch.
# Both classes still share `model_cache` (the volume) so weights download
# once per repo across both image rebuilds.
qwen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.4.0",
        # Qwen3-VL's video processor uses torchvision for decoding; same
        # eager-loading issue as Marlin's processor.
        "torchvision>=0.19.0",
        # torchcodec is the recommended Qwen3-VL video backend on
        # transformers >= 4.57.0.
        "torchcodec",
        # 4.57.0 is the floor for native Qwen3-VL support per the model's
        # GitHub README (https://github.com/QwenLM/Qwen3-VL).
        "transformers>=4.57.0",
        "accelerate>=1.0.0",
        # Optional — Qwen3-VL supports passing file:// paths directly to
        # apply_chat_template, so we don't strictly need qwen-vl-utils.
        # Including it pinned to the version the README calls out, in case
        # we later move to the process_vision_info path for fine-grained
        # pixel control on long videos.
        "qwen-vl-utils==0.0.14",
        "pillow>=10.4.0",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26.0",
        "fastapi[standard]>=0.115.0",
    )
    # Ship the shared video-input helper so the container can import it.
    .add_local_python_source("video_source")
)

# Persistent volume for model weights. First run pulls ~5 GB; subsequent
# cold starts mount the volume and skip the download.
model_cache = modal.Volume.from_name("clip-model-cache", create_if_missing=True)


@app.cls(
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: model_cache},
    # Marlin-2B is a gated repo on HuggingFace — the container needs an HF
    # token to download weights. Create the secret once:
    #     modal secret create huggingface-secret HF_TOKEN=hf_xxx
    secrets=[modal.Secret.from_name("huggingface-secret")],
    # Long videos are captioned in <=CHUNK_SEC chunks (see caption()), so a
    # single request can make N sequential model.caption() calls. Budget for
    # the worst case: a ~15 min video is ~8 chunks. 1800s gives headroom over
    # the old 600s single-call ceiling.
    timeout=1800,           # max seconds a single inference call can take
    scaledown_window=120,   # keep container warm for 2 minutes after last request
    # Don't retry .enter failures forever — surface them after one attempt.
    retries=0,
)
class MarlinModel:
    """Loaded once per container; reused for every request that container handles."""

    @modal.enter()
    def load(self):
        """Run on container startup. Pulls weights into the volume on first cold start."""
        import os
        # Point HuggingFace at the persistent volume so we don't re-download.
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
        os.environ["HF_HUB_CACHE"] = CACHE_DIR

        # Verify the gated-repo token actually made it into the container.
        # If this fails, the Modal secret isn't attached or was named wrong.
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN env var not set in the container. The Modal secret "
                "'huggingface-secret' is missing or not attached. Run "
                "`modal secret list` to verify, then re-deploy."
            )
        print(f"[modal_app] HF_TOKEN present (length={len(hf_token)}). Authenticating...")
        from huggingface_hub import login, whoami, HfApi
        login(token=hf_token, add_to_git_credential=False)

        # Verify the token works at all and that this account has access to
        # the gated repo. Either of these will throw a clear, specific error
        # if something's wrong — far better than the generic OSError that
        # transformers raises later.
        try:
            me = whoami(token=hf_token)
            print(f"[modal_app] HF user: {me.get('name')} ({me.get('email', 'no email')})")
        except Exception as e:
            raise RuntimeError(f"HF token is invalid or expired: {e}") from e

        try:
            HfApi().model_info(MODEL_REPO, token=hf_token)
            print(f"[modal_app] Access to {MODEL_REPO} confirmed.")
        except Exception as e:
            raise RuntimeError(
                f"Account does NOT have access to {MODEL_REPO}. Visit "
                f"https://huggingface.co/{MODEL_REPO} in a browser, accept "
                f"the terms, then retry. Original error: {e}"
            ) from e

        # Pre-fetch config.json directly. This isolates "can we download files
        # from this gated repo at all?" from any transformers-internal weirdness
        # with how it forwards the auth token.
        from huggingface_hub import hf_hub_download
        try:
            cfg_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="config.json",
                token=hf_token,
                cache_dir=CACHE_DIR,
            )
            print(f"[modal_app] Pre-fetched config.json → {cfg_path}")
        except Exception as e:
            raise RuntimeError(
                f"Direct download of config.json failed: {e}. "
                f"This is below transformers — points at huggingface_hub or network."
            ) from e

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"[modal_app] Loading {MODEL_REPO} on {GPU_TYPE}...")
        # Pass token= explicitly. In some transformers versions the global
        # login state isn't always picked up by from_pretrained's downloader.
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
            low_cpu_mem_usage=True,
            token=hf_token,
            cache_dir=CACHE_DIR,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            MODEL_REPO,
            trust_remote_code=True,
            token=hf_token,
            cache_dir=CACHE_DIR,
        )
        # Persist any new weights pulled this session.
        model_cache.commit()
        print("[modal_app] Model loaded and ready.")

    @modal.method()
    def caption(
        self,
        video_bytes: bytes | None = None,
        video_ext: str = "mp4",
        max_new_tokens: int = 2048,
        *,
        video_url: str | None = None,
    ) -> dict:
        """
        Run Marlin's native .caption() method on a video file.

        Accepts the video as `video_bytes` (existing path, unchanged) OR a new
        keyword-only `video_url` that is stream-downloaded in the container.
        Provide exactly one. See video_source._materialize_video.

        This is the canonical inference pathway per the Marlin model card:
        Marlin's custom modeling code (loaded via trust_remote_code=True)
        handles video decoding (torchcodec), frame sampling, AND timestamp
        tracking internally. Output already contains real-world seconds in
        its `Scene: ... Events: <a-b> desc` text.

        Distinct from generate() (raw apply_chat_template path with
        pre-sampled PIL frames) which is documented as "advanced — raw
        inference" and is the source of the timestamp compression bug
        we hit during v0.1 testing.
        """
        import math
        import os
        import re
        import subprocess
        import tempfile

        # ---- helpers (local to keep the Modal method self-contained) ----
        tag_re = re.compile(r"<\s*(\d+(?:\.\d+)?)\s*[-–:,]\s*(\d+(?:\.\d+)?)\s*>")

        def probe_duration(path: str) -> float:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, check=True,
            )
            return float(out.stdout.strip())

        def split_native(text: str):
            """Return (scene_paragraph, events_block) from Marlin's native output."""
            ev = re.search(r"^\s*Events:\s*", text, re.IGNORECASE | re.MULTILINE)
            sc = re.search(r"^\s*Scene:\s*", text, re.IGNORECASE | re.MULTILINE)
            if ev and sc and ev.start() > sc.start():
                return text[sc.end():ev.start()].strip(), text[ev.end():]
            return "", text  # no recognizable header — treat all as events

        def offset_and_filter(events_block: str, offset: float, max_rel: float) -> str:
            """Offset each chunk-relative <a-b> tag by `offset`, but first DROP
            any tag whose relative start is >= max_rel. Stream-copy segments can
            run a few seconds past their nominal CHUNK_SEC window; those overshoot
            events are re-captioned by the next chunk, so dropping them here (and
            keeping the nominal i*CHUNK_SEC offset) yields a strictly monotonic,
            non-inflated timeline. Lines without a tag are dropped (the gateway
            parser ignores them anyway)."""
            out_lines = []
            for line in events_block.splitlines():
                m = tag_re.search(line)
                if not m:
                    continue
                rel_start = float(m.group(1))
                if rel_start >= max_rel:
                    continue  # overlap into the next chunk's window — skip
                new_tag = (f"<{rel_start + offset:.1f} - "
                           f"{float(m.group(2)) + offset:.1f}>")
                out_lines.append(tag_re.sub(new_tag, line, count=1))
            return "\n".join(out_lines)

        # Marlin's internal decoder needs a real file on disk. Materialize the
        # video (from bytes or a downloaded URL) to a tempfile, then clean up
        # after the call regardless of success/failure. The chunking path below
        # still uses tempfile for its per-window segments.
        video_path = _materialize_video(
            video_bytes=video_bytes, video_url=video_url, video_ext=video_ext
        )

        segments: list[str] = []
        try:
            duration = probe_duration(video_path)

            # Short enough to caption in one pass — original behaviour.
            if duration <= CHUNK_SEC + 10:
                result = self.model.caption(
                    video_path, max_new_tokens=max_new_tokens
                )
                # Marlin's caption() returns a dict with "caption" (raw text),
                # "scene" (parsed paragraph), and "events" (list of parsed
                # dicts). We return just the raw text and let the gateway's
                # routing.parse_describe handle structured extraction — keeps
                # parsing consistent across model backends.
                return {"raw": result["caption"]}

            # Long video: split into <=CHUNK_SEC windows, caption each, offset
            # its chunk-relative timestamps by the window start, and stitch
            # back into one native-format string so routing.parse_describe is
            # unchanged. See the CHUNK_SEC comment for the ±keyframe drift the
            # stream-copy split introduces.
            n_chunks = math.ceil(duration / CHUNK_SEC)
            print(f"[modal_app] caption: duration={duration:.1f}s > {CHUNK_SEC}s "
                  f"horizon; chunking into {n_chunks} segments")

            scene_parts: list[str] = []
            event_blocks: list[str] = []
            # Each chunk is offset by its nominal i*CHUNK_SEC source position
            # (NOT cumulative probed durations, which double-count overlapping
            # stream-copy segments and inflate the timeline past the real
            # duration). offset_and_filter drops each chunk's overshoot events
            # (relative start >= CHUNK_SEC) so adjacent chunks don't overlap and
            # the stitched timeline stays strictly monotonic and bounded.
            for i in range(n_chunks):
                start = i * CHUNK_SEC
                seg = tempfile.NamedTemporaryFile(suffix=f".{video_ext}", delete=False)
                seg.close()
                segments.append(seg.name)
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(start), "-t", str(CHUNK_SEC),
                     "-i", video_path, "-c", "copy", "-reset_timestamps", "1",
                     seg.name],
                    capture_output=True, check=True,
                )
                chunk_raw = self.model.caption(
                    seg.name, max_new_tokens=max_new_tokens
                )["caption"]
                scene, events = split_native(chunk_raw)
                if scene:
                    scene_parts.append(scene)
                offset_block = offset_and_filter(
                    events, float(start), float(CHUNK_SEC)
                ).strip()
                if offset_block:
                    event_blocks.append(offset_block)

            combined = (
                "Scene: " + " ".join(scene_parts).strip()
                + "\n\nEvents:\n" + "\n".join(event_blocks)
            )
            return {"raw": combined}
        finally:
            os.unlink(video_path)
            for s in segments:
                try:
                    os.unlink(s)
                except OSError:
                    pass

    @modal.method()
    def find(
        self,
        video_bytes: bytes | None = None,
        event: str | None = None,
        video_ext: str = "mp4",
        *,
        video_url: str | None = None,
    ) -> dict:
        """
        Run Marlin's native .find() method to temporally ground a
        natural-language event query inside the video.

        Accepts `video_bytes` (existing path, unchanged) OR keyword-only
        `video_url` (downloaded in-container). `event` keeps its positional
        slot but now defaults to None so video_bytes can default to None too;
        it is still required and validated at runtime.

        Like caption(), this is the canonical pathway per the Marlin
        model card. Marlin has a separately-trained "find" mode that
        emits `From X.X to Y.Y.` and the custom modeling code parses it
        into a `(start, end)` span tuple. Going through raw generate()
        with our own FIND_PROMPT_TEMPLATE would be reinventing that mode
        from the outside — same lesson as caption().

        Returns {"raw": str, "span": (s, e) | None, "format_ok": bool}.
        """
        import os

        if event is None:
            raise ValueError("find() requires an 'event' query string.")
        video_path = _materialize_video(
            video_bytes=video_bytes, video_url=video_url, video_ext=video_ext
        )

        try:
            result = self.model.find(video_path, event=event)
            return {
                "raw": result["raw"],
                "span": result.get("span"),
                "format_ok": result.get("format_ok", False),
            }
        finally:
            os.unlink(video_path)

    @modal.method()
    def generate(
        self,
        frames: list,            # list of PIL.Image.Image (Modal pickles these)
        prompt: str,
        max_new_tokens: int = 512,
    ) -> dict:
        """
        Run Marlin via the raw apply_chat_template pathway on pre-sampled
        PIL frames + a custom prompt. Returns {'raw': str}.

        Used by /summarise and /ask — endpoints whose prompts Marlin
        wasn't natively trained on. /describe and /find go through their
        respective native methods (caption() and find() above), which
        have correct timestamps; this generate() path produces compressed
        timestamps because the chat template doesn't get source-fps info.
        """
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
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
        ).to("cuda")

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return {"raw": raw}


# ============================================================
# QwenVL — Qwen3-VL-8B-Instruct for open-ended video Q&A
# ============================================================
#
# Why this exists
# ---------------
# Marlin is a captioning + temporal-grounding model. It literally cannot
# answer open-ended questions like "why is the person doing X?" — it can
# only describe what it sees. PRD §5.4 specifies Qwen3-VL-8B-Instruct as
# the model for /ask. This class delivers that.
#
# Why not extend MarlinModel
# --------------------------
# Different model family, different transformers version floor (4.57.0 for
# native Qwen3-VL support vs. 4.46.0 for Marlin's trust_remote_code path),
# different processor. Cleaner to keep them as separate classes that Modal
# scales independently.
#
# What the API looks like
# -----------------------
# @modal.method() def ask(video_bytes, question, video_ext) -> {"raw": str}
# Same shape as MarlinModel.caption()/find(): ship raw bytes to the
# container, write to a tempfile, point the chat template at the local
# file (file:// prefix as the Qwen3-VL README documents), get back text.
@app.cls(
    image=qwen_image,
    gpu=QWEN_GPU,
    volumes={CACHE_DIR: model_cache},
    # Qwen3-VL is Apache 2.0 (not gated), so the HF secret isn't strictly
    # required for downloads. We attach it anyway — keeps the auth posture
    # identical across both classes, and lets us hit HF rate limits less.
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    scaledown_window=120,
    retries=0,
)
class QwenVL:
    """Loaded once per container; reused for every /ask request."""

    @modal.enter()
    def load(self):
        import os
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
        os.environ["HF_HUB_CACHE"] = CACHE_DIR

        # Auth is optional for Apache-2.0 Qwen3-VL, but logging in if the
        # token is present (we attached the secret) avoids public-anon HF
        # rate limits during model download.
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print(f"[modal_app/qwen] Authenticated with HF (token length={len(hf_token)}).")
        else:
            print("[modal_app/qwen] No HF_TOKEN attached; using anonymous access.")

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[modal_app/qwen] Loading {QWEN_REPO} on {QWEN_GPU}...")
        # Per the Qwen3-VL README: AutoModelForImageTextToText is the
        # canonical loader (transformers >= 4.57.0). No trust_remote_code
        # — Qwen3-VL is natively supported.
        # We skip attn_implementation="flash_attention_2" for v0.1 since
        # building flash-attn adds nvcc to the image; the README marks it
        # as a recommended-not-required optimisation.
        self.model = AutoModelForImageTextToText.from_pretrained(
            QWEN_REPO,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            QWEN_REPO,
            cache_dir=CACHE_DIR,
        )
        model_cache.commit()
        print("[modal_app/qwen] Model loaded and ready.")

    @modal.method()
    def ask(
        self,
        video_bytes: bytes | None = None,
        question: str | None = None,
        video_ext: str = "mp4",
        max_new_tokens: int = 512,
        # num_frames caps the video-token cost. 32 keeps us comfortably
        # within A10G memory for short-to-medium clips; the model card
        # warns to mind GPU memory budget on long video. fps=None makes
        # num_frames authoritative (per the README "Pixel Control" example).
        num_frames: int = 32,
        *,
        video_url: str | None = None,
    ) -> dict:
        """
        Run open-ended Q&A against the video.

        Uses the simple apply_chat_template path documented in the
        Qwen3-VL README (sections "Video inference" and "Pixel Control via
        Official Processor"). We write the incoming video bytes to a
        tempfile and reference it with a `file:///...` URI in the message
        content — Qwen3-VL's processor handles decoding + frame sampling
        + timestamp tracking internally, same way Marlin's caption() does.

        Generation: deterministic (do_sample=False) for v0.1. The model
        card recommends sampling defaults (temp=0.7, top_p=0.8 etc.) for
        Instruct models, but greedy makes the API output reproducible and
        easier to debug. Revisit once we have eval coverage.

        Returns {"raw": str} — the model's answer text.
        """
        import os
        import torch

        if question is None:
            raise ValueError("ask() requires a 'question' string.")
        video_path = _materialize_video(
            video_bytes=video_bytes, video_url=video_url, video_ext=video_ext
        )

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        # Plain absolute path — NOT a file:// URI.
                        # The Qwen3-VL README's `file:///` form is only valid
                        # on the qwen_vl_utils.process_vision_info() path; when
                        # passing messages directly to apply_chat_template,
                        # transformers' internal load_video (video_utils.py)
                        # parses the string as either an http(s) URL or a raw
                        # local path and rejects file:// URIs with
                        # "TypeError: Incorrect format used for video".
                        # (First QwenVL deploy 2026-05-27 hit exactly that.)
                        {"type": "video", "video": str(video_path)},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                # Cap the frame budget to keep video tokens manageable on
                # A10G. fps=None makes num_frames authoritative — both kwargs
                # are documented in the README.
                num_frames=num_frames,
                fps=None,
            ).to("cuda")

            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            # Trim the prompt tokens — `inputs.input_ids` is a tensor; we
            # only want the newly-generated portion (same pattern as
            # MarlinModel.generate above).
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            raw = self.processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0]
            return {"raw": raw}
        finally:
            os.unlink(video_path)


# Simple healthcheck endpoint you can curl after `modal deploy` to confirm
# the app is live. It does NOT load the model (no GPU), so it's free to hit.
@app.function()
@modal.fastapi_endpoint(method="GET")
def healthz():
    return {
        "status": "ok",
        "marlin": {"repo": MODEL_REPO, "gpu": GPU_TYPE},
        "qwen": {"repo": QWEN_REPO, "gpu": QWEN_GPU},
    }


# Optional local smoke test:
#     modal run modal_app.py
# This calls the model on a single dummy frame to verify end-to-end deploy.
@app.local_entrypoint()
def main():
    from PIL import Image
    print("Running a single-frame smoke test against the deployed model...")
    dummy = [Image.new("RGB", (448, 448), color="gray")] * 16  # 16 grey frames
    result = MarlinModel().generate.remote(dummy, "What do you see?")
    print("RAW OUTPUT:")
    print(result["raw"])
