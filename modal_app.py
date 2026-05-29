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

APP_NAME = "clip-marlin"
MODEL_REPO = "NemoStation/Marlin-2B"
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
    timeout=600,            # max seconds a single inference call can take
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
        video_bytes: bytes,
        video_ext: str = "mp4",
        max_new_tokens: int = 2048,
    ) -> dict:
        """
        Run Marlin's native .caption() method on a video file.

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
        import os
        import tempfile

        # Marlin's internal decoder needs a real file on disk. Write the
        # incoming bytes to a tempfile in the container's filesystem,
        # then clean up after the call regardless of success/failure.
        with tempfile.NamedTemporaryFile(
            suffix=f".{video_ext}", delete=False
        ) as f:
            f.write(video_bytes)
            video_path = f.name

        try:
            result = self.model.caption(
                video_path, max_new_tokens=max_new_tokens
            )
            # Marlin's caption() returns a dict with "caption" (raw text),
            # "scene" (parsed paragraph), and "events" (list of parsed
            # dicts). We return just the raw text and let the gateway's
            # routing.parse_describe handle structured extraction — keeps
            # parsing consistent across model backends.
            return {"raw": result["caption"]}
        finally:
            os.unlink(video_path)

    @modal.method()
    def find(
        self,
        video_bytes: bytes,
        event: str,
        video_ext: str = "mp4",
    ) -> dict:
        """
        Run Marlin's native .find() method to temporally ground a
        natural-language event query inside the video.

        Like caption(), this is the canonical pathway per the Marlin
        model card. Marlin has a separately-trained "find" mode that
        emits `From X.X to Y.Y.` and the custom modeling code parses it
        into a `(start, end)` span tuple. Going through raw generate()
        with our own FIND_PROMPT_TEMPLATE would be reinventing that mode
        from the outside — same lesson as caption().

        Returns {"raw": str, "span": (s, e) | None, "format_ok": bool}.
        """
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=f".{video_ext}", delete=False
        ) as f:
            f.write(video_bytes)
            video_path = f.name

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
        video_bytes: bytes,
        question: str,
        video_ext: str = "mp4",
        max_new_tokens: int = 512,
        # num_frames caps the video-token cost. 32 keeps us comfortably
        # within A10G memory for short-to-medium clips; the model card
        # warns to mind GPU memory budget on long video. fps=None makes
        # num_frames authoritative (per the README "Pixel Control" example).
        num_frames: int = 32,
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
        import tempfile
        import torch

        with tempfile.NamedTemporaryFile(
            suffix=f".{video_ext}", delete=False
        ) as f:
            f.write(video_bytes)
            video_path = f.name

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
