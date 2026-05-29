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

## 7. Open questions you should think about

1. **What's the wedge?** "General video analysis API" is a crowded market (Twelve Labs, Marqo, OpenAI). What can Clip do that they don't? My guesses: **self-hostable** (their videos never leave your infra) + **cheaper** (10x) + **structured outputs** (their APIs are chat-style). Pick one and lead with it.
2. **Distribution?** Reddit (r/LocalLLaMA), Hacker News (Show HN once Milestone 1 is done), Twitter/X with side-by-side demo videos vs. Gemini, blog post benchmarking Marlin vs. GPT-4o on a specific use case (sports? CCTV?).
3. **What's the moat?** Not the models (Tencent/NemoStation own those). The moat is: (a) the orchestration layer, (b) the routing rules tuned across many real video types, (c) the fine-tunes you'll do on top, (d) the customer relationships.
