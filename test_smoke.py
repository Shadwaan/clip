"""
Smoke test the inference + sampling layer without spinning up the API.

Usage:
  python test_smoke.py path/to/video.mp4 [marlin-2b|timelens-8b]

Skips the model load if `--probe-only` is passed (just tests sampling).
"""

import sys
from pathlib import Path

from sampling import sample_video, probe_duration


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: python test_smoke.py path/to/video.mp4 [marlin-2b|timelens-8b] [--probe-only]")
        sys.exit(1)

    path = Path(args[0])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    probe_only = "--probe-only" in args
    choice_name = next((a for a in args[1:] if not a.startswith("--")), "marlin-2b")

    print(f"--- Probing {path.name} ---")
    duration = probe_duration(path)
    print(f"Duration: {duration:.2f}s")

    print("--- Sampling frames ---")
    sv = sample_video(path)
    print(f"Frames: {len(sv.frames)}  |  Sampled FPS: {sv.fps_sampled:.2f}")
    print(f"First frame size: {sv.frames[0].size}")
    print(f"Timestamps (first/last): {sv.timestamps[0]:.2f}s / {sv.timestamps[-1]:.2f}s")

    if probe_only:
        print("--probe-only set, skipping model load.")
        return

    print(f"--- Loading {choice_name} ---")
    from inference import ModelChoice, get_model, DESCRIBE_PROMPT
    model = get_model(ModelChoice(choice_name))

    print("--- Running describe ---")
    result = model.run(path, DESCRIBE_PROMPT, max_new_tokens=256)
    print(f"Model: {result.model.value}")
    print(f"--- Raw output ---")
    print(result.raw)
    print(f"--- Cleaned ---")
    print(result.text)


if __name__ == "__main__":
    main()
