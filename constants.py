"""Shared constants — no heavy deps. Safe to import anywhere."""

from enum import Enum


class ModelChoice(str, Enum):
    MARLIN_2B = "marlin-2b"
    TIMELENS_8B = "timelens-8b"
    # Open-ended reasoning model per PRD §5.4. Routed to from /ask. Deployed
    # as a separate Modal class (QwenVL) alongside the existing MarlinModel
    # class in modal_app.py.
    QWEN3_VL_8B = "qwen3-vl-8b"


MODEL_REPOS = {
    ModelChoice.MARLIN_2B: "NemoStation/Marlin-2B",
    ModelChoice.TIMELENS_8B: "TencentARC/TimeLens-8B",
    ModelChoice.QWEN3_VL_8B: "Qwen/Qwen3-VL-8B-Instruct",
}

# Models actually deployed on the Modal backend in v0.1. TimeLens-8B is in
# the PRD roadmap (Milestone 2) but is NOT deployed, so auto-routing must
# never hand a request to it. Keep this in sync with ModalVLM._MODAL_CLASSES.
DEPLOYED_MODELS = frozenset({ModelChoice.MARLIN_2B, ModelChoice.QWEN3_VL_8B})
