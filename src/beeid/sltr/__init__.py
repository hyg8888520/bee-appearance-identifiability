"""Selective local trajectory repair (SLTR), development-only and fail-closed."""

SLTR_MODELS = ("resnet50", "dinov3")
SLTR_VARIANTS = (
    "immediate_baseline",
    "always_repair",
    "frequency_matched_random",
    "oracle_selector",
    "learned_selector",
)
SLTR_PRIMARY_VARIANT = "learned_selector"
