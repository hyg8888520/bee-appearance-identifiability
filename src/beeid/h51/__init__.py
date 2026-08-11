"""Post-H5 development-only closed-loop diagnostics."""

from __future__ import annotations

H51_MODELS = ("resnet50", "dinov3")
H51_VARIANTS = (
    "persistent_query_no_memory",
    "beetrackquery_short_memory",
    "beetrackquery_gated_memory",
)
H51_DECISION_STOP = "STOP_H51_IMPLEMENTATION_NOT_READY"
H51_DECISION_READY = "READY_FOR_H52_PROTOCOL_DESIGN"
