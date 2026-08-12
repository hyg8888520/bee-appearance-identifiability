"""H6 global spatiotemporal trajectory reasoning experiment."""

H6_MODELS = ("resnet50", "dinov3")
H6_VARIANTS = (
    "baseline_association",
    "global_trajectory_reasoner",
    "calibrated_selective_global",
)
H6_PRIMARY_VARIANT = "calibrated_selective_global"
H6_IMPLEMENTATION = "beeid.h6:v2-amp-retry-global-frame-context-pair-reasoner"
