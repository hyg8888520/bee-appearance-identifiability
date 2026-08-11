"""H5 BeeTrackQuery: trainable persistent identity queries over frozen crop features."""

H5_MODELS = ("resnet50", "dinov3")
H5_VARIANTS = (
    "frozen_h3_baseline",
    "persistent_query_no_memory",
    "beetrackquery_short_memory",
    "beetrackquery_gated_memory",
)
H5_PRIMARY_VARIANT = "beetrackquery_gated_memory"
