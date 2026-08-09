"""Measured-throughput resource estimate for H2 intervention extraction."""

from __future__ import annotations

from typing import Any

from ..config import ExperimentConfig
from .core import require_h2, validate_h2_inputs


def estimate_h2_run(
    config: ExperimentConfig,
    *,
    model_name: str,
    embedding_dimension: int,
    observations_per_second: float | None,
    peak_memory_gib: float | None,
) -> dict[str, Any]:
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be positive")
    if observations_per_second is not None and observations_per_second <= 0:
        raise ValueError("observations_per_second must be positive")
    if peak_memory_gib is not None and peak_memory_gib <= 0:
        raise ValueError("peak_memory_gib must be positive")
    h2 = require_h2(config)
    observations, _ = validate_h2_inputs(config)
    primary = next(item for item in h2.variants if item.name == h2.primary_variant)
    h1_reuse_eligible = bool(
        primary.pixel_view == "raw"
        and primary.crop_expansion == config.protocol.crop_expansion
        and primary.input_size == config.protocol.input_size
        and not h2.allow_subset
    )
    total_rows = len(observations) * len(h2.variants)
    reusable_rows = len(observations) if h1_reuse_eligible else 0
    rows_to_extract = total_rows - reusable_rows
    raw_float32_bytes = total_rows * embedding_dimension * 4
    return {
        "experiment": "H2_observation_reliability",
        "model": model_name,
        "observation_count": len(observations),
        "variant_count": len(h2.variants),
        "total_embedding_rows": total_rows,
        "h1_primary_cache_reuse_eligible": h1_reuse_eligible,
        "potential_reused_rows": reusable_rows,
        "rows_to_extract": rows_to_extract,
        "embedding_dimension": embedding_dimension,
        "uncompressed_float32_embedding_gib": raw_float32_bytes / (1024**3),
        "npz_disk_note": "actual compressed cache size must be measured; IDs and metadata are additional",
        "measured_observations_per_second": observations_per_second,
        "estimated_extraction_seconds": (
            rows_to_extract / observations_per_second
            if observations_per_second is not None
            else None
        ),
        "measured_peak_memory_gib": peak_memory_gib,
        "server_validation_status": "SERVER_VALIDATION_PENDING",
    }
