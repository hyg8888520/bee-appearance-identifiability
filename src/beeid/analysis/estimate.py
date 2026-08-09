"""Estimate cache size and runtime from measured server throughput."""

from __future__ import annotations

from typing import Any

from ..config import ExperimentConfig
from ..data.manifest import load_manifest


def estimate_run(
    config: ExperimentConfig,
    *,
    embedding_dimension: int,
    observations_per_second: float | None,
    peak_memory_gib: float | None,
) -> dict[str, Any]:
    observations = load_manifest(
        config.manifest_path, split=config.protocol.evaluation_split, valid_only=True
    )
    count = len(observations)
    result: dict[str, Any] = {
        "observation_count": count,
        "embedding_dimension": embedding_dimension,
        "float32_embedding_bytes": count * embedding_dimension * 4,
        "estimated_cache_gib_before_npz_compression": count * embedding_dimension * 4 / (1024**3),
        "observations_per_second": observations_per_second,
        "estimated_extraction_seconds": (
            count / observations_per_second if observations_per_second and observations_per_second > 0 else None
        ),
        "measured_peak_memory_gib": peak_memory_gib,
        "basis": "SERVER_VALIDATION_PENDING until populated from a real 100-300 frame or full-video smoke",
    }
    return result
