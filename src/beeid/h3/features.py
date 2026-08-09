"""Resumable H3 feature extraction over frozen project-train and development videos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..cache import FeatureCache, open_cache
from ..config import ExperimentConfig
from ..data.crops import load_crop
from ..models import FeatureExtractor, create_extractor, model_signature
from ..utils import atomic_write_json, canonical_json, sha256_text
from .core import H3Inputs, validate_h3_inputs


def h3_observation_ids_sha256(inputs: H3Inputs) -> str:
    return sha256_text(
        canonical_json([item.observation_id for item in inputs.observations])
    )


def _registry(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid H3 cache registry: {path}")
    return {str(key): str(location) for key, location in value.items()}


def _signature(
    config: ExperimentConfig,
    inputs: H3Inputs,
    model_name: str,
    extractor: FeatureExtractor,
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "H3_RAM_Bee",
        "implementation": "beeid.h3.features:v1",
        "manifest_sha256": inputs.audit["manifest_sha256"],
        "protocol_sha256": inputs.audit["protocol_sha256"],
        "project_split_sha256": inputs.audit["project_split_sha256"],
        "observation_ids_sha256": h3_observation_ids_sha256(inputs),
        "partitions": ["project_train", "development_validation"],
        "model": override or model_signature(model_name, config, extractor.details),
        "crop_expansion": config.protocol.crop_expansion,
        "input_size": config.protocol.input_size,
        "amp": config.runtime.amp,
        "amp_dtype": config.runtime.amp_dtype,
        "final_test_read": False,
    }


def extract_h3_features(
    config: ExperimentConfig,
    model_name: str,
    *,
    extractor: FeatureExtractor | None = None,
    signature_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = validate_h3_inputs(config)
    active = extractor or create_extractor(model_name, config)
    signature = _signature(
        config, inputs, model_name, active, signature_override
    )
    cache = FeatureCache(config.paths.cache_root, f"h3__{model_name}", signature)
    cache.initialize()
    observations = list(inputs.observations)
    shard_size = config.runtime.cache_shard_size
    for shard_index, start in enumerate(range(0, len(observations), shard_size)):
        shard = observations[start : start + shard_size]
        identifiers = [item.observation_id for item in shard]
        if cache.validate_shard(shard_index, identifiers):
            continue
        batches: list[np.ndarray] = []
        for batch_start in range(0, len(shard), config.runtime.batch_size):
            batch = shard[batch_start : batch_start + config.runtime.batch_size]
            crops = [load_crop(config.paths.bee24_root, item) for item in batch]
            try:
                batches.append(active.encode(crops))
            finally:
                for crop in crops:
                    crop.close()
        cache.write_shard(shard_index, identifiers, np.concatenate(batches, axis=0))
    registry_path = config.paths.output_root / "h3_cache_locations.json"
    registry = _registry(registry_path)
    registry[model_name] = str(cache.directory)
    atomic_write_json(registry_path, dict(sorted(registry.items())))
    return {
        "status": "completed",
        "model": model_name,
        "cache": str(cache.directory),
        "fingerprint": cache.fingerprint,
        "observation_count": len(observations),
        "final_test_read": False,
    }


def load_h3_embeddings(
    config: ExperimentConfig, model_name: str, inputs: H3Inputs | None = None
) -> tuple[np.ndarray, FeatureCache]:
    selected = inputs or validate_h3_inputs(config)
    registry_path = config.paths.output_root / "h3_cache_locations.json"
    registry = _registry(registry_path)
    if model_name not in registry:
        raise RuntimeError(f"No H3 feature cache is registered for {model_name}")
    cache = open_cache(Path(registry[model_name]), f"h3__{model_name}")
    signature = cache.signature
    expected = {
        "experiment": "H3_RAM_Bee",
        "implementation": "beeid.h3.features:v1",
        "manifest_sha256": selected.audit["manifest_sha256"],
        "protocol_sha256": selected.audit["protocol_sha256"],
        "project_split_sha256": selected.audit["project_split_sha256"],
        "observation_ids_sha256": h3_observation_ids_sha256(selected),
        "crop_expansion": config.protocol.crop_expansion,
        "input_size": config.protocol.input_size,
        "amp": config.runtime.amp,
        "amp_dtype": config.runtime.amp_dtype,
        "final_test_read": False,
    }
    mismatches = [key for key, value in expected.items() if signature.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f"H3 cache signature mismatch for {model_name}: {', '.join(mismatches)}"
        )
    model_value = signature.get("model")
    if not isinstance(model_value, dict) or model_value.get("name") != model_name:
        raise RuntimeError(f"H3 cache model signature mismatch for {model_name}")
    identifiers = [item.observation_id for item in selected.observations]
    embeddings = cache.load_all(identifiers, config.runtime.cache_shard_size)
    return embeddings, cache
