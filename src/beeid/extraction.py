"""Lazy crop batching into an atomic feature cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cache import FeatureCache
from .config import ExperimentConfig
from .data.crops import load_crop
from .data.manifest import load_manifest, manifest_sha256
from .models import FeatureExtractor, create_extractor, model_signature
from .utils import atomic_write_json


def extract_features(
    config: ExperimentConfig,
    model_name: str,
    *,
    extractor: FeatureExtractor | None = None,
    signature_override: dict[str, Any] | None = None,
) -> FeatureCache:
    observations = load_manifest(
        config.manifest_path, split=config.protocol.evaluation_split, valid_only=True
    )
    if not observations:
        raise RuntimeError(f"No valid observations for split {config.protocol.evaluation_split}")
    active_extractor = extractor or create_extractor(model_name, config)
    model_part = signature_override or model_signature(model_name, config, active_extractor.details)
    signature = {
        "format_version": 1,
        "manifest_sha256": manifest_sha256(config.manifest_path),
        "evaluation_split": config.protocol.evaluation_split,
        "model": model_part,
        "crop_expansion": config.protocol.crop_expansion,
        "input_size": config.protocol.input_size,
        "amp": config.runtime.amp,
        "amp_dtype": config.runtime.amp_dtype,
    }
    cache = FeatureCache(config.paths.cache_root, model_name, signature)
    cache.initialize()
    shard_size = config.runtime.cache_shard_size
    for shard_index, start in enumerate(range(0, len(observations), shard_size)):
        shard_observations = observations[start : start + shard_size]
        ids = [item.observation_id for item in shard_observations]
        if cache.validate_shard(shard_index, ids):
            continue
        feature_batches: list[np.ndarray] = []
        for batch_start in range(0, len(shard_observations), config.runtime.batch_size):
            batch = shard_observations[batch_start : batch_start + config.runtime.batch_size]
            crops = [load_crop(config.paths.bee24_root, item) for item in batch]
            feature_batches.append(active_extractor.encode(crops))
            for crop in crops:
                crop.close()
        cache.write_shard(shard_index, ids, np.concatenate(feature_batches, axis=0))

    locations: dict[str, str] = {}
    if config.cache_locations_path.is_file():
        observed = json.loads(config.cache_locations_path.read_text(encoding="utf-8"))
        if isinstance(observed, dict):
            locations = {str(key): str(value) for key, value in observed.items()}
    locations[model_name] = str(cache.directory)
    atomic_write_json(config.cache_locations_path, locations)
    return cache
