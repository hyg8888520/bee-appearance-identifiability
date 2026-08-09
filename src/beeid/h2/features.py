"""Resumable multi-intervention H2 feature extraction with safe H1-cache reuse."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..cache import FeatureCache, open_cache
from ..config import ExperimentConfig, H2VariantConfig
from ..models import REAL_MODEL_NAMES, FeatureExtractor, create_extractor, model_signature
from ..utils import atomic_write_json, canonical_json, sha256_text
from .core import require_h2, validate_h2_inputs
from .crops import load_variant_crop


def cache_key(model_name: str, variant_name: str) -> str:
    return f"{model_name}::{variant_name}"


def cache_label(model_name: str, variant_name: str) -> str:
    return f"{model_name}__{variant_name}"


def observation_ids_sha256(observation_ids: Sequence[str]) -> str:
    return sha256_text(canonical_json(list(observation_ids)))


def _locations(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid H2 cache location registry: {path}")
    return {str(key): str(location) for key, location in value.items()}


def _variant(config: ExperimentConfig, name: str) -> H2VariantConfig:
    h2 = require_h2(config)
    for item in h2.variants:
        if item.name == name:
            return item
    raise ValueError(f"Unknown H2 variant {name!r}")


def _model_signature_matches(
    config: ExperimentConfig,
    model_name: str,
    variant: H2VariantConfig,
    observed: Any,
) -> bool:
    if not isinstance(observed, dict) or observed.get("name") != model_name:
        return False
    if model_name not in REAL_MODEL_NAMES:
        return True
    variant_config = replace(
        config, protocol=replace(config.protocol, input_size=variant.input_size)
    )
    expected = model_signature(model_name, variant_config)
    return all(observed.get(key) == value for key, value in expected.items())


def _cache_matches_ids(cache: FeatureCache, expected_ids: list[str]) -> bool:
    observed_ids: list[str] = []
    shard_paths = sorted(cache.directory.glob("shard-*.npz"))
    if not shard_paths:
        return False
    for index, path in enumerate(shard_paths):
        if path != cache.shard_path(index):
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                ids = data["observation_ids"].astype(str).tolist()
        except (OSError, KeyError, ValueError):
            return False
        if not cache.validate_shard(index, ids):
            return False
        observed_ids.extend(ids)
    return observed_ids == expected_ids


def _load_cache_exact(cache: FeatureCache, expected_ids: list[str]) -> np.ndarray:
    if not _cache_matches_ids(cache, expected_ids):
        raise RuntimeError(f"Cache shards do not exactly match selected H2 observations: {cache.directory}")
    chunks: list[np.ndarray] = []
    for path in sorted(cache.directory.glob("shard-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            chunks.append(data["embeddings"].astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def _valid_registered_cache(
    config: ExperimentConfig,
    model_name: str,
    variant: H2VariantConfig,
    expected_ids: list[str],
) -> FeatureCache | None:
    location = _locations(config.h2_cache_locations_path).get(cache_key(model_name, variant.name))
    if not location:
        return None
    try:
        cache = open_cache(Path(location), cache_label(model_name, variant.name))
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError):
        return None
    signature = cache.signature
    expected = {
        "manifest_sha256": config.h2_manifest_path,
        "evaluation_split": config.protocol.evaluation_split,
        "base_model": model_name,
        "variant": asdict(variant),
        "observation_ids_sha256": observation_ids_sha256(expected_ids),
        "intervention_implementation": "beeid.h2.crops:v1",
        "amp": config.runtime.amp,
        "amp_dtype": config.runtime.amp_dtype,
    }
    from ..utils import sha256_file

    if signature.get("manifest_sha256") != sha256_file(expected["manifest_sha256"]):
        return None
    for key in (
        "evaluation_split", "base_model", "variant", "observation_ids_sha256",
        "intervention_implementation", "amp", "amp_dtype"
    ):
        if signature.get(key) != expected[key]:
            return None
    if not _model_signature_matches(config, model_name, variant, signature.get("model")):
        return None
    return cache if _cache_matches_ids(cache, expected_ids) else None


def _compatible_h1_cache(
    config: ExperimentConfig,
    model_name: str,
    variant: H2VariantConfig,
    expected_ids: list[str],
) -> FeatureCache | None:
    if (
        variant.pixel_view != "raw"
        or variant.crop_expansion != config.protocol.crop_expansion
        or variant.input_size != config.protocol.input_size
        or not config.h2_h1_cache_locations_path.is_file()
    ):
        return None
    locations = json.loads(config.h2_h1_cache_locations_path.read_text(encoding="utf-8"))
    if not isinstance(locations, dict) or model_name not in locations:
        return None
    try:
        cache = open_cache(Path(str(locations[model_name])), model_name)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError):
        return None
    signature = cache.signature
    from ..utils import sha256_file

    expected_fields = {
        "manifest_sha256": sha256_file(config.h2_manifest_path),
        "evaluation_split": config.protocol.evaluation_split,
        "crop_expansion": variant.crop_expansion,
        "input_size": variant.input_size,
        "amp": config.runtime.amp,
        "amp_dtype": config.runtime.amp_dtype,
    }
    if any(signature.get(key) != value for key, value in expected_fields.items()):
        return None
    model_part = signature.get("model")
    if not _model_signature_matches(config, model_name, variant, model_part):
        return None
    return cache if _cache_matches_ids(cache, expected_ids) else None


def _register(config: ExperimentConfig, model_name: str, variant_name: str, cache: FeatureCache) -> None:
    locations = _locations(config.h2_cache_locations_path)
    locations[cache_key(model_name, variant_name)] = str(cache.directory)
    atomic_write_json(config.h2_cache_locations_path, dict(sorted(locations.items())))


def extract_h2_features(
    config: ExperimentConfig,
    model_name: str,
    *,
    variant_names: Sequence[str] | None = None,
    extractor: FeatureExtractor | None = None,
    signature_override: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    h2 = require_h2(config)
    observations, _ = validate_h2_inputs(config)
    expected_ids = [item.observation_id for item in observations]
    requested = list(variant_names or [item.name for item in h2.variants])
    if len(set(requested)) != len(requested):
        raise ValueError("H2 variant names must be unique")
    results: dict[str, dict[str, Any]] = {}
    extractors: dict[int, FeatureExtractor] = {}
    for variant_name in requested:
        variant = _variant(config, variant_name)
        registered = _valid_registered_cache(config, model_name, variant, expected_ids)
        if registered is not None:
            results[variant.name] = {
                "cache": str(registered.directory),
                "fingerprint": registered.fingerprint,
                "status": "reused_h2_cache",
            }
            continue
        h1_cache = _compatible_h1_cache(config, model_name, variant, expected_ids)
        if h1_cache is not None:
            _register(config, model_name, variant.name, h1_cache)
            results[variant.name] = {
                "cache": str(h1_cache.directory),
                "fingerprint": h1_cache.fingerprint,
                "status": "reused_h1_cache",
            }
            continue

        if extractor is not None:
            active_extractor = extractor
        elif variant.input_size in extractors:
            active_extractor = extractors[variant.input_size]
        else:
            variant_config = replace(
                config,
                protocol=replace(config.protocol, input_size=variant.input_size),
            )
            active_extractor = create_extractor(model_name, variant_config)
            extractors[variant.input_size] = active_extractor
        model_part = signature_override or model_signature(
            model_name,
            replace(config, protocol=replace(config.protocol, input_size=variant.input_size)),
            active_extractor.details,
        )
        from ..utils import sha256_file

        signature = {
            "format_version": 2,
            "experiment": "H2_observation_reliability",
            "intervention_implementation": "beeid.h2.crops:v1",
            "manifest_sha256": sha256_file(config.h2_manifest_path),
            "evaluation_split": config.protocol.evaluation_split,
            "base_model": model_name,
            "model": model_part,
            "variant": asdict(variant),
            "observation_ids_sha256": observation_ids_sha256(expected_ids),
            "neutral_rgb": [127, 127, 127],
            "amp": config.runtime.amp,
            "amp_dtype": config.runtime.amp_dtype,
        }
        cache = FeatureCache(config.paths.cache_root, cache_label(model_name, variant.name), signature)
        cache.initialize()
        shard_size = config.runtime.cache_shard_size
        for shard_index, start in enumerate(range(0, len(observations), shard_size)):
            shard_observations = observations[start : start + shard_size]
            ids = [item.observation_id for item in shard_observations]
            if cache.validate_shard(shard_index, ids):
                continue
            feature_batches: list[np.ndarray] = []
            for batch_start in range(0, len(shard_observations), config.runtime.batch_size):
                batch = shard_observations[
                    batch_start : batch_start + config.runtime.batch_size
                ]
                crops = [
                    load_variant_crop(
                        config.paths.bee24_root,
                        item,
                        variant,
                        patch_size=h2.patch_size,
                    )
                    for item in batch
                ]
                try:
                    feature_batches.append(active_extractor.encode(crops))
                finally:
                    for crop in crops:
                        crop.close()
            cache.write_shard(shard_index, ids, np.concatenate(feature_batches, axis=0))
        _register(config, model_name, variant.name, cache)
        results[variant.name] = {
            "cache": str(cache.directory),
            "fingerprint": cache.fingerprint,
            "status": "extracted",
        }
    return results


def load_h2_embeddings(
    config: ExperimentConfig,
    model_name: str,
    variant_name: str,
    expected_ids: list[str],
) -> tuple[np.ndarray, FeatureCache]:
    variant = _variant(config, variant_name)
    cache = _valid_registered_cache(config, model_name, variant, expected_ids)
    if cache is None:
        cache = _compatible_h1_cache(config, model_name, variant, expected_ids)
    if cache is None:
        raise RuntimeError(
            f"No valid H2 cache for {model_name}/{variant_name}; run h2-extract first"
        )
    return _load_cache_exact(cache, expected_ids), cache
