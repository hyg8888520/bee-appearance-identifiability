"""Leakage-safe H5 inputs and read-only H3 feature reuse."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..cache import open_cache
from ..config import ExperimentConfig, H5Config
from ..h3.core import H3Inputs, validate_h3_inputs
from ..utils import sha256_file
from .protocol import validate_h5_protocol


class H5InputError(RuntimeError):
    """Raised when H5 provenance, split isolation or source features are invalid."""


@dataclass(frozen=True)
class H5Inputs:
    h3_inputs: H3Inputs
    audit: dict[str, Any]

    def indices(self, partition: str) -> list[int]:
        return [
            index for index, item in enumerate(self.h3_inputs.observations)
            if self.h3_inputs.partition_by_observation[item.observation_id] == partition
        ]


def require_h5(config: ExperimentConfig) -> H5Config:
    if config.h5 is None:
        raise H5InputError("H5 commands require an h5 config section")
    return config.h5


def validate_h5_inputs(config: ExperimentConfig) -> H5Inputs:
    h5 = require_h5(config)
    protocol = validate_h5_protocol(h5.protocol_lock_path, h5.protocol_checksum_path)
    h3_inputs = validate_h3_inputs(config)
    frozen_parameters = protocol["parameters"]
    configured_parameters = {
        "hidden_dim": h5.hidden_dim, "num_heads": h5.num_heads,
        "clip_length": h5.clip_length, "memory_slots": h5.memory_slots,
        "memory_top_k": h5.memory_top_k, "dropout": h5.dropout,
        "epochs": h5.epochs, "learning_rate": h5.learning_rate,
        "weight_decay": h5.weight_decay,
        "max_train_clips_per_video": h5.max_train_clips_per_video,
        "train_clip_stride": h5.train_clip_stride,
        "reliability_loss_weight": h5.reliability_loss_weight,
        "update_gate": h5.update_gate, "memory_mix": h5.memory_mix,
        "max_age": h5.max_age, "min_assignment_score": h5.min_assignment_score,
        "max_normalized_distance": h5.max_normalized_distance,
    }
    mismatches = [key for key, value in configured_parameters.items() if frozen_parameters.get(key) != value]
    if mismatches:
        raise H5InputError("H5 config differs from frozen protocol: " + ", ".join(mismatches))
    if h3_inputs.audit["protocol_sha256"] != protocol["source_h3_protocol_sha256"]:
        raise H5InputError("H5 source H3 protocol hash differs from the frozen protocol")
    if sha256_file(h5.project_split_path) != h3_inputs.audit["project_split_sha256"]:
        raise H5InputError("H5 and H3 project split hashes differ")
    source_root = config.paths.h3_output_root
    assert source_root is not None
    metadata_path = source_root / "h3_run_metadata.json"
    if not metadata_path.is_file():
        raise H5InputError(f"Completed H3 metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed_development_gt_boxes":
        raise H5InputError("H5 requires completed_development_gt_boxes H3 input")
    if metadata.get("final_test_read") is not False:
        raise H5InputError("H3 metadata does not prove final_test_read=false")
    train_count = sum(
        h3_inputs.partition_by_observation[item.observation_id] == "project_train"
        for item in h3_inputs.observations
    )
    development_count = len(h3_inputs.observations) - train_count
    if train_count == 0 or development_count == 0:
        raise H5InputError("H5 requires non-empty project_train and development partitions")
    return H5Inputs(
        h3_inputs,
        {
            **h3_inputs.audit,
            "status": "valid",
            "role": "h5_beetrackquery_development_only",
            "h5_protocol_id": protocol["protocol_id"],
            "h5_protocol_sha256": protocol["protocol_sha256"],
            "source_h3_root": str(source_root),
            "source_h3_metadata_sha256": sha256_file(metadata_path),
            "fit_partition": "project_train",
            "evaluation_partition": "development_validation",
            "project_train_observations": train_count,
            "development_observations": development_count,
            "backbone_frozen": True,
            "feature_reextraction": False,
            "final_test_read": False,
        },
    )


def load_h5_source_embeddings(
    config: ExperimentConfig, model_name: str, inputs: H5Inputs | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = inputs or validate_h5_inputs(config)
    source_root = config.paths.h3_output_root
    assert source_root is not None
    registry_path = source_root / "h3_cache_locations.json"
    if not registry_path.is_file():
        raise H5InputError(f"H3 cache registry is missing: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or model_name not in registry:
        raise H5InputError(f"H3 cache registry has no {model_name} entry")
    cache = open_cache(Path(str(registry[model_name])), f"h3__{model_name}")
    signature = cache.signature
    expected_signature = {
        "experiment": "H3_RAM_Bee",
        "implementation": "beeid.h3.features:v1",
        "manifest_sha256": selected.audit["manifest_sha256"],
        "protocol_sha256": selected.audit["protocol_sha256"],
        "project_split_sha256": selected.audit["project_split_sha256"],
        "crop_expansion": config.protocol.crop_expansion,
        "input_size": config.protocol.input_size,
        "final_test_read": False,
    }
    mismatches = [key for key, value in expected_signature.items() if signature.get(key) != value]
    model_signature = signature.get("model")
    if not isinstance(model_signature, dict) or model_signature.get("name") != model_name:
        mismatches.append("model")
    if mismatches:
        raise H5InputError(
            f"H3 source cache signature mismatch for {model_name}: " + ", ".join(mismatches)
        )
    required_ids = [item.observation_id for item in selected.h3_inputs.observations]
    required = set(required_ids)
    found: dict[str, np.ndarray] = {}
    for shard_path in sorted(cache.directory.glob("shard-*.npz")):
        try:
            with np.load(shard_path, allow_pickle=False) as shard:
                identifiers = shard["observation_ids"].astype(str).tolist()
                embeddings = shard["embeddings"].astype(np.float32, copy=False)
        except (OSError, KeyError, ValueError) as error:
            raise H5InputError(f"Cannot read H3 cache shard {shard_path}: {error}") from error
        if embeddings.ndim != 2 or embeddings.shape[0] != len(identifiers):
            raise H5InputError(f"Invalid H3 cache shard shape: {shard_path}")
        if not np.isfinite(embeddings).all() or not np.allclose(
            np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4, rtol=1e-4
        ):
            raise H5InputError(f"Invalid or non-normalized H3 embeddings: {shard_path}")
        for identifier, embedding in zip(identifiers, embeddings):
            if identifier not in required:
                continue
            if identifier in found:
                raise H5InputError(f"Duplicate observation in H3 cache: {identifier}")
            found[identifier] = embedding.copy()
    missing = [identifier for identifier in required_ids if identifier not in found]
    if missing:
        preview = ", ".join(missing[:5])
        raise H5InputError(
            f"H3 cache {model_name} is missing {len(missing)} selected observations: {preview}"
        )
    values = np.stack([found[identifier] for identifier in required_ids]).astype(np.float32)
    return values, {
        "cache_fingerprint": cache.fingerprint,
        "cache_directory": str(cache.directory),
        "read_only": True,
        "feature_reextraction": False,
        "source_cache_observation_scope": "superset_allowed_after_strict_provenance_validation",
        "selected_observation_count": len(required_ids),
    }
