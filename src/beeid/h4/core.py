"""Leakage-safe H4 inputs and read-only reuse of completed H3 artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import ExperimentConfig, H4Config
from ..data.manifest import load_manifest, manifest_sha256
from ..data.mot import Observation
from ..h3.core import read_project_split
from ..utils import sha256_file
from .protocol import H4ProtocolError, validate_h4_protocol


class H4InputError(RuntimeError):
    """Raised when H4 would use incomplete, changed, or leakage-prone inputs."""


@dataclass(frozen=True)
class H4Inputs:
    observations: tuple[Observation, ...]
    video_partitions: dict[str, tuple[str, ...]]
    audit: dict[str, Any]


def require_h4(config: ExperimentConfig) -> H4Config:
    if config.h4 is None:
        raise H4InputError("This command requires an h4 section in the YAML config")
    return config.h4


def _subset(observations: Sequence[Observation], config: ExperimentConfig) -> list[Observation]:
    selected = list(observations)
    if config.dataset.video_ids:
        requested = set(config.dataset.video_ids)
        selected = [item for item in selected if item.video_id in requested]
    if config.dataset.max_videos is not None:
        videos = sorted({item.video_id for item in selected})[: config.dataset.max_videos]
        selected = [item for item in selected if item.video_id in set(videos)]
    if config.dataset.max_frames_per_video is not None:
        selected = [
            item for item in selected if item.frame <= config.dataset.max_frames_per_video
        ]
    return selected


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise H4InputError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise H4InputError(f"{label} must be a JSON object: {path}")
    return value


def _locked_parameters(h4: H4Config) -> dict[str, Any]:
    return {
        "horizons": list(h4.horizons),
        "go_horizon": h4.go_horizon,
        "oracle_history_length": h4.oracle_history_length,
        "oracle_unique_margin": h4.oracle_unique_margin,
        "primary_horizon": h4.primary_horizon,
        "beam_width": h4.beam_width,
        "max_component_size": h4.max_component_size,
        "ambiguity_margin": h4.ambiguity_margin,
        "memory_alpha": h4.memory_alpha,
        "appearance_weight": h4.appearance_weight,
        "motion_weight": h4.motion_weight,
        "max_normalized_distance": h4.max_normalized_distance,
        "min_assignment_score": h4.min_assignment_score,
        "unmatched_penalty": h4.unmatched_penalty,
        "max_age": h4.max_age,
        "min_recoverable_fraction": h4.min_recoverable_fraction,
        "min_events_per_model": h4.min_events_per_model,
        "min_videos_with_events": h4.min_videos_with_events,
        "noninferiority_tolerance": h4.noninferiority_tolerance,
        "min_nonharmed_videos": h4.min_nonharmed_videos,
    }


def validate_h4_inputs(config: ExperimentConfig) -> H4Inputs:
    h4 = require_h4(config)
    try:
        protocol = validate_h4_protocol(h4.protocol_lock_path, h4.protocol_checksum_path)
    except H4ProtocolError as error:
        raise H4InputError(str(error)) from error
    mismatches = [
        key
        for key, value in _locked_parameters(h4).items()
        if protocol["parameters"].get(key) != value
    ]
    if mismatches:
        raise H4InputError(
            "H4 YAML differs from the frozen protocol: " + ", ".join(mismatches)
        )
    configured_split = h4.project_split_path.resolve(strict=False)
    protocol_split = Path(protocol["project_split"]).resolve(strict=False)
    if configured_split != protocol_split:
        raise H4InputError("h4.project_split must be the exact split referenced by the lock")
    if sha256_file(configured_split) != protocol["project_split_sha256"]:
        raise H4InputError("Frozen project split hash disagrees with the H4 protocol")
    split = read_project_split(configured_split)
    partitions = {
        name: tuple(str(item) for item in split["partitions"][name]["video_ids"])
        for name in ("project_train", "development_validation", "final_test")
    }

    manifest = config.h3_manifest_path
    if not manifest.is_file():
        raise H4InputError(f"Completed H1 manifest is missing: {manifest}")
    current_manifest_sha = manifest_sha256(manifest)
    frozen_manifest_sha = split.get("provenance", {}).get("manifest_sha256")
    if current_manifest_sha != frozen_manifest_sha:
        raise H4InputError(
            f"H4 manifest differs from the frozen split: expected {frozen_manifest_sha}, "
            f"got {current_manifest_sha}"
        )
    all_observations = load_manifest(manifest, valid_only=True)
    development_videos = set(partitions["development_validation"])
    final_videos = set(partitions["final_test"])
    selected = _subset(
        [item for item in all_observations if item.video_id in development_videos], config
    )
    if not selected:
        raise H4InputError("H4 selection contains no development_validation observations")
    if any(item.video_id in final_videos for item in selected):
        raise H4InputError("H4 selected a final-test observation")
    if any(
        not math.isclose(
            item.crop_expansion, config.protocol.crop_expansion,
            rel_tol=0.0, abs_tol=1e-12,
        )
        for item in selected
    ):
        raise H4InputError("H4 crop expansion differs from the frozen H1 manifest")
    observed_videos = {item.video_id for item in selected}
    if not h4.allow_subset and observed_videos != development_videos:
        missing = sorted(development_videos - observed_videos)
        raise H4InputError("Full H4 development is missing videos: " + ", ".join(missing))
    if not h4.allow_subset and (
        config.dataset.max_frames_per_video is not None
        or config.dataset.max_videos is not None
        or config.dataset.video_ids
    ):
        raise H4InputError("Full H4 development cannot use dataset subset controls")

    source_root = config.paths.h3_output_root
    assert source_root is not None
    required = (
        "h3_run_metadata.json", "h3_tracking_metadata.json",
        "h3_observation_signals.csv", "h3_thresholds.json", "h3_cache_locations.json",
    )
    missing_artifacts = [str(source_root / name) for name in required if not (source_root / name).is_file()]
    if missing_artifacts:
        raise H4InputError("Completed H3 artifacts are missing: " + ", ".join(missing_artifacts))
    run_metadata = _load_json(source_root / "h3_run_metadata.json", "H3 run metadata")
    tracking_metadata = _load_json(
        source_root / "h3_tracking_metadata.json", "H3 tracking metadata"
    )
    thresholds = _load_json(source_root / "h3_thresholds.json", "H3 thresholds")
    if run_metadata.get("status") != "completed_development_gt_boxes":
        raise H4InputError("H4 requires H3 status completed_development_gt_boxes")
    if tracking_metadata.get("status") != "completed":
        raise H4InputError("H4 requires completed H3 tracking metadata")
    for label, artifact in (
        ("H3 run metadata", run_metadata),
        ("H3 tracking metadata", tracking_metadata),
        ("H3 thresholds", thresholds),
    ):
        if artifact.get("final_test_read") is not False:
            raise H4InputError(f"{label} does not prove final_test_read=false")
    source_audit = tracking_metadata.get("input_audit")
    if not isinstance(source_audit, dict):
        raise H4InputError("H3 tracking metadata has no input audit")
    expected_source = {
        "manifest_sha256": current_manifest_sha,
        "project_split_sha256": protocol["project_split_sha256"],
        "protocol_sha256": protocol["source_h3_protocol_sha256"],
        "final_test_read": False,
    }
    source_mismatches = [
        key for key, value in expected_source.items() if source_audit.get(key) != value
    ]
    if source_mismatches:
        raise H4InputError(
            "H3 provenance is incompatible with H4: " + ", ".join(source_mismatches)
        )
    selected.sort(
        key=lambda item: (item.video_id, item.frame, item.center_x, item.center_y, item.observation_id)
    )
    audit = {
        "status": "valid",
        "role": "h4_development_only",
        "allow_subset": h4.allow_subset,
        "manifest": str(manifest),
        "manifest_sha256": current_manifest_sha,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "source_h3_protocol_sha256": protocol["source_h3_protocol_sha256"],
        "project_split_sha256": protocol["project_split_sha256"],
        "source_h3_root": str(source_root),
        "source_h3_run_metadata_sha256": sha256_file(source_root / "h3_run_metadata.json"),
        "source_h3_tracking_metadata_sha256": sha256_file(source_root / "h3_tracking_metadata.json"),
        "source_h3_thresholds_sha256": sha256_file(source_root / "h3_thresholds.json"),
        "observation_count": len(selected),
        "development_videos": sorted(observed_videos),
        "selected_final_test_videos": [],
        "oracle_gt_use": "OFFLINE_DIAGNOSTIC_ONLY",
        "method_gt_identity_input": False,
        "final_test_read": False,
    }
    return H4Inputs(tuple(selected), partitions, audit)


def load_h4_source_embeddings(
    config: ExperimentConfig, model_name: str, inputs: H4Inputs | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read only selected embeddings while validating every touched H3 shard."""
    selected = inputs or validate_h4_inputs(config)
    source_root = config.paths.h3_output_root
    assert source_root is not None
    registry = _load_json(source_root / "h3_cache_locations.json", "H3 cache registry")
    location = registry.get(model_name)
    if not isinstance(location, str) or not location:
        raise H4InputError(f"H3 cache registry has no model {model_name}")
    directory = Path(location).expanduser().resolve(strict=False)
    metadata_path = directory / "cache_manifest.json"
    metadata = _load_json(metadata_path, f"H3 {model_name} cache metadata")
    signature = metadata.get("signature")
    if not isinstance(signature, dict):
        raise H4InputError(f"H3 cache signature is invalid for {model_name}")
    model_signature = signature.get("model")
    if not isinstance(model_signature, dict) or model_signature.get("name") != model_name:
        raise H4InputError(f"H3 cache model signature mismatch for {model_name}")
    expected_signature = {
        "experiment": "H3_RAM_Bee",
        "manifest_sha256": selected.audit["manifest_sha256"],
        "project_split_sha256": selected.audit["project_split_sha256"],
        "final_test_read": False,
    }
    mismatches = [
        key for key, value in expected_signature.items() if signature.get(key) != value
    ]
    if mismatches:
        raise H4InputError(
            f"H3 cache signature mismatch for {model_name}: {', '.join(mismatches)}"
        )
    wanted = {item.observation_id for item in selected.observations}
    found: dict[str, np.ndarray] = {}
    shards = sorted(directory.glob("shard-*.npz"))
    if not shards:
        raise H4InputError(f"H3 cache has no feature shards: {directory}")
    for shard in shards:
        try:
            with np.load(shard, allow_pickle=False) as data:
                identifiers = data["observation_ids"].astype(str).tolist()
                values = data["embeddings"]
                if values.dtype != np.float32 or values.ndim != 2:
                    raise H4InputError(f"Invalid H3 embedding array in {shard}")
                if values.shape[0] != len(identifiers) or not np.isfinite(values).all():
                    raise H4InputError(f"Invalid H3 embedding rows in {shard}")
                norms = np.linalg.norm(values, axis=1)
                if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
                    raise H4InputError(f"Non-normalized H3 embeddings in {shard}")
                for index, identifier in enumerate(identifiers):
                    if identifier not in wanted:
                        continue
                    if identifier in found:
                        raise H4InputError(f"Duplicate H3 cache observation: {identifier}")
                    found[identifier] = values[index].astype(np.float32, copy=True)
        except (OSError, KeyError, ValueError) as error:
            raise H4InputError(f"Cannot validate H3 cache shard {shard}: {error}") from error
        if len(found) == len(wanted):
            break
    missing = sorted(wanted - set(found))
    if missing:
        preview = ", ".join(missing[:5])
        raise H4InputError(
            f"H3 cache {model_name} is missing {len(missing)} selected observations: {preview}"
        )
    aligned = np.stack([found[item.observation_id] for item in selected.observations])
    return aligned, {
        "directory": str(directory),
        "fingerprint": metadata.get("fingerprint"),
        "cache_manifest_sha256": sha256_file(metadata_path),
        "model_signature": model_signature,
        "selected_observation_count": len(selected.observations),
        "scanned_shard_count": len(shards),
        "read_only": True,
    }
