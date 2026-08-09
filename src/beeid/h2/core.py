"""Frozen-split and completed-H1 input validation for H2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..config import ExperimentConfig, H2Config
from ..data.manifest import load_manifest, manifest_sha256
from ..data.mot import Observation


class H2InputError(RuntimeError):
    """Raised when H2 would violate the frozen development protocol."""


def require_h2(config: ExperimentConfig) -> H2Config:
    if config.h2 is None:
        raise H2InputError("This command requires an h2 section in the YAML config")
    return config.h2


def _project_split(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise H2InputError(f"Frozen project split does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise H2InputError(f"Invalid frozen project split {path}: {error}") from error
    if not isinstance(value, dict) or value.get("status") != "FROZEN":
        raise H2InputError(f"H2 requires a split with status FROZEN: {path}")
    return value


def validate_h2_inputs(config: ExperimentConfig) -> tuple[list[Observation], dict[str, Any]]:
    h2 = require_h2(config)
    required_h1_artifacts = (
        config.h2_source_root / "run_metadata.json",
        config.h2_source_root / "summary.csv",
        config.h2_source_root / "query_results.csv",
        config.h2_source_root / "resolved_split.yaml",
        config.h2_source_root / "cache_locations.json",
    )
    missing_h1_artifacts = [str(path) for path in required_h1_artifacts if not path.is_file()]
    if missing_h1_artifacts:
        raise H2InputError(
            "H2 requires a completed H1 output; missing: " + ", ".join(missing_h1_artifacts)
        )
    if not config.h2_manifest_path.is_file():
        raise H2InputError(
            f"Completed H1 manifest is missing: {config.h2_manifest_path}. "
            "H2 never rebuilds or silently changes the H1 manifest."
        )
    split = _project_split(h2.project_split_path)
    try:
        expected = set(split["partitions"]["development_validation"]["video_ids"])
    except (KeyError, TypeError) as error:
        raise H2InputError("Frozen split is missing partitions.development_validation.video_ids") from error
    observations = load_manifest(
        config.h2_manifest_path, split=config.protocol.evaluation_split, valid_only=True
    )
    if config.dataset.video_ids:
        requested = set(config.dataset.video_ids)
        observations = [item for item in observations if item.video_id in requested]
    if config.dataset.max_videos is not None:
        selected_videos = sorted({item.video_id for item in observations})[
            : config.dataset.max_videos
        ]
        observations = [item for item in observations if item.video_id in set(selected_videos)]
    if config.dataset.max_frames_per_video is not None:
        observations = [
            item
            for item in observations
            if item.frame <= config.dataset.max_frames_per_video
        ]
    if not observations:
        raise H2InputError("H2 source manifest contains no valid development observations")
    observed = {item.video_id for item in observations}
    unexpected = sorted(observed - expected)
    if unexpected:
        raise H2InputError(
            "H2 source contains videos outside frozen development validation: "
            + ", ".join(unexpected)
        )
    if not h2.allow_subset and observed != expected:
        missing = sorted(expected - observed)
        raise H2InputError(
            "Full H2 requires every frozen development video; missing: " + ", ".join(missing)
        )
    current_manifest_hash = manifest_sha256(config.h2_manifest_path)
    expected_manifest_hash = split.get("provenance", {}).get("manifest_sha256")
    if not h2.allow_subset and expected_manifest_hash != current_manifest_hash:
        raise H2InputError(
            "Completed H1 manifest hash differs from the frozen project split provenance: "
            f"expected {expected_manifest_hash}, got {current_manifest_hash}"
        )
    h1_metadata_path = config.h2_source_root / "run_metadata.json"
    value = json.loads(h1_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise H2InputError(f"Invalid completed H1 metadata: {h1_metadata_path}")
    h1_metadata: dict[str, Any] = value
    recorded = value.get("manifest_sha256")
    if recorded and recorded != current_manifest_hash:
        raise H2InputError(
            f"H1 run_metadata manifest hash {recorded} disagrees with {current_manifest_hash}"
        )
    if h2.manual_annotations_csv is not None and not h2.manual_annotations_csv.is_file():
        raise H2InputError(
            f"Configured H2 manual annotation file does not exist: {h2.manual_annotations_csv}"
        )
    audit = {
        "status": "valid",
        "role": "development_validation",
        "allow_subset": h2.allow_subset,
        "manifest": str(config.h2_manifest_path),
        "manifest_sha256": current_manifest_hash,
        "observation_count": len(observations),
        "observation_subset_controls": {
            "video_ids": list(config.dataset.video_ids),
            "max_videos": config.dataset.max_videos,
            "max_frames_per_video": config.dataset.max_frames_per_video,
        },
        "observed_video_ids": sorted(observed),
        "expected_video_ids": sorted(expected),
        "project_split": str(h2.project_split_path),
        "project_split_id": split.get("split_id"),
        "h1_metadata_present": True,
        "final_test_read": False,
    }
    return observations, audit
