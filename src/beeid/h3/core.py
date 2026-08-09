"""Leakage-safe H3 input selection and artifact contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import math

import yaml

from ..config import ExperimentConfig, H3Config
from ..data.manifest import load_manifest, manifest_sha256
from ..data.mot import Observation
from ..utils import sha256_file
from .protocol import FrozenProtocolError, validate_h3_protocol


class H3InputError(RuntimeError):
    """Raised when H3 would cross a frozen split or artifact boundary."""


@dataclass(frozen=True)
class H3Inputs:
    observations: tuple[Observation, ...]
    partition_by_observation: dict[str, str]
    video_partitions: dict[str, tuple[str, ...]]
    audit: dict[str, Any]

    def partition(self, name: str) -> list[Observation]:
        return [
            item
            for item in self.observations
            if self.partition_by_observation[item.observation_id] == name
        ]


def require_h3(config: ExperimentConfig) -> H3Config:
    if config.h3 is None:
        raise H3InputError("This command requires an h3 section in the YAML config")
    return config.h3


def read_project_split(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise H3InputError(f"Frozen project split does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise H3InputError(f"Invalid frozen project split {path}: {error}") from error
    if not isinstance(value, dict) or value.get("status") != "FROZEN":
        raise H3InputError(f"H3 requires a project split with status FROZEN: {path}")
    try:
        partitions = value["partitions"]
        for name in ("project_train", "development_validation", "final_test"):
            videos = partitions[name]["video_ids"]
            if not isinstance(videos, list) or not videos:
                raise TypeError(name)
    except (KeyError, TypeError) as error:
        raise H3InputError("Frozen project split is missing required video partitions") from error
    return value


def _subset_observations(
    observations: Sequence[Observation], config: ExperimentConfig
) -> list[Observation]:
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


def validate_h3_inputs(config: ExperimentConfig) -> H3Inputs:
    h3 = require_h3(config)
    try:
        protocol_audit = validate_h3_protocol(
            h3.protocol_lock_path, h3.protocol_checksum_path
        )
    except FrozenProtocolError as error:
        raise H3InputError(str(error)) from error
    split = read_project_split(h3.project_split_path)
    configured_split = h3.project_split_path.resolve(strict=False)
    protocol_split = Path(str(protocol_audit["project_split"])).resolve(strict=False)
    if configured_split != protocol_split:
        raise H3InputError(
            "h3.project_split must be the exact split referenced by the frozen protocol"
        )
    if sha256_file(configured_split) != protocol_audit["project_split_sha256"]:
        raise H3InputError("Frozen project split hash disagrees with H3 protocol audit")
    if protocol_audit["final_test_access"] is not False:
        raise H3InputError("H3 development must keep final_test_access=false")

    if not config.h3_manifest_path.is_file():
        raise H3InputError(
            f"Completed H1 manifest is missing: {config.h3_manifest_path}"
        )
    h1_root = config.h3_manifest_path.parents[1]
    required = (h1_root / "run_metadata.json", h1_root / "resolved_split.yaml")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise H3InputError("H3 requires completed H1 provenance: " + ", ".join(missing))
    metadata = json.loads((h1_root / "run_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise H3InputError(f"Invalid H1 run metadata: {h1_root / 'run_metadata.json'}")

    partition_values = split["partitions"]
    video_partitions = {
        name: tuple(str(item) for item in partition_values[name]["video_ids"])
        for name in ("project_train", "development_validation", "final_test")
    }
    train_and_development = set(video_partitions["project_train"]) | set(
        video_partitions["development_validation"]
    )
    final_videos = set(video_partitions["final_test"])
    current_manifest_sha256 = manifest_sha256(config.h3_manifest_path)
    frozen_manifest_sha256 = split.get("provenance", {}).get("manifest_sha256")
    if current_manifest_sha256 != frozen_manifest_sha256:
        raise H3InputError(
            "H3 manifest differs from frozen project-split provenance: "
            f"expected {frozen_manifest_sha256}, got {current_manifest_sha256}"
        )
    all_observations = load_manifest(config.h3_manifest_path, valid_only=True)
    forbidden = sorted({item.video_id for item in all_observations} & final_videos)
    # A completed H1 manifest may contain official test rows, but H3 never selects them.
    selected = [
        item for item in all_observations if item.video_id in train_and_development
    ]
    selected = _subset_observations(selected, config)
    if not selected:
        raise H3InputError("H3 selection contains no project_train/development observations")
    if any(
        not math.isclose(
            item.crop_expansion,
            config.protocol.crop_expansion,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for item in selected
    ):
        raise H3InputError(
            "H3 configured crop expansion differs from the frozen H1 manifest"
        )
    observed_videos = {item.video_id for item in selected}
    if observed_videos & final_videos:
        raise H3InputError("H3 selected final-test observations before the freeze gate")
    if not h3.allow_subset and observed_videos != train_and_development:
        missing_videos = sorted(train_and_development - observed_videos)
        extra_videos = sorted(observed_videos - train_and_development)
        details = []
        if missing_videos:
            details.append("missing=" + ",".join(missing_videos))
        if extra_videos:
            details.append("unexpected=" + ",".join(extra_videos))
        raise H3InputError("Full H3 input video inventory mismatch: " + "; ".join(details))

    train_videos = set(video_partitions["project_train"])
    development_videos = set(video_partitions["development_validation"])
    partition_by_observation: dict[str, str] = {}
    for item in selected:
        if item.video_id in train_videos:
            partition_by_observation[item.observation_id] = "project_train"
        elif item.video_id in development_videos:
            partition_by_observation[item.observation_id] = "development_validation"
        else:  # pragma: no cover - guarded by selection above
            raise H3InputError(f"Observation is outside frozen H3 partitions: {item.video_id}")
    selected.sort(key=lambda item: (item.video_id, item.frame, item.track_id, item.observation_id))
    observed_train = sorted(observed_videos & train_videos)
    observed_development = sorted(observed_videos & development_videos)
    audit = {
        "status": "valid",
        "stage": h3.stage,
        "role": "h3_development_only",
        "allow_subset": h3.allow_subset,
        "manifest": str(config.h3_manifest_path),
        "manifest_sha256": current_manifest_sha256,
        "h1_metadata_sha256": sha256_file(h1_root / "run_metadata.json"),
        "protocol_sha256": protocol_audit["protocol_sha256"],
        "project_split_sha256": protocol_audit["project_split_sha256"],
        "observation_count": len(selected),
        "project_train_observation_count": sum(
            value == "project_train" for value in partition_by_observation.values()
        ),
        "development_observation_count": sum(
            value == "development_validation"
            for value in partition_by_observation.values()
        ),
        "project_train_videos": observed_train,
        "development_videos": observed_development,
        "manifest_contains_unselected_final_videos": forbidden,
        "selected_final_test_videos": [],
        "final_test_read": False,
    }
    return H3Inputs(
        tuple(selected), partition_by_observation, video_partitions, audit
    )
