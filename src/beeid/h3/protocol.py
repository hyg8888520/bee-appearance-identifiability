"""Strict validator for the machine-readable, final-test-locked H3 protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..utils import sha256_file


class FrozenProtocolError(RuntimeError):
    """Raised when a frozen protocol is absent, changed, or semantically unsafe."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrozenProtocolError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise FrozenProtocolError(f"{label} must equal {expected!r}; got {value!r}")


def validate_h3_protocol(path: Path, checksum_path: Path | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=False)
    if not path.is_file():
        raise FrozenProtocolError(f"H3 protocol lock does not exist: {path}")
    try:
        document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "H3 protocol")
    except yaml.YAMLError as error:
        raise FrozenProtocolError(f"Invalid H3 protocol YAML {path}: {error}") from error
    _exact(document.get("schema_version"), 1, "schema_version")
    _exact(document.get("status"), "FROZEN", "status")
    _exact(document.get("identity_definition"), ["video_id", "track_id"], "identity_definition")
    _exact(document.get("isolation_unit"), "video_id", "isolation_unit")

    split = _mapping(document.get("project_split"), "project_split")
    _exact(split.get("partitions"), ["project_train", "development_validation", "final_test"], "project_split.partitions")
    _exact(split.get("threshold_fitting_partition"), "project_train", "project_split.threshold_fitting_partition")
    _exact(split.get("final_test_access"), False, "project_split.final_test_access")
    _exact(split.get("final_test_excluded_from_fitting"), True, "project_split.final_test_excluded_from_fitting")
    _exact(split.get("final_test_excluded_from_selection"), True, "project_split.final_test_excluded_from_selection")
    split_reference = split.get("path")
    if not isinstance(split_reference, str) or not split_reference:
        raise FrozenProtocolError("project_split.path must be a non-empty relative path")
    relative = Path(split_reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise FrozenProtocolError("project_split.path must be a safe repository-relative path")
    candidates = (path.parent.parent / relative, path.parent / relative)
    split_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if split_path is None:
        raise FrozenProtocolError(f"Frozen project split does not exist: {relative}")
    try:
        split_document = _mapping(
            yaml.safe_load(split_path.read_text(encoding="utf-8")), "frozen project split"
        )
    except yaml.YAMLError as error:
        raise FrozenProtocolError(f"Invalid frozen project split {split_path}: {error}") from error
    _exact(split_document.get("status"), "FROZEN", "frozen project split status")
    partitions = _mapping(split_document.get("partitions"), "frozen project split partitions")
    _exact(sorted(partitions), ["development_validation", "final_test", "project_train"], "frozen project split partition names")
    video_sets: dict[str, set[str]] = {}
    for name in ("project_train", "development_validation", "final_test"):
        partition = _mapping(partitions.get(name), f"frozen project split {name}")
        videos = partition.get("video_ids")
        if not isinstance(videos, list) or not videos or any(not isinstance(item, str) for item in videos):
            raise FrozenProtocolError(f"frozen project split {name}.video_ids must be non-empty strings")
        video_sets[name] = set(videos)
    if any(
        video_sets[left] & video_sets[right]
        for left, right in (
            ("project_train", "development_validation"),
            ("project_train", "final_test"),
            ("development_validation", "final_test"),
        )
    ):
        raise FrozenProtocolError("Frozen project split partitions are not video-disjoint")

    model = _mapping(document.get("model_policy"), "model_policy")
    _exact(model.get("primary_backbones"), ["resnet50_imagenet1k_v2", "dinov3_vits16"], "model_policy.primary_backbones")
    _exact(model.get("official_topic_agw"), "REFERENCE_ONLY_TRAINING_OVERLAP_UNCONFIRMED", "model_policy.official_topic_agw")

    reliability = _mapping(document.get("reliability_model"), "reliability_model")
    _exact(
        reliability.get("included_features"),
        ["identity_history_outlier", "bbox_scale_change", "orientation_change_proxy", "sharpness_change", "crowding_overlap"],
        "reliability_model.included_features",
    )
    _exact(reliability.get("excluded_static_features"), ["bbox_area", "bbox_laplacian_variance"], "reliability_model.excluded_static_features")
    _exact(reliability.get("threshold_fitting_partition"), "project_train", "reliability_model.threshold_fitting_partition")
    _exact(reliability.get("causality"), "past_and_current_observations_only", "reliability_model.causality")
    _exact(document.get("stages"), ["gt_detection_boxes", "fixed_detector_boxes"], "stages")
    _exact(document.get("association_variants"), ["baseline_association", "selective_memory_update", "reliability_weighted_association", "full_ram_bee"], "association_variants")
    metrics = _mapping(document.get("metrics"), "metrics")
    _exact(metrics.get("primary"), ["AssA", "IDF1", "IDSW", "Frag", "HOTA"], "metrics.primary")
    gate = _mapping(document.get("freeze_gate"), "freeze_gate")
    _exact(gate.get("final_test_runs_allowed"), 1, "freeze_gate.final_test_runs_allowed")
    required_gate = set(gate.get("required_before_final_test") or [])
    expected_gate = {
        "method_code_frozen", "configuration_frozen", "model_checkpoints_frozen",
        "threshold_artifact_frozen", "reporting_code_frozen", "development_results_signed_off",
    }
    if required_gate != expected_gate:
        raise FrozenProtocolError("freeze_gate.required_before_final_test is incomplete or changed")
    stop_go = _mapping(document.get("stop_go"), "stop_go")
    if not stop_go.get("stop_if") or not stop_go.get("go_if"):
        raise FrozenProtocolError("stop_go must contain non-empty stop_if and go_if rules")

    digest = sha256_file(path)
    sidecar = checksum_path or path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FrozenProtocolError(f"H3 protocol checksum sidecar does not exist: {sidecar}")
    expected_digest = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if expected_digest != digest:
        raise FrozenProtocolError(
            f"H3 protocol checksum mismatch: expected {expected_digest}, got {digest}"
        )
    return {
        "status": "valid",
        "protocol_id": document.get("protocol_id"),
        "protocol_sha256": digest,
        "project_split": str(split_path.resolve(strict=False)),
        "project_split_sha256": sha256_file(split_path),
        "final_test_access": False,
        "final_test_runs_allowed_after_freeze": 1,
    }
