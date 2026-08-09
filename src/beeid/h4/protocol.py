"""Validator for the frozen H4 development protocol and its H3 dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..h3.protocol import FrozenProtocolError, validate_h3_protocol
from ..utils import sha256_file


class H4ProtocolError(RuntimeError):
    """Raised when the H4 protocol lock or its upstream lock is unsafe."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise H4ProtocolError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise H4ProtocolError(f"{label} must equal {expected!r}; got {value!r}")


def _resolve_reference(protocol_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise H4ProtocolError(f"{label} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise H4ProtocolError(f"{label} must be a safe repository-relative path")
    candidates = (protocol_path.parent.parent / relative, protocol_path.parent / relative)
    resolved = next((item for item in candidates if item.is_file()), None)
    if resolved is None:
        raise H4ProtocolError(f"Referenced file does not exist for {label}: {value}")
    return resolved.resolve(strict=False)


def validate_h4_protocol(path: Path, checksum_path: Path | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=False)
    if not path.is_file():
        raise H4ProtocolError(f"H4 protocol lock does not exist: {path}")
    try:
        document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "H4 protocol")
    except yaml.YAMLError as error:
        raise H4ProtocolError(f"Invalid H4 protocol YAML {path}: {error}") from error
    _exact(document.get("schema_version"), 1, "schema_version")
    _exact(document.get("status"), "FROZEN_DEVELOPMENT", "status")
    _exact(document.get("identity_definition"), ["video_id", "track_id"], "identity_definition")
    _exact(document.get("isolation_unit"), "video_id", "isolation_unit")

    split = _mapping(document.get("project_split"), "project_split")
    _exact(
        split.get("partitions"),
        ["project_train", "development_validation", "final_test"],
        "project_split.partitions",
    )
    _exact(split.get("evaluation_partition"), "development_validation", "project_split.evaluation_partition")
    _exact(split.get("final_test_access"), False, "project_split.final_test_access")
    _exact(split.get("final_test_excluded_from_fitting"), True, "project_split.final_test_excluded_from_fitting")
    _exact(split.get("final_test_excluded_from_selection"), True, "project_split.final_test_excluded_from_selection")
    split_path = _resolve_reference(path, split.get("path"), "project_split.path")
    split_document = _mapping(
        yaml.safe_load(split_path.read_text(encoding="utf-8")), "frozen project split"
    )
    _exact(split_document.get("status"), "FROZEN", "project split status")

    source = _mapping(document.get("source_h3"), "source_h3")
    _exact(source.get("required_status"), "completed_development_gt_boxes", "source_h3.required_status")
    _exact(source.get("reuse_policy"), "read_only_no_feature_reextraction", "source_h3.reuse_policy")
    required_artifacts = source.get("required_artifacts")
    expected_artifacts = [
        "h3_run_metadata.json", "h3_tracking_metadata.json",
        "h3_observation_signals.csv", "h3_thresholds.json", "h3_cache_locations.json",
    ]
    _exact(required_artifacts, expected_artifacts, "source_h3.required_artifacts")
    h3_protocol = _resolve_reference(path, source.get("protocol_path"), "source_h3.protocol_path")
    h3_checksum = _resolve_reference(
        path, source.get("protocol_checksum_path"), "source_h3.protocol_checksum_path"
    )
    try:
        h3_audit = validate_h3_protocol(h3_protocol, h3_checksum)
    except FrozenProtocolError as error:
        raise H4ProtocolError(f"Pinned H3 protocol is invalid: {error}") from error
    _exact(h3_audit.get("final_test_access"), False, "source H3 final_test_access")

    model = _mapping(document.get("model_policy"), "model_policy")
    _exact(model.get("primary_backbones"), ["resnet50", "dinov3"], "model_policy.primary_backbones")
    _exact(model.get("official_topic_agw"), "REFERENCE_ONLY_NOT_A_METHOD_COMPONENT", "model_policy.official_topic_agw")
    _exact(model.get("trainable_parameters"), 0, "model_policy.trainable_parameters")

    oracle = _mapping(document.get("recoverability_audit"), "recoverability_audit")
    _exact(oracle.get("horizons"), [1, 3, 5, 10], "recoverability_audit.horizons")
    _exact(oracle.get("go_horizon"), 10, "recoverability_audit.go_horizon")
    _exact(oracle.get("ground_truth_use"), "OFFLINE_ORACLE_DIAGNOSTIC_ONLY", "recoverability_audit.ground_truth_use")
    _exact(oracle.get("method_input_use"), "FORBIDDEN", "recoverability_audit.method_input_use")
    _exact(oracle.get("report_separately_from_tracker_metrics"), True, "recoverability_audit.report_separately_from_tracker_metrics")

    tracker = _mapping(document.get("hypothesis_tracker"), "hypothesis_tracker")
    _exact(tracker.get("event_trigger"), "local_assignment_ambiguity", "hypothesis_tracker.event_trigger")
    _exact(tracker.get("branch_memory_policy"), "copy_on_branch_commit_winner_only", "hypothesis_tracker.branch_memory_policy")
    _exact(tracker.get("causality"), "fixed_lag_uses_no_frames_after_decision_frame", "hypothesis_tracker.causality")
    variants = [
        "immediate_commit", "fixed_lag_frozen_memory_h5",
        "fixed_lag_isolated_memory_h1", "fixed_lag_isolated_memory_h3",
        "fixed_lag_isolated_memory_h5", "fixed_lag_isolated_memory_h10",
    ]
    _exact(document.get("association_variants"), variants, "association_variants")
    _exact(document.get("primary_variant"), "fixed_lag_isolated_memory_h5", "primary_variant")

    stop_go = _mapping(document.get("stop_go"), "stop_go")
    recovery_gate = _mapping(stop_go.get("recoverability_gate"), "stop_go.recoverability_gate")
    _exact(recovery_gate.get("failure_action"), "STOP_BEFORE_METHOD_EVALUATION", "recoverability gate action")
    method_gate = _mapping(stop_go.get("method_gate"), "stop_go.method_gate")
    _exact(method_gate.get("require_positive_idsw_reduction_each_primary_backbone"), True, "method IDSW gate")
    _exact(method_gate.get("require_idf1_and_hota_noninferiority_each_primary_backbone"), True, "method noninferiority gate")
    final_test = _mapping(document.get("final_test"), "final_test")
    _exact(final_test.get("access"), False, "final_test.access")
    _exact(final_test.get("runs_allowed_in_h4_development"), 0, "final_test.runs_allowed_in_h4_development")

    digest = sha256_file(path)
    sidecar = checksum_path or path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise H4ProtocolError(f"H4 protocol checksum sidecar does not exist: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if digest != expected:
        raise H4ProtocolError(f"H4 protocol checksum mismatch: expected {expected}, got {digest}")
    parameters = {
        "horizons": oracle.get("horizons"),
        "go_horizon": oracle.get("go_horizon"),
        "oracle_history_length": oracle.get("history_length"),
        "oracle_unique_margin": oracle.get("unique_margin"),
        "primary_horizon": tracker.get("primary_horizon"),
        "beam_width": tracker.get("beam_width"),
        "max_component_size": tracker.get("max_component_size"),
        "ambiguity_margin": tracker.get("ambiguity_margin"),
        "memory_alpha": tracker.get("memory_alpha"),
        "appearance_weight": tracker.get("appearance_weight"),
        "motion_weight": tracker.get("motion_weight"),
        "max_normalized_distance": tracker.get("max_normalized_distance"),
        "min_assignment_score": tracker.get("min_assignment_score"),
        "unmatched_penalty": tracker.get("unmatched_penalty"),
        "max_age": tracker.get("max_age"),
        "min_recoverable_fraction": recovery_gate.get("min_joint_recoverable_fraction"),
        "min_events_per_model": recovery_gate.get("min_events_per_model"),
        "min_videos_with_events": recovery_gate.get("min_videos_with_events"),
        "noninferiority_tolerance": method_gate.get("noninferiority_tolerance"),
        "min_nonharmed_videos": method_gate.get("min_nonharmed_videos"),
    }
    if any(not isinstance(value, (int, float, list)) for value in parameters.values()):
        raise H4ProtocolError("H4 protocol contains an invalid locked numeric parameter")
    return {
        "status": "valid",
        "protocol_id": document.get("protocol_id"),
        "protocol_sha256": digest,
        "project_split": str(split_path),
        "project_split_sha256": sha256_file(split_path),
        "source_h3_protocol_sha256": h3_audit["protocol_sha256"],
        "parameters": parameters,
        "variants": variants,
        "primary_variant": document.get("primary_variant"),
        "final_test_access": False,
        "final_test_runs_allowed": 0,
    }
