"""Strict validator for the post-H4-v1 frozen exploratory H4.1 protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..h4.protocol import H4ProtocolError, validate_h4_protocol
from ..utils import sha256_file


class H41ProtocolError(RuntimeError):
    """Raised when the H4.1 lock or a pinned upstream artifact is unsafe."""


H41_VARIANTS = (
    "immediate_commit",
    "fixed_lag_frozen_memory_h5",
    "fixed_lag_isolated_memory_h5",
    "adaptive_lag_frozen_memory_h5",
    "adaptive_lag_isolated_memory_h5",
)
H41_PRIMARY_VARIANT = "adaptive_lag_isolated_memory_h5"


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise H41ProtocolError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise H41ProtocolError(f"{label} must equal {expected!r}; got {value!r}")


def _resolve_reference(protocol_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise H41ProtocolError(f"{label} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise H41ProtocolError(f"{label} must be a safe repository-relative path")
    candidates = (protocol_path.parent.parent / relative, protocol_path.parent / relative)
    resolved = next((item for item in candidates if item.is_file()), None)
    if resolved is None:
        raise H41ProtocolError(f"Referenced file does not exist for {label}: {value}")
    return resolved.resolve(strict=False)


def validate_h41_protocol(
    path: Path, checksum_path: Path | None = None
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=False)
    if not path.is_file():
        raise H41ProtocolError(f"H4.1 protocol lock does not exist: {path}")
    try:
        document = _mapping(
            yaml.safe_load(path.read_text(encoding="utf-8")), "H4.1 protocol"
        )
    except yaml.YAMLError as error:
        raise H41ProtocolError(f"Invalid H4.1 protocol YAML {path}: {error}") from error
    _exact(document.get("schema_version"), 1, "schema_version")
    _exact(document.get("status"), "FROZEN_EXPLORATORY_DEVELOPMENT", "status")
    _exact(
        document.get("design_timing"),
        "DEFINED_AFTER_OBSERVING_H4_V1_DEVELOPMENT_AUDIT",
        "design_timing",
    )
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

    source = _mapping(document.get("source_h4_v1"), "source_h4_v1")
    _exact(source.get("required_run_status"), "stopped_after_recoverability_audit", "source_h4_v1.required_run_status")
    _exact(source.get("required_gate_status"), "STOP_NO_RECOVERABILITY_SIGNAL", "source_h4_v1.required_gate_status")
    _exact(source.get("reuse_policy"), "read_only_provenance_and_exact_horizon_crosscheck", "source_h4_v1.reuse_policy")
    _exact(source.get("preserved_conclusion"), "H4_V1_EXACT_T_PLUS_10_GATE_FAILED", "source_h4_v1.preserved_conclusion")
    expected_artifacts = [
        "h4_run_metadata.json",
        "h4_recoverability_metadata.json",
        "h4_protocol_decision.json",
        "h4_recoverability_baseline_assignments.csv",
        "h4_recoverability_events.csv",
    ]
    _exact(source.get("required_artifacts"), expected_artifacts, "source_h4_v1.required_artifacts")
    h4_protocol = _resolve_reference(path, source.get("protocol_path"), "source_h4_v1.protocol_path")
    h4_checksum = _resolve_reference(path, source.get("protocol_checksum_path"), "source_h4_v1.protocol_checksum_path")
    try:
        h4_audit = validate_h4_protocol(h4_protocol, h4_checksum)
    except H4ProtocolError as error:
        raise H41ProtocolError(f"Pinned H4-v1 protocol is invalid: {error}") from error
    _exact(h4_audit.get("final_test_access"), False, "source H4-v1 final_test_access")

    model = _mapping(document.get("model_policy"), "model_policy")
    _exact(model.get("primary_backbones"), ["resnet50", "dinov3"], "model_policy.primary_backbones")
    _exact(model.get("official_topic_agw"), "REFERENCE_ONLY_NOT_A_METHOD_COMPONENT", "model_policy.official_topic_agw")
    _exact(model.get("trainable_parameters"), 0, "model_policy.trainable_parameters")

    oracle = _mapping(document.get("window_recoverability_audit"), "window_recoverability_audit")
    _exact(oracle.get("exact_horizons"), list(range(1, 11)), "window_recoverability_audit.exact_horizons")
    _exact(oracle.get("cumulative_deadlines"), [1, 3, 5, 10], "window_recoverability_audit.cumulative_deadlines")
    _exact(oracle.get("gate_deadline"), 5, "window_recoverability_audit.gate_deadline")
    _exact(oracle.get("estimand"), "ever_uniquely_correct_joint_assignment_at_any_exact_lag_up_to_deadline", "window_recoverability_audit.estimand")
    _exact(oracle.get("denominator"), "all_baseline_idsw_events", "window_recoverability_audit.denominator")
    _exact(oracle.get("ground_truth_use"), "OFFLINE_ORACLE_DIAGNOSTIC_ONLY", "window_recoverability_audit.ground_truth_use")
    _exact(oracle.get("method_input_use"), "FORBIDDEN", "window_recoverability_audit.method_input_use")

    tracker = _mapping(document.get("adaptive_hypothesis_tracker"), "adaptive_hypothesis_tracker")
    _exact(tracker.get("event_trigger"), "local_assignment_ambiguity", "adaptive_hypothesis_tracker.event_trigger")
    _exact(tracker.get("early_commit_rule"), "stable_winner_and_normalized_margin_at_least_threshold", "adaptive_hypothesis_tracker.early_commit_rule")
    _exact(tracker.get("deadline_rule"), "force_best_branch_at_horizon", "adaptive_hypothesis_tracker.deadline_rule")
    _exact(tracker.get("branch_memory_policy"), "copy_on_branch_commit_winner_only", "adaptive_hypothesis_tracker.branch_memory_policy")
    _exact(tracker.get("causality"), "no_frame_after_recorded_decision_frame", "adaptive_hypothesis_tracker.causality")
    _exact(tracker.get("ground_truth_decision_input"), "FORBIDDEN", "adaptive_hypothesis_tracker.ground_truth_decision_input")
    _exact(document.get("association_variants"), list(H41_VARIANTS), "association_variants")
    _exact(document.get("primary_variant"), H41_PRIMARY_VARIANT, "primary_variant")

    stop_go = _mapping(document.get("stop_go"), "stop_go")
    recovery_gate = _mapping(
        stop_go.get("cumulative_recoverability_gate"),
        "stop_go.cumulative_recoverability_gate",
    )
    _exact(recovery_gate.get("failure_action"), "STOP_BEFORE_METHOD_EVALUATION", "cumulative gate action")
    method_gate = _mapping(stop_go.get("method_gate"), "stop_go.method_gate")
    _exact(method_gate.get("require_positive_idsw_reduction_each_primary_backbone"), True, "method IDSW gate")
    _exact(method_gate.get("require_idf1_and_hota_noninferiority_each_primary_backbone"), True, "method noninferiority gate")
    final_test = _mapping(document.get("final_test"), "final_test")
    _exact(final_test.get("access"), False, "final_test.access")
    _exact(final_test.get("runs_allowed_in_h41_development"), 0, "final_test.runs_allowed_in_h41_development")

    digest = sha256_file(path)
    sidecar = checksum_path or path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise H41ProtocolError(f"H4.1 protocol checksum sidecar does not exist: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if digest != expected:
        raise H41ProtocolError(
            f"H4.1 protocol checksum mismatch: expected {expected}, got {digest}"
        )

    parameters = {
        "audit_horizons": oracle.get("exact_horizons"),
        "cumulative_deadlines": oracle.get("cumulative_deadlines"),
        "gate_deadline": oracle.get("gate_deadline"),
        "min_cumulative_recoverable_fraction": recovery_gate.get("min_joint_recoverable_fraction"),
        "min_events_per_model": recovery_gate.get("min_events_per_model"),
        "min_videos_with_events": recovery_gate.get("min_videos_with_events"),
        "max_decision_horizon": tracker.get("max_decision_horizon"),
        "min_decision_lag": tracker.get("min_decision_lag"),
        "decision_margin": tracker.get("decision_margin"),
        "winner_stability_steps": tracker.get("winner_stability_steps"),
        "noninferiority_tolerance": method_gate.get("noninferiority_tolerance"),
        "min_nonharmed_videos": method_gate.get("min_nonharmed_videos"),
    }
    if any(not isinstance(value, (int, float, list)) for value in parameters.values()):
        raise H41ProtocolError("H4.1 protocol contains an invalid locked parameter")
    return {
        "status": "valid",
        "protocol_id": document.get("protocol_id"),
        "protocol_sha256": digest,
        "design_timing": document.get("design_timing"),
        "project_split": str(split_path),
        "project_split_sha256": sha256_file(split_path),
        "source_h4_protocol_sha256": h4_audit["protocol_sha256"],
        "source_h3_protocol_sha256": h4_audit["source_h3_protocol_sha256"],
        "parameters": parameters,
        "variants": list(H41_VARIANTS),
        "primary_variant": H41_PRIMARY_VARIANT,
        "final_test_access": False,
        "final_test_runs_allowed": 0,
    }
