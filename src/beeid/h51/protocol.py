"""Strict validator for the immutable post-H5 H5.1 diagnostic protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..h5.protocol import H5ProtocolError, validate_h5_protocol
from ..utils import sha256_file
from . import H51_VARIANTS


class H51ProtocolError(RuntimeError):
    """Raised when the H5.1 lock or a pinned upstream protocol is unsafe."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise H51ProtocolError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise H51ProtocolError(f"{label} must equal {expected!r}; got {value!r}")


def _reference(protocol_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise H51ProtocolError(f"{label} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise H51ProtocolError(f"{label} must be a safe repository-relative path")
    candidates = (protocol_path.parent.parent / relative, protocol_path.parent / relative)
    found = next((candidate for candidate in candidates if candidate.is_file()), None)
    if found is None:
        raise H51ProtocolError(f"Referenced file does not exist for {label}: {value}")
    return found.resolve(strict=False)


def validate_h51_protocol(path: Path, checksum_path: Path | None = None) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=False)
    if not source.is_file():
        raise H51ProtocolError(f"H5.1 protocol lock does not exist: {source}")
    digest = sha256_file(source)
    sidecar = checksum_path or source.with_suffix(".sha256")
    if not sidecar.is_file():
        raise H51ProtocolError(f"H5.1 checksum sidecar does not exist: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if expected != digest:
        raise H51ProtocolError(f"H5.1 protocol checksum mismatch: expected {expected}, got {digest}")
    try:
        document = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "H5.1 protocol")
    except yaml.YAMLError as error:
        raise H51ProtocolError(f"Invalid H5.1 protocol YAML {source}: {error}") from error
    _exact(document.get("schema_version"), 1, "schema_version")
    _exact(document.get("status"), "FROZEN_POST_H5_DEVELOPMENT_DIAGNOSTIC", "status")
    _exact(document.get("design_timing"), "DEFINED_AFTER_H5_DEVELOPMENT_GATE_FAILED", "design_timing")
    split = _mapping(document.get("project_split"), "project_split")
    _exact(split.get("runtime_gradient_probe_partition"), "project_train", "project_split.runtime_gradient_probe_partition")
    _exact(split.get("teacher_forced_offline_partition"), "development_validation", "project_split.teacher_forced_offline_partition")
    _exact(split.get("causal_rollout_partition"), "development_validation", "project_split.causal_rollout_partition")
    _exact(split.get("metrics_partition"), "development_validation", "project_split.metrics_partition")
    _exact(split.get("final_test_access"), False, "project_split.final_test_access")
    _exact(split.get("final_test_runs_allowed"), 0, "project_split.final_test_runs_allowed")
    split_path = _reference(source, split.get("path"), "project_split.path")

    upstream = _mapping(document.get("source_h5"), "source_h5")
    _exact(upstream.get("required_gate_status"), "STOP_OR_REVISE_BEETRACKQUERY", "source_h5.required_gate_status")
    _exact(upstream.get("retraining"), "forbidden", "source_h5.retraining")
    _exact(upstream.get("checkpoint_modification"), "forbidden", "source_h5.checkpoint_modification")
    h5_path = _reference(source, upstream.get("protocol_path"), "source_h5.protocol_path")
    h5_checksum = _reference(source, upstream.get("protocol_checksum_path"), "source_h5.protocol_checksum_path")
    try:
        h5_protocol = validate_h5_protocol(h5_path, h5_checksum)
    except H5ProtocolError as error:
        raise H51ProtocolError(f"Pinned H5 protocol is invalid: {error}") from error
    _exact(h5_protocol.get("final_test_access"), False, "source H5 final_test_access")

    diagnostics = _mapping(document.get("diagnostics"), "diagnostics")
    _exact(diagnostics.get("replay_diagnostic_schema_version"), 2, "diagnostics.replay_diagnostic_schema_version")
    _exact(
        diagnostics.get("gradient_health_ready_rule"),
        "all_trainable_parameters_grad_present_finite_and_nonzero",
        "diagnostics.gradient_health_ready_rule",
    )
    _exact(tuple(diagnostics.get("variants", [])), H51_VARIANTS, "diagnostics.variants")
    _exact(diagnostics.get("gradient_probe_data_use"), "project_train_transition_inputs_only", "diagnostics.gradient_probe_data_use")
    _exact(diagnostics.get("teacher_forced_one_step"), "development_validation_offline_gt_audit_only", "diagnostics.teacher_forced_one_step")
    _exact(diagnostics.get("causal_rollout_data_use"), "development_validation_only", "diagnostics.causal_rollout_data_use")
    _exact(diagnostics.get("max_teacher_forced_rows_full"), 250000, "diagnostics.max_teacher_forced_rows_full")
    _exact(diagnostics.get("max_teacher_forced_rows_subset_smoke"), 10000, "diagnostics.max_teacher_forced_rows_subset_smoke")
    _exact(diagnostics.get("candidate_sample_per_observation"), 3, "diagnostics.candidate_sample_per_observation")
    _exact(diagnostics.get("causal_rollout_gt_identity_decision_input"), "forbidden", "diagnostics causal boundary")
    thresholds = _mapping(document.get("thresholds"), "thresholds")
    _exact(thresholds.get("diagnostic_not_confirmatory"), True, "thresholds.diagnostic_not_confirmatory")
    decision = _mapping(document.get("decision"), "decision")
    _exact(decision.get("final_test_unlock"), "never", "decision.final_test_unlock")

    return {
        "status": "valid",
        "protocol_id": document.get("protocol_id"),
        "protocol_sha256": digest,
        "design_timing": document.get("design_timing"),
        "project_split": str(split_path),
        "project_split_sha256": sha256_file(split_path),
        "runtime_gradient_probe_partition": "project_train",
        "teacher_forced_offline_partition": "development_validation",
        "causal_rollout_partition": "development_validation",
        "metrics_partition": "development_validation",
        "replay_diagnostic_schema_version": diagnostics["replay_diagnostic_schema_version"],
        "gradient_health_ready_rule": diagnostics["gradient_health_ready_rule"],
        "max_teacher_forced_rows_full": diagnostics["max_teacher_forced_rows_full"],
        "max_teacher_forced_rows_subset_smoke": diagnostics["max_teacher_forced_rows_subset_smoke"],
        "candidate_sample_per_observation": diagnostics["candidate_sample_per_observation"],
        "source_h5_protocol_sha256": h5_protocol["protocol_sha256"],
        "source_h3_protocol_sha256": h5_protocol["source_h3_protocol_sha256"],
        "variants": list(H51_VARIANTS),
        "thresholds": dict(thresholds),
        "diagnostic_not_confirmatory": True,
        "final_test_access": False,
        "final_test_runs_allowed": 0,
    }
