"""Validator for the locked SLTR development protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..h3.protocol import FrozenProtocolError, validate_h3_protocol
from ..h4.protocol import H4ProtocolError, validate_h4_protocol
from ..utils import sha256_file
from . import SLTR_MODELS, SLTR_PRIMARY_VARIANT, SLTR_VARIANTS


class SLTRProtocolError(RuntimeError):
    """Raised when an SLTR protocol lock is absent, altered, or unsafe."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SLTRProtocolError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise SLTRProtocolError(f"{label} must equal {expected!r}; got {value!r}")


def _reference(protocol: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SLTRProtocolError(f"{label} must be a non-empty repository-relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SLTRProtocolError(f"{label} must be a safe repository-relative path")
    choices = (protocol.parent.parent / candidate, protocol.parent / candidate)
    resolved = next((item for item in choices if item.is_file()), None)
    if resolved is None:
        raise SLTRProtocolError(f"Referenced file does not exist for {label}: {value}")
    return resolved.resolve(strict=False)


def validate_sltr_protocol(path: Path, checksum_path: Path | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=False)
    if not path.is_file():
        raise SLTRProtocolError(f"SLTR protocol lock does not exist: {path}")
    sidecar = checksum_path or path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise SLTRProtocolError(f"SLTR protocol checksum sidecar does not exist: {sidecar}")
    expected_digest = sidecar.read_text(encoding="utf-8").strip().split()[0]
    digest = sha256_file(path)
    if digest != expected_digest:
        raise SLTRProtocolError(
            f"SLTR protocol checksum mismatch: expected {expected_digest}, got {digest}"
        )
    try:
        document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "SLTR protocol")
    except yaml.YAMLError as error:
        raise SLTRProtocolError(f"Invalid SLTR protocol YAML {path}: {error}") from error
    _exact(document.get("schema_version"), 1, "schema_version")
    _exact(document.get("status"), "FROZEN_DEVELOPMENT", "status")
    _exact(document.get("identity_definition"), ["video_id", "track_id"], "identity_definition")
    _exact(document.get("isolation_unit"), "video_id", "isolation_unit")
    _exact(document.get("design_timing"), "DEFINED_AFTER_H4_H41_DIAGNOSTICS", "design_timing")

    split = _mapping(document.get("project_split"), "project_split")
    _exact(split.get("partitions"), ["project_train", "development_validation", "final_test"], "project_split.partitions")
    _exact(split.get("fit_partition"), "project_train", "project_split.fit_partition")
    _exact(split.get("evaluation_partition"), "development_validation", "project_split.evaluation_partition")
    _exact(split.get("final_test_access"), False, "project_split.final_test_access")
    _exact(split.get("final_test_excluded_from_fitting"), True, "project_split.final_test_excluded_from_fitting")
    _exact(split.get("final_test_excluded_from_selection"), True, "project_split.final_test_excluded_from_selection")
    split_path = _reference(path, split.get("path"), "project_split.path")
    split_document = _mapping(yaml.safe_load(split_path.read_text(encoding="utf-8")), "frozen project split")
    _exact(split_document.get("status"), "FROZEN", "project split status")

    sources = _mapping(document.get("sources"), "sources")
    _exact(sources.get("h1_manifest"), "read_only", "sources.h1_manifest")
    _exact(sources.get("h3_cache"), "read_only_no_feature_reextraction", "sources.h3_cache")
    _exact(sources.get("h3_signals"), "read_only_observable_only", "sources.h3_signals")
    h3_protocol = _reference(path, sources.get("h3_protocol_path"), "sources.h3_protocol_path")
    h3_checksum = _reference(path, sources.get("h3_protocol_checksum_path"), "sources.h3_protocol_checksum_path")
    h4_protocol = _reference(path, sources.get("h4_protocol_path"), "sources.h4_protocol_path")
    h4_checksum = _reference(path, sources.get("h4_protocol_checksum_path"), "sources.h4_protocol_checksum_path")
    try:
        h3_audit = validate_h3_protocol(h3_protocol, h3_checksum)
        h4_audit = validate_h4_protocol(h4_protocol, h4_checksum)
    except (FrozenProtocolError, H4ProtocolError) as error:
        raise SLTRProtocolError(f"Pinned upstream protocol is invalid: {error}") from error
    _exact(h4_audit.get("source_h3_protocol_sha256"), h3_audit.get("protocol_sha256"), "H4 source H3 hash")

    model = _mapping(document.get("model_policy"), "model_policy")
    _exact(model.get("primary_backbones"), list(SLTR_MODELS), "model_policy.primary_backbones")
    _exact(model.get("official_topic_agw"), "REFERENCE_ONLY_NOT_A_METHOD_COMPONENT", "model_policy.official_topic_agw")
    _exact(model.get("trainable_parameters"), "selector_only_numpy_logistic", "model_policy.trainable_parameters")

    counterfactual = _mapping(document.get("counterfactual"), "counterfactual")
    _exact(counterfactual.get("event_trigger"), "h4_local_assignment_ambiguity", "counterfactual.event_trigger")
    _exact(counterfactual.get("option_a"), "rank_0_immediate_then_rank_0_horizon", "counterfactual.option_a")
    _exact(counterfactual.get("option_b"), "rank_1_immediate_then_rank_0_horizon", "counterfactual.option_b")
    _exact(counterfactual.get("horizon"), 1, "counterfactual.horizon")
    _exact(counterfactual.get("evidence_basis"), "H41_MEDIAN_FIRST_RECOVERY_LAG_ONE", "counterfactual.evidence_basis")
    _exact(counterfactual.get("component_policy"), "one_to_one_local_component_only", "counterfactual.component_policy")
    _exact(counterfactual.get("decision_causality"), "selected_branch_state_only_no_future_decision_input", "counterfactual.decision_causality")

    labels = _mapping(document.get("offline_labels"), "offline_labels")
    _exact(labels.get("mapping"), "event_prehistory_recent_5_unique_gt_to_pred", "offline_labels.mapping")
    _exact(labels.get("history_length"), 5, "offline_labels.history_length")
    _exact(labels.get("utility"), "delta_correct_observations_minus_2_delta_idsw", "offline_labels.utility")
    _exact(labels.get("unavailable_policy"), "exclude_from_fit_and_oof", "offline_labels.unavailable_policy")
    _exact(labels.get("method_input"), "FORBIDDEN", "offline_labels.method_input")

    selector = _mapping(document.get("selector"), "selector")
    for field, expected in (
        ("algorithm", "deterministic_numpy_l2_logistic"),
        ("split", "group_by_video_oof"),
        ("threshold_data", "oof_only"),
        ("threshold_tie_break", "higher_threshold"),
        ("feature_policy", "observable_finite_no_gt_identity_label_utility_correct"),
    ):
        _exact(selector.get(field), expected, f"selector.{field}")
    locked = {
        "horizon": counterfactual.get("horizon"),
        "selector_l2": selector.get("l2"),
        "selector_iterations": selector.get("iterations"),
        "selector_learning_rate": selector.get("learning_rate"),
        "min_train_events": selector.get("min_train_events"),
        "min_train_videos": selector.get("min_train_videos"),
        "min_positive_events": selector.get("min_positive_events"),
        "min_precision": selector.get("min_precision"),
        "max_harm": selector.get("max_harm"),
        "min_selected_events": selector.get("min_selected_events"),
        "min_selected_videos": selector.get("min_selected_videos"),
        "max_intervention_fraction": selector.get("max_intervention_fraction"),
        "noninferiority_tolerance": selector.get("noninferiority_tolerance"),
        "min_nonharmed_videos": selector.get("min_nonharmed_videos"),
        "random_seed": selector.get("random_seed"),
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in locked.values()):
        raise SLTRProtocolError("SLTR protocol has an invalid locked numeric parameter")

    variants = _mapping(document.get("variants"), "variants")
    _exact(variants.get("all"), list(SLTR_VARIANTS), "variants.all")
    _exact(variants.get("primary"), SLTR_PRIMARY_VARIANT, "variants.primary")
    _exact(variants.get("oracle_role"), "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE", "variants.oracle_role")
    final_test = _mapping(document.get("final_test"), "final_test")
    _exact(final_test.get("access"), False, "final_test.access")
    _exact(final_test.get("runs_allowed_in_sltr_development"), 0, "final_test.runs_allowed_in_sltr_development")

    return {
        "status": "valid",
        "protocol_id": document.get("protocol_id"),
        "protocol_sha256": digest,
        "project_split": str(split_path),
        "project_split_sha256": sha256_file(split_path),
        "source_h3_protocol_sha256": h3_audit["protocol_sha256"],
        "source_h4_protocol_sha256": h4_audit["protocol_sha256"],
        "parameters": locked,
        "variants": list(SLTR_VARIANTS),
        "primary_variant": SLTR_PRIMARY_VARIANT,
        "final_test_access": False,
        "final_test_runs_allowed": 0,
    }
