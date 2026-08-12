"""Strict validation for the frozen H6 development protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..h3.protocol import FrozenProtocolError, validate_h3_protocol
from ..utils import sha256_file
from . import H6_MODELS, H6_PRIMARY_VARIANT, H6_VARIANTS


class H6ProtocolError(RuntimeError):
    """Raised when H6 protocol provenance or scientific isolation is invalid."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise H6ProtocolError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise H6ProtocolError(f"{label} must equal {expected!r}; got {value!r}")


def _reference(protocol: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise H6ProtocolError(f"{label} must be a non-empty repository-relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise H6ProtocolError(f"{label} must be a safe repository-relative path")
    choices = (protocol.parent.parent / candidate, protocol.parent / candidate)
    resolved = next((item for item in choices if item.is_file()), None)
    if resolved is None:
        raise H6ProtocolError(f"Referenced file does not exist for {label}: {value}")
    return resolved.resolve(strict=False)


def validate_h6_protocol(path: Path, checksum_path: Path | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=False)
    if not path.is_file():
        raise H6ProtocolError(f"H6 protocol lock does not exist: {path}")
    sidecar = checksum_path or path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise H6ProtocolError(f"H6 protocol checksum sidecar does not exist: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    digest = sha256_file(path)
    if digest != expected:
        raise H6ProtocolError(f"H6 protocol checksum mismatch: expected {expected}, got {digest}")
    try:
        document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "H6 protocol")
    except yaml.YAMLError as error:
        raise H6ProtocolError(f"Invalid H6 protocol YAML {path}: {error}") from error
    _exact(document.get("schema_version"), 1, "schema_version")
    _exact(document.get("status"), "FROZEN_DEVELOPMENT", "status")
    _exact(document.get("identity_definition"), ["video_id", "track_id"], "identity_definition")
    _exact(document.get("isolation_unit"), "video_id", "isolation_unit")
    _exact(document.get("design_timing"), "DEFINED_AFTER_SLTR_V1_FAIL_CLOSED", "design_timing")

    split = _mapping(document.get("project_split"), "project_split")
    _exact(split.get("fit_partition"), "project_train_fit", "project_split.fit_partition")
    _exact(split.get("calibration_partition"), "project_train_calibration", "project_split.calibration_partition")
    _exact(split.get("evaluation_partition"), "development_validation", "project_split.evaluation_partition")
    _exact(split.get("calibration_algorithm"), "sha256_seed_video_group_holdout", "project_split.calibration_algorithm")
    _exact(split.get("final_test_access"), False, "project_split.final_test_access")
    split_path = _reference(path, split.get("path"), "project_split.path")
    split_document = _mapping(yaml.safe_load(split_path.read_text(encoding="utf-8")), "project split")
    _exact(split_document.get("status"), "FROZEN", "project split status")

    sources = _mapping(document.get("sources"), "sources")
    _exact(sources.get("h1_manifest"), "read_only", "sources.h1_manifest")
    _exact(sources.get("h3_cache"), "read_only_no_feature_reextraction", "sources.h3_cache")
    _exact(sources.get("h3_baseline"), "frozen_recomputed_from_h3_protocol", "sources.h3_baseline")
    h3_path = _reference(path, sources.get("h3_protocol_path"), "sources.h3_protocol_path")
    h3_checksum = _reference(path, sources.get("h3_protocol_checksum_path"), "sources.h3_protocol_checksum_path")
    try:
        h3_audit = validate_h3_protocol(h3_path, h3_checksum)
    except FrozenProtocolError as error:
        raise H6ProtocolError(f"Pinned H3 protocol is invalid: {error}") from error

    model = _mapping(document.get("model"), "model")
    _exact(model.get("primary_backbones"), list(H6_MODELS), "model.primary_backbones")
    _exact(model.get("architecture"), "global_frame_transformer_pair_reasoner", "model.architecture")
    _exact(model.get("candidate_policy"), "all_cross_frame_pairs_within_gap_no_top_k", "model.candidate_policy")
    _exact(model.get("gt_identity_at_inference"), False, "model.gt_identity_at_inference")
    _exact(model.get("future_context"), "offline_bidirectional_within_window", "model.future_context")

    training = _mapping(document.get("training"), "training")
    locked_names = (
        "baseline_variant", "window_length", "window_stride", "max_frame_gap",
        "hidden_dim", "num_heads", "num_layers", "feedforward_dim", "dropout",
        "epochs", "learning_rate", "weight_decay", "gradient_clip_norm",
        "max_tokens_per_batch", "max_train_windows_per_video",
        "negative_positive_ratio", "association_loss_weight",
        "contrastive_loss_weight", "cycle_loss_weight", "calibration_fraction",
        "min_oracle_edge_recall", "min_calibration_precision",
        "max_calibration_harm", "min_calibration_interventions",
        "min_calibration_videos", "min_assignment_probability",
        "noninferiority_tolerance", "min_nonharmed_videos",
        "checkpoint_interval_batches", "random_seed",
    )
    locked = {name: training.get(name) for name in locked_names}
    if any(value is None or isinstance(value, bool) for value in locked.values()):
        raise H6ProtocolError("H6 protocol is missing a locked numeric/string parameter")
    _exact(locked["baseline_variant"], "baseline_association", "training.baseline_variant")

    selection = _mapping(document.get("selection"), "selection")
    _exact(selection.get("threshold_source"), "project_train_calibration_only", "selection.threshold_source")
    _exact(selection.get("decision_input"), "observable_neural_and_baseline_partition_features_no_gt", "selection.decision_input")
    _exact(selection.get("association_threshold"), "calibrated_pair_precision_recall", "selection.association_threshold")
    _exact(selection.get("fallback"), "exact_frozen_baseline_when_gate_fails", "selection.fallback")
    _exact(selection.get("gate_unit"), "predicted_trajectory", "selection.gate_unit")
    variants = _mapping(document.get("variants"), "variants")
    _exact(variants.get("all"), list(H6_VARIANTS), "variants.all")
    _exact(variants.get("primary"), H6_PRIMARY_VARIANT, "variants.primary")
    final_test = _mapping(document.get("final_test"), "final_test")
    _exact(final_test.get("access"), False, "final_test.access")
    _exact(final_test.get("runs_allowed_in_h6_development"), 0, "final_test.runs_allowed_in_h6_development")
    return {
        "status": "valid", "protocol_id": document.get("protocol_id"),
        "protocol_sha256": digest, "project_split": str(split_path),
        "project_split_sha256": sha256_file(split_path),
        "source_h3_protocol_sha256": h3_audit["protocol_sha256"],
        "parameters": locked, "variants": list(H6_VARIANTS),
        "primary_variant": H6_PRIMARY_VARIANT, "final_test_access": False,
        "final_test_runs_allowed": 0,
    }
