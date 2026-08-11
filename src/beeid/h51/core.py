"""Leakage-safe, read-only H5/H3 inputs for H5.1 diagnostics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..config import ExperimentConfig, H51Config
from ..h5.core import H5Inputs, load_h5_source_embeddings, validate_h5_inputs
from ..h5.model import BeeTrackQuery
from ..h5.tracker import TRACKER_IMPLEMENTATION
from ..utils import canonical_json, sha256_file, sha256_text
from . import H51_MODELS
from .protocol import H51ProtocolError, validate_h51_protocol


class H51InputError(RuntimeError):
    """Raised when diagnostic provenance or the final-test boundary is invalid."""


@dataclass(frozen=True)
class H51Inputs:
    h5_inputs: H5Inputs
    source_h5_root: Path
    source_metadata: dict[str, Any]
    checkpoints: dict[str, dict[str, Any]]
    audit: dict[str, Any]

    @property
    def observations(self):  # type: ignore[no-untyped-def]
        return self.h5_inputs.h3_inputs.observations

    def development_indices(self) -> list[int]:
        return self.h5_inputs.indices("development_validation")


def require_h51(config: ExperimentConfig) -> H51Config:
    if config.h51 is None:
        raise H51InputError("H5.1 commands require an h51 config section")
    return config.h51


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise H51InputError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise H51InputError(f"{label} must be a JSON object: {path}")
    return value


def _false(value: dict[str, Any], label: str) -> None:
    if value.get("final_test_read") is not False:
        raise H51InputError(f"{label} does not prove final_test_read=false")


def _locked_parameters(h51: H51Config) -> dict[str, Any]:
    return {
        "collapse_new_track_rate": h51.collapse_new_track_rate,
        "collapse_prediction_inflation": h51.collapse_prediction_inflation,
        "gate_degeneracy_low": h51.gate_degeneracy_low,
        "gate_degeneracy_high": h51.gate_degeneracy_high,
        "min_gate_updates": h51.min_gate_updates,
        "max_teacher_rollout_idf1_gap": h51.max_teacher_rollout_idf1_gap,
    }


def validate_h51_inputs(config: ExperimentConfig) -> H51Inputs:
    h51 = require_h51(config)
    if config.h5 is None:
        raise H51InputError("H5.1 requires the unchanged h5 config section")
    try:
        protocol = validate_h51_protocol(h51.protocol_lock_path, h51.protocol_checksum_path)
    except H51ProtocolError as error:
        raise H51InputError(str(error)) from error
    mismatches = [
        key for key, expected in _locked_parameters(h51).items()
        if protocol["thresholds"].get(key) != expected
    ]
    if mismatches:
        raise H51InputError("H5.1 config differs from its diagnostic lock: " + ", ".join(mismatches))
    expected_teacher_rows = (
        protocol["max_teacher_forced_rows_subset_smoke"]
        if h51.allow_subset
        else protocol["max_teacher_forced_rows_full"]
    )
    output_mismatches: list[str] = []
    if h51.max_teacher_forced_rows != expected_teacher_rows:
        output_mismatches.append("max_teacher_forced_rows")
    if h51.candidate_sample_per_observation != protocol["candidate_sample_per_observation"]:
        output_mismatches.append("candidate_sample_per_observation")
    if output_mismatches:
        raise H51InputError(
            "H5.1 config differs from its locked full/subset diagnostic output contract: "
            + ", ".join(output_mismatches)
        )
    if h51.allow_subset != config.h5.allow_subset:
        raise H51InputError("h51.allow_subset must equal h5.allow_subset")
    if h51.project_split_path.resolve(strict=False) != Path(protocol["project_split"]).resolve(strict=False):
        raise H51InputError("h51.project_split must be the exact split pinned by the protocol")
    if sha256_file(h51.project_split_path) != protocol["project_split_sha256"]:
        raise H51InputError("H5.1 project split checksum mismatch")

    # This is read-only validation of H3 manifests, split and cache provenance.
    h5_inputs = validate_h5_inputs(config)
    if h5_inputs.audit.get("h5_protocol_sha256") != protocol["source_h5_protocol_sha256"]:
        raise H51InputError("Configured H5 protocol differs from the H5.1 pinned source protocol")
    if h5_inputs.audit.get("protocol_sha256") != protocol["source_h3_protocol_sha256"]:
        raise H51InputError("Configured H3 protocol differs from the H5.1 pinned source protocol")
    if h5_inputs.audit.get("project_split_sha256") != protocol["project_split_sha256"]:
        raise H51InputError("H5 source input split differs from H5.1")

    source_root = config.paths.h5_output_root
    assert source_root is not None
    required = (
        "h5_run_metadata.json", "h5_training_metadata.json", "h5_tracking_metadata.json",
        "h5_method_decision.json", "h5_assignments.csv", "h5_summary.csv",
    )
    missing = [str(source_root / name) for name in required if not (source_root / name).is_file()]
    if missing:
        raise H51InputError("Completed H5 artifacts are missing: " + ", ".join(missing))
    run = _json(source_root / "h5_run_metadata.json", "H5 run metadata")
    training = _json(source_root / "h5_training_metadata.json", "H5 training metadata")
    tracking = _json(source_root / "h5_tracking_metadata.json", "H5 tracking metadata")
    decision = _json(source_root / "h5_method_decision.json", "H5 decision")
    for label, artifact in (("H5 run metadata", run), ("H5 training metadata", training),
                            ("H5 tracking metadata", tracking), ("H5 decision", decision)):
        _false(artifact, label)
    if run.get("status") != "completed_development_gt_boxes":
        raise H51InputError("H5.1 requires completed H5 development GT-box output")
    if decision.get("status") != "STOP_OR_REVISE_BEETRACKQUERY" or decision.get("gate_passed") is not False:
        raise H51InputError("H5.1 is the post-failure diagnostic for the preserved H5 STOP")
    if training.get("status") != "completed" or tracking.get("status") != "completed_development_gt_boxes":
        raise H51InputError("H5 training/tracking metadata is incomplete")
    expected_models = set(H51_MODELS)
    for label, artifact in (("H5 run", run), ("H5 training", training), ("H5 tracking", tracking)):
        if set(artifact.get("models", [])) != expected_models:
            raise H51InputError(f"{label} metadata does not contain exactly both primary models")
    if tracking.get("tracker_implementation") != TRACKER_IMPLEMENTATION:
        raise H51InputError("H5 source tracker implementation is not the pinned tracker-v4 runtime")
    if tracking.get("gt_identity_used_for_decisions") is not False:
        raise H51InputError("H5 source tracking metadata does not prove GT-free inference decisions")
    if run.get("decision") != decision:
        raise H51InputError("H5 run metadata decision differs from h5_method_decision.json")
    input_audit = run.get("input_audit")
    if not isinstance(input_audit, dict):
        raise H51InputError("H5 run metadata has no input_audit")
    expected_input = {
        "h5_protocol_sha256": protocol["source_h5_protocol_sha256"],
        "protocol_sha256": protocol["source_h3_protocol_sha256"],
        "project_split_sha256": protocol["project_split_sha256"],
        "manifest_sha256": h5_inputs.audit["manifest_sha256"],
        "final_test_read": False,
    }
    bad = [key for key, expected in expected_input.items() if input_audit.get(key) != expected]
    if bad:
        raise H51InputError("H5 run provenance differs from H5.1 inputs: " + ", ".join(bad))
    hashes = run.get("result_hashes")
    if not isinstance(hashes, dict):
        raise H51InputError("H5 run metadata has no result hashes")
    source_hashes = {
        "assignments": sha256_file(source_root / "h5_assignments.csv"),
        "summary": sha256_file(source_root / "h5_summary.csv"),
        "decision": sha256_file(source_root / "h5_method_decision.json"),
    }
    changed = [key for key, digest in source_hashes.items() if hashes.get(key) != digest]
    if changed:
        raise H51InputError("H5 result artifacts changed after reporting: " + ", ".join(changed))

    h3_root = config.paths.h3_output_root
    assert h3_root is not None
    h3_metadata_path = h3_root / "h3_run_metadata.json"
    h3_assignment_path = h3_root / "h3_assignments.csv"
    h3_metadata = _json(h3_metadata_path, "H3 run metadata")
    _false(h3_metadata, "H3 run metadata")
    if h3_metadata.get("status") != "completed_development_gt_boxes":
        raise H51InputError("Frozen H3 baseline metadata is incomplete")
    h3_assignment_sha256 = sha256_file(h3_assignment_path)
    if h3_metadata.get("assignment_sha256") != h3_assignment_sha256:
        raise H51InputError("Frozen H3 baseline assignments changed after reporting")

    checkpoint_audit: dict[str, dict[str, Any]] = {}
    training_checkpoints = training.get("checkpoints")
    if not isinstance(training_checkpoints, dict):
        raise H51InputError("H5 training metadata has no checkpoint registry")
    for model_name in H51_MODELS:
        checkpoint = source_root / "h5_checkpoints" / model_name / "final.pt"
        metadata_path = checkpoint.parent / "checkpoint.json"
        if not checkpoint.is_file() or not metadata_path.is_file():
            raise H51InputError(f"H5 final checkpoint/metadata is missing for {model_name}")
        metadata = _json(metadata_path, f"H5 {model_name} checkpoint metadata")
        _false(metadata, f"H5 {model_name} checkpoint metadata")
        digest = sha256_file(checkpoint)
        if metadata.get("status") != "completed" or metadata.get("checkpoint_sha256") != digest:
            raise H51InputError(f"H5 {model_name} checkpoint checksum/status mismatch")
        registry = training_checkpoints.get(model_name)
        if not isinstance(registry, dict):
            raise H51InputError(f"H5 training registry has no {model_name} checkpoint")
        _false(registry, f"H5 training registry {model_name} checkpoint")
        if registry.get("fingerprint") != metadata.get("fingerprint") or registry.get("checkpoint_sha256") != digest:
            raise H51InputError(f"H5 {model_name} fingerprint differs across metadata")
        signature = metadata.get("signature")
        if not isinstance(signature, dict):
            raise H51InputError(f"H5 {model_name} checkpoint has no signature")
        expected_signature = {
            "implementation": "beeid.h5:v2-batched-transitions",
            "model": model_name,
            "h5_protocol_sha256": protocol["source_h5_protocol_sha256"],
            "project_split_sha256": protocol["project_split_sha256"],
            "manifest_sha256": h5_inputs.audit["manifest_sha256"],
            "fit_partition": "project_train",
            "final_test_read": False,
        }
        bad_signature = [key for key, expected in expected_signature.items() if signature.get(key) != expected]
        if bad_signature:
            raise H51InputError(f"H5 {model_name} checkpoint provenance mismatch: " + ", ".join(bad_signature))
        computed_fingerprint = sha256_text(canonical_json(signature))
        if metadata.get("fingerprint") != computed_fingerprint:
            raise H51InputError(f"H5 {model_name} fingerprint does not authenticate its signature")
        if registry.get("signature") != signature:
            raise H51InputError(f"H5 {model_name} signature differs across checkpoint metadata")
        checkpoint_audit[model_name] = {
            "path": str(checkpoint), "sha256": digest,
            "fingerprint": metadata["fingerprint"], "signature": signature,
            "read_only": True,
        }
    audit = {
        **h5_inputs.audit,
        "status": "valid",
        "role": "h51_post_h5_closed_loop_diagnostic_development_only",
        "protocol_id": protocol["protocol_id"],
        "h51_protocol_sha256": protocol["protocol_sha256"],
        "source_h5_root": str(source_root),
        "source_h5_run_metadata_sha256": sha256_file(source_root / "h5_run_metadata.json"),
        "source_h5_training_metadata_sha256": sha256_file(source_root / "h5_training_metadata.json"),
        "source_h5_tracking_metadata_sha256": sha256_file(source_root / "h5_tracking_metadata.json"),
        "source_h5_decision_sha256": source_hashes["decision"],
        "source_h5_gate_status": decision["status"],
        "source_h5_project_git_commit": run.get("project_git_commit"),
        "source_h5_failure_diagnosis": True,
        "gradient_probe_provenance": {
            "required_checkpoint_implementation": "beeid.h5:v2-batched-transitions",
            "runtime_probe": "current_branch_recreation_of_pinned_h5_v2_batched_transition_loss",
            "historical_training_binary_directly_attested": False,
            "limitation": "legacy_h5_metadata_records_commit_but_no_training_source_diff_or_binary_digest",
        },
        "checkpoint_retraining": False,
        "checkpoint_modification": False,
        "source_h5_checkpoints": checkpoint_audit,
        "source_h3_baseline": {
            "run_metadata_path": str(h3_metadata_path),
            "run_metadata_sha256": sha256_file(h3_metadata_path),
            "assignments_path": str(h3_assignment_path),
            "assignments_sha256": h3_assignment_sha256,
            "variant": "baseline_association",
            "models": list(H51_MODELS),
            "read_only": True,
        },
        "read_partitions": {
            "project_train": "runtime_gradient_probe_only",
            "development_validation": "teacher_forced_offline_audit_causal_rollout_and_metrics",
        },
        "runtime_gradient_probe_partition": protocol["runtime_gradient_probe_partition"],
        "teacher_forced_offline_partition": protocol["teacher_forced_offline_partition"],
        "causal_rollout_partition": protocol["causal_rollout_partition"],
        "metrics_partition": protocol["metrics_partition"],
        "replay_diagnostic_schema_version": protocol["replay_diagnostic_schema_version"],
        "gradient_health_ready_rule": protocol["gradient_health_ready_rule"],
        "diagnostic_output_limits": {
            "max_teacher_forced_rows": h51.max_teacher_forced_rows,
            "teacher_forced_rows_scope": (
                "bounded_subset_smoke_override" if h51.allow_subset else "locked_full_development"
            ),
            "candidate_sample_per_observation": h51.candidate_sample_per_observation,
        },
        "diagnostic_not_confirmatory": True,
        "final_test_read": False,
    }
    return H51Inputs(h5_inputs, source_root, run, checkpoint_audit, audit)


def load_h51_embeddings(
    config: ExperimentConfig, model_name: str, inputs: H51Inputs
) -> tuple[np.ndarray, dict[str, Any]]:
    values, audit = load_h5_source_embeddings(config, model_name, inputs.h5_inputs)
    checkpoint_cache = inputs.checkpoints[model_name]["signature"].get("source_cache_fingerprint")
    if checkpoint_cache != audit.get("cache_fingerprint"):
        raise H51InputError(
            f"H5 {model_name} checkpoint was trained from a different H3 cache fingerprint"
        )
    return values, {**audit, "read_only": True, "feature_reextraction": False}


def load_h51_model(
    config: ExperimentConfig, model_name: str, embedding_dim: int, inputs: H51Inputs,
    device: torch.device,
) -> BeeTrackQuery:
    h5 = config.h5
    assert h5 is not None
    model = BeeTrackQuery(embedding_dim, h5.hidden_dim, h5.num_heads, h5.dropout).to(device)
    checkpoint = Path(inputs.checkpoints[model_name]["path"])
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    return model


def load_frozen_h3_baseline(
    config: ExperimentConfig, model_name: str, development_ids: set[str]
) -> list[dict[str, str]]:
    root = config.paths.h3_output_root
    assert root is not None
    path = root / "h3_assignments.csv"
    metadata = _json(root / "h3_run_metadata.json", "H3 run metadata")
    _false(metadata, "H3 run metadata")
    if metadata.get("status") != "completed_development_gt_boxes":
        raise H51InputError("Frozen H3 baseline metadata is incomplete")
    if metadata.get("assignment_sha256") != sha256_file(path):
        raise H51InputError("Frozen H3 baseline assignments changed after reporting")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise H51InputError(f"Cannot read frozen H3 assignments {path}: {error}") from error
    selected = [
        row for row in rows
        if row.get("model") == model_name
        and row.get("variant") == "baseline_association"
        and row.get("observation_id") in development_ids
    ]
    selected_ids = [str(row.get("observation_id", "")) for row in selected]
    if (
        len(selected) != len(development_ids)
        or len(set(selected_ids)) != len(selected_ids)
        or set(selected_ids) != development_ids
    ):
        raise H51InputError(
            f"Frozen H3 baseline must contain exactly one row per development observation for {model_name}"
        )
    return [{**row, "variant": "frozen_h3_baseline"} for row in selected]
