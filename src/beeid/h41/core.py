"""Leakage-safe H4.1 inputs with immutable H4-v1 provenance."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExperimentConfig, H41Config
from ..h4.core import H4Inputs, load_h4_source_embeddings, validate_h4_inputs
from ..utils import sha256_file
from .protocol import H41ProtocolError, validate_h41_protocol


class H41InputError(RuntimeError):
    """Raised when H4.1 would change provenance or cross the final-test boundary."""


@dataclass(frozen=True)
class H41Inputs:
    h4_inputs: H4Inputs
    source_h4_root: Path
    source_baseline_rows: tuple[dict[str, str], ...]
    source_event_rows: tuple[dict[str, str], ...]
    audit: dict[str, Any]

    @property
    def observations(self):  # type: ignore[no-untyped-def]
        return self.h4_inputs.observations

    @property
    def video_partitions(self) -> dict[str, tuple[str, ...]]:
        return self.h4_inputs.video_partitions


def require_h41(config: ExperimentConfig) -> H41Config:
    if config.h41 is None:
        raise H41InputError("This command requires an h41 section in the YAML config")
    return config.h41


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise H41InputError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise H41InputError(f"{label} must be a JSON object: {path}")
    return value


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise H41InputError(f"Cannot read {label} {path}: {error}") from error


def _locked_parameters(h41: H41Config) -> dict[str, Any]:
    return {
        "audit_horizons": list(h41.audit_horizons),
        "cumulative_deadlines": list(h41.cumulative_deadlines),
        "gate_deadline": h41.gate_deadline,
        "min_cumulative_recoverable_fraction": h41.min_cumulative_recoverable_fraction,
        "min_events_per_model": h41.min_events_per_model,
        "min_videos_with_events": h41.min_videos_with_events,
        "max_decision_horizon": h41.max_decision_horizon,
        "min_decision_lag": h41.min_decision_lag,
        "decision_margin": h41.decision_margin,
        "winner_stability_steps": h41.winner_stability_steps,
        "noninferiority_tolerance": h41.noninferiority_tolerance,
        "min_nonharmed_videos": h41.min_nonharmed_videos,
    }


def validate_h41_inputs(config: ExperimentConfig) -> H41Inputs:
    h41 = require_h41(config)
    if config.h4 is None:
        raise H41InputError("H4.1 requires the unchanged frozen h4 section")
    try:
        protocol = validate_h41_protocol(
            h41.protocol_lock_path, h41.protocol_checksum_path
        )
    except H41ProtocolError as error:
        raise H41InputError(str(error)) from error
    mismatches = [
        key
        for key, value in _locked_parameters(h41).items()
        if protocol["parameters"].get(key) != value
    ]
    if mismatches:
        raise H41InputError(
            "H4.1 YAML differs from the frozen exploratory protocol: "
            + ", ".join(mismatches)
        )
    if h41.allow_subset != config.h4.allow_subset:
        raise H41InputError("h41.allow_subset must equal h4.allow_subset")
    configured_split = h41.project_split_path.resolve(strict=False)
    if configured_split != Path(protocol["project_split"]).resolve(strict=False):
        raise H41InputError("h41.project_split must be the exact split referenced by the lock")
    if sha256_file(configured_split) != protocol["project_split_sha256"]:
        raise H41InputError("Frozen project split hash disagrees with the H4.1 protocol")

    # Reuse the already strict H4/H3/cache split validation. This does not execute
    # H4-v1 or write any source artifact.
    h4_inputs = validate_h4_inputs(config)
    if h4_inputs.audit.get("source_h3_protocol_sha256") != protocol["source_h3_protocol_sha256"]:
        raise H41InputError("H3 protocol provenance differs from the H4.1 lock")
    if h4_inputs.audit.get("project_split_sha256") != protocol["project_split_sha256"]:
        raise H41InputError("H4 input split provenance differs from H4.1")

    source_root = config.paths.h4_output_root
    assert source_root is not None
    required = (
        "h4_run_metadata.json",
        "h4_recoverability_metadata.json",
        "h4_protocol_decision.json",
        "h4_recoverability_baseline_assignments.csv",
        "h4_recoverability_events.csv",
    )
    missing = [str(source_root / name) for name in required if not (source_root / name).is_file()]
    if missing:
        raise H41InputError("Completed H4-v1 artifacts are missing: " + ", ".join(missing))
    run = _load_json(source_root / "h4_run_metadata.json", "H4-v1 run metadata")
    recovery = _load_json(
        source_root / "h4_recoverability_metadata.json", "H4-v1 recoverability metadata"
    )
    decision = _load_json(source_root / "h4_protocol_decision.json", "H4-v1 decision")
    if run.get("status") != "stopped_after_recoverability_audit":
        raise H41InputError("H4.1 requires the preserved H4-v1 stopped audit")
    if decision.get("status") != "STOP_NO_RECOVERABILITY_SIGNAL" or decision.get("gate_passed") is not False:
        raise H41InputError("H4.1 requires the original H4-v1 STOP decision")
    if recovery.get("status") != "completed":
        raise H41InputError("H4-v1 recoverability metadata is incomplete")
    for label, artifact in (
        ("H4-v1 run metadata", run),
        ("H4-v1 recoverability metadata", recovery),
        ("H4-v1 decision", decision),
    ):
        if artifact.get("final_test_read") is not False:
            raise H41InputError(f"{label} does not prove final_test_read=false")
    if run.get("protocol_id") != "bee24-h4-deferred-identity-frozen-development-v1":
        raise H41InputError("H4-v1 run metadata has an unexpected protocol_id")
    run_input = run.get("input_audit")
    if not isinstance(run_input, dict):
        raise H41InputError("H4-v1 run metadata has no input audit")
    expected_source = {
        "protocol_sha256": protocol["source_h4_protocol_sha256"],
        "project_split_sha256": protocol["project_split_sha256"],
        "source_h3_protocol_sha256": protocol["source_h3_protocol_sha256"],
        "manifest_sha256": h4_inputs.audit["manifest_sha256"],
        "final_test_read": False,
    }
    source_mismatches = [
        key for key, value in expected_source.items() if run_input.get(key) != value
    ]
    if source_mismatches:
        raise H41InputError(
            "H4-v1 provenance is incompatible with H4.1: " + ", ".join(source_mismatches)
        )
    hashes = run.get("result_hashes")
    if not isinstance(hashes, dict):
        raise H41InputError("H4-v1 run metadata has no result hashes")
    if hashes.get("recoverability_events") != sha256_file(
        source_root / "h4_recoverability_events.csv"
    ):
        raise H41InputError("H4-v1 recoverability event file changed after reporting")
    if hashes.get("protocol_decision") != sha256_file(
        source_root / "h4_protocol_decision.json"
    ):
        raise H41InputError("H4-v1 protocol decision changed after reporting")

    baseline_rows = _read_csv(
        source_root / "h4_recoverability_baseline_assignments.csv",
        "H4-v1 baseline assignments",
    )
    event_rows = _read_csv(
        source_root / "h4_recoverability_events.csv", "H4-v1 recoverability events"
    )
    if not baseline_rows or not event_rows:
        raise H41InputError("H4-v1 source tables must not be empty")
    if {int(row["horizon"]) for row in event_rows} != {1, 3, 5, 10}:
        raise H41InputError("H4-v1 source event table does not contain the frozen horizons")
    if any(row.get("final_test_read") != "False" for row in event_rows):
        raise H41InputError("H4-v1 event rows do not prove final_test_read=false")

    audit = {
        **h4_inputs.audit,
        "status": "valid",
        "role": "h41_exploratory_development_only",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "design_timing": protocol["design_timing"],
        "source_h4_root": str(source_root),
        "source_h4_protocol_sha256": protocol["source_h4_protocol_sha256"],
        "source_h4_run_metadata_sha256": sha256_file(source_root / "h4_run_metadata.json"),
        "source_h4_recoverability_metadata_sha256": sha256_file(
            source_root / "h4_recoverability_metadata.json"
        ),
        "source_h4_decision_sha256": sha256_file(source_root / "h4_protocol_decision.json"),
        "source_h4_baseline_sha256": sha256_file(
            source_root / "h4_recoverability_baseline_assignments.csv"
        ),
        "source_h4_events_sha256": sha256_file(source_root / "h4_recoverability_events.csv"),
        "source_h4_gate_status": decision["status"],
        "source_h4_conclusion_preserved": True,
        "oracle_gt_use": "OFFLINE_DIAGNOSTIC_ONLY",
        "method_gt_identity_input": False,
        "final_test_read": False,
    }
    return H41Inputs(
        h4_inputs,
        source_root,
        tuple(baseline_rows),
        tuple(event_rows),
        audit,
    )


def load_h41_source_embeddings(
    config: ExperimentConfig, model_name: str, inputs: H41Inputs | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = inputs or validate_h41_inputs(config)
    values, audit = load_h4_source_embeddings(
        config, model_name, selected.h4_inputs
    )
    return values, {**audit, "read_only": True, "feature_reextraction": False}
