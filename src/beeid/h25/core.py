"""Frozen protocol and completed-H2 validation for H2.5."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from ..config import ExperimentConfig, H25Config
from ..data.mot import Observation
from ..h2.core import validate_h2_inputs
from ..h3.protocol import validate_h3_protocol
from ..utils import sha256_file


class H25InputError(RuntimeError):
    """Raised when H2.5 inputs or its frozen protocol are unsafe."""


EXPECTED_EVENT_TYPES = (
    "history_outlier",
    "bbox_scale_change",
    "orientation_change_proxy",
    "sharpness_change",
    "crowding",
    "wrong_identity_control",
)
EXPECTED_STRATEGIES = (
    "oracle_correct_update",
    "unconditional_update",
    "skip_update",
    "reliability_weighted_update",
)


def require_h25(config: ExperimentConfig) -> H25Config:
    if config.h25 is None:
        raise H25InputError("This command requires an h25 section in the YAML config")
    return config.h25


def read_h25_protocol(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise H25InputError(f"H2.5 protocol lock does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise H25InputError(f"Invalid H2.5 protocol YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise H25InputError("H2.5 protocol lock must be a mapping")
    required_top = {
        "schema_version", "protocol_id", "status", "role", "source", "event_selection",
        "reliability", "memory", "reporting", "scope",
    }
    if set(value) != required_top:
        raise H25InputError(
            "H2.5 protocol top-level schema changed: "
            f"expected {sorted(required_top)}, got {sorted(value)}"
        )
    if value.get("schema_version") != 1 or value.get("status") != "FROZEN":
        raise H25InputError("H2.5 protocol must have schema_version 1 and status FROZEN")
    source = value.get("source") or {}
    if source.get("split") != "development_validation" or source.get("final_test_access") is not False:
        raise H25InputError("H2.5 is development-only and must keep final_test_access false")
    selection = value.get("event_selection") or {}
    if selection.get("outcome_blind") is not True:
        raise H25InputError("H2.5 event selection must be outcome-blind")
    if tuple(selection.get("event_types") or []) != EXPECTED_EVENT_TYPES:
        raise H25InputError("H2.5 event types differ from the frozen protocol")
    if selection.get("event_quantile") != 0.75:
        raise H25InputError("H2.5 event_quantile must remain frozen at 0.75")
    if selection.get("history_length") != 5 or selection.get("trusted_history_length") != 5:
        raise H25InputError("H2.5 history lengths must remain frozen at 5")
    if selection.get("excluded_static_signals") != ["bbox_area", "bbox_laplacian_variance"]:
        raise H25InputError("Static bbox area and absolute Laplacian must remain excluded")
    memory = value.get("memory") or {}
    if tuple(memory.get("strategies") or []) != EXPECTED_STRATEGIES:
        raise H25InputError("H2.5 must compare exactly the four frozen memory strategies")
    if memory.get("ema_alpha") != 0.2 or memory.get("recovery_horizon") != 10:
        raise H25InputError("H2.5 EMA alpha and recovery horizon differ from the frozen protocol")
    if value.get("scope", {}).get("tracker_output_claim") is not False:
        raise H25InputError("H2.5 cannot claim an end-to-end tracker result")
    return value


def h2_source_config(config: ExperimentConfig) -> ExperimentConfig:
    return replace(config, paths=replace(config.paths, output_root=config.h25_source_root))


def validate_h25_inputs(
    config: ExperimentConfig,
) -> tuple[list[Observation], dict[str, Any], dict[str, Any]]:
    h25 = require_h25(config)
    protocol = read_h25_protocol(h25.protocol_lock_path)
    h3_audit = validate_h3_protocol(h25.h3_protocol_lock_path)
    if config.h2 is None:
        raise H25InputError("H2.5 requires H2 variant definitions")
    if protocol["source"]["primary_variant"] != config.h2.primary_variant:
        raise H25InputError("Configured H2 primary variant differs from the H2.5 protocol lock")
    if h25.allow_subset != config.h2.allow_subset:
        raise H25InputError("h25.allow_subset and h2.allow_subset must match")

    source = config.h25_source_root
    required = (
        source / "h2_run_metadata.json",
        source / "h2_signal_metadata.json",
        source / "h2_observation_signals.csv",
        source / "h2_cache_locations.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise H25InputError("H2.5 requires completed H2 artifacts; missing: " + ", ".join(missing))
    metadata = json.loads((source / "h2_run_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("final_test_read") is not False:
        raise H25InputError("H2 source metadata must explicitly record final_test_read=false")
    if metadata.get("primary_variant") != config.h2.primary_variant:
        raise H25InputError("H2 source primary variant does not match H2.5")
    source_config = h2_source_config(config)
    observations, h2_audit = validate_h2_inputs(source_config)
    audit = {
        "status": "valid",
        "role": "development_validation",
        "allow_subset": h25.allow_subset,
        "observation_count": len(observations),
        "observed_video_ids": sorted({item.video_id for item in observations}),
        "h1_manifest_sha256": h2_audit["manifest_sha256"],
        "h2_source_root": str(source),
        "h2_metadata_sha256": sha256_file(source / "h2_run_metadata.json"),
        "h2_signals_sha256": sha256_file(source / "h2_observation_signals.csv"),
        "h25_protocol_sha256": sha256_file(h25.protocol_lock_path),
        "h3_protocol_sha256": h3_audit["protocol_sha256"],
        "source_h2_legacy_status": metadata.get("status"),
        "final_test_read": False,
    }
    return observations, protocol, audit
