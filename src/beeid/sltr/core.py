"""Leakage-safe SLTR inputs: frozen H3 data/cache/signals and H4 local mechanics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..cache import open_cache
from ..config import ExperimentConfig, SLTRConfig
from ..h3.core import H3Inputs, validate_h3_inputs
from ..h3.features import h3_observation_ids_sha256
from ..h3.signals import read_h3_signal_rows
from ..h4.core import validate_h4_inputs
from ..utils import sha256_file
from .protocol import SLTRProtocolError, validate_sltr_protocol


class SLTRInputError(RuntimeError):
    """Raised before SLTR could fit, select, or touch an unsafe source artifact."""


@dataclass(frozen=True)
class SLTRInputs:
    h3_inputs: H3Inputs
    signals_by_observation: dict[str, dict[str, str]]
    audit: dict[str, Any]

    @property
    def observations(self):  # type: ignore[no-untyped-def]
        return self.h3_inputs.observations

    def indices(self, partition: str) -> list[int]:
        return [
            index for index, item in enumerate(self.observations)
            if self.h3_inputs.partition_by_observation[item.observation_id] == partition
        ]


def require_sltr(config: ExperimentConfig) -> SLTRConfig:
    if config.sltr is None:
        raise SLTRInputError("SLTR commands require an sltr config section")
    return config.sltr


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SLTRInputError(f"Cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SLTRInputError(f"{label} must be a JSON object: {path}")
    return value


def _locked_parameters(value: SLTRConfig) -> dict[str, Any]:
    return {
        "horizon": value.horizon, "selector_l2": value.selector_l2,
        "selector_iterations": value.selector_iterations,
        "selector_learning_rate": value.selector_learning_rate,
        "min_train_events": value.min_train_events,
        "min_train_videos": value.min_train_videos,
        "min_positive_events": value.min_positive_events,
        "min_precision": value.min_precision, "max_harm": value.max_harm,
        "min_selected_events": value.min_selected_events,
        "min_selected_videos": value.min_selected_videos,
        "max_intervention_fraction": value.max_intervention_fraction,
        "noninferiority_tolerance": value.noninferiority_tolerance,
        "min_nonharmed_videos": value.min_nonharmed_videos,
        "random_seed": value.random_seed,
    }


def _protocol_provenance(protocol: dict[str, Any]) -> dict[str, Any]:
    """Keep SLTR and source protocol identities distinct in downstream audits."""
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "source_h3_protocol_sha256": protocol["source_h3_protocol_sha256"],
        "source_h4_protocol_sha256": protocol["source_h4_protocol_sha256"],
    }


def validate_sltr_inputs(config: ExperimentConfig) -> SLTRInputs:
    sltr = require_sltr(config)
    if config.h4 is None:
        raise SLTRInputError("SLTR requires the frozen H4 tracker parameters")
    try:
        protocol = validate_sltr_protocol(sltr.protocol_lock_path, sltr.protocol_checksum_path)
    except SLTRProtocolError as error:
        raise SLTRInputError(str(error)) from error
    mismatches = [
        key for key, value in _locked_parameters(sltr).items()
        if protocol["parameters"].get(key) != value
    ]
    if mismatches:
        raise SLTRInputError("SLTR YAML differs from frozen protocol: " + ", ".join(mismatches))
    if sltr.allow_subset != config.h3.allow_subset or sltr.allow_subset != config.h4.allow_subset:
        raise SLTRInputError("sltr.allow_subset must equal h3.allow_subset and h4.allow_subset")
    if sltr.project_split_path.resolve(strict=False) != Path(protocol["project_split"]).resolve(strict=False):
        raise SLTRInputError("sltr.project_split must be the exact split in the frozen protocol")
    if sha256_file(sltr.project_split_path) != protocol["project_split_sha256"]:
        raise SLTRInputError("SLTR project split checksum differs from frozen protocol")

    # H4 validation checks the unchanged local-assignment parameters and all H3
    # provenance without running any H4 job.  H3 validation supplies train+dev.
    h4_inputs = validate_h4_inputs(config)
    h3_inputs = validate_h3_inputs(config)
    if h3_inputs.audit["protocol_sha256"] != protocol["source_h3_protocol_sha256"]:
        raise SLTRInputError("H3 protocol provenance differs from the SLTR lock")
    if h4_inputs.audit["protocol_sha256"] != protocol["source_h4_protocol_sha256"]:
        raise SLTRInputError("H4 tracker protocol provenance differs from the SLTR lock")
    if h3_inputs.audit["project_split_sha256"] != protocol["project_split_sha256"]:
        raise SLTRInputError("H3 project split provenance differs from the SLTR lock")
    if not any(h3_inputs.partition_by_observation[item.observation_id] == "project_train" for item in h3_inputs.observations):
        raise SLTRInputError("SLTR requires non-empty project_train observations")
    if not any(h3_inputs.partition_by_observation[item.observation_id] == "development_validation" for item in h3_inputs.observations):
        raise SLTRInputError("SLTR requires non-empty development_validation observations")

    source_root = config.paths.h3_output_root
    assert source_root is not None
    required = ("h3_run_metadata.json", "h3_tracking_metadata.json", "h3_observation_signals.csv", "h3_cache_locations.json")
    missing = [str(source_root / name) for name in required if not (source_root / name).is_file()]
    if missing:
        raise SLTRInputError("SLTR requires completed read-only H3 artifacts: " + ", ".join(missing))
    run = _json(source_root / "h3_run_metadata.json", "H3 run metadata")
    tracking = _json(source_root / "h3_tracking_metadata.json", "H3 tracking metadata")
    if run.get("status") != "completed_development_gt_boxes" or tracking.get("status") != "completed":
        raise SLTRInputError("SLTR requires completed H3 GT-box development artifacts")
    if run.get("final_test_read") is not False or tracking.get("final_test_read") is not False:
        raise SLTRInputError("H3 artifacts do not prove final_test_read=false")
    source_audit = tracking.get("input_audit")
    if not isinstance(source_audit, dict) or source_audit.get("final_test_read") is not False:
        raise SLTRInputError("H3 tracking metadata has no valid final-test isolation audit")
    rows = read_h3_signal_rows(source_root / "h3_observation_signals.csv")
    signal_lookup = {str(row["observation_id"]): row for row in rows}
    wanted = {item.observation_id for item in h3_inputs.observations}
    if not wanted <= set(signal_lookup):
        raise SLTRInputError("Read-only H3 signal rows are missing selected train/development observations")
    if not sltr.allow_subset and set(signal_lookup) != wanted:
        raise SLTRInputError("Full SLTR H3 signal rows do not exactly match frozen train/development observations")
    signal_lookup = {identifier: signal_lookup[identifier] for identifier in wanted}
    return SLTRInputs(
        h3_inputs, signal_lookup,
        {
            **h3_inputs.audit,
            "status": "valid", "role": "sltr_train_only_fit_development_only_evaluation",
            **_protocol_provenance(protocol),
            "source_h3_root": str(source_root),
            "source_h3_run_metadata_sha256": sha256_file(source_root / "h3_run_metadata.json"),
            "source_h3_tracking_metadata_sha256": sha256_file(source_root / "h3_tracking_metadata.json"),
            "source_h3_signals_sha256": sha256_file(source_root / "h3_observation_signals.csv"),
            "fit_partition": "project_train", "evaluation_partition": "development_validation",
            "h3_cache_reuse": "read_only", "h3_signals_reuse": "read_only_observable_only",
            "final_test_read": False,
        },
    )


def load_sltr_embeddings(
    config: ExperimentConfig, model_name: str, inputs: SLTRInputs | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Open the existing H3 cache read-only; never extract or write feature shards."""
    selected = inputs or validate_sltr_inputs(config)
    source_root = config.paths.h3_output_root
    assert source_root is not None
    registry = _json(source_root / "h3_cache_locations.json", "H3 cache registry")
    location = registry.get(model_name)
    if not isinstance(location, str) or not location:
        raise SLTRInputError(f"H3 cache registry has no required backbone {model_name}")
    cache = open_cache(Path(location), f"h3__{model_name}")
    signature = cache.signature
    expected = {
        "experiment": "H3_RAM_Bee", "implementation": "beeid.h3.features:v1",
        "manifest_sha256": selected.audit["manifest_sha256"],
        "protocol_sha256": selected.audit["source_h3_protocol_sha256"],
        "project_split_sha256": selected.audit["project_split_sha256"],
        "crop_expansion": config.protocol.crop_expansion, "input_size": config.protocol.input_size,
        "final_test_read": False,
    }
    if not require_sltr(config).allow_subset:
        expected["observation_ids_sha256"] = h3_observation_ids_sha256(selected.h3_inputs)
    mismatch = [key for key, value in expected.items() if signature.get(key) != value]
    model = signature.get("model")
    if not isinstance(model, dict) or model.get("name") != model_name:
        mismatch.append("model")
    if mismatch:
        raise SLTRInputError("H3 cache signature mismatch for %s: %s" % (model_name, ", ".join(mismatch)))
    embeddings = cache.load_all(
        [item.observation_id for item in selected.observations], config.runtime.cache_shard_size
    )
    if embeddings.dtype != np.float32 or embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise SLTRInputError("Read-only H3 embeddings are invalid")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise SLTRInputError("Read-only H3 embeddings are not normalized")
    return embeddings, {
        "directory": str(cache.directory), "fingerprint": cache.fingerprint,
        "signature": signature, "selected_observation_count": len(selected.observations),
        "read_only": True, "feature_reextraction": False, "final_test_read": False,
    }
