"""Leakage-safe H6 inputs and immutable H3 feature/baseline reuse."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..cache import open_cache
from ..config import ExperimentConfig, H6Config
from ..h3.core import H3Inputs, validate_h3_inputs
from ..h3.features import h3_observation_ids_sha256
from ..h3.signals import read_h3_signal_rows
from ..h3.thresholds import load_h3_thresholds
from ..h3.tracker import track_gt_sequence
from ..utils import canonical_json, sha256_file, sha256_text
from .protocol import H6ProtocolError, validate_h6_protocol


class H6InputError(RuntimeError):
    """Raised before H6 can train, calibrate, or inspect development data."""


@dataclass(frozen=True)
class H6Inputs:
    h3_inputs: H3Inputs
    signals_by_observation: dict[str, dict[str, str]]
    fit_videos: tuple[str, ...]
    calibration_videos: tuple[str, ...]
    development_videos: tuple[str, ...]
    partition_by_observation: dict[str, str]
    audit: dict[str, Any]

    @property
    def observations(self):  # type: ignore[no-untyped-def]
        return self.h3_inputs.observations

    def indices(self, partition: str) -> list[int]:
        return [
            index for index, item in enumerate(self.observations)
            if self.partition_by_observation[item.observation_id] == partition
        ]


def require_h6(config: ExperimentConfig) -> H6Config:
    if config.h6 is None:
        raise H6InputError("H6 commands require an h6 section in the YAML config")
    return config.h6


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise H6InputError(f"Cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise H6InputError(f"{label} must be a JSON object: {path}")
    return value


def locked_h6_parameters(value: H6Config) -> dict[str, Any]:
    return {
        "baseline_variant": value.baseline_variant,
        "window_length": value.window_length, "window_stride": value.window_stride,
        "max_frame_gap": value.max_frame_gap, "hidden_dim": value.hidden_dim,
        "num_heads": value.num_heads, "num_layers": value.num_layers,
        "feedforward_dim": value.feedforward_dim, "dropout": value.dropout,
        "epochs": value.epochs, "learning_rate": value.learning_rate,
        "weight_decay": value.weight_decay, "gradient_clip_norm": value.gradient_clip_norm,
        "max_tokens_per_batch": value.max_tokens_per_batch,
        "max_train_windows_per_video": value.max_train_windows_per_video,
        "negative_positive_ratio": value.negative_positive_ratio,
        "association_loss_weight": value.association_loss_weight,
        "contrastive_loss_weight": value.contrastive_loss_weight,
        "cycle_loss_weight": value.cycle_loss_weight,
        "calibration_fraction": value.calibration_fraction,
        "min_oracle_edge_recall": value.min_oracle_edge_recall,
        "min_calibration_precision": value.min_calibration_precision,
        "max_calibration_harm": value.max_calibration_harm,
        "min_calibration_interventions": value.min_calibration_interventions,
        "min_calibration_videos": value.min_calibration_videos,
        "min_assignment_probability": value.min_assignment_probability,
        "noninferiority_tolerance": value.noninferiority_tolerance,
        "min_nonharmed_videos": value.min_nonharmed_videos,
        "checkpoint_interval_batches": value.checkpoint_interval_batches,
        "random_seed": value.random_seed,
    }


def _calibration_videos(videos: Sequence[str], seed: int, fraction: float) -> tuple[str, ...]:
    ordered = sorted(
        videos,
        key=lambda video: (hashlib.sha256(f"{seed}:{video}".encode()).hexdigest(), video),
    )
    count = max(1, round(fraction * len(ordered)))
    if count >= len(ordered):
        count = len(ordered) - 1
    if count <= 0:
        raise H6InputError("H6 requires at least two project_train videos for fit/calibration isolation")
    return tuple(sorted(ordered[:count]))


def validate_h6_inputs(config: ExperimentConfig) -> H6Inputs:
    h6 = require_h6(config)
    if config.h3 is None:
        raise H6InputError("H6 requires the frozen H3 configuration")
    try:
        protocol = validate_h6_protocol(h6.protocol_lock_path, h6.protocol_checksum_path)
    except H6ProtocolError as error:
        raise H6InputError(str(error)) from error
    mismatches = [
        key for key, value in locked_h6_parameters(h6).items()
        if protocol["parameters"].get(key) != value
    ]
    # Subset smoke may shrink compute and gate-count parameters, but is permanently
    # marked non-scientific and can never make method_ready=true in the report.
    immutable_subset = {
        "baseline_variant", "window_length", "window_stride", "max_frame_gap",
        "negative_positive_ratio", "association_loss_weight", "contrastive_loss_weight",
        "cycle_loss_weight", "calibration_fraction", "min_assignment_probability",
        "random_seed",
    }
    unsafe_subset_mismatches = [key for key in mismatches if key in immutable_subset]
    if mismatches and (not h6.allow_subset or unsafe_subset_mismatches):
        raise H6InputError("H6 YAML differs from frozen protocol: " + ", ".join(mismatches))
    if h6.allow_subset != config.h3.allow_subset:
        raise H6InputError("h6.allow_subset must equal h3.allow_subset")
    if h6.project_split_path.resolve(strict=False) != Path(protocol["project_split"]).resolve(strict=False):
        raise H6InputError("h6.project_split must be the exact split referenced by the protocol")
    if sha256_file(h6.project_split_path) != protocol["project_split_sha256"]:
        raise H6InputError("H6 project split checksum differs from the frozen protocol")

    h3_inputs = validate_h3_inputs(config)
    if h3_inputs.audit["protocol_sha256"] != protocol["source_h3_protocol_sha256"]:
        raise H6InputError("H3 protocol provenance differs from the H6 lock")
    source = config.paths.h3_output_root
    assert source is not None
    required = (
        "h3_run_metadata.json", "h3_tracking_metadata.json", "h3_observation_signals.csv",
        "h3_thresholds.json", "h3_cache_locations.json",
    )
    missing = [str(source / name) for name in required if not (source / name).is_file()]
    if missing:
        raise H6InputError("H6 requires completed read-only H3 artifacts: " + ", ".join(missing))
    run = _json(source / "h3_run_metadata.json", "H3 run metadata")
    tracking = _json(source / "h3_tracking_metadata.json", "H3 tracking metadata")
    if run.get("status") != "completed_development_gt_boxes" or tracking.get("status") != "completed":
        raise H6InputError("H6 requires a completed H3 GT-box development run")
    if run.get("final_test_read") is not False or tracking.get("final_test_read") is not False:
        raise H6InputError("H3 provenance does not prove final_test_read=false")

    signals = read_h3_signal_rows(source / "h3_observation_signals.csv")
    signal_lookup = {str(row["observation_id"]): row for row in signals}
    wanted = {item.observation_id for item in h3_inputs.observations}
    if not wanted <= set(signal_lookup):
        raise H6InputError("H3 signal table is missing selected H6 observations")
    if not h6.allow_subset and set(signal_lookup) != wanted:
        raise H6InputError("Full H6 requires H3 signals to exactly match frozen train/development observations")
    signal_lookup = {identifier: signal_lookup[identifier] for identifier in wanted}

    original_train = set(h3_inputs.video_partitions["project_train"])
    observed_train = sorted({item.video_id for item in h3_inputs.observations} & original_train)
    observed_development = tuple(sorted(
        {item.video_id for item in h3_inputs.observations}
        & set(h3_inputs.video_partitions["development_validation"])
    ))
    if len(observed_train) < 2 or not observed_development:
        raise H6InputError("H6 selection needs at least two train videos and one development video")
    calibration = _calibration_videos(observed_train, h6.random_seed, h6.calibration_fraction)
    calibration_set = set(calibration)
    fit = tuple(sorted(set(observed_train) - calibration_set))
    partition: dict[str, str] = {}
    for item in h3_inputs.observations:
        if item.video_id in fit:
            partition[item.observation_id] = "project_train_fit"
        elif item.video_id in calibration_set:
            partition[item.observation_id] = "project_train_calibration"
        elif item.video_id in observed_development:
            partition[item.observation_id] = "development_validation"
        else:
            raise H6InputError(f"H6 selected an observation outside its partitions: {item.observation_id}")
    split_signature = sha256_text(canonical_json({
        "algorithm": "sha256_seed_video_group_holdout", "seed": h6.random_seed,
        "fraction": h6.calibration_fraction, "fit": fit, "calibration": calibration,
        "development": observed_development,
    }))
    audit = {
        **h3_inputs.audit, "status": "valid",
        "role": "h6_project_train_fit_calibration_development_evaluation",
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol["protocol_sha256"],
        "source_h3_protocol_sha256": protocol["source_h3_protocol_sha256"],
        "fit_videos": list(fit), "calibration_videos": list(calibration),
        "development_videos": list(observed_development),
        "subpartition_sha256": split_signature,
        "source_h3_root": str(source),
        "source_h3_run_metadata_sha256": sha256_file(source / "h3_run_metadata.json"),
        "source_h3_tracking_metadata_sha256": sha256_file(source / "h3_tracking_metadata.json"),
        "source_h3_signals_sha256": sha256_file(source / "h3_observation_signals.csv"),
        "source_h3_thresholds_sha256": sha256_file(source / "h3_thresholds.json"),
        "feature_reextraction": False, "final_test_read": False,
        "temporary_subset_overrides": mismatches if h6.allow_subset else [],
    }
    return H6Inputs(
        h3_inputs, signal_lookup, fit, calibration, observed_development, partition, audit
    )


def load_h6_embeddings(
    config: ExperimentConfig, model_name: str, inputs: H6Inputs | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = inputs or validate_h6_inputs(config)
    source = config.paths.h3_output_root
    assert source is not None
    registry = _json(source / "h3_cache_locations.json", "H3 cache registry")
    location = registry.get(model_name)
    if not isinstance(location, str) or not location:
        raise H6InputError(f"H3 cache registry has no {model_name} entry")
    cache = open_cache(Path(location), f"h3__{model_name}")
    signature = cache.signature
    expected = {
        "experiment": "H3_RAM_Bee", "implementation": "beeid.h3.features:v1",
        "manifest_sha256": selected.audit["manifest_sha256"],
        "protocol_sha256": selected.audit["source_h3_protocol_sha256"],
        "project_split_sha256": selected.audit["project_split_sha256"],
        "crop_expansion": config.protocol.crop_expansion,
        "input_size": config.protocol.input_size, "final_test_read": False,
    }
    if not require_h6(config).allow_subset:
        expected["observation_ids_sha256"] = h3_observation_ids_sha256(selected.h3_inputs)
    mismatch = [key for key, value in expected.items() if signature.get(key) != value]
    model = signature.get("model")
    if not isinstance(model, dict) or model.get("name") != model_name:
        mismatch.append("model")
    if mismatch:
        raise H6InputError(f"H3 cache signature mismatch for {model_name}: {', '.join(mismatch)}")
    ids = [item.observation_id for item in selected.observations]
    embeddings = cache.load_selected(ids, require_exact_ids=not require_h6(config).allow_subset)
    if embeddings.dtype != np.float32 or embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise H6InputError("Read-only H3 embeddings are invalid")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise H6InputError("Read-only H3 embeddings are not L2-normalized")
    return embeddings, {
        "directory": str(cache.directory), "fingerprint": cache.fingerprint,
        "signature": signature, "read_only": True, "feature_reextraction": False,
        "selected_observation_count": len(ids), "final_test_read": False,
    }


def h3_source_config(config: ExperimentConfig) -> ExperimentConfig:
    source = config.paths.h3_output_root
    assert source is not None
    return replace(config, paths=replace(config.paths, output_root=source))


def run_frozen_baseline(
    config: ExperimentConfig,
    inputs: H6Inputs,
    model_name: str,
    observations: Sequence[Any],
    embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    """Recompute the exact frozen H3 baseline without reading GT decisions."""
    source_config = h3_source_config(config)
    thresholds = load_h3_thresholds(source_config, [model_name])
    signals = [inputs.signals_by_observation[item.observation_id] for item in observations]
    return track_gt_sequence(
        model_name, require_h6(config).baseline_variant, observations, embeddings,
        signals, thresholds, config.h3,  # type: ignore[arg-type]
    )
