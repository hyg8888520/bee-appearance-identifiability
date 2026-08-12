"""Oracle audit, GPU training, calibration, and development evaluation for H6."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from ..config import ExperimentConfig
from ..h3.metrics import paired_video_differences, summarize_gt_assignments
from ..h4.io import write_csv
from ..utils import atomic_write_json, canonical_json, sha256_file, sha256_text
from . import H6_IMPLEMENTATION, H6_MODELS, H6_PRIMARY_VARIANT
from .core import H6Inputs, load_h6_embeddings, require_h6, run_frozen_baseline, validate_h6_inputs
from .data import (
    build_windows, collate_windows, dense_candidate_edges, identity_triples,
    token_budget_batches,
)
from .model import (
    GlobalTrajectoryReasoner, H6ModelSpec, TRAJECTORY_FEATURE_NAMES, TrajectorySelector,
    balanced_edge_loss, cycle_consistency_loss, model_parameter_report,
    supervised_contrastive_pair_loss,
)
from .oracle import audit_candidate_reachability
from .tracker import (
    assignment_rows, attach_offline_trajectory_utility, cluster_global_edges,
    global_edge_probabilities, selector_probabilities, trajectory_feature_rows,
)


ASSIGNMENT_FIELDS = (
    "model", "variant", "stage", "video_id", "frame_id", "observation_id",
    "gt_identity", "gt_track_id", "predicted_track_id", "matched_existing_track",
    "association_score", "appearance_similarity", "motion_score",
    "normalized_motion_distance", "reliability", "effective_memory_alpha",
    "memory_update_accepted", "eligible_memory_update",
    "risk_identity_history_outlier", "risk_bbox_scale_change",
    "risk_orientation_change_proxy", "risk_sharpness_change", "risk_crowding_overlap",
    "h6_intervention_accepted", "h6_exact_baseline_fallback",
    "h6_selector_probability", "h6_selector_threshold",
    "h6_ground_truth_decision_input", "final_test_read",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device(config: ExperimentConfig) -> torch.device:
    requested = torch.device(config.runtime.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("H6 real training requested CUDA but torch.cuda.is_available() is false")
    return requested


def _amp_dtype(config: ExperimentConfig) -> torch.dtype:
    return torch.float16 if config.runtime.amp_dtype == "float16" else torch.bfloat16


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _write_progress(output: Path, stage: str, **payload: Any) -> None:
    atomic_write_json(output / "logs" / f"h6_{stage}_progress.json", {
        "stage": stage, "updated_at": _now(), "implementation": H6_IMPLEMENTATION,
        "final_test_read": False, **payload,
    })


def _groups(inputs: H6Inputs, partition: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index in inputs.indices(partition):
        result[inputs.observations[index].video_id].append(index)
    for indices in result.values():
        indices.sort(key=lambda index: (
            inputs.observations[index].frame, inputs.observations[index].center_x,
            inputs.observations[index].center_y, inputs.observations[index].observation_id,
        ))
    return dict(sorted(result.items()))


def run_h6_oracle_audit(config: ExperimentConfig) -> dict[str, Any]:
    h6 = require_h6(config)
    inputs = validate_h6_inputs(config)
    rows, summary = audit_candidate_reachability(
        inputs.observations, inputs.indices("project_train_fit"),
        window_length=h6.window_length, window_stride=h6.window_stride,
        max_frame_gap=h6.max_frame_gap,
    )
    gate = "GO_DENSE_CANDIDATE_GRAPH" if summary["edge_recall"] >= h6.min_oracle_edge_recall else "STOP_INSUFFICIENT_CANDIDATE_RECALL"
    result = {
        **summary, "gate": gate, "threshold": h6.min_oracle_edge_recall,
        "fit_videos": list(inputs.fit_videos), "input_audit": inputs.audit,
        "subset_never_method_ready": h6.allow_subset, "final_test_read": False,
    }
    write_csv(config.paths.output_root / "h6_oracle_edges.csv", rows)
    atomic_write_json(config.paths.output_root / "h6_oracle_audit.json", result)
    return result


def _model_spec(config: ExperimentConfig, embedding_dim: int) -> H6ModelSpec:
    h6 = require_h6(config)
    return H6ModelSpec(
        embedding_dim=embedding_dim, hidden_dim=h6.hidden_dim, num_heads=h6.num_heads,
        num_layers=h6.num_layers, feedforward_dim=h6.feedforward_dim,
        dropout=h6.dropout, window_length=h6.window_length,
    )


def _training_signature(
    config: ExperimentConfig, inputs: H6Inputs, model_name: str,
    cache: Mapping[str, Any], spec: H6ModelSpec,
) -> dict[str, Any]:
    h6 = require_h6(config)
    identifiers = [
        inputs.observations[index].observation_id for index in inputs.indices("project_train_fit")
    ]
    return {
        "implementation": H6_IMPLEMENTATION, "model": model_name,
        "model_spec": spec.__dict__, "protocol_sha256": inputs.audit["protocol_sha256"],
        "project_split_sha256": inputs.audit["project_split_sha256"],
        "subpartition_sha256": inputs.audit["subpartition_sha256"],
        "fit_observation_ids_sha256": sha256_text(canonical_json(identifiers)),
        "h3_cache_fingerprint": cache["fingerprint"],
        "h3_signals_sha256": inputs.audit["source_h3_signals_sha256"],
        "h3_thresholds_sha256": inputs.audit["source_h3_thresholds_sha256"],
        "h3_run_metadata_sha256": inputs.audit["source_h3_run_metadata_sha256"],
        "h3_tracking_metadata_sha256": inputs.audit["source_h3_tracking_metadata_sha256"],
        "training": {
            "epochs": h6.epochs, "learning_rate": h6.learning_rate,
            "weight_decay": h6.weight_decay, "gradient_clip_norm": h6.gradient_clip_norm,
            "max_tokens_per_batch": h6.max_tokens_per_batch,
            "max_train_windows_per_video": h6.max_train_windows_per_video,
            "negative_positive_ratio": h6.negative_positive_ratio,
            "loss_weights": [h6.association_loss_weight, h6.contrastive_loss_weight, h6.cycle_loss_weight],
            "window_length": h6.window_length, "window_stride": h6.window_stride,
            "max_frame_gap": h6.max_frame_gap, "amp": config.runtime.amp,
            "amp_dtype": config.runtime.amp_dtype, "seed": h6.random_seed,
        },
        "fit_partition": "project_train_fit", "final_test_read": False,
    }


def _load_training_checkpoint(
    path: Path, signature: Mapping[str, Any], model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("signature") != dict(signature):
        raise RuntimeError(f"Refusing incompatible H6 checkpoint reuse: {path}")
    if value.get("signature_sha256") != sha256_text(canonical_json(signature)):
        raise RuntimeError(f"H6 checkpoint signature digest mismatch: {path}")
    model.load_state_dict(value["association_model"], strict=True)
    if optimizer is not None and value.get("optimizer") is not None:
        optimizer.load_state_dict(value["optimizer"])
    if not isinstance(value.get("loss_history"), list):
        raise RuntimeError(f"H6 checkpoint has invalid loss history: {path}")
    return value


def _reuse_completed_training(
    config: ExperimentConfig, inputs: H6Inputs, model_name: str,
    embeddings: np.ndarray, cache: Mapping[str, Any], device: torch.device,
) -> dict[str, Any] | None:
    """Return a strictly authenticated completed model without retraining either network."""
    checkpoint_root = config.paths.output_root / "h6_checkpoints" / model_name
    final_path = checkpoint_root / "final.pt"
    metadata_path = checkpoint_root / "checkpoint.json"
    if not final_path.exists() and not metadata_path.exists():
        return None
    if not final_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(
            f"Incomplete H6 completed checkpoint for {model_name}; final.pt and checkpoint.json "
            "must either both exist or both be absent"
        )
    spec = _model_spec(config, embeddings.shape[1])
    signature = _training_signature(config, inputs, model_name, cache, spec)
    model = GlobalTrajectoryReasoner(spec).to(device)
    checkpoint = _load_training_checkpoint(final_path, signature, model)
    assert checkpoint is not None
    required_checkpoint = {
        "selector": dict, "association_calibration": dict, "calibration": dict,
        "training_summary": dict,
    }
    invalid = [
        key for key, expected_type in required_checkpoint.items()
        if not isinstance(checkpoint.get(key), expected_type)
    ]
    if checkpoint.get("feature_names") != list(TRAJECTORY_FEATURE_NAMES):
        invalid.append("feature_names")
    if checkpoint.get("final_test_read") is not False:
        invalid.append("final_test_read")
    if invalid:
        raise RuntimeError(
            f"Completed H6 checkpoint is missing authenticated fields for {model_name}: "
            + ", ".join(invalid)
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read H6 checkpoint metadata: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise RuntimeError(f"H6 checkpoint metadata is not an object: {metadata_path}")
    expected = {
        "status": "completed", "model": model_name, "checkpoint": str(final_path),
        "checkpoint_sha256": sha256_file(final_path), "signature": signature,
        "signature_sha256": sha256_text(canonical_json(signature)),
        "final_test_read": False,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if metadata.get("training_summary") != checkpoint.get("training_summary"):
        mismatches.append("training_summary")
    if mismatches:
        raise RuntimeError(
            f"Completed H6 checkpoint metadata mismatch for {model_name}: "
            + ", ".join(sorted(set(mismatches)))
        )
    return metadata


def _train_association_model(
    config: ExperimentConfig, inputs: H6Inputs, model_name: str,
    embeddings: np.ndarray, cache: Mapping[str, Any], device: torch.device,
) -> tuple[GlobalTrajectoryReasoner, dict[str, Any], list[dict[str, Any]]]:
    h6 = require_h6(config)
    torch.manual_seed(h6.random_seed)
    np.random.seed(h6.random_seed)
    random.seed(h6.random_seed)
    spec = _model_spec(config, embeddings.shape[1])
    model = GlobalTrajectoryReasoner(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=h6.learning_rate, weight_decay=h6.weight_decay)
    signature = _training_signature(config, inputs, model_name, cache, spec)
    checkpoint_root = config.paths.output_root / "h6_checkpoints" / model_name
    final_path, partial_path = checkpoint_root / "final.pt", checkpoint_root / "partial.pt"
    if final_path.is_file():
        final_value = _load_training_checkpoint(final_path, signature, model)
        assert final_value is not None
        return model, signature, list(final_value["loss_history"])
    resumed = _load_training_checkpoint(partial_path, signature, model, optimizer)
    start_epoch = int(resumed.get("epoch", 0)) if resumed else 0
    start_batch = int(resumed.get("next_batch", 0)) if resumed else 0
    history = list(resumed["loss_history"]) if resumed else []
    windows = build_windows(
        inputs.observations, inputs.indices("project_train_fit"), h6.window_length,
        h6.window_stride, h6.max_train_windows_per_video, h6.random_seed,
    )
    if not windows:
        raise RuntimeError("H6 training produced no fit windows")
    use_amp = config.runtime.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if resumed:
        if isinstance(resumed.get("scaler"), dict):
            scaler.load_state_dict(resumed["scaler"])
        if isinstance(resumed.get("torch_rng_state"), torch.Tensor):
            torch.set_rng_state(resumed["torch_rng_state"])
        cuda_states = resumed.get("cuda_rng_states")
        if device.type == "cuda" and isinstance(cuda_states, list):
            torch.cuda.set_rng_state_all(cuda_states)
    global_batch = sum(1 for _ in history)
    for epoch in range(start_epoch, h6.epochs):
        rng = random.Random(h6.random_seed + epoch)
        shuffled = list(windows)
        rng.shuffle(shuffled)
        model.train()
        epoch_losses: list[float] = []
        processed_batches = 0
        for batch_index, window_group in enumerate(token_budget_batches(shuffled, h6.max_tokens_per_batch)):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            batch = collate_windows(window_group, inputs.observations, embeddings, h6.window_length).to(device)
            optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device=device).manual_seed(h6.random_seed * 1_000_003 + epoch * 10_007 + batch_index)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(config), enabled=use_amp):
                encoded = model.encode(batch.embeddings, batch.geometry, batch.frame_index, batch.token_mask)
                sample_losses: list[torch.Tensor] = []
                for sample_index, identities in enumerate(batch.identities):
                    count = len(identities)
                    frames = batch.frame_index[sample_index, :count]
                    left, right, labels = dense_candidate_edges(
                        frames, identities, h6.max_frame_gap,
                        h6.negative_positive_ratio, generator,
                    )
                    if left.numel() == 0:
                        continue
                    logits = model.pair_logits(
                        encoded[sample_index, :count], batch.geometry[sample_index, :count],
                        frames, left, right,
                    )
                    association = balanced_edge_loss(logits.float(), labels)
                    contrastive = supervised_contrastive_pair_loss(
                        encoded[sample_index, :count], left, right, labels
                    )
                    triples = identity_triples(frames, identities, h6.max_frame_gap)
                    cycle = cycle_consistency_loss(
                        model, encoded[sample_index, :count], batch.geometry[sample_index, :count],
                        frames, triples,
                    )
                    sample_losses.append(
                        h6.association_loss_weight * association
                        + h6.contrastive_loss_weight * contrastive
                        + h6.cycle_loss_weight * cycle
                    )
                if not sample_losses:
                    continue
                loss = torch.stack(sample_losses).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), h6.gradient_clip_norm))
            if not np.isfinite(gradient_norm):
                raise RuntimeError("H6 training produced a non-finite gradient norm")
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(loss.detach())
            if not np.isfinite(loss_value):
                raise RuntimeError("H6 training produced a non-finite loss")
            epoch_losses.append(loss_value)
            processed_batches += 1
            history.append({
                "epoch": epoch + 1, "batch": batch_index + 1, "loss": loss_value,
                "gradient_norm": gradient_norm, "window_count": len(window_group),
                "token_count": int(batch.token_mask.sum().item()),
            })
            global_batch += 1
            if global_batch % h6.checkpoint_interval_batches == 0:
                _atomic_torch_save(partial_path, {
                    "signature": signature, "signature_sha256": sha256_text(canonical_json(signature)),
                    "epoch": epoch, "next_batch": batch_index + 1,
                    "association_model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "selector": None,
                    "scaler": scaler.state_dict(), "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "loss_history": history, "final_test_read": False,
                })
            _write_progress(
                config.paths.output_root, "training", status="running", model=model_name,
                epoch=epoch + 1, epochs=h6.epochs, batch=batch_index + 1,
                windows=len(windows), loss=loss_value,
            )
        if not epoch_losses:
            # A periodic checkpoint may be written after the final batch but
            # before the epoch-boundary checkpoint. Its next_batch equals the
            # complete deterministic batch count, so resuming this epoch has
            # nothing left to execute and may advance safely.
            total_batches = sum(1 for _ in token_budget_batches(shuffled, h6.max_tokens_per_batch))
            if not (epoch == start_epoch and start_batch >= total_batches and total_batches > 0):
                raise RuntimeError("H6 epoch contained no valid cross-frame association edges")
        _atomic_torch_save(partial_path, {
            "signature": signature, "signature_sha256": sha256_text(canonical_json(signature)),
            "epoch": epoch + 1, "next_batch": 0, "association_model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "selector": None,
            "scaler": scaler.state_dict(), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "loss_history": history, "final_test_read": False,
        })
    return model, signature, history


def _video_trajectory_rows(
    config: ExperimentConfig, inputs: H6Inputs, model_name: str,
    model: GlobalTrajectoryReasoner, embeddings: np.ndarray, indices: Sequence[int],
    device: torch.device, association_threshold: float, label_utility: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, int], float], list[int]]:
    h6 = require_h6(config)
    observations = [inputs.observations[index] for index in indices]
    values = np.asarray(embeddings[list(indices)], dtype=np.float32)
    baseline = run_frozen_baseline(config, inputs, model_name, observations, values)
    probabilities = global_edge_probabilities(
        model, observations, values, window_length=h6.window_length,
        window_stride=h6.window_stride, max_frame_gap=h6.max_frame_gap,
        device=device, amp=config.runtime.amp, amp_dtype=_amp_dtype(config),
        max_tokens_per_batch=h6.max_tokens_per_batch,
    )
    neural_ids, _ = cluster_global_edges(observations, probabilities, association_threshold)
    baseline_lookup = {str(row["observation_id"]): row for row in baseline}
    trajectories = trajectory_feature_rows(observations, neural_ids, probabilities, baseline_lookup)
    if label_utility:
        attach_offline_trajectory_utility(trajectories, observations, baseline_lookup, neural_ids)
    for row in trajectories:
        row.update({"model": model_name, "video_id": observations[0].video_id})
    return baseline, trajectories, probabilities, neural_ids


def _calibrate_association_threshold(
    config: ExperimentConfig, inputs: H6Inputs, model_name: str,
    model: GlobalTrajectoryReasoner, embeddings: np.ndarray, device: torch.device,
) -> dict[str, Any]:
    """Learn the graph edge threshold from held-out project-train videos only."""
    h6 = require_h6(config)
    probabilities: list[float] = []
    labels: list[bool] = []
    video_count = 0
    for _, indices in _groups(inputs, "project_train_calibration").items():
        observations = [inputs.observations[index] for index in indices]
        values = np.asarray(embeddings[list(indices)], dtype=np.float32)
        edge_scores = global_edge_probabilities(
            model, observations, values, window_length=h6.window_length,
            window_stride=h6.window_stride, max_frame_gap=h6.max_frame_gap,
            device=device, amp=config.runtime.amp, amp_dtype=_amp_dtype(config),
            max_tokens_per_batch=h6.max_tokens_per_batch,
        )
        for (left, right), score in edge_scores.items():
            probabilities.append(float(score))
            labels.append(observations[left].identity == observations[right].identity)
        video_count += 1
    if not probabilities or not any(labels):
        return {
            "status": "stopped_fail_closed", "gate": "STOP_NO_CALIBRATION_EDGES",
            "threshold": None, "edge_count": len(probabilities), "positive_edge_count": sum(labels),
            "video_count": video_count, "source_partition": "project_train_calibration",
            "final_test_read": False,
        }
    scores = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    candidates = sorted({
        float(value) for value in np.quantile(scores, np.linspace(0.0, 1.0, 201))
        if value >= h6.min_assignment_probability
    } | {h6.min_assignment_probability}, reverse=True)
    table: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for threshold in candidates:
        selected = scores >= threshold
        true_positive = int(np.sum(selected & truth))
        false_positive = int(np.sum(selected & ~truth))
        false_negative = int(np.sum(~selected & truth))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        row = {
            "threshold": threshold, "true_positive": true_positive,
            "false_positive": false_positive, "false_negative": false_negative,
            "precision": precision, "recall": recall, "f1": f1,
            "passes_precision": precision >= h6.min_calibration_precision,
        }
        table.append(row)
        if row["passes_precision"] and true_positive > 0:
            valid.append(row)
    chosen = max(valid, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["threshold"])), default=None)
    return {
        "status": "passed" if chosen else "stopped_fail_closed",
        "gate": "GO_CALIBRATED_ASSOCIATION" if chosen else "STOP_NO_SAFE_ASSOCIATION_THRESHOLD",
        "threshold": None if chosen is None else chosen["threshold"],
        "chosen": chosen, "table": table, "edge_count": len(probabilities),
        "positive_edge_count": sum(labels), "video_count": video_count,
        "minimum_search_probability": h6.min_assignment_probability,
        "source_partition": "project_train_calibration", "development_labels_used": False,
        "final_test_read": False,
    }


def _fit_selector(
    config: ExperimentConfig, rows: Sequence[dict[str, Any]], device: torch.device,
) -> tuple[TrajectorySelector, list[dict[str, Any]]]:
    h6 = require_h6(config)
    eligible = [row for row in rows if float(row["partition_disagreement_fraction"]) > 0.0]
    selector = TrajectorySelector(dropout=h6.dropout).to(device)
    if not eligible:
        return selector, []
    values = torch.tensor([row["features"] for row in eligible], dtype=torch.float32, device=device)
    labels = torch.tensor([float(bool(row["offline_positive"])) for row in eligible], dtype=torch.float32, device=device)
    selector.set_normalization(values)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=h6.learning_rate, weight_decay=h6.weight_decay)
    history: list[dict[str, Any]] = []
    selector.train()
    iterations = max(100, h6.epochs * 10)
    positive_weight = ((len(labels) - labels.sum()) / labels.sum().clamp_min(1.0)).clamp(1.0, 100.0)
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        logits = selector(values)
        loss = functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(selector.parameters(), h6.gradient_clip_norm))
        optimizer.step()
        if step % 10 == 0 or step + 1 == iterations:
            history.append({"step": step + 1, "loss": float(loss.detach()), "gradient_norm": gradient_norm})
    return selector, history


def _calibrate_threshold(
    config: ExperimentConfig, rows: Sequence[dict[str, Any]], scores: Sequence[float],
) -> dict[str, Any]:
    h6 = require_h6(config)
    candidates = sorted({
        float(score) for row, score in zip(rows, scores)
        if float(row["partition_disagreement_fraction"]) > 0.0
    }, reverse=True)
    valid: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    for threshold in candidates:
        selected = [
            (row, score) for row, score in zip(rows, scores)
            if float(row["partition_disagreement_fraction"]) > 0.0 and score >= threshold
        ]
        positives = sum(bool(row["offline_positive"]) for row, _ in selected)
        harms = len(selected) - positives
        precision = positives / len(selected) if selected else 0.0
        harm = harms / len(selected) if selected else 0.0
        videos = len({str(row["video_id"]) for row, _ in selected})
        entry = {
            "threshold": threshold, "selected_trajectories": len(selected),
            "positive_trajectories": positives, "harmful_trajectories": harms,
            "precision": precision, "harm_rate": harm, "selected_videos": videos,
            "passes": len(selected) >= h6.min_calibration_interventions
            and videos >= h6.min_calibration_videos
            and precision >= h6.min_calibration_precision
            and harm <= h6.max_calibration_harm,
        }
        table.append(entry)
        if entry["passes"]:
            valid.append(entry)
    chosen = max(valid, key=lambda row: (int(row["selected_trajectories"]), float(row["threshold"])), default=None)
    return {
        "status": "passed" if chosen else "stopped_fail_closed",
        "gate": "GO_CALIBRATED_SELECTOR" if chosen else "STOP_NO_SAFE_SELECTOR_THRESHOLD",
        "threshold": None if chosen is None else chosen["threshold"],
        "chosen": chosen, "candidate_count": len(candidates), "table": table,
        "constraints": {
            "min_precision": h6.min_calibration_precision,
            "max_harm": h6.max_calibration_harm,
            "min_interventions": h6.min_calibration_interventions,
            "min_videos": h6.min_calibration_videos,
        },
        "source_partition": "project_train_calibration", "development_labels_used": False,
        "final_test_read": False,
    }


def train_h6(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    if not model_names or len(set(model_names)) != len(model_names) or set(model_names) - set(H6_MODELS):
        raise ValueError("H6 training requires unique supported backbone names")
    inputs = validate_h6_inputs(config)
    oracle_path = config.paths.output_root / "h6_oracle_audit.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8")) if oracle_path.is_file() else run_h6_oracle_audit(config)
    if oracle.get("gate") != "GO_DENSE_CANDIDATE_GRAPH":
        return {"status": "stopped_after_oracle_audit", "gate": oracle.get("gate"), "final_test_read": False}
    device = _device(config)
    all_metadata: dict[str, Any] = {}
    for model_name in model_names:
        embeddings, cache = load_h6_embeddings(config, model_name, inputs)
        completed_metadata = _reuse_completed_training(
            config, inputs, model_name, embeddings, cache, device
        )
        if completed_metadata is not None:
            all_metadata[model_name] = completed_metadata
            continue
        model, signature, association_history = _train_association_model(
            config, inputs, model_name, embeddings, cache, device
        )
        edge_calibration = _calibrate_association_threshold(
            config, inputs, model_name, model, embeddings, device
        )
        association_threshold = (
            1.0 if edge_calibration["threshold"] is None
            else float(edge_calibration["threshold"])
        )
        fit_rows: list[dict[str, Any]] = []
        for video_id, indices in _groups(inputs, "project_train_fit").items():
            _, trajectories, _, _ = _video_trajectory_rows(
                config, inputs, model_name, model, embeddings, indices, device,
                association_threshold, True,
            )
            fit_rows.extend(trajectories)
        selector, selector_history = _fit_selector(config, fit_rows, device)
        eligible_fit = [
            row for row in fit_rows if float(row["partition_disagreement_fraction"]) > 0.0
        ]
        fit_labels = {bool(row["offline_positive"]) for row in eligible_fit}
        selector_fit_audit = {
            "eligible_trajectory_count": len(eligible_fit),
            "positive_trajectory_count": sum(bool(row["offline_positive"]) for row in eligible_fit),
            "negative_trajectory_count": sum(not bool(row["offline_positive"]) for row in eligible_fit),
            "video_count": len({str(row["video_id"]) for row in eligible_fit}),
            "both_classes_present": fit_labels == {False, True},
            "gate": "GO_SELECTOR_FIT" if fit_labels == {False, True} else "STOP_SELECTOR_FIT_SINGLE_OR_EMPTY_CLASS",
            "source_partition": "project_train_fit", "final_test_read": False,
        }
        calibration_rows: list[dict[str, Any]] = []
        for video_id, indices in _groups(inputs, "project_train_calibration").items():
            _, trajectories, _, _ = _video_trajectory_rows(
                config, inputs, model_name, model, embeddings, indices, device,
                association_threshold, True,
            )
            calibration_rows.extend(trajectories)
        scores = selector_probabilities(selector, calibration_rows, device)
        calibration = _calibrate_threshold(config, calibration_rows, scores)
        if selector_fit_audit["gate"] != "GO_SELECTOR_FIT":
            calibration = {
                **calibration, "status": "stopped_fail_closed",
                "gate": "STOP_SELECTOR_FIT_SINGLE_OR_EMPTY_CLASS", "threshold": None,
            }
        if edge_calibration["gate"] != "GO_CALIBRATED_ASSOCIATION":
            calibration = {
                **calibration, "status": "stopped_fail_closed",
                "gate": "STOP_ASSOCIATION_CALIBRATION_FAILED",
                "threshold": None,
                "upstream_association_gate": edge_calibration["gate"],
            }
        checkpoint_root = config.paths.output_root / "h6_checkpoints" / model_name
        final_path = checkpoint_root / "final.pt"
        training_summary = {
            "association_batches": len(association_history),
            "selector_steps": len(selector_history),
            "fit_trajectory_count": len(fit_rows),
            "calibration_trajectory_count": len(calibration_rows),
            "association_calibration": edge_calibration,
            "selector_fit_audit": selector_fit_audit,
            "calibration": calibration,
            "model_parameters": model_parameter_report(model),
            "selector_parameters": model_parameter_report(selector),
        }
        _atomic_torch_save(final_path, {
            "signature": signature, "signature_sha256": sha256_text(canonical_json(signature)),
            "epoch": require_h6(config).epochs, "association_model": model.state_dict(),
            "optimizer": None, "selector": selector.state_dict(),
            "loss_history": association_history, "selector_history": selector_history,
            "association_calibration": edge_calibration,
            "selector_fit_audit": selector_fit_audit,
            "calibration": calibration, "feature_names": list(TRAJECTORY_FEATURE_NAMES),
            "training_summary": training_summary,
            "final_test_read": False,
        })
        checkpoint_sha = sha256_file(final_path)
        metadata = {
            "status": "completed", "model": model_name, "checkpoint": str(final_path),
            "checkpoint_sha256": checkpoint_sha, "signature": signature,
            "signature_sha256": sha256_text(canonical_json(signature)),
            "association_batches": len(association_history), "selector_steps": len(selector_history),
            "fit_trajectory_count": len(fit_rows), "calibration_trajectory_count": len(calibration_rows),
            "association_calibration": edge_calibration, "calibration": calibration,
            "selector_fit_audit": selector_fit_audit,
            "model_parameters": model_parameter_report(model),
            "selector_parameters": model_parameter_report(selector),
            "device": str(device), "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "amp": config.runtime.amp and device.type == "cuda", "input_audit": inputs.audit,
            "final_test_read": False,
            "training_summary": training_summary,
        }
        atomic_write_json(checkpoint_root / "checkpoint.json", metadata)
        all_metadata[model_name] = metadata
        write_csv(config.paths.output_root / f"h6_fit_trajectories_{model_name}.csv", fit_rows)
        write_csv(config.paths.output_root / f"h6_calibration_trajectories_{model_name}.csv", [
            {**row, "selector_probability": score}
            for row, score in zip(calibration_rows, scores)
        ])
        write_csv(config.paths.output_root / f"h6_calibration_curve_{model_name}.csv", calibration["table"])
        write_csv(config.paths.output_root / f"h6_edge_calibration_curve_{model_name}.csv", edge_calibration.get("table", []))
    result = {
        "status": "completed", "models": list(model_names), "training": all_metadata,
        "oracle_gate": oracle["gate"], "fit_partition": "project_train_fit",
        "calibration_partition": "project_train_calibration",
        "development_read": False, "final_test_read": False,
    }
    atomic_write_json(config.paths.output_root / "h6_training_metadata.json", result)
    _write_progress(config.paths.output_root, "training", status="completed", models=list(model_names))
    return result


def _load_models(
    config: ExperimentConfig, inputs: H6Inputs, model_name: str,
    embeddings: np.ndarray, cache: Mapping[str, Any], device: torch.device,
) -> tuple[GlobalTrajectoryReasoner, TrajectorySelector, dict[str, Any]]:
    if _reuse_completed_training(config, inputs, model_name, embeddings, cache, device) is None:
        raise RuntimeError(f"H6 completed checkpoint is missing for {model_name}; run h6-train")
    spec = _model_spec(config, embeddings.shape[1])
    signature = _training_signature(config, inputs, model_name, cache, spec)
    path = config.paths.output_root / "h6_checkpoints" / model_name / "final.pt"
    if not path.is_file():
        raise RuntimeError(f"H6 final checkpoint is missing for {model_name}; run h6-train")
    model = GlobalTrajectoryReasoner(spec).to(device)
    checkpoint = _load_training_checkpoint(path, signature, model)
    assert checkpoint is not None
    selector_state = checkpoint.get("selector")
    if selector_state is None:
        raise RuntimeError(f"H6 checkpoint has no trained trajectory selector: {path}")
    selector = TrajectorySelector(dropout=require_h6(config).dropout).to(device)
    selector.load_state_dict(selector_state, strict=True)
    if checkpoint.get("final_test_read") is not False or checkpoint.get("feature_names") != list(TRAJECTORY_FEATURE_NAMES):
        raise RuntimeError("H6 checkpoint provenance/feature schema is invalid")
    return model, selector, checkpoint


def _signed_job(path: Path, signature: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    body = {key: value.get(key) for key in ("assignments", "decisions", "trajectories")}
    if (
        value.get("status") != "completed" or value.get("signature") != dict(signature)
        or value.get("signature_sha256") != sha256_text(canonical_json(signature))
        or value.get("payload_sha256") != sha256_text(canonical_json(body))
        or value.get("final_test_read") is not False
    ):
        return None
    return value


def track_h6(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    if not model_names or len(set(model_names)) != len(model_names) or set(model_names) - set(H6_MODELS):
        raise ValueError("H6 tracking requires unique supported backbone names")
    inputs = validate_h6_inputs(config)
    device = _device(config)
    assignments: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    trajectories_out: list[dict[str, Any]] = []
    total = len(model_names) * len(_groups(inputs, "development_validation"))
    completed = reused = 0
    _write_progress(config.paths.output_root, "tracking", status="running", completed_jobs=0, total_jobs=total)
    for model_name in model_names:
        embeddings, cache = load_h6_embeddings(config, model_name, inputs)
        model, selector, checkpoint = _load_models(config, inputs, model_name, embeddings, cache, device)
        calibration = checkpoint.get("calibration")
        if not isinstance(calibration, dict):
            raise RuntimeError("H6 checkpoint has no calibration audit")
        threshold = calibration.get("threshold")
        if threshold is not None:
            threshold = float(threshold)
        edge_calibration = checkpoint.get("association_calibration")
        if not isinstance(edge_calibration, dict) or edge_calibration.get("threshold") is None:
            association_threshold = 1.0
        else:
            association_threshold = float(edge_calibration["threshold"])
        for video_id, indices in _groups(inputs, "development_validation").items():
            observations = [inputs.observations[index] for index in indices]
            signature = {
                "implementation": H6_IMPLEMENTATION, "model": model_name, "video_id": video_id,
                "observation_ids": [item.observation_id for item in observations],
                "checkpoint_sha256": sha256_file(config.paths.output_root / "h6_checkpoints" / model_name / "final.pt"),
                "cache_fingerprint": cache["fingerprint"], "selector_threshold": threshold,
                "association_threshold": association_threshold,
                "protocol_sha256": inputs.audit["protocol_sha256"], "final_test_read": False,
            }
            job_path = config.paths.output_root / "work" / "tracking" / model_name / f"{video_id}.json"
            job = _signed_job(job_path, signature)
            if job is None:
                baseline, trajectory_rows, probabilities, neural_ids = _video_trajectory_rows(
                    config, inputs, model_name, model, embeddings, indices, device,
                    association_threshold, False,
                )
                scores = selector_probabilities(selector, trajectory_rows, device)
                video_assignments, video_decisions = assignment_rows(
                    model_name, observations, neural_ids, trajectory_rows, scores,
                    threshold, baseline,
                )
                trajectory_payload = [
                    {key: value for key, value in row.items() if key != "member_indices"}
                    | {"selector_probability": score, "partition": "development_validation"}
                    for row, score in zip(trajectory_rows, scores)
                ]
                body = {
                    "assignments": video_assignments, "decisions": video_decisions,
                    "trajectories": trajectory_payload,
                }
                job = {
                    "status": "completed", "signature": signature,
                    "signature_sha256": sha256_text(canonical_json(signature)), **body,
                    "payload_sha256": sha256_text(canonical_json(body)), "final_test_read": False,
                }
                atomic_write_json(job_path, job)
            else:
                reused += 1
            assignments.extend(job["assignments"])
            decisions.extend(job["decisions"])
            trajectories_out.extend(job["trajectories"])
            completed += 1
            _write_progress(
                config.paths.output_root, "tracking", status="running",
                completed_jobs=completed, total_jobs=total, model=model_name,
                video_id=video_id, reused_jobs=reused,
            )
    assignments.sort(key=lambda row: (
        str(row["model"]), str(row["variant"]), str(row["video_id"]),
        int(row["frame_id"]), str(row["observation_id"]),
    ))
    per_video, summary = summarize_gt_assignments(assignments)
    paired = paired_video_differences(per_video)
    write_csv(config.paths.output_root / "h6_assignments.csv", assignments, ASSIGNMENT_FIELDS)
    write_csv(config.paths.output_root / "h6_trajectory_decisions.csv", decisions)
    write_csv(config.paths.output_root / "h6_trajectory_diagnostics.csv", trajectories_out)
    write_csv(config.paths.output_root / "h6_per_video_metrics.csv", per_video)
    write_csv(config.paths.output_root / "h6_summary.csv", summary)
    write_csv(config.paths.output_root / "h6_paired_video_metrics.csv", paired)
    result = {
        "status": "completed_development_gt_boxes", "models": list(model_names),
        "assignment_rows": len(assignments), "trajectory_rows": len(trajectories_out),
        "decision_rows": len(decisions), "accepted_trajectories": sum(bool(row["accepted"]) for row in decisions),
        "exact_baseline_fallback_trajectories": sum(not bool(row["accepted"]) for row in decisions),
        "reused_jobs": reused, "input_audit": inputs.audit,
        "fixed_detector_boxes": "SERVER_VALIDATION_PENDING", "final_test_read": False,
    }
    atomic_write_json(config.paths.output_root / "h6_tracking_metadata.json", result)
    _write_progress(config.paths.output_root, "tracking", status="completed", completed_jobs=completed, total_jobs=total, reused_jobs=reused)
    return result


def run_h6_all(config: ExperimentConfig, model_names: Sequence[str], confirm_full: bool) -> dict[str, Any]:
    h6 = require_h6(config)
    full = not config.dataset.video_ids and config.dataset.max_videos is None and config.dataset.max_frames_per_video is None
    if full and not confirm_full:
        raise RuntimeError("Full H6 is gated; pass --confirm-full only after H6 smoke and server review")
    if full and set(model_names) != set(H6_MODELS):
        raise RuntimeError("Full H6 requires both resnet50 and dinov3")
    from ..utils import atomic_write_text
    atomic_write_text(
        config.paths.output_root / "h6_resolved_config.yaml",
        config.config_path.read_text(encoding="utf-8"),
    )
    oracle = run_h6_oracle_audit(config)
    if oracle["gate"] != "GO_DENSE_CANDIDATE_GRAPH":
        return {"status": "stopped_after_oracle_audit", "oracle": oracle, "final_test_read": False}
    training = train_h6(config, model_names)
    if training.get("status") != "completed":
        return training
    tracking = track_h6(config, model_names)
    from .report import generate_h6_report
    report = generate_h6_report(config, model_names)
    result = {
        "status": report["status"], "decision": report["decision"],
        "oracle": "completed", "training": "completed", "tracking": "completed",
        "models": list(model_names), "subset_never_method_ready": h6.allow_subset,
        "final_test_read": False,
    }
    atomic_write_json(config.paths.output_root / "h6_run_metadata.json", result)
    return result
