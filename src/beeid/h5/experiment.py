"""Resumable BeeTrackQuery training and development-only GT-box evaluation."""

from __future__ import annotations

import csv
import io
import json
import os
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..config import ExperimentConfig, H5Config
from ..data.mot import Observation
from ..h3.metrics import paired_video_differences, summarize_gt_assignments
from ..utils import atomic_write_json, atomic_write_text, canonical_json, sha256_file, sha256_text
from . import H5_MODELS, H5_PRIMARY_VARIANT, H5_VARIANTS
from .core import H5Inputs, load_h5_source_embeddings, require_h5, validate_h5_inputs
from .model import BeeTrackQuery
from .tracker import TRACKER_IMPLEMENTATION, normalized_geometry, track_sequences_batched


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = tuple(rows[0]) if rows else ()
    buffer = io.StringIO(newline="")
    if fields:
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Required H5 source table is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_mot_results(root: Path, rows: Sequence[dict[str, Any]], observations: Sequence[Observation]) -> None:
    lookup = {item.observation_id: item for item in observations}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["variant"]), str(row["video_id"]))].append(row)
    for (model, variant, video_id), selected in groups.items():
        lines: list[str] = []
        for row in sorted(selected, key=lambda value: (int(value["frame_id"]), int(value["predicted_track_id"]))):
            item = lookup[str(row["observation_id"])]
            lines.append(",".join([str(item.frame), str(row["predicted_track_id"]), f"{item.raw_x:.6f}", f"{item.raw_y:.6f}", f"{item.raw_w:.6f}", f"{item.raw_h:.6f}", "1", "-1", "-1", "-1"]))
        atomic_write_text(root / model / variant / f"{video_id}.txt", "\n".join(lines) + "\n")


def _device(config: ExperimentConfig) -> torch.device:
    device = torch.device(config.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("H5 requested CUDA but torch.cuda.is_available() is false")
    return device


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _frames(observations: Sequence[Observation], indices: Sequence[int]) -> list[list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        grouped[observations[index].frame].append(index)
    return [
        sorted(grouped[frame], key=lambda i: (observations[i].center_x, observations[i].center_y))
        for frame in sorted(grouped)
    ]


def _training_clips(inputs: H5Inputs, h5: H5Config) -> list[list[list[int]]]:
    observations = inputs.h3_inputs.observations
    by_video: dict[str, list[int]] = defaultdict(list)
    for index in inputs.indices("project_train"):
        by_video[observations[index].video_id].append(index)
    clips: list[list[list[int]]] = []
    for video_id in sorted(by_video):
        frames = _frames(observations, by_video[video_id])
        candidates = [
            frames[start : start + h5.clip_length]
            for start in range(0, max(0, len(frames) - h5.clip_length + 1), h5.train_clip_stride)
            if all(
                observations[frames[offset + 1][0]].frame
                == observations[frames[offset][0]].frame + 1
                for offset in range(start, start + h5.clip_length - 1)
            )
        ]
        if len(candidates) > h5.max_train_clips_per_video:
            positions = np.linspace(0, len(candidates) - 1, h5.max_train_clips_per_video, dtype=int)
            candidates = [candidates[int(position)] for position in positions]
        clips.extend(candidates)
    if not clips:
        raise RuntimeError("No H5 training clips can be formed from project_train")
    return clips


@dataclass(frozen=True)
class TrainingTransition:
    """One teacher-forced consecutive-frame association problem."""

    query_indices: tuple[int, ...]
    detection_indices: tuple[int, ...]
    targets: tuple[int, ...]

    @property
    def query_count(self) -> int:
        return len(self.query_indices)

    @property
    def detection_count(self) -> int:
        return len(self.detection_indices)


def _training_transitions(inputs: H5Inputs, h5: H5Config) -> list[TrainingTransition]:
    """Flatten the original clip supervision into batchable adjacent transitions."""
    observations = inputs.h3_inputs.observations
    transitions: list[TrainingTransition] = []
    for clip in _training_clips(inputs, h5):
        for previous, current in zip(clip, clip[1:]):
            target_by_identity = {
                observations[index].identity: position for position, index in enumerate(current)
            }
            query_indices = tuple(
                index for index in previous if observations[index].identity in target_by_identity
            )
            if not query_indices or not current:
                continue
            targets = tuple(
                target_by_identity[observations[index].identity] for index in query_indices
            )
            transitions.append(TrainingTransition(query_indices, tuple(current), targets))
    if not transitions:
        raise RuntimeError("H5 training produced no supervised continuing-identity transitions")
    return transitions


def _transition_batches(
    transitions: Sequence[TrainingTransition],
    order: Sequence[int],
    max_batch_size: int,
    max_pair_elements: int,
) -> list[list[int]]:
    """Greedily pad transitions without exceeding the frozen pair-tensor budget."""
    batches: list[list[int]] = []
    current: list[int] = []
    max_queries = 0
    max_detections = 0
    for raw_index in order:
        index = int(raw_index)
        transition = transitions[index]
        next_queries = max(max_queries, transition.query_count)
        next_detections = max(max_detections, transition.detection_count)
        next_count = len(current) + 1
        next_cost = next_count * next_queries * next_detections
        if current and (next_count > max_batch_size or next_cost > max_pair_elements):
            batches.append(current)
            current = []
            max_queries = 0
            max_detections = 0
        current.append(index)
        max_queries = max(max_queries, transition.query_count)
        max_detections = max(max_detections, transition.detection_count)
    if current:
        batches.append(current)
    return batches


def _batched_transition_loss(
    model: BeeTrackQuery,
    transitions: Sequence[TrainingTransition],
    batch_indices: Sequence[int],
    embeddings: np.ndarray,
    geometry: np.ndarray,
    device: torch.device,
    reliability_loss_weight: float,
) -> tuple[torch.Tensor, int, int]:
    """Run padded attention/pair scoring once, then reduce each transition independently."""
    selected = [transitions[index] for index in batch_indices]
    batch_size = len(selected)
    max_queries = max(item.query_count for item in selected)
    max_detections = max(item.detection_count for item in selected)
    embedding_dim = int(embeddings.shape[1])
    query_embeddings = np.zeros((batch_size, max_queries, embedding_dim), dtype=np.float32)
    query_geometry = np.zeros((batch_size, max_queries, 4), dtype=np.float32)
    detection_embeddings = np.zeros((batch_size, max_detections, embedding_dim), dtype=np.float32)
    detection_geometry = np.zeros((batch_size, max_detections, 4), dtype=np.float32)
    detection_padding_mask = np.ones((batch_size, max_detections), dtype=bool)
    for batch_position, item in enumerate(selected):
        query_embeddings[batch_position, : item.query_count] = embeddings[list(item.query_indices)]
        query_geometry[batch_position, : item.query_count] = geometry[list(item.query_indices)]
        detection_embeddings[batch_position, : item.detection_count] = embeddings[list(item.detection_indices)]
        detection_geometry[batch_position, : item.detection_count] = geometry[list(item.detection_indices)]
        detection_padding_mask[batch_position, : item.detection_count] = False
    query_raw = torch.from_numpy(query_embeddings).to(device=device, non_blocking=True)
    query_geom = torch.from_numpy(query_geometry).to(device=device, non_blocking=True)
    detection_raw = torch.from_numpy(detection_embeddings).to(device=device, non_blocking=True)
    detection_geom = torch.from_numpy(detection_geometry).to(device=device, non_blocking=True)
    padding_mask = torch.from_numpy(detection_padding_mask).to(device=device, non_blocking=True)
    query_tokens = model.encode_detections(query_raw, query_geom)
    detection_tokens = model.encode_detections(detection_raw, detection_geom)
    refreshed = model.refresh_batched(query_tokens, detection_tokens, padding_mask)
    geometry_delta = torch.abs(query_geom[:, :, None, :] - detection_geom[:, None, :, :])
    logits = model.pair_logits_batched(refreshed, detection_tokens, geometry_delta)
    losses: list[torch.Tensor] = []
    pair_count = 0
    prediction_count = 0
    for batch_position, item in enumerate(selected):
        sample_logits = logits[
            batch_position, : item.query_count, : item.detection_count
        ]
        targets = torch.as_tensor(item.targets, dtype=torch.long, device=device)
        association = F.cross_entropy(sample_logits, targets)
        correct = (sample_logits.argmax(dim=1) == targets).to(sample_logits.dtype)
        calibration = F.binary_cross_entropy_with_logits(
            model.reliability_logits(sample_logits), correct
        )
        losses.append(association + reliability_loss_weight * calibration)
        pair_count += item.query_count * item.detection_count
        prediction_count += item.query_count
    return torch.stack(losses).mean(), pair_count, prediction_count


def _checkpoint_signature(config: ExperimentConfig, inputs: H5Inputs, model_name: str, embedding_dim: int, cache_audit: dict[str, Any]) -> dict[str, Any]:
    h5 = require_h5(config)
    hyperparameters = {
        "hidden_dim": h5.hidden_dim, "num_heads": h5.num_heads,
        "clip_length": h5.clip_length, "memory_slots": h5.memory_slots,
        "memory_top_k": h5.memory_top_k, "dropout": h5.dropout,
        "epochs": h5.epochs, "learning_rate": h5.learning_rate,
        "weight_decay": h5.weight_decay,
        "max_train_clips_per_video": h5.max_train_clips_per_video,
        "train_clip_stride": h5.train_clip_stride,
        "max_pair_elements_per_batch": h5.max_pair_elements_per_batch,
        "checkpoint_interval_batches": h5.checkpoint_interval_batches,
        "reliability_loss_weight": h5.reliability_loss_weight,
        "update_gate": h5.update_gate, "memory_mix": h5.memory_mix,
        "max_age": h5.max_age, "min_assignment_score": h5.min_assignment_score,
        "max_normalized_distance": h5.max_normalized_distance,
    }
    return {
        "format_version": 1,
        "experiment": "H5_BeeTrackQuery",
        "implementation": "beeid.h5:v2-batched-transitions",
        "model": model_name,
        "embedding_dim": embedding_dim,
        "h5_protocol_sha256": inputs.audit["h5_protocol_sha256"],
        "project_split_sha256": inputs.audit["project_split_sha256"],
        "manifest_sha256": inputs.audit["manifest_sha256"],
        "source_cache_fingerprint": cache_audit["cache_fingerprint"],
        "hyperparameters": hyperparameters,
        "seed": config.runtime.seed,
        "fit_partition": "project_train",
        "final_test_read": False,
    }


def _autocast(config: ExperimentConfig, device: torch.device):
    dtype = torch.float16 if config.runtime.amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=config.runtime.amp and device.type == "cuda")


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_checkpoint_state() -> dict[str, Any]:
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_checkpoint_state(progress: dict[str, Any]) -> None:
    torch_state = progress.get("torch_rng_state")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())
    cuda_states = progress.get("cuda_rng_states")
    if torch.cuda.is_available() and isinstance(cuda_states, list):
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _fit_model(
    config: ExperimentConfig,
    inputs: H5Inputs,
    embeddings: np.ndarray,
    model_name: str,
    cache_audit: dict[str, Any],
) -> tuple[BeeTrackQuery, dict[str, Any]]:
    h5 = require_h5(config)
    device = _device(config)
    _seed(config.runtime.seed)
    signature = _checkpoint_signature(config, inputs, model_name, embeddings.shape[1], cache_audit)
    fingerprint = sha256_text(canonical_json(signature))
    directory = config.paths.output_root / "h5_checkpoints" / model_name
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "final.pt"
    progress_checkpoint = directory / "training-progress.pt"
    metadata_path = directory / "checkpoint.json"
    live_progress_path = config.paths.output_root / "h5_logs" / f"{model_name}_progress.json"
    model = BeeTrackQuery(embeddings.shape[1], h5.hidden_dim, h5.num_heads, h5.dropout).to(device)
    if checkpoint.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Refusing incompatible H5 checkpoint reuse for {model_name}")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        return model, {**metadata, "status": "resumed", "checkpoint": str(checkpoint)}

    transitions = _training_transitions(inputs, h5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=h5.learning_rate, weight_decay=h5.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.runtime.amp and device.type == "cuda")
    observations = inputs.h3_inputs.observations
    geometry = np.stack([normalized_geometry(item) for item in observations])
    logs: list[dict[str, Any]] = []
    start_epoch = 1
    start_batch = 0
    partial_epoch_losses: list[float] = []
    partial_pair_count = 0
    partial_prediction_count = 0
    optimizer_steps_completed = 0
    if progress_checkpoint.is_file():
        progress = torch.load(progress_checkpoint, map_location=device, weights_only=False)
        if progress.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Refusing incompatible partial H5 checkpoint reuse for {model_name}: "
                f"move the old output directory aside before starting the v2 batched trainer"
            )
        model.load_state_dict(progress["model"], strict=True)
        optimizer.load_state_dict(progress["optimizer"])
        if scaler.is_enabled() and progress.get("scaler") is not None:
            scaler.load_state_dict(progress["scaler"])
        _restore_rng_checkpoint_state(progress)
        logs = list(progress.get("training_log", []))
        start_epoch = int(progress.get("current_epoch", int(progress["completed_epoch"]) + 1))
        start_batch = int(progress.get("next_batch_index", 0))
        partial_epoch_losses = [float(value) for value in progress.get("current_epoch_losses", [])]
        partial_pair_count = int(progress.get("current_epoch_pair_count", 0))
        partial_prediction_count = int(progress.get("current_epoch_prediction_count", 0))
        optimizer_steps_completed = int(
            progress.get(
                "optimizer_steps_completed",
                sum(int(row.get("batch_updates", 0)) for row in logs) + start_batch,
            )
        )
    model.train()
    training_started = time.monotonic()
    for epoch in range(start_epoch, h5.epochs + 1):
        order = np.random.default_rng(config.runtime.seed + epoch).permutation(len(transitions))
        batches = _transition_batches(
            transitions,
            order,
            config.runtime.batch_size,
            h5.max_pair_elements_per_batch,
        )
        first_batch = start_batch if epoch == start_epoch else 0
        if first_batch > len(batches):
            raise RuntimeError(f"Invalid H5 partial checkpoint batch index for {model_name}")
        losses = list(partial_epoch_losses) if epoch == start_epoch else []
        pair_count = partial_pair_count if epoch == start_epoch else 0
        prediction_count = partial_prediction_count if epoch == start_epoch else 0
        transitions_before_batch = sum(len(batch) for batch in batches[:first_batch])
        for batch_index in range(first_batch, len(batches)):
            batch = batches[batch_index]
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                loss, batch_pairs, batch_predictions = _batched_transition_loss(
                    model,
                    transitions,
                    batch,
                    embeddings,
                    geometry,
                    device,
                    h5.reliability_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps_completed += 1
            losses.append(float(loss.detach().cpu()))
            pair_count += batch_pairs
            prediction_count += batch_predictions
            completed_in_epoch = transitions_before_batch + sum(
                len(value) for value in batches[first_batch : batch_index + 1]
            )
            completed_total = (epoch - 1) * len(transitions) + completed_in_epoch
            elapsed = max(time.monotonic() - training_started, 1e-6)
            completed_since_resume = max(
                1,
                sum(len(value) for value in batches[first_batch : batch_index + 1]),
            )
            rate = completed_since_resume / elapsed
            remaining = h5.epochs * len(transitions) - completed_total
            live_state = {
                "status": "training",
                "model": model_name,
                "epoch": epoch,
                "epochs": h5.epochs,
                "batch_in_epoch": batch_index + 1,
                "batches_in_epoch": len(batches),
                "transitions_completed": completed_total,
                "transitions_total": h5.epochs * len(transitions),
                "mean_loss_current_epoch": float(np.mean(losses)),
                "transitions_per_second_since_resume": rate,
                "estimated_seconds_remaining": remaining / rate if rate > 0 else None,
                "checkpoint_every_batches": h5.checkpoint_interval_batches,
                "last_durable_batch": (
                    batch_index + 1
                    if (batch_index + 1) % h5.checkpoint_interval_batches == 0
                    else (batch_index + 1) // h5.checkpoint_interval_batches * h5.checkpoint_interval_batches
                ),
                "cuda_memory_allocated_mb": (
                    torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
                ),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "final_test_read": False,
            }
            atomic_write_json(live_progress_path, live_state)
            next_batch = batch_index + 1
            if next_batch % h5.checkpoint_interval_batches == 0 and next_batch < len(batches):
                _atomic_torch_save(progress_checkpoint, {
                    "fingerprint": fingerprint,
                    "completed_epoch": epoch - 1,
                    "current_epoch": epoch,
                    "next_batch_index": next_batch,
                    "current_epoch_losses": losses,
                    "current_epoch_pair_count": pair_count,
                    "current_epoch_prediction_count": prediction_count,
                    "optimizer_steps_completed": optimizer_steps_completed,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "training_log": logs,
                    **_rng_checkpoint_state(),
                })
        if not losses:
            raise RuntimeError("H5 training produced no supervised continuing-identity transitions")
        logs.append({
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "batch_updates": len(batches),
            "transition_count": len(transitions),
            "pair_count": pair_count,
            "prediction_count": prediction_count,
        })
        _atomic_torch_save(progress_checkpoint, {
            "fingerprint": fingerprint,
            "completed_epoch": epoch,
            "current_epoch": epoch + 1,
            "next_batch_index": 0,
            "current_epoch_losses": [],
            "current_epoch_pair_count": 0,
            "current_epoch_prediction_count": 0,
            "optimizer_steps_completed": optimizer_steps_completed,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "training_log": logs,
            **_rng_checkpoint_state(),
        })
        start_batch = 0
        partial_epoch_losses = []
        partial_pair_count = 0
        partial_prediction_count = 0
    _atomic_torch_save(checkpoint, model.state_dict())
    metadata = {
        "status": "completed", "model": model_name, "fingerprint": fingerprint,
        "signature": signature, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "epoch_count": h5.epochs,
        "training_transition_count": len(transitions),
        "optimizer_step_count": optimizer_steps_completed,
        "batch_size_limit": config.runtime.batch_size,
        "max_pair_elements_per_batch": h5.max_pair_elements_per_batch,
        "training_log": logs,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "backbone_trainable_parameter_count": 0, "final_test_read": False,
    }
    atomic_write_json(metadata_path, metadata)
    _write_csv(config.paths.output_root / "h5_logs" / f"{model_name}_training.csv", logs)
    atomic_write_json(live_progress_path, {
        "status": "completed",
        "model": model_name,
        "epochs": h5.epochs,
        "transitions_per_epoch": len(transitions),
        "checkpoint": str(checkpoint),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "final_test_read": False,
    })
    return model, metadata


def train_h5(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    inputs = validate_h5_inputs(config)
    states: dict[str, Any] = {}
    for model_name in model_names:
        embeddings, cache_audit = load_h5_source_embeddings(config, model_name, inputs)
        _, states[model_name] = _fit_model(config, inputs, embeddings, model_name, cache_audit)
    result = {"status": "completed", "models": list(model_names), "checkpoints": states, "input_audit": inputs.audit, "final_test_read": False}
    atomic_write_json(config.paths.output_root / "h5_training_metadata.json", result)
    return result


def _load_model(config: ExperimentConfig, model_name: str, embedding_dim: int) -> BeeTrackQuery:
    h5 = require_h5(config)
    device = _device(config)
    checkpoint = config.paths.output_root / "h5_checkpoints" / model_name / "final.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"H5 checkpoint is missing for {model_name}; run h5-train first")
    model = BeeTrackQuery(embedding_dim, h5.hidden_dim, h5.num_heads, h5.dropout).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    return model


def _baseline_rows(config: ExperimentConfig, model_name: str, development_ids: set[str]) -> list[dict[str, Any]]:
    source_root = config.paths.h3_output_root
    assert source_root is not None
    rows = _read_csv(source_root / "h3_assignments.csv")
    selected = [row for row in rows if row.get("model") == model_name and row.get("variant") == "baseline_association" and row.get("observation_id") in development_ids]
    if {row["observation_id"] for row in selected} != development_ids:
        raise RuntimeError(f"Frozen H3 baseline is incomplete for {model_name}")
    for row in selected:
        row["variant"] = "frozen_h3_baseline"
    return selected


def _tracking_signature(
    config: ExperimentConfig,
    inputs: H5Inputs,
    *,
    model_name: str,
    video_id: str,
    variant: str,
    observation_ids: Sequence[str],
    cache_fingerprint: Any,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """All inputs that can affect an atomic causal tracking job."""
    h5 = require_h5(config)
    return {
        "implementation": TRACKER_IMPLEMENTATION,
        "model": model_name,
        "video_id": video_id,
        "variant": variant,
        "observation_ids": list(observation_ids),
        "checkpoint_sha256": checkpoint_sha256,
        "source_cache_fingerprint": cache_fingerprint,
        "h5_protocol_sha256": inputs.audit["h5_protocol_sha256"],
        "manifest_sha256": inputs.audit["manifest_sha256"],
        "project_split_sha256": inputs.audit["project_split_sha256"],
        "tracking_parameters": {
            "memory_slots": h5.memory_slots,
            "memory_top_k": h5.memory_top_k,
            "update_gate": h5.update_gate,
            "memory_mix": h5.memory_mix,
            "max_age": h5.max_age,
            "min_assignment_score": h5.min_assignment_score,
            "max_normalized_distance": h5.max_normalized_distance,
        },
        "runtime": {
            "batch_size": config.runtime.batch_size,
            "inference_autocast": False,
            "embedding_dtype": "float32",
            "model_parameter_dtype": "float32",
            "score_dtype": "float64",
        },
        "seed": config.runtime.seed,
        "final_test_read": False,
    }


def _tracking_fingerprint(signature: dict[str, Any]) -> str:
    return sha256_text(canonical_json(signature))


def _tracking_rows_digest(rows: Sequence[dict[str, Any]]) -> str:
    """Digest canonical job rows before trusting an on-disk cache hit."""
    return sha256_text(canonical_json(list(rows)))


def _load_tracking_job(
    path: Path,
    fingerprint: str,
    *,
    model_name: str,
    video_id: str,
    variant: str,
    observation_ids: Sequence[str],
) -> dict[str, Any] | None:
    """Reuse only a complete, exact job; malformed partial files are ignored."""
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = value.get("rows") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("status") != "completed"
        or value.get("fingerprint") != fingerprint
        or value.get("final_test_read") is not False
        or not isinstance(rows, list)
        or len(rows) != len(observation_ids)
        or value.get("rows_sha256") != _tracking_rows_digest(rows)
    ):
        return None
    row_ids = [str(row.get("observation_id", "")) for row in rows if isinstance(row, dict)]
    if (
        len(row_ids) != len(rows)
        or sorted(row_ids) != sorted(observation_ids)
        or len(set(row_ids)) != len(row_ids)
        or any(
            row.get("model") != model_name
            or row.get("video_id") != video_id
            or row.get("variant") != variant
            for row in rows
        )
    ):
        return None
    return value


def track_h5(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    inputs = validate_h5_inputs(config)
    h5 = require_h5(config)
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("H5 tracking requires unique model names")
    observations = inputs.h3_inputs.observations
    development = inputs.indices("development_validation")
    by_video: dict[str, list[int]] = defaultdict(list)
    for index in development:
        by_video[observations[index].video_id].append(index)
    development_ids = {observations[index].observation_id for index in development}
    for indices in by_video.values():
        indices.sort(
            key=lambda index: (
                observations[index].frame,
                observations[index].center_x,
                observations[index].center_y,
                observations[index].observation_id,
            )
        )
    if not by_video:
        raise RuntimeError("H5 tracking found no development_validation observations")
    all_rows: list[dict[str, Any]] = []
    cache_audit: dict[str, Any] = {}
    output = config.paths.output_root
    total_jobs = len(model_names) * len(by_video) * len(H5_VARIANTS[1:])
    total_frames = len(model_names) * len(H5_VARIANTS[1:]) * sum(
        len({observations[index].frame for index in indices})
        for indices in by_video.values()
    )
    completed_jobs = 0
    reused_jobs = 0
    completed_frames = 0
    reported_frames = 0
    tracking_started = time.monotonic()
    progress_path = output / "h5_logs" / "tracking_progress.json"

    def write_progress(
        *,
        status: str,
        model: str | None = None,
        video_id: str | None = None,
        variant: str | None = None,
        frame: int | None = None,
        current_frames: int | None = None,
    ) -> None:
        nonlocal reported_frames
        candidate_frames = completed_frames if current_frames is None else current_frames
        # A live callback counts all frames in the in-flight batch, whereas a
        # just-completed job counts only its own frames.  Never let the public
        # progress/ETA move backward while those two notifications interleave.
        reported_frames = max(reported_frames, min(total_frames, candidate_frames))
        observed_frames = reported_frames
        elapsed = max(time.monotonic() - tracking_started, 1e-6)
        rate = observed_frames / elapsed
        remaining = max(0, total_frames - observed_frames)
        atomic_write_json(
            progress_path,
            {
                "status": status,
                "completed_jobs": completed_jobs,
                "total_jobs": total_jobs,
                "completed_frames": observed_frames,
                "total_frames": total_frames,
                "model": model,
                "video_id": video_id,
                "variant": variant,
                "frame": frame,
                "elapsed_seconds": elapsed,
                "frames_per_second": rate if observed_frames else 0.0,
                "eta_seconds": remaining / rate if rate > 0 else None,
                "implementation": TRACKER_IMPLEMENTATION,
                "final_test_read": False,
            },
        )

    write_progress(status="running")
    for model_name in model_names:
        embeddings, cache = load_h5_source_embeddings(config, model_name, inputs)
        cache_audit[model_name] = cache
        model = _load_model(config, model_name, embeddings.shape[1])
        all_rows.extend(_baseline_rows(config, model_name, development_ids))
        checkpoint_path = output / "h5_checkpoints" / model_name / "final.pt"
        checkpoint_sha256 = sha256_file(checkpoint_path)
        pending: list[dict[str, Any]] = []
        for video_id in sorted(by_video):
            indices = by_video[video_id]
            job_observations = [observations[index] for index in indices]
            observation_ids = [item.observation_id for item in job_observations]
            values = np.asarray(embeddings[indices], dtype=np.float32)
            frame_count = len({item.frame for item in job_observations})
            for variant in H5_VARIANTS[1:]:
                signature = _tracking_signature(
                    config,
                    inputs,
                    model_name=model_name,
                    video_id=video_id,
                    variant=variant,
                    observation_ids=observation_ids,
                    cache_fingerprint=cache.get("cache_fingerprint"),
                    checkpoint_sha256=checkpoint_sha256,
                )
                fingerprint = _tracking_fingerprint(signature)
                job_path = output / "h5_work" / "tracking" / model_name / variant / f"{video_id}.json"
                job = _load_tracking_job(
                    job_path,
                    fingerprint,
                    model_name=model_name,
                    video_id=video_id,
                    variant=variant,
                    observation_ids=observation_ids,
                )
                if job is not None:
                    all_rows.extend(job["rows"])
                    completed_jobs += 1
                    reused_jobs += 1
                    completed_frames += frame_count
                    write_progress(
                        status="running",
                        model=model_name,
                        video_id=video_id,
                        variant=variant,
                        frame=job_observations[-1].frame,
                    )
                    continue
                pending.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "video_id": video_id,
                        "observations": job_observations,
                        "embeddings": values,
                        "path": job_path,
                        "signature": signature,
                        "fingerprint": fingerprint,
                        "frame_count": frame_count,
                    }
                )
        if pending:
            base_completed_frames = completed_frames

            def on_live_progress(event: dict[str, Any]) -> None:
                write_progress(
                    status="running",
                    model=str(event["model"]),
                    video_id=str(event["video_id"]),
                    variant=str(event["variant"]),
                    frame=int(event["frame"]),
                    current_frames=base_completed_frames + int(event["completed_frames"]),
                )

            def on_job_complete(index: int, rows: list[dict[str, Any]]) -> None:
                nonlocal completed_jobs, completed_frames
                job = pending[index]
                expected_ids = [item.observation_id for item in job["observations"]]
                if len(rows) != len(expected_ids) or sorted(row["observation_id"] for row in rows) != sorted(expected_ids):
                    raise RuntimeError("H5 tracking job did not produce one row for every observation")
                elapsed = time.monotonic() - tracking_started
                runtime = {
                    "model": job["model"],
                    "video_id": job["video_id"],
                    "variant": job["variant"],
                    "observation_count": len(rows),
                    "frame_count": job["frame_count"],
                    "elapsed_seconds_since_tracking_start": elapsed,
                    "device_role": "causal_padded_frame_batch",
                    "feature_extraction": "reused_h3_cache",
                    "final_test_read": False,
                }
                atomic_write_json(
                    job["path"],
                    {
                        "status": "completed",
                        "fingerprint": job["fingerprint"],
                        "signature": job["signature"],
                        "rows": rows,
                        "rows_sha256": _tracking_rows_digest(rows),
                        "runtime": runtime,
                        "final_test_read": False,
                    },
                )
                all_rows.extend(rows)
                completed_jobs += 1
                completed_frames += int(job["frame_count"])
                write_progress(
                    status="running",
                    model=str(job["model"]),
                    video_id=str(job["video_id"]),
                    variant=str(job["variant"]),
                    frame=int(job["observations"][-1].frame),
                )

            sequences = [
                (job["model"], job["variant"], job["observations"], job["embeddings"])
                for job in pending
            ]
            track_sequences_batched(
                model,
                sequences,
                h5,
                _device(config),
                batch_size=config.runtime.batch_size,
                progress_callback=on_live_progress,
                job_completed_callback=on_job_complete,
            )
    all_rows.sort(key=lambda row: (row["model"], row["variant"], row["video_id"], int(row["frame_id"]), row["observation_id"]))
    per_video, summary = summarize_gt_assignments(all_rows)
    paired = paired_video_differences([
        {**row, "variant": "baseline_association" if row["variant"] == "frozen_h3_baseline" else row["variant"]}
        for row in per_video
    ])
    for row in paired:
        if row["variant"] == "baseline_association":
            row["variant"] = "frozen_h3_baseline"
    _write_csv(output / "h5_assignments.csv", all_rows)
    _write_csv(output / "h5_per_video_metrics.csv", per_video)
    _write_csv(output / "h5_summary.csv", summary)
    _write_csv(output / "h5_paired_video_metrics.csv", paired)
    _write_mot_results(output / "h5_mot_results", all_rows, observations)
    state = {
        "status": "completed_development_gt_boxes",
        "models": list(model_names),
        "variants": list(H5_VARIANTS),
        "assignment_rows": len(all_rows),
        "input_audit": inputs.audit,
        "cache_audit": cache_audit,
        "resumable_job_count": total_jobs,
        "reused_job_count": reused_jobs,
        "tracker_implementation": TRACKER_IMPLEMENTATION,
        "gt_identity_used_for_decisions": False,
        "final_test_read": False,
    }
    atomic_write_json(output / "h5_tracking_metadata.json", state)
    write_progress(status="completed")
    return state


def run_h5_all(config: ExperimentConfig, model_names: Sequence[str], confirm_full: bool) -> dict[str, Any]:
    h5 = require_h5(config)
    if not h5.allow_subset and not confirm_full:
        raise RuntimeError("Full H5 requires --confirm-full; use the smoke config for subsets")
    validate_h5_inputs(config)
    training = train_h5(config, model_names)
    tracking = track_h5(config, model_names)
    progress_path = config.paths.output_root / "h5_logs" / "tracking_progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress.update({
            "status": "reporting",
            "reporting_started_at": datetime.now(timezone.utc).isoformat(),
            "final_test_read": False,
        })
        atomic_write_json(progress_path, progress)
    from .report import generate_h5_report
    report = generate_h5_report(config, model_names)
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress.update({
            "status": "completed",
            "reporting_completed_at": datetime.now(timezone.utc).isoformat(),
            "final_test_read": False,
        })
        atomic_write_json(progress_path, progress)
    return {"status": report["status"], "training": training["status"], "tracking": tracking["status"], "models": list(model_names), "final_test_read": False}
