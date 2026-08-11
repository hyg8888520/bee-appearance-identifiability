"""Resumable BeeTrackQuery training and development-only GT-box evaluation."""

from __future__ import annotations

import csv
import io
import json
import os
import random
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..config import ExperimentConfig, H5Config
from ..data.mot import Observation
from ..h3.metrics import paired_video_differences, summarize_gt_assignments
from ..utils import atomic_write_json, atomic_write_text, canonical_json, sha256_file, sha256_text
from . import H5_MODELS, H5_PRIMARY_VARIANT, H5_VARIANTS
from .core import H5Inputs, load_h5_source_embeddings, require_h5, validate_h5_inputs
from .model import BeeTrackQuery, supervised_transition
from .tracker import normalized_geometry, track_sequence


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
        "reliability_loss_weight": h5.reliability_loss_weight,
        "update_gate": h5.update_gate, "memory_mix": h5.memory_mix,
        "max_age": h5.max_age, "min_assignment_score": h5.min_assignment_score,
        "max_normalized_distance": h5.max_normalized_distance,
    }
    return {
        "format_version": 1,
        "experiment": "H5_BeeTrackQuery",
        "implementation": "beeid.h5:v1",
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


def _fit_model(
    config: ExperimentConfig,
    inputs: H5Inputs,
    embeddings: np.ndarray,
    model_name: str,
    cache_audit: dict[str, Any],
) -> tuple[BeeTrackQuery, dict[str, Any]]:
    h5 = require_h5(config)
    device = _device(config)
    signature = _checkpoint_signature(config, inputs, model_name, embeddings.shape[1], cache_audit)
    fingerprint = sha256_text(canonical_json(signature))
    directory = config.paths.output_root / "h5_checkpoints" / model_name
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "final.pt"
    progress_checkpoint = directory / "training-progress.pt"
    metadata_path = directory / "checkpoint.json"
    model = BeeTrackQuery(embeddings.shape[1], h5.hidden_dim, h5.num_heads, h5.dropout).to(device)
    if checkpoint.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Refusing incompatible H5 checkpoint reuse for {model_name}")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        return model, {**metadata, "status": "resumed", "checkpoint": str(checkpoint)}

    _seed(config.runtime.seed)
    clips = _training_clips(inputs, h5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=h5.learning_rate, weight_decay=h5.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.runtime.amp and device.type == "cuda")
    observations = inputs.h3_inputs.observations
    logs: list[dict[str, Any]] = []
    start_epoch = 1
    if progress_checkpoint.is_file():
        progress = torch.load(progress_checkpoint, map_location=device, weights_only=False)
        if progress.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Refusing incompatible partial H5 checkpoint reuse for {model_name}")
        model.load_state_dict(progress["model"], strict=True)
        optimizer.load_state_dict(progress["optimizer"])
        if scaler.is_enabled() and progress.get("scaler") is not None:
            scaler.load_state_dict(progress["scaler"])
        logs = list(progress.get("training_log", []))
        start_epoch = int(progress["completed_epoch"]) + 1
    model.train()
    for epoch in range(start_epoch, h5.epochs + 1):
        order = np.random.default_rng(config.runtime.seed + epoch).permutation(len(clips))
        losses: list[float] = []
        for clip_index in order:
            clip = clips[int(clip_index)]
            state_tokens: dict[str, torch.Tensor] = {}
            state_geometry: dict[str, torch.Tensor] = {}
            optimizer.zero_grad(set_to_none=True)
            transition_losses: list[torch.Tensor] = []
            for frame_indices in clip:
                raw = torch.as_tensor(embeddings[frame_indices], dtype=torch.float32, device=device)
                geom = torch.as_tensor(np.stack([normalized_geometry(observations[i]) for i in frame_indices]), device=device)
                identities = [observations[i].identity for i in frame_indices]
                with _autocast(config, device):
                    tokens = model.encode_detections(raw, geom)
                    previous = sorted(set(state_tokens) & set(identities))
                    if previous:
                        previous_tokens = torch.stack([state_tokens[key] for key in previous])
                        delta = torch.stack([torch.abs(state_geometry[key][None, :] - geom) for key in previous])
                        result = supervised_transition(model, previous_tokens, previous, tokens, identities, delta)
                        if result is not None:
                            transition_losses.append(result.association_loss + h5.reliability_loss_weight * result.reliability_loss)
                for identity, token, geometry in zip(identities, tokens, geom):
                    state_tokens[identity] = token
                    state_geometry[identity] = geometry
            if transition_losses:
                loss = torch.stack(transition_losses).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
        if not losses:
            raise RuntimeError("H5 training produced no supervised continuing-identity transitions")
        logs.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), "clip_updates": len(losses)})
        progress_temporary = progress_checkpoint.with_name(
            f".{progress_checkpoint.name}.{uuid.uuid4().hex}.tmp"
        )
        torch.save(
            {
                "fingerprint": fingerprint, "completed_epoch": epoch,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "training_log": logs,
            },
            progress_temporary,
        )
        os.replace(progress_temporary, progress_checkpoint)
    temporary = checkpoint.with_name(f".{checkpoint.name}.{uuid.uuid4().hex}.tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, checkpoint)
    metadata = {
        "status": "completed", "model": model_name, "fingerprint": fingerprint,
        "signature": signature, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "epoch_count": h5.epochs, "training_clip_count": len(clips), "training_log": logs,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "backbone_trainable_parameter_count": 0, "final_test_read": False,
    }
    atomic_write_json(metadata_path, metadata)
    _write_csv(config.paths.output_root / "h5_logs" / f"{model_name}_training.csv", logs)
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


def track_h5(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    inputs = validate_h5_inputs(config)
    h5 = require_h5(config)
    observations = inputs.h3_inputs.observations
    development = inputs.indices("development_validation")
    by_video: dict[str, list[int]] = defaultdict(list)
    for index in development:
        by_video[observations[index].video_id].append(index)
    development_ids = {observations[index].observation_id for index in development}
    all_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        embeddings, _ = load_h5_source_embeddings(config, model_name, inputs)
        model = _load_model(config, model_name, embeddings.shape[1])
        all_rows.extend(_baseline_rows(config, model_name, development_ids))
        for video_id in sorted(by_video):
            indices = sorted(by_video[video_id], key=lambda i: (observations[i].frame, observations[i].center_x))
            for variant in H5_VARIANTS[1:]:
                all_rows.extend(track_sequence(model_name, variant, model, [observations[i] for i in indices], embeddings[indices], h5, _device(config)))
    all_rows.sort(key=lambda row: (row["model"], row["variant"], row["video_id"], int(row["frame_id"]), row["observation_id"]))
    per_video, summary = summarize_gt_assignments(all_rows)
    paired = paired_video_differences([
        {**row, "variant": "baseline_association" if row["variant"] == "frozen_h3_baseline" else row["variant"]}
        for row in per_video
    ])
    for row in paired:
        if row["variant"] == "baseline_association":
            row["variant"] = "frozen_h3_baseline"
    _write_csv(config.paths.output_root / "h5_assignments.csv", all_rows)
    _write_csv(config.paths.output_root / "h5_per_video_metrics.csv", per_video)
    _write_csv(config.paths.output_root / "h5_summary.csv", summary)
    _write_csv(config.paths.output_root / "h5_paired_video_metrics.csv", paired)
    _write_mot_results(config.paths.output_root / "h5_mot_results", all_rows, observations)
    state = {"status": "completed_development_gt_boxes", "models": list(model_names), "variants": list(H5_VARIANTS), "assignment_rows": len(all_rows), "input_audit": inputs.audit, "final_test_read": False}
    atomic_write_json(config.paths.output_root / "h5_tracking_metadata.json", state)
    return state


def run_h5_all(config: ExperimentConfig, model_names: Sequence[str], confirm_full: bool) -> dict[str, Any]:
    h5 = require_h5(config)
    if not h5.allow_subset and not confirm_full:
        raise RuntimeError("Full H5 requires --confirm-full; use the smoke config for subsets")
    validate_h5_inputs(config)
    training = train_h5(config, model_names)
    tracking = track_h5(config, model_names)
    from .report import generate_h5_report
    report = generate_h5_report(config, model_names)
    return {"status": report["status"], "training": training["status"], "tracking": tracking["status"], "models": list(model_names), "final_test_read": False}
