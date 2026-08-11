"""Causal BeeTrackQuery inference on exact GT detections."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from ..config import H5Config
from ..data.mot import Observation
from ..h3.assignment import maximum_weight_matching
from . import H5_VARIANTS
from .model import BeeTrackQuery


@dataclass
class QueryTrack:
    identifier: int
    query: torch.Tensor
    geometry: torch.Tensor
    last_frame: int
    memory: deque[torch.Tensor] = field(default_factory=deque)
    previous_center: tuple[float, float] | None = None
    last_center: tuple[float, float] = (0.0, 0.0)


def normalized_geometry(item: Observation) -> np.ndarray:
    return np.asarray(
        [
            item.center_x / item.image_width,
            item.center_y / item.image_height,
            item.original_width / item.image_width,
            item.original_height / item.image_height,
        ],
        dtype=np.float32,
    )


def _distance(track: QueryTrack, item: Observation) -> float:
    dx = normalized_geometry(item)[:2] - track.geometry.detach().cpu().numpy()[:2]
    diagonal = max(1e-6, float(np.linalg.norm(track.geometry.detach().cpu().numpy()[2:])))
    return float(np.linalg.norm(dx) / diagonal)


@torch.inference_mode()
def track_sequence(
    model_name: str,
    variant: str,
    model: BeeTrackQuery,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    config: H5Config,
    device: torch.device,
) -> list[dict[str, Any]]:
    if variant not in H5_VARIANTS[1:]:
        raise ValueError(f"Unknown trainable H5 variant: {variant}")
    if len(observations) != embeddings.shape[0]:
        raise ValueError("H5 embeddings must align with observations")
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        by_frame[item.frame].append(index)
    active: dict[int, QueryTrack] = {}
    next_identifier = 1
    rows: list[dict[str, Any]] = []
    use_memory = variant != "persistent_query_no_memory"
    gated = variant == "beetrackquery_gated_memory"
    model.eval()
    for frame in sorted(by_frame):
        active = {
            key: value for key, value in active.items()
            if frame - value.last_frame <= config.max_age
        }
        indices = sorted(by_frame[frame], key=lambda i: (observations[i].center_x, observations[i].center_y))
        items = [observations[i] for i in indices]
        raw = torch.as_tensor(embeddings[indices], dtype=torch.float32, device=device)
        geometry = torch.as_tensor(np.stack([normalized_geometry(item) for item in items]), device=device)
        tokens = model.encode_detections(raw, geometry)
        tracks = [active[key] for key in sorted(active)]
        matched_detection: set[int] = set()
        if tracks and items:
            queries = torch.stack([track.query for track in tracks])
            queries = model.refresh(queries, tokens)
            if use_memory:
                memory_queries: list[torch.Tensor] = []
                for query, track in zip(queries, tracks):
                    if not track.memory:
                        memory_queries.append(query)
                        continue
                    memory = torch.stack(list(track.memory))
                    similarities = torch.nn.functional.cosine_similarity(
                        memory, query.unsqueeze(0), dim=1
                    )
                    count = min(config.memory_top_k, memory.shape[0])
                    selected = memory[torch.topk(similarities, count).indices]
                    memory_queries.append(model.read_memory(query, selected))
                queries = torch.stack(memory_queries)
            geometry_delta = torch.stack([
                torch.abs(track.geometry[None, :] - geometry) for track in tracks
            ])
            logits = model.pair_logits(queries, tokens, geometry_delta)
            probabilities = torch.sigmoid(logits)
            reliability = model.reliability(logits)
            scores = probabilities.detach().cpu().numpy().astype(np.float64)
            valid = np.asarray([
                [_distance(track, item) <= config.max_normalized_distance for item in items]
                for track in tracks
            ], dtype=bool)
            matches = maximum_weight_matching(scores, valid=valid, minimum_score=config.min_assignment_score)
            for track_index, detection_index in matches:
                matched_detection.add(detection_index)
                track = tracks[track_index]
                item = items[detection_index]
                confidence = float(reliability[track_index].item())
                accepted = not gated or confidence >= config.update_gate
                if accepted:
                    refreshed = queries[track_index]
                    track.query = torch.nn.functional.normalize(
                        (1.0 - config.memory_mix) * refreshed + config.memory_mix * tokens[detection_index],
                        dim=0,
                    )
                    track.memory.append(tokens[detection_index].detach())
                    while len(track.memory) > config.memory_slots:
                        track.memory.popleft()
                track.previous_center = track.last_center
                track.last_center = (item.center_x, item.center_y)
                track.geometry = geometry[detection_index].detach()
                track.last_frame = frame
                rows.append(_row(model_name, variant, item, track.identifier, True, float(scores[track_index, detection_index]), confidence, accepted))
        for detection_index, item in enumerate(items):
            if detection_index in matched_detection:
                continue
            query = torch.nn.functional.normalize(tokens[detection_index].detach(), dim=0)
            memory: deque[torch.Tensor] = deque([query], maxlen=config.memory_slots)
            active[next_identifier] = QueryTrack(
                next_identifier, query, geometry[detection_index].detach(), frame,
                memory=memory, last_center=(item.center_x, item.center_y),
            )
            rows.append(_row(model_name, variant, item, next_identifier, False, "", "", True))
            next_identifier += 1
    return sorted(rows, key=lambda row: (row["frame_id"], row["observation_id"]))


def _row(model: str, variant: str, item: Observation, predicted: int, matched: bool, score: Any, reliability: Any, accepted: bool) -> dict[str, Any]:
    return {
        "model": model, "variant": variant, "stage": "gt_detection_boxes",
        "video_id": item.video_id, "frame_id": item.frame,
        "observation_id": item.observation_id, "gt_identity": item.identity,
        "gt_track_id": item.track_id, "predicted_track_id": predicted,
        "matched_existing_track": matched, "association_score": score,
        "appearance_similarity": "", "motion_score": "", "normalized_motion_distance": "",
        "reliability": reliability, "effective_memory_alpha": "",
        "memory_update_accepted": accepted, "eligible_memory_update": matched,
        "risk_identity_history_outlier": "", "risk_bbox_scale_change": "",
        "risk_orientation_change_proxy": "", "risk_sharpness_change": "",
        "risk_crowding_overlap": "",
    }
