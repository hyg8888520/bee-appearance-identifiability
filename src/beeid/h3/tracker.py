"""Causal GT-detection association variants for the frozen H3 protocol."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import H3Config
from ..data.mot import Observation
from .assignment import maximum_weight_matching
from .thresholds import dynamic_pair_values, reliability_from_artifact


H3_VARIANTS = (
    "baseline_association",
    "selective_memory_update",
    "reliability_weighted_association",
    "full_ram_bee",
)


@dataclass
class TrackState:
    track_id: int
    memory: np.ndarray
    last_signal: dict[str, Any]
    last_center: tuple[float, float]
    last_frame: int
    last_width: float
    last_height: float
    previous_center: tuple[float, float] | None = None
    previous_frame: int | None = None

    def predicted_center(self, frame: int) -> tuple[float, float]:
        if self.previous_center is None or self.previous_frame is None:
            return self.last_center
        elapsed = self.last_frame - self.previous_frame
        if elapsed <= 0:
            return self.last_center
        vx = (self.last_center[0] - self.previous_center[0]) / elapsed
        vy = (self.last_center[1] - self.previous_center[1]) / elapsed
        future = frame - self.last_frame
        return self.last_center[0] + vx * future, self.last_center[1] + vy * future


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("H3 tracker received a non-normalizable embedding")
    return value / norm


def _ema(memory: np.ndarray, embedding: np.ndarray, alpha: float) -> np.ndarray:
    return _normalize((1.0 - alpha) * memory + alpha * embedding)


def _motion_pair(
    track: TrackState, observation: Observation
) -> tuple[float, float]:
    predicted_x, predicted_y = track.predicted_center(observation.frame)
    distance = math.hypot(
        observation.center_x - predicted_x, observation.center_y - predicted_y
    )
    diagonal = max(1e-6, math.hypot(track.last_width, track.last_height))
    gap = max(1, observation.frame - track.last_frame)
    normalized_distance = distance / (diagonal * math.sqrt(gap))
    motion_score = math.exp(-0.5 * normalized_distance * normalized_distance)
    return normalized_distance, motion_score


def _association_score(
    appearance_score: float,
    motion_score: float,
    reliability: float,
    variant: str,
    config: H3Config,
) -> float:
    appearance_reliability = (
        reliability
        if variant in {"reliability_weighted_association", "full_ram_bee"}
        else 1.0
    )
    appearance_weight = config.appearance_weight * appearance_reliability
    denominator = appearance_weight + config.motion_weight
    if denominator <= 0:  # pragma: no cover - rejected by config validation
        return 0.0
    return (
        appearance_weight * appearance_score + config.motion_weight * motion_score
    ) / denominator


def _memory_alpha(variant: str, reliability: float, config: H3Config) -> float:
    if variant == "selective_memory_update":
        return config.memory_alpha if reliability >= config.update_gate else 0.0
    if variant == "full_ram_bee":
        return (
            config.memory_alpha * reliability
            if reliability >= config.update_gate
            else 0.0
        )
    return config.memory_alpha


def track_gt_sequence(
    model_name: str,
    variant: str,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    signals: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    config: H3Config,
) -> list[dict[str, Any]]:
    """Associate exact GT detections without consulting their GT identities."""
    if variant not in H3_VARIANTS:
        raise ValueError(f"Unknown H3 association variant: {variant}")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(observations):
        raise ValueError("H3 embeddings must align exactly with observations")
    if len(signals) != len(observations):
        raise ValueError("H3 signals must align exactly with observations")
    if not observations:
        return []
    video_ids = {item.video_id for item in observations}
    if len(video_ids) != 1:
        raise ValueError("track_gt_sequence accepts exactly one video")

    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        by_frame[item.frame].append(index)
    for indices in by_frame.values():
        # Spatial order prevents GT track-id order from deciding exact assignment ties.
        indices.sort(
            key=lambda index: (
                observations[index].center_x,
                observations[index].center_y,
                observations[index].bbox_area,
                observations[index].image_path,
            )
        )
    active: dict[int, TrackState] = {}
    next_track_id = 1
    rows: list[dict[str, Any]] = []

    for frame in range(min(by_frame), max(by_frame) + 1):
        detection_indices = by_frame.get(frame, [])
        active = {
            identifier: track
            for identifier, track in active.items()
            if frame - track.last_frame <= config.max_age
        }
        tracks = [active[key] for key in sorted(active)]
        scores = np.zeros((len(tracks), len(detection_indices)), dtype=np.float64)
        valid = np.zeros_like(scores, dtype=bool)
        pair_details: dict[tuple[int, int], dict[str, Any]] = {}
        for track_index, track in enumerate(tracks):
            for detection_index, source_index in enumerate(detection_indices):
                item = observations[source_index]
                embedding = embeddings[source_index]
                values = dynamic_pair_values(
                    track.last_signal, signals[source_index], track.memory, embedding
                )
                reliability, risks = reliability_from_artifact(
                    thresholds, model_name, values
                )
                normalized_distance, motion_score = _motion_pair(track, item)
                cosine = float(np.dot(track.memory, embedding))
                appearance_score = min(1.0, max(0.0, (cosine + 1.0) / 2.0))
                score = _association_score(
                    appearance_score, motion_score, reliability, variant, config
                )
                scores[track_index, detection_index] = score
                valid[track_index, detection_index] = (
                    normalized_distance <= config.max_normalized_distance
                )
                pair_details[(track_index, detection_index)] = {
                    "association_score": score,
                    "appearance_similarity": cosine,
                    "motion_score": motion_score,
                    "normalized_motion_distance": normalized_distance,
                    "reliability": reliability,
                    "risks": risks,
                }
        matches = maximum_weight_matching(
            scores, valid=valid, minimum_score=config.min_assignment_score
        )
        matched_detections: set[int] = set()
        for track_index, detection_index in matches:
            track = tracks[track_index]
            source_index = detection_indices[detection_index]
            matched_detections.add(detection_index)
            item = observations[source_index]
            embedding = embeddings[source_index]
            detail = pair_details[(track_index, detection_index)]
            reliability = float(detail["reliability"])
            alpha = _memory_alpha(variant, reliability, config)
            if alpha > 0:
                track.memory = _ema(track.memory, embedding, alpha)
            previous_center = track.last_center
            previous_frame = track.last_frame
            track.previous_center = previous_center
            track.previous_frame = previous_frame
            track.last_center = (item.center_x, item.center_y)
            track.last_frame = item.frame
            track.last_width = item.original_width
            track.last_height = item.original_height
            track.last_signal = dict(signals[source_index])
            risk_values = detail["risks"]
            rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "stage": "gt_detection_boxes",
                    "video_id": item.video_id,
                    "frame_id": item.frame,
                    "observation_id": item.observation_id,
                    "gt_identity": item.identity,
                    "gt_track_id": item.track_id,
                    "predicted_track_id": track.track_id,
                    "matched_existing_track": True,
                    "association_score": detail["association_score"],
                    "appearance_similarity": detail["appearance_similarity"],
                    "motion_score": detail["motion_score"],
                    "normalized_motion_distance": detail["normalized_motion_distance"],
                    "reliability": reliability,
                    "effective_memory_alpha": alpha,
                    "memory_update_accepted": alpha > 0,
                    "eligible_memory_update": True,
                    **{f"risk_{key}": value for key, value in risk_values.items()},
                }
            )

        for detection_index, source_index in enumerate(detection_indices):
            if detection_index in matched_detections:
                continue
            item = observations[source_index]
            identifier = next_track_id
            next_track_id += 1
            active[identifier] = TrackState(
                track_id=identifier,
                memory=_normalize(embeddings[source_index]),
                last_signal=dict(signals[source_index]),
                last_center=(item.center_x, item.center_y),
                last_frame=item.frame,
                last_width=item.original_width,
                last_height=item.original_height,
            )
            rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "stage": "gt_detection_boxes",
                    "video_id": item.video_id,
                    "frame_id": item.frame,
                    "observation_id": item.observation_id,
                    "gt_identity": item.identity,
                    "gt_track_id": item.track_id,
                    "predicted_track_id": identifier,
                    "matched_existing_track": False,
                    "association_score": "",
                    "appearance_similarity": "",
                    "motion_score": "",
                    "normalized_motion_distance": "",
                    "reliability": "",
                    "effective_memory_alpha": "",
                    "memory_update_accepted": True,
                    "eligible_memory_update": False,
                    "risk_identity_history_outlier": "",
                    "risk_bbox_scale_change": "",
                    "risk_orientation_change_proxy": "",
                    "risk_sharpness_change": "",
                    "risk_crowding_overlap": "",
                }
            )
    rows.sort(key=lambda row: (row["frame_id"], row["observation_id"]))
    return rows
