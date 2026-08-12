"""Event-triggered fixed-lag association with copy-on-branch identity memory."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..config import H4Config
from ..data.mot import Observation
from .assignment import ConflictComponent, local_assignment_options


H4_VARIANTS = (
    "immediate_commit",
    "fixed_lag_frozen_memory_h5",
    "fixed_lag_isolated_memory_h1",
    "fixed_lag_isolated_memory_h3",
    "fixed_lag_isolated_memory_h5",
    "fixed_lag_isolated_memory_h10",
)
H4_PRIMARY_VARIANT = "fixed_lag_isolated_memory_h5"


@dataclass
class TrackState:
    track_id: int
    memory: np.ndarray
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
        velocity_x = (self.last_center[0] - self.previous_center[0]) / elapsed
        velocity_y = (self.last_center[1] - self.previous_center[1]) / elapsed
        future = frame - self.last_frame
        return self.last_center[0] + velocity_x * future, self.last_center[1] + velocity_y * future


@dataclass
class TrackerState:
    active: dict[int, TrackState] = field(default_factory=dict)
    next_track_id: int = 1


@dataclass
class StepOutcome:
    state: TrackerState
    objective: float
    rows: list[dict[str, Any]]
    pending_memory_updates: list[tuple[int, np.ndarray]]
    conflict: ConflictComponent | None
    option_rank: int


@dataclass
class Hypothesis:
    state: TrackerState
    score: float
    rows: list[dict[str, Any]]
    pending_memory_updates: list[tuple[int, np.ndarray]]
    trace: tuple[int, ...]


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("H4 tracker received a non-normalizable embedding")
    return value / norm


def _ema(memory: np.ndarray, embedding: np.ndarray, alpha: float) -> np.ndarray:
    return _normalize((1.0 - alpha) * memory + alpha * embedding)


def _copy_state(state: TrackerState) -> TrackerState:
    return TrackerState(
        active={
            identifier: TrackState(
                track_id=track.track_id,
                memory=track.memory.copy(),
                last_center=track.last_center,
                last_frame=track.last_frame,
                last_width=track.last_width,
                last_height=track.last_height,
                previous_center=track.previous_center,
                previous_frame=track.previous_frame,
            )
            for identifier, track in state.active.items()
        },
        next_track_id=state.next_track_id,
    )


def _motion_pair(track: TrackState, observation: Observation) -> tuple[float, float]:
    predicted_x, predicted_y = track.predicted_center(observation.frame)
    distance = math.hypot(
        observation.center_x - predicted_x, observation.center_y - predicted_y
    )
    diagonal = max(1e-6, math.hypot(track.last_width, track.last_height))
    gap = max(1, observation.frame - track.last_frame)
    normalized = distance / (diagonal * math.sqrt(gap))
    return normalized, math.exp(-0.5 * normalized * normalized)


def _variant_policy(variant: str) -> tuple[int, str]:
    if variant == "immediate_commit":
        return 0, "isolated"
    if variant == "fixed_lag_frozen_memory_h5":
        return 5, "frozen"
    prefix = "fixed_lag_isolated_memory_h"
    if variant.startswith(prefix):
        return int(variant[len(prefix):]), "isolated"
    raise ValueError(f"Unknown H4 association variant: {variant}")


def _score_matrix(
    state: TrackerState,
    observations: Sequence[Observation],
    embeddings: Sequence[np.ndarray],
    frame: int,
    config: H4Config,
) -> tuple[list[int], np.ndarray, np.ndarray, dict[tuple[int, int], dict[str, float]]]:
    live_ids = sorted(
        identifier
        for identifier, track in state.active.items()
        if frame - track.last_frame <= config.max_age
    )
    scores = np.zeros((len(live_ids), len(observations)), dtype=np.float64)
    valid = np.zeros_like(scores, dtype=bool)
    details: dict[tuple[int, int], dict[str, float]] = {}
    denominator = config.appearance_weight + config.motion_weight
    for row, identifier in enumerate(live_ids):
        track = state.active[identifier]
        for column, (item, embedding) in enumerate(zip(observations, embeddings)):
            cosine = float(np.dot(track.memory, embedding))
            appearance = min(1.0, max(0.0, (cosine + 1.0) / 2.0))
            normalized_distance, motion = _motion_pair(track, item)
            score = (
                config.appearance_weight * appearance + config.motion_weight * motion
            ) / denominator
            scores[row, column] = score
            valid[row, column] = normalized_distance <= config.max_normalized_distance
            details[(row, column)] = {
                "association_score": score,
                "appearance_similarity": cosine,
                "motion_score": motion,
                "normalized_motion_distance": normalized_distance,
            }
    return live_ids, scores, valid, details


def _base_row(model_name: str, variant: str, item: Observation) -> dict[str, Any]:
    return {
        "model": model_name,
        "variant": variant,
        "stage": "gt_detection_boxes",
        "video_id": item.video_id,
        "frame_id": item.frame,
        "observation_id": item.observation_id,
        "gt_identity": item.identity,
        "gt_track_id": item.track_id,
        "event_id": "",
        "ambiguity_detected": False,
        "deferred_commit": False,
        "event_start_frame": "",
        "decision_frame": item.frame,
        "decision_latency_frames": 0,
        "branch_memory_isolated": False,
        "branch_memory_frozen": False,
        "hypothesis_count_peak": 1,
        "winning_hypothesis_rank": 1,
        "decision_score_margin": "",
    }


def _step_options(
    model_name: str,
    variant: str,
    state: TrackerState,
    observations: Sequence[Observation],
    embeddings: Sequence[np.ndarray],
    frame: int,
    config: H4Config,
    *,
    allow_branch: bool,
    memory_mode: str,
) -> list[StepOutcome]:
    live_ids, scores, valid, details = _score_matrix(
        state, observations, embeddings, frame, config
    )
    options, conflict = local_assignment_options(
        scores,
        valid=valid,
        minimum_score=config.min_assignment_score,
        ambiguity_margin=config.ambiguity_margin,
        max_component_size=config.max_component_size,
        beam_width=config.beam_width,
        unmatched_penalty=config.unmatched_penalty,
    )
    if not allow_branch or conflict is None or conflict.oversized:
        options = options[:1]
    outcomes: list[StepOutcome] = []
    for rank, option in enumerate(options, start=1):
        updated = _copy_state(state)
        updated.active = {
            identifier: track
            for identifier, track in updated.active.items()
            if frame - track.last_frame <= config.max_age
        }
        matched_columns: set[int] = set()
        rows: list[dict[str, Any]] = []
        pending: list[tuple[int, np.ndarray]] = []
        for row_index, column in option.matches:
            identifier = live_ids[row_index]
            if identifier not in updated.active:
                continue
            track = updated.active[identifier]
            item = observations[column]
            embedding = _normalize(embeddings[column])
            matched_columns.add(column)
            if memory_mode == "isolated":
                track.memory = _ema(track.memory, embedding, config.memory_alpha)
            else:
                pending.append((identifier, embedding.copy()))
            track.previous_center = track.last_center
            track.previous_frame = track.last_frame
            track.last_center = (item.center_x, item.center_y)
            track.last_frame = item.frame
            track.last_width = item.original_width
            track.last_height = item.original_height
            detail = details[(row_index, column)]
            rows.append(
                {
                    **_base_row(model_name, variant, item),
                    "predicted_track_id": identifier,
                    "matched_existing_track": True,
                    **detail,
                    "memory_update_committed": memory_mode == "isolated",
                    "memory_update_used_for_branch_scoring": memory_mode == "isolated",
                    "effective_memory_alpha": config.memory_alpha,
                    "eligible_memory_update": True,
                    "memory_update_accepted": memory_mode == "isolated",
                    "reliability": "",
                }
            )
        for column, item in enumerate(observations):
            if column in matched_columns:
                continue
            identifier = updated.next_track_id
            updated.next_track_id += 1
            updated.active[identifier] = TrackState(
                track_id=identifier,
                memory=_normalize(embeddings[column]),
                last_center=(item.center_x, item.center_y),
                last_frame=item.frame,
                last_width=item.original_width,
                last_height=item.original_height,
            )
            rows.append(
                {
                    **_base_row(model_name, variant, item),
                    "predicted_track_id": identifier,
                    "matched_existing_track": False,
                    "association_score": "",
                    "appearance_similarity": "",
                    "motion_score": "",
                    "normalized_motion_distance": "",
                    "memory_update_committed": True,
                    "memory_update_used_for_branch_scoring": True,
                    "effective_memory_alpha": "",
                    "eligible_memory_update": False,
                    "memory_update_accepted": True,
                    "reliability": "",
                }
            )
        rows.sort(key=lambda row: row["observation_id"])
        outcomes.append(
            StepOutcome(updated, option.objective, rows, pending, conflict, rank)
        )
    return outcomes


def _commit_frozen_updates(
    state: TrackerState, updates: Sequence[tuple[int, np.ndarray]], config: H4Config
) -> None:
    for identifier, embedding in updates:
        track = state.active.get(identifier)
        if track is not None:
            track.memory = _ema(track.memory, embedding, config.memory_alpha)


def _hypothesis_key(hypothesis: Hypothesis) -> tuple[Any, ...]:
    assignment_signature = tuple(
        (int(row["frame_id"]), str(row["observation_id"]), int(row["predicted_track_id"]))
        for row in hypothesis.rows
    )
    return (-hypothesis.score, hypothesis.trace, assignment_signature)


def track_h4_sequence(
    model_name: str,
    variant: str,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    config: H4Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Track one video without consulting GT identities during any decision."""
    horizon, memory_mode = _variant_policy(variant)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(observations):
        raise ValueError("H4 embeddings must align exactly with observations")
    if not observations:
        return [], []
    if len({item.video_id for item in observations}) != 1:
        raise ValueError("track_h4_sequence accepts exactly one video")
    if not np.isfinite(embeddings).all():
        raise ValueError("H4 embeddings contain non-finite values")

    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        by_frame[item.frame].append(index)
    for indices in by_frame.values():
        indices.sort(
            key=lambda index: (
                observations[index].center_x, observations[index].center_y,
                observations[index].bbox_area, observations[index].observation_id,
            )
        )
    frames = list(range(min(by_frame), max(by_frame) + 1))
    state = TrackerState()
    committed_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    position = 0
    event_number = 0
    while position < len(frames):
        frame = frames[position]
        indices = by_frame.get(frame, [])
        frame_observations = [observations[index] for index in indices]
        frame_embeddings = [embeddings[index] for index in indices]
        probe = _step_options(
            model_name, variant, state, frame_observations, frame_embeddings,
            frame, config, allow_branch=True, memory_mode=memory_mode,
        )
        conflict = probe[0].conflict if probe else None
        usable_conflict = conflict is not None and not conflict.oversized and len(probe) > 1
        if horizon == 0 or not usable_conflict:
            immediate = _step_options(
                model_name, variant, state, frame_observations, frame_embeddings,
                frame, config, allow_branch=False, memory_mode="isolated",
            )[0]
            state = immediate.state
            if conflict is not None:
                event_number += 1
                event_id = f"{observations[0].video_id}:{variant}:{frame:06d}:{event_number:04d}"
                for row in immediate.rows:
                    row["event_id"] = event_id
                    row["ambiguity_detected"] = True
                events.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "video_id": observations[0].video_id,
                        "event_id": event_id,
                        "start_frame": frame,
                        "decision_frame": frame,
                        "realized_horizon": 0,
                        "deferred_commit": False,
                        "reason": "oversized_conflict" if conflict.oversized else "immediate_policy",
                        "component_track_count": len(conflict.rows),
                        "component_detection_count": len(conflict.columns),
                        "initial_assignment_margin": (
                            "" if conflict.assignment_margin is None else conflict.assignment_margin
                        ),
                        "hypotheses_expanded": 1,
                        "hypothesis_count_peak": 1,
                        "winning_score": immediate.objective,
                        "runner_up_score": "",
                        "decision_score_margin": "",
                        "analysis_uses_gt": False,
                    }
                )
            committed_rows.extend(immediate.rows)
            position += 1
            continue

        event_number += 1
        event_id = f"{observations[0].video_id}:{variant}:{frame:06d}:{event_number:04d}"
        beam = [
            Hypothesis(
                outcome.state, outcome.objective, list(outcome.rows),
                list(outcome.pending_memory_updates), (outcome.option_rank,),
            )
            for outcome in probe
        ]
        hypotheses_expanded = len(beam)
        peak = len(beam)
        last_position = min(len(frames) - 1, position + horizon)
        for future_position in range(position + 1, last_position + 1):
            future_frame = frames[future_position]
            future_indices = by_frame.get(future_frame, [])
            future_observations = [observations[index] for index in future_indices]
            future_embeddings = [embeddings[index] for index in future_indices]
            expanded: list[Hypothesis] = []
            for hypothesis in beam:
                outcomes = _step_options(
                    model_name, variant, hypothesis.state,
                    future_observations, future_embeddings, future_frame, config,
                    allow_branch=True, memory_mode=memory_mode,
                )
                hypotheses_expanded += len(outcomes)
                for outcome in outcomes:
                    expanded.append(
                        Hypothesis(
                            outcome.state,
                            hypothesis.score + outcome.objective,
                            hypothesis.rows + outcome.rows,
                            hypothesis.pending_memory_updates + outcome.pending_memory_updates,
                            hypothesis.trace + (outcome.option_rank,),
                        )
                    )
            expanded.sort(key=_hypothesis_key)
            beam = expanded[: config.beam_width]
            peak = max(peak, len(beam))
        beam.sort(key=_hypothesis_key)
        winner = beam[0]
        runner_up = beam[1].score if len(beam) > 1 else None
        if memory_mode == "frozen":
            _commit_frozen_updates(winner.state, winner.pending_memory_updates, config)
        decision_frame = frames[last_position]
        decision_margin = "" if runner_up is None else winner.score - runner_up
        for row in winner.rows:
            row["event_id"] = event_id
            row["ambiguity_detected"] = True
            row["deferred_commit"] = True
            row["event_start_frame"] = frame
            row["decision_frame"] = decision_frame
            row["decision_latency_frames"] = decision_frame - int(row["frame_id"])
            row["branch_memory_isolated"] = memory_mode == "isolated"
            row["branch_memory_frozen"] = memory_mode == "frozen"
            row["hypothesis_count_peak"] = peak
            row["winning_hypothesis_rank"] = winner.trace[0]
            row["decision_score_margin"] = decision_margin
            if memory_mode == "frozen" and row["matched_existing_track"]:
                row["memory_update_committed"] = True
                row["memory_update_used_for_branch_scoring"] = False
                row["memory_update_accepted"] = True
        committed_rows.extend(winner.rows)
        state = winner.state
        events.append(
            {
                "model": model_name,
                "variant": variant,
                "video_id": observations[0].video_id,
                "event_id": event_id,
                "start_frame": frame,
                "decision_frame": decision_frame,
                "realized_horizon": decision_frame - frame,
                "deferred_commit": True,
                "reason": "local_assignment_ambiguity",
                "component_track_count": len(conflict.rows),
                "component_detection_count": len(conflict.columns),
                "initial_assignment_margin": conflict.assignment_margin,
                "hypotheses_expanded": hypotheses_expanded,
                "hypothesis_count_peak": peak,
                "winning_score": winner.score,
                "runner_up_score": "" if runner_up is None else runner_up,
                "decision_score_margin": decision_margin,
                "analysis_uses_gt": False,
            }
        )
        position = last_position + 1

    committed_rows.sort(key=lambda row: (row["frame_id"], row["observation_id"]))
    return committed_rows, events
