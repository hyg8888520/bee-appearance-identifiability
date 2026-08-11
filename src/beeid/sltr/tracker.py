"""Causal two-branch local repairs built directly from H4's assignment options."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..config import H4Config
from ..data.mot import Observation
from ..h4.tracker import StepOutcome, TrackerState, _copy_state, _step_options


def _frames(observations: Sequence[Observation]) -> tuple[dict[int, list[int]], list[int]]:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        by_frame[item.frame].append(index)
    for indices in by_frame.values():
        indices.sort(key=lambda i: (
            observations[i].center_x, observations[i].center_y,
            observations[i].bbox_area, observations[i].observation_id,
        ))
    return by_frame, list(range(min(by_frame), max(by_frame) + 1)) if by_frame else []


def _rows_for_component(
    rows: Sequence[dict[str, Any]], component_ids: set[str]
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["observation_id"]) in component_ids]


def event_prehistory_mapping(
    rows: Sequence[dict[str, Any]], component_gt_identities: Sequence[str],
    *, history_length: int = 5,
) -> tuple[dict[str, str], str | None]:
    """Offline label map from each identity's recent, strictly pre-event history."""
    if history_length <= 0:
        raise ValueError("SLTR label history_length must be positive")
    ordered = sorted(
        rows, key=lambda row: (int(row.get("frame_id", 0)), str(row.get("observation_id", "")))
    )
    component = {str(identity) for identity in component_gt_identities}
    recent: list[dict[str, Any]] = []
    for identity in sorted(component):
        identity_rows = [row for row in ordered if str(row["gt_identity"]) == identity]
        recent.extend(identity_rows[-history_length:])
    gt_to_predictions: dict[str, set[str]] = defaultdict(set)
    prediction_to_gt: dict[str, set[str]] = defaultdict(set)
    for row in recent:
        gt = str(row["gt_identity"])
        prediction = str(row["predicted_track_id"])
        gt_to_predictions[gt].add(prediction)
        prediction_to_gt[prediction].add(gt)
    mapping: dict[str, str] = {}
    for identity in component_gt_identities:
        candidates = gt_to_predictions.get(identity, set())
        if len(candidates) != 1:
            return {}, f"nonunique_or_missing_gt_to_pred:{identity}"
        candidate = next(iter(candidates))
        if len(prediction_to_gt[candidate]) != 1:
            return {}, f"nonunique_pred_to_gt:{identity}"
        mapping[identity] = candidate
    if len(set(mapping.values())) != len(mapping):
        return {}, "component_mapping_not_one_to_one"
    return mapping, None


def _branch_statistics(
    history_rows: Sequence[dict[str, Any]], branch_rows: Sequence[dict[str, Any]],
    mapping: dict[str, str], component: set[str],
) -> tuple[int, int]:
    selected = [row for row in branch_rows if str(row["gt_identity"]) in component]
    correct = sum(str(row["predicted_track_id"]) == mapping[str(row["gt_identity"])] for row in selected)
    previous: dict[str, str] = {}
    for row in sorted(history_rows, key=lambda r: (int(r["frame_id"]), str(r["observation_id"]))):
        gt = str(row["gt_identity"])
        if gt in component:
            previous[gt] = str(row["predicted_track_id"])
    switches = 0
    for row in sorted(selected, key=lambda r: (int(r["frame_id"]), str(r["observation_id"]))):
        gt = str(row["gt_identity"])
        current = str(row["predicted_track_id"])
        if gt in previous and previous[gt] != current:
            switches += 1
        previous[gt] = current
    return int(correct), int(switches)


def _feature_dict(
    state: TrackerState, rank0: StepOutcome, rank1: StepOutcome,
    frame_rows: Sequence[dict[str, Any]],
) -> dict[str, float]:
    scores = [
        float(row["association_score"]) for row in rank0.rows
        if row.get("association_score", "") not in {"", None}
    ]
    appearance = [
        float(row["appearance_similarity"]) for row in rank0.rows
        if row.get("appearance_similarity", "") not in {"", None}
    ]
    motion = [
        float(row["motion_score"]) for row in rank0.rows
        if row.get("motion_score", "") not in {"", None}
    ]
    conflict = rank0.conflict
    assert conflict is not None
    ages = [max(0, int(frame_rows[0]["frame_id"]) - track.last_frame) for track in state.active.values()]
    result = {
        "component_track_count": float(len(conflict.rows)),
        "component_detection_count": float(len(conflict.columns)),
        "assignment_margin": float(conflict.assignment_margin if conflict.assignment_margin is not None else -1.0),
        "rank0_objective": float(rank0.objective),
        "rank0_rank1_objective_gap": float(rank0.objective - rank1.objective),
        "rank0_mean_association_score": float(np.mean(scores)) if scores else 0.0,
        "rank0_mean_appearance_similarity": float(np.mean(appearance)) if appearance else 0.0,
        "rank0_mean_motion_score": float(np.mean(motion)) if motion else 0.0,
        "active_track_count": float(len(state.active)),
        "active_track_max_age": float(max(ages) if ages else 0),
        "event_observation_count": float(len(frame_rows)),
    }
    if not np.isfinite(np.asarray(list(result.values()), dtype=np.float64)).all():
        raise RuntimeError("SLTR constructed a non-finite observable event feature")
    return result


def _roll_deterministic(
    model_name: str, state: TrackerState, observations: Sequence[Observation], embeddings: np.ndarray,
    frames: Sequence[int], by_frame: dict[int, list[int]], config: H4Config,
) -> tuple[TrackerState, list[dict[str, Any]]]:
    current = _copy_state(state)
    rows: list[dict[str, Any]] = []
    for frame in frames:
        indices = by_frame.get(frame, [])
        items = [observations[index] for index in indices]
        values = [embeddings[index] for index in indices]
        outcome = _step_options(
            model_name, "sltr_deterministic_horizon", current, items, values, frame,
            config, allow_branch=False, memory_mode="isolated",
        )[0]
        current = outcome.state
        rows.extend(outcome.rows)
    return current, rows


@dataclass
class CounterfactualEvent:
    model: str
    video_id: str
    event_id: str
    start_frame: int
    component_observation_ids: tuple[str, ...]
    component_gt_identities: tuple[str, ...]
    features: dict[str, float]
    a_rows: list[dict[str, Any]]
    b_rows: list[dict[str, Any]]
    a_state: TrackerState
    b_state: TrackerState
    label_available: bool
    label_unavailable_reason: str
    a_correct_observations: int | None
    b_correct_observations: int | None
    a_idsw: int | None
    b_idsw: int | None
    utility: float | None

    def audit_row(self, partition: str) -> dict[str, Any]:
        return {
            "model": self.model, "partition": partition, "video_id": self.video_id,
            "event_id": self.event_id, "start_frame": self.start_frame,
            "component_observation_ids": "|".join(self.component_observation_ids),
            "component_size": len(self.component_gt_identities),
            "label_available": self.label_available,
            "label_unavailable_reason": self.label_unavailable_reason,
            "a_correct_observations": "" if self.a_correct_observations is None else self.a_correct_observations,
            "b_correct_observations": "" if self.b_correct_observations is None else self.b_correct_observations,
            "a_idsw": "" if self.a_idsw is None else self.a_idsw,
            "b_idsw": "" if self.b_idsw is None else self.b_idsw,
            "utility_b_minus_a": "" if self.utility is None else self.utility,
            "positive": "" if self.utility is None else self.utility > 0,
            "features_json": json.dumps(self.features, sort_keys=True, separators=(",", ":")),
            "counterfactual_a": "rank_0_immediate_then_rank_0_horizon",
            "counterfactual_b": "rank_1_immediate_then_rank_0_horizon",
            "component_policy": "one_to_one_local_component_only",
            "offline_label_uses_gt": True,
            "method_decision_uses_gt": False,
            "final_test_read": False,
        }


def make_counterfactual_event(
    model_name: str, event_id: str, state: TrackerState, observations: Sequence[Observation],
    embeddings: np.ndarray, by_frame: dict[int, list[int]], frames: Sequence[int], position: int,
    config: H4Config, history_rows: Sequence[dict[str, Any]], horizon: int,
) -> CounterfactualEvent | None:
    frame = frames[position]
    indices = by_frame.get(frame, [])
    items = [observations[index] for index in indices]
    values = [embeddings[index] for index in indices]
    options = _step_options(
        model_name, "sltr", state, items, values, frame, config,
        allow_branch=True, memory_mode="isolated",
    )
    if len(options) < 2 or options[0].conflict is None or options[0].conflict.oversized:
        return None
    rank0, rank1 = options[0], options[1]
    conflict = rank0.conflict
    assert conflict is not None
    component_ids = tuple(items[column].observation_id for column in conflict.columns)
    component_gt = tuple(sorted({items[column].identity for column in conflict.columns}))
    if len(component_gt) < 2:
        return None
    future_frames = frames[position + 1 : position + 1 + horizon]
    a_state, a_future = _roll_deterministic(
        model_name, rank0.state, observations, embeddings, future_frames, by_frame, config
    )
    b_state, b_future = _roll_deterministic(
        model_name, rank1.state, observations, embeddings, future_frames, by_frame, config
    )
    a_rows, b_rows = list(rank0.rows) + a_future, list(rank1.rows) + b_future
    mapping, reason = event_prehistory_mapping(history_rows, component_gt, history_length=5)
    if reason is None:
        component = set(component_gt)
        a_correct, a_switches = _branch_statistics(history_rows, a_rows, mapping, component)
        b_correct, b_switches = _branch_statistics(history_rows, b_rows, mapping, component)
        utility = float((b_correct - a_correct) - 2 * (b_switches - a_switches))
    else:
        a_correct = b_correct = a_switches = b_switches = None
        utility = None
    return CounterfactualEvent(
        model_name, observations[0].video_id, event_id, frame, component_ids, component_gt,
        _feature_dict(state, rank0, rank1, rank0.rows), a_rows, b_rows, a_state, b_state,
        reason is None, reason or "", a_correct, b_correct, a_switches, b_switches, utility,
    )


def build_counterfactual_events(
    model_name: str, observations: Sequence[Observation], embeddings: np.ndarray,
    config: H4Config, *, horizon: int = 5,
) -> tuple[list[dict[str, Any]], list[CounterfactualEvent]]:
    """Generate offline A/B events while the baseline state follows rank-0 only."""
    if not observations:
        return [], []
    if embeddings.ndim != 2 or embeddings.shape[0] != len(observations) or not np.isfinite(embeddings).all():
        raise ValueError("SLTR embeddings must be finite and exactly align with observations")
    if len({item.video_id for item in observations}) != 1:
        raise ValueError("SLTR counterfactual generation accepts exactly one video")
    by_frame, frames = _frames(observations)
    state = TrackerState()
    baseline_rows: list[dict[str, Any]] = []
    events: list[CounterfactualEvent] = []
    for position, frame in enumerate(frames):
        event = make_counterfactual_event(
            model_name, f"{observations[0].video_id}:{model_name}:sltr:{frame:06d}",
            state, observations, embeddings, by_frame, frames, position, config, baseline_rows, horizon,
        )
        indices = by_frame.get(frame, [])
        immediate = _step_options(
            model_name, "immediate_baseline", state,
            [observations[index] for index in indices], [embeddings[index] for index in indices], frame,
            config, allow_branch=False, memory_mode="isolated",
        )[0]
        state = immediate.state
        baseline_rows.extend(immediate.rows)
        if event is not None:
            events.append(event)
    baseline_rows.sort(key=lambda row: (int(row["frame_id"]), str(row["observation_id"])))
    return baseline_rows, events


def _annotate(
    rows: Sequence[dict[str, Any]], event: CounterfactualEvent, branch: str,
    reason: str, horizon: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update({
            "variant": "", "event_id": event.event_id, "sltr_event": True,
            "sltr_selected_branch": branch, "sltr_selection_reason": reason,
            "sltr_horizon": horizon, "sltr_component_size": len(event.component_gt_identities),
            "sltr_ground_truth_decision_input": False,
            "sltr_exact_a_fallback": branch == "A" and reason == "below_threshold_exact_a_fallback",
            "final_test_read": False,
        })
        result.append(item)
    return result


def track_selective_sequence(
    model_name: str, variant: str, observations: Sequence[Observation], embeddings: np.ndarray,
    config: H4Config, *, horizon: int, decide: Callable[[CounterfactualEvent], tuple[bool, str]],
    prepare_event: Callable[[CounterfactualEvent], None] | None = None,
) -> tuple[list[dict[str, Any]], list[CounterfactualEvent], list[dict[str, Any]]]:
    """Track one video; after selection only the selected branch state becomes causal state."""
    if not observations:
        return [], [], []
    by_frame, frames = _frames(observations)
    state = TrackerState()
    history: list[dict[str, Any]] = []
    events: list[CounterfactualEvent] = []
    interventions: list[dict[str, Any]] = []
    position = 0
    while position < len(frames):
        frame = frames[position]
        event = make_counterfactual_event(
            model_name, f"{observations[0].video_id}:{model_name}:sltr:{frame:06d}",
            state, observations, embeddings, by_frame, frames, position, config, history, horizon,
        )
        if event is None:
            indices = by_frame.get(frame, [])
            outcome = _step_options(
                model_name, variant, state,
                [observations[index] for index in indices], [embeddings[index] for index in indices], frame,
                config, allow_branch=False, memory_mode="isolated",
            )[0]
            state = outcome.state
            for row in outcome.rows:
                row = dict(row)
                row.update({"variant": variant, "sltr_event": False, "sltr_selected_branch": "A", "sltr_selection_reason": "no_rank1_local_option", "sltr_horizon": 0, "sltr_component_size": 0, "sltr_ground_truth_decision_input": False, "sltr_exact_a_fallback": False, "final_test_read": False})
                history.append(row)
            position += 1
            continue
        if prepare_event is not None:
            prepare_event(event)
        repair, reason = decide(event)
        branch = "B" if repair else "A"
        chosen_rows = event.b_rows if repair else event.a_rows
        state = event.b_state if repair else event.a_state
        annotated = _annotate(chosen_rows, event, branch, reason, horizon)
        for row in annotated:
            row["variant"] = variant
        history.extend(annotated)
        events.append(event)
        interventions.append({
            "model": model_name, "variant": variant, "video_id": event.video_id,
            "event_id": event.event_id, "start_frame": event.start_frame,
            "selected_branch": branch, "repaired": repair, "selection_reason": reason,
            "score": "", "threshold": "", "label_available": event.label_available,
            "offline_utility_b_minus_a": "" if event.utility is None else event.utility,
            "ground_truth_decision_input": False, "final_test_read": False,
        })
        position += 1 + min(horizon, len(frames) - position - 1)
    history.sort(key=lambda row: (int(row["frame_id"]), str(row["observation_id"])))
    return history, events, interventions


def deterministic_frequency_choice(event_ids: Sequence[str], count: int, seed: int = 24) -> set[str]:
    """Select exactly ``count`` event ids with a portable SHA-256 seed ordering."""
    if count < 0 or count > len(event_ids):
        raise ValueError("Frequency-matched random count is outside candidate range")
    ordered = sorted(
        set(event_ids),
        key=lambda event_id: hashlib.sha256(f"{seed}:{event_id}".encode("utf-8")).hexdigest(),
    )
    return set(ordered[:count])
