"""GT-separated oracle audit: can later frames resolve baseline identity switches?"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from ..config import H4Config
from ..data.mot import Observation
from ..h3.assignment import maximum_weight_matching


RECOVERABILITY_FIELDS = (
    "model", "video_id", "event_id", "event_frame", "gt_identity",
    "previous_predicted_track_id", "current_predicted_track_id",
    "component_identities", "component_size", "horizon", "future_frame",
    "available", "appearance_recoverable", "motion_recoverable",
    "joint_recoverable", "appearance_min_margin", "motion_min_margin",
    "joint_min_margin", "unavailable_reason", "oracle_uses_gt",
    "method_input", "final_test_read",
)


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Oracle audit received a non-normalizable embedding")
    return value / norm


def _prediction(history: Sequence[Observation], frame: int) -> tuple[float, float, float]:
    last = history[-1]
    predicted_x, predicted_y = last.center_x, last.center_y
    if len(history) >= 2:
        previous = history[-2]
        elapsed = last.frame - previous.frame
        if elapsed > 0:
            future = frame - last.frame
            predicted_x += (last.center_x - previous.center_x) / elapsed * future
            predicted_y += (last.center_y - previous.center_y) / elapsed * future
    diagonal = max(1e-6, math.hypot(last.original_width, last.original_height))
    return predicted_x, predicted_y, diagonal


def _is_unique_correct(
    matrix: np.ndarray, margin_threshold: float
) -> tuple[bool, float]:
    size = matrix.shape[0]
    if matrix.shape != (size, size) or size < 2:
        return False, float("-inf")
    matches = maximum_weight_matching(matrix)
    exact = len(matches) == size and all(row == column for row, column in matches)
    margins: list[float] = []
    for row in range(size):
        incorrect = max(
            float(matrix[row, column]) for column in range(size) if column != row
        )
        margins.append(float(matrix[row, row]) - incorrect)
    minimum = min(margins)
    return exact and minimum > margin_threshold + 1e-12, minimum


def _component_identities(
    event_row: dict[str, Any],
    frame_rows: Sequence[dict[str, Any]],
    observation_lookup: dict[str, Observation],
    max_size: int,
) -> list[str]:
    identity = str(event_row["gt_identity"])
    previous_track = str(event_row["previous_predicted_track_id"])
    current_track = str(event_row["predicted_track_id"])
    identities = {
        str(row["gt_identity"])
        for row in frame_rows
        if str(row["predicted_track_id"]) in {previous_track, current_track}
    }
    identities.add(identity)
    if len(identities) < 2:
        event_observation = observation_lookup[str(event_row["observation_id"])]
        others = [
            row for row in frame_rows if str(row["gt_identity"]) != identity
        ]
        others.sort(
            key=lambda row: (
                math.hypot(
                    observation_lookup[str(row["observation_id"])].center_x
                    - event_observation.center_x,
                    observation_lookup[str(row["observation_id"])].center_y
                    - event_observation.center_y,
                ),
                str(row["gt_identity"]),
            )
        )
        if others:
            identities.add(str(others[0]["gt_identity"]))
    ordered = [identity] + sorted(identities - {identity})
    return ordered[:max_size]


def audit_model_recoverability(
    model_name: str,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    baseline_rows: Sequence[dict[str, Any]],
    config: H4Config,
) -> list[dict[str, Any]]:
    """Use GT only here to measure an upper bound; never return tracker assignments."""
    observation_lookup = {item.observation_id: item for item in observations}
    embedding_lookup = {
        item.observation_id: _normalize(embeddings[index])
        for index, item in enumerate(observations)
    }
    by_identity: dict[str, list[Observation]] = defaultdict(list)
    for item in observations:
        by_identity[item.identity].append(item)
    for history in by_identity.values():
        history.sort(key=lambda item: (item.frame, item.observation_id))
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        rows_by_frame[int(row["frame_id"])].append(row)
        rows_by_identity[str(row["gt_identity"])].append(row)
    switch_rows: list[dict[str, Any]] = []
    for identity, identity_rows in rows_by_identity.items():
        ordered = sorted(identity_rows, key=lambda row: (int(row["frame_id"]), str(row["observation_id"])))
        previous_track: str | None = None
        for row in ordered:
            current = str(row["predicted_track_id"])
            if previous_track is not None and current != previous_track:
                switch_rows.append({**row, "previous_predicted_track_id": previous_track})
            previous_track = current
    switch_rows.sort(key=lambda row: (str(row["video_id"]), int(row["frame_id"]), str(row["gt_identity"])))

    output: list[dict[str, Any]] = []
    for event_index, event in enumerate(switch_rows, start=1):
        event_frame = int(event["frame_id"])
        component = _component_identities(
            event, rows_by_frame[event_frame], observation_lookup, config.max_component_size
        )
        event_id = f"{event['video_id']}:{model_name}:idsw:{event_frame:06d}:{event_index:05d}"
        histories = {
            identity: [item for item in by_identity[identity] if item.frame < event_frame]
            for identity in component
        }
        for horizon in config.horizons:
            future_frame = event_frame + horizon
            unavailable: list[str] = []
            if len(component) < 2:
                unavailable.append("no_competing_identity")
            if any(not histories[identity] for identity in component):
                unavailable.append("missing_pre_event_history")
            future_items: list[Observation] = []
            for identity in component:
                candidates = [
                    item for item in by_identity[identity] if item.frame == future_frame
                ]
                if len(candidates) != 1:
                    unavailable.append(f"future_observation_count_{identity}_{len(candidates)}")
                else:
                    future_items.append(candidates[0])
            base = {
                "model": model_name,
                "video_id": event["video_id"],
                "event_id": event_id,
                "event_frame": event_frame,
                "gt_identity": event["gt_identity"],
                "previous_predicted_track_id": event["previous_predicted_track_id"],
                "current_predicted_track_id": event["predicted_track_id"],
                "component_identities": "|".join(component),
                "component_size": len(component),
                "horizon": horizon,
                "future_frame": future_frame,
                "oracle_uses_gt": True,
                "method_input": False,
                "final_test_read": False,
            }
            if unavailable:
                output.append(
                    {
                        **base,
                        "available": False,
                        "appearance_recoverable": False,
                        "motion_recoverable": False,
                        "joint_recoverable": False,
                        "appearance_min_margin": "",
                        "motion_min_margin": "",
                        "joint_min_margin": "",
                        "unavailable_reason": ";".join(dict.fromkeys(unavailable)),
                    }
                )
                continue
            prototypes = []
            for identity in component:
                selected_history = histories[identity][-config.oracle_history_length :]
                prototypes.append(
                    _normalize(
                        np.mean(
                            [embedding_lookup[item.observation_id] for item in selected_history],
                            axis=0,
                        )
                    )
                )
            appearance = np.zeros((len(component), len(component)), dtype=np.float64)
            motion = np.zeros_like(appearance)
            for row_index, identity in enumerate(component):
                predicted_x, predicted_y, diagonal = _prediction(
                    histories[identity], future_frame
                )
                gap = max(1, future_frame - histories[identity][-1].frame)
                for column, item in enumerate(future_items):
                    cosine = float(
                        np.dot(prototypes[row_index], embedding_lookup[item.observation_id])
                    )
                    appearance[row_index, column] = (cosine + 1.0) / 2.0
                    distance = math.hypot(
                        item.center_x - predicted_x, item.center_y - predicted_y
                    ) / (diagonal * math.sqrt(gap))
                    motion[row_index, column] = math.exp(-0.5 * distance * distance)
            joint = (
                config.appearance_weight * appearance + config.motion_weight * motion
            ) / (config.appearance_weight + config.motion_weight)
            appearance_ok, appearance_margin = _is_unique_correct(
                appearance, config.oracle_unique_margin
            )
            motion_ok, motion_margin = _is_unique_correct(
                motion, config.oracle_unique_margin
            )
            joint_ok, joint_margin = _is_unique_correct(
                joint, config.oracle_unique_margin
            )
            output.append(
                {
                    **base,
                    "available": True,
                    "appearance_recoverable": appearance_ok,
                    "motion_recoverable": motion_ok,
                    "joint_recoverable": joint_ok,
                    "appearance_min_margin": appearance_margin,
                    "motion_min_margin": motion_margin,
                    "joint_min_margin": joint_margin,
                    "unavailable_reason": "",
                }
            )
    return output


def summarize_recoverability(
    event_rows: Sequence[dict[str, Any]],
    model_names: Sequence[str],
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in model_names:
        for horizon in horizons:
            selected = [
                row for row in event_rows
                if row["model"] == model and int(row["horizon"]) == horizon
            ]
            event_count = len(selected)
            available = sum(bool(row["available"]) for row in selected)
            videos = sorted({str(row["video_id"]) for row in selected})
            output.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "switch_event_count": event_count,
                    "available_event_count": available,
                    "available_fraction": available / event_count if event_count else 0.0,
                    "appearance_recoverable_count": sum(
                        bool(row["appearance_recoverable"]) for row in selected
                    ),
                    "appearance_recoverable_fraction": sum(
                        bool(row["appearance_recoverable"]) for row in selected
                    ) / event_count if event_count else 0.0,
                    "motion_recoverable_count": sum(
                        bool(row["motion_recoverable"]) for row in selected
                    ),
                    "motion_recoverable_fraction": sum(
                        bool(row["motion_recoverable"]) for row in selected
                    ) / event_count if event_count else 0.0,
                    "joint_recoverable_count": sum(
                        bool(row["joint_recoverable"]) for row in selected
                    ),
                    "joint_recoverable_fraction": sum(
                        bool(row["joint_recoverable"]) for row in selected
                    ) / event_count if event_count else 0.0,
                    "videos_with_events": len(videos),
                    "video_ids": "|".join(videos),
                    "denominator_policy": "all_baseline_idsw_events_missing_future_counts_as_not_recoverable",
                    "oracle_uses_gt": True,
                    "final_test_read": False,
                }
            )
    return output


def recoverability_decision(
    summary_rows: Sequence[dict[str, Any]],
    model_names: Sequence[str],
    config: H4Config,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for model in model_names:
        candidates = [
            row for row in summary_rows
            if row["model"] == model and int(row["horizon"]) == config.go_horizon
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Missing H4 recoverability summary for {model}")
        row = candidates[0]
        event_ok = int(row["switch_event_count"]) >= config.min_events_per_model
        videos_ok = int(row["videos_with_events"]) >= config.min_videos_with_events
        fraction_ok = (
            float(row["joint_recoverable_fraction"]) >= config.min_recoverable_fraction
        )
        checks.append(
            {
                "model": model,
                "event_count": int(row["switch_event_count"]),
                "event_count_pass": event_ok,
                "videos_with_events": int(row["videos_with_events"]),
                "video_count_pass": videos_ok,
                "joint_recoverable_fraction": float(row["joint_recoverable_fraction"]),
                "recoverable_fraction_pass": fraction_ok,
                "pass": event_ok and videos_ok and fraction_ok,
            }
        )
    passed = bool(checks) and all(item["pass"] for item in checks)
    return {
        "status": "GO_METHOD_EVALUATION" if passed else "STOP_NO_RECOVERABILITY_SIGNAL",
        "gate_passed": passed,
        "go_horizon": config.go_horizon,
        "thresholds": {
            "min_joint_recoverable_fraction": config.min_recoverable_fraction,
            "min_events_per_model": config.min_events_per_model,
            "min_videos_with_events": config.min_videos_with_events,
        },
        "model_checks": checks,
        "failure_action": "STOP_BEFORE_METHOD_EVALUATION",
        "oracle_uses_gt": True,
        "method_uses_gt": False,
        "final_test_read": False,
    }
