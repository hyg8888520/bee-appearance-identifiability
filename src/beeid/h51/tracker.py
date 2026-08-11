"""Instrumented H5 v4 causal replay without ground-truth decision inputs."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..config import H5Config
from ..data.mot import Observation
from ..h3.assignment import maximum_weight_matching
from ..h5.model import BeeTrackQuery
from ..h5.tracker import _batched_memory_read, _new_state
from .audit import ModuleCallProbe


REJECTION_REASONS = (
    "matched",
    "no_active_track",
    "no_motion_valid_candidate",
    "score_below_threshold",
    "assignment_conflict",
)


def classify_rejection(
    *, matched: bool, active_tracks: int, motion_valid_candidates: int,
    above_score_candidates: int,
) -> str:
    if matched:
        return "matched"
    if active_tracks == 0:
        return "no_active_track"
    if motion_valid_candidates == 0:
        return "no_motion_valid_candidate"
    if above_score_candidates == 0:
        return "score_below_threshold"
    return "assignment_conflict"


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1), dim=1)[0].cpu())


@torch.inference_mode()
def instrumented_track_sequence(
    model_name: str,
    variant: str,
    model: BeeTrackQuery,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    config: H5Config,
    device: torch.device,
    *,
    candidate_sample_per_observation: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    """Replay one sequence with the exact H5 state update and auditable causes."""
    if candidate_sample_per_observation < 1:
        raise ValueError("candidate_sample_per_observation must be positive")
    state = _new_state(model_name, variant, observations, embeddings, config)
    observation_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    activity: Counter[str] = Counter()
    # Audit-only labels are updated after every completed H5 decision/state
    # update and are never passed back into the tracker.
    audit_identity_by_track: dict[int, str] = {}
    model.eval()
    with ModuleCallProbe(model) as probe:
        while not state.complete:
            frame = state.current_frame
            tracks, items, raw, geometry_np = state.prepare()
            raw_tensor = torch.from_numpy(raw).to(device=device)
            geometry = torch.from_numpy(geometry_np).to(device=device)
            tokens = model.encode_detections(raw_tensor, geometry)
            queries_pre: torch.Tensor | None = None
            queries_refresh: torch.Tensor | None = None
            queries_memory: torch.Tensor | None = None
            scores: np.ndarray | None = None
            logit_values: np.ndarray | None = None
            reliability: np.ndarray | None = None
            valid: np.ndarray | None = None
            distances: np.ndarray | None = None
            matches: list[tuple[int, int]] = []
            if tracks:
                queries_pre = torch.stack([track.query for track in tracks])
                padding = torch.zeros((1, len(items)), dtype=torch.bool, device=device)
                queries_refresh = model.refresh_batched(queries_pre.unsqueeze(0), tokens.unsqueeze(0), padding)[0]
                queries_memory = _batched_memory_read(
                    model, queries_refresh.unsqueeze(0), [tracks], [len(tracks)],
                    config.memory_top_k, [variant != "persistent_query_no_memory"],
                )[0]
                track_geometry = torch.stack([track.geometry for track in tracks])
                geometry_delta = torch.abs(track_geometry[:, None, :] - geometry[None, :, :])
                logits = model.pair_logits_batched(
                    queries_memory.unsqueeze(0), tokens.unsqueeze(0), geometry_delta.unsqueeze(0)
                )[0]
                score_tensor = torch.sigmoid(logits)
                reliability_tensor = model.reliability(logits)
                diagonal = torch.linalg.vector_norm(track_geometry[:, 2:], dim=1).clamp_min(1e-6)
                distance_tensor = torch.linalg.vector_norm(
                    track_geometry[:, None, :2] - geometry[None, :, :2], dim=2
                ) / diagonal[:, None]
                scores = score_tensor.cpu().numpy().astype(np.float64, copy=False)
                logit_values = logits.cpu().numpy().astype(np.float64, copy=False)
                reliability = reliability_tensor.cpu().numpy().astype(np.float64, copy=False)
                distances = distance_tensor.cpu().numpy().astype(np.float64, copy=False)
                valid = distances <= config.max_normalized_distance
                matches = maximum_weight_matching(scores, valid=valid, minimum_score=config.min_assignment_score)
                activity["query_refresh_count"] += len(tracks)
                if variant != "persistent_query_no_memory":
                    activity["effective_memory_read_count"] += sum(bool(track.memory) for track in tracks)
            match_by_detection = {detection: track for track, detection in matches}
            reason_counts: Counter[str] = Counter()
            for detection_index, item in enumerate(items):
                matched = detection_index in match_by_detection
                track_index = match_by_detection.get(detection_index)
                if scores is None or logit_values is None or valid is None or distances is None:
                    column_scores = np.asarray([], dtype=np.float64)
                    column_logits = np.asarray([], dtype=np.float64)
                    column_valid = np.asarray([], dtype=bool)
                    column_distance = np.asarray([], dtype=np.float64)
                else:
                    column_scores = scores[:, detection_index]
                    column_logits = logit_values[:, detection_index]
                    column_valid = valid[:, detection_index]
                    column_distance = distances[:, detection_index]
                above = column_valid & (column_scores >= config.min_assignment_score)
                reason = classify_rejection(
                    matched=matched, active_tracks=len(tracks),
                    motion_valid_candidates=int(column_valid.sum()),
                    above_score_candidates=int(above.sum()),
                )
                reason_counts[reason] += 1
                order = np.argsort(-column_scores, kind="stable") if column_scores.size else np.asarray([], dtype=int)
                top = column_scores[order[:2]]
                top_logits = column_logits[order[:2]]
                candidate_sample = [
                    {
                        "track_id": tracks[int(index)].identifier,
                        "logit": float(column_logits[index]),
                        "score": float(column_scores[index]),
                        "normalized_distance": float(column_distance[index]),
                        "motion_valid": bool(column_valid[index]),
                        "above_score_threshold": bool(above[index]),
                    }
                    for index in order[:candidate_sample_per_observation]
                ]
                gate_accepted: bool | str = ""
                selected_reliability: float | str = ""
                selected_distance: float | str = ""
                norm_pre: float | str = ""
                norm_refresh: float | str = ""
                norm_memory: float | str = ""
                memory_cosine: float | str = ""
                memory_delta: float | str = ""
                memory_read_active = False
                matched_track_id: int | str = ""
                association_continuity_correct: bool | str = ""
                accepted_incorrect_update: bool | str = ""
                if matched:
                    assert track_index is not None
                    assert queries_pre is not None and queries_refresh is not None and queries_memory is not None
                    assert reliability is not None and distances is not None
                    selected_reliability = float(reliability[track_index])
                    selected_distance = float(distances[track_index, detection_index])
                    gate_accepted = (
                        variant != "beetrackquery_gated_memory"
                        or selected_reliability >= config.update_gate
                    )
                    matched_track_id = tracks[track_index].identifier
                    previous_audit_identity = audit_identity_by_track.get(int(matched_track_id))
                    if previous_audit_identity is None:
                        raise RuntimeError("H5.1 audit identity map is missing an active predicted track")
                    association_continuity_correct = previous_audit_identity == item.identity
                    accepted_incorrect_update = bool(
                        gate_accepted and not association_continuity_correct
                    )
                    norm_pre = float(torch.linalg.vector_norm(queries_pre[track_index]).cpu())
                    norm_refresh = float(torch.linalg.vector_norm(queries_refresh[track_index]).cpu())
                    norm_memory = float(torch.linalg.vector_norm(queries_memory[track_index]).cpu())
                    memory_read_active = (
                        variant != "persistent_query_no_memory" and bool(tracks[track_index].memory)
                    )
                    if memory_read_active:
                        memory_cosine = _cosine(queries_refresh[track_index], queries_memory[track_index])
                        memory_delta = float(torch.linalg.vector_norm(
                            queries_memory[track_index] - queries_refresh[track_index]
                        ).cpu())
                    activity["eligible_gate_update_count"] += 1
                    if gate_accepted:
                        activity["accepted_gate_update_count"] += 1
                        activity["memory_write_count"] += 1
                observation_rows.append(
                    {
                        "model": model_name, "variant": variant,
                        "stage": "development_gt_detection_boxes_diagnostic_replay",
                        "video_id": item.video_id, "frame_id": item.frame,
                        "observation_id": item.observation_id,
                        "active_tracks": len(tracks), "detections": len(items),
                        "motion_valid_candidate_count": int(column_valid.sum()),
                        "above_score_candidate_count": int(above.sum()),
                        "max_any_score": float(column_scores.max()) if column_scores.size else "",
                        "max_motion_valid_score": float(column_scores[column_valid].max()) if column_valid.any() else "",
                        "max_any_logit": float(column_logits.max()) if column_logits.size else "",
                        "max_motion_valid_logit": float(column_logits[column_valid].max()) if column_valid.any() else "",
                        "top1_score": float(top[0]) if len(top) else "",
                        "top2_score": float(top[1]) if len(top) > 1 else "",
                        "top2_margin": float(top[0] - top[1]) if len(top) > 1 else "",
                        "top1_logit": float(top_logits[0]) if len(top_logits) else "",
                        "top2_logit": float(top_logits[1]) if len(top_logits) > 1 else "",
                        "top2_logit_margin": float(top_logits[0] - top_logits[1]) if len(top_logits) > 1 else "",
                        "nearest_normalized_distance": float(column_distance.min()) if column_distance.size else "",
                        "matched_existing_track": matched,
                        "new_track": not matched,
                        "matched_track_id": matched_track_id,
                        "rejection_reason": reason,
                        "selected_normalized_distance": selected_distance,
                        "query_norm_pre_refresh": norm_pre,
                        "query_norm_post_refresh": norm_refresh,
                        "query_norm_post_memory": norm_memory,
                        "memory_read_active": memory_read_active,
                        "memory_read_cosine": memory_cosine,
                        "memory_read_delta": memory_delta,
                        "reliability": selected_reliability,
                        "gate_accepted": gate_accepted,
                        "association_continuity_correct": association_continuity_correct,
                        "accepted_incorrect_update": accepted_incorrect_update,
                        "post_decision_gt_audit": matched,
                        "candidate_sample_json": json.dumps(candidate_sample, sort_keys=True, separators=(",", ":")),
                        "offline_gt_audit": False, "deployable_rollout": True,
                        "input_partition": "development_validation",
                        "gt_identity_used_for_inference_decision": False,
                        "final_test_read": False,
                    }
                )
            tracker_row_start = len(state.rows)
            state.finish_frame(
                tracks, items, tokens, geometry,
                queries_memory, scores, reliability, valid,
            )
            # This map exists only for post-decision labels. Updating it after
            # finish_frame guarantees neither GT identity nor the labels can
            # affect assignment, gate, query, memory, or track creation.
            identity_by_observation = {item.observation_id: item.identity for item in items}
            for tracker_row in state.rows[tracker_row_start:]:
                audit_identity_by_track[int(tracker_row["predicted_track_id"])] = (
                    identity_by_observation[str(tracker_row["observation_id"])]
                )
            frame_rows.append(
                {
                    "model": model_name, "variant": variant,
                    "video_id": observations[0].video_id if observations else "",
                    "frame_id": frame, "active_tracks": len(tracks),
                    "detections": len(items), "matched_count": len(matches),
                    "new_track_count": len(items) - len(matches),
                    "motion_valid_candidate_count": int(valid.sum()) if valid is not None else 0,
                    "above_score_candidate_count": int(
                        (valid & (scores >= config.min_assignment_score)).sum()
                    ) if valid is not None and scores is not None else 0,
                    **{f"reason_{reason}": reason_counts.get(reason, 0) for reason in REJECTION_REASONS},
                    "gt_identity_used_for_inference_decision": False,
                    "input_partition": "development_validation",
                    "final_test_read": False,
                }
            )
    for reason in REJECTION_REASONS:
        activity[f"reason_{reason}"] = sum(row["rejection_reason"] == reason for row in observation_rows)
    for key in (
        "query_refresh_count", "effective_memory_read_count", "eligible_gate_update_count",
        "accepted_gate_update_count", "memory_write_count",
    ):
        activity.setdefault(key, 0)
    activity["accepted_incorrect_update_count"] = sum(
        row["accepted_incorrect_update"] is True for row in observation_rows
    )
    return state.rows, observation_rows, frame_rows, dict(activity), dict(sorted(probe.calls.items()))
