"""Causal adaptive-lag association built on H4-v1 branch semantics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from ..config import H4Config, H41Config
from ..data.mot import Observation
from ..h4.tracker import (
    Hypothesis,
    TrackerState,
    _commit_frozen_updates,
    _hypothesis_key,
    _step_options,
    track_h4_sequence,
)
from .protocol import H41_PRIMARY_VARIANT, H41_VARIANTS


H41_ASSIGNMENT_EXTRA_FIELDS = (
    "decision_policy",
    "early_commit",
    "forced_at_deadline",
    "forced_at_sequence_end",
    "threshold_rule_met",
    "normalized_decision_margin",
    "winner_stability_steps",
    "evidence_frames_used",
    "competing_hypothesis_survived",
)


def _enrich_control_rows(
    variant: str,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_lookup = {str(row["event_id"]): row for row in events}
    policy = "immediate" if variant == "immediate_commit" else "fixed_lag"
    for event in events:
        steps = int(event["realized_horizon"]) + 1
        realized = int(event["realized_horizon"])
        deferred = bool(event["deferred_commit"])
        raw = event.get("decision_score_margin", "")
        normalized = "" if raw == "" else float(raw) / max(1, steps)
        event.update(
            {
                "decision_policy": policy,
                "early_commit": False,
                "forced_at_deadline": deferred and realized >= 5,
                "forced_at_sequence_end": deferred and realized < 5,
                "threshold_rule_met": False,
                "normalized_decision_margin": normalized,
                "winner_stability_steps": "",
                "evidence_frames_used": steps,
                "competing_hypothesis_survived": event.get("runner_up_score", "") != "",
                "ground_truth_decision_input": False,
                "final_test_read": False,
            }
        )
    for row in rows:
        event = event_lookup.get(str(row.get("event_id", "")))
        row.update(
            {
                "decision_policy": policy,
                "early_commit": False,
                "forced_at_deadline": bool(event and event["forced_at_deadline"]),
                "forced_at_sequence_end": False if event is None else event["forced_at_sequence_end"],
                "threshold_rule_met": False if event is None else event["threshold_rule_met"],
                "normalized_decision_margin": "" if event is None else event["normalized_decision_margin"],
                "winner_stability_steps": "" if event is None else event["winner_stability_steps"],
                "evidence_frames_used": 1 if event is None else event["evidence_frames_used"],
                "competing_hypothesis_survived": False if event is None else event["competing_hypothesis_survived"],
            }
        )
    return rows, events


def _adaptive_memory_mode(variant: str) -> str:
    if variant == "adaptive_lag_frozen_memory_h5":
        return "frozen"
    if variant == "adaptive_lag_isolated_memory_h5":
        return "isolated"
    raise ValueError(f"Unknown H4.1 adaptive variant: {variant}")


def _best_competing_score(
    beam: Sequence[Hypothesis], winner: Hypothesis
) -> float | None:
    return next(
        (
            hypothesis.score
            for hypothesis in beam
            if hypothesis.trace and hypothesis.trace[0] != winner.trace[0]
        ),
        None,
    )


def _adaptive_sequence(
    model_name: str,
    variant: str,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    h4: H4Config,
    h41: H41Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memory_mode = _adaptive_memory_mode(variant)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(observations):
        raise ValueError("H4.1 embeddings must align exactly with observations")
    if not observations:
        return [], []
    if len({item.video_id for item in observations}) != 1:
        raise ValueError("track_h41_sequence accepts exactly one video")
    if not np.isfinite(embeddings).all():
        raise ValueError("H4.1 embeddings contain non-finite values")

    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        by_frame[item.frame].append(index)
    for indices in by_frame.values():
        indices.sort(
            key=lambda index: (
                observations[index].center_x,
                observations[index].center_y,
                observations[index].bbox_area,
                observations[index].observation_id,
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
            model_name,
            variant,
            state,
            frame_observations,
            frame_embeddings,
            frame,
            h4,
            allow_branch=True,
            memory_mode=memory_mode,
        )
        conflict = probe[0].conflict if probe else None
        usable = conflict is not None and not conflict.oversized and len(probe) > 1
        if not usable:
            immediate = _step_options(
                model_name,
                variant,
                state,
                frame_observations,
                frame_embeddings,
                frame,
                h4,
                allow_branch=False,
                memory_mode="isolated",
            )[0]
            state = immediate.state
            if conflict is not None:
                event_number += 1
                event_id = (
                    f"{observations[0].video_id}:{variant}:{frame:06d}:{event_number:04d}"
                )
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
                        "reason": "oversized_conflict",
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
                        "decision_policy": "adaptive_lag",
                        "early_commit": False,
                        "forced_at_deadline": False,
                        "forced_at_sequence_end": False,
                        "threshold_rule_met": False,
                        "normalized_decision_margin": "",
                        "winner_stability_steps": 1,
                        "evidence_frames_used": 1,
                        "competing_hypothesis_survived": False,
                        "ground_truth_decision_input": False,
                        "analysis_uses_gt": False,
                        "final_test_read": False,
                    }
                )
            for row in immediate.rows:
                row.update(
                    {
                        "decision_policy": "adaptive_lag",
                        "early_commit": False,
                        "forced_at_deadline": False,
                        "forced_at_sequence_end": False,
                        "threshold_rule_met": False,
                        "normalized_decision_margin": "",
                        "winner_stability_steps": 1,
                        "evidence_frames_used": 1,
                        "competing_hypothesis_survived": False,
                    }
                )
            committed_rows.extend(immediate.rows)
            position += 1
            continue

        event_number += 1
        event_id = f"{observations[0].video_id}:{variant}:{frame:06d}:{event_number:04d}"
        beam = [
            Hypothesis(
                outcome.state,
                outcome.objective,
                list(outcome.rows),
                list(outcome.pending_memory_updates),
                (outcome.option_rank,),
            )
            for outcome in probe
        ]
        beam.sort(key=_hypothesis_key)
        hypotheses_expanded = len(beam)
        peak = len(beam)
        previous_winner_rank = beam[0].trace[0]
        stability = 1
        deadline_position = min(
            len(frames) - 1, position + h41.max_decision_horizon
        )
        decision_position = deadline_position
        threshold_rule_met = False
        normalized_margin: float | str = ""
        runner_score: float | None = None
        for future_position in range(position + 1, deadline_position + 1):
            future_frame = frames[future_position]
            future_indices = by_frame.get(future_frame, [])
            future_observations = [observations[index] for index in future_indices]
            future_embeddings = [embeddings[index] for index in future_indices]
            expanded: list[Hypothesis] = []
            for hypothesis in beam:
                outcomes = _step_options(
                    model_name,
                    variant,
                    hypothesis.state,
                    future_observations,
                    future_embeddings,
                    future_frame,
                    h4,
                    allow_branch=True,
                    memory_mode=memory_mode,
                )
                hypotheses_expanded += len(outcomes)
                for outcome in outcomes:
                    expanded.append(
                        Hypothesis(
                            outcome.state,
                            hypothesis.score + outcome.objective,
                            hypothesis.rows + outcome.rows,
                            hypothesis.pending_memory_updates
                            + outcome.pending_memory_updates,
                            hypothesis.trace + (outcome.option_rank,),
                        )
                    )
            expanded.sort(key=_hypothesis_key)
            beam = expanded[: h4.beam_width]
            peak = max(peak, len(beam))
            winner = beam[0]
            winner_rank = winner.trace[0]
            if winner_rank == previous_winner_rank:
                stability += 1
            else:
                previous_winner_rank = winner_rank
                stability = 1
            runner_score = _best_competing_score(beam, winner)
            evidence_frames = future_position - position + 1
            normalized_margin = (
                ""
                if runner_score is None
                else (winner.score - runner_score) / evidence_frames
            )
            elapsed = future_frame - frame
            threshold_met = runner_score is None or float(normalized_margin) >= h41.decision_margin
            if (
                elapsed >= h41.min_decision_lag
                and stability >= h41.winner_stability_steps
                and threshold_met
            ):
                decision_position = future_position
                threshold_rule_met = True
                break

        beam.sort(key=_hypothesis_key)
        winner = beam[0]
        runner_score = _best_competing_score(beam, winner)
        evidence_frames = decision_position - position + 1
        normalized_margin = (
            "" if runner_score is None else (winner.score - runner_score) / evidence_frames
        )
        if memory_mode == "frozen":
            _commit_frozen_updates(winner.state, winner.pending_memory_updates, h4)
        decision_frame = frames[decision_position]
        raw_margin = "" if runner_score is None else winner.score - runner_score
        realized_horizon = decision_frame - frame
        early_commit = (
            threshold_rule_met and realized_horizon < h41.max_decision_horizon
        )
        forced = (
            not threshold_rule_met
            and realized_horizon >= h41.max_decision_horizon
        )
        forced_at_sequence_end = (
            not threshold_rule_met
            and realized_horizon < h41.max_decision_horizon
        )
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
            row["decision_score_margin"] = raw_margin
            row.update(
                {
                    "decision_policy": "adaptive_lag",
                    "early_commit": early_commit,
                    "forced_at_deadline": forced,
                    "forced_at_sequence_end": forced_at_sequence_end,
                    "threshold_rule_met": threshold_rule_met,
                    "normalized_decision_margin": normalized_margin,
                    "winner_stability_steps": stability,
                    "evidence_frames_used": evidence_frames,
                    "competing_hypothesis_survived": runner_score is not None,
                }
            )
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
                "realized_horizon": realized_horizon,
                "deferred_commit": True,
                "reason": (
                    "stable_margin_early_commit"
                    if early_commit
                    else "stable_margin_at_deadline"
                    if threshold_rule_met
                    else "forced_at_deadline"
                    if forced
                    else "forced_at_sequence_end"
                ),
                "component_track_count": len(conflict.rows),
                "component_detection_count": len(conflict.columns),
                "initial_assignment_margin": conflict.assignment_margin,
                "hypotheses_expanded": hypotheses_expanded,
                "hypothesis_count_peak": peak,
                "winning_score": winner.score,
                "runner_up_score": "" if runner_score is None else runner_score,
                "decision_score_margin": raw_margin,
                "decision_policy": "adaptive_lag",
                "early_commit": early_commit,
                "forced_at_deadline": forced,
                "forced_at_sequence_end": forced_at_sequence_end,
                "threshold_rule_met": threshold_rule_met,
                "normalized_decision_margin": normalized_margin,
                "winner_stability_steps": stability,
                "evidence_frames_used": evidence_frames,
                "competing_hypothesis_survived": runner_score is not None,
                "ground_truth_decision_input": False,
                "analysis_uses_gt": False,
                "final_test_read": False,
            }
        )
        position = decision_position + 1

    committed_rows.sort(key=lambda row: (row["frame_id"], row["observation_id"]))
    return committed_rows, events


def track_h41_sequence(
    model_name: str,
    variant: str,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    h4: H4Config,
    h41: H41Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one frozen H4.1 variant without reading GT in any decision."""
    if variant not in H41_VARIANTS:
        raise ValueError(f"Unknown H4.1 association variant: {variant}")
    if variant.startswith("adaptive_lag_"):
        return _adaptive_sequence(model_name, variant, observations, embeddings, h4, h41)
    rows, events = track_h4_sequence(
        model_name, variant, observations, embeddings, h4
    )
    return _enrich_control_rows(variant, rows, events)


__all__ = [
    "H41_ASSIGNMENT_EXTRA_FIELDS",
    "H41_PRIMARY_VARIANT",
    "H41_VARIANTS",
    "track_h41_sequence",
]
