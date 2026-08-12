"""Candidate-graph reachability audit performed before any H6 fitting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .data import H6Window, build_windows


def audit_candidate_reachability(
    observations: Sequence[Any], indices: Sequence[int], *, window_length: int,
    window_stride: int, max_frame_gap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure whether adjacent same-identity observations are present in any dense window graph."""
    windows = build_windows(observations, indices, window_length, window_stride)
    window_membership: set[tuple[int, int]] = set()
    for window in windows:
        members = list(window.indices)
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                gap = observations[right].frame - observations[left].frame
                if 0 < gap <= max_frame_gap:
                    window_membership.add((left, right))
    by_track: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in indices:
        item = observations[index]
        by_track[(item.video_id, item.identity)].append(index)
    rows: list[dict[str, Any]] = []
    for (video_id, identity), members in sorted(by_track.items()):
        ordered = sorted(members, key=lambda index: observations[index].frame)
        for left, right in zip(ordered, ordered[1:]):
            gap = observations[right].frame - observations[left].frame
            within_gap = 0 < gap <= max_frame_gap
            reachable = within_gap and (left, right) in window_membership
            rows.append({
                "video_id": video_id, "identity": identity,
                "left_observation_id": observations[left].observation_id,
                "right_observation_id": observations[right].observation_id,
                "frame_gap": gap, "within_max_frame_gap": within_gap,
                "candidate_edge_reachable": reachable,
                "candidate_policy": "all_cross_frame_pairs_within_gap_no_top_k",
                "ground_truth_used_for_audit_only": True, "method_decision_input": False,
                "final_test_read": False,
            })
    eligible = [row for row in rows if row["within_max_frame_gap"]]
    videos = sorted({str(row["video_id"]) for row in eligible})
    per_video = {
        video: sum(bool(row["candidate_edge_reachable"]) for row in eligible if row["video_id"] == video)
        / max(1, sum(1 for row in eligible if row["video_id"] == video))
        for video in videos
    }
    recall = sum(bool(row["candidate_edge_reachable"]) for row in eligible) / max(1, len(eligible))
    summary = {
        "status": "completed", "adjacent_identity_edge_count": len(rows),
        "eligible_edge_count": len(eligible), "reachable_edge_count": sum(
            bool(row["candidate_edge_reachable"]) for row in eligible
        ),
        "edge_recall": recall, "per_video_edge_recall": per_video,
        "window_count": len(windows), "candidate_top_k": None,
        "candidate_policy": "dense_all_pairs_within_gap",
        "ground_truth_role": "project_train_pretraining_audit_only",
        "method_decision_input": False, "final_test_read": False,
    }
    return rows, summary
