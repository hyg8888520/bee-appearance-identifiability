"""H3 identity metrics for exact GT-box detections, following TrackEval semantics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from .assignment import maximum_weight_matching


TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"


def _identity_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gt_ids = sorted({str(row["gt_identity"]) for row in rows})
    tracker_ids = sorted({str(row["predicted_track_id"]) for row in rows})
    gt_lookup = {value: index for index, value in enumerate(gt_ids)}
    tracker_lookup = {value: index for index, value in enumerate(tracker_ids)}
    counts = np.zeros((len(gt_ids), len(tracker_ids)), dtype=np.float64)
    gt_count = np.zeros(len(gt_ids), dtype=np.float64)
    tracker_count = np.zeros(len(tracker_ids), dtype=np.float64)
    for row in rows:
        gt_index = gt_lookup[str(row["gt_identity"])]
        tracker_index = tracker_lookup[str(row["predicted_track_id"])]
        counts[gt_index, tracker_index] += 1
        gt_count[gt_index] += 1
        tracker_count[tracker_index] += 1
    matches = maximum_weight_matching(counts, minimum_score=1.0)
    idtp = int(sum(counts[row, column] for row, column in matches))
    total = len(rows)
    idfn = total - idtp
    idfp = total - idtp
    idf1 = idtp / max(1.0, idtp + 0.5 * idfn + 0.5 * idfp)

    association_sum = 0.0
    for gt_index in range(len(gt_ids)):
        for tracker_index in range(len(tracker_ids)):
            match_count = counts[gt_index, tracker_index]
            if match_count <= 0:
                continue
            association_accuracy = match_count / max(
                1.0,
                gt_count[gt_index] + tracker_count[tracker_index] - match_count,
            )
            association_sum += match_count * association_accuracy
    ass_a = association_sum / max(1, total)

    by_gt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gt[str(row["gt_identity"])].append(row)
    id_switches = 0
    for identity_rows in by_gt.values():
        ordered = sorted(
            identity_rows,
            key=lambda row: (int(row["frame_id"]), str(row["observation_id"])),
        )
        previous: str | None = None
        for row in ordered:
            current = str(row["predicted_track_id"])
            if previous is not None and current != previous:
                id_switches += 1
            previous = current

    # With exact GT detections every GT detection has a matched tracker detection.
    # CLEAR fragmentation is therefore zero even if the matched tracker ID changes;
    # those changes are counted as IDSW instead.
    return {
        "detection_count": total,
        "gt_identity_count": len(gt_ids),
        "predicted_identity_count": len(tracker_ids),
        "IDTP": idtp,
        "IDFN": idfn,
        "IDFP": idfp,
        "IDF1": idf1,
        "AssA": ass_a,
        "DetA": 1.0 if total else 0.0,
        "HOTA": math.sqrt(ass_a) if total else 0.0,
        "IDSW": id_switches,
        "Frag": 0,
    }


def _diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if bool(row["eligible_memory_update"])]
    accepted = [row for row in eligible if bool(row["memory_update_accepted"])]
    reliabilities = [
        float(row["reliability"]) for row in eligible if row["reliability"] != ""
    ]
    alphas = [
        float(row["effective_memory_alpha"])
        for row in eligible
        if row["effective_memory_alpha"] != ""
    ]
    assignment_scores = [
        float(row["association_score"])
        for row in eligible
        if row["association_score"] != ""
    ]
    gt_track_lengths = Counter(str(row["gt_identity"]) for row in rows)
    return {
        "eligible_memory_update_count": len(eligible),
        "accepted_memory_update_count": len(accepted),
        "memory_update_acceptance_rate": len(accepted) / len(eligible) if eligible else 0.0,
        "mean_reliability": float(np.mean(reliabilities)) if reliabilities else "",
        "mean_effective_memory_alpha": float(np.mean(alphas)) if alphas else "",
        "mean_assignment_score": float(np.mean(assignment_scores)) if assignment_scores else "",
        "mean_gt_track_observations": (
            float(np.mean(list(gt_track_lengths.values()))) if gt_track_lengths else 0.0
        ),
    }


def summarize_gt_assignments(
    assignment_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        groups[(str(row["model"]), str(row["variant"]), str(row["video_id"]))].append(row)
    per_video: list[dict[str, Any]] = []
    for (model, variant, video_id), rows in sorted(groups.items()):
        per_video.append(
            {
                "model": model,
                "variant": variant,
                "stage": "gt_detection_boxes",
                "video_id": video_id,
                **_identity_statistics(rows),
                **_diagnostics(rows),
            }
        )

    pooled: list[dict[str, Any]] = []
    model_variants = sorted({(row["model"], row["variant"]) for row in assignment_rows})
    for model, variant in model_variants:
        selected = [
            dict(row) for row in assignment_rows
            if row["model"] == model and row["variant"] == variant
        ]
        # Predicted IDs restart per video. Prefix both sides before pooled global
        # identity assignment so no cross-video mapping is possible.
        for row in selected:
            row["gt_identity"] = f"{row['video_id']}::{row['gt_identity']}"
            row["predicted_track_id"] = f"{row['video_id']}::{row['predicted_track_id']}"
        metrics = _identity_statistics(selected)
        diagnostics = _diagnostics(selected)
        video_metrics = [
            row for row in per_video if row["model"] == model and row["variant"] == variant
        ]
        pooled.append(
            {
                "model": model,
                "variant": variant,
                "stage": "gt_detection_boxes",
                "aggregation": "micro_pooled",
                **metrics,
                **diagnostics,
                "video_count": len(video_metrics),
                "video_macro_AssA_mean": float(
                    np.mean([float(row["AssA"]) for row in video_metrics])
                ),
                "video_macro_AssA_population_std": float(
                    np.std([float(row["AssA"]) for row in video_metrics])
                ),
                "video_macro_IDF1_mean": float(
                    np.mean([float(row["IDF1"]) for row in video_metrics])
                ),
                "video_macro_IDF1_population_std": float(
                    np.std([float(row["IDF1"]) for row in video_metrics])
                ),
                "video_macro_HOTA_mean": float(
                    np.mean([float(row["HOTA"]) for row in video_metrics])
                ),
                "video_macro_HOTA_population_std": float(
                    np.std([float(row["HOTA"]) for row in video_metrics])
                ),
            }
        )
    return per_video, pooled


def paired_video_differences(
    per_video_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["model"]), str(row["variant"]), str(row["video_id"])): row
        for row in per_video_rows
    }
    output: list[dict[str, Any]] = []
    for row in per_video_rows:
        if row["variant"] == "baseline_association":
            continue
        baseline = lookup.get((row["model"], "baseline_association", row["video_id"]))
        if baseline is None:
            raise RuntimeError("H3 paired metrics are missing a baseline video row")
        output.append(
            {
                "model": row["model"],
                "variant": row["variant"],
                "stage": row["stage"],
                "video_id": row["video_id"],
                "AssA_gain_vs_baseline": float(row["AssA"]) - float(baseline["AssA"]),
                "IDF1_gain_vs_baseline": float(row["IDF1"]) - float(baseline["IDF1"]),
                "HOTA_gain_vs_baseline": float(row["HOTA"]) - float(baseline["HOTA"]),
                "IDSW_reduction_vs_baseline": int(baseline["IDSW"]) - int(row["IDSW"]),
                "Frag_reduction_vs_baseline": int(baseline["Frag"]) - int(row["Frag"]),
            }
        )
    return output
