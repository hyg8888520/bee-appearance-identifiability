"""Run the four frozen H3 association variants on development GT detections."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..utils import atomic_write_json, atomic_write_text, sha256_file
from .core import require_h3, validate_h3_inputs
from .features import load_h3_embeddings
from .metrics import paired_video_differences, summarize_gt_assignments
from .signals import read_h3_signal_rows
from .thresholds import load_h3_thresholds
from .tracker import H3_VARIANTS, track_gt_sequence


ASSIGNMENT_FIELDS = (
    "model", "variant", "stage", "video_id", "frame_id", "observation_id",
    "gt_identity", "gt_track_id", "predicted_track_id", "matched_existing_track",
    "association_score", "appearance_similarity", "motion_score",
    "normalized_motion_distance", "reliability", "effective_memory_alpha",
    "memory_update_accepted", "eligible_memory_update",
    "risk_identity_history_outlier", "risk_bbox_scale_change",
    "risk_orientation_change_proxy", "risk_sharpness_change",
    "risk_crowding_overlap",
)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    selected = tuple(fields or (tuple(rows[0]) if rows else ()))
    buffer = io.StringIO(newline="")
    if selected:
        writer = csv.DictWriter(
            buffer, fieldnames=selected, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _aligned_signals(config: ExperimentConfig, expected_ids: Sequence[str]) -> list[dict[str, str]]:
    path = config.paths.output_root / "h3_observation_signals.csv"
    rows = read_h3_signal_rows(path)
    lookup = {row["observation_id"]: row for row in rows}
    if set(lookup) != set(expected_ids):
        raise RuntimeError("H3 signal table does not match the frozen selected observations")
    return [lookup[identifier] for identifier in expected_ids]


def _write_mot_results(
    root: Path,
    assignment_rows: Sequence[dict[str, Any]],
    observation_lookup: dict[str, Any],
) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        groups[(row["model"], row["variant"], row["video_id"])].append(row)
    for (model, variant, video_id), rows in groups.items():
        lines: list[str] = []
        for row in sorted(rows, key=lambda item: (item["frame_id"], item["predicted_track_id"])):
            observation = observation_lookup[row["observation_id"]]
            lines.append(
                ",".join(
                    [
                        str(observation.frame),
                        str(row["predicted_track_id"]),
                        f"{observation.raw_x:.6f}",
                        f"{observation.raw_y:.6f}",
                        f"{observation.raw_w:.6f}",
                        f"{observation.raw_h:.6f}",
                        "1", "-1", "-1", "-1",
                    ]
                )
            )
        atomic_write_text(root / model / variant / f"{video_id}.txt", "\n".join(lines) + "\n")


def run_h3_tracking(
    config: ExperimentConfig, model_names: Sequence[str]
) -> dict[str, Any]:
    h3 = require_h3(config)
    if h3.stage != "gt_detection_boxes":
        raise RuntimeError(
            "fixed_detector_boxes requires a frozen detector-result adapter and is "
            "SERVER_VALIDATION_PENDING; run the GT-box H3 stage first"
        )
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("H3 tracking requires unique model names")
    inputs = validate_h3_inputs(config)
    expected_ids = [item.observation_id for item in inputs.observations]
    signals = _aligned_signals(config, expected_ids)
    thresholds = load_h3_thresholds(config, model_names)
    development_indices = [
        index
        for index, item in enumerate(inputs.observations)
        if inputs.partition_by_observation[item.observation_id]
        == "development_validation"
    ]
    if not development_indices:
        raise RuntimeError("H3 tracking found no development_validation observations")
    by_video: dict[str, list[int]] = defaultdict(list)
    for index in development_indices:
        by_video[inputs.observations[index].video_id].append(index)
    all_rows: list[dict[str, Any]] = []
    cache_fingerprints: dict[str, str] = {}
    for model_name in model_names:
        embeddings, cache = load_h3_embeddings(config, model_name, inputs)
        recorded = thresholds["models"][model_name].get("cache_fingerprint")
        if recorded != cache.fingerprint:
            raise RuntimeError(
                f"H3 thresholds for {model_name} were fit with a different feature cache"
            )
        cache_fingerprints[model_name] = cache.fingerprint
        for video_id in sorted(by_video):
            indices = sorted(
                by_video[video_id],
                key=lambda index: (
                    inputs.observations[index].frame,
                    inputs.observations[index].center_x,
                    inputs.observations[index].center_y,
                ),
            )
            video_observations = [inputs.observations[index] for index in indices]
            video_embeddings = np.asarray(embeddings[indices], dtype=np.float32)
            video_signals = [signals[index] for index in indices]
            for variant in H3_VARIANTS:
                all_rows.extend(
                    track_gt_sequence(
                        model_name,
                        variant,
                        video_observations,
                        video_embeddings,
                        video_signals,
                        thresholds,
                        h3,
                    )
                )
    all_rows.sort(
        key=lambda row: (
            row["model"], row["variant"], row["video_id"],
            row["frame_id"], row["observation_id"],
        )
    )
    per_video, summary = summarize_gt_assignments(all_rows)
    paired = paired_video_differences(per_video)
    output = config.paths.output_root
    _write_csv(output / "h3_assignments.csv", all_rows, ASSIGNMENT_FIELDS)
    _write_csv(output / "h3_per_video_metrics.csv", per_video)
    _write_csv(output / "h3_summary.csv", summary)
    _write_csv(output / "h3_paired_video_metrics.csv", paired)
    observation_lookup = {item.observation_id: item for item in inputs.observations}
    _write_mot_results(output / "h3_mot_results", all_rows, observation_lookup)
    state = {
        "status": "completed",
        "stage": h3.stage,
        "models": list(model_names),
        "variants": list(H3_VARIANTS),
        "assignment_row_count": len(all_rows),
        "per_video_metric_row_count": len(per_video),
        "summary_row_count": len(summary),
        "paired_video_row_count": len(paired),
        "development_videos": sorted(by_video),
        "cache_fingerprints": cache_fingerprints,
        "threshold_artifact_sha256": sha256_file(output / "h3_thresholds.json"),
        "input_audit": inputs.audit,
        "metric_scope": (
            "Exact GT detections: direct detection correspondence; TrackEval-equivalent "
            "identity association formulas; DetA=1 and Frag=0 by construction"
        ),
        "final_test_read": False,
    }
    atomic_write_json(output / "h3_tracking_metadata.json", state)
    return state
