"""Controlled EMA trajectory-template contamination experiments on fixed GT tracks."""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from statistics import mean, median
from typing import Any, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..data.mot import Observation
from ..utils import atomic_write_json, atomic_write_text
from .core import require_h2, validate_h2_inputs
from .features import load_h2_embeddings
from .signals import read_signal_rows


TRAJECTORY_FIELDS = (
    "model", "contamination_type", "video_id", "identity", "event_observation_id",
    "contaminant_observation_id", "event_frame", "step", "frames_since_event",
    "target_observation_id", "clean_positive_similarity", "contaminated_positive_similarity",
    "similarity_degradation", "clean_rank1", "contaminated_rank1", "induced_error",
    "recovered_at_this_step", "event_bbox_laplacian_variance", "event_max_bbox_iou",
    "event_neighbor_count_wide", "event_clipped",
)


def _write_csv(path: Any, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    selected = tuple(fields or (tuple(rows[0]) if rows else ()))
    buffer = io.StringIO(newline="")
    if selected:
        writer = csv.DictWriter(
            buffer, fieldnames=selected, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 0 or not math.isfinite(norm):
        raise RuntimeError("Cannot normalize a non-finite trajectory template")
    return (value / norm).astype(np.float32, copy=False)


def _ema(template: np.ndarray, feature: np.ndarray, alpha: float) -> np.ndarray:
    return _normalize((1.0 - alpha) * template + alpha * feature)


def _rank1(
    template: np.ndarray,
    positive_index: int,
    candidate_indices: Sequence[int],
    observations: Sequence[Observation],
    embeddings: np.ndarray,
) -> bool:
    positive = float(np.dot(template, embeddings[positive_index]))
    negatives = [
        float(np.dot(template, embeddings[index]))
        for index in candidate_indices
        if observations[index].identity != observations[positive_index].identity
    ]
    return not negatives or positive > max(negatives)


def _quality_rules(
    signals: dict[str, dict[str, str]], low_quantile: float
) -> tuple[dict[str, bool], dict[str, float]]:
    sharpness = np.asarray(
        [float(row["bbox_laplacian_variance"]) for row in signals.values()], dtype=np.float64
    )
    overlap = np.asarray([float(row["max_bbox_iou"]) for row in signals.values()], dtype=np.float64)
    density = np.asarray(
        [float(row["neighbor_count_wide"]) for row in signals.values()], dtype=np.float64
    )
    thresholds = {
        "sharpness_low": float(np.quantile(sharpness, low_quantile)),
        "sharpness_median": float(np.quantile(sharpness, 0.5)),
        "overlap_high": float(np.quantile(overlap, 1.0 - low_quantile)),
        "density_high": float(np.quantile(density, 1.0 - low_quantile)),
    }
    result: dict[str, bool] = {}
    for observation_id, row in signals.items():
        manual_blur = float(row["manual_blur"]) if row["manual_blur"] else 0.0
        manual_occlusion = float(row["manual_occlusion"]) if row["manual_occlusion"] else 0.0
        high_overlap = (
            thresholds["overlap_high"] > 0
            and float(row["max_bbox_iou"]) > thresholds["overlap_high"]
        )
        high_density = (
            thresholds["density_high"] > 0
            and float(row["neighbor_count_wide"]) > thresholds["density_high"]
        )
        result[observation_id] = bool(
            float(row["bbox_laplacian_variance"]) < thresholds["sharpness_low"]
            or high_overlap
            or high_density
            or row["variant_clipped"].lower() == "true"
            or manual_blur >= 0.5
            or manual_occlusion >= 0.5
        )
    return result, thresholds


def _trusted_window(
    indices: Sequence[int],
    observations: Sequence[Observation],
    low_quality: dict[str, bool],
    length: int,
) -> tuple[int, list[int]] | None:
    for start in range(0, len(indices) - length):
        window = list(indices[start : start + length])
        frames = [observations[index].frame for index in window]
        if all(right - left == 1 for left, right in zip(frames, frames[1:])) and all(
            not low_quality[observations[index].observation_id] for index in window
        ):
            return start + length, window
    return None


def run_memory_contamination(
    config: ExperimentConfig, model_names: Sequence[str]
) -> dict[str, Any]:
    h2 = require_h2(config)
    observations, _ = validate_h2_inputs(config)
    expected_ids = [item.observation_id for item in observations]
    by_identity: dict[str, list[int]] = defaultdict(list)
    by_frame: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        by_identity[item.identity].append(index)
        by_frame[(item.video_id, item.frame)].append(index)
    for indices in by_identity.values():
        indices.sort(key=lambda index: (observations[index].frame, observations[index].observation_id))

    primary_signal_rows = [
        row
        for row in read_signal_rows(config.paths.output_root / "h2_observation_signals.csv")
        if row["variant"] == h2.primary_variant
    ]
    signals = {row["observation_id"]: row for row in primary_signal_rows}
    if set(signals) != set(expected_ids):
        raise RuntimeError("Primary-variant H2 signals do not align with the source observations")
    low_quality, quality_thresholds = _quality_rules(
        signals, h2.contamination.low_quality_quantile
    )
    blur_variants = [
        item
        for item in h2.variants
        if item.pixel_view == "gaussian_blur"
        and item.crop_expansion
        == next(item.crop_expansion for item in h2.variants if item.name == h2.primary_variant)
        and item.input_size
        == next(item.input_size for item in h2.variants if item.name == h2.primary_variant)
    ]
    blur_variant = blur_variants[0].name if blur_variants else None
    trajectories: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    for model_name in model_names:
        embeddings, _ = load_h2_embeddings(
            config, model_name, h2.primary_variant, expected_ids
        )
        blurred_embeddings = None
        if blur_variant is not None:
            blurred_embeddings, _ = load_h2_embeddings(
                config, model_name, blur_variant, expected_ids
            )
        for identity, indices in sorted(by_identity.items()):
            trusted = _trusted_window(
                indices,
                observations,
                low_quality,
                h2.contamination.trusted_history_length,
            )
            if trusted is None:
                continue
            after_position, trusted_indices = trusted
            later = indices[after_position:]
            if len(later) < 2:
                continue
            template = _normalize(embeddings[trusted_indices].mean(axis=0))
            events: list[tuple[str, int, int, np.ndarray]] = []
            low_events = [
                index for index in later[:-1] if low_quality[observations[index].observation_id]
            ]
            if low_events:
                event_index = low_events[0]
                events.append(
                    ("observed_low_quality", event_index, event_index, embeddings[event_index])
                )
            if blurred_embeddings is not None:
                event_index = later[0]
                events.append(
                    ("synthetic_gaussian_blur", event_index, event_index, blurred_embeddings[event_index])
                )
            wrong_event: tuple[int, int] | None = None
            for event_index in later[:-1]:
                negatives = [
                    candidate
                    for candidate in by_frame[
                        (observations[event_index].video_id, observations[event_index].frame)
                    ]
                    if observations[candidate].identity != identity
                ]
                if negatives:
                    negative_index = min(
                        negatives,
                        key=lambda candidate: (
                            (observations[candidate].center_x - observations[event_index].center_x) ** 2
                            + (observations[candidate].center_y - observations[event_index].center_y) ** 2,
                            observations[candidate].identity,
                        ),
                    )
                    wrong_event = (event_index, negative_index)
                    break
            if wrong_event is not None:
                event_index, negative_index = wrong_event
                events.append(
                    ("wrong_identity", event_index, negative_index, embeddings[negative_index])
                )

            for contamination_type, event_index, contaminant_index, contaminant in events:
                event_position = indices.index(event_index)
                future = indices[event_position + 1 : event_position + 1 + h2.contamination.recovery_horizon]
                if not future:
                    continue
                event_template = template.copy()
                for intermediate_index in indices[after_position:event_position]:
                    event_template = _ema(
                        event_template,
                        embeddings[intermediate_index],
                        h2.contamination.ema_alpha,
                    )
                clean_template = event_template.copy()
                if contamination_type in {"synthetic_gaussian_blur", "wrong_identity"}:
                    clean_template = _ema(
                        clean_template,
                        embeddings[event_index],
                        h2.contamination.ema_alpha,
                    )
                contaminated_template = _ema(
                    event_template, contaminant, h2.contamination.ema_alpha
                )
                recovered_step: int | None = None
                event_signal = signals[observations[event_index].observation_id]
                event_row_indices: list[int] = []
                for step, positive_index in enumerate(future, start=1):
                    candidate_indices = by_frame[
                        (observations[positive_index].video_id, observations[positive_index].frame)
                    ]
                    clean_similarity = float(
                        np.dot(clean_template, embeddings[positive_index])
                    )
                    contaminated_similarity = float(
                        np.dot(contaminated_template, embeddings[positive_index])
                    )
                    clean_rank1 = _rank1(
                        clean_template,
                        positive_index,
                        candidate_indices,
                        observations,
                        embeddings,
                    )
                    contaminated_rank1 = _rank1(
                        contaminated_template,
                        positive_index,
                        candidate_indices,
                        observations,
                        embeddings,
                    )
                    degradation = clean_similarity - contaminated_similarity
                    recovered = bool(
                        recovered_step is None
                        and abs(degradation) <= h2.contamination.recovery_tolerance
                        and contaminated_rank1
                    )
                    if recovered:
                        recovered_step = step
                    event_row_indices.append(len(trajectories))
                    trajectories.append(
                        {
                            "model": model_name,
                            "contamination_type": contamination_type,
                            "video_id": observations[event_index].video_id,
                            "identity": identity,
                            "event_observation_id": observations[event_index].observation_id,
                            "contaminant_observation_id": observations[contaminant_index].observation_id,
                            "event_frame": observations[event_index].frame,
                            "step": step,
                            "frames_since_event": observations[positive_index].frame
                            - observations[event_index].frame,
                            "target_observation_id": observations[positive_index].observation_id,
                            "clean_positive_similarity": clean_similarity,
                            "contaminated_positive_similarity": contaminated_similarity,
                            "similarity_degradation": degradation,
                            "clean_rank1": clean_rank1,
                            "contaminated_rank1": contaminated_rank1,
                            "induced_error": clean_rank1 and not contaminated_rank1,
                            "recovered_at_this_step": recovered,
                            "event_bbox_laplacian_variance": event_signal["bbox_laplacian_variance"],
                            "event_max_bbox_iou": event_signal["max_bbox_iou"],
                            "event_neighbor_count_wide": event_signal["neighbor_count_wide"],
                            "event_clipped": event_signal["variant_clipped"],
                        }
                    )
                    clean_template = _ema(
                        clean_template, embeddings[positive_index], h2.contamination.ema_alpha
                    )
                    contaminated_template = _ema(
                        contaminated_template,
                        embeddings[positive_index],
                        h2.contamination.ema_alpha,
                    )
                event_records.append(
                    {
                        "model": model_name,
                        "contamination_type": contamination_type,
                        "video_id": observations[event_index].video_id,
                        "identity": identity,
                        "event_observation_id": observations[event_index].observation_id,
                        "contaminant_observation_id": observations[contaminant_index].observation_id,
                        "evaluated_steps": len(event_row_indices),
                        "recovery_step": "" if recovered_step is None else recovered_step,
                    }
                )

    _write_csv(
        config.paths.output_root / "h2_memory_trajectories.csv",
        trajectories,
        TRAJECTORY_FIELDS,
    )
    grouped_events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in event_records:
        grouped_events[(str(event["model"]), str(event["contamination_type"]))].append(event)
    for row in trajectories:
        grouped_rows[(str(row["model"]), str(row["contamination_type"]))].append(row)
    summary: list[dict[str, Any]] = []
    for key, events in sorted(grouped_events.items()):
        rows = grouped_rows[key]
        recovery = [int(event["recovery_step"]) for event in events if event["recovery_step"] != ""]
        summary.append(
            {
                "model": key[0],
                "contamination_type": key[1],
                "event_count": len(events),
                "trajectory_step_count": len(rows),
                "mean_similarity_degradation": mean(
                    float(row["similarity_degradation"]) for row in rows
                ),
                "induced_error_count": sum(bool(row["induced_error"]) for row in rows),
                "induced_error_rate": sum(bool(row["induced_error"]) for row in rows) / len(rows),
                "recovered_event_count": len(recovery),
                "not_recovered_fraction": (len(events) - len(recovery)) / len(events),
                "median_recovery_step": median(recovery) if recovery else "",
            }
        )
    _write_csv(config.paths.output_root / "h2_memory_events.csv", event_records)
    _write_csv(config.paths.output_root / "h2_memory_summary.csv", summary)
    result = {
        "status": "completed",
        "event_count": len(event_records),
        "trajectory_row_count": len(trajectories),
        "summary_row_count": len(summary),
        "quality_thresholds": quality_thresholds,
        "blur_variant": blur_variant,
        "controlled_sequence": "GT identity trajectories; no tracker output is used",
        "control_policy": {
            "observed_low_quality": "skip the event update",
            "synthetic_gaussian_blur": "update with the unblurred same-identity feature",
            "wrong_identity": "update with the correct same-frame identity feature",
        },
        "parameters": {
            "trusted_history_length": h2.contamination.trusted_history_length,
            "ema_alpha": h2.contamination.ema_alpha,
            "recovery_horizon": h2.contamination.recovery_horizon,
            "recovery_tolerance": h2.contamination.recovery_tolerance,
            "low_quality_quantile": h2.contamination.low_quality_quantile,
        },
    }
    atomic_write_json(config.paths.output_root / "h2_memory_metadata.json", result)
    return result
