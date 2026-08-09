"""Query-level H2 diagnostics across frozen crop and input interventions."""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..data.mot import Observation
from ..protocol.retrieval import evaluate_embeddings
from ..utils import atomic_write_text
from .core import require_h2, validate_h2_inputs
from .features import load_h2_embeddings
from .signals import read_signal_rows


FACTOR_FIELDS = (
    "bbox_area", "bbox_width", "bbox_height", "aspect_ratio", "foreground_fraction",
    "context_fraction", "resized_bbox_width", "resized_bbox_height", "resized_bbox_area",
    "approx_patch_coverage", "bbox_touches_image_boundary", "variant_clipped", "visibility",
    "laplacian_variance", "gradient_energy", "gradient_orientation_proxy_deg",
    "orientation_coherence", "mean_luminance", "luminance_std",
    "bbox_laplacian_variance", "bbox_gradient_energy", "bbox_orientation_proxy_deg",
    "bbox_orientation_coherence", "max_bbox_iou",
    "overlap_count", "nearest_center_distance_normalized", "neighbor_count_near",
    "neighbor_count_wide", "track_observation_count", "track_duration_frames",
    "track_age_frames", "track_remaining_frames", "track_age_fraction", "manual_blur",
    "manual_occlusion", "manual_pose_deg",
)

BASE_QUERY_FIELDS = (
    "model", "variant", "pixel_view", "crop_expansion", "input_size", "split", "video_id",
    "query_observation_id", "candidate_frame", "track_id", "identity", "delta",
    "positive_observation_id", "predicted_observation_id", "max_hard_negative_observation_id",
    "rank1", "hard_rank1", "positive_similarity", "max_full_negative_similarity",
    "max_hard_negative_similarity", "margin", "hard_negative_count", "skip_reason",
    "query_history_count", "query_history_similarity", "query_history_outlier",
    "positive_history_count", "positive_history_similarity", "positive_history_outlier",
    "absolute_orientation_change_proxy_deg", "absolute_log_bbox_area_ratio",
    "absolute_laplacian_log_ratio", "absolute_manual_pose_change_deg",
)
QUERY_DIAGNOSTIC_FIELDS = BASE_QUERY_FIELDS + tuple(f"query_{field}" for field in FACTOR_FIELDS)


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


def read_query_diagnostics(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"H2 query diagnostics do not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != QUERY_DIAGNOSTIC_FIELDS:
            raise RuntimeError(f"Unexpected H2 query diagnostic schema: {path}")
        return list(reader)


def history_consistency(
    observations: Sequence[Observation], embeddings: np.ndarray, history_length: int
) -> dict[str, tuple[int, float | str, float | str]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        groups[item.identity].append(index)
    result: dict[str, tuple[int, float | str, float | str]] = {}
    for indices in groups.values():
        indices.sort(key=lambda index: (observations[index].frame, observations[index].observation_id))
        for position, index in enumerate(indices):
            previous = indices[max(0, position - history_length) : position]
            if not previous:
                result[observations[index].observation_id] = (0, "", "")
                continue
            prototype = embeddings[previous].mean(axis=0)
            norm = float(np.linalg.norm(prototype))
            if norm <= 0 or not np.isfinite(norm):
                raise RuntimeError("Cannot construct a finite trajectory-history prototype")
            prototype = prototype / norm
            similarity = float(np.dot(embeddings[index], prototype))
            result[observations[index].observation_id] = (
                len(previous), similarity, 1.0 - similarity
            )
    return result


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _orientation_difference(left: Any, right: Any) -> float | str:
    first = _float_or_none(left)
    second = _float_or_none(right)
    if first is None or second is None:
        return ""
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def _pose_difference(left: Any, right: Any) -> float | str:
    first = _float_or_none(left)
    second = _float_or_none(right)
    if first is None or second is None:
        return ""
    difference = abs(first - second) % 360.0
    return min(difference, 360.0 - difference)


def _absolute_log_ratio(left: Any, right: Any) -> float | str:
    first = _float_or_none(left)
    second = _float_or_none(right)
    if first is None or second is None or first <= 0 or second <= 0:
        return ""
    return abs(math.log(second / first))


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    full = [row for row in rows if row["rank1"] != ""]
    hard = [row for row in rows if row["hard_rank1"] != ""]
    return {
        "query_count": len(rows),
        "full_query_count": len(full),
        "hard_query_count": len(hard),
        "rank1": sum(bool(row["rank1"]) for row in full) / len(full) if full else "",
        "hard_rank1": (
            sum(bool(row["hard_rank1"]) for row in hard) / len(hard) if hard else ""
        ),
        "mean_positive_similarity": (
            mean(float(row["positive_similarity"]) for row in full) if full else ""
        ),
        "mean_max_hard_negative_similarity": (
            mean(float(row["max_hard_negative_similarity"]) for row in hard) if hard else ""
        ),
        "mean_margin": mean(float(row["margin"]) for row in hard) if hard else "",
        "no_positive_count": sum(row["skip_reason"] == "no_positive_at_delta" for row in rows),
    }


def _context_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    by_video: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["model"]), str(row["variant"]), int(row["delta"]))
        grouped[key].append(row)
        by_video[(*key, str(row["video_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, variant, delta), values in sorted(grouped.items()):
        video_scores = [
            float(summary["rank1"])
            for key, video_rows in by_video.items()
            if key[:3] == (model, variant, delta)
            and (summary := _summarize(video_rows))["rank1"] != ""
        ]
        first = values[0]
        output.append(
            {
                "model": model,
                "variant": variant,
                "pixel_view": first["pixel_view"],
                "crop_expansion": first["crop_expansion"],
                "input_size": first["input_size"],
                "delta": delta,
                **_summarize(values),
                "video_macro_rank1_mean": mean(video_scores) if video_scores else "",
                "video_macro_rank1_population_std": pstdev(video_scores) if video_scores else "",
                "video_count": len(video_scores),
            }
        )
    return output


def _paired_ablation(
    rows: Sequence[dict[str, Any]], primary_variant: str
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["model"]), str(row["variant"]), int(row["delta"]),
            str(row["query_observation_id"]),
        ): row
        for row in rows
        if row["rank1"] != ""
    }
    groups: dict[tuple[str, str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for (model, variant, delta, observation_id), row in lookup.items():
        if variant == primary_variant:
            continue
        primary = lookup.get((model, primary_variant, delta, observation_id))
        if primary is not None:
            groups[(model, variant, delta)].append((primary, row))
    output: list[dict[str, Any]] = []
    for (model, variant, delta), pairs in sorted(groups.items()):
        differences = [int(bool(right["rank1"])) - int(bool(left["rank1"])) for left, right in pairs]
        margin_pairs = [
            (float(left["margin"]), float(right["margin"]))
            for left, right in pairs
            if left["margin"] != "" and right["margin"] != ""
        ]
        output.append(
            {
                "model": model,
                "reference_variant": primary_variant,
                "variant": variant,
                "delta": delta,
                "paired_query_count": len(pairs),
                "variant_minus_reference_rank1": mean(differences) if differences else "",
                "variant_wins": sum(value > 0 for value in differences),
                "reference_wins": sum(value < 0 for value in differences),
                "same_outcome": sum(value == 0 for value in differences),
                "mean_margin_difference": (
                    mean(right - left for left, right in margin_pairs) if margin_pairs else ""
                ),
            }
        )
    return output


def evaluate_h2(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    h2 = require_h2(config)
    observations, _ = validate_h2_inputs(config)
    expected_ids = [item.observation_id for item in observations]
    signal_rows = read_signal_rows(config.paths.output_root / "h2_observation_signals.csv")
    signals = {
        (row["variant"], row["observation_id"]): row for row in signal_rows
    }
    expected_signal_count = len(observations) * len(h2.variants)
    if len(signals) != expected_signal_count:
        raise RuntimeError(
            f"H2 signal rows are incomplete: expected {expected_signal_count}, got {len(signals)}"
        )
    all_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for variant in h2.variants:
            embeddings, _ = load_h2_embeddings(
                config, model_name, variant.name, expected_ids
            )
            history = history_consistency(observations, embeddings, h2.history_length)
            query_rows, _ = evaluate_embeddings(
                model_name,
                observations,
                embeddings,
                config.protocol.deltas,
                config.protocol.hard_negative_k,
            )
            for row in query_rows:
                query_signal = signals[(variant.name, str(row["query_observation_id"]))]
                positive_signal = signals.get(
                    (variant.name, str(row["positive_observation_id"]))
                )
                query_history = history[str(row["query_observation_id"])]
                positive_history = history.get(str(row["positive_observation_id"]), (0, "", ""))
                output = {
                    **row,
                    "model": model_name,
                    "variant": variant.name,
                    "pixel_view": variant.pixel_view,
                    "crop_expansion": variant.crop_expansion,
                    "input_size": variant.input_size,
                    "query_history_count": query_history[0],
                    "query_history_similarity": query_history[1],
                    "query_history_outlier": query_history[2],
                    "positive_history_count": positive_history[0],
                    "positive_history_similarity": positive_history[1],
                    "positive_history_outlier": positive_history[2],
                    "absolute_orientation_change_proxy_deg": (
                        _orientation_difference(
                            query_signal["bbox_orientation_proxy_deg"],
                            positive_signal["bbox_orientation_proxy_deg"] if positive_signal else "",
                        )
                    ),
                    "absolute_log_bbox_area_ratio": _absolute_log_ratio(
                        query_signal["bbox_area"],
                        positive_signal["bbox_area"] if positive_signal else "",
                    ),
                    "absolute_laplacian_log_ratio": _absolute_log_ratio(
                        query_signal["bbox_laplacian_variance"],
                        positive_signal["bbox_laplacian_variance"] if positive_signal else "",
                    ),
                    "absolute_manual_pose_change_deg": _pose_difference(
                        query_signal["manual_pose_deg"],
                        positive_signal["manual_pose_deg"] if positive_signal else "",
                    ),
                }
                output.update(
                    {f"query_{field}": query_signal[field] for field in FACTOR_FIELDS}
                )
                all_rows.append(output)
    all_rows.sort(
        key=lambda row: (
            str(row["model"]), str(row["variant"]), str(row["video_id"]),
            str(row["query_observation_id"]), int(row["delta"]),
        )
    )
    _write_csv(
        config.paths.output_root / "h2_query_diagnostics.csv",
        all_rows,
        QUERY_DIAGNOSTIC_FIELDS,
    )
    context = _context_summary(all_rows)
    paired = _paired_ablation(all_rows, h2.primary_variant)
    _write_csv(config.paths.output_root / "h2_context_ablation.csv", context)
    _write_csv(config.paths.output_root / "h2_paired_ablation.csv", paired)
    return {
        "status": "completed",
        "models": list(model_names),
        "variants": [item.name for item in h2.variants],
        "query_row_count": len(all_rows),
        "context_summary_rows": len(context),
        "paired_ablation_rows": len(paired),
    }
