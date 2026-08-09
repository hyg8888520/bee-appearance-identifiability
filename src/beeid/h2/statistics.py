"""Cluster-aware and tie-aware statistics for H2 diagnostics."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np


FACTOR_SPECS: tuple[dict[str, str], ...] = (
    {"column": "query_bbox_area", "name": "bbox_area", "direction": "higher_reliable"},
    {"column": "query_aspect_ratio", "name": "aspect_ratio", "direction": "descriptive_only"},
    {"column": "query_approx_patch_coverage", "name": "approx_patch_coverage", "direction": "higher_reliable"},
    {"column": "query_bbox_laplacian_variance", "name": "bbox_laplacian_variance", "direction": "higher_reliable"},
    {"column": "query_bbox_gradient_energy", "name": "bbox_gradient_energy", "direction": "descriptive_only"},
    {"column": "query_bbox_orientation_coherence", "name": "bbox_orientation_coherence", "direction": "higher_reliable"},
    {"column": "query_max_bbox_iou", "name": "max_bbox_iou", "direction": "higher_unreliable"},
    {"column": "query_overlap_count", "name": "overlap_count", "direction": "higher_unreliable"},
    {"column": "query_nearest_center_distance_normalized", "name": "nearest_center_distance_normalized", "direction": "higher_reliable"},
    {"column": "query_neighbor_count_near", "name": "neighbor_count_near", "direction": "higher_unreliable"},
    {"column": "query_neighbor_count_wide", "name": "neighbor_count_wide", "direction": "higher_unreliable"},
    {"column": "query_variant_clipped", "name": "variant_clipped", "direction": "higher_unreliable", "kind": "binary"},
    {"column": "query_visibility", "name": "visibility", "direction": "higher_reliable"},
    {"column": "query_track_age_fraction", "name": "track_age_fraction", "direction": "descriptive_only"},
    {"column": "query_history_similarity", "name": "history_similarity", "direction": "higher_reliable"},
    {"column": "query_history_outlier", "name": "history_outlier", "direction": "higher_unreliable"},
    {"column": "absolute_orientation_change_proxy_deg", "name": "orientation_change_proxy", "direction": "higher_unreliable"},
    {"column": "absolute_log_bbox_area_ratio", "name": "bbox_scale_change", "direction": "higher_unreliable"},
    {"column": "absolute_laplacian_log_ratio", "name": "sharpness_change", "direction": "higher_unreliable"},
    {"column": "absolute_manual_pose_change_deg", "name": "manual_pose_change", "direction": "higher_unreliable"},
    {"column": "query_manual_blur", "name": "manual_blur", "direction": "higher_unreliable"},
    {"column": "query_manual_occlusion", "name": "manual_occlusion", "direction": "higher_unreliable"},
)


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text == "true":
        return 1.0
    if text == "false":
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"Not a boolean value: {value!r}")


def quantile_bins(values: Sequence[float], bin_count: int) -> tuple[list[float], list[str]]:
    if not values:
        raise ValueError("Cannot bin an empty factor")
    edges = [
        float(value)
        for value in np.quantile(np.asarray(values, dtype=np.float64), np.arange(1, bin_count) / bin_count)
    ]
    labels = [f"Q{index + 1}" for index in range(bin_count)]
    return edges, labels


def assign_quantile(value: float, edges: Sequence[float]) -> str:
    return f"Q{int(np.searchsorted(np.asarray(edges), value, side='left')) + 1}"


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def roc_auc(scores: Sequence[float], positive: Sequence[bool]) -> float | None:
    if len(scores) != len(positive) or not scores:
        return None
    labels = np.asarray(positive, dtype=bool)
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum = float(ranks[labels].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    if float(left_ranks.std()) == 0 or float(right_ranks.std()) == 0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def factor_analysis(
    rows: Sequence[dict[str, Any]], bin_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    factor_rows: list[dict[str, Any]] = []
    predictiveness: list[dict[str, Any]] = []
    definitions: dict[str, Any] = {
        "outcome_blind_binning": True,
        "binning": f"quantiles of unique query observations within model/variant; {bin_count} requested bins",
        "factors": list(FACTOR_SPECS),
        "thresholds": {},
    }
    model_variants = sorted({(str(row["model"]), str(row["variant"])) for row in rows})
    for model, variant in model_variants:
        subset = [row for row in rows if row["model"] == model and row["variant"] == variant]
        for spec in FACTOR_SPECS:
            column = spec["column"]
            unique: dict[str, float] = {}
            for row in subset:
                value = numeric(row.get(column))
                if value is not None:
                    unique.setdefault(str(row["query_observation_id"]), value)
            if not unique:
                continue
            if spec.get("kind") == "binary":
                edges: list[float] = []
                bins = {observation_id: "true" if value >= 0.5 else "false" for observation_id, value in unique.items()}
            else:
                edges, _ = quantile_bins(list(unique.values()), bin_count)
                bins = {
                    observation_id: assign_quantile(value, edges)
                    for observation_id, value in unique.items()
                }
            definition_key = f"{model}::{variant}::{spec['name']}"
            definitions["thresholds"][definition_key] = edges
            grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
            for row in subset:
                bin_label = bins.get(str(row["query_observation_id"]))
                if bin_label is not None:
                    grouped[(int(row["delta"]), bin_label)].append(row)
            for (delta, bin_label), values in sorted(grouped.items()):
                full = [row for row in values if row["rank1"] != ""]
                hard = [row for row in values if row["margin"] != ""]
                by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in full:
                    by_video[str(row["video_id"])].append(row)
                video_scores = [
                    sum(boolean(row["rank1"]) for row in video_rows) / len(video_rows)
                    for video_rows in by_video.values()
                ]
                factor_rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "delta": delta,
                        "factor": spec["name"],
                        "direction": spec["direction"],
                        "bin": bin_label,
                        "query_count": len(values),
                        "full_query_count": len(full),
                        "rank1": (
                            sum(boolean(row["rank1"]) for row in full) / len(full)
                            if full else ""
                        ),
                        "mean_margin": (
                            mean(float(row["margin"]) for row in hard) if hard else ""
                        ),
                        "mean_positive_similarity": (
                            mean(float(row["positive_similarity"]) for row in full)
                            if full else ""
                        ),
                        "video_macro_rank1_mean": mean(video_scores) if video_scores else "",
                        "video_macro_rank1_population_std": (
                            pstdev(video_scores) if video_scores else ""
                        ),
                        "video_count": len(video_scores),
                    }
                )
            if spec["direction"] == "descriptive_only":
                continue
            for delta in sorted({int(row["delta"]) for row in subset}):
                values = [
                    row
                    for row in subset
                    if int(row["delta"]) == delta
                    and row["rank1"] != ""
                    and numeric(row.get(column)) is not None
                ]
                raw_scores = [float(numeric(row[column])) for row in values]  # type: ignore[arg-type]
                unreliability = (
                    raw_scores
                    if spec["direction"] == "higher_unreliable"
                    else [-value for value in raw_scores]
                )
                failures = [not boolean(row["rank1"]) for row in values]
                margin_pairs = [
                    (score, -float(row["margin"]))
                    for score, row in zip(unreliability, values)
                    if row["margin"] != ""
                ]
                predictiveness.append(
                    {
                        "model": model,
                        "variant": variant,
                        "delta": delta,
                        "factor": spec["name"],
                        "unreliability_direction": spec["direction"],
                        "query_count": len(values),
                        "failure_count": sum(failures),
                        "failure_roc_auc": (
                            "" if (auc := roc_auc(unreliability, failures)) is None else auc
                        ),
                        "spearman_unreliability_vs_negative_margin": (
                            ""
                            if not margin_pairs
                            or (
                                correlation := spearman(
                                    [item[0] for item in margin_pairs],
                                    [item[1] for item in margin_pairs],
                                )
                            )
                            is None
                            else correlation
                        ),
                    }
                )
    return factor_rows, predictiveness, definitions


def _seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def clustered_bootstrap(
    rows: Sequence[dict[str, Any]], replicates: int, seed: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["rank1"] != "":
            groups[(str(row["model"]), str(row["variant"]), int(row["delta"]))].append(row)
    for key, values in sorted(groups.items()):
        for cluster_unit, cluster_column, aggregation in (
            ("video", "video_id", "macro_mean"),
            ("identity", "identity", "pooled_micro"),
        ):
            clusters: dict[str, tuple[int, int]] = {}
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in values:
                grouped[str(row[cluster_column])].append(row)
            for cluster, cluster_rows in grouped.items():
                clusters[cluster] = (
                    sum(boolean(row["rank1"]) for row in cluster_rows), len(cluster_rows)
                )
            labels = sorted(clusters)
            rng = np.random.default_rng(_seed(seed, f"{key}:{cluster_unit}"))
            samples: list[float] = []
            for _ in range(replicates):
                selected = rng.choice(labels, size=len(labels), replace=True)
                if aggregation == "macro_mean":
                    samples.append(mean(clusters[str(label)][0] / clusters[str(label)][1] for label in selected))
                else:
                    successes = sum(clusters[str(label)][0] for label in selected)
                    count = sum(clusters[str(label)][1] for label in selected)
                    samples.append(successes / count)
            point = (
                mean(successes / count for successes, count in clusters.values())
                if aggregation == "macro_mean"
                else sum(item[0] for item in clusters.values()) / sum(item[1] for item in clusters.values())
            )
            output.append(
                {
                    "model": key[0],
                    "variant": key[1],
                    "delta": key[2],
                    "cluster_unit": cluster_unit,
                    "aggregation": aggregation,
                    "cluster_count": len(labels),
                    "query_count": len(values),
                    "point_rank1": point,
                    "bootstrap_mean_rank1": mean(samples),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "replicates": replicates,
                    "seed": seed,
                }
            )
    return output


def paired_clustered_bootstrap(
    rows: Sequence[dict[str, Any]],
    primary_variant: str,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["model"]), str(row["variant"]), int(row["delta"]),
            str(row["query_observation_id"]),
        ): row
        for row in rows
        if row["rank1"] != ""
    }
    comparisons: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for (model, variant, delta, observation_id), row in lookup.items():
        if variant == primary_variant:
            continue
        primary = lookup.get((model, primary_variant, delta, observation_id))
        if primary is None:
            continue
        comparisons[(model, variant, delta)].append(
            {
                "video_id": row["video_id"],
                "identity": row["identity"],
                "difference": int(boolean(row["rank1"])) - int(boolean(primary["rank1"])),
            }
        )
    output: list[dict[str, Any]] = []
    for key, values in sorted(comparisons.items()):
        for cluster_unit, aggregation in (("video", "macro_mean"), ("identity", "pooled_micro")):
            grouped: dict[str, list[int]] = defaultdict(list)
            for row in values:
                grouped[str(row[f"{cluster_unit}_id"] if cluster_unit == "video" else row["identity"])].append(
                    int(row["difference"])
                )
            labels = sorted(grouped)
            rng = np.random.default_rng(
                _seed(seed, f"paired:{key}:{cluster_unit}:{primary_variant}")
            )
            samples: list[float] = []
            for _ in range(replicates):
                selected = rng.choice(labels, size=len(labels), replace=True)
                if aggregation == "macro_mean":
                    samples.append(mean(mean(grouped[str(label)]) for label in selected))
                else:
                    sampled_values = [
                        value for label in selected for value in grouped[str(label)]
                    ]
                    samples.append(mean(sampled_values))
            point = (
                mean(mean(cluster_values) for cluster_values in grouped.values())
                if aggregation == "macro_mean"
                else mean(int(row["difference"]) for row in values)
            )
            output.append(
                {
                    "model": key[0],
                    "reference_variant": primary_variant,
                    "variant": key[1],
                    "delta": key[2],
                    "cluster_unit": cluster_unit,
                    "aggregation": aggregation,
                    "cluster_count": len(labels),
                    "paired_query_count": len(values),
                    "point_rank1_difference": point,
                    "bootstrap_mean_difference": mean(samples),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "replicates": replicates,
                    "seed": seed,
                }
            )
    return output
