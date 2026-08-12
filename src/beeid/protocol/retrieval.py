"""Frame-delta retrieval with full and spatially hard galleries."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np

from ..cache import open_cache
from ..config import ExperimentConfig
from ..data.manifest import load_manifest
from ..data.mot import Observation
from ..utils import atomic_write_json, atomic_write_text

QUERY_FIELDS = (
    "model", "split", "video_id", "query_observation_id", "candidate_frame", "track_id", "identity",
    "delta", "positive_observation_id", "predicted_observation_id", "max_hard_negative_observation_id",
    "rank1", "hard_rank1", "positive_similarity", "max_full_negative_similarity",
    "max_hard_negative_similarity", "margin",
    "hard_negative_count", "skip_reason", "bbox_area", "size_quartile", "clipped",
)


def size_quartiles(observations: Sequence[Observation]) -> tuple[tuple[float, float, float], dict[str, str]]:
    unique = {item.observation_id: item.bbox_area for item in observations}
    if not unique:
        raise ValueError("Cannot compute size quartiles without observations")
    thresholds = tuple(float(value) for value in np.quantile(list(unique.values()), [0.25, 0.5, 0.75]))

    def label(area: float) -> str:
        if area <= thresholds[0]:
            return "Q1"
        if area <= thresholds[1]:
            return "Q2"
        if area <= thresholds[2]:
            return "Q3"
        return "Q4"

    return thresholds, {observation_id: label(area) for observation_id, area in unique.items()}


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def evaluate_embeddings(
    model_name: str,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    deltas: Sequence[int],
    hard_negative_k: int,
) -> tuple[list[dict[str, Any]], tuple[float, float, float]]:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(observations):
        raise ValueError("Embedding rows must align exactly with observations")
    by_video_frame: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, observation in enumerate(observations):
        by_video_frame[(observation.video_id, observation.frame)].append(index)
    thresholds, quartile_labels = size_quartiles(observations)
    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(observations):
        for delta in deltas:
            candidate_frame = query.frame + delta
            candidates = by_video_frame.get((query.video_id, candidate_frame), [])
            positives = [index for index in candidates if observations[index].identity == query.identity]
            base: dict[str, Any] = {
                "model": model_name,
                "split": query.split,
                "video_id": query.video_id,
                "query_observation_id": query.observation_id,
                "candidate_frame": candidate_frame,
                "track_id": query.track_id,
                "identity": query.identity,
                "delta": delta,
                "positive_observation_id": "",
                "predicted_observation_id": "",
                "max_hard_negative_observation_id": "",
                "rank1": "",
                "hard_rank1": "",
                "positive_similarity": "",
                "max_full_negative_similarity": "",
                "max_hard_negative_similarity": "",
                "margin": "",
                "hard_negative_count": 0,
                "skip_reason": "",
                "bbox_area": query.bbox_area,
                "size_quartile": quartile_labels[query.observation_id],
                "clipped": query.clipped,
            }
            if len(positives) != 1:
                base["skip_reason"] = "no_positive_at_delta" if not positives else "multiple_positives_at_delta"
                rows.append(base)
                continue
            positive_index = positives[0]
            positive_similarity = _similarity(embeddings[query_index], embeddings[positive_index])
            negatives = [index for index in candidates if observations[index].identity != query.identity]
            full_negative_scores = [
                _similarity(embeddings[query_index], embeddings[index]) for index in negatives
            ]
            maximum_full_negative = max(full_negative_scores) if full_negative_scores else None
            full_rank1 = maximum_full_negative is None or positive_similarity > maximum_full_negative
            candidate_scores = [
                (index, _similarity(embeddings[query_index], embeddings[index])) for index in candidates
            ]
            maximum_score = max(score for _, score in candidate_scores)
            winners = [index for index, score in candidate_scores if score == maximum_score]
            predicted = observations[winners[0]].observation_id if len(winners) == 1 else "TIE"

            nearest_negatives = sorted(
                negatives,
                key=lambda index: (
                    (observations[index].center_x - query.center_x) ** 2
                    + (observations[index].center_y - query.center_y) ** 2,
                    observations[index].identity,
                    observations[index].observation_id,
                ),
            )[:hard_negative_k]
            hard_scores = [
                _similarity(embeddings[query_index], embeddings[index]) for index in nearest_negatives
            ]
            maximum_hard = max(hard_scores) if hard_scores else None
            maximum_hard_index = (
                nearest_negatives[hard_scores.index(maximum_hard)] if maximum_hard is not None else None
            )
            base.update(
                {
                    "positive_observation_id": observations[positive_index].observation_id,
                    "predicted_observation_id": predicted,
                    "max_hard_negative_observation_id": (
                        observations[maximum_hard_index].observation_id
                        if maximum_hard_index is not None else ""
                    ),
                    "rank1": full_rank1,
                    "hard_rank1": "" if maximum_hard is None else positive_similarity > maximum_hard,
                    "positive_similarity": positive_similarity,
                    "max_full_negative_similarity": "" if maximum_full_negative is None else maximum_full_negative,
                    "max_hard_negative_similarity": "" if maximum_hard is None else maximum_hard,
                    "margin": "" if maximum_hard is None else positive_similarity - maximum_hard,
                    "hard_negative_count": len(nearest_negatives),
                    "skip_reason": "no_hard_negative" if maximum_hard is None else "",
                }
            )
            rows.append(base)
    return rows, thresholds


def _metric_rows(query_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_video: list[dict[str, Any]] = []
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        groups[(str(row["model"]), int(row["delta"]), str(row["video_id"]))].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        full = [row for row in rows if row["rank1"] != ""]
        hard = [row for row in rows if row["hard_rank1"] != ""]
        positive = [float(row["positive_similarity"]) for row in full]
        hard_sim = [float(row["max_hard_negative_similarity"]) for row in hard]
        margins = [float(row["margin"]) for row in hard]
        return {
            "query_count": len(rows),
            "full_query_count": len(full),
            "hard_query_count": len(hard),
            "rank1": sum(bool(row["rank1"]) for row in full) / len(full) if full else "",
            "hard_rank1": sum(bool(row["hard_rank1"]) for row in hard) / len(hard) if hard else "",
            "mean_positive_similarity": mean(positive) if positive else "",
            "mean_max_hard_negative_similarity": mean(hard_sim) if hard_sim else "",
            "mean_margin": mean(margins) if margins else "",
            "no_positive_count": sum(row["skip_reason"] == "no_positive_at_delta" for row in rows),
            "no_hard_negative_count": sum(row["skip_reason"] == "no_hard_negative" for row in rows),
        }

    for (model, delta, video_id), rows in sorted(groups.items()):
        per_video.append({"model": model, "delta": delta, "video_id": video_id, **summarize(rows)})

    summary: list[dict[str, Any]] = []
    pooled_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        pooled_groups[(str(row["model"]), int(row["delta"]))].append(row)
    for (model, delta), rows in sorted(pooled_groups.items()):
        pooled = summarize(rows)
        video_values = [
            row for row in per_video if row["model"] == model and row["delta"] == delta
        ]
        rank_values = [float(row["rank1"]) for row in video_values if row["rank1"] != ""]
        hard_values = [float(row["hard_rank1"]) for row in video_values if row["hard_rank1"] != ""]
        summary.append(
            {
                "model": model,
                "delta": delta,
                **pooled,
                "video_macro_rank1_mean": mean(rank_values) if rank_values else "",
                "video_macro_rank1_population_std": pstdev(rank_values) if rank_values else "",
                "video_macro_hard_rank1_mean": mean(hard_values) if hard_values else "",
                "video_macro_hard_rank1_population_std": pstdev(hard_values) if hard_values else "",
                "video_count": len(video_values),
            }
        )
    return summary, per_video


def _size_rows(query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        groups[(str(row["model"]), int(row["delta"]), str(row["size_quartile"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, delta, quartile), rows in sorted(groups.items()):
        full = [row for row in rows if row["rank1"] != ""]
        hard = [row for row in rows if row["hard_rank1"] != ""]
        positive = [float(row["positive_similarity"]) for row in full]
        hard_similarity = [float(row["max_hard_negative_similarity"]) for row in hard]
        margins = [float(row["margin"]) for row in hard]
        output.append(
            {
                "model": model,
                "delta": delta,
                "size_quartile": quartile,
                "query_count": len(rows),
                "full_query_count": len(full),
                "hard_query_count": len(hard),
                "rank1": sum(bool(row["rank1"]) for row in full) / len(full) if full else "",
                "hard_rank1": sum(bool(row["hard_rank1"]) for row in hard) / len(hard) if hard else "",
                "mean_positive_similarity": mean(positive) if positive else "",
                "mean_max_hard_negative_similarity": mean(hard_similarity) if hard_similarity else "",
                "mean_margin": mean(margins) if margins else "",
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    selected_fields = tuple(fieldnames or (tuple(rows[0]) if rows else ()))
    buffer = io.StringIO(newline="")
    if selected_fields:
        writer = csv.DictWriter(buffer, fieldnames=selected_fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def evaluate(config: ExperimentConfig, model_names: Sequence[str]) -> list[dict[str, Any]]:
    observations = load_manifest(
        config.manifest_path, split=config.protocol.evaluation_split, valid_only=True
    )
    expected_ids = [item.observation_id for item in observations]
    if not config.cache_locations_path.is_file():
        raise RuntimeError(f"Cache locations file does not exist: {config.cache_locations_path}")
    locations = json.loads(config.cache_locations_path.read_text(encoding="utf-8"))
    all_rows: list[dict[str, Any]] = []
    thresholds_by_model: dict[str, list[float]] = {}
    for model_name in model_names:
        if model_name not in locations:
            raise RuntimeError(f"No extracted cache is registered for model {model_name}")
        cache = open_cache(Path(locations[model_name]), model_name)
        embeddings = cache.load_all(expected_ids, config.runtime.cache_shard_size)
        rows, thresholds = evaluate_embeddings(
            model_name,
            observations,
            embeddings,
            config.protocol.deltas,
            config.protocol.hard_negative_k,
        )
        all_rows.extend(rows)
        thresholds_by_model[model_name] = list(thresholds)
    summary, per_video = _metric_rows(all_rows)
    size = _size_rows(all_rows)
    _write_csv(config.paths.output_root / "query_results.csv", all_rows, QUERY_FIELDS)
    _write_csv(config.paths.output_root / "summary.csv", summary)
    _write_csv(config.paths.output_root / "per_video_results.csv", per_video)
    _write_csv(config.paths.output_root / "size_analysis.csv", size)
    atomic_write_json(
        config.paths.output_root / "size_quartile_thresholds.json",
        {
            "basis": "unique valid query observation raw bbox area in evaluation split",
            "thresholds_q25_q50_q75": next(iter(thresholds_by_model.values()), []),
            "evaluation_split": config.protocol.evaluation_split,
        },
    )
    return all_rows
