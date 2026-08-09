"""Refined, outcome-blind H2.5 memory-contamination mechanism experiment."""

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
from ..h2.diagnostics import history_consistency
from ..h2.features import load_h2_embeddings
from ..h2.signals import read_signal_rows
from ..utils import atomic_write_json, atomic_write_text
from .core import EXPECTED_STRATEGIES, h2_source_config, validate_h25_inputs
from .statistics import cluster_bootstrap


EVENT_FIELDS = (
    "event_id", "model", "event_type", "video_id", "identity", "event_frame",
    "event_observation_id", "contaminant_observation_id", "event_value", "event_threshold",
    "event_percentile", "combined_risk_percentile", "reliability_weight", "trusted_start_frame",
    "trusted_end_frame", "future_step_count",
)
TRAJECTORY_FIELDS = (
    "event_id", "model", "event_type", "strategy", "video_id", "identity", "event_frame",
    "event_observation_id", "contaminant_observation_id", "combined_risk_percentile",
    "reliability_weight", "event_update_alpha", "step", "frames_since_event",
    "target_observation_id", "positive_similarity", "oracle_positive_similarity",
    "similarity_gap_to_oracle", "rank1", "oracle_rank1", "induced_error_vs_oracle",
    "recovered_at_this_step",
)


def _write_csv(path: Any, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 0 or not math.isfinite(norm):
        raise RuntimeError("Cannot normalize a non-finite H2.5 memory template")
    return (value / norm).astype(np.float32, copy=False)


def ema_update(template: np.ndarray, feature: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("EMA alpha must be in [0,1]")
    return _normalize((1.0 - alpha) * template + alpha * feature)


def empirical_percentiles(values: Sequence[float]) -> list[float]:
    """Midrank empirical percentiles; independent of all retrieval outcomes."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Empirical percentile values must be a finite non-empty vector")
    ordered = np.sort(array)
    return [
        float((np.searchsorted(ordered, value, side="left") + np.searchsorted(ordered, value, side="right")) / (2.0 * len(ordered)))
        for value in array
    ]


def _axial_change(current: str, previous: str) -> float | None:
    if current == "" or previous == "":
        return None
    difference = abs(float(current) - float(previous)) % 180.0
    return min(difference, 180.0 - difference)


def _log_change(current: float | str, previous: float | str) -> float | None:
    if current == "" or previous == "":
        return None
    left, right = float(current), float(previous)
    if left <= 0 or right <= 0 or not math.isfinite(left) or not math.isfinite(right):
        return None
    return abs(math.log(left / right))


def _trusted_window(indices: Sequence[int], observations: Sequence[Observation], length: int) -> tuple[int, list[int]] | None:
    for start in range(0, len(indices) - length + 1):
        window = list(indices[start : start + length])
        frames = [observations[index].frame for index in window]
        if all(right - left == 1 for left, right in zip(frames, frames[1:])):
            return start + length, window
    return None


def nearest_same_frame_negative(
    event_index: int,
    candidates: Sequence[int],
    observations: Sequence[Observation],
) -> int | None:
    event = observations[event_index]
    negatives = [index for index in candidates if observations[index].identity != event.identity]
    if not negatives:
        return None
    return min(
        negatives,
        key=lambda index: (
            (observations[index].center_x - event.center_x) ** 2
            + (observations[index].center_y - event.center_y) ** 2,
            observations[index].identity,
            observations[index].observation_id,
        ),
    )


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


def _candidate_features(
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    signals: dict[str, dict[str, str]],
    history_length: int,
) -> dict[int, dict[str, float | None]]:
    history = history_consistency(observations, embeddings, history_length)
    by_identity: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        by_identity[item.identity].append(index)
    result: dict[int, dict[str, float | None]] = {}
    for indices in by_identity.values():
        indices.sort(key=lambda index: (observations[index].frame, observations[index].observation_id))
        for position, index in enumerate(indices):
            if position == 0:
                continue
            current = signals[observations[index].observation_id]
            previous = signals[observations[indices[position - 1]].observation_id]
            history_value = history[observations[index].observation_id][2]
            result[index] = {
                "history_outlier": None if history_value == "" else float(history_value),
                "bbox_scale_change": _log_change(current["bbox_area"], previous["bbox_area"]),
                "orientation_change_proxy": _axial_change(
                    current["bbox_orientation_proxy_deg"], previous["bbox_orientation_proxy_deg"]
                ),
                "sharpness_change": _log_change(
                    current["bbox_laplacian_variance"], previous["bbox_laplacian_variance"]
                ),
                "crowding_neighbor_raw": float(current["neighbor_count_wide"]),
                "crowding_iou_raw": float(current["max_bbox_iou"]),
            }
    return result


def _add_percentiles(features: dict[int, dict[str, float | None]]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for name in ("history_outlier", "bbox_scale_change", "orientation_change_proxy", "sharpness_change"):
        indices = [index for index, row in features.items() if row[name] is not None]
        values = [float(features[index][name]) for index in indices]
        percentiles = empirical_percentiles(values)
        for index, percentile in zip(indices, percentiles):
            features[index][f"{name}_percentile"] = percentile
        thresholds[name] = float(np.quantile(np.asarray(values), 0.75)) if values else float("nan")
    for raw_name in ("crowding_neighbor_raw", "crowding_iou_raw"):
        indices = list(features)
        percentiles = empirical_percentiles([float(features[index][raw_name]) for index in indices])
        for index, percentile in zip(indices, percentiles):
            features[index][f"{raw_name}_percentile"] = percentile
    for row in features.values():
        row["crowding"] = max(
            float(row["crowding_neighbor_raw_percentile"]),
            float(row["crowding_iou_raw_percentile"]),
        )
        row["crowding_percentile"] = float(row["crowding"])
    thresholds["crowding"] = 0.75
    return thresholds


def _window_rows(rows: Sequence[dict[str, Any]], low: int, high: int) -> list[dict[str, Any]]:
    return [row for row in rows if low <= int(row["step"]) <= high]


def _summaries(
    trajectories: Sequence[dict[str, Any]], windows: dict[str, list[int]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        grouped[(str(row["model"]), str(row["event_type"]), str(row["strategy"]))].append(row)
    strategy_summary: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        for window, bounds in windows.items():
            selected = _window_rows(rows, int(bounds[0]), int(bounds[1]))
            if not selected:
                continue
            event_ids = {str(row["event_id"]) for row in selected}
            recovered = {
                str(row["event_id"]) for row in selected if bool(row["recovered_at_this_step"])
            }
            recovery_steps = [
                int(row["step"]) for row in selected if bool(row["recovered_at_this_step"])
            ]
            strategy_summary.append(
                {
                    "model": key[0], "event_type": key[1], "strategy": key[2], "window": window,
                    "event_count": len(event_ids), "target_step_count": len(selected),
                    "rank1": mean(float(bool(row["rank1"])) for row in selected),
                    "mean_positive_similarity": mean(float(row["positive_similarity"]) for row in selected),
                    "mean_similarity_gap_to_oracle": mean(float(row["similarity_gap_to_oracle"]) for row in selected),
                    "induced_error_rate_vs_oracle": mean(float(bool(row["induced_error_vs_oracle"])) for row in selected),
                    "recovered_event_fraction": len(recovered) / len(event_ids),
                    "median_recovery_step": median(recovery_steps) if recovery_steps else "",
                }
            )

    lookup = {
        (row["event_id"], row["target_observation_id"], row["step"], row["strategy"]): row
        for row in trajectories
    }
    paired_detail: list[dict[str, Any]] = []
    comparison_strategies = ("skip_update", "reliability_weighted_update")
    for row in trajectories:
        if row["strategy"] not in comparison_strategies:
            continue
        unconditional = lookup[(row["event_id"], row["target_observation_id"], row["step"], "unconditional_update")]
        for window, bounds in windows.items():
            if not int(bounds[0]) <= int(row["step"]) <= int(bounds[1]):
                continue
            paired_detail.append(
                {
                    "model": row["model"], "event_type": row["event_type"],
                    "strategy": row["strategy"], "window": window,
                    "video_id": row["video_id"], "identity": row["identity"],
                    "event_id": row["event_id"], "target_observation_id": row["target_observation_id"],
                    "rank1_gain": float(bool(row["rank1"])) - float(bool(unconditional["rank1"])),
                    "similarity_gap_reduction": float(unconditional["similarity_gap_to_oracle"]) - float(row["similarity_gap_to_oracle"]),
                    "induced_error_reduction": float(bool(unconditional["induced_error_vs_oracle"])) - float(bool(row["induced_error_vs_oracle"])),
                }
            )
    paired_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_detail:
        paired_groups[(row["model"], row["event_type"], row["strategy"], row["window"])].append(row)
    paired_summary = [
        {
            "model": key[0], "event_type": key[1], "strategy": key[2], "window": key[3],
            "paired_step_count": len(rows),
            "event_count": len({row["event_id"] for row in rows}),
            "mean_rank1_gain_vs_unconditional": mean(row["rank1_gain"] for row in rows),
            "mean_similarity_gap_reduction_vs_unconditional": mean(row["similarity_gap_reduction"] for row in rows),
            "mean_induced_error_reduction_vs_unconditional": mean(row["induced_error_reduction"] for row in rows),
        }
        for key, rows in sorted(paired_groups.items())
    ]
    return strategy_summary, paired_summary, paired_detail


def run_h25_experiment(
    config: ExperimentConfig, model_names: Sequence[str]
) -> dict[str, Any]:
    observations, protocol, input_audit = validate_h25_inputs(config)
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("H2.5 model names must be a non-empty unique list")
    expected_ids = [item.observation_id for item in observations]
    h2_config = h2_source_config(config)
    primary = protocol["source"]["primary_variant"]
    selection = protocol["event_selection"]
    memory = protocol["memory"]
    reporting = protocol["reporting"]
    history_length = int(selection["history_length"])
    trusted_length = int(selection["trusted_history_length"])
    alpha = float(memory["ema_alpha"])
    horizon = int(memory["recovery_horizon"])
    tolerance = float(memory["recovery_tolerance"])
    min_weight = float(protocol["reliability"]["min_weight"])

    source_signal_rows = [
        row for row in read_signal_rows(config.h25_source_root / "h2_observation_signals.csv")
        if row["variant"] == primary
    ]
    signals = {row["observation_id"]: row for row in source_signal_rows}
    if set(signals) != set(expected_ids):
        raise RuntimeError("H2.5 primary signals do not exactly align with selected observations")
    by_identity: dict[str, list[int]] = defaultdict(list)
    by_frame: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        by_identity[item.identity].append(index)
        by_frame[(item.video_id, item.frame)].append(index)
    for indices in by_identity.values():
        indices.sort(key=lambda index: (observations[index].frame, observations[index].observation_id))

    events: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    threshold_metadata: dict[str, Any] = {
        "outcome_blind": True,
        "event_quantile": float(selection["event_quantile"]),
        "percentile_reference": selection["percentile_reference"],
        "models": {},
    }
    candidate_counts: dict[str, dict[str, int]] = {}
    for model_name in model_names:
        embeddings, cache = load_h2_embeddings(h2_config, model_name, primary, expected_ids)
        features = _candidate_features(observations, embeddings, signals, history_length)
        thresholds = _add_percentiles(features)
        threshold_metadata["models"][model_name] = {
            "thresholds": thresholds,
            "embedding_cache": str(cache.directory),
            "candidate_observation_count": len(features),
        }
        counts = {event_type: 0 for event_type in selection["event_types"]}
        for identity, indices in sorted(by_identity.items()):
            trusted = _trusted_window(indices, observations, trusted_length)
            if trusted is None:
                continue
            after_position, trusted_indices = trusted
            eligible: list[tuple[int, int, list[int]]] = []
            for event_position in range(after_position, len(indices) - 1):
                event_index = indices[event_position]
                negative = nearest_same_frame_negative(
                    event_index,
                    by_frame[(observations[event_index].video_id, observations[event_index].frame)],
                    observations,
                )
                future = indices[event_position + 1 : event_position + 1 + horizon]
                if negative is not None and future and event_index in features:
                    eligible.append((event_position, negative, future))
            if not eligible:
                continue
            chosen: list[tuple[str, int, int, list[int]]] = []
            for event_type in selection["event_types"]:
                if event_type == "wrong_identity_control":
                    event_position, negative, future = eligible[0]
                    chosen.append((event_type, event_position, negative, future))
                    continue
                threshold = thresholds[event_type]
                for event_position, negative, future in eligible:
                    value = features[indices[event_position]].get(event_type)
                    if value is not None and float(value) >= threshold:
                        chosen.append((event_type, event_position, negative, future))
                        break
            for event_type, event_position, negative_index, future in chosen:
                counts[event_type] += 1
                event_index = indices[event_position]
                row = features[event_index]
                percentiles = [
                    float(row[f"{name}_percentile"])
                    for name in ("history_outlier", "bbox_scale_change", "orientation_change_proxy", "sharpness_change", "crowding")
                    if row.get(f"{name}_percentile") is not None
                ]
                combined_risk = max(percentiles)
                reliability_weight = min(1.0, max(min_weight, 1.0 - combined_risk))
                event_value = combined_risk if event_type == "wrong_identity_control" else float(row[event_type])
                event_threshold = "" if event_type == "wrong_identity_control" else thresholds[event_type]
                event_percentile = combined_risk if event_type == "wrong_identity_control" else float(row[f"{event_type}_percentile"])
                event_id = f"{model_name}|{event_type}|{observations[event_index].observation_id}"

                template = _normalize(embeddings[trusted_indices].mean(axis=0))
                for intermediate in indices[after_position:event_position]:
                    template = ema_update(template, embeddings[intermediate], alpha)
                templates = {
                    "oracle_correct_update": ema_update(template, embeddings[event_index], alpha),
                    "unconditional_update": ema_update(template, embeddings[negative_index], alpha),
                    "skip_update": template.copy(),
                    "reliability_weighted_update": ema_update(
                        template, embeddings[negative_index], alpha * reliability_weight
                    ),
                }
                update_alphas = {
                    "oracle_correct_update": alpha,
                    "unconditional_update": alpha,
                    "skip_update": 0.0,
                    "reliability_weighted_update": alpha * reliability_weight,
                }
                recovered: dict[str, bool] = {strategy: False for strategy in EXPECTED_STRATEGIES}
                event_record = {
                    "event_id": event_id, "model": model_name, "event_type": event_type,
                    "video_id": observations[event_index].video_id, "identity": identity,
                    "event_frame": observations[event_index].frame,
                    "event_observation_id": observations[event_index].observation_id,
                    "contaminant_observation_id": observations[negative_index].observation_id,
                    "event_value": event_value, "event_threshold": event_threshold,
                    "event_percentile": event_percentile,
                    "combined_risk_percentile": combined_risk,
                    "reliability_weight": reliability_weight,
                    "trusted_start_frame": observations[trusted_indices[0]].frame,
                    "trusted_end_frame": observations[trusted_indices[-1]].frame,
                    "future_step_count": len(future),
                }
                events.append(event_record)
                for step, positive_index in enumerate(future, start=1):
                    candidates = by_frame[(observations[positive_index].video_id, observations[positive_index].frame)]
                    oracle_template = templates["oracle_correct_update"]
                    oracle_similarity = float(np.dot(oracle_template, embeddings[positive_index]))
                    oracle_rank1 = _rank1(
                        oracle_template, positive_index, candidates, observations, embeddings
                    )
                    for strategy in EXPECTED_STRATEGIES:
                        active = templates[strategy]
                        similarity = float(np.dot(active, embeddings[positive_index]))
                        rank1 = _rank1(active, positive_index, candidates, observations, embeddings)
                        gap = oracle_similarity - similarity
                        recovered_now = bool(
                            strategy != "oracle_correct_update"
                            and not recovered[strategy]
                            and abs(gap) <= tolerance
                            and rank1
                        )
                        if recovered_now:
                            recovered[strategy] = True
                        trajectories.append(
                            {
                                **{key: event_record[key] for key in (
                                    "event_id", "model", "event_type", "video_id", "identity",
                                    "event_frame", "event_observation_id", "contaminant_observation_id",
                                    "combined_risk_percentile", "reliability_weight",
                                )},
                                "strategy": strategy,
                                "event_update_alpha": update_alphas[strategy],
                                "step": step,
                                "frames_since_event": observations[positive_index].frame - observations[event_index].frame,
                                "target_observation_id": observations[positive_index].observation_id,
                                "positive_similarity": similarity,
                                "oracle_positive_similarity": oracle_similarity,
                                "similarity_gap_to_oracle": gap,
                                "rank1": rank1,
                                "oracle_rank1": oracle_rank1,
                                "induced_error_vs_oracle": oracle_rank1 and not rank1,
                                "recovered_at_this_step": recovered_now,
                            }
                        )
                    for strategy in EXPECTED_STRATEGIES:
                        templates[strategy] = ema_update(
                            templates[strategy], embeddings[positive_index], alpha
                        )
        candidate_counts[model_name] = counts

    strategy_summary, paired_summary, paired_detail = _summaries(
        trajectories, reporting["windows"]
    )
    bootstrap = cluster_bootstrap(
        paired_detail,
        replicates=int(reporting["bootstrap_replicates"]),
        seed=int(reporting["bootstrap_seed"]),
    ) if paired_detail else []
    output = config.paths.output_root
    _write_csv(output / "h25_events.csv", events, EVENT_FIELDS)
    _write_csv(output / "h25_strategy_trajectories.csv", trajectories, TRAJECTORY_FIELDS)
    _write_csv(output / "h25_strategy_summary.csv", strategy_summary, tuple(strategy_summary[0]) if strategy_summary else (
        "model", "event_type", "strategy", "window", "event_count", "target_step_count",
        "rank1", "mean_positive_similarity", "mean_similarity_gap_to_oracle",
        "induced_error_rate_vs_oracle", "recovered_event_fraction", "median_recovery_step",
    ))
    _write_csv(output / "h25_paired_summary.csv", paired_summary, tuple(paired_summary[0]) if paired_summary else (
        "model", "event_type", "strategy", "window", "paired_step_count", "event_count",
        "mean_rank1_gain_vs_unconditional", "mean_similarity_gap_reduction_vs_unconditional",
        "mean_induced_error_reduction_vs_unconditional",
    ))
    _write_csv(output / "h25_cluster_bootstrap.csv", bootstrap, tuple(bootstrap[0]) if bootstrap else (
        "model", "event_type", "strategy", "window", "metric", "cluster_unit",
        "cluster_count", "sample_count", "point_estimate", "ci95_low", "ci95_high",
        "bootstrap_replicates", "bootstrap_seed",
    ))
    threshold_metadata["candidate_event_counts"] = candidate_counts
    atomic_write_json(output / "h25_event_thresholds.json", threshold_metadata)
    result = {
        "status": "completed",
        "models": list(model_names),
        "event_count": len(events),
        "trajectory_row_count": len(trajectories),
        "strategy_summary_row_count": len(strategy_summary),
        "paired_summary_row_count": len(paired_summary),
        "cluster_bootstrap_row_count": len(bootstrap),
        "strategies": list(EXPECTED_STRATEGIES),
        "event_types": list(selection["event_types"]),
        "input_audit": input_audit,
        "outcome_blind_event_selection": True,
        "controlled_sequence": "GT identity trajectories; no tracker output is used",
        "final_test_read": False,
    }
    atomic_write_json(output / "h25_experiment_metadata.json", result)
    return result
