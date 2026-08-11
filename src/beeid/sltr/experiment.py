"""Resumable SLTR audit, project-train fit, and development-only causal tracking."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from ..config import ExperimentConfig
from ..h3.metrics import summarize_gt_assignments
from ..h4.io import write_csv
from ..utils import atomic_write_json, atomic_write_text, canonical_json, sha256_file, sha256_text
from . import SLTR_MODELS, SLTR_PRIMARY_VARIANT, SLTR_VARIANTS
from .core import load_sltr_embeddings, require_sltr, validate_sltr_inputs
from .selector import (
    LogisticModel, SelectorError, choose_oof_threshold, feature_matrix,
    fit_l2_logistic, group_oof_probabilities, validate_feature_names,
)
from .tracker import (
    CounterfactualEvent, build_counterfactual_events, deterministic_frequency_choice,
    track_selective_sequence,
)


AUDIT_FIELDS = (
    "model", "partition", "video_id", "event_id", "start_frame",
    "component_observation_ids", "component_size", "label_available",
    "label_unavailable_reason", "a_correct_observations", "b_correct_observations",
    "a_idsw", "b_idsw", "utility_b_minus_a", "positive", "features_json",
    "counterfactual_a", "counterfactual_b", "component_policy",
    "offline_label_uses_gt", "method_decision_uses_gt", "final_test_read",
)

ASSIGNMENT_FIELDS = (
    "model", "variant", "stage", "video_id", "frame_id", "observation_id",
    "gt_identity", "gt_track_id", "predicted_track_id", "matched_existing_track",
    "association_score", "appearance_similarity", "motion_score",
    "normalized_motion_distance", "eligible_memory_update", "memory_update_accepted",
    "memory_update_committed", "memory_update_used_for_branch_scoring",
    "effective_memory_alpha", "reliability", "event_id", "sltr_event",
    "sltr_selected_branch", "sltr_selection_reason", "sltr_horizon",
    "sltr_component_size", "sltr_ground_truth_decision_input",
    "sltr_exact_a_fallback", "final_test_read",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_progress(output: Path, stage: str, **payload: Any) -> None:
    atomic_write_json(output / "logs" / f"sltr_{stage}_progress.json", {
        "stage": stage, "updated_at": _now(), "final_test_read": False, **payload,
    })


def _job_signature(payload: dict[str, Any]) -> str:
    return sha256_text(canonical_json(payload))


def _load_signed_job(path: Path, signature: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("status") != "completed":
        return None
    if value.get("signature") != signature or value.get("signature_sha256") != _job_signature(signature):
        return None
    body = {key: value.get(key) for key in ("baseline_rows", "audit_rows")}
    if value.get("payload_sha256") != sha256_text(canonical_json(body)):
        return None
    if value.get("final_test_read") is not False:
        return None
    return value


def _write_signed_job(path: Path, signature: dict[str, Any], baseline_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    body = {"baseline_rows": baseline_rows, "audit_rows": audit_rows}
    value = {
        "status": "completed", "signature": signature,
        "signature_sha256": _job_signature(signature),
        **body, "payload_sha256": sha256_text(canonical_json(body)),
        "final_test_read": False,
    }
    atomic_write_json(path, value)
    return value


def _load_signed_tracking_job(path: Path, signature: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("status") != "completed" or value.get("signature") != signature:
        return None
    if value.get("signature_sha256") != _job_signature(signature) or value.get("final_test_read") is not False:
        return None
    body = {key: value.get(key) for key in ("rows", "interventions")}
    if not isinstance(body["rows"], list) or not isinstance(body["interventions"], list):
        return None
    if value.get("payload_sha256") != sha256_text(canonical_json(body)):
        return None
    return value


def _write_signed_tracking_job(path: Path, signature: dict[str, Any], rows: list[dict[str, Any]], interventions: list[dict[str, Any]]) -> dict[str, Any]:
    body = {"rows": rows, "interventions": interventions}
    value = {
        "status": "completed", "signature": signature,
        "signature_sha256": _job_signature(signature), **body,
        "payload_sha256": sha256_text(canonical_json(body)), "final_test_read": False,
    }
    atomic_write_json(path, value)
    return value


def _by_partition_video(inputs: Any) -> dict[tuple[str, str], list[int]]:
    result: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(inputs.observations):
        partition = inputs.h3_inputs.partition_by_observation[item.observation_id]
        result[(partition, item.video_id)].append(index)
    for indices in result.values():
        indices.sort(key=lambda i: (
            inputs.observations[i].frame, inputs.observations[i].center_x,
            inputs.observations[i].center_y, inputs.observations[i].observation_id,
        ))
    return result


def _attach_h3_observable_features(events: Sequence[CounterfactualEvent], inputs: Any) -> None:
    """Use only read-only H3 observation-quality/crowding signals, never their IDs."""
    fields = (
        "bbox_area", "bbox_orientation_proxy_deg", "bbox_laplacian_variance",
        "max_bbox_iou", "neighbor_count_wide",
    )
    for event in events:
        values: dict[str, list[float]] = {field: [] for field in fields}
        for observation_id in event.component_observation_ids:
            signal = inputs.signals_by_observation[observation_id]
            for field in fields:
                value = float(signal[field])
                if not np.isfinite(value):
                    raise RuntimeError(f"H3 observable signal is non-finite: {field}")
                values[field].append(value)
        event.features.update({
            f"h3_component_mean_{field}": float(np.mean(items))
            for field, items in values.items()
        })


def run_sltr_audit(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    """Build offline labelled counterfactuals; this stage never fits a selector."""
    sltr = require_sltr(config)
    if len(set(model_names)) != len(model_names) or set(model_names) - set(SLTR_MODELS):
        raise ValueError("SLTR audit requires unique required real backbone names")
    inputs = validate_sltr_inputs(config)
    groups = _by_partition_video(inputs)
    output = config.paths.output_root
    all_baseline: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    cache_audit: dict[str, Any] = {}
    total = len(model_names) * len(groups)
    complete = reused = 0
    _write_progress(output, "audit", status="running", completed_jobs=0, total_jobs=total)
    for model_name in model_names:
        embeddings, cache = load_sltr_embeddings(config, model_name, inputs)
        cache_audit[model_name] = cache
        for (partition, video_id), indices in sorted(groups.items()):
            observations = [inputs.observations[index] for index in indices]
            values = np.asarray(embeddings[indices], dtype=np.float32)
            signature = {
                "implementation": "beeid.sltr.audit:v1", "model": model_name,
                "partition": partition, "video_id": video_id,
                "observation_ids": [item.observation_id for item in observations],
                "cache_fingerprint": cache["fingerprint"], "protocol_sha256": inputs.audit["protocol_sha256"],
                "horizon": sltr.horizon, "final_test_read": False,
            }
            path = output / "work" / "audit" / model_name / partition / f"{video_id}.json"
            job = _load_signed_job(path, signature)
            if job is None:
                baseline, events = build_counterfactual_events(
                    model_name, observations, values, config.h4, horizon=sltr.horizon  # type: ignore[arg-type]
                )
                _attach_h3_observable_features(events, inputs)
                job = _write_signed_job(path, signature, baseline, [event.audit_row(partition) for event in events])
            else:
                reused += 1
            all_baseline.extend(job["baseline_rows"])
            all_events.extend(job["audit_rows"])
            complete += 1
            _write_progress(output, "audit", status="running", completed_jobs=complete, total_jobs=total, model=model_name, partition=partition, video_id=video_id)
    all_baseline.sort(key=lambda row: (row["model"], row["video_id"], int(row["frame_id"]), row["observation_id"]))
    all_events.sort(key=lambda row: (row["model"], row["partition"], row["video_id"], int(row["start_frame"]), row["event_id"]))
    write_csv(output / "sltr_baseline_assignments.csv", all_baseline, ASSIGNMENT_FIELDS)
    write_csv(output / "sltr_audit.csv", all_events, AUDIT_FIELDS)
    write_csv(output / "sltr_event_counterfactuals.csv", all_events, AUDIT_FIELDS)
    schemas = {
        tuple(sorted(json.loads(str(row["features_json"])))) for row in all_events
    }
    if len(schemas) > 1:
        raise RuntimeError("SLTR audit produced more than one observable feature schema")
    feature_names = validate_feature_names(next(iter(schemas))) if schemas else ()
    atomic_write_json(output / "sltr_feature_schema.json", {
        "feature_names": list(feature_names),
        "forbidden_tokens": ["gt", "identity", "label", "utility", "correct"],
        "ground_truth_features": False, "final_test_read": False,
    })
    metadata = {
        "status": "completed", "stage": "offline_counterfactual_audit", "models": list(model_names),
        "audit_event_count": len(all_events), "labeled_event_count": sum(row["label_available"] is True for row in all_events),
        "baseline_assignment_count": len(all_baseline), "resumable_job_count": total,
        "reused_job_count": reused, "cache_audit": cache_audit, "input_audit": inputs.audit,
        "offline_label_gt_use": "offline_only", "method_gt_identity_input": False,
        "fit_partition": "project_train", "evaluation_partition": "development_validation",
        "final_test_read": False,
    }
    atomic_write_json(output / "sltr_audit_metadata.json", metadata)
    _write_progress(output, "audit", status="completed", completed_jobs=complete, total_jobs=total, reused_jobs=reused)
    return metadata


def _audit_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError("Run sltr-audit before sltr-fit")
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for row in raw:
        if row.get("label_available") != "True":
            continue
        try:
            utility = float(row["utility_b_minus_a"])
            features = json.loads(row["features_json"])
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid SLTR audit feature/utility row {row.get('event_id')}") from error
        rows.append({
            "model": row["model"], "partition": row["partition"], "video_id": row["video_id"],
            "event_id": row["event_id"], "features": features, "utility": utility,
            "positive": utility > 0,
        })
    return rows


def fit_sltr_selector(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    """Fit only project_train labels and select a threshold solely from group OOF rows."""
    sltr = require_sltr(config)
    inputs = validate_sltr_inputs(config)
    all_audit = _audit_rows(config.paths.output_root / "sltr_audit.csv")
    model_payload: dict[str, Any] = {}
    oof_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        train = [row for row in all_audit if row["model"] == model_name and row["partition"] == "project_train"]
        train_videos = sorted({row["video_id"] for row in train})
        positives = sum(bool(row["positive"]) for row in train)
        support = {
            "train_event_count": len(train), "train_video_count": len(train_videos),
            "positive_event_count": positives, "minimums": {
                "events": sltr.min_train_events, "videos": sltr.min_train_videos,
                "positives": sltr.min_positive_events,
            },
        }
        if len(train) < sltr.min_train_events or len(train_videos) < sltr.min_train_videos or positives < sltr.min_positive_events:
            model_payload[model_name] = {
                "status": "STOP_INSUFFICIENT_PROJECT_TRAIN_EVENTS", "support": support,
                "final_test_read": False,
            }
            continue
        try:
            probabilities, names, folds = group_oof_probabilities(
                train, l2=sltr.selector_l2, iterations=sltr.selector_iterations,
                learning_rate=sltr.selector_learning_rate,
            )
            x, _ = feature_matrix(train, names)
            labels = np.asarray([int(bool(row["positive"])) for row in train], dtype=np.float64)
            model = fit_l2_logistic(
                x, labels, names, l2=sltr.selector_l2, iterations=sltr.selector_iterations,
                learning_rate=sltr.selector_learning_rate,
            )
            threshold = choose_oof_threshold(
                probabilities, [float(row["utility"]) for row in train], [str(row["video_id"]) for row in train],
                min_precision=sltr.min_precision, max_harm=sltr.max_harm,
                min_selected_events=sltr.min_selected_events, min_selected_videos=sltr.min_selected_videos,
                max_intervention_fraction=sltr.max_intervention_fraction,
            )
        except SelectorError as error:
            model_payload[model_name] = {
                "status": "STOP_SELECTOR_OOF_INVALID", "support": support, "error": str(error),
                "final_test_read": False,
            }
            continue
        for row, probability in zip(train, probabilities):
            oof_rows.append({
                "model": model_name, "partition": "project_train", "video_id": row["video_id"],
                "event_id": row["event_id"], "oof_probability": float(probability),
                "utility_b_minus_a": row["utility"], "positive": row["positive"],
                "threshold_source": "group_by_video_oof_only", "final_test_read": False,
            })
        fit_rows.extend({"model": model_name, **fold, "final_test_read": False} for fold in folds)
        model_payload[model_name] = {
            "status": threshold["status"], "support": support, "feature_schema": list(names),
            "model": model.to_dict(), "threshold": threshold, "oof_fold_count": len(folds),
            "fit_partition": "project_train", "development_used_for_fit": False,
            "final_test_read": False,
        }
    status = "completed" if all(model_payload.get(name, {}).get("status") == "GO_SAFE_OOF_THRESHOLD" for name in model_names) else "stopped_fail_closed"
    payload = {
        "status": status, "models": model_payload, "input_audit": inputs.audit,
        "selection_data": "project_train_group_oof_only", "development_used_for_fit": False,
        "final_test_read": False,
    }
    write_csv(config.paths.output_root / "sltr_oof.csv", oof_rows)
    write_csv(config.paths.output_root / "sltr_oof_predictions.csv", oof_rows)
    write_csv(config.paths.output_root / "sltr_fit.csv", fit_rows)
    atomic_write_json(config.paths.output_root / "sltr_model.json", payload)
    atomic_write_json(config.paths.output_root / "sltr_selector_models.json", payload)
    fit_decision = {
        "status": status, "model_statuses": {name: item.get("status") for name, item in model_payload.items()},
        "fit_partition": "project_train", "development_used_for_fit": False, "final_test_read": False,
    }
    atomic_write_json(config.paths.output_root / "sltr_fit.json", fit_decision)
    atomic_write_json(config.paths.output_root / "sltr_fit_decision.json", fit_decision)
    return payload


def _load_model(output: Path, model_name: str) -> tuple[LogisticModel, float, dict[str, Any]]:
    path = output / "sltr_model.json"
    if not path.is_file():
        raise RuntimeError("Run sltr-fit before sltr-track")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload.get("models", {}).get(model_name) if isinstance(payload, dict) else None
    if not isinstance(entry, dict) or entry.get("status") != "GO_SAFE_OOF_THRESHOLD":
        status = entry.get("status") if isinstance(entry, dict) else "missing"
        raise RuntimeError(f"SLTR fail-closed: {model_name} selector is not deployable ({status})")
    threshold = entry.get("threshold", {})
    if not isinstance(threshold, dict) or threshold.get("threshold") is None:
        raise RuntimeError(f"SLTR fail-closed: {model_name} has no safe OOF threshold")
    return LogisticModel.from_dict(entry["model"]), float(threshold["threshold"]), entry


def _gate(
    metrics: Sequence[dict[str, Any]], pooled_metrics: Sequence[dict[str, Any]],
    interventions: Sequence[dict[str, Any]], model_names: Sequence[str], sltr: Any,
) -> dict[str, Any]:
    lookup = {(row["model"], row["variant"], row["video_id"]): row for row in metrics}
    pooled_lookup = {(row["model"], row["variant"]): row for row in pooled_metrics}
    checks: list[dict[str, Any]] = []
    for model in model_names:
        videos = sorted({row["video_id"] for row in metrics if row["model"] == model and row["variant"] == SLTR_PRIMARY_VARIANT})
        baseline = [lookup[(model, "immediate_baseline", video)] for video in videos]
        learned = [lookup[(model, SLTR_PRIMARY_VARIANT, video)] for video in videos]
        random = [lookup[(model, "frequency_matched_random", video)] for video in videos]
        reduction = sum(int(a["IDSW"]) - int(b["IDSW"]) for a, b in zip(baseline, learned))
        random_reduction = sum(int(a["IDSW"]) - int(b["IDSW"]) for a, b in zip(baseline, random))
        pooled_baseline = pooled_lookup[(model, "immediate_baseline")]
        pooled_learned = pooled_lookup[(model, SLTR_PRIMARY_VARIANT)]
        pooled_noninferior = (
            float(pooled_learned["IDF1"]) >= float(pooled_baseline["IDF1"]) - sltr.noninferiority_tolerance
            and float(pooled_learned["HOTA"]) >= float(pooled_baseline["HOTA"]) - sltr.noninferiority_tolerance
        )
        nonharmed = sum(
            float(b["IDF1"]) >= float(a["IDF1"]) - sltr.noninferiority_tolerance
            and float(b["HOTA"]) >= float(a["HOTA"]) - sltr.noninferiority_tolerance
            for a, b in zip(baseline, learned)
        )
        model_events = [
            row for row in interventions
            if row["model"] == model and row["variant"] == SLTR_PRIMARY_VARIANT
        ]
        selected = [row for row in model_events if row["repaired"]]
        harmful = [
            row for row in selected
            if not row["label_available"]
            or row["offline_utility_b_minus_a"] in {"", None}
            or float(row["offline_utility_b_minus_a"]) < 0
        ]
        harm = len(harmful) / len(selected) if selected else 0.0
        intervention_fraction = len(selected) / len(model_events) if model_events else 0.0
        passed = (
            reduction > 0
            and reduction > random_reduction
            and pooled_noninferior
            and nonharmed >= sltr.min_nonharmed_videos
            and harm <= sltr.max_harm
            and intervention_fraction <= sltr.max_intervention_fraction + 1e-12
        )
        checks.append({
            "model": model, "idsw_reduction": reduction, "nonharmed_video_count": nonharmed,
            "frequency_matched_random_idsw_reduction": random_reduction,
            "idsw_reduction_better_than_frequency_matched_random": reduction > random_reduction,
            "pooled_idf1_hota_noninferior": pooled_noninferior,
            "harm": harm, "selected_events": len(selected),
            "unlabelled_selected_events": sum(not row["label_available"] for row in selected),
            "intervention_fraction": intervention_fraction, "pass": passed,
        })
    go = all(row["pass"] for row in checks) and len(checks) == len(model_names)
    return {
        "status": "GO_SLTR_DEVELOPMENT" if go else "STOP_SLTR_DEVELOPMENT_GATE",
        "gate_passed": go, "model_checks": checks,
        "criteria": {
            "positive_idsw_reduction_each_backbone": True,
            "better_idsw_reduction_than_frequency_matched_random": True,
            "pooled_idf1_hota_noninferiority": sltr.noninferiority_tolerance,
            "idf1_hota_noninferiority": sltr.noninferiority_tolerance,
            "max_harm": sltr.max_harm,
            "max_intervention_fraction": sltr.max_intervention_fraction,
            "unlabelled_selected_events_count_as_harm": True,
            "min_nonharmed_videos": sltr.min_nonharmed_videos,
        },
        "oracle_selector_diagnostic_only": True, "final_test_read": False,
    }


def _intervention_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["variant"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, variant), events in sorted(groups.items()):
        selected = [row for row in events if bool(row["repaired"])]
        labelled = [
            row for row in selected
            if bool(row["label_available"])
            and row["offline_utility_b_minus_a"] not in {"", None}
        ]
        beneficial = [row for row in labelled if float(row["offline_utility_b_minus_a"]) > 0]
        harmful = [row for row in labelled if float(row["offline_utility_b_minus_a"]) < 0]
        unlabelled = len(selected) - len(labelled)
        output.append({
            "model": model, "variant": variant, "event_count": len(events),
            "selected_event_count": len(selected),
            "intervention_fraction": len(selected) / len(events) if events else 0.0,
            "beneficial_selected_count": len(beneficial),
            "harmful_selected_count": len(harmful),
            "unlabelled_selected_count": unlabelled,
            "intervention_precision": len(beneficial) / len(selected) if selected else 0.0,
            "conservative_harm_rate": (len(harmful) + unlabelled) / len(selected) if selected else 0.0,
            "selected_utility_sum": sum(float(row["offline_utility_b_minus_a"]) for row in labelled),
            "final_test_read": False,
        })
    return output


def run_sltr_tracking(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    """Evaluate frozen development variants; learned decisions use no labels/GT fields."""
    sltr = require_sltr(config)
    inputs = validate_sltr_inputs(config)
    if set(model_names) != set(SLTR_MODELS):
        raise ValueError("SLTR READY/development tracking requires both resnet50 and dinov3")
    groups = _by_partition_video(inputs)
    development = {video: indices for (partition, video), indices in groups.items() if partition == "development_validation"}
    output = config.paths.output_root
    assignments: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    cache_audit: dict[str, Any] = {}
    _write_progress(output, "track", status="running", completed_jobs=0, total_jobs=len(model_names) * len(development) * len(SLTR_VARIANTS))
    complete = reused = 0
    for model_name in model_names:
        selector, threshold, fit_entry = _load_model(output, model_name)
        embeddings, cache = load_sltr_embeddings(config, model_name, inputs)
        cache_audit[model_name] = cache
        for video_id, indices in sorted(development.items()):
            observations = [inputs.observations[index] for index in indices]
            values = np.asarray(embeddings[indices], dtype=np.float32)
            _, candidate_events = build_counterfactual_events(model_name, observations, values, config.h4, horizon=sltr.horizon)  # type: ignore[arg-type]
            _attach_h3_observable_features(candidate_events, inputs)
            score_by_event = {
                event.event_id: float(selector.probability(np.asarray([[event.features[name] for name in selector.feature_names]], dtype=np.float64))[0])
                for event in candidate_events
            }
            above_threshold = sorted(
                (event_id for event_id, score in score_by_event.items() if score >= threshold),
                key=lambda event_id: (-score_by_event[event_id], event_id),
            )
            intervention_budget = int(np.floor(sltr.max_intervention_fraction * len(candidate_events)))
            learned_ids = set(above_threshold[:intervention_budget])
            random_ids = deterministic_frequency_choice(list(score_by_event), len(learned_ids), sltr.random_seed)
            for variant in SLTR_VARIANTS:
                selected_count = 0

                def decide(event: CounterfactualEvent) -> tuple[bool, str]:
                    nonlocal selected_count
                    if variant == "immediate_baseline":
                        return False, "immediate_baseline"
                    if variant == "always_repair":
                        return True, "always_repair"
                    if variant == "frequency_matched_random":
                        repair = event.event_id in random_ids
                        return repair, "frequency_matched_random" if repair else "frequency_matched_random_fallback"
                    if variant == "oracle_selector":
                        repair = bool(event.label_available and event.utility is not None and event.utility > 0)
                        return repair, "oracle_diagnostic_gt_label" if event.label_available else "oracle_label_unavailable_exact_a_fallback"
                    score = float(selector.probability(np.asarray(
                        [[event.features[name] for name in selector.feature_names]], dtype=np.float64
                    ))[0])
                    if score >= threshold and selected_count < intervention_budget:
                        selected_count += 1
                        return True, "learned_above_oof_threshold_within_budget"
                    reason = "intervention_budget_exhausted_exact_a_fallback" if score >= threshold else "below_threshold_exact_a_fallback"
                    return False, reason

                signature = {
                    "implementation": "beeid.sltr.tracking:v1", "model": model_name,
                    "variant": variant, "video_id": video_id,
                    "observation_ids": [item.observation_id for item in observations],
                    "cache_fingerprint": cache["fingerprint"], "protocol_sha256": inputs.audit["protocol_sha256"],
                    "selector_model_sha256": sha256_file(output / "sltr_model.json"),
                    "horizon": sltr.horizon, "final_test_read": False,
                }
                job_path = output / "work" / "tracking" / model_name / variant / f"{video_id}.json"
                job = _load_signed_tracking_job(job_path, signature)
                if job is None:
                    rows, actual_events, selected = track_selective_sequence(
                        model_name, variant, observations, values, config.h4, horizon=sltr.horizon,
                        decide=decide, prepare_event=lambda event: _attach_h3_observable_features([event], inputs),  # type: ignore[arg-type]
                    )
                    by_id = {event.event_id: event for event in actual_events}
                    for record in selected:
                        event = by_id[record["event_id"]]
                        score = float(selector.probability(np.asarray([[event.features[name] for name in selector.feature_names]], dtype=np.float64))[0])
                        record["score"] = score
                        record["threshold"] = threshold
                        if variant == "oracle_selector":
                            record["ground_truth_decision_input"] = True
                    for row in rows:
                        if variant == "oracle_selector":
                            row["sltr_ground_truth_decision_input"] = True
                    job = _write_signed_tracking_job(job_path, signature, rows, selected)
                else:
                    reused += 1
                assignments.extend(job["rows"])
                interventions.extend(job["interventions"])
                complete += 1
                _write_progress(output, "track", status="running", completed_jobs=complete, total_jobs=len(model_names) * len(development) * len(SLTR_VARIANTS), model=model_name, video_id=video_id, variant=variant)
    assignments.sort(key=lambda row: (row["model"], row["variant"], row["video_id"], int(row["frame_id"]), row["observation_id"]))
    interventions.sort(key=lambda row: (row["model"], row["variant"], row["video_id"], int(row["start_frame"]), row["event_id"]))
    per_video, summary = summarize_gt_assignments(assignments)
    paired: list[dict[str, Any]] = []
    baseline = {(row["model"], row["video_id"]): row for row in per_video if row["variant"] == "immediate_baseline"}
    for row in per_video:
        if row["variant"] == "immediate_baseline":
            continue
        base = baseline[(row["model"], row["video_id"])]
        paired.append({
            "model": row["model"], "variant": row["variant"], "video_id": row["video_id"],
            "IDSW_reduction_vs_immediate": int(base["IDSW"]) - int(row["IDSW"]),
            "IDF1_gain_vs_immediate": float(row["IDF1"]) - float(base["IDF1"]),
            "HOTA_gain_vs_immediate": float(row["HOTA"]) - float(base["HOTA"]),
            "final_test_read": False,
        })
    write_csv(output / "sltr_assignments.csv", assignments, ASSIGNMENT_FIELDS)
    write_csv(output / "sltr_interventions.csv", interventions)
    write_csv(output / "sltr_per_video_metrics.csv", per_video)
    write_csv(output / "sltr_metrics.csv", summary)
    write_csv(output / "sltr_summary.csv", summary)
    write_csv(output / "sltr_paired_video_metrics.csv", paired)
    write_csv(output / "sltr_intervention_metrics.csv", _intervention_summary(interventions))
    decision = _gate(per_video, summary, interventions, model_names, sltr)
    atomic_write_json(output / "sltr_decision.json", decision)
    atomic_write_json(output / "sltr_method_decision.json", decision)
    metadata = {
        "status": "completed", "stage": "development_causal_tracking", "models": list(model_names),
        "variants": list(SLTR_VARIANTS), "primary_variant": SLTR_PRIMARY_VARIANT,
        "assignment_row_count": len(assignments), "intervention_count": len(interventions),
        "development_videos": sorted(development), "cache_audit": cache_audit,
        "resumable_job_count": len(model_names) * len(development) * len(SLTR_VARIANTS),
        "reused_job_count": reused,
        "selector_source": "project_train_group_oof_threshold", "development_used_for_fit": False,
        "oracle_selector": "diagnostic_only_not_deployable", "input_audit": inputs.audit,
        "decision": decision, "final_test_read": False,
    }
    atomic_write_json(output / "sltr_tracking_metadata.json", metadata)
    _write_progress(output, "track", status="completed", completed_jobs=complete, total_jobs=len(model_names) * len(development) * len(SLTR_VARIANTS), reused_jobs=reused)
    return metadata


def run_sltr_all(config: ExperimentConfig, model_names: Sequence[str], confirm_full: bool) -> dict[str, Any]:
    sltr = require_sltr(config)
    if not sltr.allow_subset and not confirm_full:
        raise RuntimeError("Full SLTR is gated; pass --confirm-full only after synthetic and real smoke")
    audit = run_sltr_audit(config, model_names)
    fit = fit_sltr_selector(config, model_names)
    if fit["status"] != "completed":
        atomic_write_json(config.paths.output_root / "sltr_decision.json", {
            "status": "STOP_SLTR_SELECTOR_FIT", "gate_passed": False,
            "reason": "project_train_selector_fail_closed", "final_test_read": False,
        })
        atomic_write_json(config.paths.output_root / "sltr_method_decision.json", {
            "status": "STOP_SLTR_SELECTOR_FIT", "gate_passed": False,
            "reason": "project_train_selector_fail_closed", "final_test_read": False,
        })
        from .report import generate_sltr_report
        report = generate_sltr_report(config, model_names)
        return {"audit": audit, "fit": fit, "tracking": None, "report": report, "status": "stopped_fail_closed"}
    tracking = run_sltr_tracking(config, model_names)
    from .report import generate_sltr_report
    report = generate_sltr_report(config, model_names)
    return {"audit": audit, "fit": fit, "tracking": tracking, "report": report, "status": tracking["decision"]["status"]}
