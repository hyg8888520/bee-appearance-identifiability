"""H4 recoverability gate and fixed-lag hypothesis-tracking experiments."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..h3.metrics import summarize_gt_assignments
from ..utils import (
    atomic_write_json, atomic_write_text, canonical_json, sha256_file, sha256_text,
)
from .core import load_h4_source_embeddings, require_h4, validate_h4_inputs
from .io import write_csv
from .recoverability import (
    RECOVERABILITY_FIELDS,
    audit_model_recoverability,
    recoverability_decision,
    summarize_recoverability,
)
from .tracker import H4_PRIMARY_VARIANT, H4_VARIANTS, track_h4_sequence


H4_ASSIGNMENT_FIELDS = (
    "model", "variant", "stage", "video_id", "frame_id", "observation_id",
    "gt_identity", "gt_track_id", "predicted_track_id", "matched_existing_track",
    "association_score", "appearance_similarity", "motion_score",
    "normalized_motion_distance", "eligible_memory_update", "memory_update_accepted",
    "memory_update_committed", "memory_update_used_for_branch_scoring",
    "effective_memory_alpha", "reliability", "event_id", "ambiguity_detected",
    "deferred_commit", "event_start_frame", "decision_frame",
    "decision_latency_frames", "branch_memory_isolated", "branch_memory_frozen",
    "hypothesis_count_peak", "winning_hypothesis_rank", "decision_score_margin",
)


def _by_video_indices(observations: Sequence[Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        result[item.video_id].append(index)
    for indices in result.values():
        indices.sort(
            key=lambda index: (
                observations[index].frame, observations[index].center_x,
                observations[index].center_y, observations[index].observation_id,
            )
        )
    return result


def _job_fingerprint(
    stage: str,
    model_name: str,
    video_id: str,
    variant: str,
    observation_ids: Sequence[str],
    cache_fingerprint: Any,
    protocol_sha256: str,
    decision_sha256: str | None = None,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "implementation": "beeid.h4.experiment:v1",
                "stage": stage,
                "model": model_name,
                "video_id": video_id,
                "variant": variant,
                "observation_ids": list(observation_ids),
                "cache_fingerprint": cache_fingerprint,
                "protocol_sha256": protocol_sha256,
                "decision_sha256": decision_sha256,
                "final_test_read": False,
            }
        )
    )


def _load_job(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("status") != "completed"
        or value.get("fingerprint") != fingerprint
        or value.get("final_test_read") is not False
        or not isinstance(value.get("rows"), list)
        or not isinstance(value.get("events"), list)
    ):
        return None
    return value


def _write_mot_results(
    root: Path,
    rows: Sequence[dict[str, Any]],
    observation_lookup: dict[str, Any],
) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["variant"]), str(row["video_id"]))].append(row)
    for (model, variant, video_id), selected in groups.items():
        lines: list[str] = []
        for row in sorted(selected, key=lambda item: (int(item["frame_id"]), int(item["predicted_track_id"]))):
            observation = observation_lookup[str(row["observation_id"])]
            lines.append(
                ",".join(
                    [
                        str(observation.frame), str(row["predicted_track_id"]),
                        f"{observation.raw_x:.6f}", f"{observation.raw_y:.6f}",
                        f"{observation.raw_w:.6f}", f"{observation.raw_h:.6f}",
                        "1", "-1", "-1", "-1",
                    ]
                )
            )
        atomic_write_text(root / model / variant / f"{video_id}.txt", "\n".join(lines) + "\n")


def _paired_metrics(per_video: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["model"]), str(row["variant"]), str(row["video_id"])): row
        for row in per_video
    }
    output: list[dict[str, Any]] = []
    for row in per_video:
        if row["variant"] == "immediate_commit":
            continue
        baseline = lookup.get((str(row["model"]), "immediate_commit", str(row["video_id"])))
        if baseline is None:
            raise RuntimeError("H4 paired metrics are missing immediate_commit")
        output.append(
            {
                "model": row["model"],
                "variant": row["variant"],
                "video_id": row["video_id"],
                "AssA_gain_vs_immediate": float(row["AssA"]) - float(baseline["AssA"]),
                "IDF1_gain_vs_immediate": float(row["IDF1"]) - float(baseline["IDF1"]),
                "HOTA_gain_vs_immediate": float(row["HOTA"]) - float(baseline["HOTA"]),
                "IDSW_reduction_vs_immediate": int(baseline["IDSW"]) - int(row["IDSW"]),
                "nonharmed_IDF1": float(row["IDF1"]) >= float(baseline["IDF1"]),
                "nonharmed_HOTA": float(row["HOTA"]) >= float(baseline["HOTA"]),
                "final_test_read": False,
            }
        )
    return output


def _augment_metrics(
    per_video: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    assignment_rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
) -> None:
    assignment_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        assignment_groups[(str(row["model"]), str(row["variant"]), str(row["video_id"]))].append(row)
    event_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        event_groups[(str(row["model"]), str(row["variant"]), str(row["video_id"]))].append(row)
    for row in per_video:
        key = (str(row["model"]), str(row["variant"]), str(row["video_id"]))
        assignments = assignment_groups[key]
        events = event_groups.get(key, [])
        deferred = [item for item in assignments if bool(item["deferred_commit"])]
        row.update(
            {
                "event_count": len(events),
                "deferred_event_count": sum(bool(item["deferred_commit"]) for item in events),
                "oversized_conflict_count": sum(item["reason"] == "oversized_conflict" for item in events),
                "hypotheses_expanded": sum(int(item["hypotheses_expanded"]) for item in events),
                "deferred_observation_fraction": len(deferred) / len(assignments) if assignments else 0.0,
                "mean_decision_latency_frames": float(np.mean([int(item["decision_latency_frames"]) for item in deferred])) if deferred else 0.0,
            }
        )
    summary_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_video:
        summary_groups[(str(row["model"]), str(row["variant"]))].append(row)
    for row in summary:
        selected = summary_groups[(str(row["model"]), str(row["variant"]))]
        row.update(
            {
                "event_count": sum(int(item["event_count"]) for item in selected),
                "deferred_event_count": sum(int(item["deferred_event_count"]) for item in selected),
                "oversized_conflict_count": sum(int(item["oversized_conflict_count"]) for item in selected),
                "hypotheses_expanded": sum(int(item["hypotheses_expanded"]) for item in selected),
                "deferred_observation_fraction": float(np.mean([float(item["deferred_observation_fraction"]) for item in selected])) if selected else 0.0,
                "mean_decision_latency_frames": float(np.mean([float(item["mean_decision_latency_frames"]) for item in selected])) if selected else 0.0,
            }
        )


def run_h4_recoverability_audit(
    config: ExperimentConfig, model_names: Sequence[str]
) -> dict[str, Any]:
    h4 = require_h4(config)
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("H4 audit requires unique model names")
    inputs = validate_h4_inputs(config)
    by_video = _by_video_indices(inputs.observations)
    all_events: list[dict[str, Any]] = []
    all_baseline: list[dict[str, Any]] = []
    cache_audit: dict[str, Any] = {}
    progress = config.paths.output_root / "h4_logs" / "audit_progress.json"
    total = len(model_names) * len(by_video)
    completed = 0
    reused_jobs = 0
    atomic_write_json(progress, {"status": "running", "completed_model_videos": 0, "total_model_videos": total, "final_test_read": False})
    for model_name in model_names:
        embeddings, cache = load_h4_source_embeddings(config, model_name, inputs)
        cache_audit[model_name] = cache
        for video_id in sorted(by_video):
            indices = by_video[video_id]
            observations = [inputs.observations[index] for index in indices]
            values = np.asarray(embeddings[indices], dtype=np.float32)
            fingerprint = _job_fingerprint(
                "recoverability_audit", model_name, video_id, "immediate_commit",
                [item.observation_id for item in observations], cache.get("fingerprint"),
                str(inputs.audit["protocol_sha256"]),
            )
            job_path = (
                config.paths.output_root / "h4_work" / "audit" / model_name / f"{video_id}.json"
            )
            job = _load_job(job_path, fingerprint)
            if job is None:
                baseline, _ = track_h4_sequence(
                    model_name, "immediate_commit", observations, values, h4
                )
                event_rows = audit_model_recoverability(
                    model_name, observations, values, baseline, h4
                )
                job = {
                    "status": "completed",
                    "fingerprint": fingerprint,
                    "rows": baseline,
                    "events": event_rows,
                    "final_test_read": False,
                }
                atomic_write_json(job_path, job)
            else:
                reused_jobs += 1
            all_baseline.extend(job["rows"])
            all_events.extend(job["events"])
            completed += 1
            atomic_write_json(progress, {"status": "running", "completed_model_videos": completed, "total_model_videos": total, "model": model_name, "video_id": video_id, "final_test_read": False})
    all_baseline.sort(key=lambda row: (row["model"], row["video_id"], row["frame_id"], row["observation_id"]))
    all_events.sort(key=lambda row: (row["model"], row["video_id"], row["event_frame"], row["event_id"], row["horizon"]))
    summary = summarize_recoverability(all_events, model_names, h4.horizons)
    decision = recoverability_decision(summary, model_names, h4)
    output = config.paths.output_root
    write_csv(output / "h4_recoverability_baseline_assignments.csv", all_baseline, H4_ASSIGNMENT_FIELDS)
    write_csv(output / "h4_recoverability_events.csv", all_events, RECOVERABILITY_FIELDS)
    write_csv(output / "h4_horizon_summary.csv", summary)
    atomic_write_json(output / "h4_protocol_decision.json", decision)
    metadata = {
        "status": "completed",
        "stage": "recoverability_audit",
        "models": list(model_names),
        "event_horizon_row_count": len(all_events),
        "switch_event_count": len({(row["model"], row["event_id"]) for row in all_events}),
        "baseline_assignment_count": len(all_baseline),
        "cache_audit": cache_audit,
        "resumable_job_count": total,
        "reused_job_count": reused_jobs,
        "input_audit": inputs.audit,
        "decision": decision,
        "oracle_gt_use": "OFFLINE_DIAGNOSTIC_ONLY",
        "method_gt_identity_input": False,
        "final_test_read": False,
    }
    atomic_write_json(output / "h4_recoverability_metadata.json", metadata)
    atomic_write_json(progress, {"status": "completed", "completed_model_videos": completed, "total_model_videos": total, "gate_status": decision["status"], "final_test_read": False})
    return metadata


def _load_gate(config: ExperimentConfig) -> dict[str, Any]:
    path = config.paths.output_root / "h4_protocol_decision.json"
    if not path.is_file():
        raise RuntimeError("Run h4-audit before h4-track")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("final_test_read") is not False:
        raise RuntimeError("Invalid H4 recoverability decision")
    return value


def run_h4_tracking(
    config: ExperimentConfig,
    model_names: Sequence[str],
    *,
    override_audit_stop: bool = False,
) -> dict[str, Any]:
    h4 = require_h4(config)
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("H4 tracking requires unique model names")
    inputs = validate_h4_inputs(config)
    decision = _load_gate(config)
    gate_passed = decision.get("gate_passed") is True
    if not gate_passed:
        if not override_audit_stop:
            raise RuntimeError(
                "H4 recoverability gate failed; protocol requires STOP_BEFORE_METHOD_EVALUATION"
            )
        if not h4.allow_subset:
            raise RuntimeError("Audit-stop override is restricted to subset smoke configurations")
    by_video = _by_video_indices(inputs.observations)
    all_rows: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    cache_audit: dict[str, Any] = {}
    total = len(model_names) * len(by_video) * len(H4_VARIANTS)
    completed = 0
    reused_jobs = 0
    progress = config.paths.output_root / "h4_logs" / "tracking_progress.json"
    decision_sha256 = sha256_file(config.paths.output_root / "h4_protocol_decision.json")
    atomic_write_json(progress, {"status": "running", "completed_jobs": 0, "total_jobs": total, "final_test_read": False})
    for model_name in model_names:
        embeddings, cache = load_h4_source_embeddings(config, model_name, inputs)
        cache_audit[model_name] = cache
        for video_id in sorted(by_video):
            indices = by_video[video_id]
            observations = [inputs.observations[index] for index in indices]
            values = np.asarray(embeddings[indices], dtype=np.float32)
            for variant in H4_VARIANTS:
                fingerprint = _job_fingerprint(
                    "hypothesis_tracking", model_name, video_id, variant,
                    [item.observation_id for item in observations], cache.get("fingerprint"),
                    str(inputs.audit["protocol_sha256"]), decision_sha256,
                )
                job_path = (
                    config.paths.output_root / "h4_work" / "tracking"
                    / model_name / variant / f"{video_id}.json"
                )
                job = _load_job(job_path, fingerprint)
                was_reused = job is not None
                if job is None:
                    started = time.perf_counter()
                    rows, events = track_h4_sequence(
                        model_name, variant, observations, values, h4
                    )
                    elapsed = time.perf_counter() - started
                    runtime = {
                        "model": model_name,
                        "variant": variant,
                        "video_id": video_id,
                        "observation_count": len(observations),
                        "event_count": len(events),
                        "elapsed_seconds": elapsed,
                        "observations_per_second": len(observations) / elapsed if elapsed > 0 else "",
                        "device_role": "association_cpu_numpy",
                        "feature_extraction": "reused_h3_cache",
                        "final_test_read": False,
                    }
                    job = {
                        "status": "completed",
                        "fingerprint": fingerprint,
                        "rows": rows,
                        "events": events,
                        "runtime": runtime,
                        "final_test_read": False,
                    }
                    atomic_write_json(job_path, job)
                else:
                    reused_jobs += 1
                all_rows.extend(job["rows"])
                all_events.extend(job["events"])
                runtime = dict(job.get("runtime") or {})
                runtime["resumed_from_atomic_job"] = was_reused
                runtime_rows.append(runtime)
                completed += 1
                atomic_write_json(progress, {"status": "running", "completed_jobs": completed, "total_jobs": total, "model": model_name, "video_id": video_id, "variant": variant, "final_test_read": False})
    all_rows.sort(key=lambda row: (row["model"], row["variant"], row["video_id"], row["frame_id"], row["observation_id"]))
    all_events.sort(key=lambda row: (row["model"], row["variant"], row["video_id"], row["start_frame"], row["event_id"]))
    per_video, summary = summarize_gt_assignments(all_rows)
    _augment_metrics(per_video, summary, all_rows, all_events)
    paired = _paired_metrics(per_video)
    output = config.paths.output_root
    write_csv(output / "h4_assignments.csv", all_rows, H4_ASSIGNMENT_FIELDS)
    write_csv(output / "h4_conflict_events.csv", all_events)
    write_csv(output / "h4_per_video_metrics.csv", per_video)
    write_csv(output / "h4_summary.csv", summary)
    write_csv(output / "h4_paired_video_metrics.csv", paired)
    write_csv(output / "h4_runtime_profile.csv", runtime_rows)
    _write_mot_results(
        output / "h4_mot_results", all_rows,
        {item.observation_id: item for item in inputs.observations},
    )
    state = {
        "status": "completed",
        "models": list(model_names),
        "variants": list(H4_VARIANTS),
        "primary_variant": H4_PRIMARY_VARIANT,
        "assignment_row_count": len(all_rows),
        "conflict_event_count": len(all_events),
        "summary_row_count": len(summary),
        "development_videos": sorted(by_video),
        "cache_audit": cache_audit,
        "resumable_job_count": total,
        "reused_job_count": reused_jobs,
        "recoverability_gate_passed": gate_passed,
        "protocol_decision_sha256": decision_sha256,
        "audit_stop_override": override_audit_stop,
        "result_status": (
            "PROTOCOL_GATE_OVERRIDDEN_FOR_SUBSET_SMOKE_NOT_EXPERIMENT"
            if override_audit_stop and not gate_passed else "PRIMARY_DEVELOPMENT_RESULT"
        ),
        "input_audit": inputs.audit,
        "gt_identity_used_for_decisions": False,
        "final_test_read": False,
    }
    atomic_write_json(output / "h4_tracking_metadata.json", state)
    atomic_write_json(progress, {"status": "completed", "completed_jobs": completed, "total_jobs": total, "final_test_read": False})
    return state
