"""Resumable H4.1 continuous audit and adaptive-lag tracking experiments."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..h3.metrics import summarize_gt_assignments
from ..h4.experiment import (
    H4_ASSIGNMENT_FIELDS,
    _augment_metrics,
    _by_video_indices,
    _load_job,
    _paired_metrics,
    _write_mot_results,
)
from ..h4.io import write_csv
from ..h4.recoverability import (
    RECOVERABILITY_FIELDS,
    audit_model_recoverability,
    summarize_recoverability,
)
from ..h4.tracker import track_h4_sequence
from ..utils import (
    atomic_write_json,
    canonical_json,
    sha256_file,
    sha256_text,
)
from .core import load_h41_source_embeddings, require_h41, validate_h41_inputs
from .protocol import H41_PRIMARY_VARIANT, H41_VARIANTS
from .recoverability import (
    CUMULATIVE_EVENT_FIELDS,
    crosscheck_h4_v1_exact_rows,
    cumulative_event_timelines,
    cumulative_recoverability_decision,
    summarize_cumulative_recoverability,
)
from .tracker import H41_ASSIGNMENT_EXTRA_FIELDS, track_h41_sequence


H41_ASSIGNMENT_FIELDS = H4_ASSIGNMENT_FIELDS + H41_ASSIGNMENT_EXTRA_FIELDS


def _fingerprint(
    stage: str,
    model_name: str,
    video_id: str,
    variant: str,
    observation_ids: Sequence[str],
    cache_fingerprint: Any,
    protocol_sha256: str,
    source_h4_events_sha256: str,
    source_h4_baseline_sha256: str,
    decision_sha256: str | None = None,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "implementation": "beeid.h41.experiment:v1",
                "stage": stage,
                "model": model_name,
                "video_id": video_id,
                "variant": variant,
                "observation_ids": list(observation_ids),
                "cache_fingerprint": cache_fingerprint,
                "protocol_sha256": protocol_sha256,
                "source_h4_events_sha256": source_h4_events_sha256,
                "source_h4_baseline_sha256": source_h4_baseline_sha256,
                "decision_sha256": decision_sha256,
                "final_test_read": False,
            }
        )
    )


def _crosscheck_baseline(
    current_rows: Sequence[dict[str, Any]],
    source_rows: Sequence[dict[str, str]],
    model_names: Sequence[str],
) -> dict[str, Any]:
    wanted = set(model_names)

    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        return str(row["model"]), str(row["video_id"]), str(row["observation_id"])

    current = {key(row): row for row in current_rows if row["model"] in wanted}
    source = {
        identifier: row
        for row in source_rows
        if row["model"] in wanted
        and (identifier := key(row)) in current
    }
    if set(current) != set(source):
        raise RuntimeError("H4.1 recomputed baseline observation keys differ from H4-v1")
    for identifier, source_row in source.items():
        row = current[identifier]
        for field in ("predicted_track_id", "matched_existing_track"):
            if str(row[field]) != str(source_row[field]):
                raise RuntimeError(
                    f"H4.1 recomputed baseline changed H4-v1 {field} for {identifier}"
                )
    return {
        "status": "passed",
        "compared_assignment_count": len(source),
        "source_h4_v1_conclusion_preserved": True,
        "final_test_read": False,
    }


def run_h41_window_audit(
    config: ExperimentConfig, model_names: Sequence[str]
) -> dict[str, Any]:
    h41 = require_h41(config)
    h4 = config.h4
    assert h4 is not None
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("H4.1 audit requires unique model names")
    inputs = validate_h41_inputs(config)
    by_video = _by_video_indices(inputs.observations)
    all_exact: list[dict[str, Any]] = []
    all_baseline: list[dict[str, Any]] = []
    cache_audit: dict[str, Any] = {}
    progress = config.paths.output_root / "h41_logs" / "audit_progress.json"
    total = len(model_names) * len(by_video)
    completed = 0
    reused_jobs = 0
    atomic_write_json(
        progress,
        {
            "status": "running",
            "completed_model_videos": 0,
            "total_model_videos": total,
            "final_test_read": False,
        },
    )
    oracle_config = replace(h4, horizons=h41.audit_horizons)
    for model_name in model_names:
        embeddings, cache = load_h41_source_embeddings(config, model_name, inputs)
        cache_audit[model_name] = cache
        for video_id in sorted(by_video):
            indices = by_video[video_id]
            observations = [inputs.observations[index] for index in indices]
            values = np.asarray(embeddings[indices], dtype=np.float32)
            fingerprint = _fingerprint(
                "continuous_window_recoverability_audit",
                model_name,
                video_id,
                "immediate_commit",
                [item.observation_id for item in observations],
                cache.get("fingerprint"),
                str(inputs.audit["protocol_sha256"]),
                str(inputs.audit["source_h4_events_sha256"]),
                str(inputs.audit["source_h4_baseline_sha256"]),
            )
            job_path = (
                config.paths.output_root
                / "h41_work"
                / "audit"
                / model_name
                / f"{video_id}.json"
            )
            job = _load_job(job_path, fingerprint)
            if job is None:
                baseline, _ = track_h4_sequence(
                    model_name, "immediate_commit", observations, values, h4
                )
                exact = audit_model_recoverability(
                    model_name, observations, values, baseline, oracle_config
                )
                job = {
                    "status": "completed",
                    "fingerprint": fingerprint,
                    "rows": baseline,
                    "events": exact,
                    "final_test_read": False,
                }
                atomic_write_json(job_path, job)
            else:
                reused_jobs += 1
            all_baseline.extend(job["rows"])
            all_exact.extend(job["events"])
            completed += 1
            atomic_write_json(
                progress,
                {
                    "status": "running",
                    "completed_model_videos": completed,
                    "total_model_videos": total,
                    "model": model_name,
                    "video_id": video_id,
                    "final_test_read": False,
                },
            )
    all_baseline.sort(
        key=lambda row: (
            row["model"], row["video_id"], row["frame_id"], row["observation_id"]
        )
    )
    all_exact.sort(
        key=lambda row: (
            row["model"], row["video_id"], row["event_frame"],
            row["event_id"], row["horizon"],
        )
    )
    baseline_check = _crosscheck_baseline(
        all_baseline, inputs.source_baseline_rows, model_names
    )
    exact_check = crosscheck_h4_v1_exact_rows(
        all_exact,
        inputs.source_event_rows,
        model_names,
        comparison_frame_limit=config.dataset.max_frames_per_video,
    )
    timelines = cumulative_event_timelines(all_exact, h41.cumulative_deadlines)
    timelines.sort(
        key=lambda row: (
            row["model"], row["video_id"], row["event_frame"],
            row["event_id"], row["deadline"],
        )
    )
    exact_summary = summarize_recoverability(
        all_exact, model_names, h41.audit_horizons
    )
    cumulative_summary = summarize_cumulative_recoverability(
        timelines, model_names, h41.cumulative_deadlines
    )
    decision = cumulative_recoverability_decision(
        cumulative_summary,
        model_names,
        gate_deadline=h41.gate_deadline,
        min_fraction=h41.min_cumulative_recoverable_fraction,
        min_events=h41.min_events_per_model,
        min_videos=h41.min_videos_with_events,
    )
    output = config.paths.output_root
    write_csv(
        output / "h41_recomputed_baseline_assignments.csv",
        all_baseline,
        H4_ASSIGNMENT_FIELDS,
    )
    write_csv(
        output / "h41_exact_recoverability_events.csv",
        all_exact,
        RECOVERABILITY_FIELDS,
    )
    write_csv(
        output / "h41_event_timelines.csv",
        timelines,
        CUMULATIVE_EVENT_FIELDS,
    )
    write_csv(output / "h41_exact_horizon_summary.csv", exact_summary)
    write_csv(output / "h41_cumulative_summary.csv", cumulative_summary)
    crosscheck = {
        "status": "passed",
        "baseline": baseline_check,
        "exact_horizons": exact_check,
        "source_h4_v1_gate_status": "STOP_NO_RECOVERABILITY_SIGNAL",
        "source_h4_v1_conclusion_preserved": True,
        "final_test_read": False,
    }
    atomic_write_json(output / "h41_source_h4_crosscheck.json", crosscheck)
    atomic_write_json(output / "h41_protocol_decision.json", decision)
    metadata = {
        "status": "completed",
        "stage": "continuous_window_recoverability_audit",
        "models": list(model_names),
        "exact_event_horizon_row_count": len(all_exact),
        "cumulative_event_deadline_row_count": len(timelines),
        "switch_event_count": len(
            {(row["model"], row["event_id"]) for row in all_exact}
        ),
        "baseline_assignment_count": len(all_baseline),
        "cache_audit": cache_audit,
        "resumable_job_count": total,
        "reused_job_count": reused_jobs,
        "input_audit": inputs.audit,
        "source_h4_crosscheck": crosscheck,
        "decision": decision,
        "result_scope": (
            "SUBSET_SMOKE_NOT_EXPERIMENT"
            if h41.allow_subset else "EXPLORATORY_DEVELOPMENT_AUDIT"
        ),
        "design_timing": "DEFINED_AFTER_OBSERVING_H4_V1_DEVELOPMENT_AUDIT",
        "oracle_gt_use": "OFFLINE_DIAGNOSTIC_ONLY",
        "method_gt_identity_input": False,
        "final_test_read": False,
    }
    atomic_write_json(output / "h41_recoverability_metadata.json", metadata)
    atomic_write_json(
        progress,
        {
            "status": "completed",
            "completed_model_videos": completed,
            "total_model_videos": total,
            "gate_status": decision["status"],
            "final_test_read": False,
        },
    )
    return metadata


def _load_gate(config: ExperimentConfig) -> dict[str, Any]:
    path = config.paths.output_root / "h41_protocol_decision.json"
    if not path.is_file():
        raise RuntimeError("Run h41-audit before h41-track")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("final_test_read") is not False:
        raise RuntimeError("Invalid H4.1 cumulative recoverability decision")
    return value


def _augment_h41_metrics(
    per_video: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    events: Sequence[dict[str, Any]],
) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        groups[(str(row["model"]), str(row["variant"]), str(row["video_id"]))].append(row)
    for row in per_video:
        selected = groups.get(
            (str(row["model"]), str(row["variant"]), str(row["video_id"])), []
        )
        deferred = [item for item in selected if bool(item["deferred_commit"])]
        adaptive = [item for item in deferred if item.get("decision_policy") == "adaptive_lag"]
        row.update(
            {
                "early_commit_count": sum(bool(item.get("early_commit")) for item in adaptive),
                "forced_deadline_count": sum(bool(item.get("forced_at_deadline")) for item in adaptive),
                "forced_sequence_end_count": sum(
                    bool(item.get("forced_at_sequence_end")) for item in adaptive
                ),
                "early_commit_fraction": (
                    sum(bool(item.get("early_commit")) for item in adaptive) / len(adaptive)
                    if adaptive else 0.0
                ),
                "mean_realized_horizon": (
                    float(np.mean([int(item["realized_horizon"]) for item in deferred]))
                    if deferred else 0.0
                ),
            }
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_video:
        grouped[(str(row["model"]), str(row["variant"]))].append(row)
    for row in summary:
        selected = grouped[(str(row["model"]), str(row["variant"]))]
        row.update(
            {
                "early_commit_count": sum(int(item["early_commit_count"]) for item in selected),
                "forced_deadline_count": sum(int(item["forced_deadline_count"]) for item in selected),
                "forced_sequence_end_count": sum(
                    int(item["forced_sequence_end_count"]) for item in selected
                ),
                "early_commit_fraction": float(
                    np.mean([float(item["early_commit_fraction"]) for item in selected])
                ) if selected else 0.0,
                "mean_realized_horizon": float(
                    np.mean([float(item["mean_realized_horizon"]) for item in selected])
                ) if selected else 0.0,
            }
        )


def run_h41_tracking(
    config: ExperimentConfig,
    model_names: Sequence[str],
    *,
    override_audit_stop: bool = False,
) -> dict[str, Any]:
    h41 = require_h41(config)
    h4 = config.h4
    assert h4 is not None
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("H4.1 tracking requires unique model names")
    inputs = validate_h41_inputs(config)
    decision = _load_gate(config)
    gate_passed = decision.get("gate_passed") is True
    if not gate_passed:
        if not override_audit_stop:
            raise RuntimeError(
                "H4.1 cumulative recoverability gate failed; protocol requires STOP_BEFORE_METHOD_EVALUATION"
            )
        if not h41.allow_subset:
            raise RuntimeError("Audit-stop override is restricted to subset smoke configurations")
    by_video = _by_video_indices(inputs.observations)
    all_rows: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    cache_audit: dict[str, Any] = {}
    total = len(model_names) * len(by_video) * len(H41_VARIANTS)
    completed = 0
    reused_jobs = 0
    progress = config.paths.output_root / "h41_logs" / "tracking_progress.json"
    decision_sha256 = sha256_file(config.paths.output_root / "h41_protocol_decision.json")
    atomic_write_json(
        progress,
        {"status": "running", "completed_jobs": 0, "total_jobs": total, "final_test_read": False},
    )
    for model_name in model_names:
        embeddings, cache = load_h41_source_embeddings(config, model_name, inputs)
        cache_audit[model_name] = cache
        for video_id in sorted(by_video):
            indices = by_video[video_id]
            observations = [inputs.observations[index] for index in indices]
            values = np.asarray(embeddings[indices], dtype=np.float32)
            for variant in H41_VARIANTS:
                fingerprint = _fingerprint(
                    "adaptive_hypothesis_tracking",
                    model_name,
                    video_id,
                    variant,
                    [item.observation_id for item in observations],
                    cache.get("fingerprint"),
                    str(inputs.audit["protocol_sha256"]),
                    str(inputs.audit["source_h4_events_sha256"]),
                    str(inputs.audit["source_h4_baseline_sha256"]),
                    decision_sha256,
                )
                job_path = (
                    config.paths.output_root
                    / "h41_work"
                    / "tracking"
                    / model_name
                    / variant
                    / f"{video_id}.json"
                )
                job = _load_job(job_path, fingerprint)
                reused = job is not None
                if job is None:
                    started = time.perf_counter()
                    rows, events = track_h41_sequence(
                        model_name, variant, observations, values, h4, h41
                    )
                    elapsed = time.perf_counter() - started
                    runtime = {
                        "model": model_name,
                        "variant": variant,
                        "video_id": video_id,
                        "observation_count": len(observations),
                        "event_count": len(events),
                        "elapsed_seconds": elapsed,
                        "observations_per_second": (
                            len(observations) / elapsed if elapsed > 0 else ""
                        ),
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
                runtime["resumed_from_atomic_job"] = reused
                runtime_rows.append(runtime)
                completed += 1
                atomic_write_json(
                    progress,
                    {
                        "status": "running",
                        "completed_jobs": completed,
                        "total_jobs": total,
                        "model": model_name,
                        "video_id": video_id,
                        "variant": variant,
                        "final_test_read": False,
                    },
                )
    all_rows.sort(
        key=lambda row: (
            row["model"], row["variant"], row["video_id"],
            row["frame_id"], row["observation_id"],
        )
    )
    all_events.sort(
        key=lambda row: (
            row["model"], row["variant"], row["video_id"],
            row["start_frame"], row["event_id"],
        )
    )
    per_video, summary = summarize_gt_assignments(all_rows)
    _augment_metrics(per_video, summary, all_rows, all_events)
    _augment_h41_metrics(per_video, summary, all_events)
    paired = _paired_metrics(per_video)
    output = config.paths.output_root
    write_csv(output / "h41_assignments.csv", all_rows, H41_ASSIGNMENT_FIELDS)
    write_csv(output / "h41_conflict_events.csv", all_events)
    write_csv(output / "h41_per_video_metrics.csv", per_video)
    write_csv(output / "h41_summary.csv", summary)
    write_csv(output / "h41_paired_video_metrics.csv", paired)
    write_csv(output / "h41_runtime_profile.csv", runtime_rows)
    _write_mot_results(
        output / "h41_mot_results",
        all_rows,
        {item.observation_id: item for item in inputs.observations},
    )
    state = {
        "status": "completed",
        "models": list(model_names),
        "variants": list(H41_VARIANTS),
        "primary_variant": H41_PRIMARY_VARIANT,
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
            "SUBSET_SMOKE_NOT_EXPERIMENT"
            if h41.allow_subset
            else "EXPLORATORY_DEVELOPMENT_RESULT"
        ),
        "input_audit": inputs.audit,
        "gt_identity_used_for_decisions": False,
        "final_test_read": False,
    }
    atomic_write_json(output / "h41_tracking_metadata.json", state)
    atomic_write_json(
        progress,
        {"status": "completed", "completed_jobs": completed, "total_jobs": total, "final_test_read": False},
    )
    return state
