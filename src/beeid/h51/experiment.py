"""Resumable post-H5 gradient, teacher-forced and causal replay diagnostics."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..config import ExperimentConfig
from ..h5.experiment import _training_transitions
from ..h5.tracker import normalized_geometry, track_sequence
from ..utils import atomic_write_json, canonical_json, sha256_text
from . import H51_MODELS, H51_VARIANTS
from .audit import (
    build_path_audit,
    runtime_gradient_coverage,
    teacher_forced_one_step_audit,
)
from .core import (
    H51Inputs,
    load_frozen_h3_baseline,
    load_h51_embeddings,
    load_h51_model,
    require_h51,
    validate_h51_inputs,
)
from .report import (
    H51_REQUIRED_OUTPUTS,
    closed_loop_decision,
    summarize_closed_loop,
    write_csv,
    write_final_report,
)
from .tracker import instrumented_track_sequence


H51_REPLAY_IMPLEMENTATION = "beeid.h51.tracker:v2-instrumented-h5-v4-semantics"
H51_REPLAY_DIAGNOSTIC_SCHEMA_VERSION = 2
REPLAY_EQUIVALENCE_ATOL = 1e-6
REPLAY_EQUIVALENCE_RTOL = 1e-6


def _device(config: ExperimentConfig) -> torch.device:
    device = torch.device(config.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("H5.1 requested CUDA but torch.cuda.is_available() is false")
    return device


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _job_payload_digest(value: dict[str, Any]) -> str:
    sections = {key: value[key] for key in ("assignments", "observation_diagnostics", "frame_diagnostics", "activity", "module_calls")}
    return sha256_text(canonical_json(sections))


def _load_job(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Refusing malformed H5.1 replay cache {path}: {error}") from error
    if not isinstance(value, dict) or value.get("fingerprint") != fingerprint:
        raise RuntimeError(f"Refusing incompatible H5.1 replay cache: {path}")
    signature = value.get("signature")
    if (
        not isinstance(signature, dict)
        or sha256_text(canonical_json(signature)) != fingerprint
    ):
        raise RuntimeError(f"Refusing H5.1 replay cache with an invalid stored signature: {path}")
    if (
        signature.get("implementation") != H51_REPLAY_IMPLEMENTATION
        or signature.get("diagnostic_schema_version") != H51_REPLAY_DIAGNOSTIC_SCHEMA_VERSION
    ):
        raise RuntimeError(f"Refusing obsolete H5.1 replay cache diagnostic schema: {path}")
    if value.get("status") != "completed" or value.get("final_test_read") is not False:
        raise RuntimeError(f"Refusing incomplete H5.1 replay cache: {path}")
    required = ("assignments", "observation_diagnostics", "frame_diagnostics", "activity", "module_calls")
    if any(key not in value for key in required) or value.get("payload_sha256") != _job_payload_digest(value):
        raise RuntimeError(f"Refusing modified H5.1 replay cache: {path}")
    return value


def _write_progress(output: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(output / "logs" / "h51_progress.json", {
        **payload, "updated_at": datetime.now(timezone.utc).isoformat(), "final_test_read": False,
    })


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    raise RuntimeError(f"Invalid boolean in replay equivalence field {label}: {value!r}")


def _optional_float(value: Any, label: str) -> float | None:
    if value in {"", None}:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"Non-finite replay equivalence field {label}: {value!r}")
    return result


def replay_equivalence_audit(
    source_rows: Sequence[dict[str, Any]],
    replay_rows: Sequence[dict[str, Any]],
    model_names: Sequence[str],
    observation_ids: set[str],
    *,
    oracle: str,
) -> dict[str, Any]:
    """Require instrumentation assignments to equal the trusted H5 oracle."""
    checks: list[dict[str, Any]] = []
    mismatch_previews: list[dict[str, Any]] = []
    for model_name in model_names:
        for variant in H51_VARIANTS:
            source = [
                row for row in source_rows
                if row.get("model") == model_name
                and row.get("variant") == variant
                and str(row.get("observation_id", "")) in observation_ids
            ]
            replay = [
                row for row in replay_rows
                if row.get("model") == model_name
                and row.get("variant") == variant
                and str(row.get("observation_id", "")) in observation_ids
            ]
            source_lookup = {str(row["observation_id"]): row for row in source}
            replay_lookup = {str(row["observation_id"]): row for row in replay}
            if (
                len(source_lookup) != len(source)
                or len(replay_lookup) != len(replay)
                or set(source_lookup) != observation_ids
                or set(replay_lookup) != observation_ids
            ):
                raise RuntimeError(
                    f"H5.1 replay equivalence has duplicate/incomplete keys for {model_name}/{variant}"
                )
            mismatch_count = 0
            max_score_diff = 0.0
            max_reliability_diff = 0.0
            for observation_id in sorted(observation_ids):
                expected = source_lookup[observation_id]
                actual = replay_lookup[observation_id]
                categorical = {
                    "predicted_track_id": str(actual.get("predicted_track_id")) == str(expected.get("predicted_track_id")),
                    "matched_existing_track": _strict_bool(actual.get("matched_existing_track"), "matched_existing_track") == _strict_bool(expected.get("matched_existing_track"), "matched_existing_track"),
                    "memory_update_accepted": _strict_bool(actual.get("memory_update_accepted"), "memory_update_accepted") == _strict_bool(expected.get("memory_update_accepted"), "memory_update_accepted"),
                }
                numeric_ok = True
                for field, maximum_name in (
                    ("association_score", "score"), ("reliability", "reliability")
                ):
                    expected_value = _optional_float(expected.get(field), field)
                    actual_value = _optional_float(actual.get(field), field)
                    if expected_value is None or actual_value is None:
                        equal = expected_value is None and actual_value is None
                        difference = 0.0
                    else:
                        difference = abs(actual_value - expected_value)
                        equal = bool(np.isclose(
                            actual_value, expected_value,
                            atol=REPLAY_EQUIVALENCE_ATOL, rtol=REPLAY_EQUIVALENCE_RTOL,
                        ))
                    numeric_ok &= equal
                    if maximum_name == "score":
                        max_score_diff = max(max_score_diff, difference)
                    else:
                        max_reliability_diff = max(max_reliability_diff, difference)
                if not all(categorical.values()) or not numeric_ok:
                    mismatch_count += 1
                    if len(mismatch_previews) < 10:
                        mismatch_previews.append({
                            "model": model_name, "variant": variant,
                            "observation_id": observation_id,
                            "categorical_equal": categorical,
                            "numeric_equal": numeric_ok,
                        })
            checks.append({
                "model": model_name, "variant": variant,
                "observation_count": len(observation_ids),
                "mismatch_count": mismatch_count,
                "max_association_score_abs_diff": max_score_diff,
                "max_reliability_abs_diff": max_reliability_diff,
                "fields": [
                    "predicted_track_id", "matched_existing_track", "association_score",
                    "reliability", "memory_update_accepted",
                ],
                "pass": mismatch_count == 0,
            })
    if mismatch_previews:
        raise RuntimeError(
            "Instrumented H5.1 replay differs from its H5 assignment oracle: "
            + canonical_json(mismatch_previews)
        )
    return {
        "status": "passed", "oracle": oracle,
        "instrumented_replay_implementation": H51_REPLAY_IMPLEMENTATION,
        "diagnostic_schema_version": H51_REPLAY_DIAGNOSTIC_SCHEMA_VERSION,
        "real_source_h5_oracle": oracle == "hashed_source_h5_assignments",
        "absolute_tolerance": REPLAY_EQUIVALENCE_ATOL,
        "relative_tolerance": REPLAY_EQUIVALENCE_RTOL,
        "checks": checks, "mismatch_count": 0,
        "instrumentation_assignment_semantics_preserved": True,
        "final_test_read": False,
    }


def _run_loaded_diagnostic(
    config: ExperimentConfig,
    inputs: H51Inputs,
    model_names: Sequence[str],
    *,
    test_only: bool = False,
    supplied: dict[str, tuple[torch.nn.Module, np.ndarray]] | None = None,
) -> dict[str, Any]:
    h5 = config.h5
    assert h5 is not None
    h51 = require_h51(config)
    output = config.paths.output_root
    output.mkdir(parents=True, exist_ok=True)
    device = _device(config)
    _seed(config.runtime.seed)
    observations = inputs.observations
    development = inputs.development_indices()
    if not development:
        raise RuntimeError("H5.1 found no development_validation observations")
    by_video: dict[str, list[int]] = defaultdict(list)
    for index in development:
        by_video[observations[index].video_id].append(index)
    for indices in by_video.values():
        indices.sort(key=lambda i: (
            observations[i].frame, observations[i].center_x, observations[i].center_y,
            observations[i].observation_id,
        ))

    gradient_rows: list[dict[str, Any]] = []
    gradient_summaries: dict[str, dict[str, Any]] = {}
    teacher_rows: list[dict[str, Any]] = []
    teacher_summaries: dict[str, dict[str, Any]] = {}
    teacher_calls: Counter[str] = Counter()
    assignment_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    rollout_calls: dict[str, Counter[str]] = {variant: Counter() for variant in H51_VARIANTS}
    rollout_activity: dict[str, Counter[str]] = {variant: Counter() for variant in H51_VARIANTS}
    test_only_oracle_rows: list[dict[str, Any]] = []
    cache_audit: dict[str, Any] = {}
    total_jobs = len(model_names) * len(by_video) * len(H51_VARIANTS)
    completed_jobs = 0
    reused_jobs = 0
    _write_progress(output, {"status": "running", "completed_jobs": 0, "total_jobs": total_jobs})

    transitions = _training_transitions(inputs.h5_inputs, h5)
    if not transitions:
        raise RuntimeError("H5.1 found no project_train transitions for the runtime gradient probe")
    geometry = np.stack([normalized_geometry(item) for item in observations]).astype(np.float32)
    selected_transitions = transitions[: min(4, len(transitions))]
    for model_name in model_names:
        if supplied and model_name in supplied:
            supplied_model, embeddings = supplied[model_name]
            model = supplied_model.to(device)  # type: ignore[union-attr]
            cache = {"cache_fingerprint": "test-only-supplied", "test_only": True, "read_only": True}
        else:
            embeddings, cache = load_h51_embeddings(config, model_name, inputs)
            model = load_h51_model(config, model_name, embeddings.shape[1], inputs, device)
        cache_audit[model_name] = cache
        model = model  # type: ignore[assignment]
        coverage, gradient = runtime_gradient_coverage(
            model, selected_transitions, embeddings, geometry, h5, device  # type: ignore[arg-type]
        )
        gradient_rows.extend(
            {"model": model_name, "input_partition": "project_train", **row}
            for row in coverage
        )
        gradient["input_partition"] = "project_train"
        gradient["runtime_recreation"] = "checkpoint_pinned_h5_v2_loss_on_current_branch"
        gradient["historical_training_binary_directly_attested"] = False
        gradient_summaries[model_name] = gradient

        one_step, teacher, calls = teacher_forced_one_step_audit(
            model_name, model, observations, embeddings, development, h5, device,  # type: ignore[arg-type]
            h51.max_teacher_forced_rows,
        )
        teacher_rows.extend(one_step)
        teacher_summaries[model_name] = teacher
        teacher_calls.update(calls)
        development_ids = {observations[index].observation_id for index in development}
        if test_only:
            # Synthetic mode supplies a frozen baseline below after replay; it
            # never reads a real H3 result.
            baseline: list[dict[str, Any]] = []
        else:
            baseline = load_frozen_h3_baseline(config, model_name, development_ids)
            assignment_rows.extend(baseline)

        for video_id in sorted(by_video):
            indices = by_video[video_id]
            job_observations = [observations[index] for index in indices]
            values = np.asarray(embeddings[indices], dtype=np.float32)
            for variant in H51_VARIANTS:
                checkpoint = inputs.checkpoints.get(model_name, {})
                signature = {
                    "implementation": H51_REPLAY_IMPLEMENTATION,
                    "diagnostic_schema_version": H51_REPLAY_DIAGNOSTIC_SCHEMA_VERSION,
                    "model": model_name, "variant": variant, "video_id": video_id,
                    "observation_ids": [item.observation_id for item in job_observations],
                    "checkpoint_sha256": checkpoint.get("sha256", "test-only"),
                    "checkpoint_fingerprint": checkpoint.get("fingerprint", "test-only"),
                    "source_cache_fingerprint": cache.get("cache_fingerprint"),
                    "torch_device": str(device),
                    "h51_protocol_sha256": inputs.audit.get("h51_protocol_sha256", "test-only"),
                    "h5_protocol_sha256": inputs.audit.get("h5_protocol_sha256", "test-only"),
                    "candidate_sample_per_observation": h51.candidate_sample_per_observation,
                    "tracking_parameters": {
                        "memory_slots": h5.memory_slots, "memory_top_k": h5.memory_top_k,
                        "update_gate": h5.update_gate, "memory_mix": h5.memory_mix,
                        "max_age": h5.max_age, "min_assignment_score": h5.min_assignment_score,
                        "max_normalized_distance": h5.max_normalized_distance,
                    },
                    "gt_identity_decision_input": False, "final_test_read": False,
                }
                fingerprint = sha256_text(canonical_json(signature))
                job_path = output / "h51_work" / model_name / variant / f"{video_id}.json"
                job = _load_job(job_path, fingerprint)
                if job is None:
                    assignments, observations_out, frames_out, activity, module_calls = instrumented_track_sequence(
                        model_name, variant, model, job_observations, values, h5, device,  # type: ignore[arg-type]
                        candidate_sample_per_observation=h51.candidate_sample_per_observation,
                    )
                    job = {
                        "status": "completed", "fingerprint": fingerprint, "signature": signature,
                        "assignments": assignments, "observation_diagnostics": observations_out,
                        "frame_diagnostics": frames_out, "activity": activity,
                        "module_calls": module_calls, "final_test_read": False,
                    }
                    job["payload_sha256"] = _job_payload_digest(job)
                    atomic_write_json(job_path, job)
                else:
                    reused_jobs += 1
                assignment_rows.extend(job["assignments"])
                if test_only:
                    test_only_oracle_rows.extend(track_sequence(
                        model_name, variant, model, job_observations, values, h5, device,  # type: ignore[arg-type]
                    ))
                diagnostic_rows.extend(job["observation_diagnostics"])
                frame_rows.extend(job["frame_diagnostics"])
                rollout_activity[variant].update({key: int(value) for key, value in job["activity"].items()})
                rollout_calls[variant].update({key: int(value) for key, value in job["module_calls"].items()})
                completed_jobs += 1
                _write_progress(output, {
                    "status": "running", "completed_jobs": completed_jobs,
                    "reused_jobs": reused_jobs, "total_jobs": total_jobs,
                    "model": model_name, "variant": variant, "video_id": video_id,
                })
        if test_only:
            # The no-memory causal output is a fixed, inference-only comparison;
            # relabeling a copy avoids inventing a conventional backbone method.
            synthetic_baseline = [
                {**row, "variant": "frozen_h3_baseline"}
                for row in assignment_rows
                if row["model"] == model_name and row["variant"] == "persistent_query_no_memory"
            ]
            assignment_rows.extend(synthetic_baseline)

    gradient_rows.sort(key=lambda row: (row["model"], row["parameter"]))
    teacher_rows.sort(key=lambda row: (row["model"], row["video_id"], row["frame_id"], row["positive_observation_id"]))
    assignment_rows.sort(key=lambda row: (row["model"], row["variant"], row["video_id"], int(row["frame_id"]), row["observation_id"]))
    diagnostic_rows.sort(key=lambda row: (row["model"], row["variant"], row["video_id"], int(row["frame_id"]), row["observation_id"]))
    frame_rows.sort(key=lambda row: (row["model"], row["variant"], row["video_id"], int(row["frame_id"])))
    development_ids = {observations[index].observation_id for index in development}
    if test_only:
        equivalence_source = test_only_oracle_rows
        equivalence_oracle = "current_h5_tracker_test_only_oracle"
    else:
        source_assignment_path = inputs.source_h5_root / "h5_assignments.csv"
        with source_assignment_path.open("r", encoding="utf-8", newline="") as handle:
            equivalence_source = list(csv.DictReader(handle))
        equivalence_oracle = "hashed_source_h5_assignments"
    equivalence = replay_equivalence_audit(
        equivalence_source, assignment_rows, model_names, development_ids,
        oracle=equivalence_oracle,
    )
    atomic_write_json(output / "h51_replay_equivalence.json", equivalence)
    variant_rows, rejection_rows, _ = summarize_closed_loop(assignment_rows, diagnostic_rows)
    decision = closed_loop_decision(h51, gradient_summaries, teacher_summaries, variant_rows)
    path_audit = build_path_audit(
        {"called_modules": dict(Counter({
            key: sum(summary["called_modules"].get(key, 0) for summary in gradient_summaries.values())
            for key in {name for summary in gradient_summaries.values() for name in summary["called_modules"]}
        })),
         "memory_defect_detected": all(summary["memory_defect_detected"] for summary in gradient_summaries.values()),
         "memory_missing_gradient_parameters": sorted({
             name
             for summary in gradient_summaries.values()
             for name in summary["memory_missing_gradient_parameters"]
         }),
         "method_ready": all(summary["method_ready"] for summary in gradient_summaries.values())},
        dict(teacher_calls),
        {variant: dict(calls) for variant, calls in rollout_calls.items()},
        {variant: dict(activity) for variant, activity in rollout_activity.items()},
    )
    path_audit["per_model_gradient_summary"] = gradient_summaries
    path_audit["per_model_teacher_forced_summary"] = teacher_summaries
    path_audit["final_test_read"] = False

    write_csv(output / "h51_gradient_coverage.csv", gradient_rows)
    atomic_write_json(output / "h51_path_audit.json", path_audit)
    write_csv(output / "h51_teacher_forced_queries.csv", teacher_rows)
    write_csv(output / "h51_rollout_observation_diagnostics.csv", diagnostic_rows)
    write_csv(output / "h51_frame_diagnostics.csv", frame_rows)
    write_csv(output / "h51_variant_summary.csv", variant_rows)
    write_csv(output / "h51_rejection_summary.csv", rejection_rows)
    atomic_write_json(output / "h51_decision.json", decision)
    metadata = write_final_report(
        output, input_audit={**inputs.audit, "cache_audit": cache_audit},
        decision=decision, models=model_names, device=device, test_only=test_only,
    )
    missing = [name for name in H51_REQUIRED_OUTPUTS if not (output / name).is_file()]
    if not (output / "logs").is_dir():
        missing.append("logs/")
    if missing:
        raise RuntimeError("H5.1 did not produce its required outputs: " + ", ".join(missing))
    _write_progress(output, {
        "status": "completed", "completed_jobs": completed_jobs,
        "reused_jobs": reused_jobs, "total_jobs": total_jobs,
        "decision": decision["status"], "method_ready": decision["method_ready"],
    })
    return {
        "status": "passed", "decision": decision["status"],
        "method_ready": decision["method_ready"], "models": list(model_names),
        "gradient_memory_defect_detected": all(
            summary["memory_defect_detected"] for summary in gradient_summaries.values()
        ),
        "assignment_rows": len(assignment_rows), "diagnostic_rows": len(diagnostic_rows),
        "reused_jobs": reused_jobs, "metadata_status": metadata["status"],
        "output_root": str(output), "final_test_read": False,
    }


def run_h51_diagnostic(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    if len(model_names) != len(H51_MODELS) or set(model_names) != set(H51_MODELS):
        raise ValueError(
            "Real H5.1 diagnostics require exactly both frozen H5 backbones: "
            + ", ".join(H51_MODELS)
        )
    inputs = validate_h51_inputs(config)
    return _run_loaded_diagnostic(config, inputs, model_names)


def run_h51_all(
    config: ExperimentConfig, model_names: Sequence[str], confirm_full: bool
) -> dict[str, Any]:
    h51 = require_h51(config)
    if not h51.allow_subset and not confirm_full:
        raise RuntimeError(
            "Full H5.1 development diagnostic requires --confirm-full after a minimal smoke "
            "containing one project_train and one development_validation video"
        )
    return run_h51_diagnostic(config, model_names)
