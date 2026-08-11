from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from beeid.config import ConfigurationError, load_config
from beeid.h3.core import H3Inputs
from beeid.h5.core import H5Inputs
from beeid.h5.model import BeeTrackQuery
from beeid.h5.tracker import track_sequence
from beeid.h5.tracker import TRACKER_IMPLEMENTATION
from beeid.h51 import H51_MODELS, H51_VARIANTS
from beeid.h51.audit import (
    build_path_audit,
    runtime_gradient_coverage,
    synthetic_transition_batch,
    teacher_forced_one_step_audit,
)
from beeid.h51.core import H51InputError, load_frozen_h3_baseline, validate_h51_inputs
from beeid.h51.experiment import (
    H51_REPLAY_DIAGNOSTIC_SCHEMA_VERSION,
    H51_REPLAY_IMPLEMENTATION,
    _job_payload_digest,
    _load_job,
    replay_equivalence_audit,
)
from beeid.h51.protocol import H51ProtocolError, validate_h51_protocol
from beeid.h51.report import H51_REQUIRED_OUTPUTS, closed_loop_decision, runtime_validation_fields
from beeid.h51.synthetic import _observation, h51_synthetic_smoke
from beeid.h51.tracker import REJECTION_REASONS, classify_rejection, instrumented_track_sequence
from beeid.utils import atomic_write_json, canonical_json, sha256_file, sha256_text


ROOT = Path(__file__).resolve().parents[1]


def _toy_development() -> tuple[list, np.ndarray]:
    observations = [
        _observation(
            "development_validation", "toy", frame, identity,
            10.0 + frame if identity == 1 else 62.0 - frame,
        )
        for frame in range(1, 7)
        for identity in (1, 2)
    ]
    embeddings = np.asarray(
        [[1.0, 0.4 * item.track_id, item.center_x / 100.0, item.frame / 20.0] for item in observations],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return observations, embeddings


def test_h51_protocol_checksum_config_and_local_ignore(tmp_path):
    protocol = validate_h51_protocol(
        ROOT / "configs" / "h51_protocol.lock.yaml",
        ROOT / "configs" / "h51_protocol.lock.sha256",
    )
    assert protocol["final_test_access"] is False
    assert protocol["diagnostic_not_confirmatory"] is True
    assert protocol["runtime_gradient_probe_partition"] == "project_train"
    assert protocol["causal_rollout_partition"] == "development_validation"
    assert protocol["max_teacher_forced_rows_full"] == 250000
    assert protocol["max_teacher_forced_rows_subset_smoke"] == 10000
    assert protocol["candidate_sample_per_observation"] == 3
    assert protocol["replay_diagnostic_schema_version"] == H51_REPLAY_DIAGNOSTIC_SCHEMA_VERSION
    assert protocol["gradient_health_ready_rule"] == (
        "all_trainable_parameters_grad_present_finite_and_nonzero"
    )
    config = load_config(ROOT / "configs" / "h51_smoke.example.yaml")
    assert config.h51 is not None and config.paths.h5_output_root is not None
    assert config.h51.collapse_new_track_rate == protocol["thresholds"]["collapse_new_track_rate"]
    modified = tmp_path / "lock.yaml"
    modified.write_text((ROOT / "configs" / "h51_protocol.lock.yaml").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(H51ProtocolError, match="checksum mismatch"):
        validate_h51_protocol(modified, ROOT / "configs" / "h51_protocol.lock.sha256")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "configs/*.local.yaml" in ignore
    assert (ROOT / "configs" / "h51.local.yaml.example").is_file()


def test_h51_config_rejects_final_test_and_overlapping_h5_output(tmp_path):
    source = (ROOT / "configs" / "h51_smoke.example.yaml").read_text(encoding="utf-8")
    final = tmp_path / "final.yaml"
    final.write_text(source.replace("source_splits: [train]", "source_splits: [test]"), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(final)
    overlap = tmp_path / "overlap.yaml"
    overlap.write_text(
        source.replace(
            "h5_output_root: /path/to/experiments/beeid/h5-beetrackquery-dev",
            "h5_output_root: /path/to/experiments/beeid/h51-smoke/source-h5",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="H5.1 output_root"):
        load_config(overlap)


def _fake_h51_source(tmp_path: Path, monkeypatch) -> tuple:
    import beeid.h51.core as core

    config = load_config(ROOT / "configs" / "h51_smoke.example.yaml")
    source = tmp_path / "source-h5"
    h3_source = tmp_path / "source-h3"
    configured = replace(
        config,
        paths=replace(
            config.paths, output_root=tmp_path / "h51-output",
            h5_output_root=source, h3_output_root=h3_source,
        ),
    )
    protocol = validate_h51_protocol(
        configured.h51.protocol_lock_path, configured.h51.protocol_checksum_path  # type: ignore[union-attr]
    )
    audit = {
        "h5_protocol_sha256": protocol["source_h5_protocol_sha256"],
        "protocol_sha256": protocol["source_h3_protocol_sha256"],
        "project_split_sha256": protocol["project_split_sha256"],
        "manifest_sha256": "manifest-test", "final_test_read": False,
    }
    fake = H5Inputs(H3Inputs((), {}, {}, audit), audit)
    monkeypatch.setattr(core, "validate_h5_inputs", lambda _config: fake)
    source.mkdir(parents=True)
    h3_source.mkdir(parents=True)
    h3_assignments = h3_source / "h3_assignments.csv"
    h3_assignments.write_text("model,variant,observation_id\n", encoding="utf-8")
    atomic_write_json(h3_source / "h3_run_metadata.json", {
        "status": "completed_development_gt_boxes",
        "assignment_sha256": sha256_file(h3_assignments),
        "final_test_read": False,
    })
    (source / "h5_assignments.csv").write_text("model,variant\n", encoding="utf-8")
    (source / "h5_summary.csv").write_text("model,variant\n", encoding="utf-8")
    decision = {
        "status": "STOP_OR_REVISE_BEETRACKQUERY", "gate_passed": False,
        "final_test_read": False,
    }
    atomic_write_json(source / "h5_method_decision.json", decision)
    registry = {}
    for model_name in H51_MODELS:
        directory = source / "h5_checkpoints" / model_name
        directory.mkdir(parents=True)
        checkpoint = directory / "final.pt"
        torch.save({"test": torch.ones(1)}, checkpoint)
        signature = {
            "implementation": "beeid.h5:v2-batched-transitions",
            "model": model_name,
            "h5_protocol_sha256": protocol["source_h5_protocol_sha256"],
            "project_split_sha256": protocol["project_split_sha256"],
            "manifest_sha256": "manifest-test", "fit_partition": "project_train",
            "final_test_read": False,
        }
        metadata = {
            "status": "completed", "fingerprint": sha256_text(canonical_json(signature)),
            "checkpoint_sha256": sha256_file(checkpoint), "signature": signature,
            "final_test_read": False,
        }
        atomic_write_json(directory / "checkpoint.json", metadata)
        registry[model_name] = dict(metadata)
    training = {"status": "completed", "models": list(H51_MODELS), "checkpoints": registry, "final_test_read": False}
    tracking = {
        "status": "completed_development_gt_boxes", "models": list(H51_MODELS),
        "tracker_implementation": TRACKER_IMPLEMENTATION,
        "gt_identity_used_for_decisions": False, "final_test_read": False,
    }
    atomic_write_json(source / "h5_training_metadata.json", training)
    atomic_write_json(source / "h5_tracking_metadata.json", tracking)
    run = {
        "status": "completed_development_gt_boxes", "models": list(H51_MODELS),
        "input_audit": audit, "decision": decision,
        "result_hashes": {
            "assignments": sha256_file(source / "h5_assignments.csv"),
            "summary": sha256_file(source / "h5_summary.csv"),
            "decision": sha256_file(source / "h5_method_decision.json"),
        },
        "final_test_read": False,
    }
    atomic_write_json(source / "h5_run_metadata.json", run)
    return configured, source


def test_h51_input_provenance_and_final_test_metadata_lock(tmp_path, monkeypatch):
    configured, source = _fake_h51_source(tmp_path, monkeypatch)
    inputs = validate_h51_inputs(configured)
    assert inputs.audit["source_h5_failure_diagnosis"] is True
    assert set(inputs.checkpoints) == set(H51_MODELS)
    assert set(inputs.audit["source_h5_checkpoints"]) == set(H51_MODELS)
    assert all(
        len(inputs.audit["source_h5_checkpoints"][model]["sha256"]) == 64
        and inputs.audit["source_h5_checkpoints"][model]["fingerprint"]
        for model in H51_MODELS
    )
    assert inputs.audit["source_h3_baseline"]["assignments_sha256"] == sha256_file(
        Path(inputs.audit["source_h3_baseline"]["assignments_path"])
    )
    assert inputs.audit["gradient_probe_provenance"]["historical_training_binary_directly_attested"] is False
    checkpoint_metadata_path = source / "h5_checkpoints" / "resnet50" / "checkpoint.json"
    checkpoint_metadata = json.loads(checkpoint_metadata_path.read_text(encoding="utf-8"))
    checkpoint_metadata["signature"]["implementation"] = "unattested-implementation"
    atomic_write_json(checkpoint_metadata_path, checkpoint_metadata)
    with pytest.raises(H51InputError, match="checkpoint provenance mismatch"):
        validate_h51_inputs(configured)
    checkpoint_metadata["signature"]["implementation"] = "beeid.h5:v2-batched-transitions"
    atomic_write_json(checkpoint_metadata_path, checkpoint_metadata)
    tracking = json.loads((source / "h5_tracking_metadata.json").read_text(encoding="utf-8"))
    tracking["final_test_read"] = True
    atomic_write_json(source / "h5_tracking_metadata.json", tracking)
    with pytest.raises(H51InputError, match="final_test_read=false"):
        validate_h51_inputs(configured)


def test_runtime_gradient_audit_identifies_exact_six_and_other_gradients_are_finite():
    config = load_config(ROOT / "configs" / "h51_smoke.example.yaml")
    assert config.h5 is not None
    model = BeeTrackQuery(4, 16, 4, 0.0)
    transitions, embeddings, geometry = synthetic_transition_batch(4)
    rows, summary = runtime_gradient_coverage(
        model, transitions, embeddings, geometry, config.h5, torch.device("cpu")
    )
    expected = {
        "memory_attention.in_proj_weight", "memory_attention.in_proj_bias",
        "memory_attention.out_proj.weight", "memory_attention.out_proj.bias",
        "memory_norm.weight", "memory_norm.bias",
    }
    assert set(summary["memory_missing_gradient_parameters"]) == expected
    assert summary["unexpected_missing_gradient_parameters"] == []
    assert summary["memory_defect_detected"] is True
    assert summary["method_ready"] is False
    assert summary["all_trainable_gradients_healthy"] is False
    assert summary["all_nonmemory_trainable_gradients_finite"] is True
    assert summary["called_modules"].get("memory_attention", 0) == 0
    assert summary["probe_transition_count"] == len(transitions)
    assert summary["probe_current_candidate_counts"] == [2]
    assert summary["probe_multi_candidate_only"] is True
    nonmemory = [row for row in rows if row["parameter"] not in expected]
    assert all(row["grad_present"] and row["finite"] for row in nonmemory)
    finite_zero = [
        row["parameter"] for row in rows
        if row["grad_present"] and row["finite"] and not row["grad_nonzero"]
    ]
    assert summary["zero_gradient_parameters"] == finite_zero
    # Exact zero norms may differ across CPU/GPU kernels.  Any finite zero must
    # fail readiness; the six missing memory gradients do so independently.
    assert not finite_zero or summary["method_ready"] is False
    for row in nonmemory:
        assert row["used_in_training_objective"] == row["grad_nonzero"]


def test_path_hooks_and_teacher_forced_boundary_are_machine_readable():
    config = load_config(ROOT / "configs" / "h51_smoke.example.yaml")
    assert config.h5 is not None
    observations, embeddings = _toy_development()
    model = BeeTrackQuery(4, 16, 4, 0.0).eval()
    rows, summary, calls = teacher_forced_one_step_audit(
        "test", model, observations, embeddings, list(range(len(observations))),
        config.h5, torch.device("cpu"), 100,
    )
    assert rows and all(row["offline_gt_audit"] is True for row in rows)
    assert all(row["deployable_rollout"] is False and row["memory_read"] is False for row in rows)
    assert calls["frame_attention"] > 0 and calls.get("memory_attention", 0) == 0
    path = build_path_audit(
        {
            "called_modules": {"frame_attention": 1},
            "memory_defect_detected": True,
            "memory_missing_gradient_parameters": ["memory_attention.in_proj_weight"],
            "method_ready": False,
        },
        calls, {"beetrackquery_short_memory": {"memory_attention": 2}},
        {"beetrackquery_short_memory": {"effective_memory_read_count": 2}},
    )
    assert path["training_path"]["memory_read_called"] is False
    assert path["deployable_rollout_path"]["gt_identity_decision_input"] is False
    assert path["boundary"]["teacher_forced_metrics_are_method_results"] is False
    assert path["memory_parameters_used_in_training_objective"] is False
    assert summary["offline_gt_audit"] is True
    partial_missing = build_path_audit(
        {
            "called_modules": {"memory_attention": 0},
            "memory_defect_detected": False,
            "memory_missing_gradient_parameters": ["memory_norm.bias"],
            "method_ready": False,
        },
        {}, {}, {},
    )
    assert partial_missing["memory_parameters_used_in_training_objective"] is False
    complete = build_path_audit(
        {
            "called_modules": {"memory_attention": 1},
            "memory_defect_detected": False,
            "memory_missing_gradient_parameters": [],
            "method_ready": True,
        },
        {}, {}, {},
    )
    assert complete["memory_parameters_used_in_training_objective"] is True


def test_teacher_forced_rank1_ties_fail_strict_unique_gate():
    config = load_config(ROOT / "configs" / "h51_smoke.example.yaml")
    assert config.h5 is not None
    observations, embeddings = _toy_development()
    model = BeeTrackQuery(4, 16, 4, 0.0).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    rows, summary, _ = teacher_forced_one_step_audit(
        "tie", model, observations, embeddings, list(range(len(observations))),
        config.h5, torch.device("cpu"), 100,
    )
    assert rows
    assert all(row["positive_score_tied"] is True for row in rows)
    assert all(row["positive_score_tie_count"] == 2 for row in rows)
    assert all(row["strict_unique_rank1"] is False for row in rows)
    assert all(row["one_step_continuity"] is False for row in rows)
    assert summary["rank1_rate"] == 0.0
    assert summary["positive_score_tie_rate"] == 1.0
    assert summary["rank1_definition"].endswith("ties_fail")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"matched": True, "active_tracks": 1, "motion_valid_candidates": 1, "above_score_candidates": 1}, "matched"),
        ({"matched": False, "active_tracks": 0, "motion_valid_candidates": 0, "above_score_candidates": 0}, "no_active_track"),
        ({"matched": False, "active_tracks": 2, "motion_valid_candidates": 0, "above_score_candidates": 0}, "no_motion_valid_candidate"),
        ({"matched": False, "active_tracks": 2, "motion_valid_candidates": 1, "above_score_candidates": 0}, "score_below_threshold"),
        ({"matched": False, "active_tracks": 2, "motion_valid_candidates": 1, "above_score_candidates": 1}, "assignment_conflict"),
    ],
)
def test_rejection_taxonomy_all_branches(kwargs, expected):
    assert classify_rejection(**kwargs) == expected
    assert expected in REJECTION_REASONS


def test_instrumented_rollout_equals_h5_tracker_and_is_identity_permutation_invariant():
    config = load_config(ROOT / "configs" / "h51_smoke.example.yaml")
    assert config.h5 is not None
    observations, embeddings = _toy_development()
    torch.manual_seed(71)
    model = BeeTrackQuery(4, 16, 4, 0.0).eval()
    expected_all = []
    actual_all = []
    for variant in H51_VARIANTS:
        expected = track_sequence(
            "m", variant, model, observations, embeddings, config.h5, torch.device("cpu")
        )
        actual, diagnostics, frames, _, _ = instrumented_track_sequence(
            "m", variant, model, observations, embeddings, config.h5, torch.device("cpu"),
            candidate_sample_per_observation=3,
        )
        expected_all.extend(expected)
        actual_all.extend(actual)
        assert [(row["predicted_track_id"], row["matched_existing_track"]) for row in actual] == [
            (row["predicted_track_id"], row["matched_existing_track"]) for row in expected
        ]
        assert np.allclose(
            [float(row["association_score"]) for row in actual if row["association_score"] != ""],
            [float(row["association_score"]) for row in expected if row["association_score"] != ""],
        )
        assert len(diagnostics) == len(observations) and frames
        assert all(row["gt_identity_used_for_inference_decision"] is False for row in diagnostics)
        for row in diagnostics:
            sample = json.loads(row["candidate_sample_json"])
            assert all(
                np.isfinite(candidate["logit"]) and np.isfinite(candidate["score"])
                for candidate in sample
            )
            if row["active_tracks"]:
                assert np.isfinite(row["max_any_logit"])
                assert np.isfinite(row["top1_logit"])
            if row["motion_valid_candidate_count"]:
                assert np.isfinite(row["max_motion_valid_logit"])
            if row["matched_existing_track"]:
                assert isinstance(row["association_continuity_correct"], bool)
                assert isinstance(row["accepted_incorrect_update"], bool)
            else:
                assert row["association_continuity_correct"] == ""
                assert row["accepted_incorrect_update"] == ""
        permuted = [replace(item, identity=f"permuted::{99 - item.track_id}") for item in observations]
        permuted_rows, permuted_diagnostics, _, _, _ = instrumented_track_sequence(
            "m", variant, model, permuted, embeddings, config.h5, torch.device("cpu"),
            candidate_sample_per_observation=3,
        )
        assert [row["predicted_track_id"] for row in permuted_rows] == [row["predicted_track_id"] for row in actual]
        assert [row["rejection_reason"] for row in permuted_diagnostics] == [row["rejection_reason"] for row in diagnostics]
        assert [row["gate_accepted"] for row in permuted_diagnostics] == [row["gate_accepted"] for row in diagnostics]
        assert [row["association_continuity_correct"] for row in permuted_diagnostics] == [
            row["association_continuity_correct"] for row in diagnostics
        ]
        assert [row["accepted_incorrect_update"] for row in permuted_diagnostics] == [
            row["accepted_incorrect_update"] for row in diagnostics
        ]
    equivalence = replay_equivalence_audit(
        expected_all, actual_all, ["m"], {row.observation_id for row in observations},
        oracle="current_h5_tracker_test_only_oracle",
    )
    assert equivalence["mismatch_count"] == 0
    assert len(equivalence["checks"]) == len(H51_VARIANTS)
    assert equivalence["diagnostic_schema_version"] == 2
    assert equivalence["absolute_tolerance"] == 5e-6
    numeric_oracle = [dict(row) for row in expected_all]
    within_tolerance = [dict(row) for row in actual_all]
    numeric_index = 0
    numeric_oracle[numeric_index]["association_score"] = 0.5
    within_tolerance[numeric_index]["association_score"] = 0.5 + 3e-6
    assert replay_equivalence_audit(
        numeric_oracle, within_tolerance, ["m"],
        {row.observation_id for row in observations},
        oracle="current_h5_tracker_test_only_oracle",
    )["mismatch_count"] == 0
    outside_tolerance = [dict(row) for row in actual_all]
    outside_tolerance[numeric_index]["association_score"] = 0.5 + 1e-4
    with pytest.raises(RuntimeError, match="differs from its H5 assignment oracle"):
        replay_equivalence_audit(
            numeric_oracle, outside_tolerance, ["m"],
            {row.observation_id for row in observations},
            oracle="current_h5_tracker_test_only_oracle",
        )
    tampered = [dict(row) for row in actual_all]
    tampered[0]["predicted_track_id"] = int(tampered[0]["predicted_track_id"]) + 1000
    with pytest.raises(RuntimeError, match="differs from its H5 assignment oracle"):
        replay_equivalence_audit(
            expected_all, tampered, ["m"], {row.observation_id for row in observations},
            oracle="current_h5_tracker_test_only_oracle",
        )


def test_gate_degeneracy_collapse_and_ready_decisions():
    config = load_config(ROOT / "configs" / "h51_smoke.example.yaml")
    assert config.h51 is not None
    gradient_bad = {"m": {"method_ready": False, "memory_defect_detected": True}}
    teacher = {"m": {"rank1_rate": 0.9}}
    collapsed = [{
        "model": "m", "variant": "beetrackquery_gated_memory", "video_id": "v",
        "new_track_rate": 0.8, "prediction_inflation_vs_frozen_h3": 3.0,
        "eligible_gate_update_count": 20, "accepted_gate_update_count": 20,
        "accepted_incorrect_update_count": 15, "IDF1": 0.1,
    }]
    stopped = closed_loop_decision(config.h51, gradient_bad, teacher, collapsed)
    assert stopped["status"] == "STOP_H51_IMPLEMENTATION_NOT_READY"
    check = stopped["checks"][0]
    assert check["track_collapse_detected"] is True
    assert check["gate_degeneracy_detected"] is True
    assert check["gate_accepted_incorrect_update_rate"] == 0.75
    passing = [{**collapsed[0], "new_track_rate": 0.1, "prediction_inflation_vs_frozen_h3": 1.0,
                "accepted_gate_update_count": 10, "IDF1": 0.8}]
    subset = closed_loop_decision(
        config.h51, {"m": {"method_ready": True, "memory_defect_detected": False}},
        teacher, passing,
    )
    assert subset["status"] == "STOP_H51_IMPLEMENTATION_NOT_READY"
    assert subset["scope"] == "bounded_subset_diagnostic"
    assert subset["subset_scope_blocks_readiness"] is True
    full = replace(config.h51, allow_subset=False, max_teacher_forced_rows=250000)
    ready = closed_loop_decision(
        full, {"m": {"method_ready": True, "memory_defect_detected": False}}, teacher, passing,
    )
    assert ready["status"] == "READY_FOR_H52_PROTOCOL_DESIGN"
    assert ready["decision_confirmatory"] is False
    assert ready["final_test_unlocked"] is False


def test_atomic_replay_resume_and_signature_mismatch(tmp_path):
    path = tmp_path / "job.json"
    signature = {
        "implementation": H51_REPLAY_IMPLEMENTATION,
        "diagnostic_schema_version": H51_REPLAY_DIAGNOSTIC_SCHEMA_VERSION,
        "final_test_read": False,
    }
    fingerprint = sha256_text(canonical_json(signature))
    payload = {
        "status": "completed", "fingerprint": fingerprint, "signature": signature,
        "assignments": [], "observation_diagnostics": [], "frame_diagnostics": [],
        "activity": {}, "module_calls": {}, "final_test_read": False,
    }
    payload["payload_sha256"] = _job_payload_digest(payload)
    atomic_write_json(path, payload)
    assert _load_job(path, fingerprint) == payload
    with pytest.raises(RuntimeError, match="incompatible"):
        _load_job(path, "b")
    payload["signature"] = {"implementation": "tampered", "final_test_read": False}
    atomic_write_json(path, payload)
    with pytest.raises(RuntimeError, match="invalid stored signature"):
        _load_job(path, fingerprint)
    obsolete_signature = {
        "implementation": "beeid.h51.tracker:v1-instrumented-h5-v4-semantics",
        "diagnostic_schema_version": 1,
        "final_test_read": False,
    }
    obsolete_fingerprint = sha256_text(canonical_json(obsolete_signature))
    obsolete = {
        **payload, "fingerprint": obsolete_fingerprint, "signature": obsolete_signature,
        "activity": {},
    }
    obsolete["payload_sha256"] = _job_payload_digest(obsolete)
    atomic_write_json(path, obsolete)
    with pytest.raises(RuntimeError, match="obsolete H5.1 replay cache diagnostic schema"):
        _load_job(path, obsolete_fingerprint)
    payload["signature"] = signature
    payload["fingerprint"] = fingerprint
    payload["activity"] = {"changed": 1}
    atomic_write_json(path, payload)
    with pytest.raises(RuntimeError, match="modified"):
        _load_job(path, fingerprint)


def test_real_h51_entrypoint_requires_exact_two_backbones(monkeypatch):
    import beeid.h51.experiment as experiment

    config = load_config(ROOT / "configs" / "h51_smoke.example.yaml")
    with pytest.raises(ValueError, match="exactly both"):
        experiment.run_h51_diagnostic(config, ["resnet50"])
    with pytest.raises(ValueError, match="exactly both"):
        experiment.run_h51_diagnostic(config, ["resnet50", "resnet50"])
    sentinel = object()
    monkeypatch.setattr(experiment, "validate_h51_inputs", lambda _config: sentinel)
    monkeypatch.setattr(
        experiment, "_run_loaded_diagnostic",
        lambda _config, inputs, models: {"inputs": inputs, "models": list(models)},
    )
    result = experiment.run_h51_diagnostic(config, ["dinov3", "resnet50"])
    assert result == {"inputs": sentinel, "models": ["dinov3", "resnet50"]}


def test_frozen_h3_baseline_rejects_duplicate_observation_rows(tmp_path):
    config = load_config(ROOT / "configs" / "h51_smoke.example.yaml")
    source = tmp_path / "h3"
    source.mkdir()
    assignment = source / "h3_assignments.csv"
    assignment.write_text(
        "model,variant,observation_id\n"
        "resnet50,baseline_association,o1\n"
        "resnet50,baseline_association,o1\n",
        encoding="utf-8",
    )
    atomic_write_json(source / "h3_run_metadata.json", {
        "status": "completed_development_gt_boxes",
        "assignment_sha256": sha256_file(assignment),
        "final_test_read": False,
    })
    configured = replace(config, paths=replace(config.paths, h3_output_root=source))
    with pytest.raises(H51InputError, match="exactly one row"):
        load_frozen_h3_baseline(configured, "resnet50", {"o1"})


def test_h51_synthetic_smoke_has_scientific_gate_and_complete_outputs(tmp_path):
    output = tmp_path / "h51"
    result = h51_synthetic_smoke(output)
    assert result["status"] == "passed"
    assert result["gradient_memory_defect_detected"] is True
    assert result["decision"] == "STOP_H51_IMPLEMENTATION_NOT_READY"
    assert result["method_ready"] is False
    assert all(result["scientific_checks"].values())
    assert all((tmp_path / "h51" / name).is_file() for name in H51_REQUIRED_OUTPUTS)
    guide = (tmp_path / "h51" / "h51_result_guide.md").read_text(encoding="utf-8")
    assert all(f"`{name}`" in guide for name in H51_REQUIRED_OUTPUTS)
    metadata = json.loads((tmp_path / "h51" / "h51_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["test_only"] is True
    assert metadata["real_experiment_result"] is False
    assert metadata["environment"]["actual_torch_device"] == "cpu"
    assert metadata["runtime_validation"] == "test_only_scientific_smoke_completed"
    assert metadata["gpu_execution_validation"] == "SERVER_VALIDATION_PENDING"
    assert metadata["repository"]["commit"]
    assert isinstance(metadata["repository"]["dirty"], bool)
    assert len(metadata["repository"]["tracked_diff_sha256"]) == 64
    assert metadata["final_test_read"] is False
    assert metadata["input_audit"]["runtime_gradient_probe_partition"] == "project_train"
    assert metadata["input_audit"]["causal_rollout_partition"] == "development_validation"
    assert metadata["input_audit"]["replay_diagnostic_schema_version"] == 2
    assert metadata["input_audit"]["diagnostic_output_limits"] == {
        "max_teacher_forced_rows": 10000,
        "teacher_forced_rows_scope": "bounded_subset_smoke_override",
        "candidate_sample_per_observation": 3,
    }
    assert metadata["decision"]["scope"] == "bounded_subset_diagnostic"
    path_audit = json.loads((tmp_path / "h51" / "h51_path_audit.json").read_text(encoding="utf-8"))
    gradient_summary = path_audit["per_model_gradient_summary"]["test_only_encoder"]
    assert gradient_summary["probe_multi_candidate_only"] is True
    assert gradient_summary["probe_current_candidate_counts"] == [2]
    with (tmp_path / "h51" / "h51_gradient_coverage.csv").open("r", encoding="utf-8", newline="") as handle:
        gradient_rows = list(csv.DictReader(handle))
    finite_zero = [
        row["parameter"] for row in gradient_rows
        if row["grad_present"] == "True" and row["finite"] == "True"
        and row["grad_nonzero"] == "False"
    ]
    assert gradient_summary["zero_gradient_parameters"] == finite_zero
    assert not finite_zero or gradient_summary["method_ready"] is False
    assert gradient_summary["all_trainable_gradients_healthy"] is False
    equivalence = json.loads((tmp_path / "h51" / "h51_replay_equivalence.json").read_text(encoding="utf-8"))
    assert equivalence["oracle"] == "current_h5_tracker_test_only_oracle"
    assert equivalence["real_source_h5_oracle"] is False
    assert equivalence["mismatch_count"] == 0
    assert equivalence["diagnostic_schema_version"] == 2
    with (tmp_path / "h51" / "h51_variant_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    gated = [row for row in summaries if row["variant"] == "beetrackquery_gated_memory"]
    assert gated and all(row["reliability_label_scope"] for row in gated)
    assert all(int(row["reliability_audit_count"]) > 0 for row in gated)
    assert all(np.isfinite(float(row["reliability_brier_score"])) for row in gated)
    assert all(np.isfinite(float(row["reliability_ece_10_bin"])) for row in gated)
    rerun = h51_synthetic_smoke(output)
    assert rerun["status"] == "passed"
    assert rerun["reused_jobs"] == len(H51_VARIANTS)
    manifest_path = output / "h51_artifact_signatures.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "h51_artifact_signatures.json" not in manifest["artifacts"]
    assert "h51_run_metadata.json" not in manifest["artifacts"]
    assert manifest["signature"] == sha256_text(canonical_json(manifest["artifacts"]))
    assert all(
        digest == sha256_file(output / name)
        for name, digest in manifest["artifacts"].items()
    )
    rerun_metadata = json.loads((output / "h51_run_metadata.json").read_text(encoding="utf-8"))
    assert rerun_metadata["artifact_signature"] == sha256_file(manifest_path)


def test_h51_runtime_validation_status_boundaries():
    test_only = runtime_validation_fields(torch.device("cpu"), True)
    assert test_only == {
        "runtime_validation": "test_only_scientific_smoke_completed",
        "gpu_execution_validation": "SERVER_VALIDATION_PENDING",
        "real_gpu_validation": "SERVER_VALIDATION_PENDING",
        "real_diagnostic_execution": "not_applicable_test_only",
    }
    real_cpu = runtime_validation_fields(torch.device("cpu"), False)
    assert real_cpu["runtime_validation"] == "real_diagnostic_completed_on_cpu"
    assert real_cpu["real_diagnostic_execution"] == "completed_on_cpu"
    assert real_cpu["gpu_execution_validation"] == "SERVER_VALIDATION_PENDING"
    real_cuda = runtime_validation_fields(torch.device("cuda:0"), False)
    assert real_cuda["runtime_validation"] == "real_diagnostic_completed_on_cuda"
    assert real_cuda["real_diagnostic_execution"] == "completed_on_cuda"
    assert real_cuda["gpu_execution_validation"] == "COMPLETED_ON_CONFIGURED_CUDA_RUNTIME"
    assert "4090" not in canonical_json(real_cuda)


def test_h51_linux_scripts_and_documented_server_boundary():
    for name in (
        "run_h51.sh", "resume_h51.sh", "h51_smoke_test.sh", "monitor_h51.sh",
        "package_h51_results.sh",
    ):
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'CPU_THREAD_DEFAULT="${BEEID_H51_CPU_THREADS:-16}"' in script
        assert '[[ ! "${CPU_THREAD_DEFAULT}" =~ ^[1-9][0-9]*$ ]]' in script
        assert "BEEID_H51_CPU_THREADS must be a positive integer when set." in script
        assert '[[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]' in script
        assert 'export OMP_NUM_THREADS="${CPU_THREAD_DEFAULT}"' in script
        assert '[[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]' in script
        assert 'export MKL_NUM_THREADS="${OMP_NUM_THREADS}"' in script
    package = (ROOT / "scripts" / "package_h51_results.sh").read_text(encoding="utf-8")
    for name in H51_REQUIRED_OUTPUTS:
        assert name in package
    protocol = (ROOT / "docs" / "h51_protocol.md").read_text(encoding="utf-8")
    deployment = (ROOT / "docs" / "h51_server_deployment.md").read_text(encoding="utf-8")
    assert "diagnostic_not_confirmatory" in protocol
    assert "one `project_train` video" in deployment
    assert "one `development_validation` video" in deployment
    assert "`h51_replay_equivalence.json`" in deployment
    assert "assignment equivalence test logs" not in deployment
    assert "environment.actual_torch_device" in deployment
    assert "Do not run a fixed detector or final test" in deployment
    assert "SERVER_VALIDATION_PENDING" in deployment
    assert "cp configs/h51_smoke.example.yaml configs/h51_smoke.local.yaml" in deployment
    assert "cp configs/h51.local.yaml.example configs/h51_smoke.local.yaml" not in deployment
    workflow = (ROOT / ".github" / "workflows" / "cpu-ci.yml").read_text(encoding="utf-8")
    assert "beeid h51-validate-protocol --protocol configs/h51_protocol.lock.yaml --checksum configs/h51_protocol.lock.sha256" in workflow
    assert "beeid h51-synthetic-smoke" in workflow


@pytest.mark.skipif(os.name == "nt", reason="Windows bash path conversion is not portable")
def test_h51_shell_thread_fallback_behavior_when_bash_is_available(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash is unavailable")
    fake = tmp_path / "fake-python"
    fake.write_bytes(
        b"#!/usr/bin/env bash\nprintf '%s|%s\\n' \"${OMP_NUM_THREADS}\" \"${MKL_NUM_THREADS}\"\n"
    )
    fake.chmod(0o755)
    quoted_fake = shlex.quote(str(fake))
    environment = dict(os.environ)
    result = subprocess.run(
        ["bash", "-c", f"BEEID_H51_CPU_THREADS=7 OMP_NUM_THREADS=0 MKL_NUM_THREADS=bad scripts/run_h51.sh unused.yaml {quoted_fake}"],
        cwd=ROOT, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["7|7", "7|7"]

    result = subprocess.run(
        ["bash", "-c", f"BEEID_H51_CPU_THREADS=7 OMP_NUM_THREADS=5 MKL_NUM_THREADS=-1 scripts/run_h51.sh unused.yaml {quoted_fake}"],
        cwd=ROOT, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["5|5", "5|5"]

    result = subprocess.run(
        ["bash", "-c", f"BEEID_H51_CPU_THREADS=not-positive scripts/run_h51.sh unused.yaml {quoted_fake}"],
        cwd=ROOT, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    assert result.returncode == 2
    assert "must be a positive integer" in result.stderr
