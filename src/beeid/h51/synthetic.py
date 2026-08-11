"""CPU-only scientific smoke for the full H5.1 diagnostic chain."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import load_config
from ..data.mot import Observation
from ..h3.core import H3Inputs
from ..h5.core import H5Inputs
from ..h5.model import BeeTrackQuery
from .core import H51Inputs
from .experiment import _run_loaded_diagnostic
from .protocol import validate_h51_protocol


def _observation(partition: str, video: str, frame: int, identity: int, x: float) -> Observation:
    split = "validation" if partition == "development_validation" else "train"
    return Observation(
        observation_id=f"{split}:{video}:{frame:06d}:{identity}", split=split,
        source_split="train", video_id=video, track_id=identity,
        identity=f"{video}:{identity}", frame=frame, frame_id=frame,
        image_path=f"train/{video}/img1/{frame:06d}.jpg",
        image_width=100, image_height=80, original_width=15.0, original_height=15.0,
        raw_x=x + 1, raw_y=21.0, raw_w=15.0, raw_h=15.0,
        bbox_x1=x, bbox_y1=20.0, bbox_x2=x + 15, bbox_y2=35.0,
        x1=int(x), y1=20, x2=int(x + 15), y2=35, crop_expansion=0.2,
        crop_clipped=False, center_x=x + 7.5, center_y=27.5, bbox_area=225.0,
        confidence=1.0, object_class=1, visibility=1.0, skip_reason="", extra_columns="[]",
    )


def h51_synthetic_smoke(output_directory: Path | None = None) -> dict[str, Any]:
    temporary = tempfile.TemporaryDirectory(prefix="beeid-h51-synthetic-")
    output = output_directory.resolve(strict=False) if output_directory else Path(temporary.name) / "output"
    root = Path(__file__).resolve().parents[3]
    base = load_config(root / "configs" / "h51_smoke.example.yaml")
    config = replace(
        base,
        paths=replace(
            base.paths, output_root=output, h5_output_root=output.parent / "test-only-h5",
            h3_output_root=output.parent / "test-only-h3",
        ),
        runtime=replace(base.runtime, device="cpu", amp=False, batch_size=4),
    )
    assert config.h51 is not None and config.h5 is not None
    protocol = validate_h51_protocol(
        config.h51.protocol_lock_path, config.h51.protocol_checksum_path
    )
    observations: list[Observation] = []
    partitions: dict[str, str] = {}
    for partition, video in (("project_train", "fit-toy"), ("development_validation", "dev-toy")):
        for frame in range(1, 9):
            for identity in (1, 2):
                x = 10.0 + frame if identity == 1 else 62.0 - frame
                item = _observation(partition, video, frame, identity, x)
                observations.append(item)
                partitions[item.observation_id] = partition
    embeddings = np.stack([
        np.asarray(
            [1.0, 0.45 * item.track_id, item.center_x / 100.0,
             item.center_y / 80.0, item.frame / 20.0, 0.2],
            dtype=np.float32,
        )
        for item in observations
    ])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    input_audit = {
        "status": "valid_test_only", "manifest_sha256": "synthetic",
        "protocol_sha256": protocol["source_h3_protocol_sha256"],
        "h5_protocol_sha256": protocol["source_h5_protocol_sha256"],
        "h51_protocol_sha256": protocol["protocol_sha256"],
        "project_split_sha256": protocol["project_split_sha256"],
        "read_partitions": {
            "project_train": "runtime_gradient_probe_only",
            "development_validation": "teacher_forced_offline_audit_causal_rollout_and_metrics",
        },
        "runtime_gradient_probe_partition": "project_train",
        "teacher_forced_offline_partition": "development_validation",
        "causal_rollout_partition": "development_validation",
        "metrics_partition": "development_validation", "test_only": True,
        "replay_diagnostic_schema_version": protocol["replay_diagnostic_schema_version"],
        "gradient_health_ready_rule": protocol["gradient_health_ready_rule"],
        "diagnostic_output_limits": {
            "max_teacher_forced_rows": config.h51.max_teacher_forced_rows,
            "teacher_forced_rows_scope": "bounded_subset_smoke_override",
            "candidate_sample_per_observation": config.h51.candidate_sample_per_observation,
        },
        "real_experiment_result": False, "final_test_read": False,
    }
    h3_inputs = H3Inputs(
        tuple(observations), partitions,
        {"project_train": ("fit-toy",), "development_validation": ("dev-toy",), "final_test": ("never-read",)},
        input_audit,
    )
    inputs = H51Inputs(
        H5Inputs(h3_inputs, input_audit), output.parent / "test-only-h5",
        {"status": "test_only"},
        {"test_only_encoder": {"sha256": "test-only", "fingerprint": "test-only", "read_only": True}},
        input_audit,
    )
    torch.manual_seed(config.runtime.seed)
    model = BeeTrackQuery(embeddings.shape[1], 16, 4, 0.0)
    result = _run_loaded_diagnostic(
        config, inputs, ["test_only_encoder"], test_only=True,
        supplied={"test_only_encoder": (model, embeddings)},
    )
    with (output / "h51_gradient_coverage.csv").open("r", encoding="utf-8", newline="") as handle:
        gradient = list(csv.DictReader(handle))
    missing = [row["parameter"] for row in gradient if row["grad_present"] == "False"]
    zero = [
        row["parameter"] for row in gradient
        if row["grad_present"] == "True" and row["finite"] == "True"
        and row["grad_nonzero"] == "False"
    ]
    expected = [
        "memory_attention.in_proj_weight", "memory_attention.in_proj_bias",
        "memory_attention.out_proj.weight", "memory_attention.out_proj.bias",
        "memory_norm.weight", "memory_norm.bias",
    ]
    path = json.loads((output / "h51_path_audit.json").read_text(encoding="utf-8"))
    decision = json.loads((output / "h51_decision.json").read_text(encoding="utf-8"))
    with (output / "h51_rollout_observation_diagnostics.csv").open("r", encoding="utf-8", newline="") as handle:
        rollout = list(csv.DictReader(handle))
    scientific_checks = {
        "exact_six_memory_parameters_missing": sorted(missing) == sorted(expected),
        "all_nonmemory_trainable_gradients_present_and_finite": all(
            row["grad_present"] == "True"
            and row["finite"] == "True"
            for row in gradient if row["parameter"] not in expected
        ),
        "finite_zero_gradient_detected_and_fused": (
            "pair_head.3.bias" in zero
            and path["per_model_gradient_summary"]["test_only_encoder"]["zero_gradient_parameters"] == zero
            and path["per_model_gradient_summary"]["test_only_encoder"]["method_ready"] is False
        ),
        "training_memory_path_not_called": path["training_path"]["memory_read_called"] is False,
        "rollout_memory_path_called": any(
            calls.get("memory_attention", 0) > 0
            for variant, calls in path["deployable_rollout_path"]["runtime_called_modules_by_variant"].items()
            if variant != "persistent_query_no_memory"
        ),
        "teacher_forced_boundary_explicit": path["boundary"]["teacher_forced_metrics_are_method_results"] is False,
        "causal_rows_without_gt_decision_input": bool(rollout) and all(
            row["gt_identity_used_for_inference_decision"] == "False" for row in rollout
        ),
        "known_defect_yields_stop_not_pipeline_failure": (
            decision["status"] == "STOP_H51_IMPLEMENTATION_NOT_READY"
            and decision["diagnostic_pipeline_passed"] is True
        ),
    }
    passed = all(scientific_checks.values())
    smoke = {
        **result,
        "status": "passed" if passed else "failed",
        "scientific_checks": scientific_checks,
        "diagnostic_pipeline_passed": passed,
        "method_ready": False,
        "metadata_status": "SERVER_VALIDATION_PENDING",
        "test_only": True, "test_only_encoder": True,
        "real_experiment_result": False, "final_test_read": False,
    }
    if not passed:
        raise RuntimeError(f"H5.1 synthetic scientific smoke failed: {smoke}")
    temporary.cleanup()
    return smoke
