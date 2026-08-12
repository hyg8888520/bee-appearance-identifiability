"""End-to-end H4 smoke with handcrafted test-only feature-cache evidence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from ..cache import FeatureCache
from ..config import load_config
from ..data.manifest import build_manifest, load_manifest
from ..h2.synthetic import _base_payload
from ..utils import atomic_write_json, atomic_write_text, sha256_file
from .experiment import run_h4_recoverability_audit, run_h4_tracking
from .report import generate_h4_report


def _make_crossing_video(directory: Path, index: int, frames: int = 12) -> None:
    (directory / "img1").mkdir(parents=True)
    (directory / "gt").mkdir()
    atomic_write_text(
        directory / "seqinfo.ini",
        "[Sequence]\n"
        f"name={directory.name}\nimDir=img1\nframeRate=30\nseqLength={frames}\n"
        "imWidth=120\nimHeight=72\nimExt=.jpg\n",
    )
    annotations: list[str] = []
    for frame in range(1, frames + 1):
        image = Image.new("RGB", (120, 72), (18, 18, 22))
        draw = ImageDraw.Draw(image)
        left_a = 20 + (index % 2)
        left_b = 78 - (index % 2)
        color_a = (225, 45, 45) if frame != 5 else (45, 80, 225)
        color_b = (45, 80, 225) if frame != 5 else (225, 45, 45)
        draw.rectangle((left_a, 15, left_a + 20, 39), fill=color_a)
        draw.rectangle((left_b, 32, left_b + 20, 56), fill=color_b)
        image.save(directory / "img1" / f"{frame:06d}.jpg", quality=100, subsampling=0)
        annotations.extend(
            [
                f"{frame},1,{left_a + 1},16,21,25,1,1,1",
                f"{frame},2,{left_b + 1},33,21,25,1,1,1",
            ]
        )
    atomic_write_text(directory / "gt" / "gt.txt", "\n".join(annotations) + "\n")


def _smoke_embedding(track_id: int, frame: int) -> np.ndarray:
    first = np.asarray([1.0, 0.0, 0.2, 0.0], dtype=np.float32)
    second = np.asarray([0.0, 1.0, 0.0, 0.2], dtype=np.float32)
    value = first if track_id == 1 else second
    if frame == 5:
        value = second if track_id == 1 else first
    return value / np.linalg.norm(value)


def h4_synthetic_smoke(output_directory: Path | None = None) -> dict[str, Any]:
    temporary = tempfile.TemporaryDirectory(prefix="beeid-h4-synthetic-")
    work = Path(temporary.name)
    dataset = work / "BEE24"
    for index in range(5):
        _make_crossing_video(dataset / "train" / f"synthetic-h4-{index:02d}", index)
    h1_output = work / "h1-output"
    cache_root = work / "cache"
    h1_payload = _base_payload(dataset, h1_output, cache_root, work)
    h1_path = work / "h1.synthetic.local.yaml"
    atomic_write_text(h1_path, yaml.safe_dump(h1_payload, sort_keys=False))
    h1_config = load_config(h1_path)
    build_manifest(h1_config)
    atomic_write_json(
        h1_output / "run_metadata.json",
        {"status": "SERVER_VALIDATION_PENDING", "test_only_encoder": True, "final_test_read": False},
    )

    resolved = yaml.safe_load(h1_config.resolved_split_path.read_text(encoding="utf-8"))
    development = list(resolved["validation_videos"])
    project_train = [
        item for item in resolved["train_videos"] if item not in set(development)
    ]
    config_root = work / "configs"
    (config_root / "splits").mkdir(parents=True)
    split_path = config_root / "splits" / "project_split.yaml"
    atomic_write_text(
        split_path,
        yaml.safe_dump(
            {
                "schema_version": 1,
                "split_id": "synthetic-h4-smoke",
                "status": "FROZEN",
                "partitions": {
                    "project_train": {"video_ids": project_train},
                    "development_validation": {"video_ids": development},
                    "final_test": {"video_ids": ["synthetic-h4-final-locked"]},
                },
                "provenance": {"manifest_sha256": sha256_file(h1_config.manifest_path)},
            },
            sort_keys=False,
        ),
    )
    repository = Path(__file__).resolve().parents[3]
    h3_protocol = config_root / "h3_protocol.lock.yaml"
    atomic_write_text(
        h3_protocol,
        (repository / "configs" / "h3_protocol.lock.yaml").read_text(encoding="utf-8"),
    )
    h3_checksum = config_root / "h3_protocol.lock.sha256"
    atomic_write_text(h3_checksum, f"{sha256_file(h3_protocol)}  {h3_protocol.name}\n")
    h4_protocol_document = yaml.safe_load(
        (repository / "configs" / "h4_protocol.lock.yaml").read_text(encoding="utf-8")
    )
    h4_protocol_document["stop_go"]["recoverability_gate"].update(
        {
            "min_joint_recoverable_fraction": 0.0,
            "min_events_per_model": 1,
            "min_videos_with_events": 1,
        }
    )
    h4_protocol_document["stop_go"]["method_gate"]["min_nonharmed_videos"] = 1
    h4_protocol_document["hypothesis_tracker"]["ambiguity_margin"] = 1.0
    h4_protocol = config_root / "h4_protocol.lock.yaml"
    atomic_write_text(
        h4_protocol, yaml.safe_dump(h4_protocol_document, sort_keys=False)
    )
    h4_checksum = config_root / "h4_protocol.lock.sha256"
    atomic_write_text(h4_checksum, f"{sha256_file(h4_protocol)}  {h4_protocol.name}\n")

    h3_output = work / "h3-output"
    h3_output.mkdir()
    selected = [
        item
        for item in load_manifest(h1_config.manifest_path, valid_only=True)
        if item.video_id in set(development)
    ]
    selected.sort(key=lambda item: (item.video_id, item.frame, item.center_x, item.observation_id))
    model_name = "test_only_encoder"
    signature = {
        "format_version": 1,
        "experiment": "H3_RAM_Bee",
        "implementation": "beeid.h3.features:v1",
        "manifest_sha256": sha256_file(h1_config.manifest_path),
        "protocol_sha256": sha256_file(h3_protocol),
        "project_split_sha256": sha256_file(split_path),
        "model": {"name": model_name, "test_only": True, "algorithm": "handcrafted crossing fixture"},
        "crop_expansion": 0.2,
        "input_size": 224,
        "amp": False,
        "amp_dtype": "bfloat16",
        "final_test_read": False,
    }
    cache = FeatureCache(cache_root, f"h3__{model_name}", signature)
    cache.initialize()
    shard_size = 5
    for shard_index, start in enumerate(range(0, len(selected), shard_size)):
        shard = selected[start : start + shard_size]
        values = np.stack([_smoke_embedding(item.track_id, item.frame) for item in shard])
        cache.write_shard(shard_index, [item.observation_id for item in shard], values)
    atomic_write_json(h3_output / "h3_cache_locations.json", {model_name: str(cache.directory)})
    input_audit = {
        "manifest_sha256": sha256_file(h1_config.manifest_path),
        "project_split_sha256": sha256_file(split_path),
        "protocol_sha256": sha256_file(h3_protocol),
        "development_videos": development,
        "final_test_read": False,
    }
    atomic_write_json(
        h3_output / "h3_run_metadata.json",
        {"status": "completed_development_gt_boxes", "models": [model_name], "input_audit": input_audit, "test_only_encoder": True, "final_test_read": False},
    )
    atomic_write_json(
        h3_output / "h3_tracking_metadata.json",
        {"status": "completed", "models": [model_name], "input_audit": input_audit, "test_only_encoder": True, "final_test_read": False},
    )
    atomic_write_json(
        h3_output / "h3_thresholds.json",
        {"status": "completed", "test_only_encoder": True, "final_test_read": False},
    )
    atomic_write_text(h3_output / "h3_observation_signals.csv", "observation_id,final_test_read\n")

    output = output_directory.resolve(strict=False) if output_directory else work / "h4-output"
    payload = _base_payload(dataset, output, cache_root, work)
    payload["paths"].update(
        {"h1_output_root": str(h1_output), "h3_output_root": str(h3_output)}
    )
    payload["h4"] = {
        "protocol_lock": str(h4_protocol),
        "protocol_checksum": str(h4_checksum),
        "project_split": str(split_path),
        "allow_subset": False,
        "horizons": [1, 3, 5, 10],
        "go_horizon": 10,
        "primary_horizon": 5,
        "oracle_history_length": 5,
        "oracle_unique_margin": 0.0,
        "min_recoverable_fraction": 0.0,
        "min_events_per_model": 1,
        "min_videos_with_events": 1,
        "beam_width": 6,
        "max_component_size": 4,
        "ambiguity_margin": 1.0,
        "memory_alpha": 0.2,
        "appearance_weight": 0.7,
        "motion_weight": 0.3,
        "max_normalized_distance": 4.0,
        "min_assignment_score": 0.1,
        "unmatched_penalty": 0.25,
        "max_age": 25,
        "noninferiority_tolerance": 0.002,
        "min_nonharmed_videos": 1,
        "bootstrap_replicates": 20,
    }
    config_path = work / "h4.synthetic.local.yaml"
    atomic_write_text(config_path, yaml.safe_dump(payload, sort_keys=False))
    config = load_config(config_path)
    audit = run_h4_recoverability_audit(config, [model_name])
    if audit["decision"]["gate_passed"] is not True:
        raise RuntimeError("H4 synthetic fixture did not exercise the method gate")
    resumed_audit = run_h4_recoverability_audit(config, [model_name])
    tracking = run_h4_tracking(config, [model_name])
    resumed_tracking = run_h4_tracking(config, [model_name])
    report = generate_h4_report(config, [model_name])
    metadata_path = output / "h4_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {"status": "SERVER_VALIDATION_PENDING", "test_only_encoder": True, "real_experiment_result": False}
    )
    atomic_write_json(metadata_path, metadata)
    required = (
        "h4_recoverability_events.csv", "h4_horizon_summary.csv",
        "h4_protocol_decision.json", "h4_assignments.csv", "h4_conflict_events.csv",
        "h4_per_video_metrics.csv", "h4_summary.csv", "h4_paired_video_metrics.csv",
        "h4_video_cluster_bootstrap.csv", "h4_failure_cases.csv",
        "h4_runtime_profile.csv", "h4_method_decision.json", "h4_run_metadata.json",
        "h4_resolved_config.yaml", "h4_logs", "h4_figures", "h4_mot_results",
    )
    missing = [name for name in required if not (output / name).exists()]
    result = {
        "status": "passed" if not missing else "failed",
        "test_only_encoder": True,
        "output_root": str(output),
        "recoverability_switch_events": audit["switch_event_count"],
        "assignment_rows": tracking["assignment_row_count"],
        "conflict_events": tracking["conflict_event_count"],
        "resumed_audit_jobs": resumed_audit["reused_job_count"],
        "resumed_tracking_jobs": resumed_tracking["reused_job_count"],
        "report_status_before_test_marker": report["status"],
        "metadata_status": metadata["status"],
        "real_experiment_result": False,
        "missing_artifacts": missing,
        "final_test_read": False,
    }
    temporary.cleanup()
    if (
        missing
        or audit["switch_event_count"] == 0
        or tracking["conflict_event_count"] == 0
        or resumed_audit["reused_job_count"] != resumed_audit["resumable_job_count"]
        or resumed_tracking["reused_job_count"] != resumed_tracking["resumable_job_count"]
    ):
        raise RuntimeError(
            "H4 synthetic smoke failed: "
            f"missing={missing}, switches={audit['switch_event_count']}, "
            f"conflicts={tracking['conflict_event_count']}"
        )
    return result
