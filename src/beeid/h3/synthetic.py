"""End-to-end CPU H3 smoke with an explicitly test-only encoder."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ..config import load_config
from ..data.manifest import build_manifest
from ..h2.synthetic import _base_payload
from ..h25.synthetic import _make_video
from ..synthetic import _SyntheticTestEncoder
from ..utils import atomic_write_json, atomic_write_text, sha256_file
from .experiment import run_h3_tracking
from .features import extract_h3_features
from .report import generate_h3_report
from .signals import build_h3_signals
from .thresholds import fit_h3_thresholds


def h3_synthetic_smoke(output: Path | None = None) -> dict[str, Any]:
    temporary = tempfile.TemporaryDirectory(prefix="beeid-h3-synthetic-")
    work = Path(temporary.name)
    dataset = work / "BEE24"
    for index in range(3):
        _make_video(dataset / "train" / f"synthetic-h3-{index:02d}", index)

    cache = work / "cache"
    h1_output = work / "h1-output"
    h1_payload = _base_payload(dataset, h1_output, cache, work)
    h1_path = work / "h1.synthetic.local.yaml"
    h1_path.write_text(yaml.safe_dump(h1_payload, sort_keys=False), encoding="utf-8")
    h1_config = load_config(h1_path)
    build_manifest(h1_config)
    atomic_write_json(
        h1_output / "run_metadata.json",
        {
            "status": "SERVER_VALIDATION_PENDING",
            "test_only_encoder": True,
            "manifest_sha256": sha256_file(h1_config.manifest_path),
            "final_test_read": False,
        },
    )

    resolved = yaml.safe_load(h1_config.resolved_split_path.read_text(encoding="utf-8"))
    development = list(resolved["validation_videos"])
    project_train = [
        item for item in resolved["train_videos"] if item not in set(development)
    ]
    config_directory = work / "configs"
    (config_directory / "splits").mkdir(parents=True)
    split_path = config_directory / "splits" / "project_split.yaml"
    split_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "split_id": "synthetic-h3-smoke",
                "status": "FROZEN",
                "partitions": {
                    "project_train": {"video_ids": project_train},
                    "development_validation": {"video_ids": development},
                    "final_test": {"video_ids": ["synthetic-h3-final-locked"]},
                },
                "provenance": {
                    "manifest_sha256": sha256_file(h1_config.manifest_path)
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    repository = Path(__file__).resolve().parents[3]
    protocol_path = config_directory / "h3_protocol.lock.yaml"
    atomic_write_text(
        protocol_path,
        (repository / "configs" / "h3_protocol.lock.yaml").read_text(encoding="utf-8"),
    )
    checksum_path = config_directory / "h3_protocol.lock.sha256"
    atomic_write_text(
        checksum_path, f"{sha256_file(protocol_path)}  {protocol_path.name}\n"
    )

    output_root = (
        output.resolve(strict=False) if output is not None else work / "h3-output"
    )
    payload = _base_payload(dataset, output_root, cache, work)
    payload["paths"]["h1_output_root"] = str(h1_output)
    payload["dataset"].update(
        {
            "source_splits": ["train"],
            "max_videos": None,
            "max_frames_per_video": None,
            "video_ids": [],
        }
    )
    payload["h3"] = {
        "protocol_lock": str(protocol_path),
        "protocol_checksum": str(checksum_path),
        "project_split": str(split_path),
        "allow_subset": False,
        "stage": "gt_detection_boxes",
        "fixed_detections_root": None,
        "history_length": 5,
        "quantile_knots": 101,
        "min_reliability": 0.05,
        "memory_alpha": 0.2,
        "update_gate": 0.25,
        "appearance_weight": 0.7,
        "motion_weight": 0.3,
        "max_normalized_distance": 4.0,
        "min_assignment_score": 0.1,
        "max_age": 10,
        "bootstrap_replicates": 20,
    }
    h3_path = work / "h3.synthetic.local.yaml"
    h3_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_config(h3_path)
    build_h3_signals(config)
    encoder = _SyntheticTestEncoder()
    model_name = "test_only_encoder"
    extract_h3_features(
        config,
        model_name,
        extractor=encoder,
        signature_override={
            "name": model_name,
            "test_only": True,
            "algorithm": encoder.details["algorithm"],
            "version": 1,
        },
    )
    fit_h3_thresholds(config, [model_name])
    tracking = run_h3_tracking(config, [model_name])
    report = generate_h3_report(config, [model_name])
    metadata_path = output_root / "h3_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": "SERVER_VALIDATION_PENDING",
            "test_only_encoder": True,
            "real_experiment_result": False,
        }
    )
    atomic_write_json(metadata_path, metadata)
    required = (
        "h3_observation_signals.csv",
        "h3_signal_metadata.json",
        "h3_cache_locations.json",
        "h3_thresholds.json",
        "h3_assignments.csv",
        "h3_per_video_metrics.csv",
        "h3_summary.csv",
        "h3_paired_video_metrics.csv",
        "h3_video_cluster_bootstrap.csv",
        "h3_tracking_metadata.json",
        "h3_run_metadata.json",
        "h3_resolved_config.yaml",
        "h3_logs",
        "h3_mot_results",
    )
    missing = [name for name in required if not (output_root / name).exists()]
    result = {
        "status": "passed" if not missing else "failed",
        "test_only_encoder": True,
        "output_root": str(output_root),
        "assignment_rows": tracking["assignment_row_count"],
        "summary_rows": tracking["summary_row_count"],
        "variants": tracking["variants"],
        "stage": tracking["stage"],
        "metadata_status": metadata["status"],
        "report_status_before_test_marker": report["status"],
        "fixed_detector_boxes": "SERVER_VALIDATION_PENDING",
        "missing_artifacts": missing,
        "final_test_read": False,
    }
    temporary.cleanup()
    if missing:
        raise RuntimeError(f"H3 synthetic smoke is missing artifacts: {missing}")
    return result
