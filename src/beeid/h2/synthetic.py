"""End-to-end H2 smoke test with an explicitly test-only encoder."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..config import load_config
from ..analysis.report import generate_report
from ..data.manifest import build_manifest
from ..extraction import extract_features
from ..protocol.retrieval import evaluate
from ..synthetic import _SyntheticTestEncoder, _make_video
from ..utils import atomic_write_json, sha256_file
from .contamination import run_memory_contamination
from .diagnostics import evaluate_h2
from .features import extract_h2_features
from .report import generate_h2_report
from .signals import build_observation_signals


def _base_payload(dataset: Path, output: Path, cache: Path, work: Path) -> dict[str, Any]:
    return {
        "paths": {
            "bee24_root": str(dataset),
            "output_root": str(output),
            "cache_root": str(cache),
            "dinov3_repo": str(work / "unused-dinov3"),
            "dinov3_weights": str(work / "unused-dino.pth"),
            "topictrack_repo": str(work / "unused-topic"),
            "topic_agw_weights": str(work / "unused-topic.pth"),
        },
        "runtime": {
            "device": "cpu",
            "batch_size": 4,
            "num_workers": 0,
            "amp": False,
            "amp_dtype": "bfloat16",
            "seed": 24,
            "cache_shard_size": 5,
        },
        "dataset": {
            "source_splits": ["train"],
            "bbox_origin": "one",
            "invalid_bbox_policy": "error",
            "duplicate_identity_policy": "error",
            "missing_seqinfo_policy": "error",
            "confidence_min": 0.0,
            "validation_fraction": 0.2,
            "max_videos": None,
            "max_frames_per_video": None,
            "video_ids": [],
            "deep_validate_images": True,
        },
        "protocol": {
            "deltas": [1],
            "crop_expansion": 0.2,
            "input_size": 224,
            "hard_negative_k": 5,
            "evaluation_split": "validation",
            "failure_top_k": 3,
        },
        "models": {
            "resnet_weights": "IMAGENET1K_V2",
            "dinov3_hub_model": "dinov3_vits16",
            "dinov3_feature_source": "patch_mean",
            "topic_config_relative": "fast-reid/configs/bee/AGW_S50.yml",
        },
    }


def h2_synthetic_smoke(output_directory: Path | None = None) -> dict[str, Any]:
    temporary = tempfile.TemporaryDirectory(prefix="beeid-h2-synthetic-")
    work = Path(temporary.name)
    dataset = work / "BEE24"
    for index in range(5):
        _make_video(dataset / "train" / f"synthetic-{index:02d}", index)
    h1_output = work / "h1-output"
    cache = work / "cache"
    h1_payload = _base_payload(dataset, h1_output, cache, work)
    h1_config_path = work / "h1.synthetic.local.yaml"
    h1_config_path.write_text(yaml.safe_dump(h1_payload, sort_keys=False), encoding="utf-8")
    h1_config = load_config(h1_config_path)
    build_manifest(h1_config)
    encoder = _SyntheticTestEncoder()
    extract_features(
        h1_config,
        "synthetic_test",
        extractor=encoder,
        signature_override={"name": "synthetic_test", "test_only": True, "version": 2},
    )
    evaluate(h1_config, ["synthetic_test"])
    generate_report(h1_config, ["synthetic_test"])

    resolved = yaml.safe_load(h1_config.resolved_split_path.read_text(encoding="utf-8"))
    development = list(resolved["validation_videos"])
    train_videos = list(resolved["train_videos"])
    project_train = [item for item in train_videos if item not in set(development)]
    split_path = work / "project_split.yaml"
    split_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "split_id": "synthetic-h2-smoke",
                "status": "FROZEN",
                "partitions": {
                    "project_train": {"video_ids": project_train},
                    "development_validation": {"video_ids": development},
                    "final_test": {"video_ids": []},
                },
                "provenance": {"manifest_sha256": sha256_file(h1_config.manifest_path)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    h2_output = output_directory.resolve(strict=False) if output_directory else work / "h2-output"
    h2_payload = _base_payload(dataset, h2_output, cache, work)
    h2_payload["paths"]["h1_output_root"] = str(h1_output)
    h2_payload["h2"] = {
        "project_split": str(split_path),
        "allow_subset": False,
        "primary_variant": "context_e020_raw_i224",
        "patch_size": 16,
        "density_radius_multipliers": [1.0, 2.0],
        "history_length": 2,
        "factor_bins": 4,
        "bootstrap_replicates": 50,
        "manual_annotations_csv": None,
        "variants": [
            {"name": "context_e000_raw_i224", "crop_expansion": 0.0, "input_size": 224, "pixel_view": "raw", "blur_radius": 0.0},
            {"name": "context_e020_raw_i224", "crop_expansion": 0.2, "input_size": 224, "pixel_view": "raw", "blur_radius": 0.0},
            {"name": "context_e050_raw_i224", "crop_expansion": 0.5, "input_size": 224, "pixel_view": "raw", "blur_radius": 0.0},
            {"name": "context_e020_foreground_i224", "crop_expansion": 0.2, "input_size": 224, "pixel_view": "bbox_foreground_only", "blur_radius": 0.0},
            {"name": "context_e020_only_i224", "crop_expansion": 0.2, "input_size": 224, "pixel_view": "context_only", "blur_radius": 0.0},
            {"name": "context_e020_blur_i224", "crop_expansion": 0.2, "input_size": 224, "pixel_view": "gaussian_blur", "blur_radius": 2.0},
            {"name": "context_e020_raw_i256", "crop_expansion": 0.2, "input_size": 256, "pixel_view": "raw", "blur_radius": 0.0},
        ],
        "contamination": {
            "trusted_history_length": 2,
            "ema_alpha": 0.2,
            "recovery_horizon": 2,
            "recovery_tolerance": 0.01,
            "low_quality_quantile": 0.25,
        },
    }
    h2_config_path = work / "h2.synthetic.local.yaml"
    h2_config_path.write_text(yaml.safe_dump(h2_payload, sort_keys=False), encoding="utf-8")
    config = load_config(h2_config_path)
    atomic_write_json(
        config.paths.output_root / "h2_run_state.json",
        {"h2_started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    signals = build_observation_signals(config)
    caches = extract_h2_features(
        config,
        "synthetic_test",
        extractor=encoder,
        signature_override={"name": "synthetic_test", "test_only": True, "version": 2},
    )
    resumed_caches = extract_h2_features(
        config,
        "synthetic_test",
        extractor=encoder,
        signature_override={"name": "synthetic_test", "test_only": True, "version": 2},
    )
    invalid_resume = {
        name: value["status"]
        for name, value in resumed_caches.items()
        if value["status"] not in {"reused_h1_cache", "reused_h2_cache"}
    }
    if invalid_resume:
        raise RuntimeError(f"H2 cache resume unexpectedly extracted variants: {invalid_resume}")
    evaluation = evaluate_h2(config, ["synthetic_test"])
    contamination = run_memory_contamination(config, ["synthetic_test"])
    metadata = generate_h2_report(config, ["synthetic_test"])
    required = [
        "h2_summary.csv",
        "h2_observation_signals.csv",
        "h2_query_diagnostics.csv",
        "h2_context_ablation.csv",
        "h2_factor_summary.csv",
        "h2_cluster_bootstrap.csv",
        "h2_paired_bootstrap.csv",
        "h2_memory_trajectories.csv",
        "h2_memory_metadata.json",
        "h2_diagnostic_cases.csv",
        "h2_run_metadata.json",
    ]
    missing = [name for name in required if not (h2_output / name).is_file()]
    result = {
        "status": "passed" if not missing else "failed",
        "test_only_encoder": True,
        "output_root": str(h2_output),
        "signal_rows": signals["row_count"],
        "query_rows": evaluation["query_row_count"],
        "memory_events": contamination["event_count"],
        "primary_cache_status": caches["context_e020_raw_i224"]["status"],
        "resumed_variant_count": len(resumed_caches),
        "metadata_status": metadata["status"],
        "missing_artifacts": missing,
    }
    temporary.cleanup()
    if missing:
        raise RuntimeError(f"H2 synthetic smoke missed artifacts: {missing}")
    return result
