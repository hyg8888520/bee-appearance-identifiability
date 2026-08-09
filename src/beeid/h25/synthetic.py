"""End-to-end H2.5 smoke with an explicitly test-only encoder."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from ..analysis.report import generate_report
from ..config import load_config
from ..data.manifest import build_manifest
from ..extraction import extract_features
from ..h2.contamination import run_memory_contamination
from ..h2.diagnostics import evaluate_h2
from ..h2.features import extract_h2_features
from ..h2.report import generate_h2_report
from ..h2.signals import build_observation_signals
from ..h2.synthetic import _base_payload
from ..protocol.retrieval import evaluate
from ..synthetic import _SyntheticTestEncoder
from ..utils import sha256_file
from .experiment import run_h25_experiment
from .report import generate_h25_report


def _make_video(directory: Path, video_index: int, frames: int = 16) -> None:
    (directory / "img1").mkdir(parents=True)
    (directory / "gt").mkdir()
    (directory / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={directory.name}\nimDir=img1\nframeRate=30\nseqLength={frames}\n"
        "imWidth=128\nimHeight=80\nimExt=.jpg\n",
        encoding="utf-8",
    )
    annotations: list[str] = []
    for frame in range(1, frames + 1):
        image = Image.new("RGB", (128, 80), (15, 15, 18))
        draw = ImageDraw.Draw(image)
        # Deterministic scale, orientation-texture, sharpness, and crowding changes.
        size = 18 + (8 if frame in {7, 12} else 0)
        left_a = 10 + frame * 2
        left_b = 90 - frame * 2
        draw.rectangle((left_a, 12, left_a + size, 12 + size), fill=(220, 40 + video_index, 45))
        draw.line((left_a, 12, left_a + size, 12 + size), fill="white", width=2)
        draw.rectangle((left_b, 40, left_b + 20, 62), fill=(35, 90, 220 - video_index))
        draw.line((left_b + 20, 40, left_b, 62), fill="white", width=2)
        draw.rectangle((52, 26 + (frame % 3), 70, 46 + (frame % 3)), fill=(65, 205, 80))
        image.save(directory / "img1" / f"{frame:06d}.jpg", quality=100, subsampling=0)
        annotations.extend(
            [
                f"{frame},1,{left_a + 1},13,{size + 1},{size + 1},1,1,1",
                f"{frame},2,{left_b + 1},41,21,23,1,1,1",
                f"{frame},3,53,{27 + (frame % 3)},19,21,1,1,1",
            ]
        )
    (directory / "gt" / "gt.txt").write_text("\n".join(annotations) + "\n", encoding="utf-8")


def h25_synthetic_smoke(output_directory: Path | None = None) -> dict[str, Any]:
    temporary = tempfile.TemporaryDirectory(prefix="beeid-h25-synthetic-")
    work = Path(temporary.name)
    dataset = work / "BEE24"
    for index in range(5):
        _make_video(dataset / "train" / f"synthetic-{index:02d}", index)
    cache = work / "cache"
    h1_output = work / "h1-output"
    h1_payload = _base_payload(dataset, h1_output, cache, work)
    h1_path = work / "h1.synthetic.local.yaml"
    h1_path.write_text(yaml.safe_dump(h1_payload, sort_keys=False), encoding="utf-8")
    h1_config = load_config(h1_path)
    build_manifest(h1_config)
    encoder = _SyntheticTestEncoder()
    signature = {"name": "synthetic_test", "test_only": True, "version": 3}
    extract_features(h1_config, "synthetic_test", extractor=encoder, signature_override=signature)
    evaluate(h1_config, ["synthetic_test"])
    generate_report(h1_config, ["synthetic_test"])

    resolved = yaml.safe_load(h1_config.resolved_split_path.read_text(encoding="utf-8"))
    development = list(resolved["validation_videos"])
    train_videos = list(resolved["train_videos"])
    split_path = work / "project_split.yaml"
    split_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "split_id": "synthetic-h25-smoke",
                "status": "FROZEN",
                "partitions": {
                    "project_train": {"video_ids": [item for item in train_videos if item not in set(development)]},
                    "development_validation": {"video_ids": development},
                    "final_test": {"video_ids": []},
                },
                "provenance": {"manifest_sha256": sha256_file(h1_config.manifest_path)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    h2_output = work / "h2-output"
    h2_payload = _base_payload(dataset, h2_output, cache, work)
    h2_payload["paths"]["h1_output_root"] = str(h1_output)
    h2_payload["h2"] = {
        "project_split": str(split_path), "allow_subset": False,
        "primary_variant": "context_e020_raw_i224", "patch_size": 16,
        "density_radius_multipliers": [1.0, 2.0], "history_length": 5,
        "factor_bins": 4, "bootstrap_replicates": 20, "manual_annotations_csv": None,
        "variants": [
            {"name": "context_e020_raw_i224", "crop_expansion": 0.2, "input_size": 224, "pixel_view": "raw", "blur_radius": 0.0},
            {"name": "context_e020_blur_i224", "crop_expansion": 0.2, "input_size": 224, "pixel_view": "gaussian_blur", "blur_radius": 2.0},
        ],
        "contamination": {
            "trusted_history_length": 3, "ema_alpha": 0.2, "recovery_horizon": 5,
            "recovery_tolerance": 0.01, "low_quality_quantile": 0.25,
        },
    }
    h2_path = work / "h2.synthetic.local.yaml"
    h2_path.write_text(yaml.safe_dump(h2_payload, sort_keys=False), encoding="utf-8")
    h2_config = load_config(h2_path)
    build_observation_signals(h2_config)
    extract_h2_features(h2_config, "synthetic_test", extractor=encoder, signature_override=signature)
    evaluate_h2(h2_config, ["synthetic_test"])
    run_memory_contamination(h2_config, ["synthetic_test"])
    generate_h2_report(h2_config, ["synthetic_test"])

    output = output_directory.resolve(strict=False) if output_directory else work / "h25-output"
    repository = Path(__file__).resolve().parents[3]
    h25_payload = _base_payload(dataset, output, cache, work)
    h25_payload["paths"].update(
        {"h1_output_root": str(h1_output), "h2_output_root": str(h2_output)}
    )
    h25_payload["h2"] = h2_payload["h2"]
    h25_payload["h25"] = {
        "protocol_lock": str(repository / "configs" / "h25_protocol.lock.yaml"),
        "h3_protocol_lock": str(repository / "configs" / "h3_protocol.lock.yaml"),
        "allow_subset": False,
    }
    h25_path = work / "h25.synthetic.local.yaml"
    h25_path.write_text(yaml.safe_dump(h25_payload, sort_keys=False), encoding="utf-8")
    config = load_config(h25_path)
    experiment = run_h25_experiment(config, ["synthetic_test"])
    metadata = generate_h25_report(config, ["synthetic_test"], test_only_encoder=True)
    required = [
        "h25_event_thresholds.json", "h25_events.csv", "h25_strategy_trajectories.csv",
        "h25_strategy_summary.csv", "h25_paired_summary.csv", "h25_cluster_bootstrap.csv",
        "h25_run_metadata.json", "h25_resolved_config.yaml", "h25_logs", "h25_figures",
    ]
    missing = [name for name in required if not (output / name).exists()]
    result = {
        "status": "passed" if not missing else "failed",
        "test_only_encoder": True,
        "output_root": str(output),
        "event_count": experiment["event_count"],
        "trajectory_row_count": experiment["trajectory_row_count"],
        "metadata_status": metadata["status"],
        "missing_artifacts": missing,
        "final_test_read": False,
    }
    temporary.cleanup()
    if missing or experiment["event_count"] == 0:
        raise RuntimeError(f"H2.5 synthetic smoke failed: missing={missing}, events={experiment['event_count']}")
    return result
