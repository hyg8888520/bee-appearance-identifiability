from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def write_runtime(tmp_path: Path, **overrides: Any) -> tuple[Path, dict[str, Any]]:
    dataset_root = tmp_path / "dataset"
    image = dataset_root / "BEE24-01" / "img1" / "000001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-needed-for-validation")
    annotation = dataset_root / "train.json"
    annotation.write_text(
        json.dumps({"images": [{"id": 1, "file_name": "BEE24-01/img1/000001.jpg"}]}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    output_root = tmp_path / "experiment"
    config: dict[str, Any] = {
        "experiment": {"name": "test", "seed": 24, "device": "cpu"},
        "dataset": {"root": str(dataset_root), "annotation": str(annotation), "max_images": None},
        "models": {
            "dinov3": {
                "checkpoint": str(checkpoint),
                "model_name": "vit_small_patch16_dinov3.lvd1689m",
                "image_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "interpolation": "bicubic",
                "global_pool": "token",
                "batch_size": 2,
                "num_workers": 0,
                "use_amp": False,
                "amp_dtype": "bfloat16",
                "normalize_features": True,
            }
        },
        "h1": {
            "query_image_ids": [],
            "query_stride": 1,
            "top_k": 1,
            "reuse_feature_cache": True,
        },
        "output": {
            "root": str(output_root),
            "feature_cache": "cache/features.pt",
            "query_results": "queries/results.json",
            "figures": "figures",
            "failure_cases": "failures",
            "metadata": "metadata/run.json",
        },
    }
    for dotted_key, value in overrides.items():
        target = config
        parts = dotted_key.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    config_path = tmp_path / "runtime.local.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, config

