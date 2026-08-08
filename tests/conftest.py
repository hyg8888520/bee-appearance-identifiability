from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from beeid.config import ExperimentConfig, load_config


def make_video(root: Path, split: str, name: str, rows: list[str] | None = None, frames: int = 4) -> Path:
    video = root / split / name
    (video / "img1").mkdir(parents=True)
    (video / "gt").mkdir()
    (video / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={name}\n"
        "imDir=img1\nframeRate=30\n"
        f"seqLength={frames}\n"
        "imWidth=100\nimHeight=80\nimExt=.jpg\n",
        encoding="utf-8",
    )
    for frame in range(1, frames + 1):
        Image.new("RGB", (100, 80), (frame * 10, 20, 30)).save(video / "img1" / f"{frame:06d}.jpg")
    default_rows = [
        f"{frame},1,11,21,20,10,1,1,0.9" for frame in range(1, frames + 1)
    ] + [
        f"{frame},2,61,41,20,10,1,1,0.8" for frame in range(1, frames + 1)
    ]
    (video / "gt" / "gt.txt").write_text("\n".join(rows or default_rows) + "\n", encoding="utf-8")
    return video


def make_config(tmp_path: Path, **overrides: Any) -> ExperimentConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset = tmp_path / "BEE24"
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    payload: dict[str, Any] = {
        "paths": {
            "bee24_root": str(dataset), "output_root": str(output), "cache_root": str(cache),
            "dinov3_repo": str(tmp_path / "dinov3"), "dinov3_weights": str(tmp_path / "dino.pth"),
            "topictrack_repo": str(tmp_path / "topic"), "topic_agw_weights": str(tmp_path / "topic.pth"),
        },
        "runtime": {
            "device": "cpu", "batch_size": 2, "num_workers": 0, "amp": False,
            "amp_dtype": "bfloat16", "seed": 24, "cache_shard_size": 3,
        },
        "dataset": {
            "source_splits": ["train"], "bbox_origin": "one", "invalid_bbox_policy": "error",
            "confidence_min": 0.0, "validation_fraction": 0.2, "max_videos": None,
            "max_frames_per_video": None, "video_ids": [], "deep_validate_images": True,
        },
        "protocol": {
            "deltas": [1], "crop_expansion": 0.2, "input_size": 224, "hard_negative_k": 5,
            "evaluation_split": "validation", "failure_top_k": 3,
        },
        "models": {
            "resnet_weights": "IMAGENET1K_V2", "dinov3_hub_model": "dinov3_vits16",
            "dinov3_feature_source": "patch_mean",
            "topic_config_relative": "fast-reid/configs/bee/AGW_S50.yml",
        },
    }
    for section, values in overrides.items():
        payload[section].update(values)
    path = tmp_path / "config.local.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_config(path)


@pytest.fixture
def config_factory():
    return make_config


@pytest.fixture
def video_factory():
    return make_video
