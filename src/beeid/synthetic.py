"""End-to-end synthetic smoke using an explicitly test-only encoder."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .analysis.report import generate_report
from .config import load_config
from .data.manifest import build_manifest
from .extraction import extract_features
from .models.base import FeatureExtractor, validate_embeddings
from .protocol.retrieval import evaluate


class _SyntheticTestEncoder(FeatureExtractor):
    """Deterministic smoke-only pixels encoder; never exposed by the real extract CLI."""

    name = "synthetic_test"

    @property
    def details(self) -> dict[str, Any]:
        return {"test_only": True, "algorithm": "RGB mean and standard deviation"}

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        rows = []
        for image in images:
            values = np.asarray(image, dtype=np.float32) / 255.0
            rows.append(np.concatenate([values.mean(axis=(0, 1)), values.std(axis=(0, 1)), [1.0]]))
        return validate_embeddings(np.asarray(rows, dtype=np.float32), len(images))


def _make_video(directory: Path, video_index: int) -> None:
    (directory / "img1").mkdir(parents=True)
    (directory / "gt").mkdir()
    (directory / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={directory.name}\n"
        "imDir=img1\nframeRate=30\nseqLength=4\nimWidth=96\nimHeight=64\nimExt=.jpg\n",
        encoding="utf-8",
    )
    annotations: list[str] = []
    for frame in range(1, 5):
        image = Image.new("RGB", (96, 64), (20, 20, 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8 + frame, 10, 31 + frame, 35), fill=(220, 40 + video_index, 40))
        draw.rectangle((58 - frame, 25, 81 - frame, 50), fill=(40, 80, 220 - video_index))
        image.save(directory / "img1" / f"{frame:06d}.jpg", quality=100, subsampling=0)
        annotations.append(f"{frame},1,{9 + frame},11,24,26,1,1,1")
        annotations.append(f"{frame},2,{59 - frame},26,24,26,1,1,1")
    (directory / "gt" / "gt.txt").write_text("\n".join(annotations) + "\n", encoding="utf-8")


def synthetic_smoke(output_directory: Path | None = None) -> dict[str, Any]:
    temporary = tempfile.TemporaryDirectory(prefix="beeid-synthetic-")
    work = Path(temporary.name)
    dataset = work / "BEE24"
    for index in range(5):
        _make_video(dataset / "train" / f"synthetic-{index:02d}", index)
    output = output_directory.resolve(strict=False) if output_directory else work / "output"
    cache = work / "cache"
    config_path = work / "h1.synthetic.local.yaml"
    payload = {
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
            "device": "cpu", "batch_size": 4, "num_workers": 0, "amp": False,
            "amp_dtype": "bfloat16", "seed": 24, "cache_shard_size": 5,
        },
        "dataset": {
            "source_splits": ["train"], "bbox_origin": "one", "invalid_bbox_policy": "error",
            "confidence_min": 0.0, "validation_fraction": 0.2, "max_videos": None,
            "max_frames_per_video": None, "video_ids": [], "deep_validate_images": True,
        },
        "protocol": {
            "deltas": [1], "crop_expansion": 0.2, "input_size": 224,
            "hard_negative_k": 5, "evaluation_split": "validation", "failure_top_k": 3,
        },
        "models": {
            "resnet_weights": "IMAGENET1K_V2", "dinov3_hub_model": "dinov3_vits16",
            "dinov3_feature_source": "patch_mean",
            "topic_config_relative": "fast-reid/configs/bee/AGW_S50.yml",
        },
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)
    build_manifest(config)
    extract_features(
        config,
        "synthetic_test",
        extractor=_SyntheticTestEncoder(),
        signature_override={"name": "synthetic_test", "test_only": True, "version": 1},
    )
    rows = evaluate(config, ["synthetic_test"])
    metadata = generate_report(config, ["synthetic_test"])
    result = {
        "status": "passed",
        "test_only_encoder": True,
        "output_root": str(output),
        "query_rows": len(rows),
        "metadata_status": metadata["status"],
    }
    temporary.cleanup()
    return result
