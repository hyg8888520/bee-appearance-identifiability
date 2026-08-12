"""Extractor construction and reproducibility signatures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..references import DINOV3_COMMIT, TOPICTRACK_COMMIT, TORCHVISION_COMMIT
from ..utils import sha256_file
from .base import FeatureExtractor
from .dinov3 import DINOv3Extractor
from .resnet50 import ResNet50Extractor
from .topic_agw import TopicAGWExtractor

REAL_MODEL_NAMES = ("resnet50", "dinov3", "topic_agw")


def create_extractor(name: str, config: ExperimentConfig) -> FeatureExtractor:
    common = {
        "input_size": config.protocol.input_size,
        "device_name": config.runtime.device,
        "amp": config.runtime.amp,
        "amp_dtype": config.runtime.amp_dtype,
    }
    if name == "resnet50":
        return ResNet50Extractor(**common)
    if name == "dinov3":
        return DINOv3Extractor(
            config.paths.dinov3_repo,
            config.paths.dinov3_weights,
            feature_source=config.models.dinov3_feature_source,
            **common,
        )
    if name == "topic_agw":
        return TopicAGWExtractor(
            config.paths.topictrack_repo,
            config.paths.topic_agw_weights,
            config.models.topic_config_relative,
            **common,
        )
    raise ValueError(f"Unknown real extractor {name!r}; choose one of {REAL_MODEL_NAMES}")


def model_signature(name: str, config: ExperimentConfig, details: dict[str, Any] | None = None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": name,
        "input_size": config.protocol.input_size,
        "amp": config.runtime.amp,
        "amp_dtype": config.runtime.amp_dtype,
    }
    if name == "resnet50":
        base.update(
            {
                "weights": "ResNet50_Weights.IMAGENET1K_V2",
                "torchvision_commit": TORCHVISION_COMMIT,
            }
        )
    elif name == "dinov3":
        base.update(
            {
                "upstream_commit": DINOV3_COMMIT,
                "checkpoint_sha256": sha256_file(config.paths.dinov3_weights),
                "feature_source": config.models.dinov3_feature_source,
            }
        )
    elif name == "topic_agw":
        base.update(
            {
                "upstream_commit": TOPICTRACK_COMMIT,
                "checkpoint_sha256": sha256_file(config.paths.topic_agw_weights),
                "config_relative": config.models.topic_config_relative.as_posix(),
            }
        )
    else:
        raise ValueError(f"Unsupported real model signature: {name}")
    if details is not None:
        base["extractor_details"] = details
    return base
