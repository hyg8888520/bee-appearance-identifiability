"""Minimal modern-PyTorch compatibility layer for TOPICTrack's BEE AGW model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ..references import TOPICTRACK_COMMIT
from ..status import BLOCKED_TOPIC_AGW_COMPATIBILITY
from ..utils import require_git_head
from .base import FeatureExtractor, torch_autocast, validate_embeddings


class TopicCompatibilityError(RuntimeError):
    """A strict AGW load failed; other benchmark models may continue."""


def _checkpoint_state(torch: Any, checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state: Any = checkpoint.get("model", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state, Mapping) or not state:
        raise TopicCompatibilityError(f"{BLOCKED_TOPIC_AGW_COMPATIBILITY}: checkpoint has no model state")
    result = dict(state)
    if all(key.startswith("module.") for key in result):
        result = {key.removeprefix("module."): value for key, value in result.items()}
    return result


def _infer_num_classes(state: Mapping[str, Any]) -> int:
    matches = [(key, value) for key, value in state.items() if key == "heads.weight" or key.endswith(".heads.weight")]
    if len(matches) != 1 or getattr(matches[0][1], "ndim", 0) != 2:
        raise TopicCompatibilityError(
            f"{BLOCKED_TOPIC_AGW_COMPATIBILITY}: could not uniquely infer classes from heads.weight"
        )
    return int(matches[0][1].shape[0])


def strict_state_report(model: Any, state: Mapping[str, Any]) -> dict[str, list[str]]:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatch = sorted(
        f"{key}: checkpoint={tuple(state[key].shape)} model={tuple(expected[key].shape)}"
        for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    return {"missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatch": mismatch}


class TopicAGWExtractor(FeatureExtractor):
    name = "topic_agw"

    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        config_relative: Path,
        input_size: int,
        device_name: str,
        amp: bool,
        amp_dtype: str,
        *,
        model: Any | None = None,
        transform: Any | None = None,
        verify_commit: bool = True,
        state_override: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import torch
            from torchvision.transforms import v2
        except ImportError as error:
            raise TopicCompatibilityError(
                f"{BLOCKED_TOPIC_AGW_COMPATIBILITY}: torch and torchvision are required"
            ) from error
        if verify_commit:
            require_git_head(repo, TOPICTRACK_COMMIT, "TOPICTrack")
        if model is None and not checkpoint.is_file():
            raise TopicCompatibilityError(
                f"{BLOCKED_TOPIC_AGW_COMPATIBILITY}: checkpoint does not exist: {checkpoint}"
            )
        self.torch = torch
        self.device = torch.device(device_name)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.input_size = input_size
        state = dict(state_override) if state_override is not None else _checkpoint_state(torch, checkpoint)
        classes = _infer_num_classes(state)
        if model is None:
            fast_reid = repo / "fast-reid"
            config_path = repo / config_relative
            if not fast_reid.is_dir() or not config_path.is_file():
                raise TopicCompatibilityError(
                    f"{BLOCKED_TOPIC_AGW_COMPATIBILITY}: TOPIC FastReID/config path is missing"
                )
            sys.path.insert(0, str(fast_reid))
            try:
                from fastreid.config import get_cfg
                from fastreid.modeling import build_model

                cfg = get_cfg()
                cfg.merge_from_file(str(config_path))
                cfg.defrost()
                cfg.MODEL.DEVICE = device_name
                cfg.MODEL.WEIGHTS = ""
                cfg.MODEL.BACKBONE.PRETRAIN = False
                cfg.MODEL.HEADS.NUM_CLASSES = classes
                cfg.INPUT.SIZE_TEST = [input_size, input_size]
                cfg.freeze()
                model = build_model(cfg)
            except Exception as error:
                raise TopicCompatibilityError(
                    f"{BLOCKED_TOPIC_AGW_COMPATIBILITY}: FastReID construction failed: {error}"
                ) from error
            finally:
                try:
                    sys.path.remove(str(fast_reid))
                except ValueError:
                    pass
        self.load_report = strict_state_report(model, state)
        if any(self.load_report.values()):
            raise TopicCompatibilityError(
                f"{BLOCKED_TOPIC_AGW_COMPATIBILITY}: strict checkpoint comparison failed: {self.load_report}"
            )
        try:
            model.load_state_dict(state, strict=True)
        except Exception as error:
            raise TopicCompatibilityError(
                f"{BLOCKED_TOPIC_AGW_COMPATIBILITY}: strict load_state_dict failed: {error}"
            ) from error
        self.model = model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # FastReID Baseline subtracts PIXEL_MEAN/STDs internally; retain 0..255 input scale.
        self.transform = transform or v2.Compose(
            [
                v2.Resize((input_size, input_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda tensor: tensor * 255.0),
            ]
        )

    @property
    def details(self) -> dict[str, Any]:
        return {
            "architecture": "TOPICTrack BEE AGW ResNeSt-50 + GeM + BN neck",
            "config": "fast-reid/configs/bee/AGW_S50.yml",
            "input_size_override": self.input_size,
            "official_size_test": [384, 384],
            "checkpoint_loading": "precompare all keys/shapes then strict=True",
            "load_report": self.load_report,
            "input_scale": "RGB float32 0..255; model applies config normalization",
            "transform": repr(self.transform),
            "upstream_commit": TOPICTRACK_COMMIT,
        }

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        batch = self.torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        with self.torch.inference_mode(), torch_autocast(
            self.torch, self.device, self.amp, self.amp_dtype
        ):
            output = self.model(batch)
        if isinstance(output, Mapping):
            for key in ("features", "feat", "embedding", "embeddings"):
                if key in output:
                    output = output[key]
                    break
        if isinstance(output, (tuple, list)):
            output = output[0]
        if getattr(output, "ndim", 0) > 2:
            output = output.flatten(1)
        return validate_embeddings(output.float().cpu().numpy(), len(images))
