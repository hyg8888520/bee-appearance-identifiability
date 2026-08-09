"""Torchvision ResNet-50 ImageNet V2 appearance baseline."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image

from .base import FeatureExtractor, torch_autocast, validate_embeddings


class ResNet50Extractor(FeatureExtractor):
    name = "resnet50"

    def __init__(
        self,
        input_size: int,
        device_name: str,
        amp: bool,
        amp_dtype: str,
        *,
        model: Any | None = None,
        transform: Any | None = None,
    ) -> None:
        try:
            import torch
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as error:
            raise RuntimeError("ResNet50 requires torch and torchvision in the active environment") from error
        self.torch = torch
        self.device = torch.device(device_name)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.input_size = input_size
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.transform = transform or weights.transforms(crop_size=input_size, resize_size=input_size)
        self.model = model or resnet50(weights=weights)
        if hasattr(self.model, "fc"):
            self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def details(self) -> dict[str, Any]:
        return {
            "architecture": "torchvision.resnet50",
            "weights": "ResNet50_Weights.IMAGENET1K_V2",
            "head": "Identity",
            "input_size": self.input_size,
            "transform": repr(self.transform),
        }

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        batch = self.torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        with self.torch.inference_mode(), torch_autocast(
            self.torch, self.device, self.amp, self.amp_dtype
        ):
            output = self.model(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = output.flatten(1)
        return validate_embeddings(output.float().cpu().numpy(), len(images))
