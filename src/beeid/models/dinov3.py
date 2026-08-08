"""DINOv3 ViT-S/16 loaded exclusively from the pinned local torch Hub checkout."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..references import DINOV3_COMMIT
from ..utils import require_git_head
from .base import FeatureExtractor, ModelContractError, torch_autocast, validate_embeddings


class DINOv3Extractor(FeatureExtractor):
    name = "dinov3"

    def __init__(
        self,
        repo: Any,
        weights: Any,
        input_size: int,
        feature_source: str,
        device_name: str,
        amp: bool,
        amp_dtype: str,
        *,
        model: Any | None = None,
        transform: Any | None = None,
        verify_commit: bool = True,
    ) -> None:
        try:
            import torch
            from torchvision.transforms import v2
        except ImportError as error:
            raise RuntimeError("DINOv3 requires torch and torchvision in the active environment") from error
        if feature_source not in {"patch_mean", "cls"}:
            raise ValueError(f"Unsupported DINOv3 feature source: {feature_source}")
        if verify_commit:
            require_git_head(repo, DINOV3_COMMIT, "DINOv3")
        if model is None and not weights.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint does not exist: {weights}")
        self.torch = torch
        self.device = torch.device(device_name)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.input_size = input_size
        self.feature_source = feature_source
        self.transform = transform or v2.Compose(
            [
                v2.Resize((input_size, input_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model = model or torch.hub.load(
            str(repo), "dinov3_vits16", source="local", weights=str(weights)
        )
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def details(self) -> dict[str, Any]:
        return {
            "architecture": "dinov3_vits16",
            "loader": "torch.hub.load(source='local')",
            "forward_api": "forward_features",
            "feature_source": self.feature_source,
            "input_size": self.input_size,
            "transform": repr(self.transform),
            "upstream_commit": DINOV3_COMMIT,
        }

    def _select(self, features: Any) -> Any:
        if not isinstance(features, dict):
            raise ModelContractError("DINOv3 forward_features() must return a dictionary")
        required = {"x_norm_patchtokens", "x_norm_clstoken"}
        missing = sorted(required - set(features))
        if missing:
            raise ModelContractError(f"DINOv3 forward_features() missing keys: {missing}")
        patches = features["x_norm_patchtokens"]
        cls = features["x_norm_clstoken"]
        if patches.ndim != 3 or cls.ndim != 2 or patches.shape[0] != cls.shape[0]:
            raise ModelContractError(
                f"Unexpected DINOv3 shapes: patch={tuple(patches.shape)}, cls={tuple(cls.shape)}"
            )
        if not patches.is_floating_point() or not cls.is_floating_point():
            raise ModelContractError(
                f"DINOv3 features must be floating point, got patch={patches.dtype}, cls={cls.dtype}"
            )
        if not self.torch.isfinite(patches).all() or not self.torch.isfinite(cls).all():
            raise ModelContractError("DINOv3 forward_features() returned non-finite values")
        return patches.mean(dim=1) if self.feature_source == "patch_mean" else cls

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        batch = self.torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        with self.torch.inference_mode(), torch_autocast(
            self.torch, self.device, self.amp, self.amp_dtype
        ):
            selected = self._select(self.model.forward_features(batch))
        return validate_embeddings(selected.float().cpu().numpy(), len(images))

    def intermediate_layers(self, images: Sequence[Image.Image], n: int = 1) -> Any:
        """Expose the official intermediate-layer API without changing H1 defaults."""
        batch = self.torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        with self.torch.inference_mode():
            return self.model.get_intermediate_layers(batch, n=n)
