"""Shared feature-extractor contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, ContextManager, Sequence

import numpy as np
from PIL import Image


class ModelContractError(RuntimeError):
    """Raised when a model violates the H1 embedding contract."""


def validate_embeddings(values: Any, expected_rows: int | None = None) -> np.ndarray:
    embeddings = np.asarray(values, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ModelContractError(f"Expected a 2-D embedding matrix, got shape {embeddings.shape}")
    if expected_rows is not None and embeddings.shape[0] != expected_rows:
        raise ModelContractError(f"Expected {expected_rows} embeddings, got {embeddings.shape[0]}")
    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ModelContractError("Embedding matrix is empty or contains non-finite values")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ModelContractError("Embedding matrix contains a zero vector")
    embeddings = embeddings / norms
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ModelContractError("Could not L2-normalize embeddings")
    return embeddings.astype(np.float32, copy=False)


def torch_autocast(torch: Any, device: Any, enabled: bool, dtype_name: str) -> ContextManager[Any]:
    if not enabled:
        return nullcontext()
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


class FeatureExtractor(ABC):
    """Composable PIL-to-float32-L2 feature interface."""

    name: str

    @abstractmethod
    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return one finite, float32, normalized row per input image."""

    @property
    @abstractmethod
    def details(self) -> dict[str, Any]:
        """Return transform/model details included in the cache fingerprint."""
