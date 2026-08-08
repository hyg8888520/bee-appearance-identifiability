"""Atomic, resumable NumPy feature shards with strict signatures."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import atomic_write_json, canonical_json, sha256_text


class CacheError(RuntimeError):
    """Raised for corrupt or mismatched feature caches."""


def cache_fingerprint(signature: dict[str, Any]) -> str:
    return sha256_text(canonical_json(signature))


@dataclass
class FeatureCache:
    root: Path
    model_name: str
    signature: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return cache_fingerprint(self.signature)

    @property
    def directory(self) -> Path:
        return self.root / "features" / self.model_name / self.fingerprint

    @property
    def metadata_path(self) -> Path:
        return self.directory / "cache_manifest.json"

    def initialize(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        expected = {"fingerprint": self.fingerprint, "signature": self.signature, "format_version": 1}
        if self.metadata_path.exists():
            observed = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if observed != expected:
                raise CacheError(
                    f"Cache metadata does not match its fingerprint directory: {self.metadata_path}. "
                    "Refusing reuse; choose a fresh cache root or remove the corrupt directory."
                )
        else:
            atomic_write_json(self.metadata_path, expected)

    def shard_path(self, index: int) -> Path:
        return self.directory / f"shard-{index:06d}.npz"

    def validate_shard(self, index: int, expected_ids: list[str]) -> bool:
        path = self.shard_path(index)
        if not path.is_file():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                ids = data["observation_ids"].astype(str).tolist()
                embeddings = data["embeddings"]
                if ids != expected_ids or embeddings.dtype != np.float32 or embeddings.ndim != 2:
                    return False
                if embeddings.shape[0] != len(ids) or not np.isfinite(embeddings).all():
                    return False
                norms = np.linalg.norm(embeddings, axis=1)
                return bool(np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4))
        except (OSError, KeyError, ValueError):
            return False

    def write_shard(self, index: int, observation_ids: list[str], embeddings: np.ndarray) -> Path:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(observation_ids):
            raise CacheError("Embedding shard shape does not match observation IDs")
        if not np.isfinite(values).all():
            raise CacheError("Embedding shard contains non-finite values")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
            raise CacheError("Embeddings must be L2-normalized before caching")
        self.initialize()
        destination = self.shard_path(index)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp.npz")
        np.savez_compressed(
            temporary,
            observation_ids=np.asarray(observation_ids, dtype=np.str_),
            embeddings=values,
        )
        os.replace(temporary, destination)
        return destination

    def load_all(self, expected_ids: list[str], shard_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for index, start in enumerate(range(0, len(expected_ids), shard_size)):
            shard_ids = expected_ids[start : start + shard_size]
            if not self.validate_shard(index, shard_ids):
                raise CacheError(f"Missing or invalid shard {index} in {self.directory}")
            with np.load(self.shard_path(index), allow_pickle=False) as data:
                chunks.append(data["embeddings"].astype(np.float32, copy=False))
        if not chunks:
            raise CacheError("No embeddings available")
        return np.concatenate(chunks, axis=0)


def open_cache(directory: Path, model_name: str) -> FeatureCache:
    metadata_path = directory / "cache_manifest.json"
    if not metadata_path.is_file():
        raise CacheError(f"Cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("signature"), dict):
        raise CacheError(f"Invalid cache metadata: {metadata_path}")
    # <root>/features/<model>/<fingerprint>
    try:
        root = directory.parents[2]
    except IndexError as error:
        raise CacheError(f"Cache directory is not in the expected layout: {directory}") from error
    cache = FeatureCache(root, model_name, metadata["signature"])
    if cache.directory.resolve(strict=False) != directory.resolve(strict=False):
        raise CacheError(f"Cache fingerprint/path mismatch: {directory}")
    cache.initialize()
    return cache
