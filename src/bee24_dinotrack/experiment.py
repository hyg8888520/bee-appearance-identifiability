"""H1 DINOv3 global-feature retrieval experiment."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ExperimentConfig
from .dataset import ImageRecord, image_path
from .validation import ValidatedExperiment


class ExperimentError(RuntimeError):
    """Raised when the validated experiment cannot complete."""


def _selected_images(validated: ValidatedExperiment) -> tuple[ImageRecord, ...]:
    records = validated.images
    limit = validated.config.dataset.max_images
    selected = records if limit is None else records[:limit]
    if len(selected) < 2:
        raise ExperimentError(
            "H1 retrieval requires at least two selected images. "
            "Increase dataset.max_images or use an annotation with more images."
        )
    missing = [
        image_path(validated.config.dataset.root, record)
        for record in selected
        if not image_path(validated.config.dataset.root, record).is_file()
    ]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:5])
        remainder = "" if len(missing) <= 5 else f"\n  - ... and {len(missing) - 5} more"
        raise ExperimentError(
            "Selected annotation images are missing; fix dataset.root or the annotation before inference:\n"
            f"{preview}{remainder}"
        )
    return tuple(selected)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output_locations(config: ExperimentConfig) -> None:
    for directory in (config.output.figures, config.output.failure_cases):
        directory.mkdir(parents=True, exist_ok=True)
    for file_path in (config.output.feature_cache, config.output.query_results, config.output.metadata):
        file_path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Any, torch_module: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    torch_module.save(payload, temporary)
    os.replace(temporary, path)


def _image_signature(records: Sequence[ImageRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record.image_id}\0{record.file_name}\n".encode("utf-8"))
    return digest.hexdigest()


def _feature_config_signature(config: ExperimentConfig, checkpoint_sha256: str) -> str:
    payload = {
        "checkpoint_sha256": checkpoint_sha256,
        "model_name": config.dinov3.model_name,
        "image_size": config.dinov3.image_size,
        "mean": config.dinov3.mean,
        "std": config.dinov3.std,
        "interpolation": config.dinov3.interpolation,
        "global_pool": config.dinov3.global_pool,
        "use_amp": config.dinov3.use_amp,
        "amp_dtype": config.dinov3.amp_dtype,
        "normalize_features": config.dinov3.normalize_features,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cached_features(
    config: ExperimentConfig,
    records: Sequence[ImageRecord],
    checkpoint_sha256: str,
    torch: Any,
) -> Any | None:
    cache = config.output.feature_cache
    if not config.h1.reuse_feature_cache or not cache.is_file():
        return None
    try:
        payload = torch.load(cache, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ExperimentError(f"Could not read feature cache {cache}: {error}") from error
    expected_ids = [record.image_id for record in records]
    if not isinstance(payload, dict) or payload.get("image_ids") != expected_ids:
        raise ExperimentError(
            f"Feature cache does not match the selected annotation images: {cache}. "
            "Set h1.reuse_feature_cache: false to rebuild it."
        )
    expected_signature = _image_signature(records)
    if payload.get("image_signature") != expected_signature:
        raise ExperimentError(
            f"Feature cache image signature is stale: {cache}. Set h1.reuse_feature_cache: false to rebuild it."
        )
    if payload.get("model_name") != config.dinov3.model_name:
        raise ExperimentError(
            f"Feature cache was produced by a different model: {cache}. "
            "Set h1.reuse_feature_cache: false to rebuild it."
        )
    expected_feature_signature = _feature_config_signature(config, checkpoint_sha256)
    if payload.get("feature_config_signature") != expected_feature_signature:
        raise ExperimentError(
            f"Feature cache does not match the configured checkpoint or preprocessing: {cache}. "
            "Set h1.reuse_feature_cache: false to rebuild it."
        )
    features = payload.get("features")
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] != len(records):
        raise ExperimentError(f"Feature cache has an invalid tensor shape: {cache}")
    return features


def _build_transform(config: ExperimentConfig) -> Any:
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except ImportError as error:
        raise ExperimentError("torchvision is required. Run 'pip install -e .'.") from error
    interpolation_name = config.dinov3.interpolation.upper()
    try:
        interpolation = InterpolationMode[interpolation_name]
    except KeyError as error:
        choices = ", ".join(member.name.lower() for member in InterpolationMode)
        raise ExperimentError(
            f"Unsupported models.dinov3.interpolation '{config.dinov3.interpolation}'. Choose one of: {choices}."
        ) from error
    return transforms.Compose(
        [
            transforms.Resize(config.dinov3.image_size, interpolation=interpolation, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.dinov3.mean, std=config.dinov3.std),
        ]
    )


def _extract_features(config: ExperimentConfig, records: Sequence[ImageRecord], torch: Any) -> Any:
    try:
        import timm
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise ExperimentError("DINOv3 inference dependencies are missing. Run 'pip install -e .'.") from error

    transform = _build_transform(config)

    class BeeImages(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> Any:
            record = records[index]
            path = image_path(config.dataset.root, record)
            try:
                with Image.open(path) as image:
                    tensor = transform(image.convert("RGB"))
            except Exception as error:
                raise ExperimentError(f"Could not decode image {path}: {error}") from error
            return tensor, record.image_id

    device = torch.device(config.device)
    amp_dtype = getattr(torch, config.dinov3.amp_dtype, None)
    if config.dinov3.use_amp and device.type == "cuda" and not isinstance(amp_dtype, torch.dtype):
        raise ExperimentError(
            f"Unsupported models.dinov3.amp_dtype: {config.dinov3.amp_dtype}. "
            "Use a torch dtype such as float16 or bfloat16."
        )
    try:
        model = timm.create_model(
            config.dinov3.model_name,
            pretrained=True,
            pretrained_cfg_overlay={"file": str(config.dinov3.checkpoint)},
            num_classes=0,
            global_pool=config.dinov3.global_pool,
        )
    except Exception as error:
        raise ExperimentError(
            f"Could not construct {config.dinov3.model_name} from checkpoint "
            f"{config.dinov3.checkpoint}: {error}"
        ) from error
    model = model.to(device).eval()
    loader = DataLoader(
        BeeImages(),
        batch_size=config.dinov3.batch_size,
        shuffle=False,
        num_workers=config.dinov3.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.dinov3.num_workers > 0,
    )
    batches: list[Any] = []
    observed_ids: list[int] = []
    try:
        with torch.inference_mode():
            for images, image_ids in loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if config.dinov3.use_amp and device.type == "cuda"
                    else nullcontext()
                )
                with amp_context:
                    features = model(images)
                if not isinstance(features, torch.Tensor):
                    raise ExperimentError(
                        "The configured timm model did not return a tensor. "
                        "Choose a backbone/global_pool combination that returns global features."
                    )
                if features.ndim > 2:
                    features = features.flatten(start_dim=2).mean(dim=2)
                if features.ndim != 2:
                    raise ExperimentError(f"Expected a 2D feature tensor, received shape {tuple(features.shape)}.")
                features = features.float()
                if config.dinov3.normalize_features:
                    features = torch.nn.functional.normalize(features, dim=1)
                batches.append(features.cpu())
                observed_ids.extend(int(item) for item in image_ids)
    except ExperimentError:
        raise
    except Exception as error:
        raise ExperimentError(f"DINOv3 feature extraction failed: {error}") from error
    expected_ids = [record.image_id for record in records]
    if observed_ids != expected_ids:
        raise ExperimentError("Data loader returned images in an unexpected order.")
    return torch.cat(batches, dim=0)


def _query_indices(config: ExperimentConfig, records: Sequence[ImageRecord]) -> list[int]:
    id_to_index = {record.image_id: index for index, record in enumerate(records)}
    if config.h1.query_image_ids:
        unknown = [image_id for image_id in config.h1.query_image_ids if image_id not in id_to_index]
        if unknown:
            raise ExperimentError(
                "h1.query_image_ids contains IDs absent from the selected images: "
                + ", ".join(str(item) for item in unknown)
            )
        return [id_to_index[image_id] for image_id in config.h1.query_image_ids]
    return list(range(0, len(records), config.h1.query_stride))


def _run_queries(config: ExperimentConfig, records: Sequence[ImageRecord], features: Any, torch: Any) -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    top_k = min(config.h1.top_k, len(records) - 1)
    for query_index in _query_indices(config, records):
        scores = features @ features[query_index]
        scores = scores.clone()
        scores[query_index] = -torch.inf
        values, indices = torch.topk(scores, k=top_k)
        query_record = records[query_index]
        matches = [
            {
                "rank": rank,
                "image_id": records[int(match_index)].image_id,
                "file_name": records[int(match_index)].file_name,
                "cosine_similarity": float(score),
            }
            for rank, (score, match_index) in enumerate(zip(values.tolist(), indices.tolist()), start=1)
        ]
        queries.append(
            {
                "image_id": query_record.image_id,
                "file_name": query_record.file_name,
                "matches": matches,
            }
        )
    return {
        "experiment": config.name,
        "model_name": config.dinov3.model_name,
        "num_images": len(records),
        "top_k": top_k,
        "queries": queries,
    }


def run_h1(validated: ValidatedExperiment) -> dict[str, Any]:
    """Extract or reuse features, run configured queries, and write outputs."""

    config = validated.config
    records = _selected_images(validated)
    _query_indices(config, records)
    _prepare_output_locations(config)
    try:
        import torch
    except ImportError as error:
        raise ExperimentError("PyTorch is required for H1. Run 'pip install -e .'.") from error
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    started = time.time()
    checkpoint_sha256 = _checkpoint_sha256(config.dinov3.checkpoint)
    features = _load_cached_features(config, records, checkpoint_sha256, torch)
    cache_reused = features is not None
    if features is None:
        features = _extract_features(config, records, torch)
        _atomic_torch_save(
            config.output.feature_cache,
            {
                "features": features,
                "image_ids": [record.image_id for record in records],
                "file_names": [record.file_name for record in records],
                "image_signature": _image_signature(records),
                "model_name": config.dinov3.model_name,
                "checkpoint_sha256": checkpoint_sha256,
                "feature_config_signature": _feature_config_signature(config, checkpoint_sha256),
            },
            torch,
        )
    results = _run_queries(config, records, features, torch)
    _atomic_json(config.output.query_results, results)
    finished = time.time()
    metadata = {
        "status": "completed",
        "started_unix": started,
        "finished_unix": finished,
        "duration_seconds": finished - started,
        "cache_reused": cache_reused,
        "checkpoint_sha256": checkpoint_sha256,
        "feature_shape": list(features.shape),
        "config": config.serializable(),
        "outputs": {
            "feature_cache": str(config.output.feature_cache),
            "query_results": str(config.output.query_results),
            "figures": str(config.output.figures),
            "failure_cases": str(config.output.failure_cases),
            "metadata": str(config.output.metadata),
        },
    }
    _atomic_json(config.output.metadata, metadata)
    return metadata
