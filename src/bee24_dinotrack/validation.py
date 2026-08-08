"""Fail-fast validation performed before model construction or inference."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import ExperimentConfig
from .dataset import DatasetFormatError, ImageRecord, image_path, load_image_records


class StartupValidationError(RuntimeError):
    """Raised when the configured runtime environment cannot run the experiment."""


@dataclass(frozen=True)
class ValidatedExperiment:
    config: ExperimentConfig
    images: tuple[ImageRecord, ...]
    sample_image: Path


def _validate_cuda(device: str, errors: list[str]) -> None:
    match = re.fullmatch(r"cuda(?::(\d+))?", device.lower())
    if not match:
        if device.lower() != "cpu":
            errors.append("experiment.device must be 'cpu', 'cuda', or 'cuda:<index>'.")
        return
    requested_index = int(match.group(1) or 0)
    try:
        import torch
    except ImportError:
        errors.append("CUDA was requested, but PyTorch is not installed. Run 'pip install -e .'.")
        return
    if not torch.cuda.is_available():
        errors.append(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build or set experiment.device: cpu."
        )
        return
    device_count = torch.cuda.device_count()
    if requested_index >= device_count:
        errors.append(
            f"CUDA device cuda:{requested_index} was requested, but only {device_count} CUDA device(s) are visible."
        )


def _validate_output_root(root: Path, errors: list[str]) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            errors.append(f"output.root is not a directory: {root}")
            return
        probe = root / f".bee24_write_probe_{uuid.uuid4().hex}"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as error:
        errors.append(f"output.root is not writable: {root}: {error}")


def validate_startup(config: ExperimentConfig) -> ValidatedExperiment:
    """Validate all cheap prerequisites and return parsed image records."""

    errors: list[str] = []
    images: list[ImageRecord] = []
    sample_image: Path | None = None
    if not config.dataset.root.is_dir():
        errors.append(f"dataset.root does not exist or is not a directory: {config.dataset.root}")
    if not config.dataset.annotation.is_file():
        errors.append(f"dataset.annotation does not exist or is not a file: {config.dataset.annotation}")
    else:
        try:
            images = load_image_records(config.dataset.annotation)
        except DatasetFormatError as error:
            errors.append(str(error))
    if config.dataset.root.is_dir() and images:
        sample_image = next(
            (candidate for record in images if (candidate := image_path(config.dataset.root, record)).is_file()),
            None,
        )
        if sample_image is None:
            errors.append(
                "No image referenced by dataset.annotation exists beneath dataset.root. "
                "Expected paths are constructed as dataset.root / images[i].file_name."
            )
    if not config.dinov3.checkpoint.is_file():
        errors.append(
            f"models.dinov3.checkpoint does not exist or is not a file: {config.dinov3.checkpoint}"
        )
    _validate_output_root(config.output.root, errors)
    _validate_cuda(config.device, errors)
    if errors:
        details = "\n".join(f"  - {message}" for message in errors)
        raise StartupValidationError(f"Startup validation failed:\n{details}")
    assert sample_image is not None
    return ValidatedExperiment(config=config, images=tuple(images), sample_image=sample_image)


def format_summary(validated: ValidatedExperiment) -> str:
    config = validated.config
    selected = len(validated.images)
    if config.dataset.max_images is not None:
        selected = min(selected, config.dataset.max_images)
    return "\n".join(
        [
            "Experiment summary",
            f"  name: {config.name}",
            f"  dataset: {config.dataset.root}",
            f"  annotation: {config.dataset.annotation}",
            f"  images: {len(validated.images)} indexed, {selected} selected",
            f"  sample: {validated.sample_image}",
            f"  model: {config.dinov3.model_name}",
            f"  checkpoint: {config.dinov3.checkpoint}",
            f"  device: {config.device}",
            f"  output: {config.output.root}",
        ]
    )

