"""Lazy BEE24 crop loading; crops are never materialized as a dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

from .mot import MotDataError, Observation


def load_crop(dataset_root: Path, observation: Observation) -> Image.Image:
    root = dataset_root.resolve(strict=False)
    image_path = (root / observation.image_path).resolve(strict=False)
    try:
        image_path.relative_to(root)
    except ValueError as error:
        raise MotDataError(f"Manifest image escapes dataset root: {observation.image_path}") from error
    with Image.open(image_path) as image:
        if image.size != (observation.image_width, observation.image_height):
            raise MotDataError(
                f"Image dimensions changed for {image_path}: {image.size}, "
                f"expected {(observation.image_width, observation.image_height)}"
            )
        return image.convert("RGB").crop(
            (observation.crop_x0, observation.crop_y0, observation.crop_x1, observation.crop_y1)
        )


def batches(values: list[Observation], size: int) -> Iterator[list[Observation]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]
