"""Deterministic H2 crop/context interventions without materializing a crop corpus."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from ..config import H2VariantConfig
from ..data.mot import MotDataError, Observation, expanded_crop_box

_NEUTRAL_RGB = (127, 127, 127)


def variant_geometry(
    observation: Observation,
    variant: H2VariantConfig,
    *,
    patch_size: int,
) -> dict[str, Any]:
    x1, y1, x2, y2, clipped = expanded_crop_box(
        observation.bbox_x1,
        observation.bbox_y1,
        observation.bbox_x2,
        observation.bbox_y2,
        observation.image_width,
        observation.image_height,
        variant.crop_expansion,
    )
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise MotDataError(f"H2 variant {variant.name} creates an empty crop for {observation.observation_id}")
    foreground_x1 = max(0, min(width, math.floor(observation.bbox_x1) - x1))
    foreground_y1 = max(0, min(height, math.floor(observation.bbox_y1) - y1))
    foreground_x2 = max(0, min(width, math.ceil(observation.bbox_x2) - x1))
    foreground_y2 = max(0, min(height, math.ceil(observation.bbox_y2) - y1))
    foreground_area = max(0, foreground_x2 - foreground_x1) * max(
        0, foreground_y2 - foreground_y1
    )
    resized_bbox_width = variant.input_size * observation.original_width / width
    resized_bbox_height = variant.input_size * observation.original_height / height
    resized_bbox_area = resized_bbox_width * resized_bbox_height
    return {
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "crop_width": width,
        "crop_height": height,
        "variant_clipped": clipped,
        "foreground_x1": foreground_x1,
        "foreground_y1": foreground_y1,
        "foreground_x2": foreground_x2,
        "foreground_y2": foreground_y2,
        "foreground_fraction": foreground_area / (width * height),
        "context_fraction": 1.0 - foreground_area / (width * height),
        "resized_bbox_width": resized_bbox_width,
        "resized_bbox_height": resized_bbox_height,
        "resized_bbox_area": resized_bbox_area,
        "approx_patch_coverage": resized_bbox_area / float(patch_size * patch_size),
    }


def load_variant_crop(
    dataset_root: Path,
    observation: Observation,
    variant: H2VariantConfig,
    *,
    patch_size: int,
) -> Image.Image:
    root = dataset_root.resolve(strict=False)
    image_path = (root / observation.image_path).resolve(strict=False)
    try:
        image_path.relative_to(root)
    except ValueError as error:
        raise MotDataError(f"Manifest image escapes dataset root: {observation.image_path}") from error
    geometry = variant_geometry(observation, variant, patch_size=patch_size)
    with Image.open(image_path) as image:
        if image.size != (observation.image_width, observation.image_height):
            raise MotDataError(
                f"Image dimensions changed for {image_path}: {image.size}, "
                f"expected {(observation.image_width, observation.image_height)}"
            )
        crop = image.convert("RGB").crop(
            (
                geometry["crop_x1"], geometry["crop_y1"],
                geometry["crop_x2"], geometry["crop_y2"],
            )
        )
    foreground_box = (
        geometry["foreground_x1"], geometry["foreground_y1"],
        geometry["foreground_x2"], geometry["foreground_y2"],
    )
    if variant.pixel_view == "bbox_foreground_only":
        controlled = Image.new("RGB", crop.size, _NEUTRAL_RGB)
        controlled.paste(crop.crop(foreground_box), foreground_box[:2])
        crop.close()
        crop = controlled
    elif variant.pixel_view == "context_only":
        neutral = Image.new(
            "RGB",
            (
                max(0, foreground_box[2] - foreground_box[0]),
                max(0, foreground_box[3] - foreground_box[1]),
            ),
            _NEUTRAL_RGB,
        )
        crop.paste(neutral, foreground_box[:2])
        neutral.close()
    elif variant.pixel_view == "gaussian_blur":
        blurred = crop.filter(ImageFilter.GaussianBlur(radius=variant.blur_radius))
        crop.close()
        crop = blurred
    elif variant.pixel_view != "raw":
        crop.close()
        raise ValueError(f"Unsupported H2 pixel view: {variant.pixel_view}")
    return crop
