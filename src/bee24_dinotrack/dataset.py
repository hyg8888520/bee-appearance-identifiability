"""COCO-style BEE24 annotation access."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


class DatasetFormatError(ValueError):
    """Raised when the annotation file is not a supported COCO-style document."""


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    file_name: str


def load_image_records(annotation: Path) -> list[ImageRecord]:
    try:
        payload: Any = json.loads(annotation.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetFormatError(f"Annotation is not valid JSON: {annotation}: {error}") from error
    except OSError as error:
        raise DatasetFormatError(f"Could not read annotation: {annotation}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise DatasetFormatError(f"Annotation must contain a top-level 'images' list: {annotation}")
    records: list[ImageRecord] = []
    seen_ids: set[int] = set()
    for index, raw in enumerate(payload["images"]):
        if not isinstance(raw, dict):
            raise DatasetFormatError(f"images[{index}] must be an object.")
        image_id = raw.get("id")
        file_name = raw.get("file_name")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise DatasetFormatError(f"images[{index}].id must be an integer.")
        if image_id in seen_ids:
            raise DatasetFormatError(f"Duplicate image id in annotation: {image_id}")
        if not isinstance(file_name, str) or not file_name.strip():
            raise DatasetFormatError(f"images[{index}].file_name must be a non-empty string.")
        posix_path = PurePosixPath(file_name)
        windows_path = PureWindowsPath(file_name)
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or ".." in posix_path.parts
            or ".." in windows_path.parts
        ):
            raise DatasetFormatError(
                f"images[{index}].file_name must be relative to dataset.root: {file_name}"
            )
        seen_ids.add(image_id)
        records.append(ImageRecord(image_id=image_id, file_name=file_name))
    if not records:
        raise DatasetFormatError(f"Annotation contains no image records: {annotation}")
    return records


def image_path(dataset_root: Path, record: ImageRecord) -> Path:
    """Construct the required BEE24 image path as root / file_name."""

    return dataset_root / record.file_name
