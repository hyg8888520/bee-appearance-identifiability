"""Outcome-blind observation signals shared by all H3 association variants."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..config import ExperimentConfig
from ..data.mot import Observation
from ..h2.signals import image_quality_signals, spatial_context_signals
from ..utils import atomic_write_json, atomic_write_text, canonical_json, sha256_file, sha256_text
from .core import H3Inputs, require_h3, validate_h3_inputs


H3_SIGNAL_FIELDS = (
    "observation_id",
    "partition",
    "video_id",
    "frame_id",
    "track_id",
    "identity",
    "bbox_area",
    "bbox_orientation_proxy_deg",
    "bbox_laplacian_variance",
    "max_bbox_iou",
    "neighbor_count_wide",
)


def _observation_ids_sha256(observations: Sequence[Observation]) -> str:
    return sha256_text(canonical_json([item.observation_id for item in observations]))


def _signature(config: ExperimentConfig, inputs: H3Inputs) -> dict[str, Any]:
    h3 = require_h3(config)
    return {
        "format_version": 1,
        "implementation": "beeid.h3.signals:v1",
        "manifest_sha256": inputs.audit["manifest_sha256"],
        "protocol_sha256": inputs.audit["protocol_sha256"],
        "project_split_sha256": inputs.audit["project_split_sha256"],
        "observation_ids_sha256": _observation_ids_sha256(inputs.observations),
        "allow_subset": h3.allow_subset,
        "bbox_measurement": "zero-context clipped GT rectangle resized bilinearly to 64x64",
        "crowding_radius_multipliers": [1.0, 2.0],
        "outcome_blind": True,
        "final_test_read": False,
    }


def read_h3_signal_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"H3 observation signals do not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != H3_SIGNAL_FIELDS:
            raise RuntimeError(f"Unexpected H3 signal schema: {path}")
        rows = list(reader)
    if len({row["observation_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"H3 signal table contains duplicate observation IDs: {path}")
    return rows


def _bbox_quality(image: Image.Image, item: Observation) -> dict[str, float]:
    x1 = max(0, min(item.image_width, math.floor(item.bbox_x1)))
    y1 = max(0, min(item.image_height, math.floor(item.bbox_y1)))
    x2 = max(0, min(item.image_width, math.ceil(item.bbox_x2)))
    y2 = max(0, min(item.image_height, math.ceil(item.bbox_y2)))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Empty H3 bbox quality crop: {item.observation_id}")
    crop = image.crop((x1, y1, x2, y2))
    try:
        resized = crop.resize((64, 64), Image.Resampling.BILINEAR)
        try:
            values = image_quality_signals(resized)
        finally:
            resized.close()
    finally:
        crop.close()
    return {
        "bbox_orientation_proxy_deg": float(values["gradient_orientation_proxy_deg"]),
        "bbox_laplacian_variance": float(values["laplacian_variance"]),
    }


def build_h3_signals(config: ExperimentConfig) -> dict[str, Any]:
    inputs = validate_h3_inputs(config)
    signature = _signature(config, inputs)
    fingerprint = sha256_text(canonical_json(signature))
    output = config.paths.output_root / "h3_observation_signals.csv"
    metadata_path = config.paths.output_root / "h3_signal_metadata.json"
    if output.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            isinstance(metadata, dict)
            and metadata.get("fingerprint") == fingerprint
            and metadata.get("signal_csv_sha256") == sha256_file(output)
        ):
            rows = read_h3_signal_rows(output)
            if len(rows) == len(inputs.observations):
                return {**metadata, "status": "reused", "row_count": len(rows)}

    by_frame: dict[tuple[str, int], list[Observation]] = defaultdict(list)
    by_image: dict[str, list[Observation]] = defaultdict(list)
    for item in inputs.observations:
        by_frame[(item.video_id, item.frame)].append(item)
        by_image[item.image_path].append(item)

    quality_by_id: dict[str, dict[str, float]] = {}
    image_paths = sorted(by_image)
    progress_path = config.paths.output_root / "h3_logs" / "signal_progress.json"
    atomic_write_json(
        progress_path,
        {
            "status": "running",
            "processed_frame_images": 0,
            "total_frame_images": len(image_paths),
            "processed_observations": 0,
            "final_test_read": False,
        },
    )
    processed_observations = 0
    for image_index, image_path in enumerate(image_paths, start=1):
        absolute = config.paths.bee24_root / image_path
        if not absolute.is_file():
            raise RuntimeError(f"Missing H3 source frame: {absolute}")
        with Image.open(absolute) as source:
            rgb = source.convert("RGB")
            try:
                for item in by_image[image_path]:
                    quality_by_id[item.observation_id] = _bbox_quality(rgb, item)
                    processed_observations += 1
            finally:
                rgb.close()
        if image_index % 500 == 0 or image_index == len(image_paths):
            atomic_write_json(
                progress_path,
                {
                    "status": "running",
                    "processed_frame_images": image_index,
                    "total_frame_images": len(image_paths),
                    "processed_observations": processed_observations,
                    "final_test_read": False,
                },
            )

    rows: list[dict[str, Any]] = []
    for item in inputs.observations:
        spatial = spatial_context_signals(
            item, by_frame[(item.video_id, item.frame)], (1.0, 2.0)
        )
        rows.append(
            {
                "observation_id": item.observation_id,
                "partition": inputs.partition_by_observation[item.observation_id],
                "video_id": item.video_id,
                "frame_id": item.frame,
                "track_id": item.track_id,
                "identity": item.identity,
                "bbox_area": item.bbox_area,
                **quality_by_id[item.observation_id],
                "max_bbox_iou": spatial["max_bbox_iou"],
                "neighbor_count_wide": spatial["neighbor_count_wide"],
            }
        )
    rows.sort(key=lambda row: (row["video_id"], row["frame_id"], row["track_id"]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=H3_SIGNAL_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(output, buffer.getvalue())
    metadata = {
        "status": "completed",
        "fingerprint": fingerprint,
        "signal_csv_sha256": sha256_file(output),
        "signature": signature,
        "row_count": len(rows),
        "project_train_row_count": sum(
            row["partition"] == "project_train" for row in rows
        ),
        "development_row_count": sum(
            row["partition"] == "development_validation" for row in rows
        ),
        "input_audit": inputs.audit,
        "excluded_as_direct_reliability_features": [
            "bbox_area",
            "bbox_laplacian_variance",
        ],
        "definitions": {
            "bbox_area": "stored only to compute causal scale change; never used as a static reliability feature",
            "bbox_laplacian_variance": "stored only to compute causal sharpness change; never used as a static reliability feature",
            "bbox_orientation_proxy_deg": "structure-tensor proxy modulo 180 degrees, not bee pose ground truth",
            "max_bbox_iou": "same-frame GT-box overlap proxy",
            "neighbor_count_wide": "same-frame centers within two query-bbox diagonals",
        },
        "outcome_blind": True,
        "final_test_read": False,
    }
    atomic_write_json(metadata_path, metadata)
    atomic_write_json(
        progress_path,
        {
            "status": "completed",
            "processed_frame_images": len(image_paths),
            "total_frame_images": len(image_paths),
            "processed_observations": len(rows),
            "final_test_read": False,
        },
    )
    return metadata
