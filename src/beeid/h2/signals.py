"""Model-independent H2 observation signals computed without looking at outcomes."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..config import ExperimentConfig, H2VariantConfig
from ..data.mot import Observation
from ..utils import atomic_write_json, atomic_write_text, canonical_json, sha256_text
from .core import require_h2, validate_h2_inputs
from .crops import load_variant_crop, variant_geometry
from .features import observation_ids_sha256


SIGNAL_FIELDS = (
    "observation_id", "split", "video_id", "frame_id", "track_id", "identity",
    "variant", "pixel_view", "crop_expansion", "input_size", "bbox_area", "bbox_width",
    "bbox_height", "aspect_ratio", "crop_width", "crop_height", "foreground_fraction",
    "context_fraction", "resized_bbox_width", "resized_bbox_height", "resized_bbox_area",
    "approx_patch_coverage", "bbox_touches_image_boundary", "variant_clipped", "visibility",
    "laplacian_variance", "gradient_energy", "gradient_orientation_proxy_deg",
    "orientation_coherence", "mean_luminance", "luminance_std",
    "bbox_laplacian_variance", "bbox_gradient_energy", "bbox_orientation_proxy_deg",
    "bbox_orientation_coherence", "max_bbox_iou",
    "overlap_count", "nearest_center_distance_normalized", "neighbor_count_near",
    "neighbor_count_wide", "track_observation_count", "track_duration_frames",
    "track_age_frames", "track_remaining_frames", "track_age_fraction", "manual_blur",
    "manual_occlusion", "manual_pose_deg", "manual_notes",
)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SIGNAL_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def read_signal_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"H2 observation signals do not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SIGNAL_FIELDS:
            raise RuntimeError(f"Unexpected H2 signal schema: {path}")
        return list(reader)


def _luminance(image: Image.Image) -> np.ndarray:
    values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]


def image_quality_signals(image: Image.Image) -> dict[str, float]:
    gray = _luminance(image)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return {
            "laplacian_variance": 0.0,
            "gradient_energy": 0.0,
            "gradient_orientation_proxy_deg": 0.0,
            "orientation_coherence": 0.0,
            "mean_luminance": float(gray.mean()) if gray.size else 0.0,
            "luminance_std": float(gray.std()) if gray.size else 0.0,
        }
    laplacian = (
        4.0 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    gradient_y, gradient_x = np.gradient(gray)
    jxx = float(np.mean(gradient_x * gradient_x))
    jyy = float(np.mean(gradient_y * gradient_y))
    jxy = float(np.mean(gradient_x * gradient_y))
    energy = jxx + jyy
    discriminant = math.sqrt(max(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy))
    orientation = (0.5 * math.degrees(math.atan2(2.0 * jxy, jxx - jyy))) % 180.0
    return {
        "laplacian_variance": float(np.var(laplacian)),
        "gradient_energy": energy,
        "gradient_orientation_proxy_deg": orientation,
        "orientation_coherence": discriminant / (energy + 1e-12),
        "mean_luminance": float(gray.mean()),
        "luminance_std": float(gray.std()),
    }


def _iou(left: Observation, right: Observation) -> float:
    x1 = max(left.bbox_x1, right.bbox_x1)
    y1 = max(left.bbox_y1, right.bbox_y1)
    x2 = min(left.bbox_x2, right.bbox_x2)
    y2 = min(left.bbox_y2, right.bbox_y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.bbox_area + right.bbox_area - intersection
    return intersection / union if union > 0 else 0.0


def spatial_context_signals(
    query: Observation,
    frame_observations: Sequence[Observation],
    radius_multipliers: Sequence[float],
) -> dict[str, float | int]:
    others = [item for item in frame_observations if item.observation_id != query.observation_id]
    overlaps = [_iou(query, item) for item in others]
    diagonal = math.hypot(query.original_width, query.original_height)
    distances = [math.hypot(item.center_x - query.center_x, item.center_y - query.center_y) for item in others]
    normalized = [distance / max(diagonal, 1e-12) for distance in distances]
    counts = [sum(distance <= multiplier for distance in normalized) for multiplier in radius_multipliers]
    return {
        "max_bbox_iou": max(overlaps, default=0.0),
        "overlap_count": sum(value > 0 for value in overlaps),
        "nearest_center_distance_normalized": min(normalized, default=float("inf")),
        "neighbor_count_near": counts[0],
        "neighbor_count_wide": counts[-1],
    }


def _manual_annotations(path: Path | None, valid_ids: set[str]) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"observation_id"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"Manual annotation CSV requires observation_id: {path}")
        result: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            observation_id = (row.get("observation_id") or "").strip()
            if observation_id not in valid_ids:
                raise RuntimeError(
                    f"{path}:{line_number}: unknown observation_id {observation_id!r}"
                )
            if observation_id in result:
                raise RuntimeError(f"{path}:{line_number}: duplicate observation_id {observation_id}")
            values = {
                "manual_blur": (row.get("manual_blur") or "").strip(),
                "manual_occlusion": (row.get("manual_occlusion") or "").strip(),
                "manual_pose_deg": (row.get("manual_pose_deg") or "").strip(),
                "manual_notes": (row.get("manual_notes") or "").strip(),
            }
            for key in ("manual_blur", "manual_occlusion"):
                if values[key] and not 0.0 <= float(values[key]) <= 1.0:
                    raise RuntimeError(f"{path}:{line_number}: {key} must be in [0,1]")
            if values["manual_pose_deg"] and not 0.0 <= float(values["manual_pose_deg"]) < 360.0:
                raise RuntimeError(f"{path}:{line_number}: manual_pose_deg must be in [0,360)")
            result[observation_id] = values
        return result


def _track_metadata(observations: Sequence[Observation]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[Observation]] = defaultdict(list)
    for item in observations:
        groups[item.identity].append(item)
    result: dict[str, dict[str, float | int]] = {}
    for values in groups.values():
        ordered = sorted(values, key=lambda item: (item.frame, item.observation_id))
        first = ordered[0].frame
        last = ordered[-1].frame
        duration = last - first + 1
        for item in ordered:
            age = item.frame - first
            result[item.observation_id] = {
                "track_observation_count": len(ordered),
                "track_duration_frames": duration,
                "track_age_frames": age,
                "track_remaining_frames": last - item.frame,
                "track_age_fraction": age / max(1, duration - 1),
            }
    return result


def _variant_signature(
    config: ExperimentConfig, observation_ids: Sequence[str]
) -> dict[str, Any]:
    h2 = require_h2(config)
    from ..utils import sha256_file

    return {
        "format_version": 1,
        "signal_implementation": "beeid.h2.signals:v1",
        "manifest_sha256": sha256_file(config.h2_manifest_path),
        "observation_ids_sha256": observation_ids_sha256(observation_ids),
        "variants": [asdict(item) for item in h2.variants],
        "patch_size": h2.patch_size,
        "density_radius_multipliers": list(h2.density_radius_multipliers),
        "manual_annotations_csv": (
            str(h2.manual_annotations_csv) if h2.manual_annotations_csv is not None else None
        ),
        "manual_annotations_sha256": (
            sha256_file(h2.manual_annotations_csv)
            if h2.manual_annotations_csv is not None
            else None
        ),
    }


def build_observation_signals(config: ExperimentConfig) -> dict[str, Any]:
    h2 = require_h2(config)
    observations, input_audit = validate_h2_inputs(config)
    output_path = config.paths.output_root / "h2_observation_signals.csv"
    metadata_path = config.paths.output_root / "h2_signal_metadata.json"
    signature = _variant_signature(config, [item.observation_id for item in observations])
    fingerprint = sha256_text(canonical_json(signature))
    if output_path.is_file() and metadata_path.is_file():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed.get("fingerprint") == fingerprint:
            rows = read_signal_rows(output_path)
            if len(rows) == len(observations) * len(h2.variants):
                return {**observed, "status": "reused", "row_count": len(rows)}

    by_frame: dict[tuple[str, int], list[Observation]] = defaultdict(list)
    for item in observations:
        by_frame[(item.video_id, item.frame)].append(item)
    track = _track_metadata(observations)
    manual = _manual_annotations(
        h2.manual_annotations_csv, {item.observation_id for item in observations}
    )
    rows: list[dict[str, Any]] = []
    for item in observations:
        spatial = spatial_context_signals(
            item,
            by_frame[(item.video_id, item.frame)],
            h2.density_radius_multipliers,
        )
        boundary = (
            item.bbox_x1 <= 0
            or item.bbox_y1 <= 0
            or item.bbox_x2 >= item.image_width
            or item.bbox_y2 >= item.image_height
        )
        annotation = manual.get(
            item.observation_id,
            {"manual_blur": "", "manual_occlusion": "", "manual_pose_deg": "", "manual_notes": ""},
        )
        bbox_measurement_variant = H2VariantConfig(
            name="bbox_quality_measurement",
            crop_expansion=0.0,
            input_size=64,
            pixel_view="raw",
            blur_radius=0.0,
        )
        bbox_crop = load_variant_crop(
            config.paths.bee24_root,
            item,
            bbox_measurement_variant,
            patch_size=h2.patch_size,
        )
        try:
            resized_bbox_crop = bbox_crop.resize((64, 64), Image.Resampling.BILINEAR)
            try:
                bbox_quality_raw = image_quality_signals(resized_bbox_crop)
            finally:
                resized_bbox_crop.close()
        finally:
            bbox_crop.close()
        bbox_quality = {
            "bbox_laplacian_variance": bbox_quality_raw["laplacian_variance"],
            "bbox_gradient_energy": bbox_quality_raw["gradient_energy"],
            "bbox_orientation_proxy_deg": bbox_quality_raw["gradient_orientation_proxy_deg"],
            "bbox_orientation_coherence": bbox_quality_raw["orientation_coherence"],
        }
        for variant in h2.variants:
            geometry = variant_geometry(item, variant, patch_size=h2.patch_size)
            crop = load_variant_crop(
                config.paths.bee24_root, item, variant, patch_size=h2.patch_size
            )
            try:
                quality = image_quality_signals(crop)
            finally:
                crop.close()
            rows.append(
                {
                    "observation_id": item.observation_id,
                    "split": item.split,
                    "video_id": item.video_id,
                    "frame_id": item.frame,
                    "track_id": item.track_id,
                    "identity": item.identity,
                    "variant": variant.name,
                    "pixel_view": variant.pixel_view,
                    "crop_expansion": variant.crop_expansion,
                    "input_size": variant.input_size,
                    "bbox_area": item.bbox_area,
                    "bbox_width": item.original_width,
                    "bbox_height": item.original_height,
                    "aspect_ratio": item.original_width / item.original_height,
                    "crop_width": geometry["crop_width"],
                    "crop_height": geometry["crop_height"],
                    "foreground_fraction": geometry["foreground_fraction"],
                    "context_fraction": geometry["context_fraction"],
                    "resized_bbox_width": geometry["resized_bbox_width"],
                    "resized_bbox_height": geometry["resized_bbox_height"],
                    "resized_bbox_area": geometry["resized_bbox_area"],
                    "approx_patch_coverage": geometry["approx_patch_coverage"],
                    "bbox_touches_image_boundary": boundary,
                    "variant_clipped": geometry["variant_clipped"],
                    "visibility": "" if item.visibility is None else item.visibility,
                    **quality,
                    **bbox_quality,
                    **spatial,
                    **track[item.observation_id],
                    **annotation,
                }
            )
    rows.sort(key=lambda row: (row["variant"], row["video_id"], row["frame_id"], row["track_id"]))
    _write_csv(output_path, rows)
    metadata = {
        "status": "completed",
        "fingerprint": fingerprint,
        "signature": signature,
        "row_count": len(rows),
        "observation_count": len(observations),
        "input_audit": input_audit,
        "definitions": {
            "laplacian_variance": "variance of a 4-neighbour discrete grayscale Laplacian; higher is sharper, not a calibrated blur label",
            "bbox_laplacian_variance": "the same proxy on the raw 0%-context GT rectangle after deterministic 64x64 bilinear resize; used for cross-observation reliability bins",
            "gradient_orientation_proxy_deg": "dominant structure-tensor orientation modulo 180 degrees; an image proxy, not bee pose ground truth",
            "max_bbox_iou": "maximum original-GT-box IoU with another identity in the same frame; an overlap proxy, not occlusion ground truth",
            "neighbor_counts": "same-frame centers within the first/last configured multiples of the query bbox diagonal",
            "bbox_foreground_only": "rectangular GT-box pixels retained and surrounding context neutralized; not a segmentation mask",
            "context_only": "rectangular GT-box pixels neutralized while expanded surrounding context is retained",
            "approx_patch_coverage": "resized rectangular bbox area divided by patch_size squared; only a literal token-area proxy for patch-based models",
        },
        "outcome_blind": True,
    }
    atomic_write_json(metadata_path, metadata)
    return metadata
