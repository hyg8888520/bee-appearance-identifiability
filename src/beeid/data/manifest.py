"""Build and read stable observation manifests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image

from ..config import ExperimentConfig
from ..utils import atomic_write_json, atomic_write_text, count_reasons, sha256_file
from .mot import MotDataError, Observation, parse_mot_row, read_seqinfo, row_to_observation

MANIFEST_FIELDS = tuple(field.name for field in fields(Observation))
_INT_FIELDS = {
    "track_id", "frame", "frame_id", "image_width", "image_height", "x1", "y1", "x2", "y2",
}
_FLOAT_FIELDS = {
    "original_width", "original_height", "raw_x", "raw_y", "raw_w", "raw_h", "bbox_x1",
    "bbox_y1", "bbox_x2", "bbox_y2", "crop_expansion", "center_x", "center_y", "bbox_area",
    "confidence", "visibility",
}

_RawIdentityKey = tuple[str, str, int, int]


def _scan_duplicate_identities(
    config: ExperimentConfig,
    assignments: dict[str, str],
) -> tuple[set[_RawIdentityKey], dict[str, object]]:
    """Pre-scan every selected GT row before choosing how to handle conflicts."""
    first_rows: dict[_RawIdentityKey, dict[str, object]] = {}
    conflicts: dict[_RawIdentityKey, list[dict[str, object]]] = {}
    gt_sources: list[dict[str, object]] = []
    selected_row_count = 0

    for key, resolved_split in sorted(assignments.items()):
        source_split, video_id = key.split("/", 1)
        ground_truth = config.paths.bee24_root / source_split / video_id / "gt" / "gt.txt"
        if not ground_truth.is_file():
            raise MotDataError(f"Missing ground truth: {ground_truth}")
        sequence_row_count = 0
        for line_number, line in enumerate(
            ground_truth.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = parse_mot_row(line, ground_truth, line_number)
            if (
                config.dataset.max_frames_per_video is not None
                and row.frame > config.dataset.max_frames_per_video
            ):
                continue
            selected_row_count += 1
            sequence_row_count += 1
            identity_key = (source_split, video_id, row.frame, row.track_id)
            detail: dict[str, object] = {
                "source_split": source_split,
                "resolved_split": resolved_split,
                "video_id": video_id,
                "frame": row.frame,
                "track_id": row.track_id,
                "gt_path": str(ground_truth),
                "line_number": line_number,
                "raw_line": line,
                "raw_xywh": [row.raw_x, row.raw_y, row.raw_w, row.raw_h],
                "confidence": row.confidence,
                "object_class": row.object_class,
                "visibility": row.visibility,
            }
            if identity_key in first_rows:
                if identity_key not in conflicts:
                    conflicts[identity_key] = [first_rows[identity_key]]
                conflicts[identity_key].append(detail)
            else:
                first_rows[identity_key] = detail
        gt_sources.append(
            {
                "source_split": source_split,
                "video_id": video_id,
                "gt_path": str(ground_truth),
                "gt_sha256": sha256_file(ground_truth),
                "selected_row_count": sequence_row_count,
            }
        )

    conflict_details: list[dict[str, object]] = []
    for identity_key in sorted(conflicts):
        source_split, video_id, frame, track_id = identity_key
        conflict_details.append(
            {
                "source_split": source_split,
                "video_id": video_id,
                "frame": frame,
                "track_id": track_id,
                "rows": conflicts[identity_key],
            }
        )
    excluded_row_count = sum(len(item["rows"]) for item in conflict_details)  # type: ignore[arg-type]
    conflict_keys = set(conflicts)
    report: dict[str, object] = {
        "format_version": 1,
        "policy": config.dataset.duplicate_identity_policy,
        "scan_scope": "all selected source GT rows before manifest construction",
        "identity_key": ["source_split", "video_id", "frame", "track_id"],
        "selected_source_row_count": selected_row_count,
        "conflict_key_count": len(conflict_keys),
        "excluded_source_row_count": (
            excluded_row_count
            if config.dataset.duplicate_identity_policy == "exclude_conflict"
            else 0
        ),
        "excluded_source_row_fraction": (
            excluded_row_count / selected_row_count
            if selected_row_count and config.dataset.duplicate_identity_policy == "exclude_conflict"
            else 0.0
        ),
        "affected_sequences": sorted({f"{key[0]}/{key[1]}" for key in conflict_keys}),
        "action": (
            "excluded every row belonging to each conflicting identity key"
            if config.dataset.duplicate_identity_policy == "exclude_conflict"
            else "none; strict error policy"
        ),
        "gt_sources": gt_sources,
        "conflicts": conflict_details,
    }
    return conflict_keys, report


def resolved_video_splits(config: ExperimentConfig) -> tuple[dict[str, str], dict[str, object]]:
    available: dict[str, list[str]] = {}
    for source_split in config.dataset.source_splits:
        split_directory = config.paths.bee24_root / source_split
        if not split_directory.is_dir():
            raise MotDataError(f"Missing BEE24 split directory: {split_directory}")
        names = sorted(path.name for path in split_directory.iterdir() if path.is_dir())
        if config.dataset.video_ids:
            requested = set(config.dataset.video_ids)
            names = [name for name in names if name in requested]
        if config.dataset.max_videos is not None:
            names = names[: config.dataset.max_videos]
        available[source_split] = names
    train = available.get("train", [])
    validation_count = max(1, round(config.dataset.validation_fraction * len(train))) if train else 0
    ordered = sorted(
        train,
        key=lambda video_id: (
            hashlib.sha256(f"{config.runtime.seed}:{video_id}".encode("utf-8")).hexdigest(), video_id
        ),
    )
    validation = set(ordered[:validation_count])
    assignments: dict[str, str] = {}
    for source_split, names in available.items():
        for video_id in names:
            assignments[f"{source_split}/{video_id}"] = (
                "validation" if source_split == "train" and video_id in validation else source_split
            )
    metadata: dict[str, object] = {
        "algorithm": "sort by SHA256(f'{seed}:{video_id}') and select max(1, round(fraction*n_train))",
        "seed": config.runtime.seed,
        "validation_fraction": config.dataset.validation_fraction,
        "validation_count": validation_count,
        "train_videos": train,
        "validation_videos": sorted(validation),
        "test_used_for_selection": False,
        "assignments": assignments,
    }
    return assignments, metadata


def _validate_image(path: Path, width: int, height: int, deep: bool) -> None:
    if not path.is_file():
        raise MotDataError(f"Missing frame image: {path}")
    with Image.open(path) as image:
        if image.size != (width, height):
            raise MotDataError(f"Image size mismatch for {path}: {image.size}, expected {(width, height)}")
        if deep:
            image.verify()


def build_manifest(config: ExperimentConfig, *, validate_only: bool = False) -> list[Observation]:
    root = config.paths.bee24_root
    if not root.is_dir():
        raise MotDataError(f"BEE24 root does not exist: {root}")
    run_state_path = config.paths.output_root / "run_state.json"
    if not validate_only and not run_state_path.exists():
        atomic_write_json(
            run_state_path,
            {"benchmark_started_at_utc": datetime.now(timezone.utc).isoformat()},
        )
    assignments, split_metadata = resolved_video_splits(config)
    duplicate_keys, duplicate_audit = _scan_duplicate_identities(config, assignments)
    if not validate_only:
        atomic_write_json(config.duplicate_identity_audit_path, duplicate_audit)
    if duplicate_keys and config.dataset.duplicate_identity_policy == "error":
        first_conflict = duplicate_audit["conflicts"][0]  # type: ignore[index]
        rows = first_conflict["rows"]  # type: ignore[index]
        locations = ", ".join(
            f"{item['gt_path']}:{item['line_number']}" for item in rows  # type: ignore[union-attr]
        )
        raise MotDataError(
            "Duplicate identity in frame: "
            f"{first_conflict['source_split']}:{first_conflict['video_id']}:"  # type: ignore[index]
            f"{first_conflict['frame']:06d}:{first_conflict['track_id']}; "  # type: ignore[index]
            f"source rows: {locations}; total conflict keys: {len(duplicate_keys)}"
        )
    observations: list[Observation] = []
    seen_ids: set[str] = set()
    for key, split in sorted(assignments.items()):
        source_split, video_id = key.split("/", 1)
        sequence = read_seqinfo(root / source_split / video_id, source_split)
        ground_truth = sequence.directory / "gt" / "gt.txt"
        if not ground_truth.is_file():
            raise MotDataError(f"Missing ground truth: {ground_truth}")
        checked_images: set[int] = set()
        for line_number, line in enumerate(ground_truth.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            row = parse_mot_row(line, ground_truth, line_number)
            if config.dataset.max_frames_per_video is not None and row.frame > config.dataset.max_frames_per_video:
                continue
            raw_identity_key = (source_split, video_id, row.frame, row.track_id)
            if raw_identity_key in duplicate_keys:
                continue
            observation = row_to_observation(
                row,
                sequence,
                split,
                root,
                config.dataset.bbox_origin,
                config.protocol.crop_expansion,
                config.dataset.confidence_min,
            )
            invalid = [
                reason for reason in observation.skip_reason.split(";")
                if reason in {"frame_out_of_range", "non_positive_area", "fully_out_of_bounds", "empty_crop_after_clipping"}
            ]
            if invalid and config.dataset.invalid_bbox_policy == "error":
                raise MotDataError(
                    f"{ground_truth}:{line_number}: invalid observation {observation.observation_id}: {','.join(invalid)}"
                )
            if observation.observation_id in seen_ids:
                raise MotDataError(f"Duplicate identity in frame: {observation.observation_id}")
            seen_ids.add(observation.observation_id)
            if observation.frame not in checked_images and not observation.skip_reason:
                _validate_image(
                    root / observation.image_path,
                    sequence.image_width,
                    sequence.image_height,
                    config.dataset.deep_validate_images,
                )
                checked_images.add(observation.frame)
            observations.append(observation)
    observations.sort(key=lambda item: (item.split, item.video_id, item.frame, item.track_id, item.observation_id))
    if not observations:
        raise MotDataError("No observations were found under the configured BEE24 root")
    if validate_only:
        return observations

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(item.to_dict() for item in observations)
    atomic_write_text(config.manifest_path, buffer.getvalue())
    split_metadata["manifest_sha256"] = sha256_file(config.manifest_path)
    atomic_write_text(config.resolved_split_path, yaml.safe_dump(split_metadata, sort_keys=False))
    valid = [item for item in observations if item.valid]
    skip_reasons = count_reasons(
        reason
        for item in observations
        for reason in item.skip_reason.split(";")
        if reason
    )
    excluded_duplicate_rows = int(duplicate_audit["excluded_source_row_count"])
    skipped_manifest_rows = len(observations) - len(valid)
    if excluded_duplicate_rows:
        skip_reasons["duplicate_identity_conflict"] = excluded_duplicate_rows
        skip_reasons = dict(sorted(skip_reasons.items()))
    atomic_write_json(
        config.manifest_stats_path,
        {
            "source_observation_count_before_duplicate_filter": duplicate_audit[
                "selected_source_row_count"
            ],
            "observation_count": len(observations),
            "valid_observation_count": len(valid),
            "skipped_manifest_observation_count": skipped_manifest_rows,
            "skipped_observation_count": skipped_manifest_rows + excluded_duplicate_rows,
            "duplicate_identity_conflict_key_count": duplicate_audit["conflict_key_count"],
            "duplicate_identity_excluded_row_count": excluded_duplicate_rows,
            "duplicate_identity_excluded_row_fraction": duplicate_audit[
                "excluded_source_row_fraction"
            ],
            "duplicate_identity_audit": str(config.duplicate_identity_audit_path),
            "skip_reasons": skip_reasons,
            "split_counts": {
                split: sum(item.split == split for item in observations)
                for split in sorted({item.split for item in observations})
            },
            "manifest_sha256": split_metadata["manifest_sha256"],
        },
    )
    return observations


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def load_manifest(path: Path, *, split: str | None = None, valid_only: bool = True) -> list[Observation]:
    if not path.is_file():
        raise MotDataError(f"Manifest does not exist: {path}")
    result: list[Observation] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise MotDataError(f"Manifest columns do not match the H1 schema: {path}")
        for row in reader:
            values: dict[str, object] = dict(row)
            for field in _INT_FIELDS:
                values[field] = int(row[field])
            for field in _FLOAT_FIELDS - {"visibility"}:
                values[field] = float(row[field])
            values["object_class"] = _optional_int(row["object_class"])
            values["visibility"] = _optional_float(row["visibility"])
            values["crop_clipped"] = row["crop_clipped"].lower() == "true"
            observation = Observation(**values)  # type: ignore[arg-type]
            if split is not None and observation.split != split:
                continue
            if valid_only and not observation.valid:
                continue
            result.append(observation)
    return result


def manifest_sha256(path: Path) -> str:
    return sha256_file(path)
