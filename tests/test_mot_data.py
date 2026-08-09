from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from beeid.data.manifest import MANIFEST_FIELDS, build_manifest, load_manifest, resolved_video_splits
from beeid.data.mot import MotDataError, expanded_crop_box, parse_mot_row, read_seqinfo, row_to_observation


def test_parse_mot_columns_and_one_based_conversion(tmp_path, config_factory, video_factory):
    config = config_factory(tmp_path)
    video = video_factory(config.paths.bee24_root, "train", "v1")
    path = video / "gt" / "gt.txt"
    row = parse_mot_row("1,7,11.5,21.5,20,10,0.7,3,0.8,99", path, 1)
    observation = row_to_observation(
        row, read_seqinfo(video, "train"), "validation", config.paths.bee24_root, "one", 0.2, 0.0
    )
    assert (row.frame, row.track_id, row.object_class, row.visibility, row.extra_columns) == (1, 7, 3, 0.8, (99.0,))
    assert (
        observation.bbox_x1, observation.bbox_y1, observation.bbox_x2, observation.bbox_y2
    ) == (10.5, 20.5, 30.5, 30.5)
    assert observation.frame_id == 1
    assert observation.original_width == 20
    assert observation.crop_expansion == 0.2
    for required in (
        "video_id", "frame_id", "track_id", "image_path", "x1", "y1", "x2", "y2",
        "original_width", "original_height", "bbox_area", "crop_expansion", "crop_clipped",
    ):
        assert required in MANIFEST_FIELDS
    assert observation.identity == "v1:7"
    assert observation.observation_id == "validation:v1:000001:7"


def test_missing_declared_image_directory_uses_unique_canonical_fallback(
    tmp_path, config_factory, video_factory
):
    config = config_factory(tmp_path)
    video = video_factory(config.paths.bee24_root, "train", "layout-mismatch")
    seqinfo = video / "seqinfo.ini"
    seqinfo.write_text(
        seqinfo.read_text(encoding="utf-8").replace("imDir=img1", "imDir=images"),
        encoding="utf-8",
    )

    sequence = read_seqinfo(video, "train")
    assert sequence.declared_image_directory == "images"
    assert sequence.image_directory == video / "img1"
    assert sequence.image_directory_fallback_used

    build_manifest(config)
    stats = json.loads(config.manifest_stats_path.read_text(encoding="utf-8"))
    assert stats["image_directory_fallback_count"] == 1
    assert stats["image_directory_fallbacks"][0]["video_id"] == "layout-mismatch"
    assert stats["image_directory_fallbacks"][0]["declared_imDir"] == "images"


def test_missing_declared_image_directory_rejects_ambiguous_fallback(
    tmp_path, config_factory, video_factory
):
    config = config_factory(tmp_path)
    video = video_factory(config.paths.bee24_root, "train", "ambiguous-layout")
    (video / "images").mkdir()
    seqinfo = video / "seqinfo.ini"
    seqinfo.write_text(
        seqinfo.read_text(encoding="utf-8").replace("imDir=img1", "imDir=missing"),
        encoding="utf-8",
    )
    with pytest.raises(MotDataError, match="fallback is ambiguous"):
        read_seqinfo(video, "train")


def test_missing_seqinfo_can_be_inferred_from_validated_images(
    tmp_path, config_factory, video_factory
):
    config = config_factory(
        tmp_path,
        dataset={"missing_seqinfo_policy": "infer_from_images"},
    )
    video = video_factory(config.paths.bee24_root, "train", "missing-seqinfo")
    (video / "seqinfo.ini").unlink()

    sequence = read_seqinfo(video, "train", "infer_from_images")
    assert sequence.metadata_source == "inferred_from_images"
    assert sequence.image_directory == video / "img1"
    assert sequence.sequence_length == 4
    assert (sequence.image_width, sequence.image_height) == (100, 80)
    assert sequence.frame_rate == 0
    assert sequence.image_count == 4
    assert sequence.image_inventory_sha256

    build_manifest(config)
    audit = json.loads(config.sequence_metadata_audit_path.read_text(encoding="utf-8"))
    assert audit["missing_seqinfo_policy"] == "infer_from_images"
    assert audit["inferred_sequence_count"] == 1
    assert audit["sequences"][0]["video_id"] == "missing-seqinfo"
    assert audit["sequences"][0]["seqinfo_sha256"] is None
    stats = json.loads(config.manifest_stats_path.read_text(encoding="utf-8"))
    assert stats["inferred_sequence_count"] == 1


def test_missing_seqinfo_inference_rejects_inconsistent_image_dimensions(
    tmp_path, config_factory, video_factory
):
    config = config_factory(tmp_path)
    video = video_factory(config.paths.bee24_root, "train", "bad-size")
    (video / "seqinfo.ini").unlink()
    Image.new("RGB", (99, 80)).save(video / "img1" / "000004.jpg")
    with pytest.raises(MotDataError, match="inconsistent image size"):
        read_seqinfo(video, "train", "infer_from_images")


def test_missing_seqinfo_is_strict_by_default(tmp_path, config_factory, video_factory):
    config = config_factory(tmp_path)
    video = video_factory(config.paths.bee24_root, "train", "strict-missing")
    (video / "seqinfo.ini").unlink()
    with pytest.raises(MotDataError, match="Missing seqinfo.ini"):
        build_manifest(config)


@pytest.mark.parametrize("line", ["1,2,3", "x,2,1,1,2,2", "1,2,1,1,nan,2", "1.5,2,1,1,2,2"])
def test_parse_mot_rejects_bad_rows(tmp_path, line):
    with pytest.raises(MotDataError):
        parse_mot_row(line, tmp_path / "gt.txt", 4)


def test_expansion_floor_ceil_and_clipping():
    assert expanded_crop_box(10, 20, 30, 40, 100, 100, 0.2) == (8, 18, 32, 42, False)
    assert expanded_crop_box(-1.2, 2, 8.2, 10, 10, 10, 0.2) == (0, 1, 10, 10, True)


def test_manifest_split_is_video_level_deterministic_and_composite_identity(tmp_path, config_factory, video_factory):
    config = config_factory(tmp_path)
    for index in range(5):
        video_factory(config.paths.bee24_root, "train", f"video-{index}")
    first, metadata = resolved_video_splits(config)
    second, _ = resolved_video_splits(config)
    assert first == second
    assert metadata["validation_count"] == 1
    observations = build_manifest(config)
    assert config.manifest_path.is_file()
    assert config.resolved_split_path.is_file()
    loaded = load_manifest(config.manifest_path, valid_only=True)
    assert [item.observation_id for item in loaded] == [item.observation_id for item in observations if item.valid]
    assert all(item.identity == f"{item.video_id}:{item.track_id}" for item in loaded)
    assert len({item.identity for item in loaded if item.track_id == 1}) == 5


def test_invalid_box_errors_or_is_recorded_as_skipped(tmp_path, config_factory, video_factory):
    rows = ["1,1,20,20,0,5,1,1,1"]
    config = config_factory(tmp_path)
    video_factory(config.paths.bee24_root, "train", "bad", rows=rows)
    with pytest.raises(MotDataError, match="non_positive_area"):
        build_manifest(config)

    skip_root = tmp_path / "skip"
    skip_config = config_factory(skip_root, dataset={"invalid_bbox_policy": "skip"})
    video_factory(skip_config.paths.bee24_root, "train", "bad", rows=rows)
    observations = build_manifest(skip_config)
    assert "non_positive_area" in observations[0].skip_reason
    assert load_manifest(skip_config.manifest_path, valid_only=True) == []


def test_duplicate_identity_in_frame_is_rejected(tmp_path, config_factory, video_factory):
    config = config_factory(tmp_path)
    video_factory(config.paths.bee24_root, "train", "dup", rows=[
        "1,2,10,10,10,10,1,1,1", "1,2,20,20,10,10,1,1,1",
    ])
    with pytest.raises(MotDataError, match="Duplicate identity"):
        build_manifest(config)
    audit = json.loads(config.duplicate_identity_audit_path.read_text(encoding="utf-8"))
    assert audit["conflict_key_count"] == 1
    assert audit["excluded_source_row_count"] == 0
    assert [item["line_number"] for item in audit["conflicts"][0]["rows"]] == [1, 2]


def test_duplicate_identity_conflict_can_be_excluded_with_full_audit(
    tmp_path, config_factory, video_factory
):
    config = config_factory(
        tmp_path,
        dataset={"duplicate_identity_policy": "exclude_conflict"},
    )
    video_factory(config.paths.bee24_root, "train", "dup", rows=[
        "1,2,10,10,10,10,1,1,1",
        "1,2,20,20,10,10,1,1,1",
        "2,2,12,12,10,10,1,1,1",
        "2,3,40,20,10,10,1,1,1",
    ])
    observations = build_manifest(config)
    assert [(item.frame, item.track_id) for item in observations] == [(2, 2), (2, 3)]

    audit = json.loads(config.duplicate_identity_audit_path.read_text(encoding="utf-8"))
    assert audit["policy"] == "exclude_conflict"
    assert audit["conflict_key_count"] == 1
    assert audit["excluded_source_row_count"] == 2
    assert audit["excluded_source_row_fraction"] == 0.5
    assert audit["affected_sequences"] == ["train/dup"]
    assert [item["line_number"] for item in audit["conflicts"][0]["rows"]] == [1, 2]

    stats = json.loads(config.manifest_stats_path.read_text(encoding="utf-8"))
    assert stats["source_observation_count_before_duplicate_filter"] == 4
    assert stats["observation_count"] == 2
    assert stats["skipped_manifest_observation_count"] == 0
    assert stats["skipped_observation_count"] == 2
    assert stats["duplicate_identity_conflict_key_count"] == 1
    assert stats["skip_reasons"]["duplicate_identity_conflict"] == 2


def test_partial_out_of_bounds_clips_but_fully_outside_errors(tmp_path, config_factory, video_factory):
    config = config_factory(tmp_path)
    video_factory(config.paths.bee24_root, "train", "clip", rows=["1,1,-5,5,20,10,1,1,1"])
    item = build_manifest(config)[0]
    assert item.clipped and item.crop_x0 == 0

    other = tmp_path / "outside"
    other_config = config_factory(other)
    video_factory(other_config.paths.bee24_root, "train", "outside", rows=["1,1,200,5,20,10,1,1,1"])
    with pytest.raises(MotDataError, match="fully_out_of_bounds"):
        build_manifest(other_config)
