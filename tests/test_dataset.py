from pathlib import Path

import pytest

from bee24_dinotrack.dataset import DatasetFormatError, ImageRecord, image_path, load_image_records


def test_image_path_is_dataset_root_join_file_name() -> None:
    root = Path("/mnt/datasets/BEE24")
    record = ImageRecord(1, "BEE24-01/img1/000001.jpg")
    assert image_path(root, record) == root / "BEE24-01/img1/000001.jpg"


def test_annotation_rejects_absolute_file_name(tmp_path: Path) -> None:
    annotation = tmp_path / "train.json"
    annotation.write_text('{"images":[{"id":1,"file_name":"/tmp/image.jpg"}]}', encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="relative to dataset.root"):
        load_image_records(annotation)


def test_annotation_rejects_windows_absolute_file_name(tmp_path: Path) -> None:
    annotation = tmp_path / "train.json"
    annotation.write_text(
        '{"images":[{"id":1,"file_name":"C:\\\\data\\\\image.jpg"}]}', encoding="utf-8"
    )
    with pytest.raises(DatasetFormatError, match="relative to dataset.root"):
        load_image_records(annotation)
