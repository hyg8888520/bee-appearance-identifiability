from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from beeid.config import load_config
from beeid.data.mot import Observation
from beeid.h25.core import EXPECTED_STRATEGIES, read_h25_protocol
from beeid.h25.experiment import ema_update, empirical_percentiles, nearest_same_frame_negative
from beeid.h25.statistics import cluster_bootstrap
from beeid.h25.synthetic import h25_synthetic_smoke
from beeid.h3.protocol import FrozenProtocolError, validate_h3_protocol
from beeid.utils import sha256_file


REPOSITORY = Path(__file__).resolve().parents[1]


def _observation(observation_id: str, identity: str, center_x: float) -> Observation:
    video_id, track_id = identity.split(":")
    return Observation(
        observation_id=observation_id, split="validation", source_split="train",
        video_id=video_id, track_id=int(track_id), identity=identity, frame=10, frame_id=10,
        image_path="train/v/img1/000010.jpg", image_width=100, image_height=80,
        original_width=10.0, original_height=10.0, raw_x=center_x, raw_y=10.0,
        raw_w=10.0, raw_h=10.0, bbox_x1=center_x, bbox_y1=9.0,
        bbox_x2=center_x + 10.0, bbox_y2=19.0, x1=int(center_x), y1=9,
        x2=int(center_x) + 10, y2=19, crop_expansion=0.2, crop_clipped=False,
        center_x=center_x, center_y=14.0, bbox_area=100.0, confidence=1.0,
        object_class=1, visibility=1.0, skip_reason="", extra_columns="[]",
    )


def test_h3_frozen_protocol_checksum_and_tamper_detection(tmp_path):
    source = REPOSITORY / "configs" / "h3_protocol.lock.yaml"
    config_dir = tmp_path / "configs"
    (config_dir / "splits").mkdir(parents=True)
    protocol = config_dir / "h3_protocol.lock.yaml"
    protocol.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    project_split = REPOSITORY / "configs" / "splits" / "project_split.yaml"
    (config_dir / "splits" / "project_split.yaml").write_text(
        project_split.read_text(encoding="utf-8"), encoding="utf-8"
    )
    checksum = protocol.with_suffix(".sha256")
    checksum.write_text(f"{sha256_file(protocol)}  {protocol.name}\n", encoding="utf-8")
    audit = validate_h3_protocol(protocol)
    assert audit["status"] == "valid"
    assert audit["final_test_access"] is False
    protocol.write_text(protocol.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(FrozenProtocolError, match="checksum mismatch"):
        validate_h3_protocol(protocol)


def test_h25_protocol_excludes_static_rules_and_has_exact_strategies():
    protocol = read_h25_protocol(REPOSITORY / "configs" / "h25_protocol.lock.yaml")
    assert protocol["event_selection"]["outcome_blind"] is True
    assert protocol["event_selection"]["excluded_static_signals"] == [
        "bbox_area", "bbox_laplacian_variance"
    ]
    assert tuple(protocol["memory"]["strategies"]) == EXPECTED_STRATEGIES
    assert protocol["source"]["final_test_access"] is False


def test_percentiles_ema_and_nearest_negative_contracts():
    assert empirical_percentiles([1.0, 2.0, 2.0, 4.0]) == [0.125, 0.5, 0.5, 0.875]
    template = np.asarray([1.0, 0.0], dtype=np.float32)
    feature = np.asarray([0.0, 1.0], dtype=np.float32)
    weighted = ema_update(template, feature, 0.05)
    assert weighted.dtype == np.float32
    assert np.linalg.norm(weighted) == pytest.approx(1.0)
    observations = [
        _observation("q", "v:1", 10.0),
        _observation("near", "v:2", 12.0),
        _observation("far", "v:3", 40.0),
        _observation("same", "v:1", 11.0),
    ]
    assert nearest_same_frame_negative(0, [0, 1, 2, 3], observations) == 1


def test_cluster_bootstrap_is_paired_clustered_and_deterministic():
    rows = [
        {
            "model": "m", "event_type": "history_outlier", "strategy": "skip_update",
            "window": "immediate", "video_id": f"v{index % 2}", "identity": f"v{index % 2}:{index}",
            "rank1_gain": value, "similarity_gap_reduction": value / 2,
            "induced_error_reduction": value,
        }
        for index, value in enumerate([1.0, 0.0, 1.0, -1.0])
    ]
    first = cluster_bootstrap(rows, replicates=50, seed=24)
    second = cluster_bootstrap(rows, replicates=50, seed=24)
    assert first == second
    assert {row["cluster_unit"] for row in first} == {"video_id", "identity"}
    assert {row["metric"] for row in first} == {
        "rank1_gain", "similarity_gap_reduction", "induced_error_reduction"
    }


def test_h25_examples_are_strict_and_new_scripts_do_not_use_bare_python():
    config = load_config(REPOSITORY / "configs" / "h25.example.yaml")
    assert config.h25 is not None
    assert config.paths.h2_output_root is not None
    assert config.h25.allow_subset is False
    for name in ("h25_smoke_test.sh", "run_h25.sh", "resume_h25.sh"):
        text = (REPOSITORY / "scripts" / name).read_text(encoding="utf-8")
        assert "python -m beeid" not in text
        assert '"${PYTHON_BIN}" -m beeid.cli' in text
        assert "dinov3_vits16" not in text


def test_h25_synthetic_smoke_writes_four_strategy_paired_outputs(tmp_path):
    output = tmp_path / "h25-output"
    result = h25_synthetic_smoke(output)
    assert result["status"] == "passed"
    assert result["test_only_encoder"] is True
    assert result["event_count"] > 0
    assert result["metadata_status"] == "SERVER_VALIDATION_PENDING"
    with (output / "h25_strategy_trajectories.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert {row["strategy"] for row in rows} == set(EXPECTED_STRATEGIES)
    assert (output / "h25_cluster_bootstrap.csv").stat().st_size > 0
    assert (output / "h25_figures" / "index.txt").is_file()
