from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from beeid.config import load_config
from beeid.h3.assignment import maximum_weight_matching
from beeid.h3.metrics import summarize_gt_assignments
from beeid.h3.synthetic import h3_synthetic_smoke
from beeid.h3.thresholds import axial_difference, reliability_from_artifact
from beeid.h3.tracker import H3_VARIANTS


REPOSITORY = Path(__file__).resolve().parents[1]


def test_exact_assignment_supports_unmatched_and_validity_mask():
    scores = np.asarray([[0.9, 0.2], [0.3, 0.8]], dtype=np.float64)
    assert maximum_weight_matching(scores, minimum_score=0.5) == [(0, 0), (1, 1)]
    valid = np.asarray([[False, True], [False, False]])
    assert maximum_weight_matching(scores, valid=valid, minimum_score=0.1) == [(0, 1)]
    assert maximum_weight_matching(np.empty((0, 2))) == []


def test_axial_orientation_and_reliability_are_causal_and_bounded():
    assert axial_difference(179.0, 1.0) == pytest.approx(2.0)
    knots = [float(value) for value in np.linspace(0.0, 1.0, 101)]
    artifact = {
        "reliability": {"min_reliability": 0.05},
        "common_calibrators": {
            "bbox_scale_change": {"knots": knots},
            "orientation_change_proxy": {"knots": knots},
            "sharpness_change": {"knots": knots},
            "max_bbox_iou": {"knots": knots},
            "neighbor_count_wide": {"knots": knots},
        },
        "models": {"m": {"identity_history_outlier": {"knots": knots}}},
    }
    low, low_risks = reliability_from_artifact(
        artifact,
        "m",
        {
            "identity_history_outlier": 0.0,
            "bbox_scale_change": 0.0,
            "orientation_change_proxy": 0.0,
            "sharpness_change": 0.0,
            "max_bbox_iou": 0.0,
            "neighbor_count_wide": 0.0,
        },
    )
    high, high_risks = reliability_from_artifact(
        artifact,
        "m",
        {
            "identity_history_outlier": 2.0,
            "bbox_scale_change": 0.0,
            "orientation_change_proxy": 0.0,
            "sharpness_change": 0.0,
            "max_bbox_iou": 0.0,
            "neighbor_count_wide": 0.0,
        },
    )
    assert 0.95 <= low <= 1.0
    assert high == pytest.approx(0.05)
    assert set(low_risks) == {
        "identity_history_outlier", "bbox_scale_change",
        "orientation_change_proxy", "sharpness_change", "crowding_overlap",
    }
    assert max(high_risks.values()) == 1.0


def test_gt_box_identity_metrics_match_known_identity_switch_fixture():
    rows = []
    for frame, assignments in ((1, (("v:1", 1), ("v:2", 2))), (2, (("v:1", 2), ("v:2", 1)))):
        for index, (identity, predicted) in enumerate(assignments):
            rows.append(
                {
                    "model": "m", "variant": "baseline_association",
                    "stage": "gt_detection_boxes", "video_id": "v", "frame_id": frame,
                    "observation_id": f"v:{frame}:{index}", "gt_identity": identity,
                    "predicted_track_id": predicted, "eligible_memory_update": True,
                    "memory_update_accepted": True, "reliability": 1.0,
                    "effective_memory_alpha": 0.2, "association_score": 0.5,
                }
            )
    per_video, pooled = summarize_gt_assignments(rows)
    assert len(per_video) == len(pooled) == 1
    result = per_video[0]
    assert result["IDTP"] == 2
    assert result["IDFN"] == result["IDFP"] == 2
    assert result["IDF1"] == pytest.approx(0.5)
    assert result["AssA"] == pytest.approx(1.0 / 3.0)
    assert result["HOTA"] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert result["IDSW"] == 2
    assert result["Frag"] == 0


def test_h3_examples_and_scripts_keep_final_test_locked():
    for name, subset in (
        ("h3.example.yaml", False),
        ("h3.local.yaml.example", False),
        ("h3_smoke.example.yaml", True),
    ):
        path = REPOSITORY / "configs" / name
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = load_config(path)
        assert config.h3 is not None
        assert config.h3.allow_subset is subset
        assert config.h3.stage == "gt_detection_boxes"
        assert config.h3.project_split_path == REPOSITORY / "configs" / "splits" / "project_split.yaml"
        assert payload["dataset"]["source_splits"] == ["train"]
        assert payload["h3"]["fixed_detections_root"] is None
    for name in ("h3_smoke_test.sh", "run_h3.sh", "resume_h3.sh"):
        text = (REPOSITORY / "scripts" / name).read_text(encoding="utf-8")
        assert "python -m beeid" not in text
        assert '"${PYTHON_BIN}" -m beeid.cli' in text


def test_h3_synthetic_smoke_writes_all_variants_and_lock_metadata(tmp_path):
    result = h3_synthetic_smoke(tmp_path / "h3-synthetic")
    assert result["status"] == "passed"
    assert result["test_only_encoder"] is True
    assert result["final_test_read"] is False
    assert result["assignment_rows"] > 0
    metadata = json.loads(
        (tmp_path / "h3-synthetic" / "h3_run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["variants"] == list(H3_VARIANTS)
    assert metadata["fixed_detector_boxes"] == "SERVER_VALIDATION_PENDING"
    for name in (
        "h3_assignments.csv", "h3_per_video_metrics.csv", "h3_summary.csv",
        "h3_paired_video_metrics.csv", "h3_run_metadata.json",
    ):
        assert (tmp_path / "h3-synthetic" / name).is_file()
