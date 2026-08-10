from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from beeid.config import H4Config, H41Config, load_config
from beeid.data.mot import Observation
from beeid.h41.protocol import (
    H41_PRIMARY_VARIANT,
    H41_VARIANTS,
    H41ProtocolError,
    validate_h41_protocol,
)
from beeid.h41.recoverability import (
    cumulative_event_timelines,
    cumulative_recoverability_decision,
    summarize_cumulative_recoverability,
)
from beeid.h41.synthetic import h41_synthetic_smoke
from beeid.h41.tracker import track_h41_sequence


ROOT = Path(__file__).resolve().parents[1]


def _observation(frame: int, track_id: int, x: float) -> Observation:
    return Observation(
        observation_id=f"validation:v:{frame:06d}:{track_id}",
        split="validation",
        source_split="train",
        video_id="v",
        track_id=track_id,
        identity=f"v:{track_id}",
        frame=frame,
        frame_id=frame,
        image_path=f"train/v/img1/{frame:06d}.jpg",
        image_width=100,
        image_height=80,
        original_width=20.0,
        original_height=20.0,
        raw_x=x + 1,
        raw_y=21.0,
        raw_w=20.0,
        raw_h=20.0,
        bbox_x1=x,
        bbox_y1=20.0,
        bbox_x2=x + 20.0,
        bbox_y2=40.0,
        x1=int(x),
        y1=18,
        x2=int(x + 20),
        y2=42,
        crop_expansion=0.2,
        crop_clipped=False,
        center_x=x + 10.0,
        center_y=30.0,
        bbox_area=400.0,
        confidence=1.0,
        object_class=1,
        visibility=1.0,
        skip_reason="",
        extra_columns="[]",
    )


def _configs(**h41_changes: object) -> tuple[H4Config, H41Config]:
    config = load_config(ROOT / "configs" / "h41.example.yaml")
    assert config.h4 is not None and config.h41 is not None
    return replace(config.h4, ambiguity_margin=1.0), replace(
        config.h41, **h41_changes
    )


def _exact_row(event: str, horizon: int, recovered: bool, available: bool = True):
    return {
        "model": "m",
        "video_id": "v",
        "event_id": event,
        "event_frame": 10,
        "gt_identity": f"v:{event}",
        "previous_predicted_track_id": "1",
        "current_predicted_track_id": "2",
        "component_identities": "v:1|v:2",
        "component_size": 2,
        "horizon": horizon,
        "available": available,
        "appearance_recoverable": recovered,
        "motion_recoverable": recovered,
        "joint_recoverable": recovered,
    }


def test_window_estimand_preserves_transient_and_late_recovery():
    rows = []
    for horizon in range(1, 11):
        rows.append(_exact_row("e1", horizon, recovered=horizon == 1, available=horizon == 1))
        rows.append(_exact_row("e2", horizon, recovered=horizon == 3))
    timelines = cumulative_event_timelines(rows, [1, 3, 5, 10])
    e1_h10 = next(
        row for row in timelines if row["event_id"] == "e1" and row["deadline"] == 10
    )
    e2_h3 = next(
        row for row in timelines if row["event_id"] == "e2" and row["deadline"] == 3
    )
    assert e1_h10["joint_recoverable_by_deadline"] is True
    assert e1_h10["joint_first_recovery_lag"] == 1
    assert e1_h10["available_lag_count"] == 1
    assert e2_h3["joint_recoverable_by_deadline"] is True
    assert e2_h3["joint_first_recovery_lag"] == 3

    summary = summarize_cumulative_recoverability(timelines, ["m"], [1, 3, 5, 10])
    h10 = next(row for row in summary if row["deadline"] == 10)
    assert h10["switch_event_count"] == 2
    assert h10["joint_recoverable_count"] == 2
    assert h10["joint_recoverable_fraction"] == 1.0
    assert h10["denominator_policy"] == "all_baseline_idsw_events"


def test_cumulative_gate_requires_every_backbone_without_dropping_missing_lags():
    rows = []
    for model, fraction in (("resnet50", 0.4), ("dinov3", 0.2)):
        rows.append(
            {
                "model": model,
                "deadline": 5,
                "switch_event_count": 20,
                "videos_with_events": 5,
                "joint_recoverable_fraction": fraction,
            }
        )
    decision = cumulative_recoverability_decision(
        rows,
        ["resnet50", "dinov3"],
        gate_deadline=5,
        min_fraction=0.3,
        min_events=10,
        min_videos=3,
    )
    assert decision["status"] == "STOP_NO_WINDOW_RECOVERABILITY_SIGNAL"
    assert decision["gate_passed"] is False
    assert [row["pass"] for row in decision["model_checks"]] == [True, False]
    assert decision["method_uses_gt"] is False


def test_adaptive_commit_is_causal_early_and_never_uses_losing_branch_memory():
    observations: list[Observation] = []
    embeddings: list[np.ndarray] = []
    for frame in range(1, 11):
        observations.extend(
            [_observation(frame, 1, 10.0), _observation(frame, 2, 12.0)]
        )
        embeddings.extend(
            [
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 1.0], dtype=np.float32),
            ]
        )
    h4, h41 = _configs(decision_margin=0.0, winner_stability_steps=2)
    rows, events = track_h41_sequence(
        "m", H41_PRIMARY_VARIANT, observations, np.stack(embeddings), h4, h41
    )
    deferred = [event for event in events if event["deferred_commit"]]
    assert deferred
    assert any(event["early_commit"] for event in deferred)
    assert all(
        int(event["decision_frame"]) <= int(event["start_frame"]) + 5
        for event in deferred
    )
    assert all(event["ground_truth_decision_input"] is False for event in deferred)
    assert all(int(row["frame_id"]) <= int(row["decision_frame"]) for row in rows)
    assert all(int(row["decision_latency_frames"]) <= 5 for row in rows)
    assert all(
        row["branch_memory_isolated"] is True
        for row in rows if row["deferred_commit"]
    )


def test_adaptive_commit_forces_deadline_when_threshold_cannot_be_met():
    observations = [
        _observation(frame, track_id, 10.0 if track_id == 1 else 12.0)
        for frame in range(1, 9)
        for track_id in (1, 2)
    ]
    embeddings = np.stack(
        [np.asarray([1.0, 0.0], dtype=np.float32) for _ in observations]
    )
    h4, h41 = _configs(decision_margin=1.0, winner_stability_steps=99)
    _, events = track_h41_sequence(
        "m", H41_PRIMARY_VARIANT, observations, embeddings, h4, h41
    )
    deferred = [event for event in events if event["deferred_commit"]]
    assert deferred
    assert all(event["early_commit"] is False for event in deferred)
    assert all(
        event["forced_at_deadline"] is True
        or event["forced_at_sequence_end"] is True
        for event in deferred
    )
    assert any(event["forced_at_deadline"] is True for event in deferred)
    assert all(event["realized_horizon"] <= 5 for event in deferred)


def test_h41_protocol_examples_scripts_and_checksum_are_locked(tmp_path):
    audit = validate_h41_protocol(
        ROOT / "configs" / "h41_protocol.lock.yaml",
        ROOT / "configs" / "h41_protocol.lock.sha256",
    )
    assert audit["final_test_access"] is False
    assert audit["design_timing"] == "DEFINED_AFTER_OBSERVING_H4_V1_DEVELOPMENT_AUDIT"
    assert audit["primary_variant"] == H41_PRIMARY_VARIANT
    assert tuple(audit["variants"]) == H41_VARIANTS
    for name, subset in (
        ("h41.example.yaml", False),
        ("h41.local.yaml.example", False),
        ("h41_smoke.example.yaml", True),
    ):
        payload = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        config = load_config(ROOT / "configs" / name)
        assert config.h4 is not None and config.h41 is not None
        assert config.h4.allow_subset is subset
        assert config.h41.allow_subset is subset
        assert payload["paths"]["h4_output_root"] != payload["paths"]["output_root"]
        assert payload["h41"]["audit_horizons"] == list(range(1, 11))
    for name in ("h41_smoke_test.sh", "run_h41.sh", "resume_h41.sh"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "python -m beeid" not in text
        assert '"${PYTHON_BIN}" -m beeid.cli' in text
    package = (ROOT / "scripts" / "package_h41_results.sh").read_text(encoding="utf-8")
    assert "Tracking did not run" in package

    changed = tmp_path / "h41_protocol.lock.sha256"
    changed.write_text("0" * 64 + "  h41_protocol.lock.yaml\n", encoding="utf-8")
    with pytest.raises(H41ProtocolError, match="checksum mismatch"):
        validate_h41_protocol(ROOT / "configs" / "h41_protocol.lock.yaml", changed)

    nested_payload = yaml.safe_load(
        (ROOT / "configs" / "h41.example.yaml").read_text(encoding="utf-8")
    )
    nested_payload["paths"]["output_root"] = (
        nested_payload["paths"]["h4_output_root"] + "/h41-child"
    )
    nested = tmp_path / "nested.local.yaml"
    nested.write_text(yaml.safe_dump(nested_payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain or be contained"):
        load_config(nested)


def test_h41_synthetic_smoke_preserves_h4_stop_and_resumes(tmp_path):
    output = tmp_path / "h41-synthetic"
    result = h41_synthetic_smoke(output)
    assert result["status"] == "passed"
    assert result["source_h4_v1_status"] == "stopped_after_recoverability_audit"
    assert result["source_h4_v1_conclusion_preserved"] is True
    assert result["window_recoverability_switch_events"] > 0
    assert result["conflict_events"] > 0
    assert result["resumed_audit_jobs"] > 0
    assert result["resumed_tracking_jobs"] > 0
    assert result["final_test_read"] is False
    metadata = json.loads((output / "h41_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "SERVER_VALIDATION_PENDING"
    assert metadata["real_experiment_result"] is False
    assert metadata["final_test_read"] is False
    crosscheck = json.loads(
        (output / "h41_source_h4_crosscheck.json").read_text(encoding="utf-8")
    )
    assert crosscheck["status"] == "passed"
    assert (output / "h41_figures" / "cumulative_recoverability.svg").is_file()
    assert (output / "h41_figures" / "idsw_immediate_vs_adaptive.svg").is_file()
