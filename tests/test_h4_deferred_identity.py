from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from beeid.config import H4Config, load_config
from beeid.data.mot import Observation
from beeid.h4.assignment import local_assignment_options
from beeid.h4.protocol import H4ProtocolError, validate_h4_protocol
from beeid.h4.recoverability import (
    audit_model_recoverability,
    recoverability_decision,
    summarize_recoverability,
)
from beeid.h4.report import _idsw_svg
from beeid.h4.synthetic import h4_synthetic_smoke
from beeid.h4.tracker import (
    H4_PRIMARY_VARIANT,
    H4_VARIANTS,
    TrackState,
    TrackerState,
    _step_options,
    track_h4_sequence,
)


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


def _h4_config(**changes: object) -> H4Config:
    config = load_config(ROOT / "configs" / "h4.example.yaml").h4
    assert config is not None
    return replace(config, **changes)


def test_local_assignment_options_are_deterministic_and_local():
    scores = np.asarray([[0.90, 0.89], [0.89, 0.90]], dtype=np.float64)
    first, conflict = local_assignment_options(
        scores,
        valid=np.ones_like(scores, dtype=bool),
        minimum_score=0.1,
        ambiguity_margin=0.03,
        max_component_size=4,
        beam_width=6,
        unmatched_penalty=0.25,
    )
    second, _ = local_assignment_options(
        scores,
        valid=np.ones_like(scores, dtype=bool),
        minimum_score=0.1,
        ambiguity_margin=0.03,
        max_component_size=4,
        beam_width=6,
        unmatched_penalty=0.25,
    )
    assert conflict is not None and conflict.oversized is False
    assert conflict.rows == (0, 1) and conflict.columns == (0, 1)
    assert len(first) == 2
    assert [item.matches for item in first] == [item.matches for item in second]
    assert first[0].matches == ((0, 0), (1, 1))
    assert first[1].matches == ((0, 1), (1, 0))


def test_branch_memory_is_copy_on_write_and_losing_branch_cannot_mutate_source():
    e1 = np.asarray([1.0, 0.0], dtype=np.float32)
    e2 = np.asarray([0.0, 1.0], dtype=np.float32)
    state = TrackerState(
        active={
            1: TrackState(1, e1.copy(), (20.0, 30.0), 1, 20.0, 20.0),
            2: TrackState(2, e2.copy(), (22.0, 30.0), 1, 20.0, 20.0),
        },
        next_track_id=3,
    )
    observations = [_observation(2, 1, 10.5), _observation(2, 2, 12.5)]
    config = _h4_config(ambiguity_margin=1.0)
    outcomes = _step_options(
        "m", H4_PRIMARY_VARIANT, state, observations, [e1, e2], 2, config,
        allow_branch=True, memory_mode="isolated",
    )
    assert len(outcomes) >= 2
    assert np.array_equal(state.active[1].memory, e1)
    assert np.array_equal(state.active[2].memory, e2)
    assert not np.shares_memory(
        outcomes[0].state.active[1].memory, outcomes[1].state.active[1].memory
    )
    losing_before = outcomes[1].state.active[1].memory.copy()
    outcomes[0].state.active[1].memory[:] = 0.0
    assert np.array_equal(outcomes[1].state.active[1].memory, losing_before)
    assert np.array_equal(state.active[1].memory, e1)


def test_fixed_lag_decisions_never_read_after_the_recorded_decision_frame():
    observations: list[Observation] = []
    embeddings: list[np.ndarray] = []
    for frame in range(1, 9):
        observations.extend([_observation(frame, 1, 10.0), _observation(frame, 2, 12.0)])
        embeddings.extend(
            [np.asarray([1.0, 0.0], dtype=np.float32), np.asarray([0.0, 1.0], dtype=np.float32)]
        )
    rows, events = track_h4_sequence(
        "m",
        "fixed_lag_isolated_memory_h3",
        observations,
        np.stack(embeddings),
        _h4_config(ambiguity_margin=1.0),
    )
    deferred = [event for event in events if event["deferred_commit"]]
    assert deferred
    assert all(event["decision_frame"] <= event["start_frame"] + 3 for event in deferred)
    assert all(int(row["decision_frame"]) >= int(row["frame_id"]) for row in rows)
    assert all(int(row["decision_latency_frames"]) <= 3 for row in rows)


def test_oracle_recoverability_is_separate_from_method_inputs_and_ties_fail():
    observations = [
        _observation(frame, track_id, 10.0 if track_id == 1 else 50.0)
        for frame in range(1, 4)
        for track_id in (1, 2)
    ]
    embeddings = np.stack(
        [
            np.asarray([1.0, 0.0], dtype=np.float32)
            if item.track_id == 1 else np.asarray([0.0, 1.0], dtype=np.float32)
            for item in observations
        ]
    )
    rows = []
    for item in observations:
        predicted = item.track_id
        if item.frame == 2:
            predicted = 3 - item.track_id
        rows.append(
            {
                "model": "m", "video_id": "v", "frame_id": item.frame,
                "observation_id": item.observation_id, "gt_identity": item.identity,
                "predicted_track_id": predicted,
            }
        )
    config = _h4_config(min_events_per_model=1, min_videos_with_events=1)
    audited = audit_model_recoverability("m", observations, embeddings, rows, config)
    horizon_one = [row for row in audited if row["horizon"] == 1]
    assert horizon_one and all(row["oracle_uses_gt"] is True for row in horizon_one)
    assert all(row["method_input"] is False for row in horizon_one)
    assert any(row["appearance_recoverable"] for row in horizon_one)

    tied = embeddings.copy()
    tied[:] = np.asarray([1.0, 0.0], dtype=np.float32)
    tied_audit = audit_model_recoverability("m", observations, tied, rows, config)
    assert not any(
        row["appearance_recoverable"] for row in tied_audit if row["horizon"] == 1
    )


def test_recoverability_gate_requires_each_requested_model():
    config = _h4_config(
        min_events_per_model=1,
        min_videos_with_events=1,
        min_recoverable_fraction=0.3,
    )
    events = []
    for model in ("resnet50", "dinov3"):
        for horizon in config.horizons:
            events.append(
                {
                    "model": model, "video_id": "v", "event_id": f"{model}:e",
                    "horizon": horizon, "available": True,
                    "appearance_recoverable": True, "motion_recoverable": True,
                    "joint_recoverable": True,
                }
            )
    summary = summarize_recoverability(events, ["resnet50", "dinov3"], config.horizons)
    decision = recoverability_decision(summary, ["resnet50", "dinov3"], config)
    assert decision["status"] == "GO_METHOD_EVALUATION"
    assert all(item["pass"] for item in decision["model_checks"])


def test_idswitch_figure_accepts_an_all_zero_smoke_result(tmp_path):
    output = tmp_path / "idsw.svg"
    rows = [
        {"model": model, "variant": variant, "IDSW": "0"}
        for model in ("resnet50", "dinov3")
        for variant in ("immediate_commit", H4_PRIMARY_VARIANT)
    ]
    _idsw_svg(output, rows)
    rendered = output.read_text(encoding="utf-8")
    assert "Immediate vs primary fixed-lag ID switches" in rendered
    assert "nan" not in rendered.lower()
    assert "inf" not in rendered.lower()


def test_h4_protocol_examples_scripts_and_checksum_are_locked(tmp_path):
    audit = validate_h4_protocol(
        ROOT / "configs" / "h4_protocol.lock.yaml",
        ROOT / "configs" / "h4_protocol.lock.sha256",
    )
    assert audit["final_test_access"] is False
    assert audit["primary_variant"] == H4_PRIMARY_VARIANT
    assert tuple(audit["variants"]) == H4_VARIANTS
    for name, subset in (
        ("h4.example.yaml", False),
        ("h4.local.yaml.example", False),
        ("h4_smoke.example.yaml", True),
    ):
        payload = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        config = load_config(ROOT / "configs" / name)
        assert config.h4 is not None and config.h4.allow_subset is subset
        assert payload["dataset"]["source_splits"] == ["train"]
        assert payload["protocol"]["evaluation_split"] == "validation"
        assert payload["paths"]["h3_output_root"] != payload["paths"]["output_root"]
    for name in ("h4_smoke_test.sh", "run_h4.sh", "resume_h4.sh"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "python -m beeid" not in text
        assert '"${PYTHON_BIN}" -m beeid.cli' in text

    changed = tmp_path / "h4_protocol.lock.sha256"
    changed.write_text("0" * 64 + "  h4_protocol.lock.yaml\n", encoding="utf-8")
    with pytest.raises(H4ProtocolError, match="checksum mismatch"):
        validate_h4_protocol(ROOT / "configs" / "h4_protocol.lock.yaml", changed)


def test_h4_synthetic_smoke_exercises_gate_branches_and_reports(tmp_path):
    output = tmp_path / "h4-synthetic"
    result = h4_synthetic_smoke(output)
    assert result["status"] == "passed"
    assert result["test_only_encoder"] is True
    assert result["recoverability_switch_events"] > 0
    assert result["conflict_events"] > 0
    assert result["resumed_audit_jobs"] > 0
    assert result["resumed_tracking_jobs"] > 0
    assert result["final_test_read"] is False
    metadata = json.loads((output / "h4_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "SERVER_VALIDATION_PENDING"
    assert metadata["real_experiment_result"] is False
    assert metadata["final_test_read"] is False
    assert (output / "h4_figures" / "recoverability_by_horizon.svg").is_file()
    assert (output / "h4_figures" / "idsw_immediate_vs_primary.svg").is_file()
