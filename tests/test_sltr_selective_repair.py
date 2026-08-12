from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from beeid.config import ConfigurationError, load_config
from beeid.sltr.core import _protocol_provenance
from beeid.sltr.protocol import SLTRProtocolError, validate_sltr_protocol
from beeid.sltr.experiment import _gate
from beeid.sltr.selector import SelectorError, choose_oof_threshold, group_oof_probabilities
from beeid.sltr.synthetic import sltr_synthetic_smoke
from beeid.sltr.tracker import deterministic_frequency_choice, event_prehistory_mapping


ROOT = Path(__file__).resolve().parents[1]


def _rows() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for video in range(4):
        for index in range(10):
            good = index < 3
            result.append({
                "video_id": f"v{video}", "event_id": f"v{video}-{index}", "positive": good,
                "utility": 3.0 if good else -2.0,
                "features": {
                    "assignment_margin": 0.01 if good else 0.3,
                    "rank0_rank1_objective_gap": 0.01 if good else 0.5,
                },
            })
    return result


def test_sltr_checksum_split_final_lock_and_examples(tmp_path):
    audit = validate_sltr_protocol(
        ROOT / "configs" / "sltr_protocol.lock.yaml",
        ROOT / "configs" / "sltr_protocol.lock.sha256",
    )
    assert audit["final_test_access"] is False
    assert audit["parameters"]["horizon"] == 1
    provenance = _protocol_provenance(audit)
    assert provenance["protocol_sha256"] == audit["protocol_sha256"]
    assert provenance["source_h3_protocol_sha256"] == audit["source_h3_protocol_sha256"]
    assert provenance["source_h3_protocol_sha256"] != provenance["protocol_sha256"]
    for name, subset in (("sltr.example.yaml", False), ("sltr.local.yaml.example", False), ("sltr_smoke.example.yaml", True)):
        config = load_config(ROOT / "configs" / name)
        assert config.sltr is not None and config.sltr.allow_subset is subset
        assert config.paths.output_root != config.paths.h3_output_root
    changed = tmp_path / "bad.sha256"
    changed.write_text("0" * 64, encoding="utf-8")
    with pytest.raises(SLTRProtocolError, match="checksum mismatch"):
        validate_sltr_protocol(ROOT / "configs" / "sltr_protocol.lock.yaml", changed)

    nested = yaml.safe_load((ROOT / "configs" / "sltr.example.yaml").read_text(encoding="utf-8"))
    nested["paths"]["output_root"] = nested["paths"]["h3_output_root"] + "/sltr-child"
    local = tmp_path / "nested.yaml"
    local.write_text(yaml.safe_dump(nested, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must not contain or be contained"):
        load_config(local)


def test_prehistory_label_mapping_is_unique_and_component_level():
    rows = [
        {"gt_identity": "v:1", "predicted_track_id": "10", "frame_id": 1, "observation_id": "a"},
        {"gt_identity": "v:2", "predicted_track_id": "11", "frame_id": 1, "observation_id": "b"},
    ]
    mapping, reason = event_prehistory_mapping(rows, ("v:1", "v:2"))
    assert reason is None and mapping == {"v:1": "10", "v:2": "11"}
    mapping, reason = event_prehistory_mapping(rows + [{"gt_identity": "v:1", "predicted_track_id": "11"}], ("v:1", "v:2"))
    assert mapping == {} and reason is not None


def test_prehistory_mapping_uses_recent_window_not_irreversible_full_history():
    rows = [
        {"gt_identity": "v:1", "predicted_track_id": "old", "frame_id": 1, "observation_id": "old-1"},
        {"gt_identity": "v:2", "predicted_track_id": "old-2", "frame_id": 1, "observation_id": "old-2"},
    ]
    for frame in range(2, 7):
        rows.extend([
            {"gt_identity": "v:1", "predicted_track_id": "10", "frame_id": frame, "observation_id": f"a-{frame}"},
            {"gt_identity": "v:2", "predicted_track_id": "11", "frame_id": frame, "observation_id": f"b-{frame}"},
        ])
    mapping, reason = event_prehistory_mapping(rows, ("v:1", "v:2"), history_length=5)
    assert reason is None and mapping == {"v:1": "10", "v:2": "11"}


def test_feature_leakage_group_oof_threshold_and_random_are_deterministic():
    leaking = _rows()
    leaking[0]["features"] = {"gt_identity": 1.0}
    with pytest.raises(SelectorError, match="leakage"):
        group_oof_probabilities(leaking, l2=0.01, iterations=20, learning_rate=0.1)

    rows = _rows()
    probabilities, _, folds = group_oof_probabilities(rows, l2=0.01, iterations=250, learning_rate=0.1)
    result = choose_oof_threshold(
        probabilities, [float(row["utility"]) for row in rows], [str(row["video_id"]) for row in rows],
        min_precision=0.8, max_harm=0.05, min_selected_events=10,
        min_selected_videos=3, max_intervention_fraction=0.30,
    )
    assert len(folds) == 4 and result["status"] == "GO_SAFE_OOF_THRESHOLD"
    ids = [str(row["event_id"]) for row in rows]
    assert deterministic_frequency_choice(ids, 10, 24) == deterministic_frequency_choice(list(reversed(ids)), 10, 24)


def test_threshold_fails_closed_outside_harm_or_intervention_bounds():
    probabilities = [0.9, 0.8, 0.7, 0.6]
    utilities = [1.0, -1.0, 1.0, -1.0]
    result = choose_oof_threshold(
        probabilities, utilities, ["a", "b", "c", "d"],
        min_precision=0.8, max_harm=0.05, min_selected_events=2,
        min_selected_videos=2, max_intervention_fraction=0.30,
    )
    assert result["status"] == "STOP_NO_SAFE_OOF_THRESHOLD"


def test_development_gate_requires_pooled_safety_and_random_specificity():
    per_video = []
    for video in range(4):
        for variant, idsw in (
            ("immediate_baseline", 5), ("frequency_matched_random", 4),
            ("learned_selector", 3),
        ):
            per_video.append({
                "model": "resnet50", "variant": variant, "video_id": f"v{video}",
                "IDSW": idsw, "IDF1": 0.8, "HOTA": 0.8,
            })
    pooled = [
        {"model": "resnet50", "variant": variant, "IDF1": 0.8, "HOTA": 0.8}
        for variant in ("immediate_baseline", "frequency_matched_random", "learned_selector")
    ]
    interventions = [
        {"model": "resnet50", "variant": "learned_selector", "repaired": index == 0,
         "label_available": True, "offline_utility_b_minus_a": 1.0}
        for index in range(4)
    ]
    config = SimpleNamespace(
        noninferiority_tolerance=0.002, min_nonharmed_videos=4,
        max_harm=0.05, max_intervention_fraction=0.30,
    )
    decision = _gate(per_video, pooled, interventions, ["resnet50"], config)
    assert decision["gate_passed"] is True
    for row in per_video:
        if row["variant"] == "frequency_matched_random":
            row["IDSW"] = 2
    decision = _gate(per_video, pooled, interventions, ["resnet50"], config)
    assert decision["gate_passed"] is False


def test_exact_fallback_component_choice_gate_and_synthetic_artifacts(tmp_path):
    # The locked names are the contract: learned below threshold means exact A, and
    # local alternatives are rank-0/rank-1 one-to-one H4 options.
    lock = (ROOT / "configs" / "sltr_protocol.lock.yaml").read_text(encoding="utf-8")
    tracker = (ROOT / "src" / "beeid" / "sltr" / "tracker.py").read_text(encoding="utf-8")
    assert "rank_0_immediate_then_rank_0_horizon" in lock
    assert "below_threshold_exact_a_fallback" in tracker
    assert "one_to_one_local_component_only" in lock
    assert "STOP_SLTR_DEVELOPMENT_GATE" in (ROOT / "src" / "beeid" / "sltr" / "experiment.py").read_text(encoding="utf-8")
    result = sltr_synthetic_smoke(tmp_path / "synthetic")
    assert result["status"] == "passed" and result["final_test_read"] is False
    assert result["counterfactual_event_count"] > 0
    assert result["exact_baseline_fallback_verified"] is True
    assert result["identity_permutation_invariant"] is True
    for name in (
        "sltr_event_counterfactuals.csv", "sltr_feature_schema.json",
        "sltr_oof_predictions.csv", "sltr_selector_models.json", "sltr_fit_decision.json",
        "sltr_summary.csv", "sltr_intervention_metrics.csv", "sltr_failure_cases.csv",
        "sltr_method_decision.json", "sltr_run_metadata.json", "sltr_resolved_config.yaml",
        "sltr_result_guide.md",
    ):
        assert (tmp_path / "synthetic" / name).is_file()
    metadata = json.loads((tmp_path / "synthetic" / "sltr_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["test_only_encoder"] is True and metadata["real_experiment_result"] is False
    for name in ("run_sltr.sh", "resume_sltr.sh", "monitor_sltr.sh", "sltr_smoke_test.sh", "package_sltr_results.sh"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert '"${PYTHON_BIN}" -m beeid.cli' in text or name in {"monitor_sltr.sh", "package_sltr_results.sh"}
