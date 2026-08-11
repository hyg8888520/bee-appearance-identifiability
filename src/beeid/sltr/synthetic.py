"""CPU-only SLTR smoke: test-only rows/encoder, relaxed temporary lock, never data results."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..h4.io import write_csv
from ..config import load_config
from ..data.mot import Observation
from ..utils import atomic_write_json, atomic_write_text, sha256_file
from .selector import choose_oof_threshold, feature_matrix, fit_l2_logistic, group_oof_probabilities
from .tracker import build_counterfactual_events, track_selective_sequence


def _observation(frame: int, track_id: int, x: float) -> Observation:
    return Observation(
        observation_id=f"validation:synthetic-sltr:{frame:06d}:{track_id}",
        split="validation", source_split="train", video_id="synthetic-sltr",
        track_id=track_id, identity=f"synthetic-sltr:{track_id}", frame=frame,
        frame_id=frame, image_path=f"train/synthetic-sltr/img1/{frame:06d}.jpg",
        image_width=100, image_height=80, original_width=20.0, original_height=20.0,
        raw_x=x + 1.0, raw_y=21.0, raw_w=20.0, raw_h=20.0,
        bbox_x1=x, bbox_y1=20.0, bbox_x2=x + 20.0, bbox_y2=40.0,
        x1=int(x), y1=18, x2=int(x + 20), y2=42, crop_expansion=0.2,
        crop_clipped=False, center_x=x + 10.0, center_y=30.0, bbox_area=400.0,
        confidence=1.0, object_class=1, visibility=1.0, skip_reason="",
        extra_columns="[]",
    )


def sltr_synthetic_smoke(output: Path | None = None) -> dict[str, Any]:
    """Exercise selector/threshold artifacts without a manifest, real backbone, or final split."""
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if output is None:
        temporary = tempfile.TemporaryDirectory(prefix="beeid-sltr-synthetic-")
        output = Path(temporary.name)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    relaxed = output / "synthetic_relaxed_lock"
    relaxed.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[3] / "configs" / "sltr_protocol.lock.yaml"
    lock = yaml.safe_load(source.read_text(encoding="utf-8"))
    lock["status"] = "SYNTHETIC_TEST_ONLY_RELAXED_LOCK"
    lock["selector"]["min_train_events"] = 4
    lock["selector"]["min_train_videos"] = 2
    lock["selector"]["min_positive_events"] = 2
    lock_path = relaxed / "sltr_protocol.synthetic.lock.yaml"
    atomic_write_text(lock_path, yaml.safe_dump(lock, sort_keys=False))
    atomic_write_text(relaxed / "sltr_protocol.synthetic.lock.sha256", f"{sha256_file(lock_path)}  {lock_path.name}\n")

    observations = [
        _observation(frame, track_id, 10.0 if track_id == 1 else 12.0)
        for frame in range(1, 6) for track_id in (1, 2)
    ]
    embeddings = np.stack([
        np.asarray([1.0, 0.0], dtype=np.float32)
        if item.track_id == 1 else np.asarray([0.0, 1.0], dtype=np.float32)
        for item in observations
    ])
    repository = Path(__file__).resolve().parents[3]
    h4 = load_config(repository / "configs" / "h4.example.yaml").h4
    if h4 is None:  # pragma: no cover - repository contract
        raise RuntimeError("SLTR synthetic smoke cannot load the H4 config")
    h4 = replace(h4, ambiguity_margin=1.0)
    baseline_rows, counterfactual_events = build_counterfactual_events(
        "test_only_encoder", observations, embeddings, h4, horizon=1
    )
    fallback_rows, fallback_events, _ = track_selective_sequence(
        "test_only_encoder", "learned_selector", observations, embeddings, h4,
        horizon=1, decide=lambda event: (False, "below_threshold_exact_a_fallback"),
    )
    baseline_predictions = {
        str(row["observation_id"]): str(row["predicted_track_id"]) for row in baseline_rows
    }
    fallback_predictions = {
        str(row["observation_id"]): str(row["predicted_track_id"]) for row in fallback_rows
    }
    if not counterfactual_events or baseline_predictions != fallback_predictions:
        raise RuntimeError("SLTR synthetic exact-A fallback does not reproduce the baseline")
    permuted = [
        Observation(**{
            **item.to_dict(), "track_id": 100 - item.track_id,
            "identity": f"synthetic-sltr:{100 - item.track_id}",
        })
        for item in observations
    ]
    _, permuted_events = build_counterfactual_events(
        "test_only_encoder", permuted, embeddings, h4, horizon=1
    )
    original_labels = [(event.label_available, event.utility) for event in counterfactual_events]
    permuted_labels = [(event.label_available, event.utility) for event in permuted_events]
    if original_labels != permuted_labels:
        raise RuntimeError("SLTR offline labels changed under GT identity permutation")

    feature_names = ("assignment_margin", "rank0_rank1_objective_gap", "active_track_count")
    rows: list[dict[str, Any]] = []
    for model in ("resnet50", "dinov3"):
        for video_number in range(4):
            for event_number in range(10):
                positive = event_number < 3
                features = {
                    "assignment_margin": 0.01 if positive else 0.25,
                    "rank0_rank1_objective_gap": 0.01 if positive else 0.45,
                    "active_track_count": 2.0 + (event_number % 2),
                }
                rows.append({
                    "model": model, "partition": "project_train", "video_id": f"fit-{video_number}",
                    "event_id": f"{model}:fit-{video_number}:{event_number}", "features": features,
                    "utility": 3.0 if positive else -2.0, "positive": positive,
                })
    models: dict[str, Any] = {}
    oof: list[dict[str, Any]] = []
    for model_name in ("resnet50", "dinov3"):
        selected = [row for row in rows if row["model"] == model_name]
        probabilities, names, folds = group_oof_probabilities(selected, l2=0.01, iterations=300, learning_rate=0.1)
        x, _ = feature_matrix(selected, names)
        labels = np.asarray([int(row["positive"]) for row in selected], dtype=np.float64)
        model = fit_l2_logistic(x, labels, names, l2=0.01, iterations=300, learning_rate=0.1)
        threshold = choose_oof_threshold(
            probabilities, [row["utility"] for row in selected], [row["video_id"] for row in selected],
            min_precision=0.8, max_harm=0.05, min_selected_events=10,
            min_selected_videos=3, max_intervention_fraction=0.30,
        )
        if threshold["status"] != "GO_SAFE_OOF_THRESHOLD":
            raise RuntimeError(f"SLTR synthetic threshold unexpectedly stopped: {threshold}")
        models[model_name] = {"status": threshold["status"], "model": model.to_dict(), "threshold": threshold, "folds": folds}
        for row, probability in zip(selected, probabilities):
            oof.append({"model": model_name, "video_id": row["video_id"], "event_id": row["event_id"], "oof_probability": float(probability), "utility_b_minus_a": row["utility"], "final_test_read": False})

    audit_rows = [{
        "model": row["model"], "partition": row["partition"], "video_id": row["video_id"],
        "event_id": row["event_id"], "start_frame": 1, "component_observation_ids": "test-only-a|test-only-b",
        "component_size": 2, "label_available": True, "label_unavailable_reason": "",
        "a_correct_observations": 5, "b_correct_observations": 6 if row["positive"] else 4,
        "a_idsw": 1, "b_idsw": 0 if row["positive"] else 2,
        "utility_b_minus_a": row["utility"], "positive": row["positive"],
        "features_json": json.dumps(row["features"], sort_keys=True),
        "counterfactual_a": "rank_0_immediate_then_rank_0_horizon",
        "counterfactual_b": "rank_1_immediate_then_rank_0_horizon",
        "component_policy": "one_to_one_local_component_only", "offline_label_uses_gt": True,
        "method_decision_uses_gt": False, "final_test_read": False,
    } for row in rows]
    write_csv(output / "sltr_audit.csv", audit_rows)
    write_csv(output / "sltr_event_counterfactuals.csv", audit_rows)
    atomic_write_json(output / "sltr_feature_schema.json", {
        "feature_names": list(feature_names), "ground_truth_features": False,
        "final_test_read": False,
    })
    write_csv(output / "sltr_oof.csv", oof)
    write_csv(output / "sltr_oof_predictions.csv", oof)
    write_csv(output / "sltr_fit.csv", [fold for entry in models.values() for fold in entry["folds"]])
    model_payload = {"status": "completed", "models": models, "test_only": True, "final_test_read": False}
    fit_payload = {"status": "completed", "test_only": True, "development_used_for_fit": False, "final_test_read": False}
    atomic_write_json(output / "sltr_model.json", model_payload)
    atomic_write_json(output / "sltr_selector_models.json", model_payload)
    atomic_write_json(output / "sltr_fit.json", fit_payload)
    atomic_write_json(output / "sltr_fit_decision.json", fit_payload)
    for name, header in (
        ("sltr_assignments.csv", "model,variant,video_id,observation_id\n"),
        ("sltr_per_video_metrics.csv", "model,variant,video_id,IDF1,HOTA,IDSW\n"),
        ("sltr_metrics.csv", "model,variant,IDF1,HOTA,IDSW\n"),
        ("sltr_summary.csv", "model,variant,IDF1,HOTA,IDSW\n"),
        ("sltr_paired_video_metrics.csv", "model,variant,video_id,IDSW_reduction_vs_immediate,IDF1_gain_vs_immediate,HOTA_gain_vs_immediate\n"),
        ("sltr_interventions.csv", "model,variant,event_id,repaired\n"),
        ("sltr_intervention_metrics.csv", "model,variant,event_count,selected_event_count,intervention_precision,conservative_harm_rate\n"),
        ("sltr_failures.csv", "model,category,observation_id\n"),
        ("sltr_failure_cases.csv", "model,category,observation_id\n"),
    ):
        atomic_write_text(output / name, header)
    decision_payload = {
        "status": "SYNTHETIC_SMOKE_ONLY_NOT_READY", "gate_passed": False,
        "real_experiment_result": False, "final_test_read": False,
    }
    atomic_write_json(output / "sltr_decision.json", decision_payload)
    atomic_write_json(output / "sltr_method_decision.json", decision_payload)
    atomic_write_text(output / "sltr_resolved_config.yaml", "synthetic_test_only: true\nfinal_test_read: false\n")
    atomic_write_text(output / "sltr_result_guide.md", "Synthetic test-only encoder smoke; not a real experiment or final-test result.\n")
    atomic_write_text(output / "figures" / "sltr_idsw.svg", '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>\n')
    atomic_write_json(output / "logs" / "sltr_synthetic_progress.json", {"status": "completed", "test_only_encoder": True, "final_test_read": False})
    atomic_write_json(output / "sltr_run_metadata.json", {
        "status": "SERVER_VALIDATION_PENDING", "test_only": True, "test_only_encoder": True,
        "real_experiment_result": False, "final_test_read": False,
        "relaxed_temporary_lock": str(lock_path),
    })
    atomic_write_json(output / "sltr_artifact_signatures.json", {"test_only": True, "final_test_read": False})
    result = {
        "status": "passed", "test_only": True, "test_only_encoder": True,
        "relaxed_temporary_lock": True, "model_count": len(models),
        "oof_row_count": len(oof), "counterfactual_event_count": len(counterfactual_events),
        "fallback_event_count": len(fallback_events),
        "exact_baseline_fallback_verified": True,
        "identity_permutation_invariant": True, "final_test_read": False,
        "real_experiment_result": False,
    }
    if temporary is not None:
        # The returned status, not an external temporary path, is the smoke contract.
        temporary.cleanup()
    return result
