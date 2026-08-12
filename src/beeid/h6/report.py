"""H6 development decision, artifact inventory, and human-readable report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from ..config import ExperimentConfig
from ..utils import atomic_write_json, atomic_write_text, sha256_file
from . import H6_MODELS, H6_PRIMARY_VARIANT
from .core import require_h6, validate_h6_inputs


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Required H6 artifact is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_h6_report(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    h6 = require_h6(config)
    inputs = validate_h6_inputs(config)
    output = config.paths.output_root
    required = (
        "h6_oracle_audit.json", "h6_training_metadata.json", "h6_tracking_metadata.json",
        "h6_assignments.csv", "h6_summary.csv", "h6_per_video_metrics.csv",
        "h6_paired_video_metrics.csv", "h6_trajectory_decisions.csv",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError("H6 report requires completed artifacts: " + ", ".join(missing))
    oracle = json.loads((output / "h6_oracle_audit.json").read_text(encoding="utf-8"))
    training = json.loads((output / "h6_training_metadata.json").read_text(encoding="utf-8"))
    tracking = json.loads((output / "h6_tracking_metadata.json").read_text(encoding="utf-8"))
    if any(value.get("final_test_read") is not False for value in (oracle, training, tracking)):
        raise RuntimeError("H6 artifact does not prove final_test_read=false")
    summary = _csv(output / "h6_summary.csv")
    per_video = _csv(output / "h6_per_video_metrics.csv")
    decisions = _csv(output / "h6_trajectory_decisions.csv")
    checks: dict[str, Any] = {}
    for model in model_names:
        rows = {
            row["variant"]: row for row in summary
            if row["model"] == model and row["aggregation"] == "micro_pooled"
        }
        baseline, primary = rows.get("baseline_association"), rows.get(H6_PRIMARY_VARIANT)
        if baseline is None or primary is None:
            raise RuntimeError(f"H6 summary lacks baseline/primary rows for {model}")
        model_videos = [
            row for row in per_video
            if row["model"] == model and row["variant"] == H6_PRIMARY_VARIANT
        ]
        baseline_videos = {
            row["video_id"]: row for row in per_video
            if row["model"] == model and row["variant"] == "baseline_association"
        }
        nonharmed = sum(
            float(row["IDF1"]) + h6.noninferiority_tolerance
            >= float(baseline_videos[row["video_id"]]["IDF1"])
            for row in model_videos
        )
        calibration = training["training"][model]["calibration"]
        association_calibration = training["training"][model]["association_calibration"]
        accepted = sum(
            row["model"] == model and row["accepted"] == "True" for row in decisions
        )
        checks[model] = {
            "calibration_gate": calibration["gate"],
            "association_calibration_gate": association_calibration["gate"],
            "accepted_development_components": accepted,
            "IDF1_gain": float(primary["IDF1"]) - float(baseline["IDF1"]),
            "AssA_gain": float(primary["AssA"]) - float(baseline["AssA"]),
            "HOTA_gain": float(primary["HOTA"]) - float(baseline["HOTA"]),
            "IDSW_reduction": int(baseline["IDSW"]) - int(primary["IDSW"]),
            "nonharmed_videos": nonharmed,
            "passes_noninferiority": float(primary["IDF1"]) + h6.noninferiority_tolerance >= float(baseline["IDF1"]),
            "passes_improvement": float(primary["IDF1"]) > float(baseline["IDF1"])
            and float(primary["AssA"]) > float(baseline["AssA"]),
            "passes_video_support": nonharmed >= h6.min_nonharmed_videos,
            "passes_intervention": accepted > 0,
        }
    both_models = set(model_names) == set(H6_MODELS)
    ready = (
        not h6.allow_subset and both_models and oracle.get("gate") == "GO_DENSE_CANDIDATE_GRAPH"
        and all(
            value["association_calibration_gate"] == "GO_CALIBRATED_ASSOCIATION"
            and value["calibration_gate"] == "GO_CALIBRATED_SELECTOR"
            and value["passes_noninferiority"] and value["passes_improvement"]
            and value["passes_video_support"]
            and value["passes_intervention"]
            for value in checks.values()
        )
    )
    decision = "GO_FREEZE_H6_FOR_FIXED_DETECTOR_VALIDATION" if ready else "STOP_H6_DEVELOPMENT_GATE_FAILED"
    result = {
        "status": "completed_development_gt_boxes", "decision": decision,
        "method_ready": ready, "models": list(model_names), "checks": checks,
        "oracle_gate": oracle.get("gate"), "subset_never_method_ready": h6.allow_subset,
        "fixed_detector_boxes": "SERVER_VALIDATION_PENDING",
        "final_test_read": False,
    }
    atomic_write_json(output / "h6_method_decision.json", result)
    artifacts = {
        name: sha256_file(output / name) for name in required
    }
    artifacts["h6_method_decision.json"] = sha256_file(output / "h6_method_decision.json")
    atomic_write_json(output / "h6_artifact_signatures.json", {
        "sha256": artifacts, "input_audit": inputs.audit, "final_test_read": False,
    })
    lines = [
        "# H6 Global Trajectory Reasoner development report", "",
        f"- Decision: `{decision}`", f"- Method ready: `{str(ready).lower()}`",
        "- Evaluation: BEE24 development_validation with GT boxes",
        "- Final test read: `false`", "- Fixed-detector validation: `SERVER_VALIDATION_PENDING`", "",
        "## Backbone results", "",
    ]
    for model, value in checks.items():
        lines.extend([
            f"### {model}", "",
            f"- Calibration gate: `{value['calibration_gate']}`",
            f"- Association calibration gate: `{value['association_calibration_gate']}`",
            f"- Accepted components: {value['accepted_development_components']}",
            f"- IDF1 gain vs frozen baseline: {value['IDF1_gain']:.6f}",
            f"- AssA gain vs frozen baseline: {value['AssA_gain']:.6f}",
            f"- HOTA gain vs frozen baseline: {value['HOTA_gain']:.6f}",
            f"- IDSW reduction: {value['IDSW_reduction']}", "",
        ])
    lines.extend([
        "## Interpretation", "",
        "The trainable global reasoner evaluates every cross-frame pair within the frozen temporal gap.",
        "The selector accepts an entire baseline/neural overlap component or preserves the frozen baseline exactly.",
        "Development GT identities are used only after predictions for metrics, never as selector input.", "",
    ])
    atomic_write_text(output / "h6_report.md", "\n".join(lines))
    return result
