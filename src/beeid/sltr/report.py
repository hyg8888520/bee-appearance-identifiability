"""SLTR report artifacts and explicit development-only claim boundaries."""

from __future__ import annotations

import csv
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml
import numpy as np

from ..config import ExperimentConfig
from ..h4.io import write_csv
from ..utils import atomic_write_json, atomic_write_text, git_head, sha256_file
from . import SLTR_PRIMARY_VARIANT
from .core import validate_sltr_inputs


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _failures(assignments: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in assignments:
        if row.get("variant") in {"immediate_baseline", SLTR_PRIMARY_VARIANT}:
            by_key[(row["model"], row["variant"], row["gt_identity"])].append(row)
    switches: dict[tuple[str, str, str], set[str]] = {}
    for key, rows in by_key.items():
        previous: str | None = None
        found: set[str] = set()
        for row in sorted(rows, key=lambda item: (int(item["frame_id"]), item["observation_id"])):
            current = row["predicted_track_id"]
            if previous is not None and previous != current:
                found.add(row["observation_id"])
            previous = current
        switches[key] = found
    output: list[dict[str, Any]] = []
    for row in assignments:
        if row.get("variant") != SLTR_PRIMARY_VARIANT:
            continue
        key = (row["model"], row["gt_identity"])
        baseline = row["observation_id"] in switches.get((key[0], "immediate_baseline", key[1]), set())
        learned = row["observation_id"] in switches.get((key[0], SLTR_PRIMARY_VARIANT, key[1]), set())
        if baseline == learned:
            continue
        output.append({
            "model": row["model"], "video_id": row["video_id"], "frame_id": row["frame_id"],
            "observation_id": row["observation_id"], "category": "recovered_baseline_idsw" if baseline else "new_learned_idsw",
            "selected_branch": row.get("sltr_selected_branch", ""),
            "selection_reason": row.get("sltr_selection_reason", ""),
            "final_test_read": False,
        })
    return sorted(output, key=lambda row: (row["category"], row["model"], row["video_id"], int(row["frame_id"])))[:500]


def _figure(output: Path, metrics: Sequence[dict[str, str]]) -> None:
    selected = [row for row in metrics if row.get("variant") in {"immediate_baseline", SLTR_PRIMARY_VARIANT}]
    labels = sorted({row.get("model", "") for row in selected})
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="160" viewBox="0 0 760 160">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="sans-serif" font-size="18">SLTR development-only IDSW comparison</text>',
    ]
    for index, model in enumerate(labels):
        baseline = next((row for row in selected if row.get("model") == model and row.get("variant") == "immediate_baseline"), {})
        learned = next((row for row in selected if row.get("model") == model and row.get("variant") == SLTR_PRIMARY_VARIANT), {})
        y = 65 + index * 42
        lines.append(f'<text x="20" y="{y}" font-family="sans-serif" font-size="14">{model}: baseline IDSW={baseline.get("IDSW", "?")}, learned IDSW={learned.get("IDSW", "?")}</text>')
    lines.append('</svg>')
    atomic_write_text(output / "figures" / "sltr_idsw.svg", "\n".join(lines) + "\n")


def generate_sltr_report(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    inputs = validate_sltr_inputs(config)
    output = config.paths.output_root
    # A fail-closed fit may stop before development tracking.  Preserve a complete,
    # explicitly empty artifact surface rather than silently omitting downstream files.
    for name, header in (
        ("sltr_assignments.csv", "model,variant,video_id,observation_id\n"),
        ("sltr_per_video_metrics.csv", "model,variant,video_id,IDF1,HOTA,IDSW\n"),
        ("sltr_metrics.csv", "model,variant,IDF1,HOTA,IDSW\n"),
        ("sltr_summary.csv", "model,variant,IDF1,HOTA,IDSW\n"),
        ("sltr_paired_video_metrics.csv", "model,variant,video_id,IDSW_reduction_vs_immediate,IDF1_gain_vs_immediate,HOTA_gain_vs_immediate\n"),
        ("sltr_interventions.csv", "model,variant,event_id,repaired\n"),
        ("sltr_intervention_metrics.csv", "model,variant,event_count,selected_event_count,intervention_precision,conservative_harm_rate\n"),
    ):
        if not (output / name).is_file():
            atomic_write_text(output / name, header)
    audit = _rows(output / "sltr_audit.csv")
    assignments = _rows(output / "sltr_assignments.csv")
    metrics = _rows(output / "sltr_metrics.csv")
    interventions = _rows(output / "sltr_interventions.csv")
    decision_path = output / "sltr_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.is_file() else {
        "status": "STOP_SLTR_NOT_TRACKED", "gate_passed": False, "final_test_read": False,
    }
    if not (output / "sltr_method_decision.json").is_file():
        atomic_write_json(output / "sltr_method_decision.json", decision)
    failures = _failures(assignments)
    write_csv(output / "sltr_failures.csv", failures)
    write_csv(output / "sltr_failure_cases.csv", failures)
    _figure(output, metrics)
    resolved = config.serializable()
    atomic_write_text(output / "sltr_resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=True))
    guide = """# SLTR result guide

This directory is a frozen **development-only** selective local trajectory repair audit.
`project_train` supplies offline counterfactual labels and is the only selector-fit partition.
`development_validation` is evaluated with the already-fitted selector. `final_test` is never read.

The learned selector uses only finite observable event features. Offline GT-derived labels,
utilities and the `oracle_selector` are diagnostic artifacts and are not deployable inputs.
`below_threshold_exact_a_fallback` means the learned path reuses the rank-0 A branch exactly.

SLTR is inspired by the general topic of local trajectory repair; it is **not** an official
reproduction or component of TOPICTrack, AGW, DINOv3, H3, or H4.
"""
    atomic_write_text(output / "sltr_result_guide.md", guide)
    hash_names = [
        "sltr_event_counterfactuals.csv", "sltr_feature_schema.json",
        "sltr_oof_predictions.csv", "sltr_selector_models.json", "sltr_fit_decision.json",
        "sltr_assignments.csv", "sltr_summary.csv", "sltr_interventions.csv",
        "sltr_intervention_metrics.csv", "sltr_failure_cases.csv",
        "sltr_method_decision.json", "sltr_tracking_metadata.json",
        "sltr_resolved_config.yaml", "sltr_result_guide.md",
    ]
    hashes = {name: sha256_file(output / name) for name in hash_names if (output / name).is_file()}
    tracking_completed = bool(
        (output / "sltr_tracking_metadata.json").is_file()
        and json.loads((output / "sltr_tracking_metadata.json").read_text(encoding="utf-8")).get("status") == "completed"
    )
    subset = bool(config.sltr and config.sltr.allow_subset)
    metadata = {
        "status": (
            "completed_subset_smoke" if subset
            else "completed_development_gt_boxes" if tracking_completed
            else "completed_project_train_audit_stop"
        ),
        "real_experiment_result": not subset,
        "project_git_commit": git_head(Path(__file__).resolve().parents[3]),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "seed": config.runtime.seed,
        "protocol_sha256": inputs.audit["protocol_sha256"],
        "models": list(model_names), "audit_event_count": len(audit),
        "assignment_count": len(assignments), "metric_rows": len(metrics),
        "intervention_count": len(interventions), "decision": decision,
        "input_audit": inputs.audit, "artifact_hashes": hashes,
        "fit_partition": "project_train", "evaluation_partition": "development_validation",
        "final_test_read": False,
    }
    atomic_write_json(output / "sltr_run_metadata.json", metadata)
    atomic_write_json(output / "sltr_artifact_signatures.json", {
        "artifacts": hashes, "final_test_read": False,
        "resumable_work_policy": "atomic_signed_jobs_only",
    })
    return metadata
