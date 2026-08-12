"""Closed-loop health summaries, fuse decision and auditable H5.1 reporting."""

from __future__ import annotations

import csv
import io
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..config import H51Config
from ..h3.metrics import summarize_gt_assignments
from ..utils import atomic_write_json, atomic_write_text, canonical_json, git_head, sha256_file, sha256_text
from . import H51_DECISION_READY, H51_DECISION_STOP
from .tracker import REJECTION_REASONS


H51_REQUIRED_OUTPUTS = (
    "h51_gradient_coverage.csv",
    "h51_path_audit.json",
    "h51_teacher_forced_queries.csv",
    "h51_rollout_observation_diagnostics.csv",
    "h51_frame_diagnostics.csv",
    "h51_variant_summary.csv",
    "h51_rejection_summary.csv",
    "h51_replay_equivalence.json",
    "h51_decision.json",
    "h51_run_metadata.json",
    "h51_result_guide.md",
    "h51_artifact_signatures.json",
)


def runtime_validation_fields(device: torch.device, test_only: bool) -> dict[str, str]:
    """Describe observed execution without inferring a specific GPU model."""
    if test_only:
        return {
            "runtime_validation": "test_only_scientific_smoke_completed",
            "gpu_execution_validation": "SERVER_VALIDATION_PENDING",
            "real_gpu_validation": "SERVER_VALIDATION_PENDING",
            "real_diagnostic_execution": "not_applicable_test_only",
        }
    if device.type == "cuda":
        return {
            "runtime_validation": "real_diagnostic_completed_on_cuda",
            "gpu_execution_validation": "COMPLETED_ON_CONFIGURED_CUDA_RUNTIME",
            "real_gpu_validation": "COMPLETED_ON_CONFIGURED_CUDA_RUNTIME",
            "real_diagnostic_execution": "completed_on_cuda",
        }
    return {
        "runtime_validation": "real_diagnostic_completed_on_cpu",
        "gpu_execution_validation": "SERVER_VALIDATION_PENDING",
        "real_gpu_validation": "SERVER_VALIDATION_PENDING",
        "real_diagnostic_execution": "completed_on_cpu",
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] = ()) -> None:
    fieldnames = tuple(rows[0]) if rows else tuple(fields)
    buffer = io.StringIO(newline="")
    if fieldnames:
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _track_lengths(rows: Sequence[dict[str, Any]]) -> tuple[float, float]:
    counts = Counter(str(row["predicted_track_id"]) for row in rows)
    values = np.asarray(list(counts.values()), dtype=np.float64)
    if not len(values):
        return 0.0, 0.0
    return float(np.median(values)), float(np.quantile(values, 0.90))


def _reliability_health(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labelled = [
        row for row in rows
        if isinstance(row.get("association_continuity_correct"), bool)
        and row.get("reliability") not in {"", None}
    ]
    if not labelled:
        return {
            "reliability_audit_count": 0,
            "reliability_brier_score": "",
            "reliability_ece_10_bin": "",
        }
    probabilities = np.asarray([float(row["reliability"]) for row in labelled], dtype=np.float64)
    labels = np.asarray(
        [float(bool(row["association_continuity_correct"])) for row in labelled],
        dtype=np.float64,
    )
    indices = np.minimum((probabilities * 10.0).astype(np.int64), 9)
    ece = 0.0
    for index in range(10):
        mask = indices == index
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(probabilities[mask].mean() - labels[mask].mean())
            )
    return {
        "reliability_audit_count": len(labelled),
        "reliability_brier_score": float(np.mean((probabilities - labels) ** 2)),
        "reliability_ece_10_bin": ece,
    }


def summarize_closed_loop(
    assignment_rows: Sequence[dict[str, Any]],
    observation_diagnostics: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_video, pooled = summarize_gt_assignments(assignment_rows)
    assignments: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        assignments[(str(row["model"]), str(row["variant"]), str(row["video_id"]))].append(row)
    diagnostics: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observation_diagnostics:
        diagnostics[(str(row["model"]), str(row["variant"]), str(row["video_id"]))].append(row)
    metric_lookup = {
        (str(row["model"]), str(row["variant"]), str(row["video_id"])): row for row in per_video
    }
    baseline_predictions = {
        (model, video): len({str(row["predicted_track_id"]) for row in rows})
        for (model, variant, video), rows in assignments.items()
        if variant == "frozen_h3_baseline"
    }
    variant_rows: list[dict[str, Any]] = []
    for key, rows in sorted(assignments.items()):
        model, variant, video = key
        metric = metric_lookup[key]
        diag = diagnostics.get(key, [])
        predicted_count = len({str(row["predicted_track_id"]) for row in rows})
        baseline_count = baseline_predictions.get((model, video), predicted_count)
        median, p90 = _track_lengths(rows)
        new_count = sum(not bool(row.get("matched_existing_track")) for row in rows) if diag else ""
        eligible = [row for row in diag if row.get("matched_existing_track") is True]
        accepted = [row for row in eligible if row.get("gate_accepted") is True]
        continuity_correct = [
            row for row in eligible if row.get("association_continuity_correct") is True
        ]
        accepted_incorrect = [
            row for row in eligible if row.get("accepted_incorrect_update") is True
        ]
        reliability_health = _reliability_health(eligible)
        variant_rows.append(
            {
                "model": model, "variant": variant, "video_id": video,
                "aggregation": "per_video", "observation_count": len(rows),
                "new_track_count": new_count,
                "new_track_rate": float(new_count) / len(rows) if diag and rows else "",
                "predicted_track_count": predicted_count,
                "frozen_h3_predicted_track_count": baseline_count,
                "prediction_inflation_vs_frozen_h3": predicted_count / max(1, baseline_count),
                "track_length_median": median, "track_length_p90": p90,
                "IDSW": metric["IDSW"], "IDF1": metric["IDF1"],
                "AssA": metric["AssA"], "HOTA": metric["HOTA"],
                "eligible_gate_update_count": len(eligible) if diag else "",
                "accepted_gate_update_count": len(accepted) if diag else "",
                "gate_acceptance_rate": len(accepted) / len(eligible) if eligible else "",
                "matched_continuity_correct_count": len(continuity_correct) if diag else "",
                "matched_continuity_correct_rate": (
                    len(continuity_correct) / len(eligible) if eligible else ""
                ),
                "accepted_incorrect_update_count": len(accepted_incorrect) if diag else "",
                "accepted_incorrect_update_rate": (
                    len(accepted_incorrect) / len(accepted) if accepted else ""
                ),
                **(reliability_health if diag else {
                    "reliability_audit_count": "", "reliability_brier_score": "",
                    "reliability_ece_10_bin": "",
                }),
                "reliability_label_scope": (
                    "post_decision_gt_identity_continuity_audit" if diag else ""
                ),
                "diagnostic_not_confirmatory": True, "final_test_read": False,
            }
        )
    rejection_rows: list[dict[str, Any]] = []
    for key, rows in sorted(diagnostics.items()):
        counts = Counter(str(row["rejection_reason"]) for row in rows)
        for reason in REJECTION_REASONS:
            rejection_rows.append(
                {
                    "model": key[0], "variant": key[1], "video_id": key[2],
                    "rejection_reason": reason, "count": counts[reason],
                    "fraction": counts[reason] / max(1, len(rows)),
                    "complete_taxonomy": True, "final_test_read": False,
                }
            )
    return variant_rows, rejection_rows, pooled


def closed_loop_decision(
    h51: H51Config,
    gradient_summaries: dict[str, dict[str, Any]],
    teacher_summaries: dict[str, dict[str, Any]],
    variant_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for model, gradient in sorted(gradient_summaries.items()):
        selected = [row for row in variant_rows if row["model"] == model and row["variant"] != "frozen_h3_baseline"]
        gated = [row for row in selected if row["variant"] == "beetrackquery_gated_memory"]
        total_updates = sum(int(row["eligible_gate_update_count"] or 0) for row in gated)
        accepted_updates = sum(int(row["accepted_gate_update_count"] or 0) for row in gated)
        accepted_incorrect_updates = sum(
            int(row.get("accepted_incorrect_update_count") or 0) for row in gated
        )
        gate_rate = accepted_updates / total_updates if total_updates else 0.0
        gate_degenerate = total_updates >= h51.min_gate_updates and (
            gate_rate <= h51.gate_degeneracy_low or gate_rate >= h51.gate_degeneracy_high
        )
        collapse_rows = [
            row for row in selected
            if float(row["new_track_rate"]) >= h51.collapse_new_track_rate
            or float(row["prediction_inflation_vs_frozen_h3"]) >= h51.collapse_prediction_inflation
        ]
        gated_idf1 = float(np.mean([float(row["IDF1"]) for row in gated])) if gated else 0.0
        teacher_rank1 = float(teacher_summaries[model]["rank1_rate"])
        teacher_rollout_gap = teacher_rank1 - gated_idf1
        gap_failed = teacher_rollout_gap > h51.max_teacher_rollout_idf1_gap
        checks.append(
            {
                "model": model,
                "gradient_coverage_pass": bool(gradient["method_ready"]),
                "memory_training_defect_detected": bool(gradient["memory_defect_detected"]),
                "track_collapse_detected": bool(collapse_rows),
                "collapsed_video_variants": [f"{row['video_id']}::{row['variant']}" for row in collapse_rows],
                "gate_eligible_updates": total_updates,
                "gate_acceptance_rate": gate_rate,
                "gate_accepted_incorrect_updates": accepted_incorrect_updates,
                "gate_accepted_incorrect_update_rate": (
                    accepted_incorrect_updates / accepted_updates if accepted_updates else 0.0
                ),
                "gate_degeneracy_detected": gate_degenerate,
                "teacher_forced_rank1_rate": teacher_rank1,
                "gated_rollout_video_macro_IDF1": gated_idf1,
                "teacher_forced_vs_rollout_gap": teacher_rollout_gap,
                "teacher_forced_vs_rollout_gap_failed": gap_failed,
                "pass": bool(gradient["method_ready"] and not collapse_rows and not gate_degenerate and not gap_failed),
            }
        )
    checks_ready = bool(checks) and all(check["pass"] for check in checks)
    subset_scope_blocks_readiness = h51.allow_subset
    ready = checks_ready and not subset_scope_blocks_readiness
    return {
        "status": H51_DECISION_READY if ready else H51_DECISION_STOP,
        "method_ready": ready,
        "diagnostic_pipeline_passed": True,
        "scope": "bounded_subset_diagnostic" if h51.allow_subset else "full_development_diagnostic",
        "decision_confirmatory": False,
        "subset_scope_blocks_readiness": subset_scope_blocks_readiness,
        "checks": checks,
        "thresholds": {
            "collapse_new_track_rate": h51.collapse_new_track_rate,
            "collapse_prediction_inflation": h51.collapse_prediction_inflation,
            "gate_degeneracy_low": h51.gate_degeneracy_low,
            "gate_degeneracy_high": h51.gate_degeneracy_high,
            "min_gate_updates": h51.min_gate_updates,
            "max_teacher_rollout_idf1_gap": h51.max_teacher_rollout_idf1_gap,
            "diagnostic_not_confirmatory": True,
        },
        "source_h5_interpretation": "failed_method_diagnosis_not_method_improvement",
        "checkpoint_retrained": False,
        "checkpoint_modified": False,
        "final_test_unlocked": False,
        "final_test_read": False,
    }


def repository_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"Cannot inspect git state: {result.stderr.decode(errors='replace').strip()}")
        return result.stdout.decode(errors="replace")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    tracked_diff = run("diff", "--no-ext-diff", "--binary", "HEAD")
    return {
        "commit": git_head(repo),
        "dirty": bool(status),
        "tracked_diff_sha256": sha256_text(tracked_diff),
        "worktree_status_sha256": sha256_text(status),
        "user_file_contents_recorded": False,
    }


def write_final_report(
    output: Path,
    *,
    input_audit: dict[str, Any],
    decision: dict[str, Any],
    models: Sequence[str],
    device: torch.device,
    test_only: bool,
) -> dict[str, Any]:
    guide = """# H5.1 closed-loop diagnostic result guide

This directory is a post-hoc **development-only diagnostic**, not an H5.2 method result.

- `h51_gradient_coverage.csv`: runtime backward coverage for every BeeTrackQuery parameter.
- `h51_path_audit.json`: observed training, offline teacher-forced, and causal rollout paths.
- `h51_teacher_forced_queries.csv`: GT-conditioned offline one-step audit (`offline_gt_audit=true`).
- `h51_rollout_observation_diagnostics.csv`: deployable causal decisions and rejection causes.
- `h51_frame_diagnostics.csv`: frame-level candidate activity.
- `h51_variant_summary.csv`: continuity, collapse and frozen-H3 inflation health.
- `h51_rejection_summary.csv`: complete rejection taxonomy counts.
- `h51_replay_equivalence.json`: exact assignment-contract comparison with the hashed source H5
  assignments (or the current H5 tracker as a test-only synthetic oracle).
- `h51_decision.json`: diagnostic fuse decision. It never unlocks final test.
- `h51_run_metadata.json`: input provenance, decision scope, git state, and actual torch device.
- `h51_artifact_signatures.json`: SHA-256 manifest for the completed auditable artifacts.
- `h51_result_guide.md`: this interpretation boundary; `logs/` contains atomic progress state.

H5 checkpoints are read-only and are neither retrained nor modified. GT identity is used only in the
separate teacher-forced offline audit and metric calculation, never in rollout association decisions.
The gradient audit recreates the checkpoint-pinned H5 v2 batched loss on the current branch. Legacy
H5 metadata did not store a source diff or training-binary digest, so this is not direct attestation of
the exact historical server binary. Six memory-path parameters are expected to be absent from the
autograd graph. The deterministic synthetic multi-candidate probe also records a finite zero gradient
for `pair_head.3.bias`, consistent with common-logit-shift invariance on that audited batch; this is
not asserted for the single-candidate reliability path or every possible real batch. Any additional
missing, non-finite, or zero gradient is reported and stops readiness without automatic attribution.
"""
    atomic_write_text(output / "h51_result_guide.md", guide)
    repo = Path(__file__).resolve().parents[3]
    actual_device = str(device)
    if device.type == "cuda":
        actual_device = f"{device}:{torch.cuda.get_device_name(device)}"
    metadata = {
        "status": "SERVER_VALIDATION_PENDING" if test_only else "completed_development_diagnostic",
        "experiment": "H5.1_closed_loop_diagnostic",
        "models": list(models), "test_only": test_only,
        "test_only_encoder": test_only, "real_experiment_result": not test_only,
        "source_h5_failed_method_diagnosis": True,
        "checkpoint_retrained": False, "checkpoint_modified": False,
        "input_audit": input_audit, "decision": decision,
        "repository": repository_state(repo),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "actual_torch_device": actual_device,
        },
        **runtime_validation_fields(device, test_only),
        "diagnostic_not_confirmatory": True,
        "fixed_detector_boxes": "NOT_RUN_H51",
        "final_test_unlocked": False, "final_test_read": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output / "h51_run_metadata.json", metadata)
    artifacts = {
        name: sha256_file(output / name)
        for name in H51_REQUIRED_OUTPUTS
        if name not in {"h51_run_metadata.json", "h51_artifact_signatures.json"}
        and (output / name).is_file()
    }
    atomic_write_json(output / "h51_artifact_signatures.json", {
        "format_version": 1, "artifacts": artifacts,
        "signature": sha256_text(canonical_json(artifacts)), "final_test_read": False,
    })
    metadata["artifact_signature"] = sha256_file(output / "h51_artifact_signatures.json")
    atomic_write_json(output / "h51_run_metadata.json", metadata)
    return metadata
