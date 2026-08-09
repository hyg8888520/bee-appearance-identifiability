"""H4 decisions, clustered uncertainty, failure audit, and dependency-free SVGs."""

from __future__ import annotations

import html
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from ..config import ExperimentConfig
from ..utils import atomic_write_json, atomic_write_text, git_head, sha256_file
from .core import require_h4, validate_h4_inputs
from .io import read_csv, write_csv
from .tracker import H4_PRIMARY_VARIANT


def _bootstrap(
    paired: Sequence[dict[str, str]], replicates: int, seed: int
) -> list[dict[str, Any]]:
    metrics = (
        "AssA_gain_vs_immediate", "IDF1_gain_vs_immediate",
        "HOTA_gain_vs_immediate", "IDSW_reduction_vs_immediate",
    )
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in paired:
        groups[(row["model"], row["variant"])].append(row)
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for (model, variant), rows in sorted(groups.items()):
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            estimates = np.empty(replicates, dtype=np.float64)
            for index in range(replicates):
                sample = rng.integers(0, len(values), size=len(values))
                estimates[index] = float(np.mean(values[sample]))
            output.append(
                {
                    "model": model,
                    "variant": variant,
                    "metric": metric,
                    "cluster_unit": "video_id",
                    "cluster_count": len(values),
                    "point_estimate": float(np.mean(values)),
                    "ci95_low": float(np.quantile(estimates, 0.025)),
                    "ci95_high": float(np.quantile(estimates, 0.975)),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                    "final_test_read": False,
                }
            )
    return output


def _method_decision(
    summary: Sequence[dict[str, str]],
    paired: Sequence[dict[str, str]],
    models: Sequence[str],
    tolerance: float,
    min_nonharmed_videos: int,
) -> dict[str, Any]:
    lookup = {(row["model"], row["variant"]): row for row in summary}
    checks: list[dict[str, Any]] = []
    for model in models:
        baseline = lookup.get((model, "immediate_commit"))
        primary = lookup.get((model, H4_PRIMARY_VARIANT))
        if baseline is None or primary is None:
            raise RuntimeError(f"H4 summary is missing the primary comparison for {model}")
        selected = [
            row for row in paired
            if row["model"] == model and row["variant"] == H4_PRIMARY_VARIANT
        ]
        idsw_reduction = int(float(baseline["IDSW"])) - int(float(primary["IDSW"]))
        idf1_gain = float(primary["IDF1"]) - float(baseline["IDF1"])
        hota_gain = float(primary["HOTA"]) - float(baseline["HOTA"])
        nonharmed = sum(
            float(row["IDF1_gain_vs_immediate"]) >= -tolerance
            and float(row["HOTA_gain_vs_immediate"]) >= -tolerance
            for row in selected
        )
        passed = (
            idsw_reduction > 0
            and idf1_gain >= -tolerance
            and hota_gain >= -tolerance
            and nonharmed >= min_nonharmed_videos
        )
        checks.append(
            {
                "model": model,
                "IDSW_reduction": idsw_reduction,
                "IDSW_pass": idsw_reduction > 0,
                "IDF1_gain": idf1_gain,
                "IDF1_noninferiority_pass": idf1_gain >= -tolerance,
                "HOTA_gain": hota_gain,
                "HOTA_noninferiority_pass": hota_gain >= -tolerance,
                "nonharmed_video_count": nonharmed,
                "nonharmed_video_pass": nonharmed >= min_nonharmed_videos,
                "pass": passed,
            }
        )
    passed = bool(checks) and all(item["pass"] for item in checks)
    return {
        "status": "GO_FREEZE_NEXT_PROTOCOL" if passed else "STOP_METHOD_NOT_SUPPORTED",
        "gate_passed": passed,
        "primary_variant": H4_PRIMARY_VARIANT,
        "noninferiority_tolerance": tolerance,
        "min_nonharmed_videos": min_nonharmed_videos,
        "model_checks": checks,
        "final_test_action": "REMAINS_LOCKED_REQUIRES_NEW_FROZEN_PROTOCOL",
        "final_test_read": False,
    }


def _switch_observations(rows: Sequence[dict[str, str]]) -> set[tuple[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["variant"], row["gt_identity"])].append(row)
    switches: set[tuple[str, str]] = set()
    for selected in groups.values():
        previous: str | None = None
        for row in sorted(selected, key=lambda item: (int(item["frame_id"]), item["observation_id"])):
            current = row["predicted_track_id"]
            if previous is not None and current != previous:
                switches.add((row["model"], row["observation_id"]))
            previous = current
    return switches


def _failure_cases(
    assignments: Sequence[dict[str, str]],
    events: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    baseline_rows = [row for row in assignments if row["variant"] == "immediate_commit"]
    primary_rows = [row for row in assignments if row["variant"] == H4_PRIMARY_VARIANT]
    baseline_switches = _switch_observations(baseline_rows)
    primary_switches = _switch_observations(primary_rows)
    lookup = {(row["model"], row["observation_id"]): row for row in primary_rows}
    baseline_lookup = {(row["model"], row["observation_id"]): row for row in baseline_rows}
    output: list[dict[str, Any]] = []
    for model, observation_id in sorted(
        {(row["model"], row["observation_id"]) for row in baseline_rows + primary_rows}
    ):
        baseline_switch = (model, observation_id) in baseline_switches
        primary_switch = (model, observation_id) in primary_switches
        if baseline_switch == primary_switch:
            continue
        row = lookup.get((model, observation_id)) or baseline_lookup[(model, observation_id)]
        output.append(
            {
                "case_type": (
                    "baseline_switch_resolved_by_primary"
                    if baseline_switch else "switch_introduced_by_primary"
                ),
                "model": model,
                "video_id": row["video_id"],
                "frame_id": row["frame_id"],
                "observation_id": observation_id,
                "gt_identity": row["gt_identity"],
                "baseline_switch": baseline_switch,
                "primary_switch": primary_switch,
                "primary_event_id": row.get("event_id", ""),
                "primary_decision_latency_frames": row.get("decision_latency_frames", ""),
                "analysis_uses_gt": True,
                "final_test_read": False,
            }
        )
    ranked_events = [
        row for row in events
        if row["variant"] == H4_PRIMARY_VARIANT and row.get("decision_score_margin", "") != ""
    ]
    ranked_events.sort(key=lambda row: (float(row["decision_score_margin"]), row["model"], row["event_id"]))
    for row in ranked_events[:20]:
        output.append(
            {
                "case_type": "lowest_primary_decision_margin",
                "model": row["model"],
                "video_id": row["video_id"],
                "frame_id": row["start_frame"],
                "observation_id": "",
                "gt_identity": "",
                "baseline_switch": "",
                "primary_switch": "",
                "primary_event_id": row["event_id"],
                "primary_decision_latency_frames": row["realized_horizon"],
                "decision_score_margin": row["decision_score_margin"],
                "analysis_uses_gt": False,
                "final_test_read": False,
            }
        )
    return output


def _line_svg(path: Path, rows: Sequence[dict[str, str]]) -> None:
    width, height = 720, 420
    left, top, chart_w, chart_h = 80, 45, 590, 300
    horizons = sorted({int(row["horizon"]) for row in rows})
    models = sorted({row["model"] for row in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="360" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">H4 oracle joint recoverability (development only)</text>']
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="black"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="black"/>')
    for tick in range(0, 6):
        value = tick / 5
        y = top + chart_h * (1 - value)
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    for model_index, model in enumerate(models):
        selected = {int(row["horizon"]): float(row["joint_recoverable_fraction"]) for row in rows if row["model"] == model}
        points = []
        for index, horizon in enumerate(horizons):
            x = left + (chart_w * index / max(1, len(horizons) - 1))
            y = top + chart_h * (1 - selected.get(horizon, 0.0))
            points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[model_index]}"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[model_index]}" stroke-width="2"/>')
        parts.append(f'<text x="{left + 15 + model_index * 150}" y="390" font-family="sans-serif" font-size="13" fill="{colors[model_index]}">{html.escape(model)}</text>')
    for index, horizon in enumerate(horizons):
        x = left + (chart_w * index / max(1, len(horizons) - 1))
        parts.append(f'<text x="{x:.1f}" y="{top + chart_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{horizon}</text>')
    parts.append('</svg>')
    atomic_write_text(path, "\n".join(parts) + "\n")


def _idsw_svg(path: Path, rows: Sequence[dict[str, str]]) -> None:
    selected = [row for row in rows if row["variant"] in {"immediate_commit", H4_PRIMARY_VARIANT}]
    labels = [f"{row['model']}\n{row['variant'].replace('fixed_lag_isolated_memory_', '')}" for row in selected]
    values = [int(float(row["IDSW"])) for row in selected]
    width, height = 760, 440
    maximum = max(values or [1])
    bar_w = 520 / max(1, len(values))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="380" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Immediate vs primary fixed-lag ID switches</text>']
    for index, (label, value) in enumerate(zip(labels, values)):
        x = 100 + index * bar_w
        height_value = 300 * value / maximum
        y = 350 - height_value
        color = "#d62728" if "immediate" in label else "#1f77b4"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.7:.1f}" height="{height_value:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w * .35:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value}</text>')
        parts.append(f'<text x="{x + bar_w * .35:.1f}" y="375" text-anchor="middle" font-family="sans-serif" font-size="10">{html.escape(label)}</text>')
    parts.append('</svg>')
    atomic_write_text(path, "\n".join(parts) + "\n")


def generate_h4_report(
    config: ExperimentConfig, model_names: Sequence[str]
) -> dict[str, Any]:
    h4 = require_h4(config)
    inputs = validate_h4_inputs(config)
    output = config.paths.output_root
    gate = json.loads((output / "h4_protocol_decision.json").read_text(encoding="utf-8"))
    horizon = read_csv(output / "h4_horizon_summary.csv", "H4 horizon summary")
    figures = output / "h4_figures"
    _line_svg(figures / "recoverability_by_horizon.svg", horizon)
    tracking_path = output / "h4_tracking_metadata.json"
    method_decision: dict[str, Any]
    bootstrap: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    result_hashes: dict[str, str] = {
        "recoverability_events": sha256_file(output / "h4_recoverability_events.csv"),
        "horizon_summary": sha256_file(output / "h4_horizon_summary.csv"),
        "protocol_decision": sha256_file(output / "h4_protocol_decision.json"),
    }
    if tracking_path.is_file():
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
        if tracking.get("status") != "completed" or tracking.get("final_test_read") is not False:
            raise RuntimeError("H4 tracking metadata is incomplete or unsafe")
        if tracking.get("protocol_decision_sha256") != sha256_file(
            output / "h4_protocol_decision.json"
        ):
            raise RuntimeError("H4 tracking was produced under a different audit decision")
        summary = read_csv(output / "h4_summary.csv", "H4 summary")
        paired = read_csv(output / "h4_paired_video_metrics.csv", "H4 paired video metrics")
        assignments = read_csv(output / "h4_assignments.csv", "H4 assignments")
        events = read_csv(output / "h4_conflict_events.csv", "H4 conflict events")
        bootstrap = _bootstrap(paired, h4.bootstrap_replicates, config.runtime.seed)
        failure_cases = _failure_cases(assignments, events)
        write_csv(output / "h4_video_cluster_bootstrap.csv", bootstrap)
        write_csv(output / "h4_failure_cases.csv", failure_cases)
        _idsw_svg(figures / "idsw_immediate_vs_primary.svg", summary)
        method_decision = _method_decision(
            summary, paired, model_names,
            h4.noninferiority_tolerance, h4.min_nonharmed_videos,
        )
        result_hashes.update(
            {
                "assignments": sha256_file(output / "h4_assignments.csv"),
                "summary": sha256_file(output / "h4_summary.csv"),
                "paired_video_metrics": sha256_file(output / "h4_paired_video_metrics.csv"),
                "tracking_metadata": sha256_file(tracking_path),
            }
        )
        report_status = (
            "completed_subset_smoke_not_experiment"
            if tracking.get("result_status") == "PROTOCOL_GATE_OVERRIDDEN_FOR_SUBSET_SMOKE_NOT_EXPERIMENT"
            else "completed_development_gt_boxes"
        )
    else:
        method_decision = {
            "status": "NOT_EVALUATED_RECOVERABILITY_GATE_STOP",
            "gate_passed": False,
            "primary_variant": H4_PRIMARY_VARIANT,
            "final_test_read": False,
        }
        write_csv(output / "h4_video_cluster_bootstrap.csv", [])
        write_csv(output / "h4_failure_cases.csv", [])
        report_status = "stopped_after_recoverability_audit"
    atomic_write_json(output / "h4_method_decision.json", method_decision)
    atomic_write_text(
        output / "h4_resolved_config.yaml",
        yaml.safe_dump(config.serializable(), sort_keys=False, allow_unicode=True),
    )
    atomic_write_text(
        output / "h4_logs" / "report.log",
        f"{datetime.now(timezone.utc).isoformat()} H4 report={report_status}; final_test_read=false\n",
    )
    metadata = {
        "status": report_status,
        "experiment": "H4_deferred_identity_commitment",
        "protocol_id": inputs.audit["protocol_id"],
        "models": list(model_names),
        "primary_variant": H4_PRIMARY_VARIANT,
        "project_git_commit": git_head(Path(__file__).resolve().parents[3]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "input_audit": inputs.audit,
        "recoverability_decision": gate,
        "method_decision": method_decision,
        "result_hashes": result_hashes,
        "counts": {
            "recoverability_horizon_rows": len(horizon),
            "bootstrap_rows": len(bootstrap),
            "failure_case_rows": len(failure_cases),
        },
        "claims_allowed": [
            "Development-only oracle recoverability if explicitly labeled as GT diagnostic.",
            "Causal fixed-lag GT-box association differences if method evaluation ran.",
            "Effect of branch-isolated memory relative to frozen-memory and immediate ablations.",
        ],
        "claims_forbidden": [
            "MHT or fixed-lag inference is novel by itself.",
            "Oracle recoverability is a deployable tracker result.",
            "GT-box results are end-to-end MOT results.",
            "Development results are final-test results.",
            "TOPIC is a component of H4.",
        ],
        "fixed_detector_boxes": "SERVER_VALIDATION_PENDING",
        "final_test": "LOCKED_REQUIRES_NEW_PROTOCOL",
        "final_test_read": False,
    }
    atomic_write_json(output / "h4_run_metadata.json", metadata)
    return metadata
