"""H4.1 uncertainty, failure audit, figures, and claim-boundary metadata."""

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
from ..h4.io import read_csv, write_csv
from ..h4.report import _bootstrap
from ..utils import atomic_write_json, atomic_write_text, git_head, sha256_file
from .core import require_h41, validate_h41_inputs
from .protocol import H41_PRIMARY_VARIANT


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
        primary = lookup.get((model, H41_PRIMARY_VARIANT))
        if baseline is None or primary is None:
            raise RuntimeError(f"H4.1 summary lacks the primary comparison for {model}")
        selected = [
            row
            for row in paired
            if row["model"] == model and row["variant"] == H41_PRIMARY_VARIANT
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
    passed = bool(checks) and all(row["pass"] for row in checks)
    return {
        "status": "GO_FREEZE_CONFIRMATORY_PROTOCOL" if passed else "STOP_METHOD_NOT_SUPPORTED",
        "gate_passed": passed,
        "primary_variant": H41_PRIMARY_VARIANT,
        "noninferiority_tolerance": tolerance,
        "min_nonharmed_videos": min_nonharmed_videos,
        "model_checks": checks,
        "design_timing": "EXPLORATORY_AFTER_H4_V1",
        "final_test_action": "REMAINS_LOCKED_REQUIRES_NEW_CONFIRMATORY_PROTOCOL",
        "final_test_read": False,
    }


def _switch_observations(rows: Sequence[dict[str, str]]) -> set[tuple[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["variant"], row["gt_identity"])].append(row)
    switches: set[tuple[str, str]] = set()
    for selected in groups.values():
        previous: str | None = None
        for row in sorted(
            selected, key=lambda item: (int(item["frame_id"]), item["observation_id"])
        ):
            current = row["predicted_track_id"]
            if previous is not None and current != previous:
                switches.add((row["model"], row["observation_id"]))
            previous = current
    return switches


def _failure_cases(
    assignments: Sequence[dict[str, str]], events: Sequence[dict[str, str]]
) -> list[dict[str, Any]]:
    baseline = [row for row in assignments if row["variant"] == "immediate_commit"]
    primary = [row for row in assignments if row["variant"] == H41_PRIMARY_VARIANT]
    baseline_switches = _switch_observations(baseline)
    primary_switches = _switch_observations(primary)
    primary_lookup = {(row["model"], row["observation_id"]): row for row in primary}
    baseline_lookup = {(row["model"], row["observation_id"]): row for row in baseline}
    output: list[dict[str, Any]] = []
    identifiers = {
        (row["model"], row["observation_id"]) for row in baseline + primary
    }
    for model, observation_id in sorted(identifiers):
        before = (model, observation_id) in baseline_switches
        after = (model, observation_id) in primary_switches
        if before == after:
            continue
        row = primary_lookup.get((model, observation_id)) or baseline_lookup[(model, observation_id)]
        output.append(
            {
                "case_type": (
                    "baseline_switch_resolved_by_primary"
                    if before else "switch_introduced_by_primary"
                ),
                "model": model,
                "video_id": row["video_id"],
                "frame_id": row["frame_id"],
                "observation_id": observation_id,
                "gt_identity": row["gt_identity"],
                "baseline_switch": before,
                "primary_switch": after,
                "primary_event_id": row.get("event_id", ""),
                "primary_decision_latency_frames": row.get("decision_latency_frames", ""),
                "analysis_uses_gt": True,
                "final_test_read": False,
            }
        )
    ranked = [
        row
        for row in events
        if row["variant"] == H41_PRIMARY_VARIANT
        and row.get("normalized_decision_margin", "") != ""
    ]
    ranked.sort(
        key=lambda row: (
            float(row["normalized_decision_margin"]), row["model"], row["event_id"]
        )
    )
    for row in ranked[:20]:
        output.append(
            {
                "case_type": "lowest_primary_normalized_decision_margin",
                "model": row["model"],
                "video_id": row["video_id"],
                "frame_id": row["start_frame"],
                "observation_id": "",
                "gt_identity": "",
                "baseline_switch": "",
                "primary_switch": "",
                "primary_event_id": row["event_id"],
                "primary_decision_latency_frames": row["realized_horizon"],
                "normalized_decision_margin": row["normalized_decision_margin"],
                "early_commit": row.get("early_commit", ""),
                "analysis_uses_gt": False,
                "final_test_read": False,
            }
        )
    return output


def _recovery_svg(path: Path, rows: Sequence[dict[str, str]]) -> None:
    width, height = 760, 430
    left, top, chart_w, chart_h = 80, 45, 620, 300
    deadlines = sorted({int(row["deadline"]) for row in rows})
    models = sorted({row["model"] for row in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">H4.1 cumulative joint recoverability (exploratory development)</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="black"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + chart_h * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    for model_index, model in enumerate(models):
        selected = {
            int(row["deadline"]): float(row["joint_recoverable_fraction"])
            for row in rows if row["model"] == model
        }
        points: list[str] = []
        for index, deadline in enumerate(deadlines):
            x = left + chart_w * index / max(1, len(deadlines) - 1)
            y = top + chart_h * (1 - selected.get(deadline, 0.0))
            points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[model_index]}"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[model_index]}" stroke-width="2"/>')
        parts.append(f'<text x="{left + 15 + model_index * 160}" y="400" font-family="sans-serif" font-size="13" fill="{colors[model_index]}">{html.escape(model)}</text>')
    for index, deadline in enumerate(deadlines):
        x = left + chart_w * index / max(1, len(deadlines) - 1)
        parts.append(f'<text x="{x:.1f}" y="{top + chart_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">H={deadline}</text>')
    parts.append("</svg>")
    atomic_write_text(path, "\n".join(parts) + "\n")


def _idsw_svg(path: Path, rows: Sequence[dict[str, str]]) -> None:
    selected = [
        row
        for row in rows
        if row["variant"] in {"immediate_commit", H41_PRIMARY_VARIANT}
    ]
    values = [int(float(row["IDSW"])) for row in selected]
    maximum = max(1, max(values, default=0))
    width, height = 760, 440
    bar_width = 540 / max(1, len(selected))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Immediate vs adaptive-isolated ID switches</text>',
    ]
    for index, (row, value) in enumerate(zip(selected, values)):
        x = 100 + index * bar_width
        bar_height = 300 * value / maximum
        y = 350 - bar_height
        color = "#d62728" if row["variant"] == "immediate_commit" else "#1f77b4"
        label = f"{row['model']}:{'immediate' if row['variant'] == 'immediate_commit' else 'adaptive'}"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * .7:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_width * .35:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value}</text>')
        parts.append(f'<text x="{x + bar_width * .35:.1f}" y="375" text-anchor="middle" font-family="sans-serif" font-size="10">{html.escape(label)}</text>')
    parts.append("</svg>")
    atomic_write_text(path, "\n".join(parts) + "\n")


def _result_guide(
    gate: dict[str, Any], method: dict[str, Any], tracking_ran: bool
) -> str:
    return f"""# H4.1 结果阅读说明

本目录是观察 H4-v1 development 结果后定义的探索性实验，不是事前注册的确认性实验。

## 先看什么

1. `h41_source_h4_crosscheck.json`：必须为 `passed`，证明旧 H4-v1 的逐时点结果和 STOP 结论未被改写。
2. `h41_cumulative_summary.csv`：回答“截至 H 帧是否至少出现过一次可恢复证据”；分母始终是全部 baseline IDSW。
3. `h41_protocol_decision.json`：累计恢复门禁，当前状态为 `{gate.get('status')}`。
4. `h41_summary.csv` 与 `h41_paired_video_metrics.csv`：只有门禁通过并实际运行 tracker 后才有方法结果。
5. `h41_method_decision.json`：当前状态为 `{method.get('status')}`。

## 解释边界

- H4-v1 的精确 `t+10` 门禁失败仍然成立；H4.1 使用的是不同 estimand（统计量定义）：窗口内曾可恢复。
- Oracle（使用真值的离线诊断）不进入实际关联决策。
- 当前是 GT-box（真值检测框）开发集实验，不能当作端到端 MOT 结果。
- H4.1 是探索性结果；即使通过，也只能支持另行冻结确认性协议，不能直接读取 final test。
- fixed-detector 和 RTX 4090 真实服务器核验仍为 `SERVER_VALIDATION_PENDING`。

tracking_ran: `{str(tracking_ran).lower()}`<br>
final_test_read: `false`
"""


def generate_h41_report(
    config: ExperimentConfig, model_names: Sequence[str]
) -> dict[str, Any]:
    h41 = require_h41(config)
    inputs = validate_h41_inputs(config)
    output = config.paths.output_root
    gate = json.loads(
        (output / "h41_protocol_decision.json").read_text(encoding="utf-8")
    )
    cumulative = read_csv(
        output / "h41_cumulative_summary.csv", "H4.1 cumulative summary"
    )
    figures = output / "h41_figures"
    _recovery_svg(figures / "cumulative_recoverability.svg", cumulative)
    tracking_path = output / "h41_tracking_metadata.json"
    bootstrap: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    result_hashes = {
        "exact_events": sha256_file(output / "h41_exact_recoverability_events.csv"),
        "event_timelines": sha256_file(output / "h41_event_timelines.csv"),
        "cumulative_summary": sha256_file(output / "h41_cumulative_summary.csv"),
        "protocol_decision": sha256_file(output / "h41_protocol_decision.json"),
        "source_h4_crosscheck": sha256_file(output / "h41_source_h4_crosscheck.json"),
    }
    if tracking_path.is_file():
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
        if tracking.get("status") != "completed" or tracking.get("final_test_read") is not False:
            raise RuntimeError("H4.1 tracking metadata is incomplete or unsafe")
        if tracking.get("protocol_decision_sha256") != sha256_file(
            output / "h41_protocol_decision.json"
        ):
            raise RuntimeError("H4.1 tracking used a different cumulative audit decision")
        summary = read_csv(output / "h41_summary.csv", "H4.1 summary")
        paired = read_csv(
            output / "h41_paired_video_metrics.csv", "H4.1 paired metrics"
        )
        assignments = read_csv(output / "h41_assignments.csv", "H4.1 assignments")
        events = read_csv(output / "h41_conflict_events.csv", "H4.1 conflict events")
        bootstrap = _bootstrap(paired, h41.bootstrap_replicates, config.runtime.seed)
        failure_cases = _failure_cases(assignments, events)
        write_csv(output / "h41_video_cluster_bootstrap.csv", bootstrap)
        write_csv(output / "h41_failure_cases.csv", failure_cases)
        _idsw_svg(figures / "idsw_immediate_vs_adaptive.svg", summary)
        method = _method_decision(
            summary,
            paired,
            model_names,
            h41.noninferiority_tolerance,
            h41.min_nonharmed_videos,
        )
        result_hashes.update(
            {
                "assignments": sha256_file(output / "h41_assignments.csv"),
                "summary": sha256_file(output / "h41_summary.csv"),
                "paired_video_metrics": sha256_file(output / "h41_paired_video_metrics.csv"),
                "tracking_metadata": sha256_file(tracking_path),
            }
        )
        report_status = (
            "completed_subset_smoke_not_experiment"
            if tracking.get("result_status")
            == "SUBSET_SMOKE_NOT_EXPERIMENT"
            else "completed_exploratory_development_gt_boxes"
        )
        tracking_ran = True
    else:
        method = {
            "status": "NOT_EVALUATED_CUMULATIVE_GATE_STOP",
            "gate_passed": False,
            "primary_variant": H41_PRIMARY_VARIANT,
            "final_test_read": False,
        }
        write_csv(output / "h41_video_cluster_bootstrap.csv", [])
        write_csv(output / "h41_failure_cases.csv", [])
        report_status = "stopped_after_window_recoverability_audit"
        tracking_ran = False
    atomic_write_json(output / "h41_method_decision.json", method)
    atomic_write_text(
        output / "h41_resolved_config.yaml",
        yaml.safe_dump(config.serializable(), sort_keys=False, allow_unicode=True),
    )
    atomic_write_text(output / "h41_result_guide.md", _result_guide(gate, method, tracking_ran))
    atomic_write_text(
        output / "h41_logs" / "report.log",
        f"{datetime.now(timezone.utc).isoformat()} H4.1 report={report_status}; final_test_read=false\n",
    )
    metadata = {
        "status": report_status,
        "experiment": "H4.1_window_recoverability_adaptive_commitment",
        "protocol_id": inputs.audit["protocol_id"],
        "models": list(model_names),
        "primary_variant": H41_PRIMARY_VARIANT,
        "design_timing": "DEFINED_AFTER_OBSERVING_H4_V1_DEVELOPMENT_AUDIT",
        "project_git_commit": git_head(Path(__file__).resolve().parents[3]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "input_audit": inputs.audit,
        "cumulative_recoverability_decision": gate,
        "method_decision": method,
        "result_hashes": result_hashes,
        "counts": {
            "cumulative_summary_rows": len(cumulative),
            "bootstrap_rows": len(bootstrap),
            "failure_case_rows": len(failure_cases),
        },
        "claims_allowed": [
            "Exploratory development-only window recoverability when explicitly labeled as a GT oracle diagnostic.",
            "Causal adaptive-commitment GT-box association differences if method evaluation ran.",
            "Early-commit and branch-memory-isolation ablations on frozen embeddings.",
        ],
        "claims_forbidden": [
            "H4.1 was preregistered before seeing H4-v1 development results.",
            "The H4-v1 exact-t+10 STOP conclusion was overturned.",
            "Oracle recoverability is a deployable tracker metric.",
            "Adaptive stopping or multiple-hypothesis tracking is novel by itself.",
            "GT-box results are end-to-end MOT or final-test results.",
            "TOPIC is a component of H4.1.",
        ],
        "fixed_detector_boxes": "SERVER_VALIDATION_PENDING",
        "server_validation": "SERVER_VALIDATION_PENDING",
        "final_test": "LOCKED_REQUIRES_NEW_CONFIRMATORY_PROTOCOL",
        "final_test_read": False,
    }
    atomic_write_json(output / "h41_run_metadata.json", metadata)
    return metadata
