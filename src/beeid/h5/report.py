"""H5 development report, calibration audit and predeclared selection decision."""

from __future__ import annotations

import csv
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from ..config import ExperimentConfig
from ..utils import atomic_write_json, atomic_write_text, git_head, sha256_file
from . import H5_PRIMARY_VARIANT
from .core import require_h5, validate_h5_inputs


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"H5 report input is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import io
    buffer = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=tuple(rows[0]), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _bootstrap(per_video: Sequence[dict[str, str]], replicates: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in per_video}):
        lookup = {(row["variant"], row["video_id"]): row for row in per_video if row["model"] == model}
        videos = sorted({row["video_id"] for row in per_video if row["model"] == model})
        for metric in ("IDF1", "AssA", "HOTA"):
            gains = np.asarray([
                float(lookup[(H5_PRIMARY_VARIANT, video)][metric])
                - float(lookup[("frozen_h3_baseline", video)][metric])
                for video in videos
            ])
            samples = np.asarray([
                rng.choice(gains, size=len(gains), replace=True).mean()
                for _ in range(replicates)
            ])
            output.append({
                "model": model, "variant": H5_PRIMARY_VARIANT, "metric": metric,
                "video_count": len(videos), "replicates": replicates,
                "mean_gain": float(gains.mean()),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
                "cluster_unit": "video_id",
            })
    return output


def _failures(assignments: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    switches: dict[tuple[str, str, str], bool] = {}
    previous: dict[tuple[str, str, str], str] = {}
    row_lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    ordered = sorted(
        assignments,
        key=lambda item: (
            item["model"], item["variant"], item["video_id"],
            int(item["frame_id"]), item["observation_id"],
        ),
    )
    for row in ordered:
        identity_key = (row["model"], row["variant"], row["gt_identity"])
        event_key = (row["model"], row["variant"], row["observation_id"])
        switches[event_key] = (
            identity_key in previous
            and previous[identity_key] != row["predicted_track_id"]
        )
        previous[identity_key] = row["predicted_track_id"]
        row_lookup[event_key] = row
    output: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in assignments}):
        observation_ids = sorted({row["observation_id"] for row in assignments if row["model"] == model})
        for observation_id in observation_ids:
            baseline = switches.get((model, "frozen_h3_baseline", observation_id), False)
            primary = switches.get((model, H5_PRIMARY_VARIANT, observation_id), False)
            if not baseline and not primary:
                continue
            row = row_lookup.get((model, H5_PRIMARY_VARIANT, observation_id))
            if row is None:
                continue
            category = (
                "recovered_baseline_switch" if baseline and not primary
                else "new_primary_switch" if primary and not baseline
                else "shared_switch"
            )
            output.append({
                "model": model, "category": category, "video_id": row["video_id"],
                "frame_id": row["frame_id"], "observation_id": observation_id,
                "gt_identity": row["gt_identity"],
                "primary_reliability": row.get("reliability", ""),
                "final_test_read": False,
            })
    priority = {"new_primary_switch": 0, "shared_switch": 1, "recovered_baseline_switch": 2}
    return sorted(
        output,
        key=lambda row: (priority[row["category"]], row["model"], row["video_id"], int(row["frame_id"])),
    )[:500]


def _calibration(assignments: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in assignments}):
        selected = [row for row in assignments if row["model"] == model and row["variant"] == H5_PRIMARY_VARIANT and row.get("reliability", "") not in {"", None}]
        previous: dict[str, str] = {}
        pairs: list[tuple[float, float]] = []
        for row in sorted(selected, key=lambda item: (item["video_id"], int(item["frame_id"]), item["observation_id"])):
            key = f"{row['video_id']}::{row['gt_identity']}"
            predicted = row["predicted_track_id"]
            correct = 1.0 if key not in previous or previous[key] == predicted else 0.0
            previous[key] = predicted
            pairs.append((float(row["reliability"]), correct))
        if not pairs:
            continue
        probabilities = np.asarray([item[0] for item in pairs])
        labels = np.asarray([item[1] for item in pairs])
        ece = 0.0
        for lower in np.linspace(0.0, 0.9, 10):
            mask = (probabilities >= lower) & (probabilities < lower + 0.1 + 1e-12)
            if mask.any():
                ece += float(mask.mean()) * abs(float(probabilities[mask].mean() - labels[mask].mean()))
        output.append({
            "model": model, "variant": H5_PRIMARY_VARIANT, "sample_count": len(pairs),
            "brier_score": float(np.mean((probabilities - labels) ** 2)),
            "ece_10_bin": ece, "mean_confidence": float(probabilities.mean()),
            "empirical_continuity": float(labels.mean()), "label_scope": "offline_gt_identity_continuity_audit",
        })
    return output


def _svg(path: Path, summary: Sequence[dict[str, str]]) -> None:
    models = sorted({row["model"] for row in summary})
    variants = ["frozen_h3_baseline", H5_PRIMARY_VARIANT]
    width, height = 700, 360
    values = {(row["model"], row["variant"]): float(row["IDF1"]) for row in summary}
    bars: list[str] = []
    colors = {"frozen_h3_baseline": "#7f8c8d", H5_PRIMARY_VARIANT: "#e67e22"}
    x = 90
    for model in models:
        for variant in variants:
            value = values.get((model, variant), 0.0)
            y = 300 - value * 240
            bars.append(f'<rect x="{x}" y="{y:.1f}" width="55" height="{value*240:.1f}" fill="{colors[variant]}"/><text x="{x+27}" y="{y-5:.1f}" text-anchor="middle" font-size="12">{value:.3f}</text>')
            x += 65
        bars.append(f'<text x="{x-65}" y="330" text-anchor="middle" font-size="13">{model}</text>')
        x += 55
    content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="20" y="25" font-size="18">H5 development GT-box IDF1</text><line x1="70" y1="300" x2="660" y2="300" stroke="black"/>{"".join(bars)}<text x="460" y="25" fill="#7f8c8d">baseline</text><text x="550" y="25" fill="#e67e22">BeeTrackQuery</text></svg>'
    atomic_write_text(path, content)


def generate_h5_report(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    inputs = validate_h5_inputs(config)
    h5 = require_h5(config)
    output = config.paths.output_root
    summary = _rows(output / "h5_summary.csv")
    per_video = _rows(output / "h5_per_video_metrics.csv")
    assignments = _rows(output / "h5_assignments.csv")
    lookup = {(row["model"], row["variant"]): row for row in summary}
    checks: list[dict[str, Any]] = []
    for model in model_names:
        baseline = lookup.get((model, "frozen_h3_baseline"))
        primary = lookup.get((model, H5_PRIMARY_VARIANT))
        if baseline is None or primary is None:
            raise RuntimeError(f"H5 summary lacks baseline/primary rows for {model}")
        noninferior = float(primary["IDF1"]) >= float(baseline["IDF1"]) - h5.noninferiority_tolerance and float(primary["HOTA"]) >= float(baseline["HOTA"]) - h5.noninferiority_tolerance
        idsw_reduced = int(primary["IDSW"]) < int(baseline["IDSW"])
        video_lookup = {(row["variant"], row["video_id"]): row for row in per_video if row["model"] == model}
        videos = sorted({row["video_id"] for row in per_video if row["model"] == model})
        nonharmed = sum(float(video_lookup[(H5_PRIMARY_VARIANT, video)]["IDF1"]) >= float(video_lookup[("frozen_h3_baseline", video)]["IDF1"]) - h5.noninferiority_tolerance for video in videos)
        checks.append({"model": model, "idf1_hota_noninferior": noninferior, "idsw_reduced": idsw_reduced, "nonharmed_videos": nonharmed, "minimum_nonharmed_videos": h5.min_nonharmed_videos, "pass": noninferior and idsw_reduced and nonharmed >= h5.min_nonharmed_videos})
    passed = all(row["pass"] for row in checks)
    decision = {"status": "GO_FIXED_DETECTOR_VALIDATION" if passed else "STOP_OR_REVISE_BEETRACKQUERY", "gate_passed": passed, "model_checks": checks, "development_only": True, "final_test_read": False}
    calibration = _calibration(assignments)
    bootstrap = _bootstrap(per_video, h5.bootstrap_replicates, config.runtime.seed)
    failures = _failures(assignments)
    _write_rows(output / "h5_reliability_calibration.csv", calibration)
    _write_rows(output / "h5_video_cluster_bootstrap.csv", bootstrap)
    _write_rows(output / "h5_failure_cases.csv", failures)
    atomic_write_json(output / "h5_method_decision.json", decision)
    (output / "h5_figures").mkdir(parents=True, exist_ok=True)
    _svg(output / "h5_figures" / "idf1_baseline_vs_beetrackquery.svg", summary)
    atomic_write_text(output / "h5_resolved_config.yaml", yaml.safe_dump(config.serializable(), sort_keys=False))
    guide = """# H5 BeeTrackQuery 结果说明\n\n- `h5_summary.csv`：各 backbone / variant 的 pooled GT-box 指标。\n- `h5_per_video_metrics.csv`：逐视频指标，用于检查收益是否集中在少数视频。\n- `h5_paired_video_metrics.csv`：相对冻结 H3 baseline 的配对差值。\n- `h5_reliability_calibration.csv`：更新可信度的 Brier score 与 10-bin ECE。\n- `h5_video_cluster_bootstrap.csv`：逐视频配对 bootstrap 置信区间。\n- `h5_failure_cases.csv`：baseline 与主方法身份切换的恢复、新增和共同失败。\n- `h5_method_decision.json`：预先固定的 development gate。\n- `h5_assignments.csv`：逐 observation 审计表，体积可能较大。\n\n这些结果使用 GT detection boxes，只评价身份关联，不是端到端 MOT。development 结果不能写成 final-test 结论。\n"""
    atomic_write_text(output / "h5_result_guide.md", guide)
    metadata = {
        "status": "completed_development_gt_boxes", "experiment": "H5_BeeTrackQuery",
        "models": list(model_names), "primary_variant": H5_PRIMARY_VARIANT,
        "project_git_commit": git_head(Path(__file__).resolve().parents[3]),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "input_audit": inputs.audit, "decision": decision,
        "result_hashes": {
            "assignments": sha256_file(output / "h5_assignments.csv"),
            "summary": sha256_file(output / "h5_summary.csv"),
            "decision": sha256_file(output / "h5_method_decision.json"),
        },
        "metric_scope": "development GT detection boxes; DetA=1 by construction",
        "fixed_detector_boxes": "SERVER_VALIDATION_PENDING",
        "real_gpu_training": "SERVER_VALIDATION_PENDING",
        "final_test_read": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output / "h5_run_metadata.json", metadata)
    return metadata
