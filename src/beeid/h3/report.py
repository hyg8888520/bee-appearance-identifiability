"""Reproducible H3 development report and video-cluster uncertainty."""

from __future__ import annotations

import csv
import io
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
from .core import require_h3, validate_h3_inputs
from .metrics import TRACKEVAL_COMMIT


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Required H3 result does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = tuple(rows[0]) if rows else ()
    buffer = io.StringIO(newline="")
    if fields:
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _video_bootstrap(
    paired_rows: Sequence[dict[str, str]], replicates: int, seed: int
) -> list[dict[str, Any]]:
    metrics = (
        "AssA_gain_vs_baseline",
        "IDF1_gain_vs_baseline",
        "HOTA_gain_vs_baseline",
        "IDSW_reduction_vs_baseline",
        "Frag_reduction_vs_baseline",
    )
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in paired_rows:
        groups[(row["model"], row["variant"])].append(row)
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for (model, variant), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: row["video_id"])
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in ordered], dtype=np.float64)
            estimates = np.empty(replicates, dtype=np.float64)
            for index in range(replicates):
                sample = rng.integers(0, len(values), size=len(values))
                estimates[index] = float(np.mean(values[sample]))
            output.append(
                {
                    "model": model,
                    "variant": variant,
                    "stage": "gt_detection_boxes",
                    "metric": metric,
                    "cluster_unit": "video_id",
                    "cluster_count": len(values),
                    "point_estimate": float(np.mean(values)),
                    "ci95_low": float(np.quantile(estimates, 0.025)),
                    "ci95_high": float(np.quantile(estimates, 0.975)),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                }
            )
    return output


def generate_h3_report(
    config: ExperimentConfig, model_names: Sequence[str]
) -> dict[str, Any]:
    h3 = require_h3(config)
    if h3.stage != "gt_detection_boxes":
        raise RuntimeError("H3 reporting currently requires completed GT-box development")
    inputs = validate_h3_inputs(config)
    output = config.paths.output_root
    tracking_metadata_path = output / "h3_tracking_metadata.json"
    if not tracking_metadata_path.is_file():
        raise RuntimeError("Run h3-track before h3-report")
    tracking_metadata = json.loads(tracking_metadata_path.read_text(encoding="utf-8"))
    if tracking_metadata.get("status") != "completed" or tracking_metadata.get("final_test_read") is not False:
        raise RuntimeError("H3 tracking metadata is incomplete or violates final-test lock")
    paired_rows = _read_csv(output / "h3_paired_video_metrics.csv")
    bootstrap = _video_bootstrap(
        paired_rows, h3.bootstrap_replicates, config.runtime.seed
    )
    _write_csv(output / "h3_video_cluster_bootstrap.csv", bootstrap)
    atomic_write_text(
        output / "h3_resolved_config.yaml",
        yaml.safe_dump(config.serializable(), sort_keys=False, allow_unicode=True),
    )
    log_text = (
        f"{datetime.now(timezone.utc).isoformat()} H3 GT-box development report generated; "
        "final_test_read=false; fixed_detector_boxes=SERVER_VALIDATION_PENDING\n"
    )
    atomic_write_text(output / "h3_logs" / "report.log", log_text)

    primary_models = {"resnet50", "dinov3"}
    evaluated_primary = sorted(primary_models & set(model_names))
    metadata = {
        "status": "completed_development_gt_boxes",
        "experiment": "H3_RAM_Bee",
        "protocol_id": "bee24-h3-ram-bee-frozen-v1",
        "stage": "gt_detection_boxes",
        "models": list(model_names),
        "variants": tracking_metadata["variants"],
        "fixed_detector_boxes": "SERVER_VALIDATION_PENDING",
        "primary_models_evaluated": evaluated_primary,
        "project_git_commit": git_head(Path(__file__).resolve().parents[3]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "input_audit": inputs.audit,
        "tracking_metadata_sha256": sha256_file(tracking_metadata_path),
        "threshold_artifact_sha256": sha256_file(output / "h3_thresholds.json"),
        "assignment_sha256": sha256_file(output / "h3_assignments.csv"),
        "metric_reference": {
            "name": "TrackEval",
            "commit": TRACKEVAL_COMMIT,
            "scope": (
                "GT-box direct correspondence; identity/HOTA association equations audited "
                "against pinned TrackEval. Fixed-detector metrics remain pending."
            ),
        },
        "counts": {
            "assignment_rows": tracking_metadata["assignment_row_count"],
            "videos": len(tracking_metadata["development_videos"]),
            "bootstrap_rows": len(bootstrap),
        },
        "claims_allowed": [
            "GT-box development association differences among the four frozen variants.",
            "Project-train-calibrated causal reliability controls memory and association.",
        ],
        "claims_forbidden": [
            "GT-box results are end-to-end detector/tracker results.",
            "Development results are final-test results.",
            "Continuous reliability weighting is superior unless it beats selective memory update.",
            "Official TOPIC AGW is split-clean.",
            "Frag is informative in an exact-GT-detection stage where it is zero by construction.",
        ],
        "next_gate": {
            "gt_detection_boxes": "completed_development",
            "fixed_detector_boxes": "SERVER_VALIDATION_PENDING",
            "final_test": "LOCKED_DO_NOT_READ",
        },
        "final_test_read": False,
    }
    atomic_write_json(output / "h3_run_metadata.json", metadata)
    return metadata
