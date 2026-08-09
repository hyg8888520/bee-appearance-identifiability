"""Dependency-light H2.5 report, provenance, and figure generation."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml
from PIL import Image, ImageDraw

from ..config import ExperimentConfig
from ..status import SERVER_VALIDATION_PENDING
from ..utils import atomic_write_json, atomic_write_text, sha256_file
from .core import validate_h25_inputs


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Required H2.5 result does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _project_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _paired_figure(rows: Sequence[dict[str, str]], directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    selected = [row for row in rows if row["window"] == "immediate"]
    figures: list[str] = []
    for model in sorted({row["model"] for row in selected}):
        values = [row for row in selected if row["model"] == model]
        if not values:
            continue
        width = 900
        height = max(260, 70 + 34 * len(values))
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 15), f"{model}: immediate Rank-1 gain vs unconditional", fill="black")
        zero_x = 650
        draw.line((zero_x, 45, zero_x, height - 20), fill=(80, 80, 80), width=1)
        for index, row in enumerate(values):
            y = 55 + index * 34
            label = f"{row['event_type']} / {row['strategy']}"
            gain = float(row["mean_rank1_gain_vs_unconditional"])
            draw.text((20, y), label[:72], fill="black")
            end = int(zero_x + max(-1.0, min(1.0, gain)) * 180)
            color = (35, 125, 75) if gain >= 0 else (180, 65, 55)
            draw.rectangle((min(zero_x, end), y + 16, max(zero_x + 1, end), y + 27), fill=color)
            draw.text((835, y), f"{gain:+.3f}", fill="black")
        path = directory / f"paired_immediate_{model}.png"
        image.save(path)
        figures.append(path.name)
    atomic_write_text(
        directory / "index.txt",
        "H2.5 controlled GT-track figures; positive bars favor the named strategy over unconditional update.\n"
        + "\n".join(figures)
        + ("\n" if figures else ""),
    )
    return figures


def generate_h25_report(
    config: ExperimentConfig,
    model_names: Sequence[str],
    *,
    test_only_encoder: bool = False,
) -> dict[str, Any]:
    observations, protocol, input_audit = validate_h25_inputs(config)
    output = config.paths.output_root
    required = (
        "h25_event_thresholds.json", "h25_events.csv", "h25_strategy_trajectories.csv",
        "h25_strategy_summary.csv", "h25_paired_summary.csv", "h25_cluster_bootstrap.csv",
        "h25_experiment_metadata.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError("Run h25-run before h25-report; missing: " + ", ".join(missing))
    events = _read_csv(output / "h25_events.csv")
    trajectories = _read_csv(output / "h25_strategy_trajectories.csv")
    paired = _read_csv(output / "h25_paired_summary.csv")
    bootstrap = _read_csv(output / "h25_cluster_bootstrap.csv")
    figures = _paired_figure(paired, output / "h25_figures")
    resolved = config.serializable()
    resolved.pop("config_path", None)
    resolved["frozen_protocol_sha256"] = sha256_file(config.h25.protocol_lock_path)  # type: ignore[union-attr]
    resolved["h3_protocol_sha256"] = input_audit["h3_protocol_sha256"]
    atomic_write_text(output / "h25_resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=False))
    logs = output / "h25_logs"
    logs.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    atomic_write_text(
        logs / "report.log",
        f"{generated} H2.5 refined memory-contamination report generated; "
        f"test_only_encoder={str(test_only_encoder).lower()}; final_test_read=false\n",
    )
    metadata = {
        "experiment": "H2.5_refined_memory_contamination",
        "status": SERVER_VALIDATION_PENDING if test_only_encoder else "completed_development_diagnostic",
        "generated_at_utc": generated,
        "project_git_commit": _project_commit(),
        "models_requested": list(model_names),
        "test_only_encoder": test_only_encoder,
        "official_topic_agw_role": "REFERENCE_ONLY_TRAINING_OVERLAP_UNCONFIRMED",
        "protocol_id": protocol["protocol_id"],
        "input_audit": input_audit,
        "final_test_read": False,
        "counts": {
            "observation_count": len(observations),
            "event_count": len(events),
            "trajectory_row_count": len(trajectories),
            "paired_summary_row_count": len(paired),
            "cluster_bootstrap_row_count": len(bootstrap),
            "figure_count": len(figures),
        },
        "artifacts": [*required, "h25_resolved_config.yaml", "h25_logs", "h25_figures"],
        "claims_allowed": [
            "Dynamic outcome-blind reliability signals define controlled contamination events.",
            "Four memory update strategies are compared on exactly paired GT-track targets.",
        ],
        "claims_forbidden": [
            "This is an end-to-end tracker result.",
            "Development-validation results are final-test results.",
            "Official TOPIC AGW is split-clean.",
            "Static bbox area or absolute Laplacian alone identifies contamination.",
        ],
        "limitations": [
            "Events are controlled wrong-identity injections on GT identity tracks.",
            "Empirical development percentiles diagnose mechanism behavior; H3 thresholds must be fit on project_train.",
            "Few videos make cluster-bootstrap intervals descriptive rather than definitive.",
        ],
    }
    atomic_write_json(output / "h25_run_metadata.json", metadata)
    return metadata
