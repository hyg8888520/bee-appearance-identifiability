"""Generate failure cases, metadata, and resolved run records."""

from __future__ import annotations

import csv
import io
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml
from PIL import Image, ImageDraw

from ..config import ExperimentConfig
from ..data.crops import load_crop
from ..data.manifest import load_manifest
from ..references import (
    DINOV3_COMMIT,
    FASTREID_COMMIT,
    TOPICTRACK_COMMIT,
    TRACKEVAL_COMMIT,
    TORCHVISION_COMMIT,
)
from ..status import SERVER_VALIDATION_PENDING
from ..utils import atomic_write_json, atomic_write_text, count_reasons, git_head, sha256_file


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = tuple(rows[0]) if rows else (
        "category", "video_id", "query_observation_id", "delta", "model_outcomes", "minimum_margin",
        "bbox_area", "clipped",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _failure_cases(rows: list[dict[str, str]], top_k: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["rank1"] != "":
            grouped[(row["video_id"], row["query_observation_id"], row["delta"])].append(row)
    base_rows: list[dict[str, Any]] = []
    for (video, observation_id, delta), model_rows in grouped.items():
        outcomes = {
            row["model"]: {
                "rank1": row["rank1"],
                "hard_rank1": row["hard_rank1"],
                "margin": row["margin"],
            }
            for row in model_rows
        }
        margins = [float(row["margin"]) for row in model_rows if row["margin"] != ""]
        base_rows.append(
            {
                "category": "correctness_combination:" + ",".join(
                    f"{name}={'correct' if value['rank1'] == 'True' else 'wrong'}"
                    for name, value in sorted(outcomes.items())
                ),
                "video_id": video,
                "query_observation_id": observation_id,
                "delta": int(delta),
                "model_outcomes": json.dumps(outcomes, sort_keys=True, separators=(",", ":")),
                "positive_observation_id": model_rows[0]["positive_observation_id"],
                "hard_negative_observation_id": next(
                    (
                        row["max_hard_negative_observation_id"]
                        for row in model_rows
                        if row["model"] == "dinov3" and row["max_hard_negative_observation_id"]
                    ),
                    next(
                        (
                            row["max_hard_negative_observation_id"]
                            for row in model_rows
                            if row["max_hard_negative_observation_id"]
                        ),
                        "",
                    ),
                ),
                "minimum_margin": min(margins) if margins else "",
                "bbox_area": float(model_rows[0]["bbox_area"]),
                "clipped": model_rows[0]["clipped"],
            }
        )
    selected: list[dict[str, Any]] = []
    combinations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        combinations[row["category"]].append(row)
    for category, values in sorted(combinations.items()):
        selected.extend(sorted(values, key=lambda item: (item["minimum_margin"] == "", item["minimum_margin"]))[:top_k])
    specific = {
        "dinov3_correct_resnet50_wrong": lambda outcomes: (
            outcomes.get("dinov3", {}).get("rank1") == "True"
            and outcomes.get("resnet50", {}).get("rank1") == "False"
        ),
        "dinov3_correct_topic_agw_wrong": lambda outcomes: (
            outcomes.get("dinov3", {}).get("rank1") == "True"
            and outcomes.get("topic_agw", {}).get("rank1") == "False"
        ),
        "dinov3_wrong_topic_agw_correct": lambda outcomes: (
            outcomes.get("dinov3", {}).get("rank1") == "False"
            and outcomes.get("topic_agw", {}).get("rank1") == "True"
        ),
        "all_three_wrong": lambda outcomes: (
            all(name in outcomes for name in ("resnet50", "dinov3", "topic_agw"))
            and all(outcomes[name].get("rank1") == "False" for name in ("resnet50", "dinov3", "topic_agw"))
        ),
    }
    for category, predicate in specific.items():
        matching = [row for row in base_rows if predicate(json.loads(row["model_outcomes"]))]
        for row in sorted(matching, key=lambda item: (item["minimum_margin"] == "", item["minimum_margin"]))[:top_k]:
            selected.append({**row, "category": category})
    for row in sorted(base_rows, key=lambda item: (item["minimum_margin"] == "", item["minimum_margin"]))[:top_k]:
        selected.append({**row, "category": "lowest_margin"})
    for row in sorted(base_rows, key=lambda item: item["bbox_area"])[:top_k]:
        selected.append({**row, "category": "smallest_target"})
    clipped = [row for row in base_rows if row["clipped"].lower() == "true"]
    for row in clipped[:top_k]:
        selected.append({**row, "category": "boundary_clipped"})
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for row in selected:
        key = (str(row["category"]), str(row["query_observation_id"]), int(row["delta"]))
        if key not in seen:
            deduplicated.append(row)
            seen.add(key)
    return deduplicated


def _environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["torch_cuda_build"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
    except ImportError:
        result["torch"] = "not installed in reporting interpreter"
    return result


def _project_git_commit() -> str:
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / ".git").exists():
            try:
                return git_head(parent)
            except RuntimeError:
                break
    return "GIT_COMMIT_UNAVAILABLE"


def _cache_signatures(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    if not config.cache_locations_path.is_file():
        return {}
    locations = json.loads(config.cache_locations_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    for model in model_names:
        location = locations.get(model) if isinstance(locations, dict) else None
        metadata_path = Path(location) / "cache_manifest.json" if location else None
        if metadata_path and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            result[model] = {
                "cache_directory": str(metadata_path.parent),
                "fingerprint": metadata.get("fingerprint"),
                "signature": metadata.get("signature"),
            }
    return result


def _render_failure_figures(
    config: ExperimentConfig, failures: list[dict[str, Any]], figures: Path
) -> list[str]:
    observations = {
        item.observation_id: item
        for item in load_manifest(config.manifest_path, split=config.protocol.evaluation_split, valid_only=True)
    }
    rendered: list[str] = []
    for index, failure in enumerate(failures):
        observation = observations.get(str(failure["query_observation_id"]))
        if observation is None:
            continue
        image_path = config.paths.bee24_root / observation.image_path
        with Image.open(image_path) as source:
            frame = source.convert("RGB")
        draw = ImageDraw.Draw(frame)
        draw.rectangle(
            (observation.bbox_x1, observation.bbox_y1, observation.bbox_x2, observation.bbox_y2),
            outline=(255, 215, 0), width=2,
        )
        draw.rectangle(
            (observation.crop_x0, observation.crop_y0, observation.crop_x1 - 1, observation.crop_y1 - 1),
            outline=(255, 0, 0), width=2,
        )
        label = str(failure["category"])
        draw.rectangle((0, 0, min(frame.width, 8 * len(label) + 8), 16), fill=(0, 0, 0))
        draw.text((3, 2), label, fill=(255, 255, 255))
        frame.thumbnail((480, 280))
        panels: list[tuple[str, Image.Image]] = [("query", load_crop(config.paths.bee24_root, observation))]
        for panel_label, key in (
            ("positive", "positive_observation_id"),
            ("hard negative", "hard_negative_observation_id"),
        ):
            candidate = observations.get(str(failure.get(key, "")))
            if candidate is not None:
                panels.append((panel_label, load_crop(config.paths.bee24_root, candidate)))
        canvas = Image.new("RGB", (max(480, len(panels) * 160), frame.height + 180), (245, 245, 245))
        canvas.paste(frame, (0, 0))
        canvas_draw = ImageDraw.Draw(canvas)
        for panel_index, (panel_label, panel) in enumerate(panels):
            panel.thumbnail((150, 150))
            x = panel_index * 160
            y = frame.height + 20
            canvas.paste(panel, (x, y))
            canvas_draw.text((x + 2, frame.height + 3), panel_label, fill=(0, 0, 0))
            panel.close()
        name = f"failure-{index:04d}.png"
        canvas.save(figures / name)
        canvas.close()
        frame.close()
        rendered.append(name)
    atomic_write_text(
        figures / "index.txt",
        "Yellow: original GT box; red: expanded/clipped crop. "
        "Panels: query, positive, model hard negative.\n" + "\n".join(rendered) + "\n",
    )
    return rendered


def generate_report(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    report_started = datetime.now(timezone.utc).isoformat()
    query_path = config.paths.output_root / "query_results.csv"
    if not query_path.is_file():
        raise RuntimeError(f"Run evaluate before report: {query_path}")
    rows = _read_csv(query_path)
    failures = _failure_cases(rows, config.protocol.failure_top_k)
    _write_csv(config.paths.output_root / "failure_cases.csv", failures)
    figures = config.paths.output_root / "figures"
    logs = config.paths.output_root / "logs"
    figures.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    rendered_figures = _render_failure_figures(config, failures, figures)
    atomic_write_text(
        logs / "report.log",
        f"{datetime.now(timezone.utc).isoformat()} report generated; GPU/real-data validation status: {SERVER_VALIDATION_PENDING}\n",
    )
    resolved = config.serializable()
    resolved.pop("config_path", None)
    atomic_write_text(config.paths.output_root / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=False))
    run_state_path = config.paths.output_root / "run_state.json"
    run_state = json.loads(run_state_path.read_text(encoding="utf-8")) if run_state_path.is_file() else {}
    query_skip_reasons = count_reasons(row["skip_reason"] for row in rows if row["skip_reason"])
    metadata = {
        "benchmark_started_at_utc": run_state.get("benchmark_started_at_utc", report_started),
        "report_started_at_utc": report_started,
        "benchmark_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": SERVER_VALIDATION_PENDING,
        "project_git_commit": _project_git_commit(),
        "models_requested": list(model_names),
        "seed": config.runtime.seed,
        "evaluation_split": config.protocol.evaluation_split,
        "resolved_split": (
            yaml.safe_load(config.resolved_split_path.read_text(encoding="utf-8"))
            if config.resolved_split_path.is_file() else None
        ),
        "resolved_config": resolved,
        "manifest_sha256": sha256_file(config.manifest_path),
        "duplicate_identity_audit": (
            json.loads(config.duplicate_identity_audit_path.read_text(encoding="utf-8"))
            if config.duplicate_identity_audit_path.is_file() else None
        ),
        "dataset_license": "LICENSE_NOT_STATED_BY_SOURCE",
        "environment": _environment(),
        "query_counts": {
            "total_rows": len(rows),
            "valid_full_rows": sum(row["rank1"] != "" for row in rows),
            "valid_hard_rows": sum(row["hard_rank1"] != "" for row in rows),
            "skipped_reasons": query_skip_reasons,
        },
        "feature_caches": _cache_signatures(config, model_names),
        "upstream_commits": {
            "dinov3": DINOV3_COMMIT,
            "topictrack": TOPICTRACK_COMMIT,
            "fastreid_audit_only": FASTREID_COMMIT,
            "trackeval_audit_only": TRACKEVAL_COMMIT,
            "torchvision": TORCHVISION_COMMIT,
        },
        "artifacts": [
            "summary.csv", "per_video_results.csv", "size_analysis.csv", "query_results.csv",
            "failure_cases.csv", "resolved_config.yaml", "resolved_split.yaml", "logs", "figures",
            "manifests/duplicate_identity_audit.json",
        ],
        "failure_figure_count": len(rendered_figures),
        "limitations": [
            "GT boxes measure an appearance upper bound, not end-to-end tracking.",
            "Crop context can leak background and location cues.",
            "The 20 percent expansion can alter model comparisons.",
            "H1 alone cannot establish fewer ID switches.",
        ],
    }
    atomic_write_json(config.paths.output_root / "run_metadata.json", metadata)
    return metadata
