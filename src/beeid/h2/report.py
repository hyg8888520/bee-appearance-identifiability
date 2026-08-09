"""H2 factor summaries, clustered uncertainty, figures, and run metadata."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml
from PIL import Image, ImageDraw

from ..analysis.report import _environment, _project_git_commit
from ..config import ExperimentConfig
from ..data.crops import load_crop
from ..data.mot import Observation
from ..references import DINOV3_COMMIT, TOPICTRACK_COMMIT, TORCHVISION_COMMIT
from ..status import SERVER_VALIDATION_PENDING
from ..utils import atomic_write_json, atomic_write_text, sha256_file
from .core import require_h2, validate_h2_inputs
from .crops import load_variant_crop
from .diagnostics import read_query_diagnostics
from .statistics import clustered_bootstrap, factor_analysis, paired_clustered_bootstrap


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Required H2 artifact does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = tuple(rows[0]) if rows else ("section", "model", "metric", "value")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _context_figures(
    context_rows: Sequence[dict[str, str]], output: Path
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    palette = [
        (47, 85, 151), (237, 125, 49), (112, 173, 71),
        (165, 165, 165), (255, 192, 0), (91, 155, 213), (112, 48, 160),
    ]
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in context_rows:
        if row.get("rank1"):
            by_model[row["model"]].append(row)
    for model, rows in sorted(by_model.items()):
        variants = sorted({row["variant"] for row in rows})
        deltas = sorted({int(row["delta"]) for row in rows})
        width, height = 1000, 560
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((25, 15), f"H2 context/input ablation: {model}", fill="black")
        left, top, right, bottom = 70, 55, 970, 470
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        for tick in range(0, 11, 2):
            y = bottom - int((bottom - top) * tick / 10)
            draw.line((left - 4, y, right, y), fill=(225, 225, 225), width=1)
            draw.text((25, y - 7), f"{tick / 10:.1f}", fill="black")
        group_width = (right - left) / max(1, len(deltas))
        bar_width = max(4, int(group_width / (len(variants) + 1)))
        lookup = {(int(row["delta"]), row["variant"]): float(row["rank1"]) for row in rows}
        for delta_index, delta in enumerate(deltas):
            start = left + delta_index * group_width + bar_width / 2
            for variant_index, variant in enumerate(variants):
                value = lookup.get((delta, variant))
                if value is None:
                    continue
                x1 = int(start + variant_index * bar_width)
                x2 = x1 + max(2, bar_width - 2)
                y1 = bottom - int((bottom - top) * value)
                draw.rectangle((x1, y1, x2, bottom - 1), fill=palette[variant_index % len(palette)])
            draw.text((int(start), bottom + 8), f"d={delta}", fill="black")
        legend_y = 500
        for index, variant in enumerate(variants):
            x = 20 + (index % 4) * 245
            y = legend_y + (index // 4) * 20
            draw.rectangle((x, y, x + 12, y + 12), fill=palette[index % len(palette)])
            draw.text((x + 17, y), variant, fill="black")
        name = f"context-ablation-{model}.png"
        canvas.save(output / name)
        canvas.close()
        rendered.append(name)
    atomic_write_text(
        output / "index.txt",
        "Rank-1 plots generated from h2_context_ablation.csv. These are development diagnostics, not final-test figures.\n"
        + "\n".join(rendered)
        + "\n",
    )
    return rendered


def _diagnostic_cases(
    rows: Sequence[dict[str, str]], primary_variant: str, top_k: int
) -> list[dict[str, Any]]:
    primary = [
        row
        for row in rows
        if row["variant"] == primary_variant and row["rank1"] == "False"
    ]
    selectors = (
        ("lowest_sharpness_failure", "query_bbox_laplacian_variance", False),
        ("highest_overlap_failure", "query_max_bbox_iou", True),
        ("highest_history_outlier_failure", "query_history_outlier", True),
        ("largest_orientation_change_failure", "absolute_orientation_change_proxy_deg", True),
    )
    selected: list[dict[str, Any]] = []
    for category, column, reverse in selectors:
        candidates = [row for row in primary if row.get(column) not in (None, "")]
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in candidates:
            groups[(row["model"], row["delta"])].append(row)
        for values in groups.values():
            values.sort(key=lambda row: float(row[column]), reverse=reverse)
            for row in values[:top_k]:
                selected.append({**row, "category": category, "factor_value": row[column]})
    clipped = [
        row
        for row in primary
        if row.get("query_variant_clipped", "").lower() == "true"
    ]
    clipped_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in clipped:
        clipped_groups[(row["model"], row["delta"])].append(row)
    for values in clipped_groups.values():
        for row in values[:top_k]:
            selected.append(
                {**row, "category": "boundary_clipped_failure", "factor_value": "true"}
            )

    lookup = {
        (
            row["model"], row["variant"], row["delta"], row["query_observation_id"]
        ): row
        for row in rows
        if row["rank1"] != ""
    }
    for key, row in lookup.items():
        model, variant, delta, observation_id = key
        if variant == primary_variant:
            continue
        reference = lookup.get((model, primary_variant, delta, observation_id))
        if reference is None or reference["rank1"] == row["rank1"]:
            continue
        category = (
            "variant_rescues_primary"
            if row["rank1"] == "True"
            else "variant_breaks_primary"
        )
        selected.append(
            {
                **row,
                "category": category,
                "factor_value": (
                    float(row["margin"]) - float(reference["margin"])
                    if row["margin"] and reference["margin"]
                    else ""
                ),
            }
        )
    paired: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        paired[
            (
                str(row["category"]), str(row["model"]),
                str(row["variant"]), str(row["delta"]),
            )
        ].append(row)
    limited: list[dict[str, Any]] = []
    for values in paired.values():
        limited.extend(values[:top_k])
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in limited:
        key = (
            str(row["category"]), str(row["model"]), str(row["variant"]),
            str(row["delta"]), str(row["query_observation_id"]),
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(row)
    return deduplicated


def _render_diagnostic_cases(
    config: ExperimentConfig,
    observations: Sequence[Observation],
    cases: Sequence[dict[str, Any]],
    output: Path,
) -> list[str]:
    h2 = require_h2(config)
    observation_map = {item.observation_id: item for item in observations}
    variants = {item.name: item for item in h2.variants}
    rendered: list[str] = []
    for index, case in enumerate(cases):
        query = observation_map.get(str(case["query_observation_id"]))
        positive = observation_map.get(str(case.get("positive_observation_id", "")))
        variant = variants.get(str(case["variant"]))
        if query is None or variant is None:
            continue
        panels: list[tuple[str, Image.Image]] = [
            ("query H1 view", load_crop(config.paths.bee24_root, query)),
            (
                f"query {variant.name}",
                load_variant_crop(
                    config.paths.bee24_root, query, variant, patch_size=h2.patch_size
                ),
            ),
        ]
        if positive is not None:
            panels.append(("positive H1 view", load_crop(config.paths.bee24_root, positive)))
        canvas = Image.new("RGB", (660, 230), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        title = f"{case['category']} | {case['model']} | d={case['delta']}"
        draw.text((5, 5), title, fill="black")
        for panel_index, (label, panel) in enumerate(panels):
            panel.thumbnail((200, 175))
            x = 5 + panel_index * 215
            canvas.paste(panel, (x, 45))
            draw.text((x, 27), label, fill="black")
            panel.close()
        name = f"diagnostic-case-{index:04d}.png"
        canvas.save(output / name)
        canvas.close()
        rendered.append(name)
    index_path = output / "index.txt"
    existing = index_path.read_text(encoding="utf-8") if index_path.is_file() else ""
    atomic_write_text(
        index_path,
        existing
        + "\nDiagnostic panels: H1 primary query, intervention query, and positive primary crop.\n"
        + "\n".join(rendered)
        + "\n",
    )
    return rendered


def _cache_metadata(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    if not config.h2_cache_locations_path.is_file():
        return {}
    locations = json.loads(config.h2_cache_locations_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    if not isinstance(locations, dict):
        return result
    for key, location in sorted(locations.items()):
        if str(key).split("::", 1)[0] not in model_names:
            continue
        metadata_path = Path(str(location)) / "cache_manifest.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            signature = metadata.get("signature")
            result[str(key)] = {
                "directory": str(metadata_path.parent),
                "fingerprint": metadata.get("fingerprint"),
                "cache_source": (
                    "h2_intervention_cache"
                    if isinstance(signature, dict)
                    and signature.get("experiment") == "H2_observation_reliability"
                    else "reused_h1_cache"
                ),
                "signature": signature,
            }
    return result


def generate_h2_report(
    config: ExperimentConfig, model_names: Sequence[str]
) -> dict[str, Any]:
    h2 = require_h2(config)
    observations, input_audit = validate_h2_inputs(config)
    query_rows = read_query_diagnostics(config.paths.output_root / "h2_query_diagnostics.csv")
    requested_rows = [row for row in query_rows if row["model"] in model_names]
    if not requested_rows:
        raise RuntimeError("H2 query diagnostics contain none of the requested models")
    factor_rows, predictiveness, definitions = factor_analysis(
        requested_rows, h2.factor_bins
    )
    bootstrap = clustered_bootstrap(
        requested_rows, h2.bootstrap_replicates, config.runtime.seed
    )
    paired_bootstrap = paired_clustered_bootstrap(
        requested_rows,
        h2.primary_variant,
        h2.bootstrap_replicates,
        config.runtime.seed,
    )
    _write_csv(config.paths.output_root / "h2_factor_summary.csv", factor_rows)
    _write_csv(config.paths.output_root / "h2_predictiveness.csv", predictiveness)
    _write_csv(config.paths.output_root / "h2_cluster_bootstrap.csv", bootstrap)
    _write_csv(config.paths.output_root / "h2_paired_bootstrap.csv", paired_bootstrap)
    atomic_write_json(config.paths.output_root / "h2_factor_definitions.json", definitions)

    context = _read_csv(config.paths.output_root / "h2_context_ablation.csv")
    paired = _read_csv(config.paths.output_root / "h2_paired_ablation.csv")
    memory = _read_csv(config.paths.output_root / "h2_memory_summary.csv")
    summary_rows: list[dict[str, Any]] = []
    for row in context:
        if row["model"] in model_names and row["variant"] == h2.primary_variant:
            summary_rows.extend(
                [
                    {
                        "section": "primary_retrieval",
                        "model": row["model"],
                        "variant": row["variant"],
                        "delta": row["delta"],
                        "metric": metric,
                        "value": row[metric],
                        "sample_count": row["full_query_count"],
                        "note": "development validation",
                    }
                    for metric in ("rank1", "hard_rank1", "mean_margin")
                ]
            )
    for row in paired:
        if row["model"] in model_names:
            summary_rows.append(
                {
                    "section": "paired_ablation",
                    "model": row["model"],
                    "variant": row["variant"],
                    "delta": row["delta"],
                    "metric": "variant_minus_primary_rank1",
                    "value": row["variant_minus_reference_rank1"],
                    "sample_count": row["paired_query_count"],
                    "note": f"reference={h2.primary_variant}",
                }
            )
    for row in memory:
        if row["model"] in model_names:
            summary_rows.append(
                {
                    "section": "memory_contamination",
                    "model": row["model"],
                    "variant": h2.primary_variant,
                    "delta": "",
                    "metric": f"{row['contamination_type']}_induced_error_rate",
                    "value": row["induced_error_rate"],
                    "sample_count": row["trajectory_step_count"],
                    "note": f"events={row['event_count']}",
                }
            )
    _write_csv(config.paths.output_root / "h2_summary.csv", summary_rows)

    cases = _diagnostic_cases(requested_rows, h2.primary_variant, config.protocol.failure_top_k)
    _write_csv(config.paths.output_root / "h2_diagnostic_cases.csv", cases)
    figures = _context_figures(context, config.paths.output_root / "h2_figures")
    figures.extend(
        _render_diagnostic_cases(
            config,
            observations,
            cases,
            config.paths.output_root / "h2_figures",
        )
    )
    logs = config.paths.output_root / "h2_logs"
    logs.mkdir(parents=True, exist_ok=True)
    resolved = config.serializable()
    resolved.pop("config_path", None)
    atomic_write_text(
        config.paths.output_root / "h2_resolved_config.yaml",
        yaml.safe_dump(resolved, sort_keys=False),
    )
    atomic_write_text(
        logs / "report.log",
        f"{datetime.now(timezone.utc).isoformat()} H2 development report generated; "
        f"server validation status: {SERVER_VALIDATION_PENDING}\n",
    )
    h1_metadata_path = config.h2_source_root / "run_metadata.json"
    h1_metadata = (
        json.loads(h1_metadata_path.read_text(encoding="utf-8"))
        if h1_metadata_path.is_file()
        else {}
    )
    artifacts = [
        "h2_summary.csv",
        "h2_observation_signals.csv",
        "h2_query_diagnostics.csv",
        "h2_context_ablation.csv",
        "h2_paired_ablation.csv",
        "h2_factor_summary.csv",
        "h2_predictiveness.csv",
        "h2_cluster_bootstrap.csv",
        "h2_paired_bootstrap.csv",
        "h2_memory_events.csv",
        "h2_memory_trajectories.csv",
        "h2_memory_summary.csv",
        "h2_memory_metadata.json",
        "h2_factor_definitions.json",
        "h2_diagnostic_cases.csv",
        "h2_signal_metadata.json",
        "h2_resolved_config.yaml",
        "h2_logs",
        "h2_figures",
    ]
    run_state_path = config.paths.output_root / "h2_run_state.json"
    run_state = (
        json.loads(run_state_path.read_text(encoding="utf-8"))
        if run_state_path.is_file()
        else {}
    )
    metadata = {
        "experiment": "H2_observation_reliability_and_memory_contamination",
        "status": SERVER_VALIDATION_PENDING,
        "h2_started_at_utc": run_state.get("h2_started_at_utc"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": _project_git_commit(),
        "models_requested": list(model_names),
        "official_topic_agw_role": "REFERENCE_ONLY_TRAINING_OVERLAP_UNCONFIRMED",
        "primary_variant": h2.primary_variant,
        "variants": [item.name for item in h2.variants],
        "input_audit": input_audit,
        "final_test_read": False,
        "environment": _environment(),
        "resolved_config": resolved,
        "source_h1": {
            "root": str(config.h2_source_root),
            "manifest_sha256": sha256_file(config.h2_manifest_path),
            "project_git_commit": h1_metadata.get("project_git_commit"),
            "benchmark_finished_at_utc": h1_metadata.get("benchmark_finished_at_utc"),
        },
        "feature_caches": _cache_metadata(config, model_names),
        "upstream_commits": {
            "dinov3": DINOV3_COMMIT,
            "topictrack": TOPICTRACK_COMMIT,
            "torchvision": TORCHVISION_COMMIT,
        },
        "counts": {
            "query_diagnostic_rows": len(requested_rows),
            "factor_summary_rows": len(factor_rows),
            "predictiveness_rows": len(predictiveness),
            "bootstrap_rows": len(bootstrap),
            "paired_bootstrap_rows": len(paired_bootstrap),
            "memory_summary_rows": len(memory),
            "figure_count": len(figures),
        },
        "artifacts": artifacts,
        "limitations": [
            "H2 is development-validation diagnosis, not final-test evaluation.",
            "Laplacian variance and structure-tensor orientation are image proxies, not manual blur or pose truth.",
            "GT-box overlap is an occlusion proxy and must not be called an occlusion label.",
            "bbox_foreground_only is a rectangular control, not bee segmentation.",
            "Cluster bootstrap intervals are descriptive with few videos and are not proof of causal effects.",
            "The controlled EMA experiment uses GT tracks and does not establish end-to-end MOT improvement.",
        ],
    }
    atomic_write_json(config.paths.output_root / "h2_run_metadata.json", metadata)
    return metadata
