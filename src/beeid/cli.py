"""Composable command-line interface for the BEE24 H1/H2 benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .analysis.estimate import estimate_run
from .analysis.report import generate_report
from .config import load_config
from .data.manifest import build_manifest
from .extraction import extract_features
from .models import REAL_MODEL_NAMES, TopicCompatibilityError
from .orchestration import orchestrate
from .protocol.retrieval import evaluate
from .server import check_server_environment, gpu_smoke
from .status import BLOCKED_TOPIC_AGW_COMPATIBILITY
from .synthetic import synthetic_smoke
from .utils import atomic_write_json
from .h2.contamination import run_memory_contamination
from .h2.core import validate_h2_inputs
from .h2.diagnostics import evaluate_h2
from .h2.estimate import estimate_h2_run
from .h2.features import extract_h2_features
from .h2.orchestration import orchestrate_h2
from .h2.report import generate_h2_report
from .h2.signals import build_observation_signals
from .h2.synthetic import h2_synthetic_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beeid", description="BEE24 H1 appearance and H2 reliability benchmarks"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def configured(name: str, help_text: str) -> argparse.ArgumentParser:
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        return command

    configured("validate-data", "Validate the configured MOT tree without writing a manifest")
    configured("build-manifest", "Validate data and write stable manifest/split records")
    extract = configured("extract", "Extract or resume one real model's feature cache")
    extract.add_argument("--model", choices=REAL_MODEL_NAMES, required=True)
    evaluation = configured("evaluate", "Evaluate one or more extracted feature caches")
    evaluation.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, required=True)
    report = configured("report", "Generate failures, metadata, logs, and report records")
    report.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, required=True)
    estimate = configured("estimate", "Estimate runtime/cache from a measured server smoke")
    estimate.add_argument("--embedding-dimension", type=int, required=True)
    estimate.add_argument("--observations-per-second", type=float)
    estimate.add_argument("--peak-memory-gib", type=float)
    run_all = configured("run-all", "Run one-environment H1 workflow")
    run_all.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, default=list(REAL_MODEL_NAMES))
    run_all.add_argument("--confirm-full", action="store_true")
    orchestration = configured("orchestrate", "Run DINO and TOPIC workers using YAML interpreter paths")
    orchestration.add_argument("--confirm-full", action="store_true")
    configured("server-check", "Check pinned repositories, paths, PyTorch, and CUDA")
    configured("gpu-smoke", "Run all required models on one real BEE24 crop")
    synthetic = subparsers.add_parser("synthetic-smoke", help="Run end-to-end smoke with a test-only encoder")
    synthetic.add_argument("--output", type=Path)

    configured("h2-validate", "Validate frozen H1 inputs and the locked H2 development split")
    h2_extract = configured("h2-extract", "Extract/resume all configured H2 variants for one model")
    h2_extract.add_argument("--model", choices=REAL_MODEL_NAMES, required=True)
    h2_extract.add_argument("--variants", nargs="+")
    configured("h2-signals", "Build outcome-blind H2 observation reliability signals")
    h2_evaluate = configured("h2-evaluate", "Evaluate all H2 model/variant retrieval diagnostics")
    h2_evaluate.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, required=True)
    h2_memory = configured("h2-contamination", "Run controlled EMA template-contamination experiments")
    h2_memory.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, required=True)
    h2_report = configured("h2-report", "Generate factor, cluster-bootstrap, figure, and metadata artifacts")
    h2_report.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, required=True)
    h2_estimate = configured("h2-estimate", "Estimate H2 extraction time and embedding storage")
    h2_estimate.add_argument("--model", choices=REAL_MODEL_NAMES, required=True)
    h2_estimate.add_argument("--embedding-dimension", type=int, required=True)
    h2_estimate.add_argument("--observations-per-second", type=float)
    h2_estimate.add_argument("--peak-memory-gib", type=float)
    h2_all = configured("h2-run-all", "Run one-environment H2 workflow")
    h2_all.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, default=list(REAL_MODEL_NAMES))
    h2_all.add_argument("--confirm-full", action="store_true")
    h2_orchestration = configured("h2-orchestrate", "Run H2 across configured DINO and TOPIC environments")
    h2_orchestration.add_argument("--confirm-full", action="store_true")
    h2_synthetic = subparsers.add_parser(
        "h2-synthetic-smoke", help="Run end-to-end H2 smoke with a test-only encoder"
    )
    h2_synthetic.add_argument("--output", type=Path)
    return parser


def _full_is_gated(config: object, confirm_full: bool) -> None:
    dataset = config.dataset  # type: ignore[attr-defined]
    full = not dataset.video_ids and dataset.max_videos is None and dataset.max_frames_per_video is None
    if full and not confirm_full:
        raise RuntimeError("Full H1 is gated; pass --confirm-full only after real smoke estimates and user confirmation")


def _run_all(config: object, models: list[str], confirm_full: bool) -> dict[str, object]:
    _full_is_gated(config, confirm_full)
    build_manifest(config)  # type: ignore[arg-type]
    completed: list[str] = []
    blocked: dict[str, str] = {}
    for model in models:
        try:
            extract_features(config, model)  # type: ignore[arg-type]
            completed.append(model)
        except TopicCompatibilityError as error:
            blocked[model] = str(error)
            atomic_write_json(
                config.paths.output_root / "topic_agw_status.json",  # type: ignore[attr-defined]
                {"status": BLOCKED_TOPIC_AGW_COMPATIBILITY, "error": str(error)},
            )
    if not completed:
        raise RuntimeError("No feature extractor completed")
    evaluate(config, completed)  # type: ignore[arg-type]
    generate_report(config, completed)  # type: ignore[arg-type]
    return {"completed_models": completed, "blocked_models": blocked}


def _h2_full_is_gated(config: object, confirm_full: bool) -> None:
    h2 = config.h2  # type: ignore[attr-defined]
    if h2 is None:
        raise RuntimeError("H2 commands require an h2 config section")
    if not h2.allow_subset and not confirm_full:
        raise RuntimeError(
            "Full H2 is gated; pass --confirm-full only after H2 synthetic and real-data smoke"
        )


def _run_h2_all(config: object, models: list[str], confirm_full: bool) -> dict[str, object]:
    _h2_full_is_gated(config, confirm_full)
    validate_h2_inputs(config)  # type: ignore[arg-type]
    atomic_write_json(
        config.paths.output_root / "h2_run_state.json",  # type: ignore[attr-defined]
        {"h2_started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    build_observation_signals(config)  # type: ignore[arg-type]
    completed: list[str] = []
    blocked: dict[str, str] = {}
    for model in models:
        try:
            extract_h2_features(config, model)  # type: ignore[arg-type]
            completed.append(model)
        except TopicCompatibilityError as error:
            blocked[model] = str(error)
            atomic_write_json(
                config.paths.output_root / "h2_topic_agw_status.json",  # type: ignore[attr-defined]
                {"status": BLOCKED_TOPIC_AGW_COMPATIBILITY, "error": str(error)},
            )
    if not completed:
        raise RuntimeError("No H2 feature extractor completed")
    evaluate_h2(config, completed)  # type: ignore[arg-type]
    run_memory_contamination(config, completed)  # type: ignore[arg-type]
    generate_h2_report(config, completed)  # type: ignore[arg-type]
    return {"completed_models": completed, "blocked_models": blocked}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "synthetic-smoke":
            result = synthetic_smoke(arguments.output)
        elif arguments.command == "h2-synthetic-smoke":
            result = h2_synthetic_smoke(arguments.output)
        else:
            config = load_config(arguments.config)
            if arguments.command == "validate-data":
                observations = build_manifest(config, validate_only=True)
                result = {"status": "valid", "observation_count": len(observations)}
            elif arguments.command == "build-manifest":
                observations = build_manifest(config)
                result = {"manifest": str(config.manifest_path), "observation_count": len(observations)}
            elif arguments.command == "extract":
                cache = extract_features(config, arguments.model)
                result = {"model": arguments.model, "cache": str(cache.directory), "fingerprint": cache.fingerprint}
            elif arguments.command == "evaluate":
                rows = evaluate(config, arguments.models)
                result = {"query_rows": len(rows), "models": arguments.models}
            elif arguments.command == "report":
                result = generate_report(config, arguments.models)
            elif arguments.command == "estimate":
                result = estimate_run(
                    config,
                    embedding_dimension=arguments.embedding_dimension,
                    observations_per_second=arguments.observations_per_second,
                    peak_memory_gib=arguments.peak_memory_gib,
                )
                atomic_write_json(config.paths.output_root / "run_estimate.json", result)
            elif arguments.command == "run-all":
                result = _run_all(config, arguments.models, arguments.confirm_full)
            elif arguments.command == "orchestrate":
                orchestrate(config, confirm_full=arguments.confirm_full)
                result = {"status": "completed"}
            elif arguments.command == "server-check":
                result = check_server_environment(config)
            elif arguments.command == "gpu-smoke":
                result = gpu_smoke(config)
            elif arguments.command == "h2-validate":
                _, result = validate_h2_inputs(config)
            elif arguments.command == "h2-extract":
                result = extract_h2_features(
                    config, arguments.model, variant_names=arguments.variants
                )
            elif arguments.command == "h2-signals":
                result = build_observation_signals(config)
            elif arguments.command == "h2-evaluate":
                result = evaluate_h2(config, arguments.models)
            elif arguments.command == "h2-contamination":
                result = run_memory_contamination(config, arguments.models)
            elif arguments.command == "h2-report":
                result = generate_h2_report(config, arguments.models)
            elif arguments.command == "h2-estimate":
                result = estimate_h2_run(
                    config,
                    model_name=arguments.model,
                    embedding_dimension=arguments.embedding_dimension,
                    observations_per_second=arguments.observations_per_second,
                    peak_memory_gib=arguments.peak_memory_gib,
                )
                atomic_write_json(
                    config.paths.output_root / f"h2_estimate_{arguments.model}.json", result
                )
            elif arguments.command == "h2-run-all":
                result = _run_h2_all(config, arguments.models, arguments.confirm_full)
            elif arguments.command == "h2-orchestrate":
                orchestrate_h2(config, confirm_full=arguments.confirm_full)
                result = {"status": "completed"}
            else:
                raise AssertionError(arguments.command)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except TopicCompatibilityError as error:
        print(f"{BLOCKED_TOPIC_AGW_COMPATIBILITY}: {error}", file=sys.stderr)
        return 3
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
