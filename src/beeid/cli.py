"""Composable command-line interface for the BEE24 H1 benchmark."""

from __future__ import annotations

import argparse
import json
import sys
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beeid", description="BEE24 H1 appearance identifiability benchmark")
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


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "synthetic-smoke":
            result = synthetic_smoke(arguments.output)
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
