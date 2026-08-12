"""Composable command-line interface for the BEE24 H1 through H5 experiments."""

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
from .h25.core import validate_h25_inputs
from .h25.experiment import run_h25_experiment
from .h25.report import generate_h25_report
from .h25.synthetic import h25_synthetic_smoke
from .h3.protocol import validate_h3_protocol
from .h3.core import validate_h3_inputs
from .h3.experiment import run_h3_tracking
from .h3.features import extract_h3_features
from .h3.report import generate_h3_report
from .h3.signals import build_h3_signals
from .h3.synthetic import h3_synthetic_smoke
from .h3.thresholds import fit_h3_thresholds
from .h4 import H4_PRIMARY_MODELS
from .h4.core import validate_h4_inputs
from .h4.experiment import run_h4_recoverability_audit, run_h4_tracking
from .h4.protocol import validate_h4_protocol
from .h4.report import generate_h4_report
from .h4.synthetic import h4_synthetic_smoke
from .h41 import H41_PRIMARY_MODELS
from .h41.core import validate_h41_inputs
from .h41.experiment import run_h41_tracking, run_h41_window_audit
from .h41.protocol import validate_h41_protocol
from .h41.report import generate_h41_report
from .h41.synthetic import h41_synthetic_smoke
from .h5 import H5_MODELS
from .h5.core import validate_h5_inputs
from .h5.experiment import run_h5_all, track_h5, train_h5
from .h5.protocol import validate_h5_protocol
from .h5.report import generate_h5_report
from .h5.synthetic import h5_synthetic_smoke
from .h51 import H51_MODELS
from .h51.core import validate_h51_inputs
from .h51.experiment import run_h51_all, run_h51_diagnostic
from .h51.protocol import validate_h51_protocol
from .h51.synthetic import h51_synthetic_smoke
from .sltr import SLTR_MODELS
from .sltr.core import validate_sltr_inputs
from .sltr.experiment import fit_sltr_selector, run_sltr_all, run_sltr_audit, run_sltr_tracking
from .sltr.protocol import validate_sltr_protocol
from .sltr.report import generate_sltr_report
from .sltr.synthetic import sltr_synthetic_smoke
from .h6 import H6_MODELS
from .h6.core import validate_h6_inputs
from .h6.experiment import run_h6_all, run_h6_oracle_audit, track_h6, train_h6
from .h6.protocol import validate_h6_protocol
from .h6.report import generate_h6_report
from .h6.synthetic import h6_synthetic_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beeid", description="BEE24 H1 diagnostics through H6 global trajectory experiments"
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

    configured("h25-validate", "Validate completed H2 inputs and both frozen protocol locks")
    h25_run = configured("h25-run", "Run refined outcome-blind four-strategy contamination diagnostics")
    h25_run.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, required=True)
    h25_report = configured("h25-report", "Generate H2.5 paired figures, provenance, and claim boundaries")
    h25_report.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, required=True)
    h25_all = configured("h25-run-all", "Validate, run, and report H2.5 in one environment")
    h25_all.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, default=list(REAL_MODEL_NAMES))
    h25_all.add_argument("--confirm-full", action="store_true")
    h25_synthetic = subparsers.add_parser(
        "h25-synthetic-smoke", help="Run end-to-end H2.5 smoke with a test-only encoder"
    )
    h25_synthetic.add_argument("--output", type=Path)
    h3_validate = subparsers.add_parser(
        "h3-validate-protocol", help="Validate the frozen H3 protocol and checksum"
    )
    h3_validate.add_argument("--protocol", type=Path, required=True)
    h3_validate.add_argument("--checksum", type=Path)
    configured("h3-validate", "Validate H3 development inputs and final-test isolation")
    h3_extract = configured(
        "h3-extract", "Extract/resume frozen features for project-train and development"
    )
    h3_extract.add_argument("--model", choices=REAL_MODEL_NAMES, required=True)
    configured("h3-signals", "Build outcome-blind H3 observation signal cache")
    h3_fit = configured(
        "h3-fit-thresholds", "Fit causal reliability calibrators on project_train only"
    )
    h3_fit.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, required=True)
    h3_track = configured("h3-track", "Run the four frozen H3 association variants")
    h3_track.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, required=True)
    h3_report = configured("h3-report", "Generate H3 GT-box metrics and locked metadata")
    h3_report.add_argument("--models", nargs="+", choices=REAL_MODEL_NAMES, required=True)
    h3_all = configured("h3-run-all", "Run the complete H3 GT-box development workflow")
    h3_all.add_argument(
        "--models", nargs="+", choices=REAL_MODEL_NAMES, default=["resnet50", "dinov3"]
    )
    h3_all.add_argument("--confirm-full", action="store_true")
    h3_synthetic = subparsers.add_parser(
        "h3-synthetic-smoke", help="Run CPU-only H3 tracking smoke with a test-only encoder"
    )
    h3_synthetic.add_argument("--output", type=Path)
    h4_validate_protocol = subparsers.add_parser(
        "h4-validate-protocol", help="Validate the frozen H4 development protocol"
    )
    h4_validate_protocol.add_argument("--protocol", type=Path, required=True)
    h4_validate_protocol.add_argument("--checksum", type=Path)
    configured("h4-validate", "Validate completed H3 inputs and final-test isolation")
    h4_audit = configured(
        "h4-audit", "Run the GT-separated future-evidence recoverability gate"
    )
    h4_audit.add_argument(
        "--models", nargs="+", choices=H4_PRIMARY_MODELS, required=True
    )
    h4_track = configured(
        "h4-track", "Run immediate and fixed-lag branch-isolation ablations"
    )
    h4_track.add_argument(
        "--models", nargs="+", choices=H4_PRIMARY_MODELS, required=True
    )
    h4_track.add_argument("--override-audit-stop", action="store_true")
    h4_report = configured("h4-report", "Generate H4 decisions, uncertainty, and failures")
    h4_report.add_argument(
        "--models", nargs="+", choices=H4_PRIMARY_MODELS, required=True
    )
    h4_all = configured("h4-run-all", "Run the gated H4 development workflow")
    h4_all.add_argument(
        "--models", nargs="+", choices=H4_PRIMARY_MODELS, default=list(H4_PRIMARY_MODELS)
    )
    h4_all.add_argument("--confirm-full", action="store_true")
    h4_all.add_argument("--override-audit-stop", action="store_true")
    h4_synthetic = subparsers.add_parser(
        "h4-synthetic-smoke", help="Run CPU-only H4 smoke with test-only cached features"
    )
    h4_synthetic.add_argument("--output", type=Path)
    h41_validate_protocol = subparsers.add_parser(
        "h41-validate-protocol", help="Validate the frozen exploratory H4.1 protocol"
    )
    h41_validate_protocol.add_argument("--protocol", type=Path, required=True)
    h41_validate_protocol.add_argument("--checksum", type=Path)
    configured(
        "h41-validate", "Validate H3 and preserved H4-v1 inputs without final-test access"
    )
    h41_audit = configured(
        "h41-audit", "Run continuous-lag and cumulative-window recoverability audit"
    )
    h41_audit.add_argument(
        "--models", nargs="+", choices=H41_PRIMARY_MODELS, required=True
    )
    h41_track = configured(
        "h41-track", "Run fixed and adaptive-lag branch-memory ablations"
    )
    h41_track.add_argument(
        "--models", nargs="+", choices=H41_PRIMARY_MODELS, required=True
    )
    h41_track.add_argument("--override-audit-stop", action="store_true")
    h41_report = configured(
        "h41-report", "Generate H4.1 exploratory decisions, uncertainty, and failures"
    )
    h41_report.add_argument(
        "--models", nargs="+", choices=H41_PRIMARY_MODELS, required=True
    )
    h41_all = configured("h41-run-all", "Run the gated H4.1 development workflow")
    h41_all.add_argument(
        "--models", nargs="+", choices=H41_PRIMARY_MODELS,
        default=list(H41_PRIMARY_MODELS),
    )
    h41_all.add_argument("--confirm-full", action="store_true")
    h41_all.add_argument("--override-audit-stop", action="store_true")
    h41_synthetic = subparsers.add_parser(
        "h41-synthetic-smoke",
        help="Run CPU-only H4.1 smoke with a preserved synthetic H4-v1 STOP",
    )
    h41_synthetic.add_argument("--output", type=Path)
    h5_validate_protocol = subparsers.add_parser(
        "h5-validate-protocol", help="Validate the frozen H5 BeeTrackQuery protocol"
    )
    h5_validate_protocol.add_argument("--protocol", type=Path, required=True)
    h5_validate_protocol.add_argument("--checksum", type=Path)
    configured("h5-validate", "Validate frozen H3 features and H5 split isolation")
    h5_train = configured("h5-train", "Train or resume BeeTrackQuery heads on project_train")
    h5_train.add_argument("--models", nargs="+", choices=H5_MODELS, required=True)
    h5_track = configured("h5-track", "Evaluate BeeTrackQuery ablations on development GT boxes")
    h5_track.add_argument("--models", nargs="+", choices=H5_MODELS, required=True)
    h5_report = configured("h5-report", "Generate H5 metrics, calibration, decision and guide")
    h5_report.add_argument("--models", nargs="+", choices=H5_MODELS, required=True)
    h5_all = configured("h5-run-all", "Run the complete H5 development workflow")
    h5_all.add_argument("--models", nargs="+", choices=H5_MODELS, default=list(H5_MODELS))
    h5_all.add_argument("--confirm-full", action="store_true")
    h5_synthetic = subparsers.add_parser(
        "h5-synthetic-smoke", help="Run CPU-only BeeTrackQuery smoke with test-only features"
    )
    h5_synthetic.add_argument("--output", type=Path)
    h51_validate_protocol = subparsers.add_parser(
        "h51-validate-protocol", help="Validate the frozen post-H5 diagnostic protocol"
    )
    h51_validate_protocol.add_argument("--protocol", type=Path, required=True)
    h51_validate_protocol.add_argument("--checksum", type=Path)
    configured("h51-validate", "Validate read-only failed H5/H3 diagnostic inputs")
    h51_run = configured(
        "h51-diagnose", "Run the H5.1 development-only closed-loop diagnostic"
    )
    h51_run.add_argument("--models", nargs="+", choices=H51_MODELS, required=True)
    h51_all = configured("h51-run-all", "Validate and run the full H5.1 diagnostic")
    h51_all.add_argument("--models", nargs="+", choices=H51_MODELS, default=list(H51_MODELS))
    h51_all.add_argument("--confirm-full", action="store_true")
    h51_synthetic = subparsers.add_parser(
        "h51-synthetic-smoke", help="Run CPU-only H5.1 scientific diagnostic smoke"
    )
    h51_synthetic.add_argument("--output", type=Path)
    sltr_validate_protocol = subparsers.add_parser(
        "sltr-validate-protocol", help="Validate the frozen SLTR protocol and checksum"
    )
    sltr_validate_protocol.add_argument("--protocol", type=Path, required=True)
    sltr_validate_protocol.add_argument("--checksum", type=Path)
    configured("sltr-validate", "Validate read-only H1/H3 inputs and final-test isolation for SLTR")
    sltr_audit = configured("sltr-audit", "Build offline local A/B counterfactual audit rows")
    sltr_audit.add_argument("--models", nargs="+", choices=SLTR_MODELS, required=True)
    sltr_fit = configured("sltr-fit", "Fit project-train-only SLTR selector and OOF threshold")
    sltr_fit.add_argument("--models", nargs="+", choices=SLTR_MODELS, required=True)
    sltr_track = configured("sltr-track", "Evaluate frozen SLTR variants on development only")
    sltr_track.add_argument("--models", nargs="+", choices=SLTR_MODELS, required=True)
    sltr_report = configured("sltr-report", "Write SLTR development metrics, audit guide, and failure artifacts")
    sltr_report.add_argument("--models", nargs="+", choices=SLTR_MODELS, required=True)
    sltr_all = configured("sltr-run-all", "Run audited train-only SLTR fit and development tracking")
    sltr_all.add_argument("--models", nargs="+", choices=SLTR_MODELS, default=list(SLTR_MODELS))
    sltr_all.add_argument("--confirm-full", action="store_true")
    sltr_synthetic = subparsers.add_parser(
        "sltr-synthetic-smoke", help="Run SLTR selector smoke with a test-only encoder"
    )
    sltr_synthetic.add_argument("--output", type=Path)
    h6_validate_protocol = subparsers.add_parser(
        "h6-validate-protocol", help="Validate the frozen H6 global trajectory protocol"
    )
    h6_validate_protocol.add_argument("--protocol", type=Path, required=True)
    h6_validate_protocol.add_argument("--checksum", type=Path)
    configured("h6-validate", "Validate read-only H3 inputs and H6 split isolation")
    configured("h6-oracle-audit", "Audit dense candidate graph recall before fitting")
    h6_train = configured("h6-train", "Train global association and trajectory selector networks")
    h6_train.add_argument("--models", nargs="+", choices=H6_MODELS, required=True)
    h6_track = configured("h6-track", "Evaluate calibrated global tracking on development")
    h6_track.add_argument("--models", nargs="+", choices=H6_MODELS, required=True)
    h6_report = configured("h6-report", "Generate H6 scientific gate and signed report")
    h6_report.add_argument("--models", nargs="+", choices=H6_MODELS, required=True)
    h6_all = configured("h6-run-all", "Run H6 oracle, GPU training, calibration, tracking, and report")
    h6_all.add_argument("--models", nargs="+", choices=H6_MODELS, default=list(H6_MODELS))
    h6_all.add_argument("--confirm-full", action="store_true")
    h6_synthetic = subparsers.add_parser(
        "h6-synthetic-smoke", help="Run CPU-only H6 scientific smoke with test-only features"
    )
    h6_synthetic.add_argument("--output", type=Path)
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


def _run_h25_all(config: object, models: list[str], confirm_full: bool) -> dict[str, object]:
    h25 = config.h25  # type: ignore[attr-defined]
    if h25 is None:
        raise RuntimeError("H2.5 commands require an h25 config section")
    if not h25.allow_subset and not confirm_full:
        raise RuntimeError(
            "Full H2.5 is gated; pass --confirm-full only after synthetic and real-data smoke"
        )
    validate_h25_inputs(config)  # type: ignore[arg-type]
    experiment = run_h25_experiment(config, models)  # type: ignore[arg-type]
    metadata = generate_h25_report(config, models)  # type: ignore[arg-type]
    return {
        "status": "completed",
        "models": models,
        "event_count": experiment["event_count"],
        "metadata_status": metadata["status"],
        "final_test_read": False,
    }


def _run_h3_all(config: object, models: list[str], confirm_full: bool) -> dict[str, object]:
    h3 = config.h3  # type: ignore[attr-defined]
    if h3 is None:
        raise RuntimeError("H3 commands require an h3 config section")
    if not h3.allow_subset and not confirm_full:
        raise RuntimeError(
            "Full H3 development is gated; pass --confirm-full only after synthetic "
            "and real-data smoke tests"
        )
    validate_h3_inputs(config)  # type: ignore[arg-type]
    build_h3_signals(config)  # type: ignore[arg-type]
    for model in models:
        extract_h3_features(config, model)  # type: ignore[arg-type]
    fit_h3_thresholds(config, models)  # type: ignore[arg-type]
    tracking = run_h3_tracking(config, models)  # type: ignore[arg-type]
    metadata = generate_h3_report(config, models)  # type: ignore[arg-type]
    return {
        "status": "completed_development_gt_boxes",
        "models": models,
        "assignment_rows": tracking["assignment_row_count"],
        "metadata_status": metadata["status"],
        "fixed_detector_boxes": "SERVER_VALIDATION_PENDING",
        "final_test_read": False,
    }


def _run_h4_all(
    config: object,
    models: list[str],
    confirm_full: bool,
    override_audit_stop: bool,
) -> dict[str, object]:
    h4 = config.h4  # type: ignore[attr-defined]
    if h4 is None:
        raise RuntimeError("H4 commands require an h4 config section")
    if not h4.allow_subset and not confirm_full:
        raise RuntimeError(
            "Full H4 development is gated; pass --confirm-full only after synthetic "
            "and real-data subset smoke tests"
        )
    validate_h4_inputs(config)  # type: ignore[arg-type]
    audit = run_h4_recoverability_audit(config, models)  # type: ignore[arg-type]
    gate_passed = audit["decision"]["gate_passed"] is True
    tracking: dict[str, object] | None = None
    if gate_passed or override_audit_stop:
        tracking = run_h4_tracking(  # type: ignore[arg-type]
            config, models, override_audit_stop=override_audit_stop
        )
    metadata = generate_h4_report(config, models)  # type: ignore[arg-type]
    return {
        "status": metadata["status"],
        "models": models,
        "recoverability_gate": audit["decision"]["status"],
        "method_evaluated": tracking is not None,
        "assignment_rows": None if tracking is None else tracking["assignment_row_count"],
        "final_test_read": False,
    }


def _run_h41_all(
    config: object,
    models: list[str],
    confirm_full: bool,
    override_audit_stop: bool,
) -> dict[str, object]:
    h41 = config.h41  # type: ignore[attr-defined]
    if h41 is None:
        raise RuntimeError("H4.1 commands require an h41 config section")
    if not h41.allow_subset and not confirm_full:
        raise RuntimeError(
            "Full H4.1 development is gated; pass --confirm-full only after synthetic "
            "and real-data subset smoke tests"
        )
    validate_h41_inputs(config)  # type: ignore[arg-type]
    audit = run_h41_window_audit(config, models)  # type: ignore[arg-type]
    gate_passed = audit["decision"]["gate_passed"] is True
    tracking: dict[str, object] | None = None
    if gate_passed or override_audit_stop:
        tracking = run_h41_tracking(  # type: ignore[arg-type]
            config, models, override_audit_stop=override_audit_stop
        )
    metadata = generate_h41_report(config, models)  # type: ignore[arg-type]
    return {
        "status": metadata["status"],
        "models": models,
        "window_recoverability_gate": audit["decision"]["status"],
        "source_h4_v1_conclusion_preserved": True,
        "method_evaluated": tracking is not None,
        "assignment_rows": None if tracking is None else tracking["assignment_row_count"],
        "fixed_detector_boxes": "SERVER_VALIDATION_PENDING",
        "final_test_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "synthetic-smoke":
            result = synthetic_smoke(arguments.output)
        elif arguments.command == "h2-synthetic-smoke":
            result = h2_synthetic_smoke(arguments.output)
        elif arguments.command == "h25-synthetic-smoke":
            result = h25_synthetic_smoke(arguments.output)
        elif arguments.command == "h3-synthetic-smoke":
            result = h3_synthetic_smoke(arguments.output)
        elif arguments.command == "h4-synthetic-smoke":
            result = h4_synthetic_smoke(arguments.output)
        elif arguments.command == "h41-synthetic-smoke":
            result = h41_synthetic_smoke(arguments.output)
        elif arguments.command == "h5-synthetic-smoke":
            result = h5_synthetic_smoke(arguments.output)
        elif arguments.command == "h51-synthetic-smoke":
            result = h51_synthetic_smoke(arguments.output)
        elif arguments.command == "sltr-synthetic-smoke":
            result = sltr_synthetic_smoke(arguments.output)
        elif arguments.command == "h6-synthetic-smoke":
            result = h6_synthetic_smoke(arguments.output)
        elif arguments.command == "h3-validate-protocol":
            result = validate_h3_protocol(arguments.protocol, arguments.checksum)
        elif arguments.command == "h4-validate-protocol":
            result = validate_h4_protocol(arguments.protocol, arguments.checksum)
        elif arguments.command == "h41-validate-protocol":
            result = validate_h41_protocol(arguments.protocol, arguments.checksum)
        elif arguments.command == "h5-validate-protocol":
            result = validate_h5_protocol(arguments.protocol, arguments.checksum)
        elif arguments.command == "h51-validate-protocol":
            result = validate_h51_protocol(arguments.protocol, arguments.checksum)
        elif arguments.command == "sltr-validate-protocol":
            result = validate_sltr_protocol(arguments.protocol, arguments.checksum)
        elif arguments.command == "h6-validate-protocol":
            result = validate_h6_protocol(arguments.protocol, arguments.checksum)
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
            elif arguments.command == "h25-validate":
                _, _, result = validate_h25_inputs(config)
            elif arguments.command == "h25-run":
                result = run_h25_experiment(config, arguments.models)
            elif arguments.command == "h25-report":
                result = generate_h25_report(config, arguments.models)
            elif arguments.command == "h25-run-all":
                result = _run_h25_all(
                    config, arguments.models, arguments.confirm_full
                )
            elif arguments.command == "h3-validate":
                result = validate_h3_inputs(config).audit
            elif arguments.command == "h3-extract":
                result = extract_h3_features(config, arguments.model)
            elif arguments.command == "h3-signals":
                result = build_h3_signals(config)
            elif arguments.command == "h3-fit-thresholds":
                result = fit_h3_thresholds(config, arguments.models)
            elif arguments.command == "h3-track":
                result = run_h3_tracking(config, arguments.models)
            elif arguments.command == "h3-report":
                result = generate_h3_report(config, arguments.models)
            elif arguments.command == "h3-run-all":
                result = _run_h3_all(config, arguments.models, arguments.confirm_full)
            elif arguments.command == "h4-validate":
                result = validate_h4_inputs(config).audit
            elif arguments.command == "h4-audit":
                result = run_h4_recoverability_audit(config, arguments.models)
            elif arguments.command == "h4-track":
                result = run_h4_tracking(
                    config,
                    arguments.models,
                    override_audit_stop=arguments.override_audit_stop,
                )
            elif arguments.command == "h4-report":
                result = generate_h4_report(config, arguments.models)
            elif arguments.command == "h4-run-all":
                result = _run_h4_all(
                    config,
                    arguments.models,
                    arguments.confirm_full,
                    arguments.override_audit_stop,
                )
            elif arguments.command == "h41-validate":
                result = validate_h41_inputs(config).audit
            elif arguments.command == "h41-audit":
                result = run_h41_window_audit(config, arguments.models)
            elif arguments.command == "h41-track":
                result = run_h41_tracking(
                    config,
                    arguments.models,
                    override_audit_stop=arguments.override_audit_stop,
                )
            elif arguments.command == "h41-report":
                result = generate_h41_report(config, arguments.models)
            elif arguments.command == "h41-run-all":
                result = _run_h41_all(
                    config,
                    arguments.models,
                    arguments.confirm_full,
                    arguments.override_audit_stop,
                )
            elif arguments.command == "h5-validate":
                result = validate_h5_inputs(config).audit
            elif arguments.command == "h5-train":
                result = train_h5(config, arguments.models)
            elif arguments.command == "h5-track":
                result = track_h5(config, arguments.models)
            elif arguments.command == "h5-report":
                result = generate_h5_report(config, arguments.models)
            elif arguments.command == "h5-run-all":
                result = run_h5_all(config, arguments.models, arguments.confirm_full)
            elif arguments.command == "h51-validate":
                result = validate_h51_inputs(config).audit
            elif arguments.command == "h51-diagnose":
                result = run_h51_diagnostic(config, arguments.models)
            elif arguments.command == "h51-run-all":
                result = run_h51_all(config, arguments.models, arguments.confirm_full)
            elif arguments.command == "sltr-validate":
                result = validate_sltr_inputs(config).audit
            elif arguments.command == "sltr-audit":
                result = run_sltr_audit(config, arguments.models)
            elif arguments.command == "sltr-fit":
                result = fit_sltr_selector(config, arguments.models)
            elif arguments.command == "sltr-track":
                result = run_sltr_tracking(config, arguments.models)
            elif arguments.command == "sltr-report":
                result = generate_sltr_report(config, arguments.models)
            elif arguments.command == "sltr-run-all":
                result = run_sltr_all(config, arguments.models, arguments.confirm_full)
            elif arguments.command == "h6-validate":
                result = validate_h6_inputs(config).audit
            elif arguments.command == "h6-oracle-audit":
                result = run_h6_oracle_audit(config)
            elif arguments.command == "h6-train":
                result = train_h6(config, arguments.models)
            elif arguments.command == "h6-track":
                result = track_h6(config, arguments.models)
            elif arguments.command == "h6-report":
                result = generate_h6_report(config, arguments.models)
            elif arguments.command == "h6-run-all":
                result = run_h6_all(config, arguments.models, arguments.confirm_full)
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
