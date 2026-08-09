"""Two-modern-environment orchestration for H2."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..config import ExperimentConfig
from ..status import BLOCKED_TOPIC_AGW_COMPATIBILITY
from ..utils import atomic_write_json
from .core import require_h2


def _python(configured: Path | None, label: str) -> Path:
    if configured is None:
        raise RuntimeError(f"paths.{label}_python must be filled for H2 orchestration")
    if not configured.is_file():
        raise RuntimeError(f"Configured {label} interpreter does not exist: {configured}")
    return configured


def _run(python: Path, arguments: Sequence[str]) -> None:
    subprocess.run([str(python), "-m", "beeid.cli", *arguments], check=True)


def orchestrate_h2(config: ExperimentConfig, *, confirm_full: bool = False) -> None:
    h2 = require_h2(config)
    if not h2.allow_subset and not confirm_full:
        raise RuntimeError(
            "Full H2 is gated. Re-run with --confirm-full only after synthetic and real-data smoke."
        )
    dino_python = _python(config.paths.dino_python, "dino")
    topic_python = _python(config.paths.topic_python, "topic")
    config_path = str(config.config_path)
    atomic_write_json(
        config.paths.output_root / "h2_run_state.json",
        {"h2_started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    _run(dino_python, ["h2-validate", "--config", config_path])
    _run(dino_python, ["h2-signals", "--config", config_path])
    _run(dino_python, ["h2-extract", "--config", config_path, "--model", "resnet50"])
    _run(dino_python, ["h2-extract", "--config", config_path, "--model", "dinov3"])
    topic_ok = True
    try:
        _run(topic_python, ["h2-extract", "--config", config_path, "--model", "topic_agw"])
    except subprocess.CalledProcessError as error:
        if error.returncode != 3:
            raise
        topic_ok = False
        atomic_write_json(
            config.paths.output_root / "h2_topic_agw_status.json",
            {
                "status": BLOCKED_TOPIC_AGW_COMPATIBILITY,
                "role": "REFERENCE_ONLY_TRAINING_OVERLAP_UNCONFIRMED",
                "detail": "TOPIC H2 worker returned the dedicated compatibility exit code.",
            },
        )
    models = ["resnet50", "dinov3", *(["topic_agw"] if topic_ok else [])]
    _run(dino_python, ["h2-evaluate", "--config", config_path, "--models", *models])
    _run(dino_python, ["h2-contamination", "--config", config_path, "--models", *models])
    _run(dino_python, ["h2-report", "--config", config_path, "--models", *models])
