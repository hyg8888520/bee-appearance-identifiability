"""Run the two modern environments without hard-coded interpreter paths."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from .config import ExperimentConfig
from .status import BLOCKED_TOPIC_AGW_COMPATIBILITY
from .utils import atomic_write_json


def _python(configured: Path | None, label: str) -> Path:
    if configured is None:
        raise RuntimeError(f"paths.{label}_python must be filled for two-environment orchestration")
    if not configured.is_file():
        raise RuntimeError(f"Configured {label} interpreter does not exist: {configured}")
    return configured


def _run(python: Path, arguments: Sequence[str]) -> None:
    subprocess.run([str(python), "-m", "beeid.cli", *arguments], check=True)


def orchestrate(config: ExperimentConfig, *, confirm_full: bool = False) -> None:
    dino_python = _python(config.paths.dino_python, "dino")
    topic_python = _python(config.paths.topic_python, "topic")
    config_path = str(config.config_path)
    full = not config.dataset.video_ids and config.dataset.max_videos is None and config.dataset.max_frames_per_video is None
    if full and not confirm_full:
        raise RuntimeError("Full H1 is gated. Re-run orchestrate with --confirm-full after smoke estimates and user confirmation.")
    _run(dino_python, ["build-manifest", "--config", config_path])
    _run(dino_python, ["extract", "--config", config_path, "--model", "resnet50"])
    _run(dino_python, ["extract", "--config", config_path, "--model", "dinov3"])
    topic_ok = True
    try:
        _run(topic_python, ["extract", "--config", config_path, "--model", "topic_agw"])
    except subprocess.CalledProcessError as error:
        if error.returncode != 3:
            raise
        topic_ok = False
        atomic_write_json(
            config.paths.output_root / "topic_agw_status.json",
            {
                "status": BLOCKED_TOPIC_AGW_COMPATIBILITY,
                "detail": "TOPIC worker returned the dedicated compatibility exit code; see worker stderr.",
            },
        )
    models = ["resnet50", "dinov3", *(["topic_agw"] if topic_ok else [])]
    _run(dino_python, ["evaluate", "--config", config_path, "--models", *models])
    _run(dino_python, ["report", "--config", config_path, "--models", *models])
