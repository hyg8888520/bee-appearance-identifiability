from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from beeid.config import ConfigurationError, load_config
from beeid.synthetic import synthetic_smoke


def test_synthetic_smoke_writes_required_artifacts(tmp_path):
    output = tmp_path / "synthetic-output"
    result = synthetic_smoke(output)
    assert result["status"] == "passed" and result["test_only_encoder"] is True
    for name in (
        "summary.csv", "per_video_results.csv", "size_analysis.csv", "query_results.csv",
        "failure_cases.csv", "run_metadata.json", "resolved_config.yaml", "resolved_split.yaml",
    ):
        assert (output / name).is_file(), name
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "SERVER_VALIDATION_PENDING"
    assert metadata["models_requested"] == ["synthetic_test"]
    assert metadata["project_git_commit"]
    assert metadata["benchmark_started_at_utc"] <= metadata["benchmark_finished_at_utc"]
    assert metadata["query_counts"]["valid_full_rows"] > 0
    assert metadata["resolved_config"]["runtime"]["seed"] == 24
    assert metadata["feature_caches"]["synthetic_test"]["signature"]["model"]["test_only"] is True
    assert (output / "figures" / "index.txt").is_file()
    assert list((output / "figures").glob("failure-*.png"))


def test_config_requires_all_model_keys_and_rejects_unknown(tmp_path, config_factory):
    config = config_factory(tmp_path)
    text = config.config_path.read_text(encoding="utf-8")
    config.config_path.write_text(text.replace("  dinov3_hub_model: dinov3_vits16\n", ""), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Missing required key"):
        load_config(config.config_path)


def test_config_preserves_virtualenv_interpreter_symlink(tmp_path, config_factory):
    target = tmp_path / "system" / "python3.11"
    target.parent.mkdir(parents=True)
    target.touch()
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    try:
        interpreter.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    config = config_factory(
        tmp_path,
        paths={"dino_python": str(interpreter), "topic_python": str(interpreter)},
    )
    assert config.paths.dino_python == interpreter.absolute()
    assert config.paths.topic_python == interpreter.absolute()
    assert config.paths.dino_python != target.resolve()


def test_repository_does_not_offer_fallbacks_or_commit_local_artifacts():
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(path.read_text(encoding="utf-8") for path in (root / "src" / "beeid").rglob("*.py"))
    lowered = source.lower()
    assert "timm" not in lowered
    assert "huggingface" not in lowered
    assert "dinov2" not in lowered
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "configs/*.local.yaml" in ignore
    assert "*.npz" in ignore
    assert not (root / "LICENSE").exists()
    assert "C:\\Users\\" not in source


def test_cli_help_lists_composable_commands():
    result = subprocess.run(
        [sys.executable, "-m", "beeid.cli", "--help"], capture_output=True, text=True, check=True
    )
    for command in (
        "validate-data", "build-manifest", "extract", "evaluate", "report", "estimate", "run-all",
        "synthetic-smoke",
    ):
        assert command in result.stdout
