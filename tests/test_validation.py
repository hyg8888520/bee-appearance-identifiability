from pathlib import Path

import pytest

from bee24_dinotrack.config import load_config
from bee24_dinotrack.validation import StartupValidationError, format_summary, validate_startup
from conftest import write_runtime


def test_valid_runtime_creates_and_checks_output_root(tmp_path: Path) -> None:
    config_path, _ = write_runtime(tmp_path)
    validated = validate_startup(load_config(config_path))
    assert validated.config.output.root.is_dir()
    assert validated.sample_image.name == "000001.jpg"
    assert "1 indexed" in format_summary(validated)


def test_missing_checkpoint_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "absent.pth"
    config_path, _ = write_runtime(tmp_path, models__dinov3__checkpoint=str(missing))
    with pytest.raises(StartupValidationError, match=r"models\.dinov3\.checkpoint.*absent\.pth"):
        validate_startup(load_config(config_path))


def test_requires_at_least_one_referenced_image(tmp_path: Path) -> None:
    config_path, raw = write_runtime(tmp_path)
    Path(raw["dataset"]["root"], "BEE24-01", "img1", "000001.jpg").unlink()
    with pytest.raises(StartupValidationError, match="No image referenced"):
        validate_startup(load_config(config_path))


def test_invalid_cuda_request_fails_before_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, _ = write_runtime(tmp_path, experiment__device="cuda:999999")
    try:
        import torch
    except ImportError:
        with pytest.raises(StartupValidationError, match="PyTorch is not installed"):
            validate_startup(load_config(config_path))
        return
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(StartupValidationError, match="cuda:999999"):
        validate_startup(load_config(config_path))

