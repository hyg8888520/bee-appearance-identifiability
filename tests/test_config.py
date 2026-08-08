from __future__ import annotations

from pathlib import Path

import pytest

from bee24_dinotrack.config import ConfigurationError, load_config
from conftest import write_runtime


def test_loads_every_runtime_path_from_yaml(tmp_path: Path) -> None:
    config_path, raw = write_runtime(tmp_path)
    config = load_config(config_path)
    assert config.name == "test"
    assert config.dataset.root == Path(raw["dataset"]["root"])
    assert config.dataset.annotation == Path(raw["dataset"]["annotation"])
    assert config.dinov3.checkpoint == Path(raw["models"]["dinov3"]["checkpoint"])
    assert config.output.feature_cache == Path(raw["output"]["root"]) / "cache" / "features.pt"
    assert config.output.query_results == Path(raw["output"]["root"]) / "queries" / "results.json"


def test_relative_primary_paths_are_resolved_from_config_directory(tmp_path: Path) -> None:
    config_path, _ = write_runtime(tmp_path)
    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["dataset"]["root"] = "dataset"
    raw["dataset"]["annotation"] = "dataset/train.json"
    raw["models"]["dinov3"]["checkpoint"] = "model.pth"
    raw["output"]["root"] = "experiment"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)
    assert config.dataset.root == (tmp_path / "dataset").resolve()
    assert config.output.root == (tmp_path / "experiment").resolve()


def test_rejects_output_outside_root(tmp_path: Path) -> None:
    config_path, _ = write_runtime(tmp_path, output__feature_cache="../escaped/features.pt")
    with pytest.raises(ConfigurationError, match="outside output.root"):
        load_config(config_path)


def test_rejects_unknown_keys_to_catch_typos(tmp_path: Path) -> None:
    config_path, _ = write_runtime(tmp_path, dataset__roots="wrong")
    with pytest.raises(ConfigurationError, match="dataset.roots"):
        load_config(config_path)


def test_rejects_duplicate_output_paths(tmp_path: Path) -> None:
    config_path, _ = write_runtime(tmp_path, output__metadata="queries/results.json")
    with pytest.raises(ConfigurationError, match="must be unique"):
        load_config(config_path)
