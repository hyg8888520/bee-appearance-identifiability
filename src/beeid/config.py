"""Typed YAML configuration with all machine-specific paths kept external."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .utils import ensure_within


class ConfigurationError(ValueError):
    """Raised when a benchmark YAML is incomplete or inconsistent."""


@dataclass(frozen=True)
class PathsConfig:
    bee24_root: Path
    output_root: Path
    cache_root: Path
    dinov3_repo: Path
    dinov3_weights: Path
    topictrack_repo: Path
    topic_agw_weights: Path
    dino_python: Path | None = None
    topic_python: Path | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    batch_size: int
    num_workers: int
    amp: bool
    amp_dtype: str
    seed: int
    cache_shard_size: int


@dataclass(frozen=True)
class DatasetConfig:
    source_splits: tuple[str, ...]
    bbox_origin: str
    invalid_bbox_policy: str
    confidence_min: float
    validation_fraction: float
    max_videos: int | None
    max_frames_per_video: int | None
    video_ids: tuple[str, ...]
    deep_validate_images: bool


@dataclass(frozen=True)
class ProtocolConfig:
    deltas: tuple[int, ...]
    crop_expansion: float
    input_size: int
    hard_negative_k: int
    evaluation_split: str
    failure_top_k: int


@dataclass(frozen=True)
class ModelConfig:
    resnet_weights: str
    dinov3_hub_model: str
    dinov3_feature_source: str
    topic_config_relative: Path


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    paths: PathsConfig
    runtime: RuntimeConfig
    dataset: DatasetConfig
    protocol: ProtocolConfig
    models: ModelConfig

    @property
    def manifest_path(self) -> Path:
        return self.paths.output_root / "manifests" / "observations.csv"

    @property
    def manifest_stats_path(self) -> Path:
        return self.paths.output_root / "manifests" / "manifest_stats.json"

    @property
    def resolved_split_path(self) -> Path:
        return self.paths.output_root / "resolved_split.yaml"

    @property
    def cache_locations_path(self) -> Path:
        return self.paths.output_root / "cache_locations.json"

    def serializable(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return convert(asdict(self))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a YAML mapping")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown {label} key(s): {', '.join(unknown)}")


def _require(value: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in value:
        raise ConfigurationError(f"Missing required key: {label}.{key}")
    return value[key]


def _path(
    value: Any,
    base: Path,
    label: str,
    *,
    optional: bool = False,
    preserve_symlink: bool = False,
) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    # A venv's bin/python is normally a symlink to the base interpreter. Running
    # the resolved target bypasses the venv and therefore loses its site-packages.
    return candidate.absolute() if preserve_symlink else candidate.resolve(strict=False)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def _optional_positive_int(value: Any, label: str) -> int | None:
    return None if value is None else _positive_int(value, label)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{label} must be true or false")
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{label} must be a list of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ConfigurationError(f"{label} must contain non-empty strings")
    return result


def _ints(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{label} must be a list of positive integers")
    result = tuple(value)
    if not result or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in result):
        raise ConfigurationError(f"{label} must contain positive integers")
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.is_file():
        raise ConfigurationError(f"Config file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {error}") from error
    root = _mapping(raw, "config")
    _keys(root, {"paths", "runtime", "dataset", "protocol", "models"}, "config")
    base = config_path.parent

    p = _mapping(_require(root, "paths", "config"), "paths")
    path_keys = {
        "bee24_root", "output_root", "cache_root", "dinov3_repo", "dinov3_weights",
        "topictrack_repo", "topic_agw_weights", "dino_python", "topic_python",
    }
    _keys(p, path_keys, "paths")
    paths = PathsConfig(
        bee24_root=_path(_require(p, "bee24_root", "paths"), base, "paths.bee24_root"),  # type: ignore[arg-type]
        output_root=_path(_require(p, "output_root", "paths"), base, "paths.output_root"),  # type: ignore[arg-type]
        cache_root=_path(_require(p, "cache_root", "paths"), base, "paths.cache_root"),  # type: ignore[arg-type]
        dinov3_repo=_path(_require(p, "dinov3_repo", "paths"), base, "paths.dinov3_repo"),  # type: ignore[arg-type]
        dinov3_weights=_path(_require(p, "dinov3_weights", "paths"), base, "paths.dinov3_weights"),  # type: ignore[arg-type]
        topictrack_repo=_path(_require(p, "topictrack_repo", "paths"), base, "paths.topictrack_repo"),  # type: ignore[arg-type]
        topic_agw_weights=_path(_require(p, "topic_agw_weights", "paths"), base, "paths.topic_agw_weights"),  # type: ignore[arg-type]
        dino_python=_path(
            p.get("dino_python"), base, "paths.dino_python", optional=True, preserve_symlink=True
        ),
        topic_python=_path(
            p.get("topic_python"), base, "paths.topic_python", optional=True, preserve_symlink=True
        ),
    )
    try:
        ensure_within(paths.output_root, paths.output_root / "manifests", "manifest output")
        ensure_within(paths.cache_root, paths.cache_root / "features", "feature cache")
    except ValueError as error:
        raise ConfigurationError(str(error)) from error

    r = _mapping(_require(root, "runtime", "config"), "runtime")
    _keys(r, {"device", "batch_size", "num_workers", "amp", "amp_dtype", "seed", "cache_shard_size"}, "runtime")
    device = _require(r, "device", "runtime")
    amp_dtype = _require(r, "amp_dtype", "runtime")
    if not isinstance(device, str) or not device:
        raise ConfigurationError("runtime.device must be a non-empty string")
    if not isinstance(amp_dtype, str) or amp_dtype not in {"float16", "bfloat16"}:
        raise ConfigurationError("runtime.amp_dtype must be float16 or bfloat16")
    num_workers = _require(r, "num_workers", "runtime")
    seed = _require(r, "seed", "runtime")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ConfigurationError("runtime.num_workers must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError("runtime.seed must be a non-negative integer")
    runtime = RuntimeConfig(
        device=device,
        batch_size=_positive_int(_require(r, "batch_size", "runtime"), "runtime.batch_size"),
        num_workers=num_workers,
        amp=_bool(_require(r, "amp", "runtime"), "runtime.amp"),
        amp_dtype=amp_dtype,
        seed=seed,
        cache_shard_size=_positive_int(_require(r, "cache_shard_size", "runtime"), "runtime.cache_shard_size"),
    )

    d = _mapping(_require(root, "dataset", "config"), "dataset")
    dataset_keys = {
        "source_splits", "bbox_origin", "invalid_bbox_policy", "confidence_min",
        "validation_fraction", "max_videos", "max_frames_per_video", "video_ids",
        "deep_validate_images",
    }
    _keys(d, dataset_keys, "dataset")
    source_splits = _strings(_require(d, "source_splits", "dataset"), "dataset.source_splits")
    if any(item not in {"train", "test"} for item in source_splits):
        raise ConfigurationError("dataset.source_splits may only contain train and test")
    bbox_origin = _require(d, "bbox_origin", "dataset")
    invalid_policy = _require(d, "invalid_bbox_policy", "dataset")
    if bbox_origin not in {"one", "zero"}:
        raise ConfigurationError("dataset.bbox_origin must be one or zero")
    if invalid_policy not in {"error", "skip"}:
        raise ConfigurationError("dataset.invalid_bbox_policy must be error or skip")
    confidence_min = float(_require(d, "confidence_min", "dataset"))
    validation_fraction = float(_require(d, "validation_fraction", "dataset"))
    if not 0.0 < validation_fraction < 1.0:
        raise ConfigurationError("dataset.validation_fraction must be between 0 and 1")
    dataset = DatasetConfig(
        source_splits=source_splits,
        bbox_origin=bbox_origin,
        invalid_bbox_policy=invalid_policy,
        confidence_min=confidence_min,
        validation_fraction=validation_fraction,
        max_videos=_optional_positive_int(d.get("max_videos"), "dataset.max_videos"),
        max_frames_per_video=_optional_positive_int(d.get("max_frames_per_video"), "dataset.max_frames_per_video"),
        video_ids=_strings(d.get("video_ids", []), "dataset.video_ids"),
        deep_validate_images=_bool(_require(d, "deep_validate_images", "dataset"), "dataset.deep_validate_images"),
    )

    q = _mapping(_require(root, "protocol", "config"), "protocol")
    _keys(q, {"deltas", "crop_expansion", "input_size", "hard_negative_k", "evaluation_split", "failure_top_k"}, "protocol")
    crop_expansion = float(_require(q, "crop_expansion", "protocol"))
    if crop_expansion < 0:
        raise ConfigurationError("protocol.crop_expansion must be non-negative")
    input_size = _positive_int(_require(q, "input_size", "protocol"), "protocol.input_size")
    if input_size not in {224, 256}:
        raise ConfigurationError("protocol.input_size must be 224 or 256 for H1")
    evaluation_split = _require(q, "evaluation_split", "protocol")
    if evaluation_split not in {"train", "validation", "test"}:
        raise ConfigurationError("protocol.evaluation_split must be train, validation, or test")
    protocol = ProtocolConfig(
        deltas=_ints(_require(q, "deltas", "protocol"), "protocol.deltas"),
        crop_expansion=crop_expansion,
        input_size=input_size,
        hard_negative_k=_positive_int(_require(q, "hard_negative_k", "protocol"), "protocol.hard_negative_k"),
        evaluation_split=evaluation_split,
        failure_top_k=_positive_int(_require(q, "failure_top_k", "protocol"), "protocol.failure_top_k"),
    )

    m = _mapping(_require(root, "models", "config"), "models")
    _keys(m, {"resnet_weights", "dinov3_hub_model", "dinov3_feature_source", "topic_config_relative"}, "models")
    required_model_keys = (
        "resnet_weights", "dinov3_hub_model", "dinov3_feature_source", "topic_config_relative",
    )
    model_values = {key: _require(m, key, "models") for key in required_model_keys}
    if any(not isinstance(value, str) or not value for value in model_values.values()):
        raise ConfigurationError("all models values must be non-empty strings")
    if model_values["resnet_weights"] != "IMAGENET1K_V2":
        raise ConfigurationError("H1 pins models.resnet_weights to IMAGENET1K_V2")
    if model_values["dinov3_hub_model"] != "dinov3_vits16":
        raise ConfigurationError("H1 pins models.dinov3_hub_model to dinov3_vits16")
    if model_values["dinov3_feature_source"] not in {"patch_mean", "cls"}:
        raise ConfigurationError("models.dinov3_feature_source must be patch_mean or cls")
    topic_relative = Path(model_values["topic_config_relative"])
    if topic_relative.is_absolute() or ".." in topic_relative.parts:
        raise ConfigurationError("models.topic_config_relative must be a safe relative path")
    if topic_relative.as_posix() != "fast-reid/configs/bee/AGW_S50.yml":
        raise ConfigurationError(
            "H1 pins models.topic_config_relative to fast-reid/configs/bee/AGW_S50.yml"
        )
    models = ModelConfig(
        resnet_weights=model_values["resnet_weights"],
        dinov3_hub_model=model_values["dinov3_hub_model"],
        dinov3_feature_source=model_values["dinov3_feature_source"],
        topic_config_relative=topic_relative,
    )

    return ExperimentConfig(config_path, paths, runtime, dataset, protocol, models)
