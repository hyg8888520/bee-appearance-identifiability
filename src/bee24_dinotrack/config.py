"""Typed YAML configuration with no machine-specific path defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class ConfigurationError(ValueError):
    """Raised when a YAML configuration is missing or invalid."""


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    annotation: Path
    max_images: int | None


@dataclass(frozen=True)
class DinoV3Config:
    checkpoint: Path
    model_name: str
    image_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str
    global_pool: str
    batch_size: int
    num_workers: int
    use_amp: bool
    amp_dtype: str
    normalize_features: bool


@dataclass(frozen=True)
class H1Config:
    query_image_ids: tuple[int, ...]
    query_stride: int
    top_k: int
    reuse_feature_cache: bool


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    feature_cache: Path
    query_results: Path
    figures: Path
    failure_cases: Path
    metadata: Path


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    name: str
    seed: int
    device: str
    dataset: DatasetConfig
    dinov3: DinoV3Config
    h1: H1Config
    output: OutputConfig

    def serializable(self) -> dict[str, Any]:
        """Return the resolved configuration as JSON-compatible values."""

        payload = asdict(self)

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return convert(payload)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"'{location}' must be a YAML mapping.")
    return value


def _require(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing required configuration key: {location}.{key}")
    return mapping[key]


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unexpected = sorted(set(mapping) - allowed)
    if unexpected:
        joined = ", ".join(f"'{location}.{key}'" for key in unexpected)
        raise ConfigurationError(f"Unknown configuration key(s): {joined}")


def _resolve_config_path(value: Any, base_dir: Path, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{location}' must be a non-empty path string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"'{location}' must be a positive integer.")
    return value


def _nonnegative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"'{location}' must be a non-negative integer.")
    return value


def _bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"'{location}' must be true or false.")
    return value


def _number_tuple(
    value: Any, location: str, length: int, cast: type[int] | type[float]
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ConfigurationError(f"'{location}' must contain exactly {length} values.")
    try:
        result = tuple(cast(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"'{location}' contains a non-numeric value.") from error
    return result


def _output_path(root: Path, value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{location}' must be a non-empty path string.")
    configured = Path(value).expanduser()
    candidate = configured if configured.is_absolute() else root / configured
    candidate = candidate.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ConfigurationError(
            f"'{location}' resolves outside output.root: {candidate}. "
            "Use a relative path or a path beneath output.root."
        ) from error
    return candidate


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate the full H1 YAML configuration."""

    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.is_file():
        raise ConfigurationError(
            f"Config file does not exist: {config_path}. Copy the example config and edit its paths."
        )
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {error}") from error
    root = _mapping(raw, "config")
    _check_keys(root, {"experiment", "dataset", "models", "h1", "output"}, "config")
    base_dir = config_path.parent

    experiment = _mapping(_require(root, "experiment", "config"), "experiment")
    _check_keys(experiment, {"name", "seed", "device"}, "experiment")
    name = _require(experiment, "name", "experiment")
    device = _require(experiment, "device", "experiment")
    seed = _require(experiment, "seed", "experiment")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("'experiment.name' must be a non-empty string.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError("'experiment.seed' must be a non-negative integer.")
    if not isinstance(device, str) or not device.strip():
        raise ConfigurationError("'experiment.device' must be a non-empty string.")

    dataset_raw = _mapping(_require(root, "dataset", "config"), "dataset")
    _check_keys(dataset_raw, {"root", "annotation", "max_images"}, "dataset")
    max_images = _require(dataset_raw, "max_images", "dataset")
    if max_images is not None:
        max_images = _positive_int(max_images, "dataset.max_images")
    dataset = DatasetConfig(
        root=_resolve_config_path(_require(dataset_raw, "root", "dataset"), base_dir, "dataset.root"),
        annotation=_resolve_config_path(
            _require(dataset_raw, "annotation", "dataset"), base_dir, "dataset.annotation"
        ),
        max_images=max_images,
    )

    models = _mapping(_require(root, "models", "config"), "models")
    _check_keys(models, {"dinov3"}, "models")
    dino = _mapping(_require(models, "dinov3", "models"), "models.dinov3")
    dino_keys = {
        "checkpoint",
        "model_name",
        "image_size",
        "mean",
        "std",
        "interpolation",
        "global_pool",
        "batch_size",
        "num_workers",
        "use_amp",
        "amp_dtype",
        "normalize_features",
    }
    _check_keys(dino, dino_keys, "models.dinov3")
    image_size = _number_tuple(
        _require(dino, "image_size", "models.dinov3"), "models.dinov3.image_size", 2, int
    )
    if any(item <= 0 for item in image_size):
        raise ConfigurationError("'models.dinov3.image_size' values must be positive.")
    mean = _number_tuple(_require(dino, "mean", "models.dinov3"), "models.dinov3.mean", 3, float)
    std = _number_tuple(_require(dino, "std", "models.dinov3"), "models.dinov3.std", 3, float)
    if any(item <= 0 for item in std):
        raise ConfigurationError("'models.dinov3.std' values must be positive.")
    model_name = _require(dino, "model_name", "models.dinov3")
    interpolation = _require(dino, "interpolation", "models.dinov3")
    global_pool = _require(dino, "global_pool", "models.dinov3")
    amp_dtype = _require(dino, "amp_dtype", "models.dinov3")
    for value, location in (
        (model_name, "models.dinov3.model_name"),
        (interpolation, "models.dinov3.interpolation"),
        (global_pool, "models.dinov3.global_pool"),
        (amp_dtype, "models.dinov3.amp_dtype"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"'{location}' must be a non-empty string.")
    dinov3 = DinoV3Config(
        checkpoint=_resolve_config_path(
            _require(dino, "checkpoint", "models.dinov3"), base_dir, "models.dinov3.checkpoint"
        ),
        model_name=model_name,
        image_size=(image_size[0], image_size[1]),
        mean=(mean[0], mean[1], mean[2]),
        std=(std[0], std[1], std[2]),
        interpolation=interpolation,
        global_pool=global_pool,
        batch_size=_positive_int(_require(dino, "batch_size", "models.dinov3"), "models.dinov3.batch_size"),
        num_workers=_nonnegative_int(
            _require(dino, "num_workers", "models.dinov3"), "models.dinov3.num_workers"
        ),
        use_amp=_bool(_require(dino, "use_amp", "models.dinov3"), "models.dinov3.use_amp"),
        amp_dtype=amp_dtype,
        normalize_features=_bool(
            _require(dino, "normalize_features", "models.dinov3"),
            "models.dinov3.normalize_features",
        ),
    )

    h1_raw = _mapping(_require(root, "h1", "config"), "h1")
    _check_keys(h1_raw, {"query_image_ids", "query_stride", "top_k", "reuse_feature_cache"}, "h1")
    query_ids = _require(h1_raw, "query_image_ids", "h1")
    if isinstance(query_ids, (str, bytes)) or not isinstance(query_ids, Sequence):
        raise ConfigurationError("'h1.query_image_ids' must be a list of integer image IDs.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in query_ids):
        raise ConfigurationError("'h1.query_image_ids' must contain only integer image IDs.")
    h1 = H1Config(
        query_image_ids=tuple(query_ids),
        query_stride=_positive_int(_require(h1_raw, "query_stride", "h1"), "h1.query_stride"),
        top_k=_positive_int(_require(h1_raw, "top_k", "h1"), "h1.top_k"),
        reuse_feature_cache=_bool(
            _require(h1_raw, "reuse_feature_cache", "h1"), "h1.reuse_feature_cache"
        ),
    )

    output_raw = _mapping(_require(root, "output", "config"), "output")
    output_keys = {"root", "feature_cache", "query_results", "figures", "failure_cases", "metadata"}
    _check_keys(output_raw, output_keys, "output")
    output_root = _resolve_config_path(_require(output_raw, "root", "output"), base_dir, "output.root")
    output = OutputConfig(
        root=output_root,
        feature_cache=_output_path(
            output_root, _require(output_raw, "feature_cache", "output"), "output.feature_cache"
        ),
        query_results=_output_path(
            output_root, _require(output_raw, "query_results", "output"), "output.query_results"
        ),
        figures=_output_path(output_root, _require(output_raw, "figures", "output"), "output.figures"),
        failure_cases=_output_path(
            output_root, _require(output_raw, "failure_cases", "output"), "output.failure_cases"
        ),
        metadata=_output_path(output_root, _require(output_raw, "metadata", "output"), "output.metadata"),
    )
    named_outputs = {
        "output.feature_cache": output.feature_cache,
        "output.query_results": output.query_results,
        "output.figures": output.figures,
        "output.failure_cases": output.failure_cases,
        "output.metadata": output.metadata,
    }
    for candidate in named_outputs.values():
        duplicates = [other_name for other_name, other in named_outputs.items() if other == candidate]
        if len(duplicates) > 1:
            raise ConfigurationError(
                f"Output paths must be unique; {', '.join(duplicates)} all resolve to {candidate}."
            )
    file_outputs = {output.feature_cache, output.query_results, output.metadata}
    directory_outputs = {output.figures, output.failure_cases}
    overlap = file_outputs & directory_outputs
    if overlap:
        raise ConfigurationError(f"Output path is configured as both a file and directory: {overlap.pop()}")
    for file_path in file_outputs:
        if file_path == output.root:
            raise ConfigurationError(f"Output file path cannot equal output.root: {file_path}")
        for other in named_outputs.values():
            if other != file_path and file_path in other.parents:
                raise ConfigurationError(
                    f"Output file path cannot be the parent of another output path: {file_path}"
                )

    return ExperimentConfig(
        config_path=config_path,
        name=name,
        seed=seed,
        device=device,
        dataset=dataset,
        dinov3=dinov3,
        h1=h1,
        output=output,
    )
