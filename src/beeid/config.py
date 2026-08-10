"""Typed YAML configuration with all machine-specific paths kept external."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import re

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
    h1_output_root: Path | None = None
    h2_output_root: Path | None = None
    h3_output_root: Path | None = None
    h4_output_root: Path | None = None
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
    duplicate_identity_policy: str
    missing_seqinfo_policy: str
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
class H2VariantConfig:
    name: str
    crop_expansion: float
    input_size: int
    pixel_view: str
    blur_radius: float


@dataclass(frozen=True)
class H2ContaminationConfig:
    trusted_history_length: int
    ema_alpha: float
    recovery_horizon: int
    recovery_tolerance: float
    low_quality_quantile: float


@dataclass(frozen=True)
class H2Config:
    project_split_path: Path
    allow_subset: bool
    primary_variant: str
    variants: tuple[H2VariantConfig, ...]
    patch_size: int
    density_radius_multipliers: tuple[float, ...]
    history_length: int
    factor_bins: int
    bootstrap_replicates: int
    manual_annotations_csv: Path | None
    contamination: H2ContaminationConfig


@dataclass(frozen=True)
class H25Config:
    protocol_lock_path: Path
    h3_protocol_lock_path: Path
    allow_subset: bool


@dataclass(frozen=True)
class H3Config:
    protocol_lock_path: Path
    protocol_checksum_path: Path
    project_split_path: Path
    allow_subset: bool
    stage: str
    fixed_detections_root: Path | None
    history_length: int
    quantile_knots: int
    min_reliability: float
    memory_alpha: float
    update_gate: float
    appearance_weight: float
    motion_weight: float
    max_normalized_distance: float
    min_assignment_score: float
    max_age: int
    bootstrap_replicates: int


@dataclass(frozen=True)
class H4Config:
    protocol_lock_path: Path
    protocol_checksum_path: Path
    project_split_path: Path
    allow_subset: bool
    horizons: tuple[int, ...]
    go_horizon: int
    primary_horizon: int
    oracle_history_length: int
    oracle_unique_margin: float
    min_recoverable_fraction: float
    min_events_per_model: int
    min_videos_with_events: int
    beam_width: int
    max_component_size: int
    ambiguity_margin: float
    memory_alpha: float
    appearance_weight: float
    motion_weight: float
    max_normalized_distance: float
    min_assignment_score: float
    unmatched_penalty: float
    max_age: int
    noninferiority_tolerance: float
    min_nonharmed_videos: int
    bootstrap_replicates: int


@dataclass(frozen=True)
class H41Config:
    protocol_lock_path: Path
    protocol_checksum_path: Path
    project_split_path: Path
    allow_subset: bool
    audit_horizons: tuple[int, ...]
    cumulative_deadlines: tuple[int, ...]
    gate_deadline: int
    min_cumulative_recoverable_fraction: float
    min_events_per_model: int
    min_videos_with_events: int
    max_decision_horizon: int
    min_decision_lag: int
    decision_margin: float
    winner_stability_steps: int
    noninferiority_tolerance: float
    min_nonharmed_videos: int
    bootstrap_replicates: int


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    paths: PathsConfig
    runtime: RuntimeConfig
    dataset: DatasetConfig
    protocol: ProtocolConfig
    models: ModelConfig
    h2: H2Config | None
    h25: H25Config | None
    h3: H3Config | None
    h4: H4Config | None
    h41: H41Config | None

    @property
    def manifest_path(self) -> Path:
        return self.paths.output_root / "manifests" / "observations.csv"

    @property
    def manifest_stats_path(self) -> Path:
        return self.paths.output_root / "manifests" / "manifest_stats.json"

    @property
    def duplicate_identity_audit_path(self) -> Path:
        return self.paths.output_root / "manifests" / "duplicate_identity_audit.json"

    @property
    def sequence_metadata_audit_path(self) -> Path:
        return self.paths.output_root / "manifests" / "sequence_metadata_audit.json"

    @property
    def resolved_split_path(self) -> Path:
        return self.paths.output_root / "resolved_split.yaml"

    @property
    def cache_locations_path(self) -> Path:
        return self.paths.output_root / "cache_locations.json"

    @property
    def h2_source_root(self) -> Path:
        if self.paths.h1_output_root is None:
            raise ConfigurationError("paths.h1_output_root is required by H2 commands")
        return self.paths.h1_output_root

    @property
    def h2_manifest_path(self) -> Path:
        return self.h2_source_root / "manifests" / "observations.csv"

    @property
    def h2_h1_cache_locations_path(self) -> Path:
        return self.h2_source_root / "cache_locations.json"

    @property
    def h2_cache_locations_path(self) -> Path:
        return self.paths.output_root / "h2_cache_locations.json"

    @property
    def h25_source_root(self) -> Path:
        if self.paths.h2_output_root is None:
            raise ConfigurationError("paths.h2_output_root is required by H2.5 commands")
        return self.paths.h2_output_root

    @property
    def h3_manifest_path(self) -> Path:
        if self.paths.h1_output_root is None:
            raise ConfigurationError("paths.h1_output_root is required by H3 commands")
        return self.paths.h1_output_root / "manifests" / "observations.csv"

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


def _floats(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{label} must be a list of positive numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or float(item) <= 0:
            raise ConfigurationError(f"{label} must contain positive numbers")
        result.append(float(item))
    if not result:
        raise ConfigurationError(f"{label} must not be empty")
    return tuple(result)


def _bounded_float(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be numeric")
    result = float(value)
    if not low <= result <= high:
        raise ConfigurationError(f"{label} must be between {low} and {high}")
    return result


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return true when either resolved path contains the other."""
    first = left.resolve(strict=False)
    second = right.resolve(strict=False)
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.is_file():
        raise ConfigurationError(f"Config file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {error}") from error
    root = _mapping(raw, "config")
    _keys(
        root,
        {"paths", "runtime", "dataset", "protocol", "models", "h2", "h25", "h3", "h4", "h41"},
        "config",
    )
    base = config_path.parent

    p = _mapping(_require(root, "paths", "config"), "paths")
    path_keys = {
        "bee24_root", "output_root", "cache_root", "dinov3_repo", "dinov3_weights",
        "topictrack_repo", "topic_agw_weights", "h1_output_root", "h2_output_root",
        "h3_output_root", "h4_output_root",
        "dino_python", "topic_python",
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
        h1_output_root=_path(p.get("h1_output_root"), base, "paths.h1_output_root", optional=True),
        h2_output_root=_path(p.get("h2_output_root"), base, "paths.h2_output_root", optional=True),
        h3_output_root=_path(p.get("h3_output_root"), base, "paths.h3_output_root", optional=True),
        h4_output_root=_path(p.get("h4_output_root"), base, "paths.h4_output_root", optional=True),
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
        "deep_validate_images", "duplicate_identity_policy", "missing_seqinfo_policy",
    }
    _keys(d, dataset_keys, "dataset")
    source_splits = _strings(_require(d, "source_splits", "dataset"), "dataset.source_splits")
    if any(item not in {"train", "test"} for item in source_splits):
        raise ConfigurationError("dataset.source_splits may only contain train and test")
    bbox_origin = _require(d, "bbox_origin", "dataset")
    invalid_policy = _require(d, "invalid_bbox_policy", "dataset")
    duplicate_policy = _require(d, "duplicate_identity_policy", "dataset")
    missing_seqinfo_policy = _require(d, "missing_seqinfo_policy", "dataset")
    if bbox_origin not in {"one", "zero"}:
        raise ConfigurationError("dataset.bbox_origin must be one or zero")
    if invalid_policy not in {"error", "skip"}:
        raise ConfigurationError("dataset.invalid_bbox_policy must be error or skip")
    if duplicate_policy not in {"error", "exclude_conflict"}:
        raise ConfigurationError(
            "dataset.duplicate_identity_policy must be error or exclude_conflict"
        )
    if missing_seqinfo_policy not in {"error", "infer_from_images"}:
        raise ConfigurationError(
            "dataset.missing_seqinfo_policy must be error or infer_from_images"
        )
    confidence_min = float(_require(d, "confidence_min", "dataset"))
    validation_fraction = float(_require(d, "validation_fraction", "dataset"))
    if not 0.0 < validation_fraction < 1.0:
        raise ConfigurationError("dataset.validation_fraction must be between 0 and 1")
    dataset = DatasetConfig(
        source_splits=source_splits,
        bbox_origin=bbox_origin,
        invalid_bbox_policy=invalid_policy,
        duplicate_identity_policy=duplicate_policy,
        missing_seqinfo_policy=missing_seqinfo_policy,
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

    h2: H2Config | None = None
    if "h2" in root and root["h2"] is not None:
        h = _mapping(root["h2"], "h2")
        _keys(
            h,
            {
                "project_split", "allow_subset", "primary_variant", "variants", "patch_size",
                "density_radius_multipliers", "history_length", "factor_bins",
                "bootstrap_replicates", "manual_annotations_csv", "contamination",
            },
            "h2",
        )
        if paths.h1_output_root is None:
            raise ConfigurationError("paths.h1_output_root is required when h2 is configured")
        if paths.h1_output_root.resolve(strict=False) == paths.output_root.resolve(strict=False):
            raise ConfigurationError("H2 output_root must differ from paths.h1_output_root")
        if protocol.evaluation_split != "validation" or dataset.source_splits != ("train",):
            raise ConfigurationError(
                "H2 diagnostics are restricted to the frozen train-derived development validation split"
            )

        variant_values = _require(h, "variants", "h2")
        if isinstance(variant_values, (str, bytes)) or not isinstance(variant_values, Sequence):
            raise ConfigurationError("h2.variants must be a list of mappings")
        variants: list[H2VariantConfig] = []
        names: set[str] = set()
        for index, value in enumerate(variant_values):
            variant = _mapping(value, f"h2.variants[{index}]")
            _keys(
                variant,
                {"name", "crop_expansion", "input_size", "pixel_view", "blur_radius"},
                f"h2.variants[{index}]",
            )
            name = _require(variant, "name", f"h2.variants[{index}]")
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ConfigurationError(
                    f"h2.variants[{index}].name must match [a-z][a-z0-9_]*"
                )
            if name in names:
                raise ConfigurationError(f"Duplicate H2 variant name: {name}")
            names.add(name)
            expansion = _bounded_float(
                _require(variant, "crop_expansion", f"h2.variants[{index}]"),
                f"h2.variants[{index}].crop_expansion",
                0.0,
                1.0,
            )
            variant_input = _positive_int(
                _require(variant, "input_size", f"h2.variants[{index}]"),
                f"h2.variants[{index}].input_size",
            )
            if variant_input not in {224, 256}:
                raise ConfigurationError("H2 variant input_size must be 224 or 256")
            pixel_view = _require(variant, "pixel_view", f"h2.variants[{index}]")
            if pixel_view not in {"raw", "bbox_foreground_only", "context_only", "gaussian_blur"}:
                raise ConfigurationError(
                    "H2 pixel_view must be raw, bbox_foreground_only, context_only, or gaussian_blur"
                )
            blur_radius = _bounded_float(
                variant.get("blur_radius", 0.0),
                f"h2.variants[{index}].blur_radius",
                0.0,
                20.0,
            )
            if pixel_view == "gaussian_blur" and blur_radius <= 0:
                raise ConfigurationError("gaussian_blur variants require blur_radius > 0")
            if pixel_view != "gaussian_blur" and blur_radius != 0:
                raise ConfigurationError("blur_radius must be 0 except for gaussian_blur variants")
            if pixel_view == "context_only" and expansion <= 0:
                raise ConfigurationError("context_only variants require positive crop_expansion")
            variants.append(H2VariantConfig(name, expansion, variant_input, pixel_view, blur_radius))
        if not variants:
            raise ConfigurationError("h2.variants must not be empty")
        primary_variant = _require(h, "primary_variant", "h2")
        if primary_variant not in names:
            raise ConfigurationError("h2.primary_variant must name one configured variant")

        contamination_raw = _mapping(_require(h, "contamination", "h2"), "h2.contamination")
        _keys(
            contamination_raw,
            {
                "trusted_history_length", "ema_alpha", "recovery_horizon",
                "recovery_tolerance", "low_quality_quantile",
            },
            "h2.contamination",
        )
        contamination = H2ContaminationConfig(
            trusted_history_length=_positive_int(
                _require(contamination_raw, "trusted_history_length", "h2.contamination"),
                "h2.contamination.trusted_history_length",
            ),
            ema_alpha=_bounded_float(
                _require(contamination_raw, "ema_alpha", "h2.contamination"),
                "h2.contamination.ema_alpha",
                0.0,
                1.0,
            ),
            recovery_horizon=_positive_int(
                _require(contamination_raw, "recovery_horizon", "h2.contamination"),
                "h2.contamination.recovery_horizon",
            ),
            recovery_tolerance=_bounded_float(
                _require(contamination_raw, "recovery_tolerance", "h2.contamination"),
                "h2.contamination.recovery_tolerance",
                0.0,
                1.0,
            ),
            low_quality_quantile=_bounded_float(
                _require(contamination_raw, "low_quality_quantile", "h2.contamination"),
                "h2.contamination.low_quality_quantile",
                0.01,
                0.49,
            ),
        )
        factor_bins = _positive_int(_require(h, "factor_bins", "h2"), "h2.factor_bins")
        if factor_bins < 2 or factor_bins > 10:
            raise ConfigurationError("h2.factor_bins must be between 2 and 10")
        h2 = H2Config(
            project_split_path=_path(
                _require(h, "project_split", "h2"), base, "h2.project_split"
            ),  # type: ignore[arg-type]
            allow_subset=_bool(_require(h, "allow_subset", "h2"), "h2.allow_subset"),
            primary_variant=str(primary_variant),
            variants=tuple(variants),
            patch_size=_positive_int(_require(h, "patch_size", "h2"), "h2.patch_size"),
            density_radius_multipliers=_floats(
                _require(h, "density_radius_multipliers", "h2"),
                "h2.density_radius_multipliers",
            ),
            history_length=_positive_int(
                _require(h, "history_length", "h2"), "h2.history_length"
            ),
            factor_bins=factor_bins,
            bootstrap_replicates=_positive_int(
                _require(h, "bootstrap_replicates", "h2"),
                "h2.bootstrap_replicates",
            ),
            manual_annotations_csv=_path(
                h.get("manual_annotations_csv"), base, "h2.manual_annotations_csv", optional=True
            ),
            contamination=contamination,
        )

    h25: H25Config | None = None
    if "h25" in root and root["h25"] is not None:
        value = _mapping(root["h25"], "h25")
        _keys(value, {"protocol_lock", "h3_protocol_lock", "allow_subset"}, "h25")
        if h2 is None:
            raise ConfigurationError("h25 requires the h2 section that defines the reused primary variant")
        if paths.h2_output_root is None:
            raise ConfigurationError("paths.h2_output_root is required when h25 is configured")
        roots = {
            paths.output_root.resolve(strict=False),
            paths.h1_output_root.resolve(strict=False) if paths.h1_output_root else None,
            paths.h2_output_root.resolve(strict=False),
        }
        if len(roots) != 3:
            raise ConfigurationError("H2.5 output_root must differ from H1 and H2 source roots")
        if protocol.evaluation_split != "validation" or dataset.source_splits != ("train",):
            raise ConfigurationError(
                "H2.5 is restricted to the frozen train-derived development validation split"
            )
        h25 = H25Config(
            protocol_lock_path=_path(
                _require(value, "protocol_lock", "h25"), base, "h25.protocol_lock"
            ),  # type: ignore[arg-type]
            h3_protocol_lock_path=_path(
                _require(value, "h3_protocol_lock", "h25"), base, "h25.h3_protocol_lock"
            ),  # type: ignore[arg-type]
            allow_subset=_bool(_require(value, "allow_subset", "h25"), "h25.allow_subset"),
        )

    h3: H3Config | None = None
    if "h3" in root and root["h3"] is not None:
        value = _mapping(root["h3"], "h3")
        _keys(
            value,
            {
                "protocol_lock", "protocol_checksum", "project_split", "allow_subset",
                "stage", "fixed_detections_root", "history_length", "quantile_knots",
                "min_reliability", "memory_alpha", "update_gate", "appearance_weight",
                "motion_weight", "max_normalized_distance", "min_assignment_score",
                "max_age", "bootstrap_replicates",
            },
            "h3",
        )
        if paths.h1_output_root is None:
            raise ConfigurationError("paths.h1_output_root is required when h3 is configured")
        if paths.h1_output_root.resolve(strict=False) == paths.output_root.resolve(strict=False):
            raise ConfigurationError("H3 output_root must differ from paths.h1_output_root")
        if dataset.source_splits != ("train",):
            raise ConfigurationError("H3 development is restricted to the official train tree")
        stage = _require(value, "stage", "h3")
        if stage not in {"gt_detection_boxes", "fixed_detector_boxes"}:
            raise ConfigurationError(
                "h3.stage must be gt_detection_boxes or fixed_detector_boxes"
            )
        fixed_root = _path(
            value.get("fixed_detections_root"),
            base,
            "h3.fixed_detections_root",
            optional=True,
        )
        if stage == "fixed_detector_boxes" and fixed_root is None:
            raise ConfigurationError(
                "h3.fixed_detections_root is required for fixed_detector_boxes"
            )
        quantile_knots = _positive_int(
            _require(value, "quantile_knots", "h3"), "h3.quantile_knots"
        )
        if quantile_knots < 11 or quantile_knots > 1001 or quantile_knots % 2 == 0:
            raise ConfigurationError("h3.quantile_knots must be an odd integer in [11, 1001]")
        appearance_weight = _bounded_float(
            _require(value, "appearance_weight", "h3"),
            "h3.appearance_weight",
            0.0,
            1.0,
        )
        motion_weight = _bounded_float(
            _require(value, "motion_weight", "h3"), "h3.motion_weight", 0.0, 1.0
        )
        if appearance_weight + motion_weight <= 0:
            raise ConfigurationError("H3 appearance_weight and motion_weight cannot both be zero")
        history_length = _positive_int(
            _require(value, "history_length", "h3"), "h3.history_length"
        )
        if history_length != 5:
            raise ConfigurationError("h3.history_length is frozen at 5")
        h3 = H3Config(
            protocol_lock_path=_path(
                _require(value, "protocol_lock", "h3"), base, "h3.protocol_lock"
            ),  # type: ignore[arg-type]
            protocol_checksum_path=_path(
                _require(value, "protocol_checksum", "h3"),
                base,
                "h3.protocol_checksum",
            ),  # type: ignore[arg-type]
            project_split_path=_path(
                _require(value, "project_split", "h3"), base, "h3.project_split"
            ),  # type: ignore[arg-type]
            allow_subset=_bool(_require(value, "allow_subset", "h3"), "h3.allow_subset"),
            stage=str(stage),
            fixed_detections_root=fixed_root,
            history_length=history_length,
            quantile_knots=quantile_knots,
            min_reliability=_bounded_float(
                _require(value, "min_reliability", "h3"),
                "h3.min_reliability",
                0.0,
                1.0,
            ),
            memory_alpha=_bounded_float(
                _require(value, "memory_alpha", "h3"), "h3.memory_alpha", 0.0, 1.0
            ),
            update_gate=_bounded_float(
                _require(value, "update_gate", "h3"), "h3.update_gate", 0.0, 1.0
            ),
            appearance_weight=appearance_weight,
            motion_weight=motion_weight,
            max_normalized_distance=_bounded_float(
                _require(value, "max_normalized_distance", "h3"),
                "h3.max_normalized_distance",
                0.01,
                100.0,
            ),
            min_assignment_score=_bounded_float(
                _require(value, "min_assignment_score", "h3"),
                "h3.min_assignment_score",
                0.0,
                1.0,
            ),
            max_age=_positive_int(_require(value, "max_age", "h3"), "h3.max_age"),
            bootstrap_replicates=_positive_int(
                _require(value, "bootstrap_replicates", "h3"),
                "h3.bootstrap_replicates",
            ),
        )

    h4: H4Config | None = None
    if "h4" in root and root["h4"] is not None:
        value = _mapping(root["h4"], "h4")
        _keys(
            value,
            {
                "protocol_lock", "protocol_checksum", "project_split", "allow_subset",
                "horizons", "go_horizon", "primary_horizon", "oracle_history_length",
                "oracle_unique_margin", "min_recoverable_fraction", "min_events_per_model",
                "min_videos_with_events", "beam_width", "max_component_size",
                "ambiguity_margin", "memory_alpha", "appearance_weight", "motion_weight",
                "max_normalized_distance", "min_assignment_score", "unmatched_penalty",
                "max_age", "noninferiority_tolerance", "min_nonharmed_videos",
                "bootstrap_replicates",
            },
            "h4",
        )
        if paths.h1_output_root is None:
            raise ConfigurationError("paths.h1_output_root is required when h4 is configured")
        if paths.h3_output_root is None:
            raise ConfigurationError("paths.h3_output_root is required when h4 is configured")
        roots = {
            paths.output_root.resolve(strict=False),
            paths.h1_output_root.resolve(strict=False),
            paths.h3_output_root.resolve(strict=False),
        }
        if len(roots) != 3:
            raise ConfigurationError("H4 output_root must differ from H1 and H3 source roots")
        if dataset.source_splits != ("train",) or protocol.evaluation_split != "validation":
            raise ConfigurationError(
                "H4 development is restricted to the frozen train-derived validation split"
            )
        horizons = _ints(_require(value, "horizons", "h4"), "h4.horizons")
        if tuple(sorted(set(horizons))) != horizons:
            raise ConfigurationError("h4.horizons must be unique and strictly increasing")
        go_horizon = _positive_int(_require(value, "go_horizon", "h4"), "h4.go_horizon")
        primary_horizon = _positive_int(
            _require(value, "primary_horizon", "h4"), "h4.primary_horizon"
        )
        if go_horizon not in horizons or primary_horizon not in horizons:
            raise ConfigurationError("h4.go_horizon and primary_horizon must occur in h4.horizons")
        appearance_weight = _bounded_float(
            _require(value, "appearance_weight", "h4"), "h4.appearance_weight", 0.0, 1.0
        )
        motion_weight = _bounded_float(
            _require(value, "motion_weight", "h4"), "h4.motion_weight", 0.0, 1.0
        )
        if appearance_weight + motion_weight <= 0:
            raise ConfigurationError("H4 appearance_weight and motion_weight cannot both be zero")
        max_component_size = _positive_int(
            _require(value, "max_component_size", "h4"), "h4.max_component_size"
        )
        if max_component_size < 2 or max_component_size > 6:
            raise ConfigurationError("h4.max_component_size must be in [2, 6]")
        h4 = H4Config(
            protocol_lock_path=_path(
                _require(value, "protocol_lock", "h4"), base, "h4.protocol_lock"
            ),  # type: ignore[arg-type]
            protocol_checksum_path=_path(
                _require(value, "protocol_checksum", "h4"), base, "h4.protocol_checksum"
            ),  # type: ignore[arg-type]
            project_split_path=_path(
                _require(value, "project_split", "h4"), base, "h4.project_split"
            ),  # type: ignore[arg-type]
            allow_subset=_bool(_require(value, "allow_subset", "h4"), "h4.allow_subset"),
            horizons=horizons,
            go_horizon=go_horizon,
            primary_horizon=primary_horizon,
            oracle_history_length=_positive_int(
                _require(value, "oracle_history_length", "h4"),
                "h4.oracle_history_length",
            ),
            oracle_unique_margin=_bounded_float(
                _require(value, "oracle_unique_margin", "h4"),
                "h4.oracle_unique_margin", 0.0, 1.0,
            ),
            min_recoverable_fraction=_bounded_float(
                _require(value, "min_recoverable_fraction", "h4"),
                "h4.min_recoverable_fraction", 0.0, 1.0,
            ),
            min_events_per_model=_positive_int(
                _require(value, "min_events_per_model", "h4"), "h4.min_events_per_model"
            ),
            min_videos_with_events=_positive_int(
                _require(value, "min_videos_with_events", "h4"),
                "h4.min_videos_with_events",
            ),
            beam_width=_positive_int(_require(value, "beam_width", "h4"), "h4.beam_width"),
            max_component_size=max_component_size,
            ambiguity_margin=_bounded_float(
                _require(value, "ambiguity_margin", "h4"), "h4.ambiguity_margin", 0.0, 1.0
            ),
            memory_alpha=_bounded_float(
                _require(value, "memory_alpha", "h4"), "h4.memory_alpha", 0.0, 1.0
            ),
            appearance_weight=appearance_weight,
            motion_weight=motion_weight,
            max_normalized_distance=_bounded_float(
                _require(value, "max_normalized_distance", "h4"),
                "h4.max_normalized_distance", 0.01, 100.0,
            ),
            min_assignment_score=_bounded_float(
                _require(value, "min_assignment_score", "h4"),
                "h4.min_assignment_score", 0.0, 1.0,
            ),
            unmatched_penalty=_bounded_float(
                _require(value, "unmatched_penalty", "h4"),
                "h4.unmatched_penalty", 0.0, 1.0,
            ),
            max_age=_positive_int(_require(value, "max_age", "h4"), "h4.max_age"),
            noninferiority_tolerance=_bounded_float(
                _require(value, "noninferiority_tolerance", "h4"),
                "h4.noninferiority_tolerance", 0.0, 0.1,
            ),
            min_nonharmed_videos=_positive_int(
                _require(value, "min_nonharmed_videos", "h4"),
                "h4.min_nonharmed_videos",
            ),
            bootstrap_replicates=_positive_int(
                _require(value, "bootstrap_replicates", "h4"),
                "h4.bootstrap_replicates",
            ),
        )

    h41: H41Config | None = None
    if "h41" in root and root["h41"] is not None:
        value = _mapping(root["h41"], "h41")
        _keys(
            value,
            {
                "protocol_lock", "protocol_checksum", "project_split", "allow_subset",
                "audit_horizons", "cumulative_deadlines", "gate_deadline",
                "min_cumulative_recoverable_fraction", "min_events_per_model",
                "min_videos_with_events", "max_decision_horizon", "min_decision_lag",
                "decision_margin", "winner_stability_steps", "noninferiority_tolerance",
                "min_nonharmed_videos", "bootstrap_replicates",
            },
            "h41",
        )
        if h4 is None:
            raise ConfigurationError("h41 requires the frozen h4 section for shared tracker parameters")
        if paths.h4_output_root is None:
            raise ConfigurationError("paths.h4_output_root is required when h41 is configured")
        if paths.h1_output_root is None or paths.h3_output_root is None:
            raise ConfigurationError("H4.1 requires paths.h1_output_root and paths.h3_output_root")
        roots = {
            paths.output_root.resolve(strict=False),
            paths.h1_output_root.resolve(strict=False),
            paths.h3_output_root.resolve(strict=False),
            paths.h4_output_root.resolve(strict=False),
        }
        if len(roots) != 4:
            raise ConfigurationError("H4.1 output_root must differ from H1, H3, and H4-v1 roots")
        for label, source_root in (
            ("H1", paths.h1_output_root),
            ("H3", paths.h3_output_root),
            ("H4-v1", paths.h4_output_root),
        ):
            if _paths_overlap(paths.output_root, source_root):
                raise ConfigurationError(
                    f"H4.1 output_root must not contain or be contained by the {label} source root"
                )
        if dataset.source_splits != ("train",) or protocol.evaluation_split != "validation":
            raise ConfigurationError(
                "H4.1 development is restricted to the frozen train-derived validation split"
            )
        audit_horizons = _ints(
            _require(value, "audit_horizons", "h41"), "h41.audit_horizons"
        )
        if audit_horizons != tuple(range(1, 11)):
            raise ConfigurationError("h41.audit_horizons is frozen at every lag from 1 through 10")
        deadlines = _ints(
            _require(value, "cumulative_deadlines", "h41"),
            "h41.cumulative_deadlines",
        )
        if tuple(sorted(set(deadlines))) != deadlines or any(
            item not in audit_horizons for item in deadlines
        ):
            raise ConfigurationError(
                "h41.cumulative_deadlines must be unique, increasing audit horizons"
            )
        gate_deadline = _positive_int(
            _require(value, "gate_deadline", "h41"), "h41.gate_deadline"
        )
        max_decision_horizon = _positive_int(
            _require(value, "max_decision_horizon", "h41"),
            "h41.max_decision_horizon",
        )
        if gate_deadline not in deadlines:
            raise ConfigurationError("h41.gate_deadline must occur in cumulative_deadlines")
        if max_decision_horizon != gate_deadline:
            raise ConfigurationError(
                "h41.max_decision_horizon must equal the cumulative gate deadline"
            )
        min_decision_lag = _positive_int(
            _require(value, "min_decision_lag", "h41"), "h41.min_decision_lag"
        )
        if min_decision_lag > max_decision_horizon:
            raise ConfigurationError("h41.min_decision_lag cannot exceed max_decision_horizon")
        h41 = H41Config(
            protocol_lock_path=_path(
                _require(value, "protocol_lock", "h41"), base, "h41.protocol_lock"
            ),  # type: ignore[arg-type]
            protocol_checksum_path=_path(
                _require(value, "protocol_checksum", "h41"),
                base,
                "h41.protocol_checksum",
            ),  # type: ignore[arg-type]
            project_split_path=_path(
                _require(value, "project_split", "h41"), base, "h41.project_split"
            ),  # type: ignore[arg-type]
            allow_subset=_bool(_require(value, "allow_subset", "h41"), "h41.allow_subset"),
            audit_horizons=audit_horizons,
            cumulative_deadlines=deadlines,
            gate_deadline=gate_deadline,
            min_cumulative_recoverable_fraction=_bounded_float(
                _require(value, "min_cumulative_recoverable_fraction", "h41"),
                "h41.min_cumulative_recoverable_fraction",
                0.0,
                1.0,
            ),
            min_events_per_model=_positive_int(
                _require(value, "min_events_per_model", "h41"),
                "h41.min_events_per_model",
            ),
            min_videos_with_events=_positive_int(
                _require(value, "min_videos_with_events", "h41"),
                "h41.min_videos_with_events",
            ),
            max_decision_horizon=max_decision_horizon,
            min_decision_lag=min_decision_lag,
            decision_margin=_bounded_float(
                _require(value, "decision_margin", "h41"),
                "h41.decision_margin",
                0.0,
                1.0,
            ),
            winner_stability_steps=_positive_int(
                _require(value, "winner_stability_steps", "h41"),
                "h41.winner_stability_steps",
            ),
            noninferiority_tolerance=_bounded_float(
                _require(value, "noninferiority_tolerance", "h41"),
                "h41.noninferiority_tolerance",
                0.0,
                0.1,
            ),
            min_nonharmed_videos=_positive_int(
                _require(value, "min_nonharmed_videos", "h41"),
                "h41.min_nonharmed_videos",
            ),
            bootstrap_replicates=_positive_int(
                _require(value, "bootstrap_replicates", "h41"),
                "h41.bootstrap_replicates",
            ),
        )

    return ExperimentConfig(
        config_path, paths, runtime, dataset, protocol, models, h2, h25, h3, h4, h41
    )
