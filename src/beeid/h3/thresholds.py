"""Project-train-only calibration for causal H3 reliability features."""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..utils import atomic_write_json, canonical_json, sha256_file, sha256_text
from .core import H3Inputs, require_h3, validate_h3_inputs
from .features import load_h3_embeddings
from .signals import read_h3_signal_rows


COMMON_CALIBRATORS = (
    "bbox_scale_change",
    "orientation_change_proxy",
    "sharpness_change",
    "max_bbox_iou",
    "neighbor_count_wide",
)


def _finite_float(value: str | float | int, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite H3 signal {label}: {value!r}")
    return result


def axial_difference(left: float, right: float) -> float:
    raw = abs(float(left) - float(right)) % 180.0
    return min(raw, 180.0 - raw)


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Cannot normalize an empty/non-finite H3 template")
    return value / norm


def _quantile_knots(values: Sequence[float], count: int, label: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise RuntimeError(f"Insufficient finite project_train values for {label}")
    return [
        float(value)
        for value in np.quantile(array, np.linspace(0.0, 1.0, count))
    ]


def percentile_from_knots(knots: Sequence[float], value: float) -> float:
    if not knots:
        raise ValueError("Reliability calibrator knots must not be empty")
    finite = float(value)
    if not math.isfinite(finite):
        return 1.0
    left = bisect.bisect_left(knots, finite)
    right = bisect.bisect_right(knots, finite)
    # Midrank treatment prevents a common tied value (for example zero IoU)
    # from being assigned the maximum edge of its tie block.
    percentile = (left + right) / (2.0 * len(knots))
    return min(1.0, max(0.0, percentile))


def dynamic_pair_values(
    previous_signal: Mapping[str, Any],
    current_signal: Mapping[str, Any],
    memory: np.ndarray,
    embedding: np.ndarray,
) -> dict[str, float]:
    previous_area = max(_finite_float(previous_signal["bbox_area"], "bbox_area"), 1e-12)
    current_area = max(_finite_float(current_signal["bbox_area"], "bbox_area"), 1e-12)
    previous_sharpness = max(
        _finite_float(previous_signal["bbox_laplacian_variance"], "bbox_laplacian_variance"),
        1e-12,
    )
    current_sharpness = max(
        _finite_float(current_signal["bbox_laplacian_variance"], "bbox_laplacian_variance"),
        1e-12,
    )
    cosine = float(np.dot(_normalize(memory), _normalize(embedding)))
    return {
        "identity_history_outlier": max(0.0, 1.0 - cosine),
        "bbox_scale_change": abs(math.log(current_area / previous_area)),
        "orientation_change_proxy": axial_difference(
            _finite_float(previous_signal["bbox_orientation_proxy_deg"], "orientation"),
            _finite_float(current_signal["bbox_orientation_proxy_deg"], "orientation"),
        ),
        "sharpness_change": abs(math.log(current_sharpness / previous_sharpness)),
        "max_bbox_iou": _finite_float(current_signal["max_bbox_iou"], "max_bbox_iou"),
        "neighbor_count_wide": _finite_float(
            current_signal["neighbor_count_wide"], "neighbor_count_wide"
        ),
    }


def reliability_from_artifact(
    artifact: Mapping[str, Any], model_name: str, values: Mapping[str, float]
) -> tuple[float, dict[str, float]]:
    try:
        common = artifact["common_calibrators"]
        history_knots = artifact["models"][model_name]["identity_history_outlier"]["knots"]
        minimum = float(artifact["reliability"]["min_reliability"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid H3 threshold artifact") from error
    risks = {
        "identity_history_outlier": percentile_from_knots(
            history_knots, values["identity_history_outlier"]
        ),
        "bbox_scale_change": percentile_from_knots(
            common["bbox_scale_change"]["knots"], values["bbox_scale_change"]
        ),
        "orientation_change_proxy": percentile_from_knots(
            common["orientation_change_proxy"]["knots"],
            values["orientation_change_proxy"],
        ),
        "sharpness_change": percentile_from_knots(
            common["sharpness_change"]["knots"], values["sharpness_change"]
        ),
    }
    risks["crowding_overlap"] = max(
        percentile_from_knots(
            common["max_bbox_iou"]["knots"], values["max_bbox_iou"]
        ),
        percentile_from_knots(
            common["neighbor_count_wide"]["knots"],
            values["neighbor_count_wide"],
        ),
    )
    reliability = max(minimum, 1.0 - max(risks.values()))
    return min(1.0, reliability), risks


def _aligned_signal_rows(inputs: H3Inputs, path: Path) -> list[dict[str, str]]:
    rows = read_h3_signal_rows(path)
    by_id = {row["observation_id"]: row for row in rows}
    expected = [item.observation_id for item in inputs.observations]
    if set(by_id) != set(expected):
        raise RuntimeError("H3 signal rows do not exactly match selected H3 observations")
    aligned = [by_id[identifier] for identifier in expected]
    for item, row in zip(inputs.observations, aligned):
        expected_partition = inputs.partition_by_observation[item.observation_id]
        if row["partition"] != expected_partition:
            raise RuntimeError("H3 signal partition does not match the frozen split")
    return aligned


def fit_h3_thresholds(
    config: ExperimentConfig, model_names: Sequence[str]
) -> dict[str, Any]:
    h3 = require_h3(config)
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("H3 threshold fitting requires unique model names")
    inputs = validate_h3_inputs(config)
    signal_path = config.paths.output_root / "h3_observation_signals.csv"
    signal_metadata = config.paths.output_root / "h3_signal_metadata.json"
    if not signal_metadata.is_file():
        raise RuntimeError("Run h3-signals before fitting thresholds")
    signals = _aligned_signal_rows(inputs, signal_path)
    train_indices = [
        index
        for index, item in enumerate(inputs.observations)
        if inputs.partition_by_observation[item.observation_id] == "project_train"
    ]
    if not train_indices:
        raise RuntimeError("H3 threshold fitting found no project_train observations")
    by_identity: dict[str, list[int]] = defaultdict(list)
    for index in train_indices:
        by_identity[inputs.observations[index].identity].append(index)
    for indices in by_identity.values():
        indices.sort(
            key=lambda index: (
                inputs.observations[index].frame,
                inputs.observations[index].observation_id,
            )
        )

    common_values: dict[str, list[float]] = {
        key: [] for key in COMMON_CALIBRATORS
    }
    for indices in by_identity.values():
        for previous_index, current_index in zip(indices, indices[1:]):
            previous = signals[previous_index]
            current = signals[current_index]
            previous_area = max(_finite_float(previous["bbox_area"], "bbox_area"), 1e-12)
            current_area = max(_finite_float(current["bbox_area"], "bbox_area"), 1e-12)
            previous_sharpness = max(
                _finite_float(previous["bbox_laplacian_variance"], "sharpness"), 1e-12
            )
            current_sharpness = max(
                _finite_float(current["bbox_laplacian_variance"], "sharpness"), 1e-12
            )
            common_values["bbox_scale_change"].append(
                abs(math.log(current_area / previous_area))
            )
            common_values["orientation_change_proxy"].append(
                axial_difference(
                    _finite_float(previous["bbox_orientation_proxy_deg"], "orientation"),
                    _finite_float(current["bbox_orientation_proxy_deg"], "orientation"),
                )
            )
            common_values["sharpness_change"].append(
                abs(math.log(current_sharpness / previous_sharpness))
            )
            common_values["max_bbox_iou"].append(
                _finite_float(current["max_bbox_iou"], "max_bbox_iou")
            )
            common_values["neighbor_count_wide"].append(
                _finite_float(current["neighbor_count_wide"], "neighbor_count_wide")
            )

    common_calibrators = {
        name: {
            "sample_count": len(values),
            "knots": _quantile_knots(values, h3.quantile_knots, name),
            "q75": float(np.quantile(np.asarray(values, dtype=np.float64), 0.75)),
        }
        for name, values in common_values.items()
    }
    models: dict[str, Any] = {}
    cache_fingerprints: dict[str, str] = {}
    for model_name in model_names:
        embeddings, cache = load_h3_embeddings(config, model_name, inputs)
        history_outliers: list[float] = []
        for indices in by_identity.values():
            template: np.ndarray | None = None
            seen = 0
            for index in indices:
                if template is not None and seen >= h3.history_length:
                    history_outliers.append(
                        max(0.0, 1.0 - float(np.dot(template, embeddings[index])))
                    )
                if template is None:
                    template = _normalize(embeddings[index])
                else:
                    template = _normalize(
                        (1.0 - h3.memory_alpha) * template
                        + h3.memory_alpha * embeddings[index]
                    )
                seen += 1
        models[model_name] = {
            "identity_history_outlier": {
                "sample_count": len(history_outliers),
                "knots": _quantile_knots(
                    history_outliers,
                    h3.quantile_knots,
                    f"{model_name}.identity_history_outlier",
                ),
                "q75": float(np.quantile(history_outliers, 0.75)),
            },
            "cache_fingerprint": cache.fingerprint,
            "model_signature_sha256": sha256_text(
                canonical_json(cache.signature.get("model"))
            ),
        }
        cache_fingerprints[model_name] = cache.fingerprint

    signature = {
        "format_version": 1,
        "implementation": "beeid.h3.thresholds:v1",
        "threshold_fitting_partition": "project_train",
        "manifest_sha256": inputs.audit["manifest_sha256"],
        "protocol_sha256": inputs.audit["protocol_sha256"],
        "project_split_sha256": inputs.audit["project_split_sha256"],
        "signals_sha256": sha256_file(signal_path),
        "signal_metadata_sha256": sha256_file(signal_metadata),
        "cache_fingerprints": cache_fingerprints,
        "models": list(model_names),
        "history_length": h3.history_length,
        "memory_alpha": h3.memory_alpha,
        "quantile_knots": h3.quantile_knots,
        "min_reliability": h3.min_reliability,
        "allow_subset": h3.allow_subset,
        "project_train_videos": inputs.audit["project_train_videos"],
        "final_test_read": False,
    }
    artifact = {
        "status": "completed",
        "fingerprint": sha256_text(canonical_json(signature)),
        "signature": signature,
        "reliability": {
            "formula": "clip(1 - max(train_CDF(dynamic_risk_features)), min_reliability, 1)",
            "min_reliability": h3.min_reliability,
            "included_features": [
                "identity_history_outlier",
                "bbox_scale_change",
                "orientation_change_proxy",
                "sharpness_change",
                "crowding_overlap",
            ],
            "excluded_static_features": ["bbox_area", "bbox_laplacian_variance"],
        },
        "common_calibrators": common_calibrators,
        "models": models,
        "input_audit": inputs.audit,
        "final_test_read": False,
    }
    artifact["calibration_sha256"] = sha256_text(
        canonical_json(
            {
                "reliability": artifact["reliability"],
                "common_calibrators": common_calibrators,
                "models": models,
            }
        )
    )
    output = config.paths.output_root / "h3_thresholds.json"
    atomic_write_json(output, artifact)
    return artifact


def load_h3_thresholds(config: ExperimentConfig, model_names: Sequence[str]) -> dict[str, Any]:
    path = config.paths.output_root / "h3_thresholds.json"
    if not path.is_file():
        raise RuntimeError("H3 threshold artifact is missing; run h3-fit-thresholds")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or artifact.get("final_test_read") is not False:
        raise RuntimeError(f"Invalid H3 threshold artifact: {path}")
    observed_calibration_sha = sha256_text(
        canonical_json(
            {
                "reliability": artifact.get("reliability"),
                "common_calibrators": artifact.get("common_calibrators"),
                "models": artifact.get("models"),
            }
        )
    )
    if artifact.get("calibration_sha256") != observed_calibration_sha:
        raise RuntimeError("H3 threshold calibration checksum mismatch")
    inputs = validate_h3_inputs(config)
    signature = artifact.get("signature")
    expected = {
        "threshold_fitting_partition": "project_train",
        "manifest_sha256": inputs.audit["manifest_sha256"],
        "protocol_sha256": inputs.audit["protocol_sha256"],
        "project_split_sha256": inputs.audit["project_split_sha256"],
        "signals_sha256": sha256_file(config.paths.output_root / "h3_observation_signals.csv"),
        "history_length": require_h3(config).history_length,
        "memory_alpha": require_h3(config).memory_alpha,
        "quantile_knots": require_h3(config).quantile_knots,
        "min_reliability": require_h3(config).min_reliability,
        "final_test_read": False,
    }
    if not isinstance(signature, dict):
        raise RuntimeError("H3 threshold artifact has no signature")
    mismatches = [key for key, value in expected.items() if signature.get(key) != value]
    if mismatches:
        raise RuntimeError("H3 threshold artifact mismatch: " + ", ".join(mismatches))
    missing_models = [name for name in model_names if name not in artifact.get("models", {})]
    if missing_models:
        raise RuntimeError(
            "H3 threshold artifact is missing models: " + ", ".join(missing_models)
        )
    return artifact
