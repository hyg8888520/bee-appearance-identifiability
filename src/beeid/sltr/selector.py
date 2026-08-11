"""Deterministic, leakage-checked NumPy selector fit and OOF thresholding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


FORBIDDEN_FEATURE_TOKENS = ("gt", "identity", "label", "utility", "correct")


class SelectorError(RuntimeError):
    """Raised for feature leakage, invalid OOF rows, or a failed selector gate."""


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    l2: float
    iterations: int
    learning_rate: float

    def probability(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise SelectorError("Selector feature dimension differs from fitted model")
        if not np.isfinite(values).all():
            raise SelectorError("Selector received non-finite observable features")
        logits = ((values - self.mean) / self.scale) @ self.weights + self.intercept
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "weights": self.weights.tolist(), "intercept": self.intercept,
            "l2": self.l2, "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "implementation": "deterministic_numpy_l2_logistic_v1",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LogisticModel":
        names = tuple(str(item) for item in value["feature_names"])
        model = cls(
            names,
            np.asarray(value["mean"], dtype=np.float64),
            np.asarray(value["scale"], dtype=np.float64),
            np.asarray(value["weights"], dtype=np.float64),
            float(value["intercept"]), float(value["l2"]),
            int(value["iterations"]), float(value["learning_rate"]),
        )
        validate_feature_names(model.feature_names)
        if model.mean.shape != model.scale.shape or model.mean.shape != model.weights.shape:
            raise SelectorError("Stored selector model has incompatible vector shapes")
        if not np.isfinite(model.mean).all() or not np.isfinite(model.scale).all() or not np.isfinite(model.weights).all():
            raise SelectorError("Stored selector model contains non-finite values")
        if np.any(model.scale <= 0) or not np.isfinite(model.intercept):
            raise SelectorError("Stored selector model has invalid normalization")
        return model


def validate_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise SelectorError("Selector feature names must be non-empty and unique")
    leaking = [
        name for name in names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if leaking:
        raise SelectorError(
            "SLTR selector feature leakage is forbidden: " + ", ".join(sorted(leaking))
        )
    return names


def feature_matrix(rows: Sequence[dict[str, Any]], feature_names: Sequence[str] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    if not rows:
        raise SelectorError("Selector requires at least one labeled event")
    names = validate_feature_names(feature_names or tuple(sorted(rows[0].get("features", {}))))
    values = np.empty((len(rows), len(names)), dtype=np.float64)
    for index, row in enumerate(rows):
        features = row.get("features")
        if not isinstance(features, dict) or set(features) != set(names):
            raise SelectorError("Every SLTR event must contain the same observable feature schema")
        for column, name in enumerate(names):
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise SelectorError(f"SLTR event feature {name} is not finite numeric")
            values[index, column] = float(value)
    return values, names


def fit_l2_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    *,
    l2: float,
    iterations: int,
    learning_rate: float,
) -> LogisticModel:
    """Fit a stable full-batch L2 logistic model without random initialization."""
    names = validate_feature_names(feature_names)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape != (len(y), len(names)) or not np.isfinite(x).all():
        raise SelectorError("Invalid finite feature matrix for logistic fit")
    if len(y) < 2 or not np.isin(y, (0.0, 1.0)).all() or y.min() == y.max():
        raise SelectorError("Logistic fit requires both binary classes")
    if l2 < 0 or iterations <= 0 or learning_rate <= 0:
        raise SelectorError("Invalid deterministic logistic hyperparameters")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1], dtype=np.float64)
    prevalence = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    intercept = float(np.log(prevalence / (1.0 - prevalence)))
    for _ in range(iterations):
        logits = np.clip(z @ weights + intercept, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        residual = probability - y
        weights -= learning_rate * ((z.T @ residual) / len(y) + l2 * weights)
        intercept -= learning_rate * float(residual.mean())
    if not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise SelectorError("Deterministic logistic optimization became non-finite")
    return LogisticModel(tuple(names), mean, scale, weights, intercept, l2, iterations, learning_rate)


def group_oof_probabilities(
    rows: Sequence[dict[str, Any]],
    *,
    l2: float,
    iterations: int,
    learning_rate: float,
) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, Any]]]:
    """Leave-one-video-out predictions; each held-out video is never fit on itself."""
    x, names = feature_matrix(rows)
    labels = np.asarray([int(bool(row["positive"])) for row in rows], dtype=np.float64)
    videos = np.asarray([str(row["video_id"]) for row in rows])
    unique_videos = sorted(set(videos.tolist()))
    if len(unique_videos) < 2:
        raise SelectorError("Group OOF needs at least two training videos")
    result = np.full(len(rows), np.nan, dtype=np.float64)
    fit_rows: list[dict[str, Any]] = []
    for video in unique_videos:
        held_out = videos == video
        fit = ~held_out
        if len(set(labels[fit].tolist())) < 2:
            # OOF must not quietly leak the held-out group to solve a one-class fold.
            raise SelectorError(f"OOF fit excluding video {video} has one class")
        model = fit_l2_logistic(x[fit], labels[fit], names, l2=l2, iterations=iterations, learning_rate=learning_rate)
        result[held_out] = model.probability(x[held_out])
        fit_rows.append({
            "held_out_video_id": video, "fit_event_count": int(fit.sum()),
            "held_out_event_count": int(held_out.sum()),
            "fit_video_count": len(unique_videos) - 1,
            "held_out_in_fit": False,
        })
    if not np.isfinite(result).all():
        raise SelectorError("Group OOF did not produce finite predictions for every event")
    return result, names, fit_rows


def choose_oof_threshold(
    probabilities: Sequence[float],
    utilities: Sequence[float],
    videos: Sequence[str],
    *,
    min_precision: float = 0.8,
    max_harm: float = 0.05,
    min_selected_events: int = 10,
    min_selected_videos: int = 3,
    max_intervention_fraction: float = 0.30,
) -> dict[str, Any]:
    """Choose only from OOF scores, preferring coverage then utility then threshold."""
    score = np.asarray(probabilities, dtype=np.float64)
    utility = np.asarray(utilities, dtype=np.float64)
    group = np.asarray([str(item) for item in videos])
    if not (len(score) == len(utility) == len(group)) or not len(score):
        raise SelectorError("Threshold selection requires aligned non-empty OOF rows")
    if not np.isfinite(score).all() or not np.isfinite(utility).all():
        raise SelectorError("Threshold selection requires finite OOF scores and utilities")
    candidates: list[dict[str, Any]] = []
    for threshold in sorted(set(score.tolist())):
        selected = score >= threshold
        count = int(selected.sum())
        precision = float((utility[selected] > 0).mean()) if count else 0.0
        harm = float((utility[selected] < 0).mean()) if count else 1.0
        selected_video_count = len(set(group[selected].tolist()))
        fraction = count / len(score)
        row = {
            "threshold": float(threshold), "selected_event_count": count,
            "selected_video_count": selected_video_count,
            "intervention_fraction": fraction, "precision": precision,
            "harm": harm, "utility_sum": float(utility[selected].sum()),
        }
        row["eligible"] = bool(
            precision >= min_precision and harm <= max_harm
            and count >= min_selected_events and selected_video_count >= min_selected_videos
            and fraction <= max_intervention_fraction + 1e-12
        )
        candidates.append(row)
    feasible = [row for row in candidates if row["eligible"]]
    if not feasible:
        return {
            "status": "STOP_NO_SAFE_OOF_THRESHOLD", "threshold": None,
            "eligible": False, "candidates": candidates,
            "selection_rule": "oof_only_precision_harm_coverage_utility_higher_threshold_tie",
        }
    # Higher threshold is the final deterministic tie break.
    best = max(feasible, key=lambda row: (row["selected_event_count"], row["utility_sum"], row["threshold"]))
    return {
        "status": "GO_SAFE_OOF_THRESHOLD", "threshold": best["threshold"],
        "eligible": True, "selected_event_count": best["selected_event_count"],
        "selected_video_count": best["selected_video_count"],
        "intervention_fraction": best["intervention_fraction"], "precision": best["precision"],
        "harm": best["harm"], "utility_sum": best["utility_sum"],
        "candidates": candidates,
        "selection_rule": "oof_only_precision_harm_coverage_utility_higher_threshold_tie",
    }
