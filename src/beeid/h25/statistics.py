"""Deterministic paired cluster bootstrap for H2.5 strategy contrasts."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from statistics import mean
from typing import Any, Sequence

import numpy as np


METRICS = ("rank1_gain", "similarity_gap_reduction", "induced_error_reduction")


def deterministic_seed(base_seed: int, *parts: object) -> int:
    payload = "\0".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _cluster_mean_bootstrap(
    rows: Sequence[dict[str, Any]], metric: str, cluster_key: str, replicates: int, seed: int
) -> tuple[float, float, float, int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(float(row[metric]))
    cluster_means = np.asarray(
        [mean(values) for _, values in sorted(grouped.items())], dtype=np.float64
    )
    if cluster_means.size == 0:
        raise RuntimeError("Cannot bootstrap an empty H2.5 comparison")
    rng = np.random.default_rng(seed)
    draws = rng.choice(cluster_means, size=(replicates, cluster_means.size), replace=True).mean(axis=1)
    return (
        float(cluster_means.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        int(cluster_means.size),
    )


def cluster_bootstrap(
    paired_rows: Sequence[dict[str, Any]], *, replicates: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        key = (
            str(row["model"]), str(row["event_type"]),
            str(row["strategy"]), str(row["window"]),
        )
        groups[key].append(row)
    result: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        for cluster_unit, cluster_key in (("video_id", "video_id"), ("identity", "identity")):
            for metric in METRICS:
                point, low, high, count = _cluster_mean_bootstrap(
                    rows,
                    metric,
                    cluster_key,
                    replicates,
                    deterministic_seed(seed, *key, cluster_unit, metric),
                )
                result.append(
                    {
                        "model": key[0],
                        "event_type": key[1],
                        "strategy": key[2],
                        "window": key[3],
                        "metric": metric,
                        "cluster_unit": cluster_unit,
                        "cluster_count": count,
                        "sample_count": len(rows),
                        "point_estimate": point,
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_replicates": replicates,
                        "bootstrap_seed": seed,
                    }
                )
    return result
