from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from beeid.h2.diagnostics import history_consistency
from beeid.h2.statistics import (
    clustered_bootstrap,
    factor_analysis,
    paired_clustered_bootstrap,
    quantile_bins,
    roc_auc,
    spearman,
)
def test_tie_aware_predictiveness_and_quantile_binning():
    edges, labels = quantile_bins([1.0, 2.0, 3.0, 4.0], 4)
    assert len(edges) == 3 and labels == ["Q1", "Q2", "Q3", "Q4"]
    assert roc_auc([0.1, 0.2, 0.9, 0.8], [False, False, True, True]) == 1.0
    assert roc_auc([1.0, 1.0], [False, True]) == 0.5
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0


def test_history_consistency_uses_only_previous_same_identity():
    observations = [
        SimpleNamespace(
            observation_id=f"validation:v1:{frame:06d}:1",
            identity="v1:1",
            frame=frame,
        )
        for frame in range(1, 4)
    ]
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    history = history_consistency(observations, embeddings, history_length=2)
    assert history[observations[0].observation_id] == (0, "", "")
    assert history[observations[1].observation_id][1] == 1.0
    assert history[observations[2].observation_id][1] == 0.0


def _diagnostic_rows():
    rows = []
    for video, identity, area, outcomes in (
        ("v1", "v1:1", 1.0, (True, False)),
        ("v2", "v2:1", 4.0, (True, True)),
    ):
        for index, outcome in enumerate(outcomes):
            rows.append(
                {
                    "model": "m",
                    "variant": "primary",
                    "delta": 1,
                    "video_id": video,
                    "identity": identity,
                    "query_observation_id": f"{identity}:{index}",
                    "rank1": str(outcome),
                    "margin": "0.1" if outcome else "-0.1",
                    "positive_similarity": "0.8",
                    "query_bbox_area": str(area),
                    "query_variant_clipped": "False",
                }
            )
    return rows


def test_factor_thresholds_are_outcome_blind_and_bootstrap_is_deterministic():
    rows = _diagnostic_rows()
    _, _, definitions = factor_analysis(rows, 4)
    flipped = [{**row, "rank1": "False" if row["rank1"] == "True" else "True"} for row in rows]
    _, _, flipped_definitions = factor_analysis(flipped, 4)
    assert definitions["thresholds"] == flipped_definitions["thresholds"]

    first = clustered_bootstrap(rows, replicates=50, seed=24)
    second = clustered_bootstrap(rows, replicates=50, seed=24)
    assert first == second
    assert {row["cluster_unit"] for row in first} == {"video", "identity"}

    comparison = [
        *rows,
        *[
            {
                **row,
                "variant": "alternative",
                "rank1": "True",
            }
            for row in rows
        ],
    ]
    paired = paired_clustered_bootstrap(
        comparison, "primary", replicates=50, seed=24
    )
    assert {row["cluster_unit"] for row in paired} == {"video", "identity"}
    assert all(row["point_rank1_difference"] >= 0 for row in paired)
