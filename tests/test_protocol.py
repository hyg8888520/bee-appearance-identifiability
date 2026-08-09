from __future__ import annotations

from dataclasses import replace

import numpy as np

from beeid.data.mot import Observation
from beeid.protocol.retrieval import evaluate_embeddings, size_quartiles


def observation(video: str, frame: int, track: int, x: float, area: float = 100.0) -> Observation:
    return Observation(
        observation_id=f"validation:{video}:{frame:06d}:{track}", split="validation", source_split="train",
        video_id=video, track_id=track, identity=f"{video}:{track}", frame=frame, frame_id=frame,
        image_path=f"train/{video}/img1/{frame:06d}.jpg", image_width=100, image_height=80,
        original_width=10, original_height=area / 10,
        raw_x=x + 1, raw_y=11, raw_w=10, raw_h=area / 10,
        bbox_x1=x, bbox_y1=10, bbox_x2=x + 10, bbox_y2=10 + area / 10,
        x1=int(x), y1=10, x2=int(x + 10), y2=int(10 + area / 10),
        crop_expansion=0.2, crop_clipped=False, center_x=x + 5, center_y=10 + area / 20,
        bbox_area=area, confidence=1.0, object_class=1, visibility=1.0,
        skip_reason="", extra_columns="[]",
    )


def normalized(rows):
    values = np.asarray(rows, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_delta_positive_full_and_hard_gallery_constraints():
    values = [
        observation("v", 1, 1, 10),
        observation("v", 2, 1, 11),
        observation("v", 2, 2, 14),
        observation("v", 2, 3, 70),
    ]
    embeddings = normalized([[1, 0], [1, 0], [0.8, 0.2], [0, 1]])
    rows, _ = evaluate_embeddings("m", values, embeddings, [1], hard_negative_k=1)
    row = next(item for item in rows if item["query_observation_id"] == values[0].observation_id)
    assert row["positive_observation_id"] == values[1].observation_id
    assert row["rank1"] is True and row["hard_rank1"] is True
    assert row["hard_negative_count"] == 1
    assert row["margin"] > 0


def test_cosine_tie_is_rank1_failure():
    values = [observation("v", 1, 1, 10), observation("v", 2, 1, 10), observation("v", 2, 2, 20)]
    rows, _ = evaluate_embeddings("m", values, normalized([[1, 0], [1, 0], [1, 0]]), [1], 5)
    row = rows[0]
    assert row["rank1"] is False and row["hard_rank1"] is False
    assert row["predicted_observation_id"] == "TIE"
    assert row["margin"] == 0


def test_missing_positive_and_missing_negative_are_excluded_with_reasons():
    no_positive_values = [observation("v", 1, 1, 10), observation("v", 2, 2, 20)]
    rows, _ = evaluate_embeddings("m", no_positive_values, normalized([[1, 0], [0, 1]]), [1], 5)
    assert rows[0]["rank1"] == "" and rows[0]["skip_reason"] == "no_positive_at_delta"

    no_negative_values = [observation("v", 1, 1, 10), observation("v", 2, 1, 11)]
    rows, _ = evaluate_embeddings("m", no_negative_values, normalized([[1, 0], [1, 0]]), [1], 5)
    assert rows[0]["rank1"] is True
    assert rows[0]["hard_rank1"] == "" and rows[0]["skip_reason"] == "no_hard_negative"


def test_size_quartiles_use_unique_query_areas():
    values = [observation("v", index, index, 1, float(index * 100)) for index in range(1, 9)]
    thresholds, labels = size_quartiles(values)
    assert thresholds == (275.0, 450.0, 625.0)
    assert labels[values[0].observation_id] == "Q1"
    assert labels[values[-1].observation_id] == "Q4"
