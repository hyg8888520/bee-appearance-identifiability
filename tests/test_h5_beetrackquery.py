from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from beeid.config import load_config
from beeid.cache import FeatureCache
from beeid.data.mot import Observation
from beeid.h5 import H5_PRIMARY_VARIANT, H5_VARIANTS
from beeid.h5.model import BeeTrackQuery, supervised_transition
from beeid.h5.core import H5Inputs, load_h5_source_embeddings
from beeid.h5.experiment import (
    TrainingTransition,
    _batched_transition_loss,
    _fit_model,
    _transition_batches,
)
from beeid.h3.core import H3Inputs
from beeid.h5.protocol import H5ProtocolError, validate_h5_protocol
from beeid.h5.synthetic import h5_synthetic_smoke
from beeid.h5.tracker import track_sequence


ROOT = Path(__file__).resolve().parents[1]


def _observation(frame: int, identity: int, x: float) -> Observation:
    return Observation(
        observation_id=f"validation:v:{frame:06d}:{identity}", split="validation",
        source_split="train", video_id="v", track_id=identity, identity=f"v:{identity}",
        frame=frame, frame_id=frame, image_path=f"train/v/img1/{frame:06d}.jpg",
        image_width=100, image_height=80, original_width=20.0, original_height=20.0,
        raw_x=x+1, raw_y=21, raw_w=20, raw_h=20, bbox_x1=x, bbox_y1=20,
        bbox_x2=x+20, bbox_y2=40, x1=int(x), y1=20, x2=int(x+20), y2=40,
        crop_expansion=0.2, crop_clipped=False, center_x=x+10, center_y=30,
        bbox_area=400, confidence=1, object_class=1, visibility=1,
        skip_reason="", extra_columns="[]",
    )


def test_h5_protocol_config_and_checksum_are_frozen(tmp_path):
    result = validate_h5_protocol(
        ROOT / "configs" / "h5_protocol.lock.yaml",
        ROOT / "configs" / "h5_protocol.lock.sha256",
    )
    assert result["primary_variant"] == H5_PRIMARY_VARIANT
    assert tuple(result["variants"]) == H5_VARIANTS
    assert result["fit_partition"] == "project_train"
    assert result["final_test_access"] is False
    for name, subset in (("h5.example.yaml", False), ("h5.local.yaml.example", False), ("h5_smoke.example.yaml", True)):
        config = load_config(ROOT / "configs" / name)
        assert config.h5 is not None and config.h3 is not None
        assert config.h5.allow_subset is subset
        assert config.paths.h3_output_root != config.paths.output_root
    bad = tmp_path / "bad.sha256"
    bad.write_text("0" * 64 + "  h5_protocol.lock.yaml\n", encoding="utf-8")
    with pytest.raises(H5ProtocolError, match="checksum mismatch"):
        validate_h5_protocol(ROOT / "configs" / "h5_protocol.lock.yaml", bad)
    legacy = tmp_path / "legacy-local.yaml"
    legacy.write_text(
        (ROOT / "configs" / "h5_smoke.example.yaml").read_text(encoding="utf-8")
        .replace(", max_pair_elements_per_batch: 262144", "")
        .replace(", checkpoint_interval_batches: 25", ""),
        encoding="utf-8",
    )
    legacy_config = load_config(legacy)
    assert legacy_config.h5 is not None
    assert legacy_config.h5.max_pair_elements_per_batch == 262144
    assert legacy_config.h5.checkpoint_interval_batches == 25


def test_beetrackquery_contract_and_supervised_reliability_loss():
    model = BeeTrackQuery(8, 16, 4, 0.0)
    embeddings = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    geometry = torch.rand(3, 4)
    tokens = model.encode_detections(embeddings, geometry)
    assert tokens.shape == (3, 16)
    refreshed = model.refresh(tokens[:2], tokens)
    delta = torch.abs(geometry[:2, None, :] - geometry[None, :, :])
    logits = model.pair_logits(refreshed, tokens, delta)
    assert logits.shape == (2, 3)
    reliability = model.reliability(logits)
    assert reliability.shape == (2,)
    assert torch.all((reliability >= 0) & (reliability <= 1))
    result = supervised_transition(model, tokens[:2], ["v:1", "v:2"], tokens, ["v:1", "v:2", "v:3"], delta)
    assert result is not None
    (result.association_loss + result.reliability_loss).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_reliability_loss_is_amp_safe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    model = BeeTrackQuery(8, 16, 4, 0.0).to(device)
    embeddings = torch.nn.functional.normalize(torch.randn(3, 8, device=device), dim=1)
    geometry = torch.rand(3, 4, device=device)
    with torch.autocast(device_type=device.type, dtype=dtype):
        tokens = model.encode_detections(embeddings, geometry)
        delta = torch.abs(geometry[:2, None, :] - geometry[None, :, :])
        result = supervised_transition(
            model, tokens[:2], ["v:1", "v:2"], tokens,
            ["v:1", "v:2", "v:3"], delta,
        )
        assert result is not None
        loss = result.association_loss + result.reliability_loss
    loss.backward()


def test_batched_transition_matches_single_transition_objective():
    torch.manual_seed(24)
    model = BeeTrackQuery(8, 16, 4, 0.0)
    model.eval()
    embeddings = np.random.default_rng(24).normal(size=(5, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    geometry = np.random.default_rng(25).random((5, 4), dtype=np.float32)
    transition = TrainingTransition((0, 1), (2, 3, 4), (0, 1))
    actual, pair_count, prediction_count = _batched_transition_loss(
        model, [transition], [0], embeddings, geometry, torch.device("cpu"), 0.25
    )
    previous_tokens = model.encode_detections(
        torch.from_numpy(embeddings[[0, 1]]), torch.from_numpy(geometry[[0, 1]])
    )
    current_tokens = model.encode_detections(
        torch.from_numpy(embeddings[[2, 3, 4]]), torch.from_numpy(geometry[[2, 3, 4]])
    )
    delta = torch.abs(
        torch.from_numpy(geometry[[0, 1]])[:, None, :]
        - torch.from_numpy(geometry[[2, 3, 4]])[None, :, :]
    )
    expected = supervised_transition(
        model,
        previous_tokens,
        ["v:1", "v:2"],
        current_tokens,
        ["v:1", "v:2", "v:3"],
        delta,
    )
    assert expected is not None
    expected_loss = expected.association_loss + 0.25 * expected.reliability_loss
    assert torch.allclose(actual, expected_loss, atol=1e-6, rtol=1e-6)
    assert pair_count == 6
    assert prediction_count == 2


def test_transition_batcher_respects_batch_and_pair_budgets():
    transitions = [
        TrainingTransition(tuple(range(queries)), tuple(range(detections)), tuple(0 for _ in range(queries)))
        for queries, detections in ((2, 3), (3, 4), (1, 2), (4, 4), (2, 5))
    ]
    batches = _transition_batches(transitions, range(len(transitions)), 3, 32)
    assert [index for batch in batches for index in batch] == list(range(len(transitions)))
    for batch in batches:
        assert len(batch) <= 3
        assert len(batch) * max(transitions[index].query_count for index in batch) * max(
            transitions[index].detection_count for index in batch
        ) <= 32


def test_batched_trainer_writes_live_and_resumable_checkpoints(tmp_path):
    observations = tuple(
        _observation(frame, identity, 10 if identity == 1 else 55)
        for frame in range(1, 7)
        for identity in (1, 2)
    )
    audit = {
        "manifest_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "project_split_sha256": "c" * 64,
        "h5_protocol_sha256": "d" * 64,
    }
    h3_inputs = H3Inputs(
        observations,
        {item.observation_id: "project_train" for item in observations},
        {"project_train": ("v",), "development_validation": ("d",), "final_test": ("t",)},
        audit,
    )
    inputs = H5Inputs(h3_inputs, audit)
    config = load_config(ROOT / "configs" / "h5_smoke.example.yaml")
    assert config.h5 is not None
    configured = replace(
        config,
        paths=replace(config.paths, output_root=tmp_path / "h5-output"),
        runtime=replace(config.runtime, device="cpu", amp=False, batch_size=2),
        h5=replace(
            config.h5,
            epochs=2,
            clip_length=3,
            train_clip_stride=1,
            max_train_clips_per_video=4,
            max_pair_elements_per_batch=32,
            checkpoint_interval_batches=1,
        ),
    )
    embeddings = np.random.default_rng(24).normal(size=(len(observations), 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    _, metadata = _fit_model(
        configured,
        inputs,
        embeddings,
        "test_model",
        {"cache_fingerprint": "cache-test"},
    )
    assert metadata["status"] == "completed"
    assert metadata["training_transition_count"] > 0
    assert metadata["optimizer_step_count"] > 0
    progress = json.loads(
        (configured.paths.output_root / "h5_logs" / "test_model_progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert progress["status"] == "completed"
    assert (configured.paths.output_root / "h5_checkpoints" / "test_model" / "training-progress.pt").is_file()
    _, resumed = _fit_model(
        configured,
        inputs,
        embeddings,
        "test_model",
        {"cache_fingerprint": "cache-test"},
    )
    assert resumed["status"] == "resumed"


def test_tracker_is_causal_and_gt_identity_permutation_does_not_change_predictions():
    observations = [_observation(frame, identity, 10 if identity == 1 else 55) for frame in range(1, 5) for identity in (1, 2)]
    embeddings = np.stack([np.asarray([1, 0, item.center_x / 100, 0], dtype=np.float32) for item in observations])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    config = load_config(ROOT / "configs" / "h5_smoke.example.yaml")
    assert config.h5 is not None
    model = BeeTrackQuery(4, 128, 4, 0.1)
    first = track_sequence("m", H5_PRIMARY_VARIANT, model, observations, embeddings, config.h5, torch.device("cpu"))
    permuted = [Observation(**{**item.to_dict(), "track_id": 99 - item.track_id, "identity": f"v:{99-item.track_id}"}) for item in observations]
    second = track_sequence("m", H5_PRIMARY_VARIANT, model, permuted, embeddings, config.h5, torch.device("cpu"))
    assert [row["predicted_track_id"] for row in first] == [row["predicted_track_id"] for row in second]
    assert all(row["frame_id"] == observations[index].frame for index, row in enumerate(first))


def test_h5_can_select_a_subset_from_a_strictly_valid_h3_superset_cache(tmp_path):
    selected_observations = (_observation(1, 1, 10), _observation(2, 1, 11))
    audit = {
        "manifest_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "project_split_sha256": "c" * 64,
    }
    h3_inputs = H3Inputs(
        selected_observations,
        {item.observation_id: "project_train" for item in selected_observations},
        {"project_train": ("v",), "development_validation": ("d",), "final_test": ("t",)},
        audit,
    )
    inputs = H5Inputs(h3_inputs, audit)
    config = load_config(ROOT / "configs" / "h5_smoke.example.yaml")
    source_root = tmp_path / "h3-output"
    cache = FeatureCache(
        tmp_path / "cache",
        "h3__resnet50",
        {
            "experiment": "H3_RAM_Bee", "implementation": "beeid.h3.features:v1",
            **audit, "crop_expansion": 0.2, "input_size": 224,
            "final_test_read": False, "model": {"name": "resnet50"},
        },
    )
    cache.initialize()
    extra_id = "validation:v:000003:1"
    values = np.eye(3, dtype=np.float32)
    cache.write_shard(
        0,
        [selected_observations[0].observation_id, extra_id, selected_observations[1].observation_id],
        values,
    )
    source_root.mkdir()
    (source_root / "h3_cache_locations.json").write_text(
        json.dumps({"resnet50": str(cache.directory)}), encoding="utf-8"
    )
    configured = replace(
        config,
        paths=replace(config.paths, h3_output_root=source_root, output_root=tmp_path / "h5-output"),
    )
    loaded, source_audit = load_h5_source_embeddings(configured, "resnet50", inputs)
    assert loaded.shape == (2, 3)
    assert np.allclose(loaded[0], values[0])
    assert np.allclose(loaded[1], values[2])
    assert source_audit["source_cache_observation_scope"].startswith("superset_allowed")
    assert source_audit["aligned_cache_status"] == "built"
    reused, reused_audit = load_h5_source_embeddings(configured, "resnet50", inputs)
    assert reused_audit["aligned_cache_status"] == "reused"
    assert isinstance(reused, np.memmap)
    assert np.allclose(reused, loaded)


def test_h5_synthetic_smoke_is_resumable_and_not_a_real_result(tmp_path):
    output = tmp_path / "h5-synthetic"
    result = h5_synthetic_smoke(output)
    assert result["status"] == "passed"
    assert result["metadata_status"] == "SERVER_VALIDATION_PENDING"
    assert result["checkpoint_resume_verified"] is True
    assert result["final_test_read"] is False
    metadata = json.loads((output / "h5_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["test_only_encoder"] is True
    assert metadata["real_experiment_result"] is False
