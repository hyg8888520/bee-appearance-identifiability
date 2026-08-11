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
    configured = replace(config, paths=replace(config.paths, h3_output_root=source_root))
    loaded, source_audit = load_h5_source_embeddings(configured, "resnet50", inputs)
    assert loaded.shape == (2, 3)
    assert np.allclose(loaded[0], values[0])
    assert np.allclose(loaded[1], values[2])
    assert source_audit["source_cache_observation_scope"].startswith("superset_allowed")


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
