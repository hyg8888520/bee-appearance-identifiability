from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from beeid.data.mot import Observation
from beeid.h6.data import build_windows, collate_windows, dense_candidate_edges
from beeid.h6.model import GlobalTrajectoryReasoner, H6ModelSpec
from beeid.h6.model import (
    balanced_edge_loss, cycle_consistency_loss, supervised_contrastive_pair_loss,
)
from beeid.h6.oracle import audit_candidate_reachability
from beeid.h6.protocol import H6ProtocolError, validate_h6_protocol
from beeid.h6.synthetic import h6_synthetic_smoke
from beeid.h6.experiment import _calibrate_threshold
from beeid.h6.experiment import (
    _atomic_torch_save, _load_training_checkpoint, _memory_bounded_window_loss,
    _train_step_with_amp_recovery,
)
from beeid.h6.tracker import (
    assignment_rows, attach_offline_trajectory_utility, cluster_global_edges,
    trajectory_feature_rows,
)
from beeid.utils import canonical_json, sha256_text


ROOT = Path(__file__).parents[1]


def _obs(frame: int, track: int, x: float | None = None) -> Observation:
    x = float(x if x is not None else track * 50 + frame)
    identifier = f"validation:v:{frame:06d}:{track}"
    return Observation(
        identifier, "validation", "train", "v", track, f"v:{track}", frame, frame,
        f"{frame:06d}.jpg", 200, 100, 10.0, 10.0, x, 20.0, 10.0, 10.0,
        x, 20.0, x + 10, 30.0, int(x), 20, int(x) + 10, 30, 0.2, False,
        x + 5, 25.0, 100.0, 1.0, 1, 1.0, "", "",
    )


def _baseline(observations: list[Observation], ids: list[int]) -> list[dict[str, object]]:
    return [
        {
            "model": "resnet50", "variant": "baseline_association", "stage": "gt_detection_boxes",
            "video_id": item.video_id, "frame_id": item.frame, "observation_id": item.observation_id,
            "gt_identity": item.identity, "gt_track_id": item.track_id,
            "predicted_track_id": predicted, "matched_existing_track": item.frame > 1,
            "association_score": "", "appearance_similarity": "", "motion_score": "",
            "normalized_motion_distance": "", "reliability": "", "effective_memory_alpha": "",
            "memory_update_accepted": False, "eligible_memory_update": False,
            "risk_identity_history_outlier": "", "risk_bbox_scale_change": "",
            "risk_orientation_change_proxy": "", "risk_sharpness_change": "",
            "risk_crowding_overlap": "",
        }
        for item, predicted in zip(observations, ids)
    ]


def test_protocol_checksum_fails_before_yaml_semantics(tmp_path: Path) -> None:
    source = ROOT / "configs" / "h6_protocol.lock.yaml"
    target = tmp_path / "h6_protocol.lock.yaml"
    target.write_bytes(source.read_bytes() + b"\n")
    checksum = tmp_path / "h6_protocol.lock.sha256"
    checksum.write_text("0" * 64 + "  h6_protocol.lock.yaml\n", encoding="utf-8")
    with pytest.raises(H6ProtocolError, match="checksum mismatch"):
        validate_h6_protocol(target, checksum)


def test_locked_protocol_is_valid() -> None:
    result = validate_h6_protocol(
        ROOT / "configs" / "h6_protocol.lock.yaml",
        ROOT / "configs" / "h6_protocol.lock.sha256",
    )
    assert result["final_test_access"] is False
    assert result["source_h3_protocol_sha256"] == "747c4e6d30de698466140d299b4d1b37e5585b805c461e4e5eceb1291037bd19"


def test_dense_edges_have_no_same_frame_and_no_top_k() -> None:
    frames = torch.tensor([0, 0, 1, 1, 2])
    identities = ["a", "b", "a", "b", "a"]
    left, right, labels = dense_candidate_edges(frames, identities, 2)
    assert len(left) == 8
    assert torch.all(frames[right] > frames[left])
    assert labels.sum().item() == 4


def test_model_training_and_inference_paths_are_identical() -> None:
    torch.manual_seed(24)
    model = GlobalTrajectoryReasoner(H6ModelSpec(8, 16, 4, 2, 32, 0.0, 6)).eval()
    observations = [_obs(frame, track) for frame in range(1, 4) for track in (1, 2)]
    embeddings = np.random.default_rng(24).normal(size=(6, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    window = build_windows(observations, range(6), 6, 3)[0]
    batch = collate_windows([window], observations, embeddings, 6)
    with torch.no_grad():
        first = model.encode(batch.embeddings, batch.geometry, batch.frame_index, batch.token_mask)
        second = model.encode(batch.embeddings, batch.geometry, batch.frame_index, batch.token_mask)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_oracle_audit_excludes_long_gap_from_recall_denominator() -> None:
    observations = [_obs(1, 1), _obs(2, 1), _obs(10, 1)]
    rows, summary = audit_candidate_reachability(
        observations, range(3), window_length=5, window_stride=2, max_frame_gap=3
    )
    assert len(rows) == 2
    assert summary["eligible_edge_count"] == 1
    assert summary["edge_recall"] == 1.0


def test_overlap_component_is_atomic_and_fallback_exact() -> None:
    observations = [_obs(frame, track) for frame in range(1, 4) for track in (1, 2)]
    baseline_ids = [1, 2, 1, 2, 11, 12]
    baseline = _baseline(observations, baseline_ids)
    neural = [1, 2, 1, 2, 1, 2]
    probabilities = {(0, 2): 0.9, (2, 4): 0.9, (1, 3): 0.9, (3, 5): 0.9}
    lookup = {str(row["observation_id"]): row for row in baseline}
    trajectories = trajectory_feature_rows(observations, neural, probabilities, lookup)
    attach_offline_trajectory_utility(trajectories, observations, lookup, neural)
    assert len(trajectories) == 2
    assert all(row["offline_harm_pairs"] == 0 for row in trajectories)
    assert all(row["offline_gain_pairs"] > 0 for row in trajectories)
    fallback, decisions = assignment_rows(
        "resnet50", observations, neural, trajectories, [0.99, 0.99], None, baseline
    )
    selected = [row for row in fallback if row["variant"] == "calibrated_selective_global"]
    assert [str(row["predicted_track_id"]) for row in selected] == [str(value) for value in baseline_ids]
    assert all(row["h6_exact_baseline_fallback"] for row in selected)
    assert not any(row["accepted"] for row in decisions)


def test_method_features_are_invariant_to_gt_identity_renaming() -> None:
    observations = [_obs(frame, track) for frame in range(1, 4) for track in (1, 2)]
    renamed = [replace(item, identity=f"renamed:{3 - item.track_id}", track_id=100 + item.track_id) for item in observations]
    baseline = _baseline(observations, [1, 2, 1, 2, 11, 12])
    lookup = {str(row["observation_id"]): row for row in baseline}
    neural = [1, 2, 1, 2, 1, 2]
    probabilities = {(0, 2): 0.9, (2, 4): 0.9, (1, 3): 0.9, (3, 5): 0.9}
    original = trajectory_feature_rows(observations, neural, probabilities, lookup)
    permuted = trajectory_feature_rows(renamed, neural, probabilities, lookup)
    assert [row["features"] for row in original] == [row["features"] for row in permuted]


def test_cluster_never_places_two_observations_from_one_frame_in_track() -> None:
    observations = [_obs(1, 1), _obs(1, 2), _obs(2, 1), _obs(2, 2)]
    ids, audit = cluster_global_edges(
        observations, {(0, 2): 0.9, (1, 2): 0.8, (1, 3): 0.7}, 0.5
    )
    tracks: dict[int, list[int]] = {}
    for index, identity in enumerate(ids):
        tracks.setdefault(identity, []).append(observations[index].frame)
    assert all(len(values) == len(set(values)) for values in tracks.values())
    assert any(row["rejection_reason"] == "same_component_or_frame_conflict" for row in audit)


def test_synthetic_smoke(tmp_path: Path) -> None:
    result = h6_synthetic_smoke(tmp_path / "h6")
    assert result["status"] == "passed"
    assert result["exact_baseline_fallback_verified"] is True
    assert result["final_test_read"] is False
    assert result["real_experiment_result"] is False


def test_selector_calibration_is_fail_closed_without_safe_threshold() -> None:
    class H6:
        min_calibration_interventions = 2
        min_calibration_videos = 2
        min_calibration_precision = 0.9
        max_calibration_harm = 0.05

    class Config:
        h6 = H6()

    rows = [
        {"partition_disagreement_fraction": 1.0, "offline_positive": False, "video_id": "v1"},
        {"partition_disagreement_fraction": 1.0, "offline_positive": True, "video_id": "v2"},
    ]
    result = _calibrate_threshold(Config(), rows, [0.99, 0.5])  # type: ignore[arg-type]
    assert result["gate"] == "STOP_NO_SAFE_SELECTOR_THRESHOLD"
    assert result["threshold"] is None


def test_checkpoint_signature_is_strict_and_loads_exact_state(tmp_path: Path) -> None:
    model = GlobalTrajectoryReasoner(H6ModelSpec(8, 16, 4, 1, 32, 0.0, 6))
    signature = {"protocol": "locked", "cache": "immutable", "final_test_read": False}
    path = tmp_path / "partial.pt"
    _atomic_torch_save(path, {
        "signature": signature,
        "signature_sha256": sha256_text(canonical_json(signature)),
        "association_model": model.state_dict(), "optimizer": None,
        "loss_history": [], "final_test_read": False,
    })
    loaded = _load_training_checkpoint(path, signature, model)
    assert loaded is not None
    with pytest.raises(RuntimeError, match="incompatible H6 checkpoint"):
        _load_training_checkpoint(path, {**signature, "cache": "changed"}, model)


class _SimulatedGradScaler:
    """Small CPU test double for CUDA GradScaler's interface."""

    def __init__(self, scale: float = 4.0) -> None:
        self.scale_value = scale

    def get_scale(self) -> float:
        return self.scale_value

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self, new_scale: float | None = None) -> None:
        if new_scale is not None:
            self.scale_value = float(new_scale)


def test_amp_overflow_retries_same_batch_then_succeeds() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    parameter = model.weight
    parameter.data.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _SimulatedGradScaler(4.0)
    random_draws: list[float] = []

    def loss(amp_enabled: bool):  # type: ignore[no-untyped-def]
        for _ in range(2):
            random_draws.append(float(torch.rand(())))
            if amp_enabled and scaler.get_scale() > 2.0:
                yield parameter.sum() * torch.tensor(float("inf"))
            else:
                yield parameter.square().sum()

    result = _train_step_with_amp_recovery(
        model, optimizer, scaler, loss, use_amp=True,
        gradient_clip_norm=1.0, device=torch.device("cpu"), loss_denominator=2,
        max_amp_retries=4,
    )
    assert result is not None
    assert result["amp_overflow_retries"] == 1
    assert result["fp32_fallback"] is False
    assert random_draws[:2] == random_draws[2:]
    assert parameter.item() < 1.0


def test_amp_exhaustion_recomputes_same_batch_in_fp32_and_remains_strict() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    parameter = model.weight
    parameter.data.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _SimulatedGradScaler(4.0)
    modes: list[bool] = []

    def recoverable_loss(amp_enabled: bool):  # type: ignore[no-untyped-def]
        modes.append(amp_enabled)
        if amp_enabled:
            yield parameter.sum() * torch.tensor(float("inf"))
        else:
            yield parameter.square().sum()

    result = _train_step_with_amp_recovery(
        model, optimizer, scaler, recoverable_loss, use_amp=True,
        gradient_clip_norm=1.0, device=torch.device("cpu"), max_amp_retries=2,
    )
    assert result is not None
    assert result["amp_overflow_retries"] == 2
    assert result["fp32_fallback"] is True
    assert modes == [True, True, False]

    def irrecoverable_loss(amp_enabled: bool):  # type: ignore[no-untyped-def]
        del amp_enabled
        yield parameter.sum() * torch.tensor(float("inf"))

    with pytest.raises(RuntimeError, match="GPU FP32 fallback"):
        _train_step_with_amp_recovery(
            model, optimizer, scaler, irrecoverable_loss, use_amp=True,
            gradient_clip_norm=1.0, device=torch.device("cpu"), max_amp_retries=1,
        )


def test_memory_bounded_window_loss_matches_original_loss_and_gradients() -> None:
    torch.manual_seed(24)
    original = GlobalTrajectoryReasoner(H6ModelSpec(8, 16, 4, 1, 32, 0.0, 6))
    bounded = GlobalTrajectoryReasoner(original.spec)
    bounded.load_state_dict(original.state_dict(), strict=True)
    observations = [_obs(frame, track) for frame in range(1, 5) for track in (1, 2)]
    embeddings = np.random.default_rng(24).normal(size=(8, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    window = build_windows(observations, range(8), 6, 3)[0]
    batch = collate_windows([window], observations, embeddings, 6)
    frames = batch.frame_index[0]
    left, right, labels = dense_candidate_edges(frames, batch.identities[0], 3)
    triples = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.long)

    encoded = original.encode(
        batch.embeddings, batch.geometry, batch.frame_index, batch.token_mask
    )[0]
    original_loss = (
        balanced_edge_loss(
            original.pair_logits(encoded, batch.geometry[0], frames, left, right), labels
        )
        + 0.25 * supervised_contrastive_pair_loss(encoded, left, right, labels)
        + 0.1 * cycle_consistency_loss(
            original, encoded, batch.geometry[0], frames, triples
        )
    )
    original_loss.backward()

    bounded_encoded = bounded.encode(
        batch.embeddings, batch.geometry, batch.frame_index, batch.token_mask
    )[0]
    bounded_loss = _memory_bounded_window_loss(
        bounded, bounded_encoded, batch.geometry[0], frames, left, right, labels, triples,
        association_weight=1.0, contrastive_weight=0.25, cycle_weight=0.1,
        chunk_size=3,
    )
    bounded_loss.backward()

    torch.testing.assert_close(bounded_loss, original_loss, rtol=1e-5, atol=1e-6)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        original.named_parameters(), bounded.named_parameters()
    ):
        assert name_a == name_b
        assert parameter_a.grad is not None and parameter_b.grad is not None
        torch.testing.assert_close(parameter_b.grad, parameter_a.grad, rtol=2e-4, atol=2e-6)
