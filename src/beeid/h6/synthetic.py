"""CPU scientific smoke for H6, explicitly not an experiment result."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import torch

from ..data.mot import Observation
from ..utils import atomic_write_json
from .data import build_windows, collate_windows, dense_candidate_edges, identity_triples
from .model import GlobalTrajectoryReasoner, H6ModelSpec, balanced_edge_loss, cycle_consistency_loss
from .oracle import audit_candidate_reachability
from .tracker import assignment_rows, attach_offline_trajectory_utility, cluster_global_edges, trajectory_feature_rows


def _observation(frame: int, track: int, x: float) -> Observation:
    identifier = f"synthetic:video:{frame:06d}:{track}"
    return Observation(
        identifier, "validation", "train", "synthetic-video", track,
        f"synthetic-video:{track}", frame, frame, f"{frame:06d}.jpg", 640, 480,
        20.0, 20.0, x, 100.0 + track * 25, 20.0, 20.0,
        x, 100.0 + track * 25, x + 20, 120.0 + track * 25,
        int(x), 100 + track * 25, int(x) + 20, 120 + track * 25,
        0.2, False, x + 10, 110.0 + track * 25, 400.0, 1.0, 1, 1.0, "", "",
    )


def h6_synthetic_smoke(output: Path | None = None) -> dict[str, object]:
    temporary = tempfile.TemporaryDirectory(prefix="beeid-h6-synthetic-") if output is None else None
    root = (Path(temporary.name) / "output") if temporary else output.expanduser().resolve(strict=False)  # type: ignore[union-attr]
    root.mkdir(parents=True, exist_ok=True)
    observations = [
        _observation(frame, track, 40.0 + track * 90 + (frame * (3 if track == 1 else -2)))
        for frame in range(1, 9) for track in (1, 2)
    ]
    rng = np.random.default_rng(24)
    prototypes = rng.normal(size=(2, 16)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    embeddings = np.stack([
        prototypes[item.track_id - 1] + rng.normal(scale=0.03, size=16) for item in observations
    ]).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    oracle_rows, oracle = audit_candidate_reachability(
        observations, list(range(len(observations))), window_length=6,
        window_stride=3, max_frame_gap=3,
    )
    windows = build_windows(observations, list(range(len(observations))), 6, 3)
    batch = collate_windows(windows[:2], observations, embeddings, 6)
    model = GlobalTrajectoryReasoner(H6ModelSpec(16, 32, 4, 2, 64, 0.0, 6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    gradient_parameters: set[str] = set()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        encoded = model.encode(batch.embeddings, batch.geometry, batch.frame_index, batch.token_mask)
        losses: list[torch.Tensor] = []
        for sample, identities in enumerate(batch.identities):
            count = len(identities)
            frames = batch.frame_index[sample, :count]
            left, right, labels = dense_candidate_edges(frames, identities, 3, 4.0)
            logits = model.pair_logits(encoded[sample, :count], batch.geometry[sample, :count], frames, left, right)
            triples = identity_triples(frames, identities, 3)
            losses.append(balanced_edge_loss(logits, labels) + 0.1 * cycle_consistency_loss(
                model, encoded[sample, :count], batch.geometry[sample, :count], frames, triples
            ))
        loss = torch.stack(losses).mean()
        loss.backward()
        gradient_parameters.update(name for name, value in model.named_parameters() if value.grad is not None)
        optimizer.step()
    required_gradients = {name for name, value in model.named_parameters() if value.requires_grad}
    if gradient_parameters != required_gradients:
        raise RuntimeError("H6 smoke found trainable parameters outside the actual training graph")
    probabilities = {
        (index, index + 2): 0.95 for index in range(len(observations) - 2)
        if observations[index].track_id == observations[index + 2].track_id
    }
    neural_ids, _ = cluster_global_edges(observations, probabilities, 0.5)
    baseline = []
    for index, item in enumerate(observations):
        # Deliberately fragment both GT identities once.
        baseline.append({
            "model": "test_only_encoder", "variant": "baseline_association",
            "stage": "gt_detection_boxes", "video_id": item.video_id, "frame_id": item.frame,
            "observation_id": item.observation_id, "gt_identity": item.identity,
            "gt_track_id": item.track_id, "predicted_track_id": item.track_id + (10 if item.frame >= 5 else 0),
            "matched_existing_track": item.frame > 1, "association_score": "",
            "appearance_similarity": "", "motion_score": "", "normalized_motion_distance": "",
            "reliability": "", "effective_memory_alpha": "", "memory_update_accepted": False,
            "eligible_memory_update": False, "risk_identity_history_outlier": "",
            "risk_bbox_scale_change": "", "risk_orientation_change_proxy": "",
            "risk_sharpness_change": "", "risk_crowding_overlap": "",
        })
    baseline_lookup = {row["observation_id"]: row for row in baseline}
    trajectories = trajectory_feature_rows(observations, neural_ids, probabilities, baseline_lookup)
    attach_offline_trajectory_utility(trajectories, observations, baseline_lookup, neural_ids)
    method_features_before = [list(row["features"]) for row in trajectories]
    renamed = [replace(item, identity=f"permuted:{3 - item.track_id}") for item in observations]
    permuted = trajectory_feature_rows(renamed, neural_ids, probabilities, baseline_lookup)
    if method_features_before != [list(row["features"]) for row in permuted]:
        raise RuntimeError("H6 method features changed after GT identity permutation")
    scores = [0.99 for _ in trajectories]
    accepted_rows, decisions = assignment_rows(
        "test_only_encoder", observations, neural_ids, trajectories, scores, 0.9, baseline
    )
    fallback_rows, fallback_decisions = assignment_rows(
        "test_only_encoder", observations, neural_ids, trajectories, scores, None, baseline
    )
    fallback = [row for row in fallback_rows if row["variant"] == "calibrated_selective_global"]
    if any(str(row["predicted_track_id"]) != str(baseline_lookup[row["observation_id"]]["predicted_track_id"]) for row in fallback):
        raise RuntimeError("H6 fail-closed selector did not reproduce baseline exactly")
    result = {
        "status": "passed", "test_only": True, "test_only_encoder": True,
        "real_experiment_result": False, "oracle_edge_recall": oracle["edge_recall"],
        "oracle_rows": len(oracle_rows), "window_count": len(windows),
        "trained_parameter_count": len(gradient_parameters),
        "trajectory_component_count": len(trajectories),
        "accepted_component_count": sum(bool(row["accepted"]) for row in decisions),
        "fallback_component_count": len(fallback_decisions),
        "exact_baseline_fallback_verified": True, "output_root": str(root),
        "gt_identity_permutation_invariant": True,
        "metadata_status": "SERVER_VALIDATION_PENDING", "final_test_read": False,
    }
    atomic_write_json(root / "h6_synthetic_metadata.json", result)
    return result
