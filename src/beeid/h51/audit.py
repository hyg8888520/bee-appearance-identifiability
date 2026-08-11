"""Runtime gradient, execution-path and teacher-forced H5.1 audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from typing import Any, Sequence

import numpy as np
import torch

from ..config import H5Config
from ..data.mot import Observation
from ..h5.experiment import TrainingTransition, _batched_transition_loss
from ..h5.model import BeeTrackQuery
from ..h5.tracker import normalized_geometry


MEMORY_PARAMETER_PREFIXES = ("memory_attention.", "memory_norm.")
GRADIENT_NONZERO_THRESHOLD = 0.0


class ModuleCallProbe(AbstractContextManager["ModuleCallProbe"]):
    """Forward hooks that record executed modules without changing tensors."""

    def __init__(self, model: BeeTrackQuery) -> None:
        self.model = model
        self.calls: Counter[str] = Counter()
        self._handles: list[Any] = []

    def __enter__(self) -> "ModuleCallProbe":
        for name, module in self.model.named_modules():
            if not name or any(True for _ in module.children()):
                continue
            self._handles.append(
                module.register_forward_hook(
                    lambda _module, _args, _output, module_name=name: self.calls.update([module_name])
                )
            )
        # MultiheadAttention has an out_proj child that is implemented through
        # functional calls, so hook the parent as the authoritative call record.
        for name in ("frame_attention", "memory_attention"):
            module = getattr(self.model, name)
            self._handles.append(
                module.register_forward_hook(
                    lambda _module, _args, _output, module_name=name: self.calls.update([module_name])
                )
            )
        return self

    def __exit__(self, *_args: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def synthetic_transition_batch(embedding_dim: int, seed: int = 24) -> tuple[
    list[TrainingTransition], np.ndarray, np.ndarray
]:
    """Create a deterministic batch with the exact H5 batched-loss tensor contract."""
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(8, embedding_dim)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    geometry = np.asarray(
        [
            [0.10, 0.25, 0.15, 0.20], [0.70, 0.25, 0.15, 0.20],
            [0.12, 0.25, 0.15, 0.20], [0.68, 0.25, 0.15, 0.20],
            [0.20, 0.35, 0.16, 0.20], [0.75, 0.35, 0.16, 0.20],
            [0.22, 0.35, 0.16, 0.20], [0.73, 0.35, 0.16, 0.20],
        ],
        dtype=np.float32,
    )
    transitions = [
        TrainingTransition((0, 1), (2, 3), (0, 1)),
        TrainingTransition((4, 5), (6, 7), (0, 1)),
    ]
    return transitions, embeddings, geometry


def runtime_gradient_coverage(
    model: BeeTrackQuery,
    transitions: Sequence[TrainingTransition],
    embeddings: np.ndarray,
    geometry: np.ndarray,
    h5: H5Config,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward through the actual H5 loss and report every named parameter."""
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    with ModuleCallProbe(model) as probe:
        loss, pair_count, prediction_count = _batched_transition_loss(
            model, transitions, list(range(len(transitions))), embeddings, geometry,
            device, h5.reliability_loss_weight,
        )
        loss.backward()
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        present = gradient is not None
        finite = bool(torch.isfinite(gradient).all().item()) if present else False
        norm = float(torch.linalg.vector_norm(gradient.detach()).cpu()) if present else ""
        nonzero = bool(present and finite and float(norm) > GRADIENT_NONZERO_THRESHOLD)
        rows.append(
            {
                "parameter": name,
                "parameter_count": parameter.numel(),
                "requires_grad": bool(parameter.requires_grad),
                "grad_present": present,
                "finite": finite,
                "grad_norm": norm,
                "grad_nonzero": nonzero,
                "grad_nonzero_threshold": GRADIENT_NONZERO_THRESHOLD,
                "autograd_connected_to_training_objective": bool(parameter.requires_grad and present),
                "used_in_training_objective": bool(
                    parameter.requires_grad and present and finite and nonzero
                ),
                "gradient_health_pass": bool(parameter.requires_grad and present and finite and nonzero),
                "expected_memory_path_parameter": name.startswith(MEMORY_PARAMETER_PREFIXES),
                "audit_kind": "runtime_forward_backward",
                "final_test_read": False,
            }
        )
    missing = [row["parameter"] for row in rows if row["requires_grad"] and not row["grad_present"]]
    memory_missing = [name for name in missing if str(name).startswith(MEMORY_PARAMETER_PREFIXES)]
    unexpected_missing = [name for name in missing if name not in memory_missing]
    zero_gradients = [
        row["parameter"] for row in rows
        if row["requires_grad"] and row["grad_present"] and row["finite"] and not row["grad_nonzero"]
    ]
    expected_memory = [name for name, _ in model.named_parameters() if name.startswith(MEMORY_PARAMETER_PREFIXES)]
    summary = {
        "loss": float(loss.detach().cpu()),
        "pair_count": pair_count,
        "prediction_count": prediction_count,
        "probe_transition_count": len(transitions),
        "probe_query_counts": sorted({item.query_count for item in transitions}),
        "probe_current_candidate_counts": sorted({item.detection_count for item in transitions}),
        "probe_multi_candidate_only": bool(transitions) and all(
            item.detection_count >= 2 for item in transitions
        ),
        "called_modules": dict(sorted(probe.calls.items())),
        "parameter_count": len(rows),
        "missing_gradient_parameters": missing,
        "memory_parameter_names": expected_memory,
        "memory_missing_gradient_parameters": memory_missing,
        "unexpected_missing_gradient_parameters": unexpected_missing,
        "zero_gradient_parameters": zero_gradients,
        "zero_gradient_interpretation": (
            "finite_zero_on_this_runtime_probe_stops_readiness_without_automatic_causal_attribution"
        ),
        "gradient_nonzero_threshold": GRADIENT_NONZERO_THRESHOLD,
        "memory_defect_detected": sorted(memory_missing) == sorted(expected_memory) and len(expected_memory) == 6,
        "all_nonmemory_trainable_gradients_finite": all(
            row["grad_present"] and row["finite"]
            for row in rows if row["requires_grad"] and not row["expected_memory_path_parameter"]
        ),
        "all_nonmemory_trainable_gradients_finite_and_nonzero": all(
            row["grad_present"] and row["finite"] and row["grad_nonzero"]
            for row in rows if row["requires_grad"] and not row["expected_memory_path_parameter"]
        ),
        "all_trainable_gradients_healthy": all(
            row["grad_present"] and row["finite"] and row["grad_nonzero"]
            for row in rows if row["requires_grad"]
        ),
        "method_ready": all(
            row["grad_present"] and row["finite"] and row["grad_nonzero"]
            for row in rows if row["requires_grad"]
        ),
        "diagnostic_completed": True,
        "final_test_read": False,
    }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return rows, summary


@torch.inference_mode()
def teacher_forced_one_step_audit(
    model_name: str,
    model: BeeTrackQuery,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    selected_indices: Sequence[int],
    h5: H5Config,
    device: torch.device,
    max_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    """Offline GT-conditioned one-step audit, explicitly outside deployable rollout."""
    by_video_frame: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in selected_indices:
        by_video_frame[(observations[index].video_id, observations[index].frame)].append(index)
    rows: list[dict[str, Any]] = []
    model.eval()
    with ModuleCallProbe(model) as probe:
        for video_id in sorted({key[0] for key in by_video_frame}):
            frames = sorted(frame for candidate, frame in by_video_frame if candidate == video_id)
            for previous_frame, current_frame in zip(frames, frames[1:]):
                if current_frame != previous_frame + 1:
                    continue
                previous = sorted(by_video_frame[(video_id, previous_frame)], key=lambda i: observations[i].observation_id)
                current = sorted(by_video_frame[(video_id, current_frame)], key=lambda i: observations[i].observation_id)
                target_by_identity = {observations[index].identity: position for position, index in enumerate(current)}
                continuing = [index for index in previous if observations[index].identity in target_by_identity]
                if not continuing or not current:
                    continue
                prev_raw = torch.as_tensor(np.asarray(embeddings[continuing]), device=device)
                cur_raw = torch.as_tensor(np.asarray(embeddings[current]), device=device)
                prev_geometry = torch.as_tensor(
                    np.stack([normalized_geometry(observations[index]) for index in continuing]), device=device
                )
                cur_geometry = torch.as_tensor(
                    np.stack([normalized_geometry(observations[index]) for index in current]), device=device
                )
                previous_tokens = model.encode_detections(prev_raw, prev_geometry)
                current_tokens = model.encode_detections(cur_raw, cur_geometry)
                refreshed = model.refresh(previous_tokens, current_tokens)
                delta = torch.abs(prev_geometry[:, None, :] - cur_geometry[None, :, :])
                logits = model.pair_logits(refreshed, current_tokens, delta)
                scores = torch.sigmoid(logits)
                reliability = model.reliability(logits)
                diagonal = torch.linalg.vector_norm(prev_geometry[:, 2:], dim=1).clamp_min(1e-6)
                distances = torch.linalg.vector_norm(
                    prev_geometry[:, None, :2] - cur_geometry[None, :, :2], dim=2
                ) / diagonal[:, None]
                for query_position, previous_index in enumerate(continuing):
                    positive_position = target_by_identity[observations[previous_index].identity]
                    values = scores[query_position]
                    positive = float(values[positive_position].cpu())
                    order = torch.argsort(values, descending=True, stable=True)
                    rank = int((order == positive_position).nonzero(as_tuple=False)[0, 0]) + 1
                    negatives = torch.cat((values[:positive_position], values[positive_position + 1 :]))
                    best_negative = float(negatives.max().cpu()) if negatives.numel() else 0.0
                    strict_unique_rank1 = bool(
                        not negatives.numel() or positive > best_negative
                    )
                    positive_tie_count = int((values == values[positive_position]).sum().cpu())
                    positive_distance = float(distances[query_position, positive_position].cpu())
                    item = observations[current[positive_position]]
                    rows.append(
                        {
                            "model": model_name, "video_id": video_id,
                            "previous_frame_id": previous_frame, "frame_id": current_frame,
                            "previous_observation_id": observations[previous_index].observation_id,
                            "positive_observation_id": item.observation_id,
                            "candidate_count": len(current), "positive_score": positive,
                            "positive_rank": rank, "positive_margin": positive - best_negative,
                            "stable_order_rank1": rank == 1,
                            "strict_unique_rank1": strict_unique_rank1,
                            "positive_score_tie_count": positive_tie_count,
                            "positive_score_tied": positive_tie_count > 1,
                            "absolute_threshold_pass": positive >= h5.min_assignment_score,
                            "positive_normalized_distance": positive_distance,
                            "motion_gate_pass": positive_distance <= h5.max_normalized_distance,
                            "reliability": float(reliability[query_position].cpu()),
                            "one_step_continuity": strict_unique_rank1,
                            "query_norm_initial_raw_token": float(torch.linalg.vector_norm(previous_tokens[query_position]).cpu()),
                            "query_norm_after_frame_refresh": float(torch.linalg.vector_norm(refreshed[query_position]).cpu()),
                            "memory_read": False, "memory_write": False,
                            "offline_gt_audit": True, "deployable_rollout": False,
                            "input_partition": "development_validation",
                            "gt_identity_used_to_construct_query_target": True,
                            "gt_identity_used_for_inference_decision": False,
                            "final_test_read": False,
                        }
                    )
                    if len(rows) >= max_rows:
                        break
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break
    summary = {
        "model": model_name,
        "query_count": len(rows),
        "rank1_rate": float(np.mean([row["strict_unique_rank1"] for row in rows])) if rows else 0.0,
        "rank1_definition": "positive_score_strictly_greater_than_every_negative_ties_fail",
        "stable_order_rank1_rate_descriptive_only": float(
            np.mean([row["stable_order_rank1"] for row in rows])
        ) if rows else 0.0,
        "positive_score_tie_rate": float(np.mean([row["positive_score_tied"] for row in rows])) if rows else 0.0,
        "mean_positive_score": float(np.mean([row["positive_score"] for row in rows])) if rows else 0.0,
        "mean_positive_margin": float(np.mean([row["positive_margin"] for row in rows])) if rows else 0.0,
        "absolute_threshold_pass_rate": float(np.mean([row["absolute_threshold_pass"] for row in rows])) if rows else 0.0,
        "motion_gate_pass_rate": float(np.mean([row["motion_gate_pass"] for row in rows])) if rows else 0.0,
        "mean_reliability": float(np.mean([row["reliability"] for row in rows])) if rows else 0.0,
        "offline_gt_audit": True,
        "deployable_rollout": False,
        "input_partition": "development_validation",
        "diagnostic_row_limit": max_rows,
        "diagnostic_not_confirmatory": True,
        "final_test_read": False,
    }
    return rows, summary, dict(sorted(probe.calls.items()))


def build_path_audit(
    gradient_summary: dict[str, Any],
    teacher_calls: dict[str, int],
    rollout_calls: dict[str, dict[str, int]],
    rollout_activity: dict[str, Any],
) -> dict[str, Any]:
    """Join runtime evidence with explicit training/deployment state boundaries."""
    training_calls = dict(gradient_summary["called_modules"])
    return {
        "training_path": {
            "runtime_called_modules": training_calls,
            "query_initialization": "raw_teacher_forced_previous_detection_token",
            "query_l2_normalized_before_refresh": False,
            "memory_read_called": training_calls.get("memory_attention", 0) > 0,
            "memory_write_called": False,
            "loss_entrypoint": "beeid.h5.experiment._batched_transition_loss",
            "gt_identity_role": "construct_supervised_transition_targets_on_project_train",
        },
        "teacher_forced_offline_audit_path": {
            "runtime_called_modules": teacher_calls,
            "query_initialization": "previous_detection_token_selected_by_gt_identity",
            "memory_read_called": teacher_calls.get("memory_attention", 0) > 0,
            "offline_gt_audit": True,
            "deployable_rollout": False,
        },
        "deployable_rollout_path": {
            "runtime_called_modules_by_variant": rollout_calls,
            "query_initialization": "l2_normalized_detection_token_for_new_track",
            "query_refresh": "frame_attention_then_optional_memory_attention",
            "query_update": "l2_normalized_mixture_after_accepted_match",
            "memory_read_write_activity": rollout_activity,
            "gt_identity_decision_input": False,
            "teacher_forcing": False,
            "deployable_rollout": True,
        },
        "boundary": {
            "teacher_forced_metrics_are_method_results": False,
            "rollout_assignments_are_method_results": True,
            "checkpoint_modified": False,
            "retraining_performed": False,
            "gradient_probe_historical_binary_direct_attestation": False,
            "gradient_probe_scope": "current_branch_recreation_of_checkpoint_pinned_h5_v2_loss",
        },
        "memory_parameters_used_in_training_objective": not bool(
            gradient_summary["memory_missing_gradient_parameters"]
        ),
        "method_ready": gradient_summary["method_ready"],
        "diagnostic_not_confirmatory": True,
        "final_test_read": False,
    }
