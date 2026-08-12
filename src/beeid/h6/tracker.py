"""Offline global graph association and trajectory-level selective fallback."""

from __future__ import annotations

from collections import defaultdict
from collections import Counter
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .data import build_windows, candidate_pairs, collate_windows
from .model import GlobalTrajectoryReasoner, TRAJECTORY_FEATURE_NAMES, TrajectorySelector


class _UnionFind:
    def __init__(self, frames: Sequence[int]) -> None:
        self.parent = list(range(len(frames)))
        self.members = {index: {index} for index in range(len(frames))}
        self.frames = {index: {int(frame)} for index, frame in enumerate(frames)}

    def root(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> bool:
        a, b = self.root(left), self.root(right)
        if a == b:
            return False
        if self.frames[a] & self.frames[b]:
            return False
        if len(self.members[a]) < len(self.members[b]):
            a, b = b, a
        self.parent[b] = a
        self.members[a].update(self.members.pop(b))
        self.frames[a].update(self.frames.pop(b))
        return True


@torch.inference_mode()
def global_edge_probabilities(
    model: GlobalTrajectoryReasoner, observations: Sequence[Any], embeddings: np.ndarray,
    *, window_length: int, window_stride: int, max_frame_gap: int,
    device: torch.device, amp: bool, amp_dtype: torch.dtype, max_tokens_per_batch: int,
) -> dict[tuple[int, int], float]:
    indices = list(range(len(observations)))
    windows = build_windows(observations, indices, window_length, window_stride)
    totals: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    model.eval()
    from .data import token_budget_batches
    for group in token_budget_batches(windows, max_tokens_per_batch):
        batch = collate_windows(group, observations, embeddings, window_length).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp and device.type == "cuda"):
            encoded_batch = model.encode(batch.embeddings, batch.geometry, batch.frame_index, batch.token_mask)
            for sample_index, window in enumerate(group):
                count = int(batch.token_mask[sample_index].sum().item())
                encoded = encoded_batch[sample_index, :count]
                left, right = candidate_pairs(batch.frame_index[sample_index, :count], max_frame_gap)
                for start in range(0, len(left), 65_536):
                    a, b = left[start : start + 65_536], right[start : start + 65_536]
                    probabilities = model.pair_logits(
                        encoded, batch.geometry[sample_index, :count],
                        batch.frame_index[sample_index, :count], a, b
                    ).sigmoid().float().cpu().tolist()
                    for local_left, local_right, probability in zip(a.cpu().tolist(), b.cpu().tolist(), probabilities):
                        global_left = window.indices[local_left]
                        global_right = window.indices[local_right]
                        key = (min(global_left, global_right), max(global_left, global_right))
                        totals[key][0] += float(probability)
                        totals[key][1] += 1.0
    return {key: value[0] / value[1] for key, value in totals.items()}


def cluster_global_edges(
    observations: Sequence[Any], probabilities: Mapping[tuple[int, int], float], threshold: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    frames = [int(item.frame) for item in observations]
    union = _UnionFind(frames)
    accepted: list[dict[str, Any]] = []
    ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    for (left, right), probability in ordered:
        if probability < threshold:
            break
        merged = union.union(left, right)
        accepted.append({
            "left_index": left, "right_index": right, "probability": probability,
            "accepted": merged, "rejection_reason": "" if merged else "same_component_or_frame_conflict",
        })
    roots = [union.root(index) for index in range(len(observations))]
    ordering = {
        root: position + 1 for position, root in enumerate(sorted(
            set(roots), key=lambda value: min(union.members[union.root(value)])
        ))
    }
    return [ordering[union.root(index)] for index in range(len(observations))], accepted


def trajectory_feature_rows(
    observations: Sequence[Any], neural_ids: Sequence[int], probabilities: Mapping[tuple[int, int], float],
    baseline_by_observation: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    neural_groups: dict[int, list[int]] = defaultdict(list)
    for index, identity in enumerate(neural_ids):
        neural_groups[int(identity)].append(index)
    baseline_ids = [
        str(baseline_by_observation[item.observation_id]["predicted_track_id"])
        for item in observations
    ]
    # Connected components of the overlap bipartite graph. This is the smallest
    # safe unit that can accept an entire split or merge without mixing baseline
    # and neural labels inside the same proposed repair.
    neural_to_baseline: dict[int, set[str]] = defaultdict(set)
    baseline_to_neural: dict[str, set[int]] = defaultdict(set)
    for index, neural_id in enumerate(neural_ids):
        neural_to_baseline[int(neural_id)].add(baseline_ids[index])
        baseline_to_neural[baseline_ids[index]].add(int(neural_id))
    components: list[tuple[list[int], list[str]]] = []
    unvisited = set(neural_groups)
    while unvisited:
        pending_neural = [min(unvisited)]
        component_neural: set[int] = set()
        component_baseline: set[str] = set()
        while pending_neural:
            neural_id = pending_neural.pop()
            if neural_id in component_neural:
                continue
            component_neural.add(neural_id)
            unvisited.discard(neural_id)
            for baseline_id in neural_to_baseline[neural_id]:
                if baseline_id not in component_baseline:
                    component_baseline.add(baseline_id)
                    pending_neural.extend(sorted(baseline_to_neural[baseline_id]))
        components.append((sorted(component_neural), sorted(component_baseline)))
    rows: list[dict[str, Any]] = []
    incident: dict[int, list[tuple[float, tuple[int, int]]]] = defaultdict(list)
    for key, value in probabilities.items():
        incident[key[0]].append((float(value), key))
        incident[key[1]].append((float(value), key))
    for values in incident.values():
        values.sort(reverse=True)

    def competitor(edge: tuple[int, int]) -> float:
        candidates = incident.get(edge[0], [])[:2] + incident.get(edge[1], [])[:2]
        return max((value for value, key in candidates if key != edge), default=0.0)

    for component_id, (component_neural, component_baseline) in enumerate(components, start=1):
        members = sorted(
            (index for neural_id in component_neural for index in neural_groups[neural_id]),
            key=lambda index: (observations[index].frame, observations[index].observation_id),
        )
        members.sort(key=lambda index: (observations[index].frame, observations[index].observation_id))
        edge_probabilities: list[float] = []
        margins: list[float] = []
        spatial_steps: list[float] = []
        for neural_id in component_neural:
            ordered = sorted(neural_groups[neural_id], key=lambda index: observations[index].frame)
            for left, right in zip(ordered, ordered[1:]):
                probability = float(probabilities.get((min(left, right), max(left, right)), 0.0))
                edge_probabilities.append(probability)
                margins.append(probability - competitor((min(left, right), max(left, right))))
                diagonal = max(1e-6, math.hypot(observations[left].original_width, observations[left].original_height))
                spatial_steps.append(math.hypot(
                    observations[right].center_x - observations[left].center_x,
                    observations[right].center_y - observations[left].center_y,
                ) / diagonal)
        frames = [observations[index].frame for index in members]
        def choose2(value: int) -> int:
            return value * (value - 1) // 2

        neural_count = Counter(int(neural_ids[index]) for index in members)
        baseline_count = Counter(baseline_ids[index] for index in members)
        joint_count = Counter((int(neural_ids[index]), baseline_ids[index]) for index in members)
        pair_count = choose2(len(members))
        disagreement_count = (
            sum(choose2(value) for value in neural_count.values())
            + sum(choose2(value) for value in baseline_count.values())
            - 2 * sum(choose2(value) for value in joint_count.values())
        )
        values = {
            "log_length": math.log1p(len(members)),
            "temporal_coverage": min(1.0, len(set(frames)) / max(1, max(frames) - min(frames) + 1)),
            "mean_edge_probability": float(np.mean(edge_probabilities)) if edge_probabilities else 0.0,
            "minimum_edge_probability": min(edge_probabilities, default=0.0),
            "mean_edge_margin": float(np.mean(margins)) if margins else 0.0,
            "minimum_edge_margin": min(margins, default=0.0),
            "mean_spatial_step": float(np.mean(spatial_steps)) if spatial_steps else 0.0,
            "baseline_track_count": float(len(component_baseline)),
            "neural_track_count": float(len(component_neural)),
            "partition_disagreement_fraction": disagreement_count / max(1, pair_count),
        }
        rows.append({
            "component_id": component_id, "neural_track_ids": component_neural,
            "baseline_track_ids": component_baseline, "member_indices": members,
            "observation_ids": [observations[index].observation_id for index in members],
            "features": [values[name] for name in TRAJECTORY_FEATURE_NAMES], **values,
        })
    return rows


def attach_offline_trajectory_utility(
    rows: list[dict[str, Any]], observations: Sequence[Any],
    baseline_by_observation: Mapping[str, Mapping[str, Any]], neural_ids: Sequence[int],
) -> None:
    """Offline label only: exact pair-partition gain/harm versus frozen baseline."""
    for row in rows:
        members = list(row["member_indices"])
        triples = [
            (
                str(observations[index].identity),
                str(baseline_by_observation[observations[index].observation_id]["predicted_track_id"]),
                int(neural_ids[index]),
            )
            for index in members
        ]
        def choose2(value: int) -> int:
            return value * (value - 1) // 2

        g = Counter(value[0] for value in triples)
        b = Counter(value[1] for value in triples)
        n = Counter(value[2] for value in triples)
        gb = Counter((value[0], value[1]) for value in triples)
        gn = Counter((value[0], value[2]) for value in triples)
        bn = Counter((value[1], value[2]) for value in triples)
        gbn = Counter(triples)
        same_gn_diff_b = sum(choose2(value) for value in gn.values()) - sum(choose2(value) for value in gbn.values())
        diff_gb_same_n = (
            sum(choose2(value) for value in n.values())
            - sum(choose2(value) for value in gn.values())
            - sum(choose2(value) for value in bn.values())
            + sum(choose2(value) for value in gbn.values())
        )
        same_gb_diff_n = sum(choose2(value) for value in gb.values()) - sum(choose2(value) for value in gbn.values())
        diff_gn_same_b = (
            sum(choose2(value) for value in b.values())
            - sum(choose2(value) for value in gb.values())
            - sum(choose2(value) for value in bn.values())
            + sum(choose2(value) for value in gbn.values())
        )
        gain = same_gn_diff_b + diff_gn_same_b
        harm = same_gb_diff_n + diff_gb_same_n
        denominator = max(1, gain + harm)
        row.update({
            "offline_gain_pairs": gain, "offline_harm_pairs": harm,
            "offline_utility": (gain - 2.0 * harm) / denominator,
            "offline_positive": gain > 2 * harm,
            "offline_label_uses_gt": True, "method_decision_uses_gt": False,
        })


@torch.inference_mode()
def selector_probabilities(selector: TrajectorySelector, rows: Sequence[dict[str, Any]], device: torch.device) -> list[float]:
    if not rows:
        return []
    values = torch.tensor([row["features"] for row in rows], dtype=torch.float32, device=device)
    selector.eval()
    return selector(values).sigmoid().cpu().tolist()


def assignment_rows(
    model_name: str, observations: Sequence[Any], neural_ids: Sequence[int],
    trajectory_rows: Sequence[dict[str, Any]], selector_scores: Sequence[float],
    selector_threshold: float | None, baseline_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = {str(row["observation_id"]): row for row in baseline_rows}
    by_neural: dict[int, tuple[dict[str, Any], float]] = {}
    for row, score in zip(trajectory_rows, selector_scores):
        for neural_id in row["neural_track_ids"]:
            by_neural[int(neural_id)] = (row, float(score))
    max_baseline = max((int(row["predicted_track_id"]) for row in baseline_rows), default=0)
    next_new = max_baseline + 1
    accepted_ids: dict[int, int] = {}
    decisions: list[dict[str, Any]] = []
    for trajectory, score in zip(trajectory_rows, selector_scores):
        changes_baseline = float(trajectory["partition_disagreement_fraction"]) > 0.0
        accepted = selector_threshold is not None and score >= selector_threshold and changes_baseline
        if accepted:
            for neural_id in trajectory["neural_track_ids"]:
                accepted_ids[int(neural_id)] = next_new
                next_new += 1
        decisions.append({
            "model": model_name, "video_id": observations[0].video_id,
            "component_id": trajectory["component_id"],
            "neural_track_ids": trajectory["neural_track_ids"],
            "baseline_track_ids": trajectory["baseline_track_ids"],
            "selector_probability": score,
            "selector_threshold": "" if selector_threshold is None else selector_threshold,
            "changes_baseline": changes_baseline, "accepted": accepted,
            "reason": "accepted" if accepted else (
                "calibration_gate_failed" if selector_threshold is None else
                "no_baseline_change" if not changes_baseline else "below_threshold"
            ),
            "observation_count": len(trajectory["member_indices"]),
            "method_decision_uses_gt": False, "final_test_read": False,
        })
    neural_output: list[dict[str, Any]] = []
    selective_output: list[dict[str, Any]] = []
    for index, item in enumerate(observations):
        base = baseline[item.observation_id]
        neural_id = int(neural_ids[index])
        neural_output.append({**base, "variant": "global_trajectory_reasoner", "predicted_track_id": neural_id})
        accepted = neural_id in accepted_ids
        selective_output.append({
            **base, "variant": "calibrated_selective_global",
            "predicted_track_id": accepted_ids[neural_id] if accepted else base["predicted_track_id"],
            "h6_intervention_accepted": accepted,
            "h6_exact_baseline_fallback": not accepted,
            "h6_selector_probability": by_neural[neural_id][1],
            "h6_selector_threshold": "" if selector_threshold is None else selector_threshold,
            "h6_ground_truth_decision_input": False, "final_test_read": False,
        })
    baseline_output = [
        {**row, "h6_intervention_accepted": False, "h6_exact_baseline_fallback": True,
         "h6_selector_probability": "", "h6_selector_threshold": "",
         "h6_ground_truth_decision_input": False, "final_test_read": False}
        for row in baseline_rows
    ]
    return baseline_output + neural_output + selective_output, decisions
