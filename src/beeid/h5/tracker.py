"""Causal BeeTrackQuery inference on exact GT detections.

The batched entry point advances at most one frame from each independent
``model x video x variant`` sequence per round. This is deliberately not a
temporal batch: a sequence is only advanced after its previous association has
updated its own state, so it preserves the original strictly-causal tracker.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..config import H5Config
from ..data.mot import Observation
from ..h3.assignment import maximum_weight_matching
from . import H5_VARIANTS
from .model import BeeTrackQuery


TRACKER_IMPLEMENTATION = "beeid.h5.tracker:v4-causal-frame-batching"


@dataclass
class QueryTrack:
    identifier: int
    query: torch.Tensor
    geometry: torch.Tensor
    last_frame: int
    memory: deque[torch.Tensor] = field(default_factory=deque)
    previous_center: tuple[float, float] | None = None
    last_center: tuple[float, float] = (0.0, 0.0)


def normalized_geometry(item: Observation) -> np.ndarray:
    return np.asarray(
        [
            item.center_x / item.image_width,
            item.center_y / item.image_height,
            item.original_width / item.image_width,
            item.original_height / item.image_height,
        ],
        dtype=np.float32,
    )


def _distance(track: QueryTrack, item: Observation) -> float:
    """Reference scalar motion gate retained for audit and single-sequence tests."""
    dx = normalized_geometry(item)[:2] - track.geometry.detach().cpu().numpy()[:2]
    diagonal = max(1e-6, float(np.linalg.norm(track.geometry.detach().cpu().numpy()[2:])))
    return float(np.linalg.norm(dx) / diagonal)


@dataclass
class _SequenceState:
    model_name: str
    variant: str
    observations: Sequence[Observation]
    embeddings: np.ndarray
    config: H5Config
    frames: tuple[int, ...]
    by_frame: dict[int, tuple[int, ...]]
    active: dict[int, QueryTrack] = field(default_factory=dict)
    next_identifier: int = 1
    next_frame_index: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.next_frame_index >= len(self.frames)

    @property
    def current_frame(self) -> int:
        if self.complete:
            raise RuntimeError("Completed H5 tracking state has no next frame")
        return self.frames[self.next_frame_index]

    def prepare(self) -> tuple[list[QueryTrack], list[Observation], np.ndarray, np.ndarray]:
        """Prune expired tracks and expose one causally-ready frame."""
        frame = self.current_frame
        self.active = {
            key: value
            for key, value in self.active.items()
            if frame - value.last_frame <= self.config.max_age
        }
        indices = self.by_frame[frame]
        items = [self.observations[index] for index in indices]
        return (
            [self.active[key] for key in sorted(self.active)],
            items,
            np.asarray(self.embeddings[list(indices)], dtype=np.float32),
            np.stack([normalized_geometry(item) for item in items]).astype(np.float32, copy=False),
        )

    def finish_frame(
        self,
        tracks: Sequence[QueryTrack],
        items: Sequence[Observation],
        tokens: torch.Tensor,
        geometry: torch.Tensor,
        queries: torch.Tensor | None,
        scores: np.ndarray | None,
        reliability: np.ndarray | None,
        valid: np.ndarray | None,
    ) -> None:
        """Apply a completed frame association and then make the next frame ready."""
        frame = self.current_frame
        matched_detection: set[int] = set()
        if tracks and items:
            assert queries is not None and scores is not None and reliability is not None and valid is not None
            matches = maximum_weight_matching(
                scores,
                valid=valid,
                minimum_score=self.config.min_assignment_score,
            )
            gated = self.variant == "beetrackquery_gated_memory"
            for track_index, detection_index in matches:
                matched_detection.add(detection_index)
                track = tracks[track_index]
                item = items[detection_index]
                confidence = float(reliability[track_index])
                accepted = not gated or confidence >= self.config.update_gate
                if accepted:
                    refreshed = queries[track_index]
                    track.query = F.normalize(
                        (1.0 - self.config.memory_mix) * refreshed
                        + self.config.memory_mix * tokens[detection_index],
                        dim=0,
                    ).detach()
                    track.memory.append(tokens[detection_index].detach())
                    while len(track.memory) > self.config.memory_slots:
                        track.memory.popleft()
                track.previous_center = track.last_center
                track.last_center = (item.center_x, item.center_y)
                track.geometry = geometry[detection_index].detach()
                track.last_frame = frame
                self.rows.append(
                    _row(
                        self.model_name,
                        self.variant,
                        item,
                        track.identifier,
                        True,
                        float(scores[track_index, detection_index]),
                        confidence,
                        accepted,
                    )
                )
        for detection_index, item in enumerate(items):
            if detection_index in matched_detection:
                continue
            query = F.normalize(tokens[detection_index].detach(), dim=0)
            memory: deque[torch.Tensor] = deque([query], maxlen=self.config.memory_slots)
            self.active[self.next_identifier] = QueryTrack(
                self.next_identifier,
                query,
                geometry[detection_index].detach(),
                frame,
                memory=memory,
                last_center=(item.center_x, item.center_y),
            )
            self.rows.append(_row(self.model_name, self.variant, item, self.next_identifier, False, "", "", True))
            self.next_identifier += 1
        self.next_frame_index += 1


def _new_state(
    model_name: str,
    variant: str,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    config: H5Config,
) -> _SequenceState:
    if variant not in H5_VARIANTS[1:]:
        raise ValueError(f"Unknown trainable H5 variant: {variant}")
    if len(observations) != embeddings.shape[0]:
        raise ValueError("H5 embeddings must align with observations")
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        by_frame[item.frame].append(index)
    ordered = {
        frame: tuple(
            sorted(indices, key=lambda index: (
                observations[index].center_x,
                observations[index].center_y,
            ))
        )
        for frame, indices in by_frame.items()
    }
    return _SequenceState(
        model_name=model_name,
        variant=variant,
        observations=observations,
        embeddings=embeddings,
        config=config,
        frames=tuple(sorted(ordered)),
        by_frame=ordered,
    )


def _batched_memory_read(
    model: BeeTrackQuery,
    queries: torch.Tensor,
    tracks_by_job: Sequence[Sequence[QueryTrack]],
    query_counts: Sequence[int],
    memory_top_k: int,
    use_memory: Sequence[bool],
) -> torch.Tensor:
    """Select per-track short memories with a few batched GPU kernels.

    Python only collects references and index positions.  A single stack and
    scatter build padded memory tokens; rows are grouped by their true memory
    length (at most ``memory_slots``) before cosine/top-k/gather.  Grouping
    keeps ``topk``'s input length and tie behaviour identical to the original
    one-track call while avoiding one tiny GPU kernel per active track.
    """
    batch, max_queries, hidden = queries.shape
    if max_queries == 0:
        return queries
    slots = max(1, memory_top_k)
    flat_count = batch * max_queries
    flat_memory = queries.new_zeros((flat_count, slots, hidden))
    flat_padding = torch.ones((flat_count, slots), dtype=torch.bool, device=queries.device)
    flat_read_enabled = torch.zeros(flat_count, dtype=torch.bool, device=queries.device)
    token_references: list[torch.Tensor] = []
    source_rows: list[int] = []
    source_columns: list[int] = []
    memory_length_by_row: dict[int, int] = {}
    for job_index, tracks in enumerate(tracks_by_job):
        for track_index, track in enumerate(tracks):
            if not use_memory[job_index] or not track.memory:
                continue
            flat_row = job_index * max_queries + track_index
            memory_length_by_row[flat_row] = len(track.memory)
            for memory_index, token in enumerate(track.memory):
                token_references.append(token)
                source_rows.append(flat_row)
                source_columns.append(memory_index)
    if token_references:
        max_memory = max(memory_length_by_row.values())
        source = queries.new_zeros((flat_count, max_memory, hidden))
        device_rows = torch.as_tensor(source_rows, dtype=torch.long, device=queries.device)
        device_columns = torch.as_tensor(source_columns, dtype=torch.long, device=queries.device)
        # One stack/scatter collects all heterogeneous deque entries.
        source[device_rows, device_columns] = torch.stack(token_references)
        for memory_length in sorted(set(memory_length_by_row.values())):
            selected_rows = [
                row for row, length in memory_length_by_row.items()
                if length == memory_length
            ]
            device_selected = torch.as_tensor(selected_rows, dtype=torch.long, device=queries.device)
            values = source[device_selected, :memory_length]
            query = queries.reshape(flat_count, hidden)[device_selected]
            similarities = F.cosine_similarity(values, query[:, None, :], dim=2)
            selected_count = min(memory_top_k, memory_length)
            selected_indices = torch.topk(similarities, selected_count, dim=1).indices
            selected = values.gather(
                1,
                selected_indices[:, :, None].expand(-1, -1, hidden),
            )
            flat_memory[device_selected, :selected_count] = selected
            flat_padding[device_selected, :selected_count] = False
            flat_read_enabled[device_selected] = True
    # MultiheadAttention rejects all-masked rows.  Such rows are ignored below,
    # but receive one finite key so they can safely share the attention kernel.
    inactive = ~flat_read_enabled
    flat_memory[inactive, 0] = queries.reshape(flat_count, hidden)[inactive]
    flat_padding[inactive, 0] = False
    read = model.read_memory_batched(
        queries,
        flat_memory.reshape(batch, max_queries, slots, hidden),
        flat_padding.reshape(batch, max_queries, slots),
    )
    return torch.where(
        flat_read_enabled.reshape(batch, max_queries, 1),
        read,
        queries,
    )


@torch.inference_mode()
def track_sequences_batched(
    model: BeeTrackQuery,
    sequences: Sequence[tuple[str, str, Sequence[Observation], np.ndarray]],
    config: H5Config,
    device: torch.device,
    *,
    batch_size: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    job_completed_callback: Callable[[int, list[dict[str, Any]]], None] | None = None,
    progress_interval_frames: int = 64,
) -> list[list[dict[str, Any]]]:
    """Track independent sequences in causal padded frame batches.

    Each sequence owns query and memory state.  The GPU only batches independent
    next frames, which leaves the original per-sequence temporal ordering and
    inference inputs unchanged.  Association remains exact and is performed
    after one batched device-to-host transfer per job/frame.
    """
    if batch_size < 1:
        raise ValueError("H5 tracking batch_size must be positive")
    if progress_interval_frames < 1:
        raise ValueError("H5 tracking progress_interval_frames must be positive")
    states = [_new_state(*sequence, config) for sequence in sequences]
    state_indices = {id(state): index for index, state in enumerate(states)}
    model.eval()
    completed_frames = 0
    total_frames = sum(len(state.frames) for state in states)
    while True:
        ready = [state for state in states if not state.complete]
        if not ready:
            break
        # Prepare after the preceding association is complete, then greedily
        # bound padded pair tensors by the frozen training memory budget.  This
        # keeps the 4090 busy without letting one crowded frame make a batch
        # unexpectedly unsafe.
        prepared_ready = [(state, state.prepare()) for state in ready]
        batches: list[list[tuple[_SequenceState, tuple[list[QueryTrack], list[Observation], np.ndarray, np.ndarray]]]] = []
        current: list[tuple[_SequenceState, tuple[list[QueryTrack], list[Observation], np.ndarray, np.ndarray]]] = []
        max_queries = 0
        max_detections = 0
        for candidate in prepared_ready:
            tracks, items, _, _ = candidate[1]
            next_queries = max(max_queries, len(tracks))
            next_detections = max(max_detections, len(items))
            next_pairs = (len(current) + 1) * next_queries * next_detections
            if current and (
                len(current) >= batch_size
                or next_pairs > config.max_pair_elements_per_batch
            ):
                batches.append(current)
                current = []
                max_queries = 0
                max_detections = 0
            current.append(candidate)
            max_queries = max(max_queries, len(tracks))
            max_detections = max(max_detections, len(items))
        if current:
            batches.append(current)
        # Keeping contiguous order makes the batching schedule deterministic.
        for batch in batches:
            selected = [value[0] for value in batch]
            prepared = [value[1] for value in batch]
            tracks_by_job = [value[0] for value in prepared]
            items_by_job = [value[1] for value in prepared]
            raw_by_job = [value[2] for value in prepared]
            geometry_by_job = [value[3] for value in prepared]
            query_counts = [len(value) for value in tracks_by_job]
            detection_counts = [len(value) for value in items_by_job]
            max_queries = max(query_counts, default=0)
            max_detections = max(detection_counts, default=0)
            if max_detections == 0:
                raise RuntimeError("H5 tracking found an empty frame")
            embedding_dim = raw_by_job[0].shape[1]
            raw = np.zeros((len(selected), max_detections, embedding_dim), dtype=np.float32)
            geometry_np = np.zeros((len(selected), max_detections, 4), dtype=np.float32)
            detection_padding = np.ones((len(selected), max_detections), dtype=bool)
            for job_index, (raw_values, geometry_values) in enumerate(zip(raw_by_job, geometry_by_job)):
                count = detection_counts[job_index]
                raw[job_index, :count] = raw_values
                geometry_np[job_index, :count] = geometry_values
                detection_padding[job_index, :count] = False
            raw_tensor = torch.from_numpy(raw).to(device=device, non_blocking=True)
            geometry_tensor = torch.from_numpy(geometry_np).to(device=device, non_blocking=True)
            tokens = model.encode_detections(raw_tensor, geometry_tensor)
            query_tensor: torch.Tensor | None = None
            scores: np.ndarray | None = None
            reliability: np.ndarray | None = None
            valid: np.ndarray | None = None
            if max_queries:
                query_tensor = tokens.new_zeros((len(selected), max_queries, tokens.shape[-1]))
                track_geometry = geometry_tensor.new_zeros((len(selected), max_queries, 4))
                for job_index, tracks in enumerate(tracks_by_job):
                    if not tracks:
                        continue
                    count = len(tracks)
                    query_tensor[job_index, :count] = torch.stack([track.query for track in tracks])
                    track_geometry[job_index, :count] = torch.stack([track.geometry for track in tracks])
                padding_tensor = torch.from_numpy(detection_padding).to(device=device, non_blocking=True)
                query_tensor = model.refresh_batched(query_tensor, tokens, padding_tensor)
                query_tensor = _batched_memory_read(
                    model,
                    query_tensor,
                    tracks_by_job,
                    query_counts,
                    config.memory_top_k,
                    [state.variant != "persistent_query_no_memory" for state in selected],
                )
                geometry_delta = torch.abs(track_geometry[:, :, None, :] - geometry_tensor[:, None, :, :])
                logits = model.pair_logits_batched(query_tensor, tokens, geometry_delta)
                # The old gate called tensor.cpu().numpy() once per pair. The
                # batch-by-query-by-detection gate now transfers once, after
                # all device work.
                diagonal = torch.linalg.vector_norm(track_geometry[:, :, 2:], dim=2).clamp_min(1e-6)
                normalized_distance = torch.linalg.vector_norm(
                    track_geometry[:, :, None, :2] - geometry_tensor[:, None, :, :2],
                    dim=3,
                ) / diagonal[:, :, None]
                scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64, copy=False)
                reliability_tensor = logits.new_zeros((len(selected), max_queries))
                for job_index, (query_count, detection_count) in enumerate(
                    zip(query_counts, detection_counts)
                ):
                    if query_count:
                        # Reliability normalises over real detections only;
                        # padded columns would otherwise change the confidence.
                        reliability_tensor[job_index, :query_count] = model.reliability(
                            logits[job_index, :query_count, :detection_count]
                        )
                reliability = reliability_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
                valid = (normalized_distance <= config.max_normalized_distance).detach().cpu().numpy()
            for job_index, state in enumerate(selected):
                query_count = query_counts[job_index]
                detection_count = detection_counts[job_index]
                state.finish_frame(
                    tracks_by_job[job_index],
                    items_by_job[job_index],
                    tokens[job_index, :detection_count],
                    geometry_tensor[job_index, :detection_count],
                    None if query_tensor is None else query_tensor[job_index, :query_count],
                    None if scores is None else scores[job_index, :query_count, :detection_count],
                    None if reliability is None else reliability[job_index, :query_count],
                    None if valid is None else valid[job_index, :query_count, :detection_count],
                )
                completed_frames += 1
                if progress_callback and (
                    completed_frames % progress_interval_frames == 0 or state.complete
                ):
                    progress_callback(
                        {
                            "completed_frames": completed_frames,
                            "total_frames": total_frames,
                            "model": state.model_name,
                            "video_id": state.observations[0].video_id if state.observations else "",
                            "variant": state.variant,
                            "frame": state.frames[state.next_frame_index - 1],
                        }
                    )
                if state.complete and job_completed_callback:
                    job_completed_callback(
                        state_indices[id(state)],
                        sorted(
                            state.rows,
                            key=lambda row: (row["frame_id"], row["observation_id"]),
                        ),
                    )
    return [
        sorted(state.rows, key=lambda row: (row["frame_id"], row["observation_id"]))
        for state in states
    ]


@torch.inference_mode()
def track_sequence(
    model_name: str,
    variant: str,
    model: BeeTrackQuery,
    observations: Sequence[Observation],
    embeddings: np.ndarray,
    config: H5Config,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for one causal sequence.

    Tests and synthetic smoke use this entry point; production passes multiple
    independent jobs to :func:`track_sequences_batched` for GPU utilization.
    """
    return track_sequences_batched(
        model,
        [(model_name, variant, observations, embeddings)],
        config,
        device,
        batch_size=1,
    )[0]


def _row(
    model: str,
    variant: str,
    item: Observation,
    predicted: int,
    matched: bool,
    score: Any,
    reliability: Any,
    accepted: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "variant": variant,
        "stage": "gt_detection_boxes",
        "video_id": item.video_id,
        "frame_id": item.frame,
        "observation_id": item.observation_id,
        "gt_identity": item.identity,
        "gt_track_id": item.track_id,
        "predicted_track_id": predicted,
        "matched_existing_track": matched,
        "association_score": score,
        "appearance_similarity": "",
        "motion_score": "",
        "normalized_motion_distance": "",
        "reliability": reliability,
        "effective_memory_alpha": "",
        "memory_update_accepted": accepted,
        "eligible_memory_update": matched,
        "risk_identity_history_outlier": "",
        "risk_bbox_scale_change": "",
        "risk_orientation_change_proxy": "",
        "risk_sharpness_change": "",
        "risk_crowding_overlap": "",
    }
