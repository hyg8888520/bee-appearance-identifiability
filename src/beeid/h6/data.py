"""Window construction, dense candidate edges, and padded GPU batches for H6."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class H6Window:
    video_id: str
    start_frame: int
    end_frame: int
    indices: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.video_id}:{self.start_frame:06d}-{self.end_frame:06d}"


@dataclass
class WindowBatch:
    embeddings: Tensor
    geometry: Tensor
    frame_index: Tensor
    token_mask: Tensor
    identities: list[list[str]]
    windows: tuple[H6Window, ...]

    def to(self, device: torch.device) -> "WindowBatch":
        return WindowBatch(
            self.embeddings.to(device), self.geometry.to(device),
            self.frame_index.to(device), self.token_mask.to(device),
            self.identities, self.windows,
        )


def build_windows(
    observations: Sequence[Any],
    indices: Sequence[int],
    window_length: int,
    stride: int,
    max_windows_per_video: int | None = None,
    seed: int = 24,
) -> list[H6Window]:
    by_video: dict[str, list[int]] = {}
    for index in indices:
        by_video.setdefault(observations[index].video_id, []).append(index)
    output: list[H6Window] = []
    for video_id, members in sorted(by_video.items()):
        members.sort(key=lambda index: (
            observations[index].frame, observations[index].center_x,
            observations[index].center_y, observations[index].observation_id,
        ))
        first = min(observations[index].frame for index in members)
        last = max(observations[index].frame for index in members)
        starts = list(range(first, last + 1, stride))
        if starts and starts[-1] + window_length - 1 < last:
            starts.append(max(first, last - window_length + 1))
        windows: list[H6Window] = []
        for start in sorted(set(starts)):
            end = start + window_length - 1
            selected = tuple(index for index in members if start <= observations[index].frame <= end)
            if len(selected) >= 2:
                windows.append(H6Window(video_id, start, end, selected))
        if max_windows_per_video is not None and len(windows) > max_windows_per_video:
            windows.sort(key=lambda item: (
                hashlib.sha256(f"{seed}:{item.key}".encode()).hexdigest(), item.key
            ))
            windows = sorted(windows[:max_windows_per_video], key=lambda item: item.start_frame)
        output.extend(windows)
    return output


def geometry_features(observation: Any, start_frame: int, window_length: int) -> np.ndarray:
    width = max(1.0, float(observation.image_width))
    height = max(1.0, float(observation.image_height))
    return np.asarray([
        observation.center_x / width, observation.center_y / height,
        np.log(max(1e-6, observation.original_width / width)),
        np.log(max(1e-6, observation.original_height / height)),
        np.log(max(1e-6, observation.bbox_area / (width * height))),
        (observation.frame - start_frame) / max(1, window_length - 1),
    ], dtype=np.float32)


def collate_windows(
    windows: Sequence[H6Window], observations: Sequence[Any], embeddings: np.ndarray,
    window_length: int,
) -> WindowBatch:
    if not windows:
        raise ValueError("Cannot collate an empty H6 window batch")
    maximum = max(len(window.indices) for window in windows)
    dimension = int(embeddings.shape[1])
    values = np.zeros((len(windows), maximum, dimension), dtype=np.float32)
    geometry = np.zeros((len(windows), maximum, 6), dtype=np.float32)
    frames = np.zeros((len(windows), maximum), dtype=np.int64)
    mask = np.zeros((len(windows), maximum), dtype=bool)
    identities: list[list[str]] = []
    for batch_index, window in enumerate(windows):
        members = list(window.indices)
        count = len(members)
        values[batch_index, :count] = embeddings[members]
        geometry[batch_index, :count] = np.stack([
            geometry_features(observations[index], window.start_frame, window_length)
            for index in members
        ])
        frames[batch_index, :count] = [observations[index].frame - window.start_frame for index in members]
        mask[batch_index, :count] = True
        identities.append([observations[index].identity for index in members])
    return WindowBatch(
        torch.from_numpy(values), torch.from_numpy(geometry), torch.from_numpy(frames),
        torch.from_numpy(mask), identities, tuple(windows),
    )


def dense_candidate_edges(
    frames: Tensor, identities: Sequence[str], max_frame_gap: int,
    negative_positive_ratio: float | None = None, generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """All cross-frame edges within gap; optional negative sampling is training-only."""
    count = int(frames.numel())
    left, right = torch.triu_indices(count, count, offset=1, device=frames.device)
    gaps = frames[right] - frames[left]
    keep = (gaps > 0) & (gaps <= max_frame_gap)
    left, right = left[keep], right[keep]
    identity_lookup = {identity: index for index, identity in enumerate(sorted(set(identities)))}
    identity_codes = torch.tensor(
        [identity_lookup[identity] for identity in identities],
        dtype=torch.long, device=frames.device,
    )
    labels = (identity_codes[left] == identity_codes[right]).to(torch.float32)
    if negative_positive_ratio is not None and bool((labels > 0).any()):
        positive = torch.where(labels > 0)[0]
        negative = torch.where(labels == 0)[0]
        limit = min(len(negative), max(1, round(len(positive) * negative_positive_ratio)))
        if limit < len(negative):
            order = torch.randperm(len(negative), generator=generator, device=negative.device)[:limit]
            negative = negative[order]
        selected = torch.cat((positive, negative)).sort().values
        left, right, labels = left[selected], right[selected], labels[selected]
    return left, right, labels


def candidate_pairs(frames: Tensor, max_frame_gap: int) -> tuple[Tensor, Tensor]:
    count = int(frames.numel())
    left, right = torch.triu_indices(count, count, offset=1, device=frames.device)
    gaps = frames[right] - frames[left]
    keep = (gaps > 0) & (gaps <= max_frame_gap)
    return left[keep], right[keep]


def identity_triples(frames: Tensor, identities: Sequence[str], max_frame_gap: int) -> Tensor:
    by_identity: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        by_identity.setdefault(identity, []).append(index)
    triples: list[tuple[int, int, int]] = []
    frame_values = frames.detach().cpu().tolist()
    for members in by_identity.values():
        ordered = sorted(members, key=lambda index: frame_values[index])
        for offset in range(len(ordered) - 2):
            a, b, c = ordered[offset : offset + 3]
            if 0 < frame_values[c] - frame_values[a] <= max_frame_gap:
                triples.append((a, b, c))
    if not triples:
        return torch.empty((0, 3), dtype=torch.long, device=frames.device)
    return torch.tensor(triples, dtype=torch.long, device=frames.device)


def token_budget_batches(windows: Sequence[H6Window], max_tokens: int) -> Iterable[list[H6Window]]:
    current: list[H6Window] = []
    tokens = 0
    for window in windows:
        size = len(window.indices)
        if size > max_tokens:
            raise ValueError(f"H6 window {window.key} has {size} observations, above max_tokens_per_batch={max_tokens}")
        if current and tokens + size > max_tokens:
            yield current
            current, tokens = [], 0
        current.append(window)
        tokens += size
    if current:
        yield current
