"""Trainable global frame-context association model and trajectory selector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


TRAJECTORY_FEATURE_NAMES = (
    "log_length", "temporal_coverage", "mean_edge_probability",
    "minimum_edge_probability", "mean_edge_margin", "minimum_edge_margin",
    "mean_spatial_step", "baseline_track_count", "neural_track_count",
    "partition_disagreement_fraction",
)


@dataclass(frozen=True)
class H6ModelSpec:
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    feedforward_dim: int
    dropout: float
    window_length: int


class GlobalTrajectoryReasoner(nn.Module):
    """Efficient long-window reasoner: observation tokens plus global frame context."""

    def __init__(self, spec: H6ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.appearance = nn.Sequential(
            nn.Linear(spec.embedding_dim, spec.hidden_dim), nn.LayerNorm(spec.hidden_dim),
            nn.GELU(), nn.Dropout(spec.dropout),
        )
        self.geometry = nn.Sequential(
            nn.Linear(6, spec.hidden_dim), nn.GELU(), nn.Linear(spec.hidden_dim, spec.hidden_dim),
        )
        self.frame_position = nn.Embedding(spec.window_length, spec.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=spec.hidden_dim, nhead=spec.num_heads,
            dim_feedforward=spec.feedforward_dim, dropout=spec.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.frame_reasoner = nn.TransformerEncoder(
            layer, num_layers=spec.num_layers, norm=nn.LayerNorm(spec.hidden_dim)
        )
        self.output_norm = nn.LayerNorm(spec.hidden_dim)
        self.pair_head = nn.Sequential(
            nn.Linear(4 * spec.hidden_dim + 7, spec.hidden_dim), nn.GELU(),
            nn.Dropout(spec.dropout), nn.Linear(spec.hidden_dim, spec.hidden_dim // 2),
            nn.GELU(), nn.Linear(spec.hidden_dim // 2, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def encode(
        self,
        embeddings: Tensor,
        geometry: Tensor,
        frame_index: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        """Encode padded windows without quadratic attention over observations."""
        if embeddings.ndim != 3 or geometry.shape[:2] != embeddings.shape[:2]:
            raise ValueError("H6 window tensors must have shapes [batch,tokens,*]")
        if geometry.shape[-1] != 6 or frame_index.shape != embeddings.shape[:2]:
            raise ValueError("H6 geometry/frame index shapes are inconsistent")
        if token_mask.shape != embeddings.shape[:2] or token_mask.dtype != torch.bool:
            raise ValueError("H6 token_mask must be boolean [batch,tokens]")
        frames = self.spec.window_length
        safe_frame = frame_index.clamp(0, frames - 1)
        token = self.appearance(embeddings) + self.geometry(geometry)
        one_hot = functional.one_hot(safe_frame, num_classes=frames).to(token.dtype)
        one_hot = one_hot * token_mask.unsqueeze(-1)
        frame_sum = torch.einsum("bnf,bnh->bfh", one_hot, token)
        frame_count = one_hot.sum(dim=1).clamp_min(1.0)
        frame_tokens = frame_sum / frame_count.unsqueeze(-1)
        positions = torch.arange(frames, device=token.device).unsqueeze(0)
        frame_tokens = frame_tokens + self.frame_position(positions)
        frame_present = one_hot.sum(dim=1) > 0
        # A fully padded sample is rejected by data collation, but unmask position
        # zero defensively so Transformer never receives an all-masked row.
        all_empty = ~frame_present.any(dim=1)
        if bool(all_empty.any()):
            frame_present = frame_present.clone()
            frame_present[all_empty, 0] = True
        context = self.frame_reasoner(frame_tokens, src_key_padding_mask=~frame_present)
        gathered = context.gather(
            1, safe_frame.unsqueeze(-1).expand(-1, -1, self.spec.hidden_dim)
        )
        output = self.output_norm(token + gathered)
        return output * token_mask.unsqueeze(-1)

    def pair_logits(
        self,
        encoded: Tensor,
        geometry: Tensor,
        frame_index: Tensor,
        left: Tensor,
        right: Tensor,
    ) -> Tensor:
        """Score arbitrary all-pair candidate edges for one encoded window."""
        if encoded.ndim != 2 or left.ndim != 1 or right.shape != left.shape:
            raise ValueError("H6 pair scoring expects one [tokens,hidden] window")
        a, b = encoded[left], encoded[right]
        ga, gb = geometry[left], geometry[right]
        delta_geometry = gb - ga
        frame_delta = (frame_index[right] - frame_index[left]).to(encoded.dtype).unsqueeze(-1)
        relative = torch.cat((delta_geometry, frame_delta), dim=-1)
        features = torch.cat((a, b, torch.abs(a - b), a * b, relative), dim=-1)
        learned = self.pair_head(features).squeeze(-1)
        cosine = functional.cosine_similarity(a, b, dim=-1)
        return learned + self.logit_scale.exp().clamp(max=100.0) * cosine


class TrajectorySelector(nn.Module):
    """Learns whether a complete proposed trajectory should replace baseline rows."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(len(TRAJECTORY_FEATURE_NAMES)))
        self.register_buffer("feature_scale", torch.ones(len(TRAJECTORY_FEATURE_NAMES)))
        self.network = nn.Sequential(
            nn.Linear(len(TRAJECTORY_FEATURE_NAMES), hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_normalization(self, values: Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != len(TRAJECTORY_FEATURE_NAMES):
            raise ValueError("Invalid selector feature matrix")
        with torch.no_grad():
            self.feature_mean.copy_(values.mean(dim=0))
            self.feature_scale.copy_(values.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, values: Tensor) -> Tensor:
        return self.network((values - self.feature_mean) / self.feature_scale).squeeze(-1)


def balanced_edge_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.shape != labels.shape or logits.numel() == 0:
        raise ValueError("H6 association loss requires aligned non-empty edges")
    positives = labels.sum()
    negatives = labels.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 1000.0)
    return functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight)


def supervised_contrastive_pair_loss(encoded: Tensor, left: Tensor, right: Tensor, labels: Tensor) -> Tensor:
    cosine = functional.cosine_similarity(encoded[left], encoded[right], dim=-1)
    positive = labels * (1.0 - cosine)
    negative = (1.0 - labels) * functional.relu(cosine - 0.2)
    return (positive + negative).mean()


def cycle_consistency_loss(
    model: GlobalTrajectoryReasoner,
    encoded: Tensor,
    geometry: Tensor,
    frames: Tensor,
    triples: Tensor,
) -> Tensor:
    if triples.numel() == 0:
        return encoded.sum() * 0.0
    first = model.pair_logits(encoded, geometry, frames, triples[:, 0], triples[:, 1]).sigmoid()
    second = model.pair_logits(encoded, geometry, frames, triples[:, 1], triples[:, 2]).sigmoid()
    direct = model.pair_logits(encoded, geometry, frames, triples[:, 0], triples[:, 2]).sigmoid()
    return functional.mse_loss(direct, first * second)


def model_parameter_report(module: nn.Module) -> dict[str, int]:
    trainable = sum(value.numel() for value in module.parameters() if value.requires_grad)
    return {"trainable_parameters": trainable, "total_parameters": sum(value.numel() for value in module.parameters())}
