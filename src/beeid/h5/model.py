"""Small trainable BeeTrackQuery head; frozen crop backbones remain external."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class BeeTrackQuery(nn.Module):
    """Persistent query refresher, short-memory reader, pair scorer and reliability head."""

    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.detection_projection = nn.Sequential(
            nn.Linear(embedding_dim + 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.frame_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.frame_norm = nn.LayerNorm(hidden_dim)
        self.memory_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 4, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1)
        )

    def encode_detections(self, embeddings: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return self.detection_projection(torch.cat([embeddings, geometry], dim=-1))

    def refresh(self, queries: torch.Tensor, detections: torch.Tensor) -> torch.Tensor:
        if queries.numel() == 0 or detections.numel() == 0:
            return queries
        attended, _ = self.frame_attention(
            queries.unsqueeze(0), detections.unsqueeze(0), detections.unsqueeze(0),
            need_weights=False,
        )
        return self.frame_norm(queries + attended.squeeze(0))

    def refresh_batched(
        self,
        queries: torch.Tensor,
        detections: torch.Tensor,
        detection_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Refresh padded transition queries in one attention kernel."""
        if queries.numel() == 0 or detections.numel() == 0:
            return queries
        attended, _ = self.frame_attention(
            queries,
            detections,
            detections,
            key_padding_mask=detection_padding_mask,
            need_weights=False,
        )
        return self.frame_norm(queries + attended)

    def read_memory(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if memory.numel() == 0:
            return query
        attended, _ = self.memory_attention(
            query.reshape(1, 1, -1), memory.unsqueeze(0), memory.unsqueeze(0),
            need_weights=False,
        )
        return self.memory_norm(query + attended.reshape(-1))

    def pair_logits(
        self, queries: torch.Tensor, detections: torch.Tensor, geometry_delta: torch.Tensor
    ) -> torch.Tensor:
        q = queries[:, None, :].expand(-1, detections.shape[0], -1)
        d = detections[None, :, :].expand(queries.shape[0], -1, -1)
        return self.pair_head(torch.cat([q, d, torch.abs(q - d), geometry_delta], dim=-1)).squeeze(-1)

    def pair_logits_batched(
        self,
        queries: torch.Tensor,
        detections: torch.Tensor,
        geometry_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Score every padded query/detection pair for a transition batch."""
        q = queries[:, :, None, :].expand(-1, -1, detections.shape[1], -1)
        d = detections[:, None, :, :].expand(-1, queries.shape[1], -1, -1)
        features = torch.cat([q, d, torch.abs(q - d), geometry_delta], dim=-1)
        return self.pair_head(features).squeeze(-1)

    def reliability_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            margin = torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
            entropy = torch.zeros_like(margin)
            top_probability = torch.sigmoid(logits[:, 0])
        else:
            probabilities = logits.softmax(dim=1)
            top_values = probabilities.topk(2, dim=1).values
            top_probability = top_values[:, 0]
            margin = top_values[:, 0] - top_values[:, 1]
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
            entropy = entropy / math.log(logits.shape[1])
        features = torch.stack([top_probability, margin, 1.0 - entropy], dim=1)
        return self.reliability_head(features).squeeze(1)

    def reliability(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities for inference outside the training BCE."""
        return torch.sigmoid(self.reliability_logits(logits))


@dataclass(frozen=True)
class TrainingBatchResult:
    association_loss: torch.Tensor
    reliability_loss: torch.Tensor
    pair_count: int
    prediction_count: int


def supervised_transition(
    model: BeeTrackQuery,
    previous_tokens: torch.Tensor,
    previous_identities: list[str],
    current_tokens: torch.Tensor,
    current_identities: list[str],
    geometry_delta: torch.Tensor,
) -> TrainingBatchResult | None:
    """Teacher-forced transition; labels affect loss/state construction, never inference."""
    target_by_identity = {identity: index for index, identity in enumerate(current_identities)}
    continuing = [index for index, identity in enumerate(previous_identities) if identity in target_by_identity]
    if not continuing or current_tokens.numel() == 0:
        return None
    query = model.refresh(previous_tokens[continuing], current_tokens)
    logits = model.pair_logits(query, current_tokens, geometry_delta[continuing])
    targets = torch.tensor(
        [target_by_identity[previous_identities[index]] for index in continuing],
        dtype=torch.long,
        device=logits.device,
    )
    association = F.cross_entropy(logits, targets)
    predicted = logits.argmax(dim=1)
    correct = (predicted == targets).to(logits.dtype)
    reliability_logits = model.reliability_logits(logits)
    calibration = F.binary_cross_entropy_with_logits(reliability_logits, correct)
    return TrainingBatchResult(association, calibration, len(continuing) * len(current_identities), len(continuing))
