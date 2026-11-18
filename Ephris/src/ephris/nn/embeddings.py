from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers.weights import Embedding


class LabelEmbedding(nn.Module):
    """Checkpoint-backed class embeddings for observed context labels."""

    def __init__(self, *, num_classes: int, embedding_dim: int):
        super().__init__()
        if int(num_classes) <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if int(embedding_dim) <= 0:
            raise ValueError(f"embedding_width must be positive, got {embedding_dim}")
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.embedding = Embedding(self.num_classes, self.embedding_dim)

    def forward(self, y: Tensor) -> Tensor:
        return self.embedding(y.view(-1).long())


def embed_observed_labels(
    embedding: LabelEmbedding,
    y: Tensor,
    context_mask: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    y = y.view(-1).long().to(device=device)
    context_mask = context_mask.view(-1).to(device=device, dtype=torch.bool)
    if int(context_mask.numel()) != int(y.numel()):
        raise ValueError(
            f"context_mask must have length {int(y.numel())}, got {int(context_mask.numel())}"
        )
    label_embeddings = torch.zeros(
        int(y.numel()), embedding.embedding_dim, dtype=dtype, device=device
    )
    if bool(context_mask.any()):
        labeled_y = y[context_mask]
        if (
            int(labeled_y.min().item()) < 0
            or int(labeled_y.max().item()) >= embedding.num_classes
        ):
            raise ValueError(
                f"Encountered class id outside [0, {embedding.num_classes - 1}]."
            )
        label_embeddings[context_mask] = embedding(labeled_y).to(dtype=dtype)
    return label_embeddings
