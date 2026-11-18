"""A minimal character-level bigram language model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Maps each token directly to next-token logits through one embedding table."""

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        if (
            not isinstance(vocabulary_size, int)
            or isinstance(vocabulary_size, bool)
            or vocabulary_size <= 0
        ):
            raise ValueError("vocabulary_size must be a positive integer")
        self.vocabulary_size = vocabulary_size
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch, time]")
        if idx.dtype != torch.long:
            raise ValueError("idx must use torch.long token IDs")
        logits = self.token_embedding_table(idx)
        if targets is None:
            return logits, None
        if targets.ndim != 2 or targets.shape != idx.shape:
            raise ValueError("targets must have the same [batch, time] shape as idx")
        if targets.dtype != torch.long:
            raise ValueError("targets must use torch.long token IDs")
        loss = F.cross_entropy(logits.reshape(-1, self.vocabulary_size), targets.reshape(-1))
        return logits, loss

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
