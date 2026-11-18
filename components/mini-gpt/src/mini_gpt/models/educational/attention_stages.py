"""Inspectable progression from fixed causal averages to learned Q/K/V attention."""

from __future__ import annotations

import torch
from torch import nn


def _validate_input(x: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, time, channels]")


def causal_average_weights(
    length: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return a [T, T] lower-triangular matrix whose rows each sum to one."""
    if length <= 0:
        raise ValueError("sequence length must be positive")
    weights = torch.tril(torch.ones((length, length), device=device, dtype=dtype))
    return weights / weights.sum(dim=-1, keepdim=True)


def explicit_context_average(x: torch.Tensor) -> torch.Tensor:
    """Stage 1: each position is the explicit average of its causal prefix."""
    _validate_input(x)
    _, time, _ = x.shape
    output = torch.empty_like(x)
    for position in range(time):
        output[:, position] = x[:, : position + 1].mean(dim=1)
    return output


def matrix_context_average(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 2: express the same causal prefix average as weights @ x."""
    _validate_input(x)
    weights = causal_average_weights(x.shape[1], device=x.device, dtype=x.dtype)
    return weights @ x, weights


def softmax_weighted_aggregation(
    x: torch.Tensor, scores: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage 3: causally masked softmax weights make aggregation data-dependent."""
    _validate_input(x)
    batch, time, _ = x.shape
    if scores is None:
        scores = x @ x.transpose(-2, -1)
    if scores.shape != (batch, time, time):
        raise ValueError("scores must have shape [batch, time, time]")
    mask = torch.tril(torch.ones((time, time), device=x.device, dtype=torch.bool))
    weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    return weights @ x, weights, scores


class EducationalQKVAttention(nn.Module):
    """Stage 4: queries seek keys, and normalized matches aggregate values."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_input(x)
        queries, keys, values = self.query(x), self.key(x), self.value(x)
        scores = queries @ keys.transpose(-2, -1)
        mask = torch.tril(torch.ones(scores.shape[-2:], device=x.device, dtype=torch.bool))
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        return weights @ values, weights, scores
