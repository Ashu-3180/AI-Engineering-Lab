"""Reusable scaled causal single- and multi-head self-attention modules."""

from __future__ import annotations

import math

import torch
from torch import nn


def _validate_attention_input(x: torch.Tensor, embedding_dim: int) -> None:
    if x.ndim != 3 or x.shape[-1] != embedding_dim:
        raise ValueError(f"x must have shape [batch, time, {embedding_dim}]")


def _causal_mask(time: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones((time, time), device=device, dtype=torch.bool))


class CausalSelfAttention(nn.Module):
    """Stage 5: scaled dot-product causal attention over one C-dimensional head."""

    def __init__(self, embedding_dim: int, attention_dropout: float = 0.0) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        self.embedding_dim = embedding_dim
        self.scale = math.sqrt(embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, x: torch.Tensor, *, return_attention: bool = False):
        _validate_attention_input(x, self.embedding_dim)
        queries, keys, values = self.query(x), self.key(x), self.value(x)
        scores = (queries @ keys.transpose(-2, -1)) / self.scale
        mask = _causal_mask(x.shape[1], x.device)
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        output = self.attention_dropout(weights) @ values
        if return_attention:
            return output, weights
        return output


class MultiHeadCausalSelfAttention(nn.Module):
    """Stage 6: split C channels into heads, attend independently, then recombine."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be positive and divisible by num_heads")
        if not 0.0 <= attention_dropout < 1.0 or not 0.0 <= output_dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x: torch.Tensor, *, return_attention: bool = False):
        _validate_attention_input(x, self.embedding_dim)
        batch, time, _ = x.shape
        qkv = self.qkv(x).reshape(batch, time, 3, self.num_heads, self.head_dim)
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4)
        scores = (queries @ keys.transpose(-2, -1)) / self.scale
        mask = _causal_mask(time, x.device)
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        heads = self.attention_dropout(weights) @ values
        combined = heads.transpose(1, 2).reshape(batch, time, self.embedding_dim)
        output = self.output_dropout(self.output(combined))
        if return_attention:
            return output, weights
        return output
