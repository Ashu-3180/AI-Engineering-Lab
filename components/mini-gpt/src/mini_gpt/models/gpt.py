"""Custom decoder-only Transformer architecture for M4 forward-pass experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .attention import MultiHeadCausalSelfAttention


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float


def validate_gpt_config(config: GPTConfig) -> None:
    values = (config.vocab_size, config.block_size, config.n_embd, config.n_head, config.n_layer)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
        raise ValueError("GPT dimensions must be positive integers")
    if config.n_embd % config.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if not isinstance(config.dropout, (int, float)) or not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")


class FeedForward(nn.Module):
    """Per-token MLP: C -> 4C -> ReLU -> C, with output dropout."""

    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder block: x + attention(ln1(x)); x + ffwd(ln2(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = MultiHeadCausalSelfAttention(
            config.n_embd,
            config.n_head,
            attention_dropout=config.dropout,
            output_dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config.n_embd, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.feed_forward(self.ln2(x))


class GPTLanguageModel(nn.Module):
    """A decoder-only GPT with learned token/position embeddings and causal blocks."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        validate_gpt_config(config)
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layer)))
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_tokens(idx, "idx")
        _, time = idx.shape
        positions = torch.arange(time, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(positions)
        logits = self.lm_head(self.ln_final(self.blocks(x)))
        if targets is None:
            return logits, None
        self._validate_tokens(targets, "targets")
        if targets.shape != idx.shape:
            raise ValueError("targets must have the same [batch, time] shape as idx")
        loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    def _validate_tokens(self, tokens: torch.Tensor, name: str) -> None:
        if tokens.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, time]")
        if tokens.dtype != torch.long:
            raise ValueError(f"{name} must use torch.long token IDs")
        if tokens.shape[1] > self.config.block_size:
            raise ValueError("sequence length exceeds block_size")
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.config.vocab_size):
            raise ValueError(f"{name} contains token IDs outside the vocabulary")
