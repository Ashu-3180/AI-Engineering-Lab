"""Model-ready next-token batch sampling within one split only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .config import DataConfig


def make_batch(
    token_ids: list[int], *, batch_size: int, context_length: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicit torch.Generator")
    if len(token_ids) < context_length + 1:
        raise ValueError("Split has insufficient tokens for a context and its shifted target")
    starts = torch.randint(len(token_ids) - context_length, (batch_size,), generator=generator)
    inputs = torch.tensor(
        [token_ids[start : start + context_length] for start in starts.tolist()], dtype=torch.long
    )
    targets = torch.tensor(
        [token_ids[start + 1 : start + context_length + 1] for start in starts.tolist()],
        dtype=torch.long,
    )
    return inputs, targets


def make_seeded_generator(seed: int) -> torch.Generator:
    """Create the explicit CPU generator used for reproducible batch sampling."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    return torch.Generator().manual_seed(seed)


def make_batch_from_config(
    token_ids: list[int], *, config: DataConfig, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a batch using the configured dimensions and seed unless a generator is supplied."""
    return make_batch(
        token_ids,
        batch_size=config.default_batch_size,
        context_length=config.default_context_length,
        generator=generator if generator is not None else make_seeded_generator(config.seed),
    )
