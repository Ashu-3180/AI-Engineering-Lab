"""Deterministic sequential split functions for encoded corpora."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DataConfig


@dataclass(frozen=True)
class DatasetSplit:
    train: list[int]
    validation: list[int]
    train_boundary: int
    train_ratio: float


def sequential_split(token_ids: list[int], train_ratio: float = 0.9) -> DatasetSplit:
    if not token_ids:
        raise ValueError("Cannot split an empty encoded corpus")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")
    boundary = math.floor(train_ratio * len(token_ids))
    return DatasetSplit(token_ids[:boundary], token_ids[boundary:], boundary, train_ratio)


def split_from_config(token_ids: list[int], config: DataConfig) -> DatasetSplit:
    """Apply the configured sequential split ratio to encoded corpus tokens."""
    return sequential_split(token_ids, train_ratio=config.train_ratio)
