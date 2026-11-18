"""TOML loading for named M4 GPT architecture presets."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .models.gpt import GPTConfig, validate_gpt_config


@dataclass(frozen=True)
class GPTPreset:
    model: GPTConfig
    batch_size: int


def load_gpt_preset(path: Path, preset_name: str, *, tokenizer_vocab_size: int) -> GPTPreset:
    try:
        with path.open("rb") as config_file:
            values = tomllib.load(config_file)[preset_name]
        preset = GPTPreset(
            model=GPTConfig(
                vocab_size=values["vocab_size"],
                block_size=values["block_size"],
                n_embd=values["n_embd"],
                n_head=values["n_head"],
                n_layer=values["n_layer"],
                dropout=values["dropout"],
            ),
            batch_size=values["batch_size"],
        )
    except (FileNotFoundError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"GPT preset {preset_name!r} is invalid") from error
    validate_gpt_config(preset.model)
    if preset.model.vocab_size != tokenizer_vocab_size:
        raise ValueError("GPT vocab_size must match the tokenizer vocabulary size")
    if (
        not isinstance(preset.batch_size, int)
        or isinstance(preset.batch_size, bool)
        or preset.batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    return preset
