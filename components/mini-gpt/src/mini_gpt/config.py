"""Loading and validation for the small, explicit data configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str
    source_url: str
    raw_data_dir: Path
    processed_data_dir: Path
    provenance_dir: Path
    train_ratio: float
    default_context_length: int
    default_batch_size: int
    seed: int


def load_data_config(path: Path) -> DataConfig:
    """Load a TOML data configuration and reject invalid values early."""
    try:
        with path.open("rb") as config_file:
            values = tomllib.load(config_file)["data"]
    except FileNotFoundError as error:
        raise ValueError(f"Configuration file does not exist: {path}") from error
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Configuration must contain a valid [data] table: {path}") from error

    try:
        config = DataConfig(
            dataset_id=values["dataset_id"],
            source_url=values["source_url"],
            raw_data_dir=Path(values["raw_data_dir"]),
            processed_data_dir=Path(values["processed_data_dir"]),
            provenance_dir=Path(values["provenance_dir"]),
            train_ratio=values["train_ratio"],
            default_context_length=values["default_context_length"],
            default_batch_size=values["default_batch_size"],
            seed=values["seed"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Configuration has missing or invalid settings: {path}") from error
    if not isinstance(config.dataset_id, str) or not isinstance(config.source_url, str):
        raise ValueError("dataset_id and source_url must be strings")
    if not config.dataset_id or not config.source_url.startswith("https://"):
        raise ValueError("dataset_id and an HTTPS source_url are required")
    if not isinstance(config.train_ratio, (int, float)) or isinstance(config.train_ratio, bool):
        raise ValueError("train_ratio must be numeric")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")
    if (
        not isinstance(config.default_context_length, int)
        or isinstance(config.default_context_length, bool)
        or not isinstance(config.default_batch_size, int)
        or isinstance(config.default_batch_size, bool)
    ):
        raise ValueError("default context length and batch size must be integers")
    if config.default_context_length <= 0 or config.default_batch_size <= 0:
        raise ValueError("default context length and batch size must be positive")
    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        raise ValueError("seed must be an integer")
    return config


def resolve_component_path(component_root: Path, configured_path: Path) -> Path:
    """Resolve a configured path only when it remains inside the component root."""
    if configured_path.is_absolute():
        raise ValueError("Configured data paths must be relative to the component root")
    root = component_root.resolve()
    resolved = (root / configured_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Configured data path escapes the component root")
    return resolved
