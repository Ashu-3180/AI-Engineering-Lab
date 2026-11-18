"""Bigram experiment configuration and machine-readable result handling."""

from __future__ import annotations

import json
import platform
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .config import DataConfig, load_data_config
from .corpus import load_provenance, read_corpus, resolve_dataset_paths
from .dataset import split_from_config
from .generation import generate_text
from .models.bigram import BigramLanguageModel
from .tokenizer import CharacterTokenizer
from .training import train_bigram


@dataclass(frozen=True)
class BigramConfig:
    experiment_id: str
    device: str
    iterations: int
    learning_rate: float
    weight_decay: float
    evaluation_interval: int
    evaluation_batches: int
    batch_size: int
    context_length: int
    seed: int


def load_bigram_config(path: Path) -> BigramConfig:
    """Load the dedicated bigram experiment TOML file."""
    try:
        with path.open("rb") as config_file:
            values = tomllib.load(config_file)
        experiment = values["experiment"]
        training = values["training"]
        config = BigramConfig(
            experiment_id=experiment["id"],
            device=experiment["device"],
            iterations=training["iterations"],
            learning_rate=training["learning_rate"],
            weight_decay=training["weight_decay"],
            evaluation_interval=training["evaluation_interval"],
            evaluation_batches=training["evaluation_batches"],
            batch_size=training["batch_size"],
            context_length=training["context_length"],
            seed=training["seed"],
        )
    except FileNotFoundError as error:
        raise ValueError(f"Bigram configuration file does not exist: {path}") from error
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Bigram configuration is invalid: {path}") from error
    validate_bigram_config(config)
    return config


def validate_bigram_config(config: BigramConfig) -> None:
    if not isinstance(config.experiment_id, str) or not config.experiment_id:
        raise ValueError("experiment_id must be a non-empty string")
    if config.device != "cpu":
        raise ValueError("M2 supports only the CPU device")
    integer_fields = (
        config.iterations,
        config.evaluation_interval,
        config.evaluation_batches,
        config.batch_size,
        config.context_length,
        config.seed,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_fields):
        raise ValueError("integer training settings must be integers")
    if (
        config.iterations <= 0
        or config.evaluation_interval <= 0
        or config.evaluation_batches <= 0
        or config.batch_size <= 0
        or config.context_length <= 0
    ):
        raise ValueError("iterations, evaluation, batch, and context settings must be positive")
    if not isinstance(config.learning_rate, (int, float)) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not isinstance(config.weight_decay, (int, float)) or config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")


def run_bigram_experiment(
    project_root: Path, config: BigramConfig, data_config: DataConfig
) -> dict[str, object]:
    """Run M2 end-to-end using only the established M1 corpus pipeline."""
    start = time.perf_counter()
    paths = resolve_dataset_paths(
        project_root,
        data_config.dataset_id,
        raw_data_dir=data_config.raw_data_dir,
        provenance_dir=data_config.provenance_dir,
    )
    text = read_corpus(
        paths.raw_file, paths.provenance_file, expected_dataset_id=data_config.dataset_id
    )
    tokenizer = CharacterTokenizer.from_text(text)
    split = split_from_config(tokenizer.encode(text), data_config)
    model = BigramLanguageModel(tokenizer.vocabulary_size)
    result = train_bigram(model, split.train, split.validation, config)
    provenance = load_provenance(paths.provenance_file)
    sample = generate_text(
        model,
        tokenizer,
        prompt="To be",
        max_new_tokens=100,
        seed=config.seed,
        context_length=config.context_length,
    )
    return {
        "experiment_id": config.experiment_id,
        "dataset_checksum": provenance.sha256,
        "tokenizer_vocabulary_size": tokenizer.vocabulary_size,
        "model_parameter_count": model.parameter_count,
        "model_configuration": {"type": "bigram", "vocabulary_size": tokenizer.vocabulary_size},
        "training_configuration": asdict(config),
        "random_seed": config.seed,
        "training": result.to_metadata(),
        "runtime_seconds": time.perf_counter() - start,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "execution_device": config.device,
        "generated_sample": sample,
    }


def write_experiment_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_config = load_data_config(project_root / "config" / "data.toml")
    config = load_bigram_config(project_root / "config" / "bigram.toml")
    result = run_bigram_experiment(project_root, config, data_config)
    write_experiment_result(project_root / "outputs" / "bigram-baseline.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
