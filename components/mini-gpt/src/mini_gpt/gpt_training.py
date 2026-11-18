"""Deterministic CPU training, evaluation, and atomic checkpoints for M5 GPT."""

from __future__ import annotations

import math
import random
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .batching import make_batch, make_seeded_generator
from .models.gpt import GPTConfig, GPTLanguageModel


@dataclass(frozen=True)
class TrainingConfig:
    model_preset: str
    iterations: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    gradient_clip: float
    batch_size: int
    context_length: int
    evaluation_interval: int
    evaluation_batches: int
    seed: int
    device: str
    checkpoint_interval: int
    generation_tokens: int


def load_training_config(path: Path, preset: str) -> TrainingConfig:
    try:
        with path.open("rb") as config_file:
            values = tomllib.load(config_file)[preset]
        config = TrainingConfig(
            model_preset=values["model_preset"],
            iterations=values["iterations"],
            learning_rate=values["learning_rate"],
            weight_decay=values["weight_decay"],
            beta1=values["beta1"],
            beta2=values["beta2"],
            gradient_clip=values["gradient_clip"],
            batch_size=values["batch_size"],
            context_length=values["context_length"],
            evaluation_interval=values["evaluation_interval"],
            evaluation_batches=values["evaluation_batches"],
            seed=values["seed"],
            device=values["device"],
            checkpoint_interval=values["checkpoint_interval"],
            generation_tokens=values["generation_tokens"],
        )
    except (FileNotFoundError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Training preset {preset!r} is invalid") from error
    validate_training_config(config)
    return config


def validate_training_config(config: TrainingConfig) -> None:
    if config.device != "cpu":
        raise ValueError("M5 supports CPU execution only")
    if config.iterations <= 0 or config.batch_size <= 0 or config.context_length <= 0:
        raise ValueError("iterations, batch_size, and context_length must be positive")
    if (
        config.evaluation_interval <= 0
        or config.evaluation_batches <= 0
        or config.checkpoint_interval <= 0
    ):
        raise ValueError("evaluation and checkpoint intervals must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.gradient_clip <= 0:
        raise ValueError("optimizer settings are invalid")
    if not 0 <= config.beta1 < 1 or not 0 <= config.beta2 < 1:
        raise ValueError("AdamW betas must be in [0, 1)")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: GPTLanguageModel, tokens: list[int], config: TrainingConfig, generator: torch.Generator
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(config.evaluation_batches):
            x, y = make_batch(
                tokens,
                batch_size=config.batch_size,
                context_length=config.context_length,
                generator=generator,
            )
            _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss):
                raise ValueError("evaluation loss is non-finite")
            losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)


def train_gpt(
    model: GPTLanguageModel,
    train_tokens: list[int],
    validation_tokens: list[int],
    config: TrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    start_iteration: int = 0,
    train_generator: torch.Generator | None = None,
    validation_generator: torch.Generator | None = None,
    on_checkpoint=None,
) -> tuple[torch.optim.Optimizer, list[dict[str, float]]]:
    validate_training_config(config)
    seed_everything(config.seed)
    optimizer = optimizer or torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )
    train_generator = train_generator or make_seeded_generator(config.seed)
    validation_generator = validation_generator or make_seeded_generator(config.seed + 1)
    history: list[dict[str, float]] = []
    for iteration in range(start_iteration + 1, config.iterations + 1):
        model.train()
        x, y = make_batch(
            train_tokens,
            batch_size=config.batch_size,
            context_length=config.context_length,
            generator=train_generator,
        )
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise ValueError("training loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise ValueError("gradient norm is non-finite")
        optimizer.step()
        if iteration % config.evaluation_interval == 0 or iteration == config.iterations:
            validation_loss = evaluate(model, validation_tokens, config, validation_generator)
            history.append(
                {
                    "iteration": float(iteration),
                    "train_loss": loss.item(),
                    "validation_loss": validation_loss,
                }
            )
        if on_checkpoint and (
            iteration % config.checkpoint_interval == 0 or iteration == config.iterations
        ):
            on_checkpoint(iteration, optimizer, train_generator, validation_generator, history)
    return optimizer, history


def perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        raise ValueError("loss must be finite")
    return math.exp(loss)


def save_checkpoint_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def make_checkpoint(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    model_config: GPTConfig,
    training_config: TrainingConfig,
    dataset_checksum: str,
    vocabulary_size: int,
    train_generator: torch.Generator,
    validation_generator: torch.Generator,
    best_validation_loss: float,
    history: list[dict[str, float]],
) -> dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "dataset_checksum": dataset_checksum,
        "vocabulary_size": vocabulary_size,
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "train_rng_state": train_generator.get_state(),
        "validation_rng_state": validation_generator.get_state(),
        "best_validation_loss": best_validation_loss,
        "history": history,
    }


def load_checkpoint(
    path: Path,
    *,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    dataset_checksum: str,
    vocabulary_size: int,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["dataset_checksum"] != dataset_checksum
        or payload["vocabulary_size"] != vocabulary_size
    ):
        raise ValueError("checkpoint dataset or vocabulary is incompatible")
    if payload["model_config"] != asdict(model.config):
        raise ValueError("checkpoint model architecture is incompatible")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    return payload
