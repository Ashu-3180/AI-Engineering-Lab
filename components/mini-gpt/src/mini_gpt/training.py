"""Focused CPU training and evaluation for the educational bigram baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .batching import make_batch, make_seeded_generator
from .models.bigram import BigramLanguageModel

if TYPE_CHECKING:
    from .experiment import BigramConfig


@dataclass(frozen=True)
class TrainingResult:
    initial_train_loss: float
    final_train_loss: float
    validation_loss: float
    train_perplexity: float
    validation_perplexity: float
    metric_history: tuple[dict[str, float], ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "initial_train_loss": self.initial_train_loss,
            "final_train_loss": self.final_train_loss,
            "validation_loss": self.validation_loss,
            "train_perplexity": self.train_perplexity,
            "validation_perplexity": self.validation_perplexity,
            "metric_history": list(self.metric_history),
        }


def evaluate_loss(
    model: BigramLanguageModel,
    token_ids: list[int],
    *,
    config: BigramConfig,
    generator: torch.Generator,
) -> float:
    """Evaluate mean next-token cross-entropy without changing model parameters."""
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(config.evaluation_batches):
            inputs, targets = make_batch(
                token_ids,
                batch_size=config.batch_size,
                context_length=config.context_length,
                generator=generator,
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            _require_finite_loss(loss)
            losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)


def train_bigram(
    model: BigramLanguageModel,
    train_tokens: list[int],
    validation_tokens: list[int],
    config: BigramConfig,
) -> TrainingResult:
    """Train the baseline with AdamW and periodic validation evaluation."""
    if config.device != "cpu":
        raise ValueError("M2 supports only CPU execution")
    torch.manual_seed(config.seed)
    model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_generator = make_seeded_generator(config.seed)
    validation_generator = make_seeded_generator(config.seed + 1)
    initial_train_loss = evaluate_loss(
        model, train_tokens, config=config, generator=train_generator
    )
    history: list[dict[str, float]] = []
    for iteration in range(1, config.iterations + 1):
        model.train()
        inputs, targets = make_batch(
            train_tokens,
            batch_size=config.batch_size,
            context_length=config.context_length,
            generator=train_generator,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        _require_finite_loss(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % config.evaluation_interval == 0 or iteration == config.iterations:
            validation_loss = evaluate_loss(
                model, validation_tokens, config=config, generator=validation_generator
            )
            history.append(
                {
                    "iteration": float(iteration),
                    "train_loss": loss.item(),
                    "validation_loss": validation_loss,
                }
            )
    final_train_loss = history[-1]["train_loss"]
    validation_loss = history[-1]["validation_loss"]
    return TrainingResult(
        initial_train_loss=initial_train_loss,
        final_train_loss=final_train_loss,
        validation_loss=validation_loss,
        train_perplexity=math.exp(final_train_loss),
        validation_perplexity=math.exp(validation_loss),
        metric_history=tuple(history),
    )


def _require_finite_loss(loss: torch.Tensor) -> None:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Loss must be a finite scalar")
