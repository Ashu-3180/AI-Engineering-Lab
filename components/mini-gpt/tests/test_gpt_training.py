from dataclasses import replace

import pytest
import torch

from mini_gpt.gpt_generation import generate_gpt_text
from mini_gpt.gpt_training import (
    TrainingConfig,
    evaluate,
    load_checkpoint,
    make_checkpoint,
    perplexity,
    save_checkpoint_atomic,
    train_gpt,
)
from mini_gpt.models.gpt import GPTConfig, GPTLanguageModel
from mini_gpt.tokenizer import CharacterTokenizer


def training_config(iterations=4):
    return TrainingConfig(
        "smoke", iterations, 0.01, 0.0, 0.9, 0.95, 1.0, 2, 4, 2, 2, 9, "cpu", 2, 5
    )


def test_training_evaluation_checkpoint_and_resume(tmp_path):
    model_config = GPTConfig(4, 4, 8, 2, 1, 0.0)
    model = GPTLanguageModel(model_config)
    tokens = [0, 1, 2, 3] * 10
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer, history = train_gpt(model, tokens, tokens, training_config())
    assert len(history) == 2 and all(
        torch.isfinite(torch.tensor(row["validation_loss"])) for row in history
    )
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
    checkpoint = tmp_path / "checkpoint.pt"
    payload = make_checkpoint(
        model,
        optimizer,
        4,
        model_config,
        training_config(),
        "checksum",
        4,
        torch.Generator().manual_seed(1),
        torch.Generator().manual_seed(2),
        1.0,
        history,
    )
    save_checkpoint_atomic(checkpoint, payload)
    restored = GPTLanguageModel(model_config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    loaded = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        dataset_checksum="checksum",
        vocabulary_size=4,
    )
    assert loaded["iteration"] == 4
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.parameters(), restored.parameters(), strict=True)
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(
            checkpoint,
            model=restored,
            optimizer=restored_optimizer,
            dataset_checksum="wrong",
            vocabulary_size=4,
        )


def test_evaluation_does_not_update_and_generation_is_reproducible():
    model = GPTLanguageModel(GPTConfig(3, 3, 6, 2, 1, 0.0))
    config = replace(training_config(), context_length=3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = evaluate(model, [0, 1, 2] * 8, config, torch.Generator().manual_seed(1))
    assert perplexity(loss) > 0
    assert all(
        torch.equal(left, right) for left, right in zip(before, model.parameters(), strict=True)
    )
    tokenizer = CharacterTokenizer.from_text("abc")
    first = generate_gpt_text(model, tokenizer, prompt="ab", max_new_tokens=4, seed=3)
    assert first == generate_gpt_text(model, tokenizer, prompt="ab", max_new_tokens=4, seed=3)
    assert len(generate_gpt_text(model, tokenizer, prompt="", max_new_tokens=4, seed=3)) == 4


def test_invalid_training_configuration_is_rejected():
    with pytest.raises(ValueError):
        train_gpt(
            GPTLanguageModel(GPTConfig(3, 2, 4, 2, 1, 0.0)),
            [0, 1, 2],
            [0, 1, 2],
            replace(training_config(), device="cuda"),
        )
