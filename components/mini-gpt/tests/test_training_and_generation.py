import pytest
import torch

from mini_gpt.experiment import BigramConfig, load_bigram_config
from mini_gpt.generation import generate_text
from mini_gpt.models.bigram import BigramLanguageModel
from mini_gpt.tokenizer import CharacterTokenizer
from mini_gpt.training import evaluate_loss, train_bigram


def small_config() -> BigramConfig:
    return BigramConfig(
        experiment_id="test",
        device="cpu",
        iterations=8,
        learning_rate=0.1,
        weight_decay=0.0,
        evaluation_interval=4,
        evaluation_batches=2,
        batch_size=2,
        context_length=2,
        seed=7,
    )


def test_short_training_is_finite_reproducible_and_updates_parameters():
    tokens = [0, 1, 2, 0, 1, 2] * 10
    first = BigramLanguageModel(3)
    second = BigramLanguageModel(3)
    torch.manual_seed(2)
    first.token_embedding_table.reset_parameters()
    second.load_state_dict(first.state_dict())
    before = first.token_embedding_table.weight.detach().clone()
    result = train_bigram(first, tokens, tokens, small_config())
    repeated = train_bigram(second, tokens, tokens, small_config())
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (result.final_train_loss, result.validation_loss)
    )
    assert result == repeated
    assert not torch.equal(before, first.token_embedding_table.weight)


def test_validation_evaluation_does_not_modify_parameters():
    model = BigramLanguageModel(3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = evaluate_loss(
        model,
        [0, 1, 2, 0, 1, 2],
        config=small_config(),
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.isfinite(torch.tensor(loss))
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))


def test_invalid_training_configuration_is_rejected(tmp_path):
    path = tmp_path / "bigram.toml"
    path.write_text(
        """[experiment]
id = "x"
device = "cuda"
[training]
iterations = 1
learning_rate = 1
weight_decay = 0
evaluation_interval = 1
evaluation_batches = 1
batch_size = 1
context_length = 1
seed = 1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CPU"):
        load_bigram_config(path)


def test_generation_handles_prompts_empty_input_and_seeded_reproducibility():
    tokenizer = CharacterTokenizer.from_text("abc")
    model = BigramLanguageModel(tokenizer.vocabulary_size)
    grad_enabled: list[bool] = []
    hook = model.register_forward_pre_hook(lambda *_: grad_enabled.append(torch.is_grad_enabled()))
    first = generate_text(model, tokenizer, prompt="ab", max_new_tokens=5, seed=4, context_length=2)
    hook.remove()
    second = generate_text(
        model, tokenizer, prompt="ab", max_new_tokens=5, seed=4, context_length=2
    )
    empty = generate_text(model, tokenizer, prompt="", max_new_tokens=5, seed=4, context_length=2)
    assert first == second
    assert first.startswith("ab") and len(first) == 7
    assert len(empty) == 5 and set(empty).issubset({"a", "b", "c"})
    assert grad_enabled and not any(grad_enabled)
    with pytest.raises(ValueError, match="Unknown character"):
        generate_text(model, tokenizer, prompt="z", max_new_tokens=1, seed=4, context_length=2)
    with pytest.raises(ValueError, match="non-negative"):
        generate_text(model, tokenizer, prompt="", max_new_tokens=-1, seed=4, context_length=2)
