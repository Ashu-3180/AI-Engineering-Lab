from pathlib import Path

import pytest
import torch

from mini_gpt.gpt_config import load_gpt_preset
from mini_gpt.models.gpt import FeedForward, GPTConfig, GPTLanguageModel, TransformerBlock


def config(dropout=0.0):
    return GPTConfig(vocab_size=11, block_size=6, n_embd=8, n_head=2, n_layer=2, dropout=dropout)


def test_embeddings_logits_loss_and_gradients():
    model = GPTLanguageModel(config())
    idx = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    logits, loss = model(idx, idx)
    token_embeddings = model.token_embedding_table(idx)
    position_embeddings = model.position_embedding_table(torch.arange(3))
    assert token_embeddings.shape == (2, 3, 8)
    assert position_embeddings.shape == (3, 8)
    assert (token_embeddings + position_embeddings).shape == (2, 3, 8)
    assert logits.shape == (2, 3, 11) and loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model.parameter_count > 0


@pytest.mark.parametrize(
    "idx,targets",
    [
        (torch.tensor([0, 1]), None),
        (torch.zeros((1, 7), dtype=torch.long), None),
        (torch.tensor([[11]], dtype=torch.long), None),
        (torch.tensor([[1]], dtype=torch.int32), None),
        (torch.tensor([[1]], dtype=torch.long), torch.tensor([1], dtype=torch.long)),
    ],
)
def test_gpt_rejects_invalid_inputs(idx, targets):
    with pytest.raises(ValueError):
        GPTLanguageModel(config())(idx, targets)


def test_complete_gpt_is_causal_in_evaluation_mode():
    model = GPTLanguageModel(config()).eval()
    idx = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    changed = idx.clone()
    changed[:, 3:] = torch.tensor([6, 7])
    first, _ = model(idx)
    second, _ = model(changed)
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-6)


def test_feedforward_and_block_preserve_shape_and_residuals():
    x = torch.randn(2, 3, 8, requires_grad=True)
    feed_forward = FeedForward(8, 0.5)
    assert feed_forward.net[0].out_features == 32
    feed_forward.eval()
    assert torch.equal(feed_forward(x), feed_forward(x))
    block = TransformerBlock(config())
    output = block(x)
    assert output.shape == x.shape and not torch.equal(output, x)
    output.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_named_presets_and_configuration_validation(tmp_path):
    path = Path(__file__).parents[1] / "config" / "gpt.toml"
    smoke = load_gpt_preset(path, "smoke", tokenizer_vocab_size=65)
    experiment = load_gpt_preset(path, "experiment", tokenizer_vocab_size=65)
    assert (smoke.model.n_embd, smoke.batch_size) == (64, 8)
    assert (experiment.model.n_embd, experiment.batch_size) == (128, 16)
    with pytest.raises(ValueError, match="vocab_size"):
        load_gpt_preset(path, "smoke", tokenizer_vocab_size=64)
    bad = tmp_path / "bad.toml"
    bad.write_text("[x]\nvocab_size=65\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_gpt_preset(bad, "x", tokenizer_vocab_size=65)
    with pytest.raises(ValueError, match="divisible"):
        GPTLanguageModel(GPTConfig(11, 6, 7, 2, 1, 0.0))
    with pytest.raises(ValueError, match="dropout"):
        GPTLanguageModel(GPTConfig(11, 6, 8, 2, 1, 1.0))
