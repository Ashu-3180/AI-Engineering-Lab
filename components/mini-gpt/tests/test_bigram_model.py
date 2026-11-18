import pytest
import torch
from torch.nn import functional as F

from mini_gpt.models.bigram import BigramLanguageModel


def test_bigram_output_shape_loss_and_vocabulary_dimension():
    model = BigramLanguageModel(5)
    inputs = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3], [4, 0, 1]], dtype=torch.long)
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 3, 5)
    assert loss is not None and torch.isfinite(loss)
    assert torch.allclose(loss, F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1)))
    assert model.parameter_count == 25


def test_bigram_gradients_propagate_and_optimizer_updates_parameters():
    model = BigramLanguageModel(3)
    inputs = torch.tensor([[0, 1]], dtype=torch.long)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    before = model.token_embedding_table.weight.detach().clone()
    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()
    assert model.token_embedding_table.weight.grad is not None
    torch.optim.AdamW(model.parameters(), lr=0.1).step()
    assert not torch.equal(before, model.token_embedding_table.weight)


@pytest.mark.parametrize(
    "inputs,targets",
    [
        (torch.tensor([0, 1], dtype=torch.long), None),
        (torch.tensor([[0, 1]], dtype=torch.int32), None),
        (torch.tensor([[0, 1]], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)),
        (torch.tensor([[0, 1]], dtype=torch.long), torch.tensor([[0, 1]], dtype=torch.int32)),
    ],
)
def test_bigram_rejects_invalid_input_or_target_shapes(inputs, targets):
    with pytest.raises(ValueError):
        BigramLanguageModel(3)(inputs, targets)
