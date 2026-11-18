import math

import pytest
import torch

from mini_gpt.models.attention import CausalSelfAttention, MultiHeadCausalSelfAttention
from mini_gpt.models.educational.attention_stages import (
    EducationalQKVAttention,
    causal_average_weights,
    explicit_context_average,
    matrix_context_average,
    softmax_weighted_aggregation,
)


def test_explicit_and_matrix_averaging_match_prefixes_and_preserve_gradients():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_()
    explicit = explicit_context_average(x)
    matrix, weights = matrix_context_average(x)
    assert explicit.shape == matrix.shape == x.shape
    assert torch.allclose(explicit, matrix)
    assert torch.allclose(explicit[:, 1], (x[:, 0] + x[:, 1]) / 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3))
    assert torch.equal(weights.triu(1), torch.zeros_like(weights))
    explicit.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_averaging_has_no_future_influence_and_handles_single_token():
    x = torch.randn(2, 4, 3)
    changed = x.clone()
    changed[:, 3] += 1000
    assert torch.allclose(
        explicit_context_average(x)[:, :3], explicit_context_average(changed)[:, :3]
    )
    assert torch.allclose(matrix_context_average(x[:, :1])[0], x[:, :1])
    assert torch.allclose(
        causal_average_weights(1, device=x.device, dtype=x.dtype), torch.ones(1, 1)
    )


def test_softmax_stage_masks_future_and_normalizes_rows():
    x = torch.randn(2, 4, 3)
    output, weights, scores = softmax_weighted_aggregation(x)
    assert output.shape == x.shape and scores.shape == weights.shape == (2, 4, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4))
    assert torch.equal(weights.triu(1), torch.zeros_like(weights).triu(1))
    assert torch.isfinite(output).all()


def test_educational_qkv_has_causal_weights_and_gradients():
    module = EducationalQKVAttention(6)
    x = torch.randn(2, 4, 6, requires_grad=True)
    output, weights, scores = module(x)
    assert output.shape == x.shape and weights.shape == scores.shape == (2, 4, 4)
    assert torch.equal(weights.triu(1), torch.zeros_like(weights).triu(1))
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("module", [CausalSelfAttention(8), MultiHeadCausalSelfAttention(8, 2)])
def test_final_attention_shapes_weights_and_causal_invariance(module):
    x = torch.randn(2, 5, 8, requires_grad=True)
    output, weights = module(x, return_attention=True)
    changed = x.detach().clone()
    changed[:, 3:] += 1000
    changed_output = module(changed)
    assert output.shape == x.shape
    assert torch.allclose(output[:, :3], changed_output[:, :3], atol=1e-5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    assert torch.equal(weights.triu(1), torch.zeros_like(weights).triu(1))
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_single_head_scale_and_dropout_modes():
    module = CausalSelfAttention(4, attention_dropout=0.8)
    assert module.scale == math.sqrt(4)
    x = torch.randn(2, 3, 4)
    module.eval()
    assert torch.equal(module(x), module(x))
    module.train()
    assert torch.isfinite(module(x)).all()


def test_multihead_dimensions_and_parameter_update():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadCausalSelfAttention(6, 4)
    module = MultiHeadCausalSelfAttention(8, 2)
    x = torch.randn(2, 3, 8)
    before = module.qkv.weight.detach().clone()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    loss = module(x).square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, module.qkv.weight)


@pytest.mark.parametrize(("batch", "time"), [(1, 1), (3, 4)])
@pytest.mark.parametrize("module", [CausalSelfAttention(8), MultiHeadCausalSelfAttention(8, 2)])
def test_final_attention_supports_multiple_valid_batch_and_sequence_sizes(module, batch, time):
    output = module(torch.randn(batch, time, 8))
    assert output.shape == (batch, time, 8)
