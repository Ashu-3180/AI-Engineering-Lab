import pytest
import torch

from mini_gpt.batching import make_batch
from mini_gpt.dataset import sequential_split


def test_sequential_split_is_exact_and_deterministic():
    split = sequential_split(list(range(10)))
    assert split.train == list(range(9))
    assert split.validation == [9]
    assert split.train_boundary == 9
    assert sequential_split(list(range(10))) == split


def test_batch_has_expected_dimensions_and_shift():
    tokens = list(range(20))
    inputs, targets = make_batch(
        tokens, batch_size=3, context_length=4, generator=torch.Generator().manual_seed(1)
    )
    assert inputs.shape == (3, 4)
    assert targets.shape == (3, 4)
    assert inputs.dtype == targets.dtype == torch.long
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert torch.equal(targets[:, -1], inputs[:, -1] + 1)


def test_seeded_sampling_is_reproducible_and_cannot_cross_split_boundary():
    split = sequential_split(list(range(30)))
    first = make_batch(
        split.train, batch_size=5, context_length=4, generator=torch.Generator().manual_seed(7)
    )
    second = make_batch(
        split.train, batch_size=5, context_length=4, generator=torch.Generator().manual_seed(7)
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    assert first[1].max().item() < split.train_boundary


def test_validation_batches_cannot_access_training_tokens():
    split = sequential_split(list(range(90)) + list(range(100, 110)))
    inputs, targets = make_batch(
        split.validation,
        batch_size=4,
        context_length=3,
        generator=torch.Generator().manual_seed(5),
    )
    assert inputs.min().item() >= 100
    assert targets.min().item() >= 100
    assert torch.equal(targets[:, :-1], inputs[:, 1:])


@pytest.mark.parametrize(("batch_size", "context_length"), [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_invalid_batch_parameters_are_rejected(batch_size, context_length):
    with pytest.raises(ValueError):
        make_batch(
            [0, 1, 2],
            batch_size=batch_size,
            context_length=context_length,
            generator=torch.Generator(),
        )


def test_insufficient_split_length_is_rejected():
    with pytest.raises(ValueError, match="insufficient"):
        make_batch([0, 1, 2], batch_size=1, context_length=3, generator=torch.Generator())


def test_invalid_generator_is_rejected():
    with pytest.raises(TypeError, match="explicit torch.Generator"):
        make_batch([0, 1, 2], batch_size=1, context_length=1, generator=None)
